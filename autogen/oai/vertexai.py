import base64
from io import BytesIO
from typing import Dict, Any, Union, List
from PIL import Image
from google.auth.credentials import Credentials
import requests
import google.auth
import vertexai
import warnings
from vertexai.preview.generative_models import GenerativeModel, ChatSession, Content, Part, FunctionDeclaration, Tool
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import ChatCompletionMessage, Choice
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function
from openai.types.completion_usage import CompletionUsage
import random
import time
import json
import copy
import re

class VertexClient:
    """Client for Google's Gemini API.

    Please visit this [page](https://github.com/microsoft/autogen/issues/2387) for the roadmap of Gemini integration
    of AutoGen.
    """

    # Mapping, where Key is a term used by Autogen, and Value is a term used by Gemini
    PARAMS_MAPPING = {
        "max_tokens": "max_output_tokens",
        # "n": "candidate_count", # Gemini supports only `n=1`
        "stop_sequences": "stop_sequences",
        "temperature": "temperature",
        "top_p": "top_p",
        "top_k": "top_k",
        "max_output_tokens": "max_output_tokens",
    }

    def __init__(self, **kwargs):
        
        location = kwargs.get("location", None)
        project = kwargs.get("project", None)
        
        assert(location is not None and project is not None), "Please provide location and project in the kwargs."
        
        try:
            if(kwargs.get("credentials", None)):
                credentials = google.auth.load_credentials_from_dict(kwargs.get("credentials"))
            elif(kwargs.get("credentials_path", None)):
                credentials = google.auth.load_credentials_from_file(kwargs.get("credentials_path"))
            else:
                credentials = None
                google.auth.default()
        except:
            raise ValueError("Please provide credentials or credentials_path or run 'gcloud auth application-default login' to set the default credentials.")
        
        if(credentials):
            vertexai.init(credentials=credentials, project=project, location=location)
        else:
            vertexai.init(project=project, location=location)

    def message_retrieval(self, response) -> List:
        """
        Retrieve and return a list of strings or a list of Choice.Message from the response.

        NOTE: if a list of Choice.Message is returned, it currently needs to contain the fields of OpenAI's ChatCompletion Message object,
        since that is expected for function or tool calling in the rest of the codebase at the moment, unless a custom agent is being used.
        """
        return [choice.message for choice in response.choices]

    def cost(self, response) -> float:
        return response.cost

    @staticmethod
    def get_usage(response) -> Dict:
        """Return usage summary of the response using RESPONSE_USAGE_KEYS."""
        # ...  # pragma: no cover
        return {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "cost": response.cost,
            "model": response.model,
        }

    def create(self, params: Dict) -> ChatCompletion:
        model_name: str = params.get("model", "gemini-1.5-pro-001")
        if not model_name or model_name.startswith("gemini") is False:
            raise ValueError(
                "Please provide a model name for the Gemini Client. "
                "You can configurate it in the OAI Config List file. "
                "See this [LLM configuration tutorial](https://microsoft.github.io/autogen/docs/topics/llm_configuration/) for more details."
            )

        params.get("api_type", "vertex")  # not used
        messages = params.get("messages", [])
        tools = params.get("tools", [])
        stream = params.get("stream", False)
        n_response = params.get("n", 1)

        generation_config = {
            gemini_term: params[autogen_term]
            for autogen_term, gemini_term in self.PARAMS_MAPPING.items()
            if autogen_term in params
        }
        safety_settings = params.get("safety_settings", {})

        if stream:
            warnings.warn(
                "Streaming is not supported for Gemini yet, and it will have no effect. Please set stream=False.",
                UserWarning,
            )

        if n_response > 1:
            warnings.warn("Gemini only supports `n=1` for now. We only generate one response.", UserWarning)

        # A. create and call the chat model.
        gemini_messages = oai_messages_to_gemini_messages(messages)
        gemini_tools = oai_tools_to_gemini_tools(tools)

        # we use chat model by default
        model = GenerativeModel(
            model_name, generation_config=generation_config, safety_settings=safety_settings, tools=gemini_tools
        )
        print("MESSAGES", gemini_messages)
        chat: ChatSession = model.start_chat(history=gemini_messages[:-1])
        max_retries = 5
        for attempt in range(max_retries):
            ans: Content = None
            try:
                chat.send_message(gemini_messages[-1].parts, stream=stream)
            except Exception as e:
                delay = 5 * (2**attempt)
                warnings.warn(
                    f"InternalServerError `500` occurs when calling Gemini's chat model. Retry in {delay} seconds...",
                    UserWarning,
                )
                time.sleep(delay)
            else:
                # `ans = response.text` is unstable. Use the following code instead.
                ans: Content = chat.history[-1]
                break

        if ans is None:
            raise RuntimeError(f"Fail to get response from Google AI after retrying {attempt + 1} times.")

        prompt_tokens = model.count_tokens(chat.history[:-1]).total_tokens
        completion_tokens = model.count_tokens(contents=Content(parts=ans.parts)).total_tokens

        # 3. convert output
        choices = gemini_content_to_oai_choices(ans)

        response_oai = ChatCompletion(
            id=str(random.randint(0, 1000)),
            model=model_name,
            created=int(time.time() * 1000),
            object="chat.completion",
            choices=choices,
            usage=CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            cost=calculate_gemini_cost(prompt_tokens, completion_tokens, model_name),
        )

        return response_oai


def calculate_gemini_cost(input_tokens: int, output_tokens: int, model_name: str) -> float:
    if "1.5-flash" in model_name:
        # "gemini-1.5-flash-001"
        # Cost is $7 per million input tokens and $21 per million output tokens
        return 7.0 * input_tokens / 1e6 + 21.0 * output_tokens / 1e6
    
    if "1.5-pro" in model_name:
        # "gemini-1.5-pro-001"
        # Cost is $7 per million input tokens and $21 per million output tokens
        return 7.0 * input_tokens / 1e6 + 21.0 * output_tokens / 1e6

    if "gemini-pro" not in model_name and "gemini-1.0-pro" not in model_name:
        warnings.warn(f"Cost calculation is not implemented for model {model_name}. Using Gemini-1.0-Pro.", UserWarning)

    # Cost is $0.5 per million input tokens and $1.5 per million output tokens
    return 0.5 * input_tokens / 1e6 + 1.5 * output_tokens / 1e6


def oai_content_to_gemini_content(message: Dict[str, Any]) -> List[Part]:
    """Convert content from OAI format to Gemini format"""
    rst = []
    if isinstance(message, str):
        rst.append(Part.from_text(message))
        return rst
        
    if "tool_calls" in message:
        rst.append(Part.from_dict({
            "functionCall": {
                "name": message["tool_calls"][0]["function"]["name"],
                "args": json.loads(message["tool_calls"][0]["function"]["arguments"])
            }
        }))
        return rst
    
    if message["role"] == "tool":
        rst.append(Part.from_function_response(
            name=message["name"],
            response=json.loads(message["content"])
        ))
        return rst
    
    if isinstance(message["content"], str):
        rst.append(Part.from_text(message["content"]))
        return rst

    assert isinstance(message["content"], list)

    for msg in message["content"]:
        if isinstance(msg, dict):
            assert "type" in msg, f"Missing 'type' field in message: {msg}"
            if msg["type"] == "text":
                rst.append(Part.from_text(msg["text"]))
            elif msg["type"] == "image_url":
                b64_img = get_image_data(msg["image_url"]["url"])
                img = _to_pil(b64_img)
                rst.append(img)
            else:
                raise ValueError(f"Unsupported message type: {msg['type']}")
        else:
            raise ValueError(f"Unsupported message type: {type(msg)}")
    return rst


def concat_parts(parts: List[Part]) -> List:
    """Concatenate parts with the same type.
    If two adjacent parts both have the "text" attribute, then it will be joined into one part.
    """
    if not parts:
        return []
    
    if len(parts) == 1:
        return parts

    concatenated_parts = []
    previous_part = parts[0]

    for current_part in parts[1:]:
        if previous_part.text != "":
            previous_part = Part.from_text(previous_part.text + current_part.text)
        else:
            concatenated_parts.append(previous_part)
            previous_part = current_part

    if previous_part.text == "":
        previous_part = Part.from_text("empty")  # Empty content is not allowed.
    concatenated_parts.append(previous_part)

    return concatenated_parts


def oai_messages_to_gemini_messages(messages: list[Dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert messages from OAI format to Gemini format.
    Make sure the "user" role and "model" role are interleaved.
    Also, make sure the last item is from the "user" role.
    """
    prev_role = None
    rst = []
    curr_parts = []
    for i, message in enumerate(messages):
        
        # Since the tool call message does not have the "name" field, we need to find the corresponding tool message.
        if message["role"] == "tool":
            message["name"] = [m for m in messages if "tool_calls" in m and m["tool_calls"][0]["id"] == message["tool_call_id"]][0]["tool_calls"][0]["function"]["name"]
        
        parts = oai_content_to_gemini_content(message)
        role = "user" if message["role"] in ["user", "system"] else "model"
        
        if prev_role is None or role == prev_role:
            # If the message is a function call or a function response, we need to separate it from the previous message.
            if parts[0].function_call or parts[0].function_response:
                if len(curr_parts) > 1:
                    rst.append(Content(parts=concat_parts(curr_parts), role=prev_role))
                elif len(curr_parts) == 1:
                    rst.append(Content(parts=curr_parts, role=None if curr_parts[0].function_response else role))
                rst.append(Content(parts=parts, role="user" if parts[0].function_response else role))
                rst.append(Content(parts=oai_content_to_gemini_content("continue"), role="model"))
                curr_parts = []
            else:
                curr_parts += parts
        elif role != prev_role:
            if len(curr_parts) > 0:
                rst.append(Content(parts=concat_parts(curr_parts), role=prev_role))
            curr_parts = parts
        prev_role = role

    # handle the last message
    if len(curr_parts) > 0:
        rst.append(Content(parts=concat_parts(curr_parts), role=role))

    # The Gemini is restrict on order of roles, such that
    # 1. The messages should be interleaved between user and model.
    # 2. The last message must be from the user role.
    # We add a dummy message "continue" if the last role is not the user.
    if rst[-1].role != "user":
        rst.append(Content(parts=oai_content_to_gemini_content("continue"), role="user"))

    return rst


def oai_tools_to_gemini_tools(tools: List[Dict[str, Any]]) -> List[Tool]:
    """Convert tools from OAI format to Gemini format."""
    if len(tools) == 0:
        return None
    function_declarations = []
    for tool in tools:
        function_declarations.append(FunctionDeclaration(
            name=tool["function"]["name"],
            description=tool["function"]["description"],
            parameters=tool["function"]["parameters"],
        ))
    return [Tool(
        function_declarations=function_declarations
    )]


def gemini_content_to_oai_choices(response: Content) -> List[Choice]:
    """Convert response from Gemini format to OAI format."""
    text = None
    tool_calls = None
    for part in response.parts:
        if part.function_call:
            arguments = Part.to_dict(part)["function_call"]["args"]
            tool_calls = [
                ChatCompletionMessageToolCall(
                    id=str(random.randint(0, 1000)),
                    type="function",
                    function=Function(
                        name=part.function_call.name,
                        arguments=json.dumps(arguments)
                    )
                )
            ]
        elif part.text:
            text = part.text
    message = ChatCompletionMessage(role="assistant", content=text, function_call=None, tool_calls=tool_calls)
    return [Choice(finish_reason="tool_calls" if tool_calls else "stop", index=0, message=message)]


def _to_pil(data: str) -> Image.Image:
    """
    Converts a base64 encoded image data string to a PIL Image object.

    This function first decodes the base64 encoded string to bytes, then creates a BytesIO object from the bytes,
    and finally creates and returns a PIL Image object from the BytesIO object.

    Parameters:
        data (str): The base64 encoded image data string.

    Returns:
        Image.Image: The PIL Image object created from the input data.
    """
    return Image.open(BytesIO(base64.b64decode(data)))


def get_image_data(image_file: str, use_b64=True) -> bytes:
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        content = response.content
    elif re.match(r"data:image/(?:png|jpeg);base64,", image_file):
        return re.sub(r"data:image/(?:png|jpeg);base64,", "", image_file)
    else:
        image = Image.open(image_file).convert("RGB")
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        content = buffered.getvalue()

    if use_b64:
        return base64.b64encode(content).decode("utf-8")
    else:
        return content