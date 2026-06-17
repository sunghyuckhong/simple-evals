import time
from typing import Any

import openai
from openai import OpenAI

from ..types import MessageList, SamplerBase, SamplerResponse

OPENAI_SYSTEM_MESSAGE_API = "You are a helpful assistant."
OPENAI_SYSTEM_MESSAGE_CHATGPT = (
    "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture."
    + "\nKnowledge cutoff: 2023-12\nCurrent date: 2024-04-01"
)


class ChatCompletionSampler(SamplerBase):
    """
    Sample from OpenAI's chat completion API
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int | None = 1024,
        top_p: float = 1.0,
        base_url: str | None = None,
        extra_body: dict | None = None,
        timeout: float = None, #Default seems to be 60
    ):
        self.api_key_name = "OPENAI_API_KEY"
        self.client = OpenAI(base_url=base_url, timeout=timeout) if base_url else OpenAI(timeout=timeout)
        # using api_key=os.environ.get("OPENAI_API_KEY")  # please set your API_KEY
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.extra_body = extra_body or {}
        self.image_format = "url"

    def _handle_image(
        self,
        image: str,
        encoding: str = "base64",
        format: str = "png",
        fovea: int = 768,
    ):
        new_image = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{format};{encoding},{image}",
            },
        }
        return new_image

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        if self.system_message:
            message_list = [
                self._pack_message("system", self.system_message)
            ] + message_list
        trial = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message_list,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    extra_body=self.extra_body,
                )
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("OpenAI API returned empty response; retrying")
                # Reasoning models split the <think> trace into a separate field (the
                # reasoning parser keeps it out of `content`). Capture it — it's an
                # extra/untyped field on the OpenAI message — so callers can display it.
                reasoning_content = getattr(
                    response.choices[0].message, "reasoning_content", None
                )
                return SamplerResponse(
                    response_text=content,
                    response_metadata={
                        "usage": response.usage,
                        "reasoning_content": reasoning_content,
                    },
                    actual_queried_message_list=message_list,
                )
            # NOTE: BadRequestError is triggered once for MMMU, please uncomment if you are reruning MMMU
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return SamplerResponse(
                    response_text="No response (bad request).",
                    response_metadata={"usage": None},
                    actual_queried_message_list=message_list,
                )
            except Exception as e:
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception
