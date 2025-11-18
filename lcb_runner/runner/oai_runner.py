import os
from time import sleep

try:
    import openai
    from openai import OpenAI
except ImportError as e:
    pass

from lcb_runner.lm_styles import LMStyle
from lcb_runner.runner.base_runner import BaseRunner

SYSTEM_PROMPT = (
    "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate "
    "with yourself via systematic reasoning processes to help come to a correct solution prior to answering. "
    "You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem."
)
class OpenAIRunner(BaseRunner):
    client = OpenAI(
        base_url="http://localhost:34885/v1",
        api_key="sk-1234567",
        timeout=None,
    )

    def __init__(self, args, model):
        super().__init__(args, model)
        if model.model_style == LMStyle.OpenAIReasonPreview:
            self.client_kwargs: dict[str | str] = {
                "model": args.model,
                "max_completion_tokens": 25000,
            }
        elif model.model_style == LMStyle.OpenAIReason:
            assert (
                "__" in args.model
            ), f"Model {args.model} is not a valid OpenAI Reasoning model as we require reasoning effort in model name."
            model, reasoning_effort = args.model.split("__")
            self.client_kwargs: dict[str | str] = {
                "model": "Motif-Technologies/Motif-2.6B",
                "reasoning_effort": reasoning_effort,
            }
        else:
            self.client_kwargs: dict[str | str] = {
                #"model": "Motif-Technologies/Motif-2.6B",
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "top_p": args.top_p,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "n": args.n,
                "timeout": None,
                # "stop": args.stop, --> stop is only used for base models currently
            }

    def _run_single(self, prompt: list[dict[str, str]], n: int = 10) -> list[str]:
        assert isinstance(prompt, list)

        if n == 0:
            print("Max retries reached. Returning empty response.")
            return []

        try:
            if prompt[0]["role"] == "system":
                prompt[0]["content"] = SYSTEM_PROMPT
            else:
                prompt.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
            
            response = OpenAIRunner.client.chat.completions.create(
                model=None,
                messages=prompt,
                **self.client_kwargs,
                extra_body={
                    "skip_special_tokens": False,
                    "stop_token_ids": [219395, 219396, 219403, 219405],
                    "chat_template_kwargs": {
                        "enable_thinking": True # 또는 False
                    },
                }
            )
            for c in response.choices:
                whole_response = c.message.content
                whole_response = whole_response.split("</think>")
                if len(whole_response) > 1:
                    whole_response = whole_response[1]
                else:
                    whole_response = ""
                c.message.content = whole_response
            return [c.message.content for c in response.choices]
        except (
            openai.APIError,
            openai.RateLimitError,
            openai.InternalServerError,
            openai.OpenAIError,
            openai.APIStatusError,
            openai.APITimeoutError,
            openai.InternalServerError,
            openai.APIConnectionError,
        ) as e:
            print("Exception: ", repr(e))
            print("Sleeping for 30 seconds...")
            print("Consider reducing the number of parallel processes.")
            sleep(30)
            return self._run_single(prompt, n=n - 1)
        except Exception as e:
            print(f"Failed to run the model for {prompt}!")
            print("Exception: ", repr(e))
            raise e
        return [c.message.content for c in response.choices]
