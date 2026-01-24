import os
import json
from typing import Dict, Any, cast
from litellm import completion


class ModelInterface:
    def __init__(self):
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        if self.gemini_key:
            os.environ["GEMINI_API_KEY"] = self.gemini_key

    def call_model(self, model_name: str, prompt: str, temperature: float = 0.5, max_tokens: int = 500) -> str:
        return self._call_litellm(model_name, prompt, temperature, max_tokens)

    def _is_verbose(self) -> bool:
        value = os.getenv("RIM_VERBOSE", "0")
        return value.lower() not in {"0", "false", "no", ""}

    def _call_litellm(self, model_name: str, prompt: str, temperature: float, max_tokens: int) -> str:
        try:
            if self._is_verbose():
                print(f"\n=== Calling {model_name} ===")
                print(f"Temperature: {temperature}, Max tokens: {max_tokens}")

            response = cast(
                Any,
                completion(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )
            )

            if self._is_verbose():
                print(f"Response object: {response}")
                print(f"Choices: {response.choices}")

            content = response.choices[0].message.content
            if self._is_verbose():
                print(f"Content: {content}")
                print(f"Content type: {type(content)}")

            if content is None:
                if self._is_verbose():
                    print("WARNING: Content is None")
                return json.dumps(self._default_error_payload("Empty response"))

            return content

        except Exception as e:
            print(f"ERROR calling {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return json.dumps(self._default_error_payload(f"Model error: {str(e)}"))

    def _default_error_payload(self, message: str) -> Dict[str, Any]:
        return {
            "score": 0.0,
            "score_a": 0.0,
            "score_b": 0.0,
            "rationale": message,
            "explanation": message,
            "uncertainty": 1.0,
            "principles": []
        }


model_interface = ModelInterface()
