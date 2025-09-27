import os
import json
from typing import Dict, Any
from litellm import completion


class ModelInterface:
    def __init__(self):
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        if self.gemini_key:
            os.environ["GEMINI_API_KEY"] = self.gemini_key

    def call_model(self, model_name: str, prompt: str, temperature: float = 0.5, max_tokens: int = 500) -> str:
        return self._call_litellm(model_name, prompt, temperature, max_tokens)

    def _call_litellm(self, model_name: str, prompt: str, temperature: float, max_tokens: int) -> str:
        try:
            print(f"\n=== Calling {model_name} ===")
            print(f"Temperature: {temperature}, Max tokens: {max_tokens}")

            response = completion(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )

            print(f"Response object: {response}")
            print(f"Choices: {response.choices}")

            content = response.choices[0].message.content
            print(f"Content: {content}")
            print(f"Content type: {type(content)}")

            if content is None:
                print("WARNING: Content is None")
                return json.dumps({"score_a": 0.5, "score_b": 0.5, "explanation": "Empty response", "uncertainty": 1.0})

            return content

        except Exception as e:
            print(f"ERROR calling {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return json.dumps({"score_a": 0.5, "score_b": 0.5, "explanation": f"Model error: {str(e)}", "uncertainty": 1.0})


model_interface = ModelInterface()