"""Wrap a simple HTTP microservice with the Atlas SDK."""

from __future__ import annotations

import json
import tempfile
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from atlas import run

CONFIG_TEMPLATE = textwrap.dedent(
    """
    agent:
      type: http_api
      name: sdk-http-example
      system_prompt: |
        You are an HTTP-based agent that echoes structured answers.
      tools: []
      transport:
        base_url: http://127.0.0.1:{port}/agent
        headers: {{}}
        timeout_seconds: 30
        retry:
          attempts: 1
          backoff_seconds: 0.5
      payload_template:
        mode: inference
      result_path:
        - output
    student:
      prompts:
        planner: |
          {base_prompt}

          Break the task into two concise steps as JSON.
        executor: |
          {base_prompt}

          Call the HTTP agent and describe the result.
        synthesizer: |
          {base_prompt}

          Summarize the findings clearly for the user.
      max_plan_tokens: 512
      max_step_tokens: 512
      max_synthesis_tokens: 512
      tool_choice: auto
    teacher:
      llm:
        provider: openai
        model: gpt-4o-mini
        api_key_env: OPENAI_API_KEY
        temperature: 0.1
        max_output_tokens: 512
      max_review_tokens: 512
      plan_cache_seconds: 60
      guidance_max_tokens: 256
      validation_max_tokens: 256
    orchestration:
      max_retries: 1
      step_timeout_seconds: 120
      rim_guidance_tag: rim_feedback
      emit_intermediate_steps: true
    rim:
      judges:
        - identifier: process
          kind: process
          weight: 0.5
          principles:
            - Followed instructions
          llm:
            provider: openai
            model: gpt-4o-mini
            api_key_env: OPENAI_API_KEY
            temperature: 0.0
            max_output_tokens: 256
        - identifier: helpfulness
          kind: helpfulness
          weight: 0.5
          principles:
            - User impact
          llm:
            provider: openai
            model: gpt-4o-mini
            api_key_env: OPENAI_API_KEY
            temperature: 0.0
            max_output_tokens: 256
      temperatures: [0.0, 0.3]
      variance_threshold: 0.2
      uncertainty_threshold: 0.3
      arbiter:
        provider: openai
        model: gpt-4o-mini
        api_key_env: OPENAI_API_KEY
        temperature: 0.1
        max_output_tokens: 256
      success_threshold: 0.7
      retry_threshold: 0.6
      aggregation_strategy: weighted_mean
    storage: null
    prompt_rewrite:
      llm:
        provider: openai
        model: gpt-4o-mini
        api_key_env: OPENAI_API_KEY
        temperature: 0.1
        max_output_tokens: 512
      max_tokens: 1024
      temperature: 0.1
    """
)


class EchoHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 (http server signature)
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length)) if length else {}
        prompt = payload.get("prompt", "")
        metadata = payload.get("metadata", {}) or {}
        mode = metadata.get("mode") if isinstance(metadata, dict) else None
        if mode == "planning":
            plan = {
                "steps": [
                    {"id": 1, "description": "Call the HTTP agent to fetch an AI news headline", "depends_on": [], "tool": None, "tool_params": None},
                    {"id": 2, "description": "Summarize the headline for the user", "depends_on": [1], "tool": None, "tool_params": None},
                ]
            }
            response: dict[str, Any] = {"output": plan}
        elif mode == "synthesis":
            response = {"output": "The custom HTTP agent shared a single AI news headline."}
        else:
            response = {"output": f"Echo service received: {prompt[:100]}"}
        body = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: Any, **_kwargs: Any) -> None:  # Silence default logging
        return


def run_server(port: int) -> HTTPServer:
    server = HTTPServer(("127.0.0.1", port), EchoHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def write_temp_config(port: int) -> str:
    rendered = CONFIG_TEMPLATE.format(port=port)
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    tmp.write(rendered)
    tmp.flush()
    tmp.close()
    return tmp.name


if __name__ == "__main__":
    port = 8040
    server = run_server(port)
    config_path = write_temp_config(port)
    try:
        result = run(
            task="Ask the HTTP agent to list one interesting AI news headline",
            config_path=config_path,
        )
        print(result.final_answer)
    finally:
        server.shutdown()
        server.server_close()
        Path(config_path).unlink(missing_ok=True)
