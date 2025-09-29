import json
from typing import List, Dict, Any, Callable, Optional
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


class TrajectoryExtractor:

    @staticmethod
    def extract_from_spans(spans: List) -> str:
        trajectory_steps = []

        for span in spans:
            if hasattr(span, "to_json"):
                span_json = span.to_json()
                if isinstance(span_json, str):
                    try:
                        span_dict = json.loads(span_json)
                    except json.JSONDecodeError:
                        span_dict = {}
                else:
                    span_dict = span_json
            elif isinstance(span, dict):
                span_dict = span
            else:
                span_dict = {}

            operation = span_dict.get('attributes', {}).get('gen_ai.operation.name', 'unknown')

            if operation == 'execute_tool':
                tool_name = span_dict.get('attributes', {}).get('gen_ai.tool.name', 'unknown_tool')
                tool_input = span_dict.get('attributes', {}).get('gen_ai.input.messages', '')
                tool_output = span_dict.get('attributes', {}).get('gen_ai.output.messages', '')

                step = {
                    'type': 'tool_call',
                    'tool': tool_name,
                    'input': tool_input,
                    'output': tool_output
                }
                trajectory_steps.append(step)

            elif operation == 'invoke_agent':
                agent_input = span_dict.get('attributes', {}).get('gen_ai.input.messages', '')
                agent_output = span_dict.get('attributes', {}).get('gen_ai.output.messages', '')

                step = {
                    'type': 'agent_step',
                    'input': agent_input,
                    'output': agent_output
                }
                trajectory_steps.append(step)

            events = span_dict.get('events', [])
            for event in events:
                if event.get('name') == 'gen_ai.client.inference.operation.details':
                    event_data = event.get('attributes', {})
                    step = {
                        'type': 'reasoning_step',
                        'operation': event_data.get('gen_ai.operation.name', ''),
                        'details': event_data
                    }
                    trajectory_steps.append(step)

        if not trajectory_steps:
            return ""

        formatted_trajectory = "Agent Execution Trajectory:\n"
        for i, step in enumerate(trajectory_steps, 1):
            formatted_trajectory += f"\nStep {i} ({step['type']}):\n"
            if step['type'] == 'tool_call':
                formatted_trajectory += f"  Tool: {step['tool']}\n"
                formatted_trajectory += f"  Input: {step['input']}\n"
                formatted_trajectory += f"  Output: {step['output']}\n"
            elif step['type'] == 'agent_step':
                formatted_trajectory += f"  Input: {step['input']}\n"
                formatted_trajectory += f"  Output: {step['output']}\n"
            elif step['type'] == 'reasoning_step':
                formatted_trajectory += f"  Operation: {step['operation']}\n"
                formatted_trajectory += f"  Details: {json.dumps(step['details'], indent=4)}\n"

        return formatted_trajectory


class InstrumentedAgentWrapper:

    def __init__(self, user_agent: Callable, framework: str = 'auto'):
        self.user_agent = user_agent
        self.framework = framework

        self.span_exporter = InMemorySpanExporter()
        self.tracer_provider = TracerProvider()
        self.tracer_provider.add_span_processor(SimpleSpanProcessor(self.span_exporter))
        trace.set_tracer_provider(self.tracer_provider)

        self.tracer = trace.get_tracer(__name__)

        self._setup_instrumentation()
        self._available_tools = self._extract_available_tools()

    def _setup_instrumentation(self):
        if self.framework == 'crewai' or self.framework == 'auto':
            try:
                from openinference.instrumentation.crewai import CrewAIInstrumentor
                CrewAIInstrumentor().instrument(skip_dep_check=True)
            except ImportError:
                pass

    def _extract_available_tools(self) -> str:
        tools_info = []

        if hasattr(self.user_agent, 'tools'):
            tools = self.user_agent.tools
            for tool in tools:
                tool_desc = self._format_tool_info(tool)
                if tool_desc:
                    tools_info.append(tool_desc)

        elif hasattr(self.user_agent, 'agent') and hasattr(self.user_agent.agent, 'tools'):
            tools = self.user_agent.agent.tools
            for tool in tools:
                tool_desc = self._format_tool_info(tool)
                if tool_desc:
                    tools_info.append(tool_desc)

        if not tools_info:
            return "No tool information available."

        return "\n".join(tools_info)

    def _format_tool_info(self, tool) -> Optional[str]:
        try:
            name = getattr(tool, 'name', getattr(tool, '__name__', 'unknown'))
            description = getattr(tool, 'description', getattr(tool, '__doc__', ''))

            if hasattr(tool, 'args_schema') and tool.args_schema:
                schema = tool.args_schema
                if hasattr(schema, 'schema'):
                    params = schema.schema().get('properties', {})
                    param_str = ', '.join([f"{k}: {v.get('type', 'any')}" for k, v in params.items()])
                    return f"- {name}({param_str}): {description}"

            return f"- {name}: {description}"
        except Exception:
            return None

    def get_available_tools(self) -> str:
        return self._available_tools

    def __call__(self, prompts: List[str]) -> tuple[List[str], List[str]]:
        responses = []
        trajectories = []

        for prompt in prompts:
            self.span_exporter.clear()

            with self.tracer.start_as_current_span(
                "invoke_agent",
                attributes={
                    "gen_ai.operation.name": "invoke_agent",
                }
            ):
                response = self.user_agent([prompt])
                if isinstance(response, list):
                    response = response[0]
                responses.append(response)

            spans = self.span_exporter.get_finished_spans()
            trajectory = TrajectoryExtractor.extract_from_spans(spans)
            trajectories.append(trajectory)

        return responses, trajectories
