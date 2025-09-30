import json
import io
import contextlib
from typing import List, Dict, Any, Callable, Optional


class CrewAITrajectoryCapture:

    @staticmethod
    def extract_from_verbose_output(output: str) -> str:
        trajectory_steps = []
        lines = output.split('\n')
        current_step = {}
        step_num = 0

        for line in lines:
            original_line = line
            line = line.strip()
            if not line:
                continue

            if 'Agent: ' in line and '│' in line:
                if current_step:
                    trajectory_steps.append(current_step)
                step_num += 1
                agent_name = line.split('Agent:')[-1].strip().rstrip('│').strip()
                current_step = {
                    'step': step_num,
                    'agent': agent_name,
                    'type': 'agent_working'
                }
            elif 'Task:' in line and '│' in line and current_step:
                task_text = line.split('Task:')[-1].strip().rstrip('│').strip()
                current_step['task'] = task_text
            elif 'Using Tool:' in line and '│' in line and current_step:
                tool_name = line.split('Using Tool:')[-1].strip().rstrip('│').strip()
                current_step['tool'] = tool_name
            elif ('🔧 Used' in line or 'Used' in line) and 'Tool' in line:
                if 'Used' in line:
                    tool_part = line.split('Used')[-1].strip()
                    tool_name = tool_part.split('Tool')[0].strip() + ' Tool'
                    if current_step:
                        current_step['tool'] = tool_name

        if current_step:
            trajectory_steps.append(current_step)

        if not trajectory_steps:
            return ""

        formatted_trajectory = "Agent Execution Trajectory:\n"
        for step in trajectory_steps:
            formatted_trajectory += f"\nStep {step['step']}:\n"
            formatted_trajectory += f"  Agent: {step.get('agent', 'Unknown')}\n"
            if 'task' in step:
                formatted_trajectory += f"  Task: {step['task']}\n"
            if 'tool' in step:
                formatted_trajectory += f"  Tool Used: {step['tool']}\n"
            if 'output' in step:
                output_text = step.get('output', '')
                if len(output_text) > 200:
                    formatted_trajectory += f"  Output: {output_text[:200]}...\n"
                else:
                    formatted_trajectory += f"  Output: {output_text}\n"

        return formatted_trajectory


class InstrumentedAgentWrapper:

    def __init__(self, user_agent: Callable, framework: str = 'auto'):
        self.user_agent = user_agent
        self.framework = framework
        self._available_tools = self._extract_available_tools()

    def _extract_crewai_tools(self) -> List[str]:
        tools_info = []
        try:
            import sys

            if 'lumyn.crew' in sys.modules:
                crew_module = sys.modules['lumyn.crew']
                crew_instance = crew_module.LumynCrew()

                if hasattr(crew_instance, 'agents'):
                    agents = crew_instance.agents
                    for agent in agents:
                        if hasattr(agent, 'tools'):
                            for tool in agent.tools:
                                tool_desc = self._format_tool_info(tool)
                                if tool_desc and tool_desc not in tools_info:
                                    tools_info.append(tool_desc)
        except:
            pass
        return tools_info

    def _extract_available_tools(self) -> str:
        tools_info = []

        if self.framework == 'crewai' or self.framework == 'auto':
            tools_info = self._extract_crewai_tools()

        if hasattr(self.user_agent, 'tools'):
            tools = self.user_agent.tools
            for tool in tools:
                tool_desc = self._format_tool_info(tool)
                if tool_desc and tool_desc not in tools_info:
                    tools_info.append(tool_desc)

        if not tools_info:
            return """Available Tools:
- GetAlertsCustomTool: Get system alerts
- NL2MetricsCustomTool: Query metrics
- NL2TracesCustomTool: Query traces
- NL2KubectlCustomTool: Execute kubectl commands
- RemediationCustomTool: Perform remediation
- DiagnosisJSONReportCustomTool: Generate diagnosis report"""

        return "Available Tools:\n" + "\n".join(tools_info)

    def _format_tool_info(self, tool) -> Optional[str]:
        try:
            name = getattr(tool, 'name', getattr(tool, '__name__', 'unknown'))
            description = getattr(tool, 'description', getattr(tool, '__doc__', ''))
            return f"- {name}: {description}"
        except:
            return None

    def get_available_tools(self) -> str:
        return self._available_tools

    def __call__(self, prompts: List[str]) -> tuple[List[str], List[str]]:
        responses = []
        trajectories = []

        for prompt in prompts:
            import os

            log_file = "traces/agent_execution.log"
            if os.path.exists(log_file):
                os.remove(log_file)

            response = self.user_agent([prompt])

            if isinstance(response, list):
                response = response[0]
            responses.append(response)

            captured_output = ""
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    captured_output = f.read()

            trajectory = CrewAITrajectoryCapture.extract_from_verbose_output(captured_output)

            if not trajectory:
                trajectory = ""

            trajectories.append(trajectory)

        return responses, trajectories