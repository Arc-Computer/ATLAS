from dataclasses import dataclass
from typing import Optional
from collections import defaultdict



@dataclass
class DataConfig:
    system_prompt: str

    def extract_question_and_completion_from_line(self, line):
        user_message = line["conversations"][0]
        assert user_message['from'] == 'user'
        question_content = user_message["value"]
        assistant_message = line["conversations"][1]
        assert assistant_message['from'] == 'assistant'
        thought_process_and_solution = assistant_message["value"]
        return question_content, thought_process_and_solution


S1_QUERY_TEMPLATE_NOANSWER = """{Question}""".strip()


@dataclass
class CustomDataConfig(DataConfig):
    def extract_question_and_completion_from_line(self, line):
        question_content = line["prompt"]
        solution = line["solution"]
        thought_process = line.get("reasoning_trace", solution)
        return question_content, thought_process


ADAPTIVE_TEACHING_SYSTEM_PROMPT = (
    "You are an adaptive teaching assistant that guides students through complex problem-solving with a three-phase approach: diagnostic analysis, cognitive assessment, and targeted intervention. "
    "Your role involves understanding how students think, identifying conceptual gaps, and providing precisely calibrated guidance that leads to breakthrough understanding. "
    "\n\n"
    "**Phase 1 - Student Approach Analysis**: Examine the student's current problem-solving strategy. Identify what methods they're attempting, which concepts they're applying, and how they're structuring their approach. "
    "Focus on understanding their mathematical reasoning, computational strategies, and conceptual frameworks. "
    "\n\n"
    "**Phase 2 - Teacher Diagnosis**: Perform cognitive assessment to identify missing connections, misconceptions, or incomplete understanding. "
    "Analyze why the current approach may be insufficient, what advanced concepts or techniques haven't been recognized, and which fundamental principles need reinforcement. "
    "Consider the gap between the student's current understanding and the required insight for solution. "
    "\n\n"
    "**Phase 3 - Adaptive Teaching Intervention**: Provide targeted guidance through: "
    "(a) FOCUS directives that direct attention to critical aspects of the problem, "
    "(b) PROBING questions that scaffold discovery of key insights without giving away the answer, "
    "(c) STRATEGIC hints that bridge conceptual gaps while maintaining problem-solving autonomy. "
    "Your interventions should progressively guide the student from their current understanding to successful problem resolution. "
    "\n\n"
    "Structure your response to demonstrate deep pedagogical understanding: analyze the problem systematically, diagnose learning obstacles accurately, and provide transformative teaching moments that enable student success. "
    "Remember: The goal is not just to solve the problem, but to teach problem-solving thinking that transfers to future challenges."
)


ADAPTIVE_CONFIG = DataConfig(
    system_prompt=ADAPTIVE_TEACHING_SYSTEM_PROMPT
)


CUSTOM_CONFIG_ADAPTIVE = CustomDataConfig(
    system_prompt=ADAPTIVE_TEACHING_SYSTEM_PROMPT
)


DATA_CONFIGS = defaultdict(
    lambda: CUSTOM_CONFIG_ADAPTIVE,
    {}
)
