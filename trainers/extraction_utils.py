import re
from typing import List, Optional


class ATLASExtractionUtils:
    
    @staticmethod
    def extract_teaching_content(teacher_response: str) -> str:
        teaching_match = re.search(r'<teaching>(.*?)</teaching>', teacher_response, re.DOTALL)
        if teaching_match:
            return teaching_match.group(1).strip()
        return teacher_response.strip()
    
    @staticmethod
    def normalize_answer(text: str) -> str:
        if not text:
            return ""
        
        text = str(text).strip().lower()
        text = text.rstrip('.')
        
        try:
            num = float(text)
            if num == int(num):
                return str(int(num))
            else:
                return f"{num:.2f}"
        except:
            pass
        
        return text
    
    @staticmethod
    def extract_solution(response: str) -> str:
        boxed_matches = re.findall(r'\\boxed\{([^}]+)\}', response)
        if boxed_matches:
            return boxed_matches[-1].strip()

        solution_matches = re.findall(r'<solution>(.*?)</solution>', response, re.DOTALL)
        if solution_matches:
            return solution_matches[-1].strip()

        answer_patterns = [
            r'answer is[:\s]+([^\n.]+)',
            r'answer[:\s]+([^\n.]+)',
            r'= ([^\n.]+)$',
        ]

        for pattern in answer_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
            if matches:
                return matches[-1].strip()

        numbers = re.findall(r'-?\d+(?:\.\d+)?', response)
        if numbers:
            return numbers[-1]

        return response
    
    @staticmethod
    def extract_solutions(responses: List[str]) -> List[str]:
        return [ATLASExtractionUtils.extract_solution(resp) for resp in responses]
    
    @staticmethod
    def check_correctness(solution: str, ground_truth: str) -> bool:
        if not ground_truth or not solution:
            return False
        
        solution_normalized = ATLASExtractionUtils.normalize_answer(solution)
        ground_truth_normalized = ATLASExtractionUtils.normalize_answer(ground_truth)
        
        return solution_normalized == ground_truth_normalized