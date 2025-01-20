import re
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class EvaluationResult:
    total_problems: int
    correct_first_try: int
    correct_majority: int
    has_correct_answer: int
    total_correct: int
    correct_per_problem: List[int]

    def print_statistics(self) -> None:
        total_attempts = self.total_problems * 16
        print(f"\nTotal AMC Correct: {self.total_correct}/{total_attempts}")
        print(f"Single try accuracy: {self.correct_first_try/self.total_problems:.2%}")
        print(
            f"Majority vote accuracy: {self.correct_majority/self.total_problems:.2%}"
        )
        print(
            f"Problems with at least one correct answer: "
            f"{self.has_correct_answer/self.total_problems:.2%}"
        )
        print(
            f"Average correct per problem: "
            f"{self.total_correct/(self.total_problems*16):.2%}"
        )


class ProblemEvaluator:
    @staticmethod
    def extract_answer(text: str) -> float:
        """Extract and convert boxed answer from generated text."""
        match = re.findall(r"\\boxed{(.+?)}", text)
        if not match:
            raise ValueError("No boxed answer found")
        return float(match[0])

    @staticmethod
    def evaluate_sample(generated: str, solution: float) -> bool:
        """Evaluate if a single sample's answer matches the solution."""
        try:
            predicted = ProblemEvaluator.extract_answer(generated)
            return float(solution) == predicted
        except (ValueError, IndexError):
            return False
