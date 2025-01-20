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
        print(
            f"\nTotal AIME (correct attempts/total attempts): {self.total_correct}/{total_attempts} "
            f"({self.total_correct/total_attempts:.2%})"
        )
        print(
            f"Single try accuracy: {self.correct_first_try}/{self.total_problems} "
            f"({self.correct_first_try/self.total_problems:.2%})"
        )
        print(
            f"Majority vote accuracy: {self.correct_majority}/{self.total_problems} "
            f"({self.correct_majority/self.total_problems:.2%})"
        )
        print(
            f"Problems with at least one correct answer: "
            f"{self.has_correct_answer}/{self.total_problems} "
            f"({self.has_correct_answer/self.total_problems:.2%})"
        )


class ProblemEvaluator:
    @staticmethod
    def extract_answer(text: str) -> float:
        """Extract and convert the last boxed answer from generated text."""
        # Find all \boxed{...} patterns in the text, handling potential whitespace
        matches = list(re.finditer(r"\\boxed\s*{\s*([^{}]+?)\s*}", text))
        if not matches:
            raise ValueError("No \\boxed{} answer found in the generated text")

        # Get the last match
        last_match = matches[-1]
        try:
            # Clean up the answer string and convert to float
            answer_str = last_match.group(1).strip()
            return float(answer_str)
        except ValueError:
            raise ValueError(
                f"Failed to convert answer '{last_match.group(1)}' to a number"
            )

    @staticmethod
    def evaluate_sample(generated: str, solution: float) -> bool:
        """Evaluate if a single sample's answer matches the solution."""
        try:
            predicted = ProblemEvaluator.extract_answer(generated)
            return float(solution) == predicted
        except (ValueError, IndexError) as e:
            print(f"Error evaluating sample: {str(e)}")
            return False
