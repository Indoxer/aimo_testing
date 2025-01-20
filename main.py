import os
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from pylatex import Document, NoEscape
from vllm import LLM, SamplingParams

from constants import *
from evaluator import EvaluationResult, ProblemEvaluator


def setup_directories() -> None:
    """Set up and clean output directories."""
    dirs_to_setup = [DATA_DIR]
    if ENABLE_LATEX:
        dirs_to_setup.append(LATEX_DIR)

    for dir_path in dirs_to_setup:
        if dir_path.exists():
            for file in dir_path.glob("*"):
                file.unlink()
        else:
            dir_path.mkdir(parents=True)


def load_problems() -> Tuple[List[str], List[float]]:
    """Load problems and solutions from dataset."""
    df = pd.read_parquet(DATASET_PATH)
    # get first 10 problems
    df = df.head(10)
    problems = [PROBLEM_PROMPT.format(p) for p in df["problem"]]
    solutions = df["answer"].tolist()  # Changed from "solutions" to "answer"
    return problems, solutions


def render_latex(
    prompt: str,
    generated_text: str,
    predicted_answer: str,
    real_answer: float,
    filename: Path,
) -> None:
    """Generate LaTeX document with problem and solution."""
    doc = Document()
    doc.preamble.append(NoEscape(r"\usepackage{amsmath,amssymb}"))
    doc.append(NoEscape(prompt))
    doc.append(NoEscape(r"\ \ \%"))
    doc.append(NoEscape(generated_text))
    doc.append(NoEscape(r"\ \ \%"))
    doc.append(NoEscape("Predicted answer: " + predicted_answer))
    doc.append(NoEscape("\n"))
    doc.append(NoEscape("Valid answer: " + str(real_answer)))
    doc.generate_pdf(str(filename))


def get_most_common_answer(
    samples: List[str], evaluator: ProblemEvaluator
) -> Optional[float]:
    """Return the most common answer from a list of sample outputs."""
    answers = []
    for sample in samples:
        try:
            answer = evaluator.extract_answer(sample)
            answers.append(answer)
        except ValueError:
            continue

    if not answers:
        return None

    return Counter(answers).most_common(1)[0][0]


def main():
    setup_directories()

    # Initialize model and load problems
    llm = LLM(**MODEL_CONFIG)
    problems, solutions = load_problems()

    # Generate solutions
    outputs = llm.generate(problems, SamplingParams(**SAMPLING_CONFIG))

    evaluator = ProblemEvaluator()
    correct_per_problem = [0] * len(outputs)
    correct_first_try = 0
    has_correct_answer = 0
    correct_majority = 0

    # Process outputs
    for prob_idx, output in enumerate(outputs):
        real_answer = float(solutions[prob_idx])  # Convert to float here

        # Get most common answer from all samples
        sample_texts = [sample.text for sample in output.outputs]
        majority_answer = get_most_common_answer(sample_texts, evaluator)
        if majority_answer is not None and abs(majority_answer - real_answer) < 1e-6:
            correct_majority += 1

        # Process samples for current problem
        for sample_idx, sample in enumerate(output.outputs):
            predicted_answer = "N/A"
            try:
                if evaluator.evaluate_sample(sample.text, real_answer):
                    correct_per_problem[prob_idx] += 1
                predicted_answer = str(evaluator.extract_answer(sample.text))
            except ValueError as e:
                print(
                    f"Error evaluating problem {prob_idx}, sample {sample_idx}: {str(e)}"
                )

            # Save output files
            output_path = DATA_DIR / f"output_{prob_idx}_{sample_idx}.txt"
            with open(output_path, "w") as f:
                f.write(output.prompt + "\n\n")
                f.write(sample.text + "\n\n")
                f.write(f"Predicted answer: {predicted_answer}\n")
                f.write(f"Valid answer: {real_answer}")

            # Generate LaTeX if enabled
            if ENABLE_LATEX:
                try:
                    render_latex(
                        output.prompt,
                        sample.text,
                        predicted_answer,
                        real_answer,
                        LATEX_DIR / f"latex_{prob_idx}_{sample_idx}",
                    )
                except Exception as e:
                    print(
                        f"Error generating LaTeX for problem {prob_idx}, sample {sample_idx}: {str(e)}"
                    )

        # Update statistics
        if correct_per_problem[prob_idx] > 0:
            has_correct_answer += 1

        # Check first try
        if evaluator.evaluate_sample(output.outputs[0].text, real_answer):
            correct_first_try += 1

        print(
            f"Problem {prob_idx}: {correct_per_problem[prob_idx]}/16 correct "
            f"(First try: {correct_first_try > prob_idx}, "
            f"Majority answer: {majority_answer})"
        )

    # Clean up non-PDF files if LaTeX was enabled
    if ENABLE_LATEX:
        for file in LATEX_DIR.glob("*"):
            if file.suffix != ".pdf":
                file.unlink()

    # Create and print evaluation results
    results = EvaluationResult(
        total_problems=len(outputs),
        correct_first_try=correct_first_try,
        correct_majority=correct_majority,
        has_correct_answer=has_correct_answer,
        total_correct=sum(correct_per_problem),
        correct_per_problem=correct_per_problem,
    )
    results.print_statistics()


if __name__ == "__main__":
    main()
