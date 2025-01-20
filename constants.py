from pathlib import Path

# Directories
DATA_DIR = Path("data")
LATEX_DIR = Path("data2")

# Model configuration
MODEL_CONFIG = {
    "model": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "dtype": "half",
    "gpu_memory_utilization": 0.85,
    "enable_chunked_prefill": True,
    "max_num_seqs": 32,
    "max_num_batched_tokens": 4096,
}

# Sampling parameters
SAMPLING_CONFIG = {
    "temperature": 0.6,
    "repetition_penalty": 1.05,
    "top_p": 0.95,
    "min_tokens": 32,
    "max_tokens": 32768,
    "skip_special_tokens": True,
    "n": 16,
}

# Data paths
DATASET_PATH = (
    "hf://datasets/AI-MO/aimo-validation-aime/data/train-00000-of-00001.parquet"
)

# Problem prompt template
PROBLEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{{}}. Provide only one \\boxed{{}}.\n The problem: \n{}\n"

# Feature flags
ENABLE_LATEX = False  # Set to True to enable LaTeX compilation

# Evaluation settings
MAJORITY_THRESHOLD = 8  # More than half of samples needed for majority vote
