import os

import torch

# Suppress advisory warnings from the Transformers library
# TODO Fix the issue causing the warning to be raised
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

# Enable TensorFloat-32 (TF32) support for faster matrix multiplications on CUDA
torch.backends.cuda.matmul.allow_tf32 = True

# Get the number of available GPUs
NUM_GPUS = torch.cuda.device_count()

# Set the number of workers for data processing (capped at 16 or half the CPU cores)
NUM_WORKERS = min(16, os.cpu_count() // 2)

# Define the batch size for dataset writing operations
DATASET_WRITER_BATCH_SIZE = 3000

# Set the audio sample rate for processing
AUDIO_SAMPLE_RATE = 16000
