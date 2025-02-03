import os

import torch

# Temporarily disable advisory warnings from the Transformers library.
# TODO: Revisit and properly handle warnings in the future.
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

# Enable TensorFloat-32 (TF32) mode for CUDA matrix multiplications.
# TF32 can provide a good balance between performance and precision on compatible hardware.
torch.backends.cuda.matmul.allow_tf32 = True

# Retrieve the number of GPUs available on the current machine.
# This is useful for parallelizing computations across multiple devices.
NUM_GPUS = torch.cuda.device_count()

# Determines the number of worker processes to use for data loading.
# It takes half of the available CPU cores but caps the number at 16 to prevent excessive resource usage.
# This helps in optimizing data loading performance without overwhelming the system.
NUM_WORKERS = min(16, os.cpu_count() // 2)

# Specifies the batch size for writing datasets to disk.
# A larger batch size can improve write efficiency by reducing the number of write operations,
# but it also requires more memory. Here, it's set to process 3,000 samples per batch.
DATASET_WRITER_BATCH_SIZE = 3000

# Defines the sampling rate for audio data in Hertz (Hz).
# All audio files will be resampled to this rate to ensure consistency across the dataset,
# which is crucial for accurate feature extraction and model training.
AUDIO_SAMPLE_RATE = 16000
