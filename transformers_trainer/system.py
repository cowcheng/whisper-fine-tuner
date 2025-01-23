import os

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
