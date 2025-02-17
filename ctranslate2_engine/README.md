# 🚀 CTranslate2 Engine

CTranslate2 is a high-performance inference engine optimized for neural machine translation and other sequence-to-sequence tasks. It enhances speed and reduces memory usage by leveraging modern hardware capabilities.

## ✨ Features

- **Efficient Inference**: Supports both CPU and GPU execution with optimized quantization.
- **Advanced Quantization**: Reduces model size while improving performance with INT8, FP16, and BF16 precision.
- **Cross-Platform Support**: Seamlessly runs on Linux, macOS, and Windows.
- **Scalability**: Manages multiple requests efficiently with customizable threading and batch settings.

## 🛠 Usage

### 🔄 Converting a Model to CTranslate2 Format

Use the following command to convert a Transformers model into the CTranslate2 format:

```bash
ct2-transformers-converter --model {huggingface_repository_id,model_path} \
    --output_dir {output_directory_path} \
    --quantization {int8,int8_float32,int8_float16,int8_bfloat16,int16,float16,bfloat16,float32} \
    --copy_files tokenizer_config.json preprocessor_config.json

# Example
ct2-transformers-converter --model "cowcheng/whisper-tiny" \
    --output_dir ./faster-whisper-tiny \
    --quantization "bfloat16" \
    --copy_files tokenizer_config.json preprocessor_config.json
```

### 🎙️ Running Inference

Run inference on an audio file using the converted model:

```bash
python inference_model.py --model {model_dir} --input {input_audio}

# Example
python inference_model.py --model ./faster-whisper-tiny --input ./sample.wav
```

### ☁️ Uploading a Model to Hugging Face

Upload the converted model to Hugging Face:

```bash
python push_model.py --model {model_dir} --repo_id {hf_repo_id} --commit "{commit_message}" [--private]

# Example
python push_model.py --model ./faster-whisper-tiny \
    --repo_id "cowcheng/faster-whisper-tiny" \
    --commit "Upload CTranslate2-converted Whisper Tiny model" \
    --private
```
