# 🎯 Transformers Trainer Guide

Leverage Hugging Face's Transformers Trainer framework to fine-tune the Whisper model efficiently.

## 🛠 Fine-Tuning the Model

### 1. Configure Training Settings

Refer to [`train_sample.yaml`](./configs/train_sample.yaml) under the `./configs` directory.

### 2. Preprocess the Dataset

```bash
python prepare_dataset.py -c {config_path}

# Example
python prepare_dataset.py -c ./configs/train_sample.yaml
```

### 3. Start Training

```bash
accelerate launch finetune_model.py -c {config_path}

# Example
accelerate launch finetune_model.py -c ./configs/train_sample.yaml 
```

## 🔍 Running Inference

```bash
python inference_model.py -m {model_dir} -i {input_audio} -p {precision|{fp16,bf16,fp32}}

# Example
python inference_model.py -m ./models/final_checkpoint -i ./sample.wav -p bf16
```

## 📈 Evaluating the Model

### 1. Configure Evaluation Settings

Refer to [`eval_sample.yaml`](./configs/eval_sample.yaml) under the `./configs` directory.

### 2. Run Evaluation

```bash
python evaluate_model.py -c {config_path}

# Example
python evaluate_model.py -c ./configs/eval_sample.yaml
```

## ☁️ Uploading Model to Hugging Face

```bash
python push_model.py -m {model_dir} -id {hf_repo_id} -cm {commit_message} [--private]

# Example
python push_model.py -m ./models/final_checkpoint -id "cowcheng/whisper-tiny" -cm "Upload natively fine-tuned Whisper Tiny model" --private
```
