# 👂 Whisper Fine-Tuner

A high-performance, multi-GPU framework tailored for fine-tuning OpenAI's Whisper model on multilingual datasets. Designed for scalability and efficiency, it enables seamless distributed training, intuitive configuration management, and real-time monitoring. The framework simplifies fine-tuning across diverse domains and languages while optimizing GPU utilization.

## ⚡ Features

- **Scalable Distributed Training**: Efficiently train across multiple GPUs with optimized parallel processing.
- **Flexible Configuration**: Easily customize datasets, training parameters, and checkpoints using YAML files.
- **Real-Time Monitoring**: Track training progress with real-time metrics and validation reports.
- **Inference Engine Integration**: Seamlessly integrate with inference engines to enhance workflows.
- **Performance Evaluation**: Assess model accuracy using built-in CER (Character Error Rate) metrics.

## 📌 Requirements

- **NVIDIA Driver >= 565**
- **CUDA >= 12.6**
- **cuDNN >= 9.6**
- **TensorRT >= 10.6**
- **Python 3.11+**
- **PyTorch >= 2.5.0**

## 📥 Installation

Follow these steps to set up the environment:

```bash
git clone https://github.com/cowcheng/whisper-fine-tuner.git
cd whisper-fine-tuner

python3.11 -m venv .venv
source .venv/bin/activate

pip install -U pip wheel setuptools
pip install -r requirements.txt
```

## 🛠️ Configure Accelerate

Set up the `accelerate` library for distributed training:

```bash
accelerate config
```

Recommended configuration:

```yaml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU
downcast_bf16: "no"
enable_cpu_affinity: false
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

## 🎯 Trainer

### 📝 Transformers Trainer

For details on using the Transformers Trainer, refer to the [Transformers Trainer](./transformers_trainer/).

### _Additional training strategies will be introduced soon to further enhance performance._

## 🚀 Inference Engine

### 🚀 CTranslate2 Engine

For details on using the CTranslate2 Engine, refer to the [CTranslate2 Engine](./ctranslate2_engine/).

### _Upcoming enhancements will include more inference acceleration techniques for improved efficiency._

## 🌱 Contributing

Contributions are welcome! Please fork the repository, create a branch, and submit a pull request.

## 📜 License

This project is licensed under the MIT License.

## 💡 Acknowledgments

Special thanks to the open-source community for their amazing tools and resources.
