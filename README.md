<div align="center">

# 🎵 Diffusion-Speech-Tokenizer 🚀

<img src="https://img.shields.io/badge/🔥-TaDiCodec-red?style=for-the-badge" alt="TaDiCodec"/>
<img src="https://img.shields.io/badge/🎯-Text--aware-blue?style=for-the-badge" alt="Text-aware"/>
<img src="https://img.shields.io/badge/🌊-Diffusion-purple?style=for-the-badge" alt="Diffusion"/>
<img src="https://img.shields.io/badge/🗣️-Speech-green?style=for-the-badge" alt="Speech"/>

**🔬 Official PyTorch Implementation of TaDiCodec**

<!-- *A series of works on diffusion-based speech tokenizers* -->

<!-- TODO: replace the paper link to the arXiv link -->
📄 **Paper**: [TaDiCodec: Text-aware Diffusion Speech Tokenizer for Speech Language Modeling](https://arxiv.org/abs/2508.16790)

[![GitHub Stars](https://img.shields.io/github/stars/HeCheng0625/Diffusion-Speech-Tokenizer?style=social)](https://github.com/HeCheng0625/Diffusion-Speech-Tokenizer)
[![arXiv](https://img.shields.io/badge/arXiv-2508.16790-b31b1b.svg)](https://arxiv.org/abs/2508.16790)
[![Demo](https://img.shields.io/badge/🎬%20Demo-tadicodec-green)](https://tadicodec.github.io/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776ab.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/🤗%20HuggingFace-tadicodec-yellow)](https://huggingface.co/amphion/TaDiCodec)

<!-- [![ModelScope](https://img.shields.io/badge/🔮%20ModelScope-tadicodec-blue)](https://modelscope.cn/models/amphion/TaDiCodec) -->


</div>

---

## 📋 Overview

This repository is designed to provide comprehensive implementations for our series of diffusion-based speech tokenizer research works. Currently, it primarily features [**TaDiCodec**](https://arxiv.org/abs/2508.16790), with plans to include additional in-progress works in the future. Specifically, the repository includes:

- 🧠 A simple PyTorch implementation of the TaDiCodec tokenizer
- 🎯 Token-based zero-shot TTS models based on TaDiCodec:
  - 🤖 Autoregressive based TTS models
  - 🌊 [Masked diffusion (a.k.a. Masked Genrative Model (MGM)](https://arxiv.org/abs/2409.00750) based TTS models
- 🏋️ Training scripts for tokenizer and TTS models
- 🤗 [Hugging Face](https://huggingface.co/amphion/tadicodec) and 🔮 [ModelScope](https://modelscope.cn/models/amphion/tadicodec) (to be updated) for easy access to pre-trained models

Short Intro on [**TaDiCodec: Text-aware Diffusion Speech Tokenizer for Speech Language Modeling**](https://arxiv.org/abs/2508.16790):

We introduce the **T**ext-**a**ware **Di**ffusion Transformer Speech **Codec** (TaDiCodec), a novel approach to speech tokenization that employs end-to-end optimization for quantization and reconstruction through a **diffusion autoencoder**, while integrating **text guidance** into the diffusion decoder to enhance reconstruction quality and achieve **optimal compression**. TaDiCodec achieves an extremely low frame rate of **6.25 Hz** and a corresponding bitrate of **0.0875 kbps** with a single-layer codebook for **24 kHz speech**, while maintaining superior performance on critical speech generation evaluation metrics such as Word Error Rate (WER), speaker similarity (SIM), and speech quality (UTMOS).

<!-- Notably, TaDiCodec employs a single-stage, end-to-end training paradigm, and obviating the need for auxiliary pre-trained models. -->

---

## 📢 News & Updates

<div align="center">

### 🔥 Latest Updates 🔥

</div>

<!-- - **🎉 [2025-08-11]** TaDiCodec is accepted by NeurIPS 2025! -->
- **🚀 [2025-08-25]** We release the offical implementation of TaDiCodec and the TTS models based on TaDiCodec.
- **🔥 [2025-08-25]** TaDiCodec paper released! Check out our [arXiv preprint](https://arxiv.org/abs/2508.16790)
- **📦 [2025-08-25]** Added auto-download functionality from Hugging Face for all models!

---

## 🚧 Development Roadmap & TODO List

<div align="center">

**🔥 Current Status: Active Development 🔥**

*This project is under active development. Check back frequently for updates!*

</div>

### 🎯 Core TaDiCodec Implementation
- [x] 🏗️ **Repository Structure Setup**
- [x] 📝 **Documentation Framework**
- [x] 🧠 **TaDiCodec Model Architecture**
  - [x] NAR Llama-style transformers for encoder and decoder architectures
  - [x] text-aware flow matching (diffusion) decoder
  - [x] vocoder for mel2wav
- [x] ⚡ **Inference Pipeline**
  - [x] Basic inference pipeline
  - [x] Auto-download from Hugging Face
  - [ ] Add auto-ASR for text input

### 🎓 Training Infrastructure
- [ ] 🏋️ **TaDiCodec Training Scripts**
- [ ] 💾 **Dataset and Dataloader**

### 🎤 Text-to-Speech Models
- [ ] 🤖 **Autoregressive Models**
  - [x] Model architecture
  - [x] Pre-training models loading and inference
  - [ ] Training scripts
- [ ] 🌊 **Masked Diffusion Models**
  - [x] Model architecture
  - [x] Pre-training models loading and inference
  - [ ] Training scripts

### 📊 Evaluation
- [ ] Add evaluation scripts

### 🪐 Future Works
- [ ] 🛸 **Diffusion-based Speech Tokenizer without text conditioning**

---


## 🤗 Pre-trained Models

### 📦 Model Zoo - Ready to Use!

*Download our pre-trained models for instant inference*

### 🎵 TaDiCodec

| Model | 🤗 Hugging Face | 👷 Status |
|:-----:|:---------------:|:------:|
| **🚀 TaDiCodec** | [![HF](https://img.shields.io/badge/🤗%20HF-TaDiCodec-yellow)](https://huggingface.co/amphion/TaDiCodec) | ✅ |
| **🚀 TaDiCodec-old** | [![HF](https://img.shields.io/badge/🤗%20HF-TaDiCodec--old-yellow)](https://huggingface.co/amphion/TaDiCodec-old) | 🚧 |

*Note: TaDiCodec-old is the old version of TaDiCodec, the TaDiCodec-TTS-AR-Phi-3.5-4B is based on TaDiCodec-old.*

### 🎤 TTS Models

| Model | Type | LLM | 🤗 Hugging Face | 👷 Status |
|:-----:|:----:|:---:|:---------------:|:-------------:|
| **🤖 TaDiCodec-TTS-AR-Qwen2.5-0.5B** | AR | Qwen2.5-0.5B-Instruct | [![HF](https://img.shields.io/badge/🤗%20HF-TaDiCodec--AR--0.5B-yellow)](https://huggingface.co/amphion/TaDiCodec-TTS-AR-Qwen2.5-0.5B) | ✅ |
| **🤖 TaDiCodec-TTS-AR-Qwen2.5-3B** | AR | Qwen2.5-3B-Instruct | [![HF](https://img.shields.io/badge/🤗%20HF-TaDiCodec--AR--3B-yellow)](https://huggingface.co/amphion/TaDiCodec-TTS-AR-Qwen2.5-3B) | ✅ |
| **🤖 TaDiCodec-TTS-AR-Phi-3.5-4B** | AR | Phi-3.5-mini-instruct | [![HF](https://img.shields.io/badge/🤗%20HF-TaDiCodec--AR--4B-yellow)](https://huggingface.co/amphion/TaDiCodec-TTS-AR-Phi-3.5-4B) | 🚧 |
| **🌊 TaDiCodec-TTS-MGM** | MGM | - | [![HF](https://img.shields.io/badge/🤗%20HF-TaDiCodec--MGM-yellow)](https://huggingface.co/amphion/TaDiCodec-TTS-MGM) | ✅ |

- [ ] ModelScope will be updated soon.

### 🔧 Quick Model Usage

```python
# 🤗 Load from Hugging Face with Auto-Download
from models.tts.tadicodec.inference_tadicodec import TaDiCodecPipline
from models.tts.llm_tts.inference_llm_tts import TTSInferencePipeline
from models.tts.llm_tts.inference_mgm_tts import MGMInferencePipeline

# Load TaDiCodec tokenizer (auto-downloads from HF if not found locally)
tokenizer = TaDiCodecPipline.from_pretrained("amphion/TaDiCodec")

# Load AR TTS model (auto-downloads from HF if not found locally)
tts_model = TTSInferencePipeline.from_pretrained(
    tadicodec_path="amphion/TaDiCodec",
    llm_path="amphion/TaDiCodec-TTS-AR-Qwen2.5-0.5B"
)

# Load MGM TTS model (auto-downloads from HF if not found locally)
mgm_model = MGMInferencePipeline.from_pretrained(
    tadicodec_path="amphion/TaDiCodec",
    mgm_path="amphion/TaDiCodec-TTS-MGM-0.6B"
)

# You can also use local paths if you have models downloaded
# tts_model = TTSInferencePipeline.from_pretrained(
#     tadicodec_path="./ckpt/TaDiCodec",
#     llm_path="./ckpt/TaDiCodec-TTS-AR-Qwen2.5-0.5B"
# )
```

---

## 🚀 Quick Start

### Installation

#### Conda Linux

Select one of the 2 pytorch lines depending on your hardware

```bash
# Clone the repository
git clone https://github.com/HeCheng0625/Diffusion-Speech-Tokenizer.git
cd Diffusion-Speech-Tokenizer

# Install dependencies
conda create -n tadicodec python=3.10
conda activate tadicodec
pip install setuptools wheel psutil packaging ninja numpy hf_xet

# pytorch
# CUDA
pip install torch==2.8.0 torchaudio --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128
# OR CPU only
pip install torch==2.8.0 torchaudio

pip install flash_attn==2.7.4.post1
pip install -r requirements.txt
```

#### Conda Windows

This assumes you are using powershell
Select one of the 2 pytorch lines depending on your hardware
Select one of the 2 flash_attn sections on if you want to use a pre-build whl or to compile your own

```powershell
# Clone the repository
git clone https://github.com/HeCheng0625/Diffusion-Speech-Tokenizer.git
cd Diffusion-Speech-Tokenizer

# Install dependencies
conda create -n tadicodec python=3.10
conda activate tadicodec
pip install setuptools wheel psutil packaging ninja numpy hf_xet

# pytorch
# CUDA
pip install torch==2.8.0 torchaudio --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128
# OR CPU only
pip install torch==2.8.0 torchaudio

# flash_attn
# use a pre-built wheel
pip install https://huggingface.co/kim512/flash_attn-2.7.4.post1/resolve/main/flash_attn-2.7.4.post1-cu128-torch2.8.0-cp310-cp310-win_amd64.whl
# OR compile your own, change MAX_JOBS to match your CPU, ideally 4 to 8. If you have lots of RAM make this number smaller.
$Env:MAX_JOBS="6"
$Env:CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
pip install -v flash-attn==2.7.4.post1 --no-build-isolation


# install requirements
pip install -r requirements.txt
```

#### UV Linux

Select one of the 2 pytorch lines depending on your hardware

```bash
# Clone the repository
git clone https://github.com/HeCheng0625/Diffusion-Speech-Tokenizer.git
cd Diffusion-Speech-Tokenizer

# Install python and dependencies
uv python install 3.10
uv venv --python 3.10
uv pip install setuptools wheel psutil packaging ninja numpy hf_xet

# pytorch
# CUDA
pip install torch==2.8.0 torchaudio --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128
# CPU only
pip install torch==2.8.0 torchaudio

pip install flash_attn==2.7.4.post1
pip install -r requirements.txt
```

#### UV Windows

This assumes you are using powershell
Select one of the 2 pytorch lines depending on your hardware
Select one of the 2 flash_attn sections on if you want to use a pre-build whl or to compile your own
 
```powershell
# Clone the repository
git clone https://github.com/HeCheng0625/Diffusion-Speech-Tokenizer.git
cd Diffusion-Speech-Tokenizer

# Install python and dependencies
uv python install 3.10
uv venv --python 3.10
uv pip install setuptools wheel psutil packaging ninja numpy hf_xet

# pytorch
# CUDA
uv pip install torch==2.8.0 torchaudio --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128
# OR CPU only
uv pip install torch==2.8.0 torchaudio

# flash_attn
# use a pre-built wheel
uv pip install https://huggingface.co/kim512/flash_attn-2.7.4.post1/resolve/main/flash_attn-2.7.4.post1-cu128-torch2.8.0-cp310-cp310-win_amd64.whl
# OR compile your own, change MAX_JOBS to match your CPU, ideally 4 to 8. If you have lots of RAM make this number smaller.
$Env:MAX_JOBS="6"
$Env:CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
uv pip install -v flash-attn==2.7.4.post1 --no-build-isolation


# install requirements
uv pip install -r requirements.txt
```

### 🔄 Auto-Download from Hugging Face

All models support automatic download from Hugging Face! Simply use the Hugging Face model ID instead of local paths:

```python
# Models will be automatically downloaded on first use
from models.tts.tadicodec.inference_tadicodec import TaDiCodecPipline
from models.tts.llm_tts.inference_llm_tts import TTSInferencePipeline

# Auto-download TaDiCodec
tokenizer = TaDiCodecPipline.from_pretrained("amphion/TaDiCodec")

# Auto-download TTS pipeline (downloads both TaDiCodec and LLM)
pipeline = TTSInferencePipeline.from_pretrained(
    tadicodec_path="amphion/TaDiCodec",
    llm_path="amphion/TaDiCodec-TTS-AR-Qwen2.5-0.5B"
)
```

**Note**: Models are cached locally after first download for faster subsequent use.

### Basic Usage

**Please refer to the [use_examples](./use_examples) folder for more detailed usage examples.**


```bash
cd Diffusion-Speech-Tokenizer

# conda linux and windows
conda activate tadicodec
# UV linux
./venv/Scripts/activate.sh
# UV powershell
.\.venv\Scripts\activate.ps1

# download models
python test_auto_download.py

cd use_examples
python test_auto_download.py
python test_llm_tts.py
python test_mgm_tts.py
python test_rec.py

```

#### Speech Tokenization and Reconstruction

```python
# Example: Using TaDiCodec for speech tokenization
import torch
import soundfile as sf
from models.tts.tadicodec.inference_tadicodec import TaDiCodecPipline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Auto-download from Hugging Face if not found locally
pipe = TaDiCodecPipline.from_pretrained(ckpt_dir="amphion/TaDiCodec", device=device)

# Or use local path if you have models downloaded
# pipe = TaDiCodecPipline.from_pretrained(ckpt_dir="./ckpt/TaDiCodec", device=device)

# Text of the prompt audio
prompt_text = "In short, we embarked on a mission to make America great again, for all Americans."
# Text of the target audio
target_text = "But to those who knew her well, it was a symbol of her unwavering determination and spirit."

# Input audio path of the prompt audio
prompt_speech_path = "./use_examples/test_audio/trump_0.wav"
# Input audio path of the target audio
speech_path = "./use_examples/test_audio/trump_1.wav"

rec_audio = pipe(
    text=target_text,
    speech_path=speech_path,
    prompt_text=prompt_text,
    prompt_speech_path=prompt_speech_path
)
sf.write("./use_examples/test_audio/trump_rec.wav", rec_audio, 24000)
```

#### Zero-shot TTS with TaDiCodec

```python
import torch
import soundfile as sf
from models.tts.llm_tts.inference_llm_tts import TTSInferencePipeline
# from models.tts.llm_tts.inference_mgm_tts import MGMInferencePipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create AR TTS pipeline with auto-download from Hugging Face
pipeline = TTSInferencePipeline.from_pretrained(
    tadicodec_path="amphion/TaDiCodec",
    llm_path="amphion/TaDiCodec-TTS-AR-Qwen2.5-0.5B",
    device=device,
)

# Or use local paths if you have models downloaded
# pipeline = TTSInferencePipeline.from_pretrained(
#     tadicodec_path="./ckpt/TaDiCodec",
#     llm_path="./ckpt/TaDiCodec-TTS-AR-Qwen2.5-0.5B",
#     device=device,
# )

# Generate speech with code-switching support
audio = pipeline(
    text="但是 to those who 知道 her well, it was a 标志 of her unwavering 决心 and spirit.",
    prompt_text="In short, we embarked on a mission to make America great again, for all Americans.",
    prompt_speech_path="./use_examples/test_audio/trump_0.wav",
)

sf.write("./use_examples/test_audio/lm_tts_output.wav", audio, 24000)
```

---

## 📊 Performance

### 🏆 TaDiCodec Performance Metrics

![TaDiCodec Performance Metrics](./pics/tadicodec_performance.png)

### 🎭 TTS Model Performance Metrics

![TTS Model Performance Metrics](./pics/tts_performance.png)

---

## 🛠️ Training Examples

### Training TaDiCodec

- [ ] To be updated

### Training TTS Models

- [ ] To be updated

---

## 📚 Citation

If you find this repository useful, please cite our paper:

TaDiCodec:
```bibtex
@article{tadicodec2025,
  title={TaDiCodec: Text-aware Diffusion Speech Tokenizer for Speech Language Modeling},
  author={Yuancheng Wang, Dekun Chen, Xueyao Zhang, Junan Zhang, Jiaqi Li, Zhizheng Wu},
  journal={arXiv preprint},
  year={2025},
  url={https://arxiv.org/abs/2508.16790}
}
```

Amphion:
```bibtex
@inproceedings{amphion,
    author={Xueyao Zhang and Liumeng Xue and Yicheng Gu and Yuancheng Wang and Jiaqi Li and Haorui He and Chaoren Wang and Ting Song and Xi Chen and Zihao Fang and Haopeng Chen and Junan Zhang and Tze Ying Tang and Lexiao Zou and Mingxuan Wang and Jun Han and Kai Chen and Haizhou Li and Zhizheng Wu},
    title={Amphion: An Open-Source Audio, Music and Speech Generation Toolkit},
    booktitle={{IEEE} Spoken Language Technology Workshop, {SLT} 2024},
    year={2024}
}
```

MaskGCT:
```bibtex
@inproceedings{wang2024maskgct,
  author={Wang, Yuancheng and Zhan, Haoyue and Liu, Liwei and Zeng, Ruihong and Guo, Haotian and Zheng, Jiachen and Zhang, Qiang and Zhang, Xueyao and Zhang, Shunsi and Wu, Zhizheng},
  title={MaskGCT: Zero-Shot Text-to-Speech with Masked Generative Codec Transformer},
  booktitle    = {{ICLR}},
  publisher    = {OpenReview.net},
  year         = {2025}
}
```

---

## 📄 License

TaDiCodec is licensed under the Apache 2.0.

## 🙏 Acknowledgments

- **MGM-based TTS** is built upon [MaskGCT](https://github.com/open-mmlab/Amphion/tree/main/models/tts/maskgct).

- **Vocos vocoder** is built upon [Vocos](https://github.com/gemelo-ai/vocos).

- **NAR Llama-style transformers** is built upon [transformers](https://github.com/huggingface/transformers).

- **(Binary Spherical Quantization) BSQ** is built upon [vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch) and [bsq-vit](https://github.com/zhaoyue-zephyrus/bsq-vit).

- **Training codebase** is built upon [Amphion](https://github.com/open-mmlab/Amphion) and [accelerate](https://github.com/huggingface/accelerate).

---

<div align="center">

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=HeCheng0625/Diffusion-Speech-Tokenizer&type=Date)](https://star-history.com/#HeCheng0625/Diffusion-Speech-Tokenizer&Date)

---

### 💫 Made with ❤️ for the Speech AI Community

**🚀 If you find this project helpful, please consider giving it a ⭐!**

<div style="margin: 20px 0;">
  <img src="https://img.shields.io/badge/🎵-Speech%20AI-ff6b6b?style=for-the-badge" alt="Speech AI"/>
  <img src="https://img.shields.io/badge/🤖-Deep%20Learning-4ecdc4?style=for-the-badge" alt="Deep Learning"/>
  <img src="https://img.shields.io/badge/🔊-Audio%20Processing-45b7d1?style=for-the-badge" alt="Audio Processing"/>
</div>

*Building the future of speech synthesis, one token at a time* 🎯

---

<sub>🔔 **Stay Updated**: Watch this repository for the latest developments in diffusion-based speech tokenization!</sub>

</div>
