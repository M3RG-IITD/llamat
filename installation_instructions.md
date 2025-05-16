# Environment Setup Guide

This guide provides step-by-step instructions for setting up a Python environment with PyTorch, NVIDIA Apex, Flash Attention, and Megatron-LLM. Explanatory notes are included to clarify each step and highlight important caveats.

---

## Python Version

**Use Python 3.10.**

- PyTorch does not currently support Python 3.12.
- Ensure your environment uses Python 3.10 (e.g., via `conda` or `venv`).

---

## PyTorch Installation

Install PyTorch, torchvision, and torchaudio with CUDA 11.8 support:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

*This command ensures you get pre-built CUDA-enabled wheels directly from the official PyTorch repository.*

---

## Load GCC Compiler (if on HPC/cluster)

If your system uses environment modules (common on clusters), load GCC 11.2.0:

```bash
module load compiler/gcc/11.2.0
```

*This ensures compatibility with CUDA and C++ extensions required by some packages.*

---

## NVIDIA Apex Installation

Apex is used for mixed-precision training and requires a custom install.

1. **Clone the Apex repository (22.04-dev branch):**

```bash
git clone -b 22.04-dev https://github.com/NVIDIA/apex.git
cd apex
```

2. **Install Apex with CUDA and C++ extensions:**

```bash
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

3. **Patch for Python 3.10 compatibility:**
    - Open:
`/path/to/env/lib/python3.10/site-packages/apex/amp/_initialize.py`
    - Replace line 2 with:

```python
string_classes = str
```


*This patch fixes a compatibility issue with Python 3.10.*

---

## Flash Attention Installation

Flash Attention accelerates transformer models but requires special installation steps.

### Option 1: Build from Source

1. **Clone the repository:**

```bash
git clone https://github.com/Dao-AILab/flash-attn.git
cd flash-attn
```

2. **Install `ninja` for faster compilation:**

```bash
pip install ninja
```

*`ninja` enables parallel builds, greatly speeding up compilation[^1].*
3. **Build and install Flash Attention:**

```bash
python setup.py install
```

    - Use at least 10 CPU cores for faster installation.
    - If your machine has limited RAM, consider limiting parallel jobs:

```bash
MAX_JOBS=4 python setup.py install
```


*If you encounter SSL errors at the end of installation, run:*

```bash
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org flash-attn
```


### Option 2: Pre-built Wheels

- Pre-built wheels are available on the [GitHub releases page](https://github.com/Dao-AILab/flash-attention/releases) for Flash Attention.
- Download the wheel matching your Python, CUDA, and PyTorch versions, then install:

```bash
pip install --no-dependencies --upgrade <wheel-file>.whl
```


*Building from source is more flexible, but pre-built wheels install much faster if available for your configuration[^2].*

---

## Megatron-LLM Installation

Megatron is a large-scale language model training framework.

1. **Clone the repository:**

```bash
git clone https://github.com/epfLLM/Megatron-LLM.git
```

2. **Install build dependency:**

```bash
pip install pybind11
```

3. **Compile C++ helpers:**

```bash
cd Megatron-LLM/megatron/data
make
```

*This compiles `helpers.cpp` into `helpers.so`, which is required by the Megatron codebase.*

---

## Additional Notes

- **Check CUDA compatibility:** Ensure your CUDA toolkit matches the versions required by the packages.
- **Linux is recommended:** Most of these libraries are best supported on Linux. Windows support for Flash Attention is experimental[^1][^3].
- **RAM and CPU:** Building Flash Attention from source can be RAM- and CPU-intensive; adjust `MAX_JOBS` if you encounter memory issues[^1].

---

## Summary Table

| Component | Install Method | Notes |
| :-- | :-- | :-- |
| Python | 3.10 | PyTorch not compatible with 3.12 |
| PyTorch | pip, CUDA 11.8 wheel | Use official index URL |
| Apex | Source, patch required | Patch `_initialize.py` for Python 3.10 |
| Flash Attention | Source or pre-built wheel | Use `ninja` for speed; pre-built wheels are fastest |
| Megatron-LLM | Source, compile helpers | Requires `pybind11` and `make` in `megatron/data` |


---

## Troubleshooting

- **Torch install fails:** Double-check Python version and CUDA compatibility.
- **Apex issues:** Ensure the patch is applied for Python 3.10.
- **Flash Attention build slow:** Confirm `ninja` is installed and working; limit `MAX_JOBS` if low on RAM.
- **SSL errors:** Use the `--trusted-host` pip flags as shown above.
