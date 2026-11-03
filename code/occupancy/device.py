"""
device.py
---------
Where training runs, and how repeatable it is.

* select_device("auto" | "cpu" | "cuda" | "cuda:N"): "auto" takes the CUDA GPU
  when PyTorch sees one and the CPU otherwise (config train.device; the
  command-line option train --device overrides it).
* configure_determinism() asks PyTorch for deterministic computation wherever
  it is available:
    - CUBLAS_WORKSPACE_CONFIG=:4096:8, which cuBLAS needs for repeatable
      matrix products; it only takes effect if it is set before CUDA starts,
      so the train command sets it before anything touches the GPU;
    - torch.backends.cudnn.deterministic = True and cudnn.benchmark = False
      (no run-time choice between convolution algorithms);
    - torch.use_deterministic_algorithms(True, warn_only=True).
  warn_only is needed because the backward pass of bilinear upsampling, used
  in the decoders of conv_ae and conv_vae, has no deterministic CUDA kernel:
  it accumulates gradients with atomic additions whose order varies from run
  to run. The architecture is not changed to avoid it. Consequently, training
  on a GPU is NOT guaranteed to give bit-identical weights when a run is
  repeated, although the seed still fixes the initial weights (drawn on the
  CPU) and the order of the training frames. On the CPU, a run is repeatable
  for the same thread count and library versions, also across an interruption
  and resume.
* runtime_environment(device) is the record written into every training
  output: device type, GPU name, memory and compute capability, CUDA, cuDNN
  and driver versions, PyTorch and the other library versions, Python, the
  determinism flags, the thread count and the code version.
"""

from __future__ import annotations

import functools
import os
import re
import subprocess
import sys

import torch

from . import provenance

CUBLAS_WORKSPACE = ":4096:8"
GPU_REPEATABILITY = ("not guaranteed bit-identical: the backward pass of bilinear upsampling "
                     "(conv_ae and conv_vae decoders) has no deterministic CUDA kernel; the seed "
                     "fixes the initial weights and the order of the training frames")
CPU_REPEATABILITY = ("repeatable (bit-identical) for the same thread count and library versions, "
                     "including across an interruption and resume")


class DeviceError(RuntimeError):
    """The requested device is not available."""


def configure_determinism() -> dict:
    """Deterministic settings for training (idempotent); returns determinism_flags()."""
    cuda_started = torch.cuda.is_initialized()
    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    flags = determinism_flags()
    flags["cuda_started_before_configuration"] = cuda_started
    return flags


def determinism_flags() -> dict:
    return {"use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def select_device(name: str | None = "auto") -> torch.device:
    """auto | cpu | cuda | cuda:N -> torch.device. "auto" falls back to the CPU."""
    name = str(name or "auto").strip().lower()
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cpu":
        return torch.device("cpu")
    if re.fullmatch(r"cuda(:\d+)?", name):
        if not torch.cuda.is_available():
            raise DeviceError(f"device {name!r} was requested, but PyTorch {torch.__version__} sees no "
                              f"CUDA GPU (built with CUDA {torch.version.cuda})")
        dev = torch.device(name)
        index = dev.index if dev.index is not None else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise DeviceError(f"{name}: only {torch.cuda.device_count()} CUDA device(s)")
        return torch.device("cuda", index)
    raise DeviceError(f"unknown device {name!r}: use auto, cpu, cuda or cuda:N")


@functools.lru_cache(maxsize=1)
def _nvidia_driver() -> str | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else None
    except (OSError, subprocess.SubprocessError):
        return None


def describe(device: torch.device) -> str:
    """One line for the log, e.g. 'cuda:0 (Tesla T4, CUDA 12.4)' or 'cpu (4 threads)'."""
    device = torch.device(device)
    if device.type == "cuda":
        return f"{device} ({torch.cuda.get_device_name(device)}, CUDA {torch.version.cuda})"
    return f"cpu ({torch.get_num_threads()} threads)"


def runtime_environment(device) -> dict:
    """Device, GPU, CUDA/cuDNN/driver, library and Python versions, determinism flags, code version."""
    device = torch.device(device)
    cudnn = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    code = provenance.code_version(with_files=False)
    env = {
        "device": str(device),
        "device_type": device.type,
        "torch": str(torch.__version__),
        "torch_cuda_build": torch.version.cuda,
        "cudnn_version": cudnn,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": None,
        "threads": torch.get_num_threads(),
        "python": sys.version.split()[0],
        "versions": provenance.library_versions(),
        "host": provenance.cpu_info(),
        "determinism": determinism_flags(),
        "code_sha256": code["code_sha256"],
        "git": code["git"],
    }
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        env.update(gpu_name=p.name, gpu_memory_bytes=int(p.total_memory),
                   gpu_compute_capability=f"{p.major}.{p.minor}", gpu_count=torch.cuda.device_count(),
                   nvidia_driver=_nvidia_driver(), bit_identical_expected=False,
                   repeatability=GPU_REPEATABILITY)
    else:
        env.update(bit_identical_expected=True, repeatability=CPU_REPEATABILITY)
    return env
