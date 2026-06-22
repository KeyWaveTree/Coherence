# CudaBridge

**English** | **[日本語](README_JA.md)** | **[中文](README_ZH.md)** | **[한국어](../../README.md)**

An open-source project enabling CUDA computing on Apple Silicon Macs via external NVIDIA eGPUs

## Overview

CudaBridge is a software stack that enables CUDA computing on external NVIDIA GPUs
connected via USB4/Thunderbolt to Apple Silicon (M1/M2/M3/M4) Macs.

## Architecture

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                        User Applications (CUDA Code)                     │
├─────────────────────────────────────────────────────────────────────────┤
│   Python API (cudabridge.py)  │  CudaBridge Userspace Library            │
│   numpy-compatible GPU ops    │  (libcudabridge.dylib / API Layer)       │
├─────────────────────────────────────────────────────────────────────────┤
│   CLI Tool (cudabridge-cli)   │  CUDA Runtime Bridge                     │
│   GPU monitoring/config       │  (cudaMemcpy, cudaMalloc, kernel, etc.)  │
├─────────────────────────────────────────────────────────────────────────┤
│   eGPU Safety Manager         │  GPU Driver Compatibility Layer           │
│   Connect/disconnect/recover  │  (NVIDIA GPU init, cmd queue, memory)    │
├─────────────────────────────────────────────────────────────────────────┤
│   Logging System              │  PCIe Tunneling Over USB4/Thunderbolt     │
│   Structured logs/rotation    │  (PCIe encapsulation, DMA, interrupts)   │
├─────────────────────────────────────────────────────────────────────────┤
│                       macOS DriverKit Extension                          │
│                    (USB4/Thunderbolt hardware access)                    │
├─────────────────────────────────────────────────────────────────────────┤
│                         Hardware Layer                                   │
│            Apple Silicon ←→ USB4/TB4 ←→ eGPU Enclosure ←→ NVIDIA GPU    │
└─────────────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. GPU Driver CLI (`cudabridge-cli`)

A command-line tool for GPU control and monitoring, similar to nvidia-smi.

**Commands:**

| Command | Alias | Description |
|---------|-------|-------------|
| `info` | `i` | Show GPU device details |
| `status` | `s` | Show eGPU connection status and health metrics |
| `connect` | `c` | Connect eGPU (auto-detect/compatibility check) |
| `disconnect` | `dc` | Safe eGPU disconnect (or `--force` for forced) |
| `monitor` | `m` | Real-time GPU monitoring (temp, power, error rate) |
| `config` | `cf` | GPU settings (clock, power, fan, P-state) |
| `diag` | `d` | Run system diagnostics |
| `log` | `l` | View driver logs |
| `benchmark` | `b` | Run transfer and compute benchmarks |
| `reset` | `r` | GPU reset (soft/hard) |

**CLI Examples:**

```bash
# Show GPU info
cudabridge-cli info

# Connect eGPU
cudabridge-cli connect

# Real-time monitoring (500ms interval)
cudabridge-cli monitor -i 500

# Show current settings
cudabridge-cli config show

# Change clock speeds
cudabridge-cli config clock 2100 1200

# Set power limit
cudabridge-cli config power 350

# JSON output (for scripting)
cudabridge-cli info --json

# Safe disconnect
cudabridge-cli disconnect

# Force disconnect (emergency)
cudabridge-cli disconnect --force
```

**Global Options:**

| Option | Description |
|--------|-------------|
| `-v, --verbose` | Verbose output (includes debug logs) |
| `-j, --json` | JSON format output |
| `--no-color` | Disable ANSI colors |
| `-d, --device N` | Target device index |
| `-f, --force` | Force execution |

### 2. eGPU Connection Safety Manager

Manages safe connection, disconnection, and error recovery for eGPUs connected via Thunderbolt/USB4.

**Connection Sequence:**
1. Device detection (USB4/Thunderbolt scan)
2. Compatibility check (vendor, bandwidth, power)
3. Thunderbolt security authentication
4. PCIe tunnel setup
5. GPU initialization
6. Health monitoring start

**Safety Features:**
- **Compatibility Check**: NVIDIA GPUs only, minimum bandwidth/power verification. Incompatible devices are reported and forcefully disconnected
- **Auto Recovery**: On link errors, automatically attempts soft reset -> hard reset -> reconnection (up to 3 retries)
- **Data Integrity**: CRC32-based transfer data verification
- **Thermal Protection**: Emergency shutdown when GPU temperature exceeds 90°C (configurable)
- **Heartbeat Monitoring**: Periodic connection status checks, auto-recovery on timeout
- **Safe Disconnect**: Wait for pending operations -> release resources -> teardown tunnel

**Safety Policy Settings:**

| Setting | Default | Description |
|---------|---------|-------------|
| `max_retry_count` | 3 | Maximum retry attempts |
| `retry_delay_ms` | 500 | Retry interval (ms) |
| `heartbeat_interval_ms` | 1000 | Heartbeat period (ms) |
| `heartbeat_timeout_ms` | 5000 | Heartbeat timeout (ms) |
| `error_rate_threshold` | 0.01 | Error rate threshold (1%) |
| `thermal_limit` | 90 | Temperature limit (°C) |
| `enable_auto_recovery` | 1 | Enable auto recovery |
| `enable_data_integrity` | 1 | Enable data integrity checks |

### 3. Logging System

Structured logging for GPU driver debugging.

**Features:**
- 6 log levels: TRACE, DEBUG, INFO, WARN, ERROR, FATAL
- Category filtering: GENERAL, DRIVER, EGPU, MEMORY, PCIE, USB4, CUDA, CLI, PYTHON
- Automatic log file rotation (default 10MB, 5 files retained)
- ANSI color console output
- Thread-safe (pthread mutex)
- Custom callback support

**Log file location:** `/tmp/cudabridge_logs/`

### 4. Python API (`cudabridge.py`)

Use eGPU CUDA computing from Python with virtually the same syntax as numpy - no additional CUDA libraries required.

**Core Concept:**
- numpy array -> `cb.to_device()` -> GPU memory
- GPU operations (add, multiply, matmul, etc.)
- GPU memory -> `cb.from_device()` -> numpy array

**Python Example:**

```python
import numpy as np
import cudabridge as cb

cb.init()

# Transfer data to GPU
a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)

gpu_a = cb.to_device(a)
gpu_b = cb.to_device(b)

# GPU operations (standard Python operators work!)
gpu_c = gpu_a + gpu_b          # Element-wise addition
gpu_d = gpu_a * gpu_b          # Element-wise multiplication

# Get results back as numpy
result = cb.from_device(gpu_c)
print(result)  # [6.0, 8.0, 10.0, 12.0]

# Reduction operations
total = gpu_a.sum()
avg = gpu_a.mean()

# Matrix multiplication
mat_a = cb.to_device(np.random.rand(100, 200).astype(np.float32))
mat_b = cb.to_device(np.random.rand(200, 50).astype(np.float32))
mat_c = mat_a @ mat_b   # (100, 50) result

cb.shutdown()
```

### 5. USB4/Thunderbolt PCIe Tunneling Driver
- Direct communication with Apple Silicon USB4 controller
- PCIe transaction encapsulation in USB4 protocol
- DMA (Direct Memory Access) handling
- Hot-plug support

### 6. NVIDIA GPU Compatibility Layer
- GPU initialization and reset sequences
- BAR (Base Address Register) memory mapping
- Command submission and completion handling
- Power management

### 7. CUDA Runtime Bridge
- CUDA Runtime API compatible interface
- Kernel compilation and execution
- Memory management (Host <-> Device)
- Stream and event synchronization

## System Requirements

### Hardware
- Apple Silicon Mac (M1, M2, M3, M4 series)
- USB4 or Thunderbolt 4 port
- Compatible eGPU enclosure (Thunderbolt 3/4 support)
- NVIDIA GPU (Ampere, Ada Lovelace, or newer recommended)

### Software
- macOS 13.0 (Ventura) or later
- Xcode Command Line Tools
- CMake 3.20+
- Python 3.8+ (for Python API)
- numpy (for Python API)

## Building

```bash
git clone https://github.com/KeyWaveTree/CudaBridge.git
cd CudaBridge
mkdir build && cd build
cmake ..
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)
sudo make install
```

Build outputs:
- `cudabridge-cli` - GPU driver CLI tool
- `libcudabridge.so` / `libcudabridge.dylib` - Shared library
- `libcudabridge.a` - Static library

## Quick Start Guide

### Using the CLI

```bash
cudabridge-cli connect        # Connect eGPU
cudabridge-cli info            # Show GPU info
cudabridge-cli status          # Show status
cudabridge-cli diag            # Run diagnostics
cudabridge-cli disconnect      # Safe disconnect
```

### Using the Python API

```python
import numpy as np
import cudabridge as cb

cb.init()

a = cb.to_device(np.array([1, 2, 3, 4], dtype=np.float32))
b = cb.to_device(np.array([5, 6, 7, 8], dtype=np.float32))

c = a + b                           # Addition on GPU
result = cb.from_device(c)          # Get result as numpy
print(result)                       # [6. 8. 10. 12.]

cb.shutdown()
```

## Learning Guide

### Step 1: Start with the CLI

The CLI is the easiest starting point - check GPU status and change settings without coding.

### Step 2: Try GPU Computing with the Python API

The Python API uses virtually the same syntax as numpy for GPU operations.

### Step 3: Low-level Control with the C API

Use the C API for direct control over memory management, streams, and events.
See `src/userspace/examples/` for examples.

### Step 4: Understand eGPU Safety Management

Understanding the safety mechanisms helps you build stable applications.

### Step 5: Use the Logging System

Leverage the logging system for faster debugging.

```bash
cudabridge-cli -v connect                    # Verbose mode
tail -f /tmp/cudabridge_logs/*.log           # Live log monitoring
cudabridge-cli log --level ERROR             # Error logs only
```

## Known Limitations

1. **macOS SIP**: System Integrity Protection may need to be disabled
2. **Driver Signing**: Developer signing required during development
3. **Bandwidth**: USB4 theoretical max is 40Gbps (PCIe 3.0 x4 level)
4. **Compatibility**: Not all CUDA features may be supported
5. **Simulation Mode**: Development/testing possible without a real eGPU

## Contributing

Contributions are welcome! Please create an issue before submitting a Pull Request.

## Attribution Recommendation

This project is released under the MIT License for free use and contribution.
While not legally required, we kindly encourage the following for the sustainability
of the open-source ecosystem:

> "If you've built something great with this project, please credit us. It means the world to us!"

- **Credit and Thanks**: If you use this code in a product or service, please mention the
  original authors (CudaBridge Contributors) in your product page, blog, or source code comments.

- **Forking**: If you fork or build upon this project, please explicitly acknowledge that
  this project served as the foundation.

Your acknowledgment is the greatest motivation for us to keep improving this project. Thank you!

## License

MIT License - See [LICENSE](../../LICENSE) for details

## Disclaimer

This project is experimental and provided for educational purposes.
Production use is not recommended.
Apple, NVIDIA, and CUDA are trademarks of their respective companies.
