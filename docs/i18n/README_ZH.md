# CudaBridge

**[English](README_EN.md)** | **[日本語](README_JA.md)** | **中文** | **[한국어](../../README.md)**

通过外部NVIDIA eGPU在Apple Silicon Mac上实现CUDA计算的开源项目

## 概述

CudaBridge是一个软件栈，使Apple Silicon (M1/M2/M3/M4) Mac能够通过USB4/Thunderbolt连接的
外部NVIDIA GPU执行CUDA计算。

## 架构

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                        User Applications (CUDA Code)                     │
├─────────────────────────────────────────────────────────────────────────┤
│   Python API (cudabridge.py)  │  CudaBridge Userspace Library            │
│   numpy兼容GPU运算             │  (libcudabridge.dylib / API Layer)       │
├─────────────────────────────────────────────────────────────────────────┤
│   CLI Tool (cudabridge-cli)   │  CUDA Runtime Bridge                     │
│   GPU监控/配置管理             │  (cudaMemcpy, cudaMalloc, kernel等)       │
├─────────────────────────────────────────────────────────────────────────┤
│   eGPU Safety Manager         │  GPU Driver Compatibility Layer           │
│   连接/断开/恢复管理           │  (NVIDIA GPU初始化、命令队列、内存管理)     │
├─────────────────────────────────────────────────────────────────────────┤
│   Logging System              │  PCIe Tunneling Over USB4/Thunderbolt     │
│   结构化日志/轮转              │  (PCIe封装、DMA处理、中断)                │
├─────────────────────────────────────────────────────────────────────────┤
│                       macOS DriverKit Extension                          │
│                    (USB4/Thunderbolt硬件访问)                            │
├─────────────────────────────────────────────────────────────────────────┤
│                         Hardware Layer                                   │
│            Apple Silicon ←→ USB4/TB4 ←→ eGPU Enclosure ←→ NVIDIA GPU    │
└─────────────────────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. GPU驱动CLI (`cudabridge-cli`)

类似nvidia-smi的GPU控制和监控命令行工具。

**主要命令:**

| 命令 | 别名 | 说明 |
|------|------|------|
| `info` | `i` | 显示GPU设备详细信息 |
| `status` | `s` | 显示eGPU连接状态和健康指标 |
| `connect` | `c` | 连接eGPU（自动检测/兼容性检查） |
| `disconnect` | `dc` | 安全断开eGPU（或使用`--force`强制断开） |
| `monitor` | `m` | 实时GPU监控（温度、功率、错误率） |
| `config` | `cf` | GPU设置管理（时钟、功率、风扇、P-state） |
| `diag` | `d` | 运行系统诊断 |
| `log` | `l` | 查看驱动日志 |
| `benchmark` | `b` | 传输和计算性能基准测试 |
| `reset` | `r` | GPU重置（软/硬） |

**CLI使用示例:**

```bash
# 查看GPU信息
cudabridge-cli info

# 连接eGPU
cudabridge-cli connect

# 实时监控（500ms间隔）
cudabridge-cli monitor -i 500

# 查看当前设置
cudabridge-cli config show

# 更改时钟速度
cudabridge-cli config clock 2100 1200

# JSON输出（用于脚本集成）
cudabridge-cli info --json

# 安全断开
cudabridge-cli disconnect

# 强制断开（紧急情况）
cudabridge-cli disconnect --force
```

### 2. eGPU连接安全管理器

管理通过Thunderbolt/USB4连接的eGPU的安全连接、断开和错误恢复。

**安全功能:**
- **兼容性检查**: 仅支持NVIDIA GPU，验证最低带宽/功率要求。检测到不兼容设备时报错并强制断开
- **自动恢复**: 检测到链路错误时，自动尝试软重置 → 硬重置 → 重新连接（最多3次）
- **数据完整性**: 基于CRC32的传输数据验证
- **热保护**: GPU温度超过90°C（可配置）时紧急关停
- **心跳监控**: 定期检查连接状态，超时时自动恢复
- **安全断开**: 等待挂起操作完成 → 释放资源 → 拆除隧道

### 3. 日志系统

通过结构化日志支持GPU驱动调试。

- 6级日志等级: TRACE, DEBUG, INFO, WARN, ERROR, FATAL
- 按类别过滤: GENERAL, DRIVER, EGPU, MEMORY, PCIE, USB4, CUDA, CLI, PYTHON
- 日志文件自动轮转（默认10MB，保留5个文件）
- 线程安全（pthread mutex）

### 4. Python API (`cudabridge.py`)

无需额外CUDA库，使用与numpy几乎相同的语法在Python中进行eGPU CUDA计算。

```python
import numpy as np
import cudabridge as cb

cb.init()

# 将数据传输到GPU
a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)

gpu_a = cb.to_device(a)
gpu_b = cb.to_device(b)

# GPU上运算（直接使用Python运算符！）
gpu_c = gpu_a + gpu_b          # 逐元素加法
gpu_d = gpu_a * gpu_b          # 逐元素乘法

# 将结果转回numpy
result = cb.from_device(gpu_c)
print(result)  # [6.0, 8.0, 10.0, 12.0]

# 矩阵乘法
mat_a = cb.to_device(np.random.rand(100, 200).astype(np.float32))
mat_b = cb.to_device(np.random.rand(200, 50).astype(np.float32))
mat_c = mat_a @ mat_b   # (100, 50) 结果

cb.shutdown()
```

### 5. USB4/Thunderbolt PCIe隧道驱动
- 与Apple Silicon USB4控制器直接通信
- PCIe事务封装到USB4协议
- DMA（直接内存访问）处理
- 热插拔支持

### 6. NVIDIA GPU兼容层
- GPU初始化和重置序列
- BAR（基地址寄存器）内存映射
- 命令提交和完成处理
- 电源管理

### 7. CUDA Runtime Bridge
- CUDA Runtime API兼容接口
- 内核编译和执行
- 内存管理（主机 ↔ 设备）
- 流和事件同步

## 系统要求

### 硬件
- Apple Silicon Mac (M1, M2, M3, M4 系列)
- USB4或Thunderbolt 4端口
- 兼容的eGPU外壳（支持Thunderbolt 3/4）
- NVIDIA GPU（推荐Ampere、Ada Lovelace或更新架构）

### 软件
- macOS 13.0 (Ventura) 或更高版本
- Xcode Command Line Tools
- CMake 3.20+
- Python 3.8+（使用Python API时）
- numpy（使用Python API时）

## 构建方法

```bash
git clone https://github.com/KeyWaveTree/CudaBridge.git
cd CudaBridge
mkdir build && cd build
cmake ..
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)
sudo make install
```

## 快速入门

```bash
cudabridge-cli connect        # 连接eGPU
cudabridge-cli info            # 显示GPU信息
cudabridge-cli status          # 显示状态
cudabridge-cli diag            # 运行诊断
cudabridge-cli disconnect      # 安全断开
```

## 学习指南

### 第1步：从CLI工具开始
CLI工具是最简单的起点——无需编程即可检查GPU状态和更改设置。

### 第2步：使用Python API体验GPU计算
Python API使用与numpy几乎相同的语法进行GPU运算。

### 第3步：使用C API进行底层控制
使用C API可以直接控制内存管理、流和事件。参见 `src/userspace/examples/` 中的示例。

### 第4步：理解eGPU安全管理
理解安全机制有助于构建稳定的应用程序。

### 第5步：利用日志系统
调试时利用日志系统可以更快地定位问题。

## 已知限制

1. **macOS SIP**: 可能需要禁用系统完整性保护
2. **驱动签名**: 开发过程中需要开发者签名
3. **带宽**: USB4理论最大带宽为40Gbps（PCIe 3.0 x4级别）
4. **兼容性**: 并非所有CUDA功能都受支持
5. **模拟模式**: 无需真实eGPU即可进行开发/测试

## 贡献

欢迎贡献！请在提交Pull Request之前先创建Issue。

## 致谢与署名建议 (Attribution Recommendation)

本项目以MIT许可证开源，供所有人自由使用和贡献。虽然法律上没有强制要求，
但为了开源生态的良性循环和创作者的持续动力，我们建议遵循以下准则：

> "如果您使用本项目创造了出色的成果，请注明出处。这对我们来说是巨大的成就感和动力！"

- **署名与致谢**: 如果您在企业或个人项目中使用了本代码来制作产品或服务，请在产品介绍页面、
  博客、演示文稿或源代码注释中简要提及原作者（CudaBridge Contributors）。

- **Fork项目时**: 如果您基于本项目进行Fork或在此基础上发展并发布新项目，请明确标注本项目
  是新作品的基础。

您的一句温暖的致谢，是开发者持续改进项目的最大动力。感谢您！

## 许可证

MIT License - 详见 [LICENSE](../../LICENSE)

## 免责声明

本项目为实验性质，仅供教育目的。不建议在生产环境中使用。
Apple、NVIDIA、CUDA是各公司的商标。
