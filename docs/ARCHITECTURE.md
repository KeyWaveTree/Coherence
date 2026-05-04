# Coherence Architecture

## 개요

Coherence는 Apple Silicon Mac에서 USB4/Thunderbolt를 통해 연결된 NVIDIA eGPU에서
CUDA 연산을 수행할 수 있게 해주는 소프트웨어 스택입니다.

## 레이어 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                    Application Layer                             │
│                   (CUDA 애플리케이션)                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Coherence Public API                           │
│                    (cudabridge.h)                               │
│   - cbMalloc, cbMemcpy, cbLaunchKernel 등                       │
│   - CUDA 호환 모드 지원                                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   CUDA Runtime Bridge                           │
│                  (cuda_runtime.c)                               │
│   - CUDA Runtime API 호환 레이어                                 │
│   - 디바이스 관리, 스트림, 이벤트                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Memory Manager                               │
│                 (memory_manager.c)                              │
│   - GPU 메모리 할당/해제                                         │
│   - 통합 메모리 (Unified Memory)                                 │
│   - 페이지 테이블 관리                                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   NVIDIA GPU Driver                             │
│                   (nvidia_gpu.c)                                │
│   - GPU 초기화 및 리셋                                           │
│   - 레지스터 접근 (BAR0)                                         │
│   - 명령 제출 (Push Buffer)                                      │
│   - 커널 실행                                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PCIe Tunnel Layer                             │
│                   (pcie_tunnel.c)                               │
│   - PCIe Configuration Space 접근                               │
│   - MMIO (Memory-Mapped I/O)                                    │
│   - DMA 전송                                                     │
│   - TLP (Transaction Layer Packet) 처리                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  USB4 Controller Driver                         │
│                  (usb4_controller.c)                            │
│   - USB4 라우터 관리                                             │
│   - PCIe 터널 생성/관리                                          │
│   - 대역폭 할당                                                  │
│   - 핫플러그 처리                                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      macOS IOKit                                │
│           (IOThunderboltController, IOUSBHostDevice)            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Hardware                                  │
│     Apple Silicon ←→ USB4/TB4 ←→ eGPU Enclosure ←→ NVIDIA GPU  │
└─────────────────────────────────────────────────────────────────┘
```

## 주요 컴포넌트 상세

### 1. USB4 Controller Driver

Apple Silicon의 USB4 컨트롤러와 직접 통신합니다.

**주요 기능:**
- USB4 라우터 열거 및 관리
- PCIe 터널 생성 (USB4 프로토콜을 통한 PCIe 캡슐화)
- 대역폭 관리 (USB4는 최대 40Gbps, PCIe 터널에 할당 가능)
- 핫플러그 이벤트 처리

**주요 구조체:**
```c
USB4ControllerContext  // 컨트롤러 전역 상태
USB4Router             // USB4 라우터 (호스트, 디바이스, 허브)
USB4Adapter            // 라우터의 어댑터 (PCIe, DP, USB3 등)
USB4PCIeTunnel         // PCIe 터널 정보
```

### 2. PCIe Tunnel Layer

USB4를 통해 PCIe 트랜잭션을 터널링합니다.

**주요 기능:**
- PCIe Configuration Space 읽기/쓰기
- BAR (Base Address Register) 메모리 매핑
- MMIO 액세스
- DMA 전송

**TLP 타입:**
- Configuration Read/Write (Type 0/1)
- Memory Read/Write (32/64-bit)
- Completion

### 3. NVIDIA GPU Driver

NVIDIA GPU 하드웨어를 직접 제어합니다.

**주요 기능:**
- GPU 초기화 및 리셋 시퀀스
- 레지스터 접근 (PMC, PFIFO, PGRAPH 등)
- 채널 (Channel) 관리
- Push Buffer를 통한 명령 제출
- 커널 실행

**지원 아키텍처:**
- Turing (Compute 7.5)
- Ampere (Compute 8.x)
- Ada Lovelace (Compute 8.9)
- Hopper (Compute 9.0)

### 4. Memory Manager

GPU 메모리를 효율적으로 관리합니다.

**메모리 풀:**
- Device Pool: GPU VRAM
- Pinned Pool: 호스트 페이지 락 메모리
- Managed Pool: 통합 메모리

**할당 전략:**
- First Fit 알고리즘
- 페이지 크기 정렬 (4KB, 64KB, 2MB)
- 블록 병합 (Coalescing)

### 5. CUDA Runtime Bridge

NVIDIA CUDA Runtime API와 호환되는 인터페이스를 제공합니다.

**지원 API:**
- 디바이스 관리: cudaGetDeviceCount, cudaSetDevice, ...
- 메모리: cudaMalloc, cudaFree, cudaMemcpy, ...
- 스트림: cudaStreamCreate, cudaStreamSynchronize, ...
- 이벤트: cudaEventCreate, cudaEventRecord, ...
- 커널 실행: cudaLaunchKernel

## 데이터 흐름

### Host → Device (H2D) 전송

```
1. 애플리케이션에서 cbMemcpy(d_ptr, h_ptr, size, H2D) 호출
2. CUDA Runtime이 NV GPU 드라이버의 memcpy_h2d 호출
3. GPU 드라이버가 DMA 요청 생성
4. PCIe Tunnel이 TLP (Memory Write) 생성
5. USB4 Controller가 TLP를 USB4 패킷으로 캡슐화
6. USB4를 통해 eGPU로 전송
7. eGPU의 GPU가 VRAM에 데이터 쓰기
```

### Device → Host (D2H) 전송

```
1. cbMemcpy(h_ptr, d_ptr, size, D2H) 호출
2. GPU 드라이버가 DMA 읽기 요청 생성
3. PCIe Tunnel이 TLP (Memory Read) 생성
4. USB4를 통해 요청 전송
5. GPU가 데이터 읽어서 Completion TLP로 응답
6. 호스트 메모리에 데이터 쓰기
```

## 대역폭 고려사항

### USB4/Thunderbolt 대역폭

| 인터페이스 | 총 대역폭 | PCIe 터널 할당 |
|-----------|----------|---------------|
| USB4 Gen 3 | 40 Gbps | ~32 Gbps |
| TB4 | 40 Gbps | ~32 Gbps |
| TB3 | 40 Gbps | ~22 Gbps |

### PCIe 비교

| PCIe 버전 | x4 대역폭 | x8 대역폭 | x16 대역폭 |
|----------|----------|----------|-----------|
| PCIe 3.0 | 32 Gbps | 64 Gbps | 128 Gbps |
| PCIe 4.0 | 64 Gbps | 128 Gbps | 256 Gbps |

USB4를 통한 eGPU는 대략 PCIe 3.0 x4 수준의 대역폭을 제공합니다.

## 제한사항

1. **대역폭**: 내장 PCIe 슬롯 대비 대역폭 제한
2. **지연 시간**: USB4 프로토콜 오버헤드로 인한 추가 지연
3. **드라이버 서명**: macOS에서 커널 확장 로드를 위해 개발자 서명 필요
4. **SIP**: System Integrity Protection 설정 필요할 수 있음
5. **호환성**: 모든 CUDA 기능이 지원되지 않을 수 있음

## 향후 개선 사항

1. **실제 CUDA 커널 지원**: PTX 컴파일러 통합
2. **Multi-GPU 지원**: 여러 eGPU 동시 사용
3. **더 나은 메모리 관리**: 페이지 폴트 기반 통합 메모리
4. **성능 최적화**: DMA 버퍼링, 프리페칭
5. **디버깅 도구**: GPU 디버거, 프로파일러 통합
