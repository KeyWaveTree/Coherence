# Coherence

**[English](docs/i18n/README_EN.md)** | **[日本語](docs/i18n/README_JA.md)** | **[中文](docs/i18n/README_ZH.md)** | **한국어**

Apple Silicon Mac에서 외부 NVIDIA eGPU를 통해 CUDA 연산을 가능하게 하는 오픈소스 프로젝트

## 개요

CudaBridge는 Apple Silicon (M1/M2/M3/M4) 기반 Mac에서 USB4/Thunderbolt를 통해 연결된
외부 NVIDIA GPU에서 CUDA 연산을 수행할 수 있게 해주는 소프트웨어 스택입니다.

## 아키텍처

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                        User Applications (CUDA Code)                     │
├─────────────────────────────────────────────────────────────────────────┤
│   Python API (cudabridge.py)  │  CudaBridge Userspace Library            │
│   numpy 호환 GPU 연산          │  (libcudabridge.dylib / API Layer)       │
├─────────────────────────────────────────────────────────────────────────┤
│   CLI Tool (cudabridge-cli)   │  CUDA Runtime Bridge                     │
│   GPU 모니터링/설정 관리        │  (cudaMemcpy, cudaMalloc, kernel 등)      │
├─────────────────────────────────────────────────────────────────────────┤
│   eGPU Safety Manager         │  GPU Driver Compatibility Layer           │
│   연결/해제/복구 관리           │  (NVIDIA GPU 초기화, 명령 큐, 메모리 관리)  │
├─────────────────────────────────────────────────────────────────────────┤
│   Logging System              │  PCIe Tunneling Over USB4/Thunderbolt     │
│   구조화된 로그/로테이션        │  (PCIe 패킷 캡슐화, DMA 처리, 인터럽트)    │
├─────────────────────────────────────────────────────────────────────────┤
│                       macOS DriverKit Extension                          │
│                    (USB4/Thunderbolt 하드웨어 접근)                       │
├─────────────────────────────────────────────────────────────────────────┤
│                         Hardware Layer                                   │
│            Apple Silicon ←→ USB4/TB4 ←→ eGPU Enclosure ←→ NVIDIA GPU    │
└─────────────────────────────────────────────────────────────────────────┘
```

## 주요 컴포넌트

### 1. GPU Driver CLI (`cudabridge-cli`)

nvidia-smi와 유사한 GPU 제어/모니터링 명령줄 도구입니다.

**주요 명령:**

| 명령 | 별칭 | 설명 |
|------|------|------|
| `info` | `i` | GPU 디바이스 상세 정보 표시 |
| `status` | `s` | eGPU 연결 상태 및 건강 지표 표시 |
| `connect` | `c` | eGPU 연결 (자동 감지/호환성 검사) |
| `disconnect` | `dc` | eGPU 안전 해제 (또는 `--force`로 강제 해제) |
| `monitor` | `m` | 실시간 GPU 모니터링 (온도, 전력, 오류율) |
| `config` | `cf` | GPU 설정 관리 (클럭, 전력, 팬, P-state) |
| `diag` | `d` | 시스템 진단 실행 |
| `log` | `l` | 드라이버 로그 조회 |
| `benchmark` | `b` | 전송 및 연산 성능 벤치마크 |
| `reset` | `r` | GPU 리셋 (소프트/하드) |

**CLI 사용 예시:**

```bash
# GPU 정보 확인
cudabridge-cli info

# eGPU 연결
cudabridge-cli connect

# 실시간 모니터링 (500ms 간격)
cudabridge-cli monitor -i 500

# GPU 설정 확인
cudabridge-cli config show

# 클럭 속도 변경
cudabridge-cli config clock 2100 1200

# 전력 제한 설정
cudabridge-cli config power 350

# 팬 속도 수동 설정 (또는 auto)
cudabridge-cli config fan 80
cudabridge-cli config fan auto

# 성능 상태 변경
cudabridge-cli config pstate P0

# 시스템 진단
cudabridge-cli diag

# 오류 로그 확인
cudabridge-cli log -n 50 --level ERROR

# 벤치마크 실행
cudabridge-cli benchmark

# JSON 출력 (스크립트 연동용)
cudabridge-cli info --json
cudabridge-cli status --json

# eGPU 안전 해제
cudabridge-cli disconnect

# 강제 해제 (비상시)
cudabridge-cli disconnect --force
```

**글로벌 옵션:**

| 옵션 | 설명 |
|------|------|
| `-v, --verbose` | 상세 출력 (디버그 로그 포함) |
| `-j, --json` | JSON 형식 출력 |
| `--no-color` | ANSI 색상 비활성화 |
| `-d, --device N` | 대상 장치 인덱스 지정 |
| `-f, --force` | 강제 실행 |

### 2. eGPU 연결 안전 관리자

Thunderbolt/USB4를 통한 eGPU의 안전한 연결, 해제, 오류 복구를 관리합니다.

**연결 순서:**
1. 장치 감지 (USB4/Thunderbolt 스캔)
2. 호환성 검사 (벤더, 대역폭, 전력 확인)
3. Thunderbolt 보안 인증
4. PCIe 터널 설정
5. GPU 초기화
6. 건강 모니터링 시작

**안전 기능:**
- **호환성 검사**: NVIDIA GPU 전용, 최소 대역폭/전력 검증. 비호환 장치 감지 시 오류 알림 후 강제 해제
- **자동 복구**: 링크 오류 감지 시 자동으로 소프트 리셋 → 하드 리셋 → 재연결 순서로 복구 시도 (최대 3회)
- **데이터 무결성**: CRC32 기반 전송 데이터 검증
- **열 보호**: GPU 온도가 90°C (설정 가능) 초과 시 긴급 중단
- **하트비트 모니터링**: 주기적 연결 상태 확인, 타임아웃 시 자동 복구
- **안전 해제**: 진행 중인 작업 완료 대기 → 리소스 해제 → 터널 해제 순서

**안전 정책 설정:**

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `max_retry_count` | 3 | 최대 재시도 횟수 |
| `retry_delay_ms` | 500 | 재시도 간격 (ms) |
| `heartbeat_interval_ms` | 1000 | 하트비트 주기 (ms) |
| `heartbeat_timeout_ms` | 5000 | 하트비트 타임아웃 (ms) |
| `error_rate_threshold` | 0.01 | 오류율 임계값 (1%) |
| `thermal_limit` | 90 | 온도 제한 (°C) |
| `enable_auto_recovery` | 1 | 자동 복구 활성화 |
| `enable_data_integrity` | 1 | 데이터 무결성 검증 |

### 3. 로깅 시스템

구조화된 로깅으로 GPU 드라이버 디버깅을 지원합니다.

**특징:**
- 6단계 로그 레벨: TRACE, DEBUG, INFO, WARN, ERROR, FATAL
- 카테고리별 필터링: GENERAL, DRIVER, EGPU, MEMORY, PCIE, USB4, CUDA, CLI, PYTHON
- 파일 로그 자동 로테이션 (기본 10MB, 5개 파일 보관)
- ANSI 색상 콘솔 출력
- 스레드 안전 (pthread mutex)
- 사용자 정의 콜백 지원

**로그 파일 위치:** `/tmp/cudabridge_logs/`

**C 코드에서 사용:**

```c
#include "cb_log.h"

// 초기화
cb_log_init_default();

// 로그 작성
CB_LOG_INFO(CB_LOG_CAT_DRIVER, "GPU initialized: %s", gpu_name);
CB_LOG_ERROR(CB_LOG_CAT_EGPU, "Connection lost: %s", reason);
CB_LOG_WARN(CB_LOG_CAT_MEMORY, "High VRAM usage: %.1f%%", usage);

// 카테고리별 레벨 설정
cb_log_set_category_level(CB_LOG_CAT_EGPU, CB_LOG_DEBUG);

// 종료
cb_log_shutdown();
```

### 4. Python API (`cudabridge.py`)

Python에서 별도의 CUDA 라이브러리 없이 기존 numpy 코드와 거의 동일한 방식으로
eGPU CUDA 연산을 수행할 수 있습니다.

**핵심 개념:**
- numpy 배열 → `cb.to_device()` → GPU 메모리
- GPU에서 연산 (add, multiply, matmul 등)
- GPU 메모리 → `cb.from_device()` → numpy 배열

**Python 사용 예시:**

```python
import numpy as np
import cudabridge as cb

# 1. 초기화
cb.init()

# 2. 데이터를 GPU로 전송
a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)

gpu_a = cb.to_device(a)
gpu_b = cb.to_device(b)

# 3. GPU에서 연산 (기존 Python 연산자 사용 가능!)
gpu_c = gpu_a + gpu_b          # 요소별 덧셈
gpu_d = gpu_a * gpu_b          # 요소별 곱셈
gpu_e = gpu_a * 2.0            # 스칼라 곱셈

# 4. 결과를 numpy로 가져오기
result = cb.from_device(gpu_c)
print(result)  # [6.0, 8.0, 10.0, 12.0]

# 5. 리덕션 연산
total = gpu_a.sum()     # 합계
avg = gpu_a.mean()      # 평균
maximum = gpu_a.max()   # 최댓값

# 6. 행렬 곱
mat_a = cb.to_device(np.random.rand(100, 200).astype(np.float32))
mat_b = cb.to_device(np.random.rand(200, 50).astype(np.float32))
mat_c = mat_a @ mat_b   # (100, 50) 행렬 결과

# 7. 편의 함수
zeros = cb.zeros((1000,))          # GPU 0 배열
ones = cb.ones((10, 10))           # GPU 1 배열
random = cb.rand((256, 256))       # GPU 랜덤 배열

# 8. 메모리 정보
free_mem, total_mem = cb.mem_info()
print(f"GPU Memory: {free_mem / 1e9:.1f} GB free / {total_mem / 1e9:.1f} GB total")

# 9. 정리
cb.shutdown()
```

**지원 데이터 타입:**

| numpy dtype | CudaBridge | 크기 |
|-------------|-----------|------|
| `np.float32` | `CB_DTYPE_FLOAT32` | 4 bytes |
| `np.float64` | `CB_DTYPE_FLOAT64` | 8 bytes |
| `np.int32` | `CB_DTYPE_INT32` | 4 bytes |
| `np.int64` | `CB_DTYPE_INT64` | 8 bytes |
| `np.uint8` | `CB_DTYPE_UINT8` | 1 byte |
| `np.int8` | `CB_DTYPE_INT8` | 1 byte |
| `np.int16` | `CB_DTYPE_INT16` | 2 bytes |
| `np.bool_` | `CB_DTYPE_BOOL` | 1 byte |

**지원 연산:**

| 연산 | 함수 | 연산자 | 설명 |
|------|------|--------|------|
| 덧셈 | `cb.add(a, b)` | `a + b` | 요소별 덧셈 |
| 곱셈 | `cb.multiply(a, b)` | `a * b` | 요소별 곱셈 |
| 뺄셈 | - | `a - b` | 요소별 뺄셈 |
| 행렬 곱 | `cb.matmul(a, b)` | `a @ b` | 행렬 곱셈 |
| 스칼라 연산 | `cb.scalar_op()` | `a + 1.0` | 스칼라 사칙연산 |
| 합계 | `cb.reduce(a, 'sum')` | `a.sum()` | 전체 합계 |
| 평균 | `cb.reduce(a, 'mean')` | `a.mean()` | 전체 평균 |
| 최대 | `cb.reduce(a, 'max')` | `a.max()` | 최댓값 |
| 최소 | `cb.reduce(a, 'min')` | `a.min()` | 최솟값 |

### 5. USB4/Thunderbolt PCIe 터널링 드라이버
- Apple Silicon의 USB4 컨트롤러와 직접 통신
- PCIe 트랜잭션을 USB4 프로토콜로 캡슐화
- DMA(Direct Memory Access) 처리
- 핫플러그 지원

### 6. NVIDIA GPU 호환 레이어
- GPU 초기화 및 리셋 시퀀스
- BAR(Base Address Register) 메모리 매핑
- 명령 제출 및 완료 처리
- 전력 관리

### 7. CUDA Runtime Bridge
- CUDA Runtime API 호환 인터페이스
- 커널 컴파일 및 실행
- 메모리 관리 (호스트 ↔ 디바이스)
- 스트림 및 이벤트 동기화

## 시스템 요구사항

### 하드웨어
- Apple Silicon Mac (M1, M2, M3, M4 시리즈)
- USB4 또는 Thunderbolt 4 포트
- 호환되는 eGPU 인클로저 (Thunderbolt 3/4 지원)
- NVIDIA GPU (Ampere, Ada Lovelace, 또는 최신 아키텍처 권장)

### 소프트웨어
- macOS 13.0 (Ventura) 이상
- Xcode Command Line Tools
- CMake 3.20+
- Python 3.8+ (Python API 사용 시)
- numpy (Python API 사용 시)

## 빌드 방법

```bash
# 저장소 클론
git clone https://github.com/KeyWaveTree/CudaBridge.git
cd CudaBridge

# 빌드 디렉토리 생성
mkdir build && cd build

# CMake 구성
cmake ..

# 빌드 (CLI 도구 + 라이브러리)
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)

# 설치 (관리자 권한 필요)
sudo make install
```

빌드 후 생성되는 파일:
- `cudabridge-cli` - GPU 드라이버 CLI 도구
- `libcudabridge.so` / `libcudabridge.dylib` - 공유 라이브러리
- `libcudabridge.a` - 정적 라이브러리

### 빌드 옵션

```bash
cmake .. -DENABLE_DEBUG=ON          # 디버그 모드
cmake .. -DENABLE_ASAN=ON          # Address Sanitizer
cmake .. -DBUILD_TESTS=OFF         # 테스트 빌드 제외
cmake .. -DBUILD_EXAMPLES=OFF      # 예제 빌드 제외
```

## 빠른 시작 가이드

### CLI 도구 사용

```bash
# 1. eGPU 연결
cudabridge-cli connect

# 2. GPU 정보 확인
cudabridge-cli info

# 3. 상태 모니터링
cudabridge-cli status

# 4. 진단 실행
cudabridge-cli diag

# 5. 사용 완료 후 안전 해제
cudabridge-cli disconnect
```

### C API 사용

```c
#include <cudabridge.h>

int main() {
    // CudaBridge 초기화
    cbInit();

    // eGPU 디바이스 열거
    int deviceCount;
    cbGetDeviceCount(&deviceCount);

    // 디바이스 선택
    cbSetDevice(0);

    // 메모리 할당
    float *d_data;
    cbMalloc((void**)&d_data, 1024 * sizeof(float));

    // 데이터 전송
    float h_data[1024];
    cbMemcpy(d_data, h_data, 1024 * sizeof(float), CB_MEMCPY_HOST_TO_DEVICE);

    // CUDA 커널 실행
    dim3 blocks(1);
    dim3 threads(1024);
    cbLaunchKernel("my_kernel", blocks, threads, ...);

    // 동기화
    cbDeviceSynchronize();

    // 정리
    cbFree(d_data);
    cbShutdown();

    return 0;
}
```

### Python API 사용

```python
import numpy as np
import cudabridge as cb

cb.init()

# numpy와 동일한 방식으로 GPU 연산
a = cb.to_device(np.array([1, 2, 3, 4], dtype=np.float32))
b = cb.to_device(np.array([5, 6, 7, 8], dtype=np.float32))

c = a + b                           # GPU에서 덧셈
d = a @ b.reshape(4, 1)             # 연산자도 지원

result = cb.from_device(c)          # 결과를 numpy로
print(result)                       # [6. 8. 10. 12.]

cb.shutdown()
```

## 학습 가이드

### 1단계: CLI 도구로 시작하기

CLI 도구는 코딩 없이 GPU 상태를 확인하고 설정을 변경할 수 있는 가장 쉬운 시작점입니다.

```bash
# 도움말 확인
cudabridge-cli help

# 각 명령의 상세 도움말
cudabridge-cli help connect
cudabridge-cli help config

# GPU 연결 및 정보 확인
cudabridge-cli connect
cudabridge-cli info
cudabridge-cli status
```

### 2단계: Python API로 GPU 연산 체험

Python API는 기존 numpy 코드와 거의 동일한 구조로 GPU 연산을 수행합니다.

```python
import numpy as np
import cudabridge as cb

cb.init()

# 기존 numpy 코드
a_np = np.random.rand(1000).astype(np.float32)
b_np = np.random.rand(1000).astype(np.float32)
c_np = a_np + b_np  # CPU 연산

# CudaBridge로 변환 (3줄만 추가)
a_gpu = cb.to_device(a_np)
b_gpu = cb.to_device(b_np)
c_gpu = a_gpu + b_gpu  # GPU 연산! (동일한 연산자)
c_result = cb.from_device(c_gpu)

# 결과 검증
assert np.allclose(c_np, c_result)

cb.shutdown()
```

### 3단계: C API로 저수준 제어

C API를 사용하면 메모리 관리, 스트림, 이벤트 등을 직접 제어할 수 있습니다.
`src/userspace/examples/` 디렉토리의 예제를 참고하세요.

- `vector_add.c` - 벡터 덧셈 (기본 사용법)
- `device_info.c` - 디바이스 정보 조회
- `bandwidth_test.c` - 대역폭 성능 측정

### 4단계: eGPU 안전 관리 이해하기

eGPU 연결의 안전 메커니즘을 이해하면 안정적인 애플리케이션을 개발할 수 있습니다.

```c
#include "egpu_safety.h"
#include "cb_log.h"

// 이벤트 콜백 등록
void my_callback(EGPUConnectionState state, EGPUError err,
                 const char *msg, void *data) {
    printf("State: %d, Error: %s, Message: %s\n",
           state, egpu_error_string(err), msg);
}

// 사용법
egpu_safety_init(NULL);  // 기본 정책
egpu_set_event_callback(my_callback, NULL);
egpu_connect(-1);  // 자동 감지 연결

// ... GPU 작업 ...

egpu_safe_disconnect();
egpu_safety_shutdown();
```

### 5단계: 로깅 시스템 활용

디버깅 시 로깅 시스템을 활용하면 문제를 빠르게 찾을 수 있습니다.

```bash
# CLI에서 상세 모드로 실행
cudabridge-cli -v connect

# 로그 파일 실시간 확인
tail -f /tmp/cudabridge_logs/*.log

# 오류 로그만 확인
cudabridge-cli log --level ERROR
```

## 프로젝트 구조

```text
CudaBridge/
├── src/
│   ├── cli/              # GPU 드라이버 CLI 도구
│   │   ├── cudabridge_cli.h
│   │   └── cudabridge_cli.c
│   ├── logging/          # 구조화된 로깅 시스템
│   │   ├── cb_log.h
│   │   └── cb_log.c
│   ├── egpu/             # eGPU 연결 안전 관리자
│   │   ├── egpu_safety.h
│   │   └── egpu_safety.c
│   ├── python/           # Python API 브릿지
│   │   ├── cudabridge_python.h
│   │   ├── cudabridge_python.c
│   │   └── cudabridge.py
│   ├── kernel/           # 커널 레벨 코드
│   │   ├── usb4/         # USB4 컨트롤러 드라이버
│   │   └── pcie/         # PCIe 터널링
│   ├── driver/           # DriverKit 확장
│   │   └── nvidia/       # NVIDIA GPU 드라이버
│   ├── cuda/             # CUDA 호환 레이어
│   │   ├── runtime/      # CUDA Runtime API
│   │   └── memory/       # 메모리 관리
│   └── userspace/        # 유저스페이스 라이브러리
│       ├── lib/          # 라이브러리 구현
│       ├── include/      # 공개 헤더
│       └── examples/     # 예제 코드
├── tests/                # 테스트
├── docs/                 # 문서
│   └── i18n/             # 다국어 README
│       ├── README_EN.md  # English
│       ├── README_JA.md  # 日本語
│       └── README_ZH.md  # 中文
└── scripts/              # 빌드/설치 스크립트
```

## 알려진 제한사항

1. **macOS SIP**: System Integrity Protection 비활성화 필요
2. **드라이버 서명**: 개발 중에는 개발자 서명 필요
3. **대역폭**: USB4의 이론적 최대 대역폭은 40Gbps (PCIe 3.0 x4 수준)
4. **호환성**: 모든 CUDA 기능이 지원되지 않을 수 있음
5. **시뮬레이션 모드**: 실제 eGPU 없이도 시뮬레이션 모드로 개발/테스트 가능

## 기여하기

기여를 환영합니다! Pull Request를 보내기 전에 이슈를 먼저 생성해 주세요.

## 감사 및 출처 표기 권장 (Attribution Recommendation)

이 프로젝트는 누구나 자유롭게 활용하고 기여할 수 있도록 MIT 라이선스로 공개되었습니다.
법적인 강제 사항은 아니지만, 오픈소스 생태계의 선순환과 창작자의 동기부여를 위해
아래와 같은 가이드라인을 지켜주시기를 권장합니다.

> "이 프로젝트를 활용해서 멋진 결과물을 만드셨다면, 출처를 남겨주세요. 제게 큰 성취감과 동기부여가 됩니다!"

- **출처 표기 및 감사 인사**: 기업이나 개인 프로젝트에서 이 코드를 활용해 제품 및 서비스를
  만드셨다면, 제품 소개 페이지, 블로그, 발표 자료, 또는 소스코드 주석 등에 짧막하게라도
  원작자(CudaBridge Contributors)와 프로젝트 이름을 기반으로 만들었다는 감사 인사나
  소개를 남겨주시면 감사하겠습니다.

- **프로젝트 재구성(Fork) 시**: 만약 이 프로젝트를 기반으로 재구성(Fork)하거나 발전시켜
  새로운 프로젝트를 배포하실 경우, 원작자인 제가 이 새로운 창작물의 토대(기반)가 되었다는
  메시지를 명시적으로 강조하여 포함해 주시기를 부탁드립니다.

여러분의 따뜻한 언급 한 줄이 개발자가 계속해서 프로젝트를 발전시켜 나갈 수 있는
가장 큰 원동력이 됩니다. 감사합니다!

## 라이선스

MIT License - 자세한 내용은 [LICENSE](LICENSE) 파일 참조

## 면책 조항

이 프로젝트는 실험적이며 교육 목적으로 제공됩니다. 프로덕션 환경에서의 사용은 권장하지 않습니다.
Apple, NVIDIA, CUDA는 각 회사의 상표입니다.
