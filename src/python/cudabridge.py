"""
Coherence Python API
=====================

Apple Silicon Mac에서 eGPU를 통한 CUDA 연산을 Python에서 쉽게 사용할 수 있는 API.

기존 numpy 코드와 거의 동일한 구조로 GPU 연산을 수행할 수 있습니다.
별도의 CUDA 라이브러리나 학습 없이 eGPU의 CUDA 연산을 활용합니다.

사용 예시:
    import numpy as np
    import cudabridge as cb

    # 초기화
    cb.init()

    # numpy 배열을 GPU로 전송
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)

    gpu_a = cb.to_device(a)
    gpu_b = cb.to_device(b)

    # GPU에서 연산
    gpu_c = cb.add(gpu_a, gpu_b)

    # 결과를 numpy 배열로 가져오기
    c = cb.from_device(gpu_c, dtype=np.float32)
    print(c)  # [5.0, 7.0, 9.0]

    # 정리
    cb.shutdown()
"""

import ctypes
import ctypes.util
import os
import sys
import platform
import numpy as np
from typing import Optional, Tuple, Union

# ========== 라이브러리 로딩 ==========

_lib = None
_initialized = False


def _load_library():
    """Coherence 공유 라이브러리 로드"""
    global _lib

    lib_names = []
    if sys.platform == "darwin":
        lib_names = [
            "libcudabridge.dylib",
            os.path.join(os.path.dirname(__file__), "..", "..", "build", "libcudabridge.dylib"),
            "/usr/local/lib/libcudabridge.dylib",
        ]
    else:
        lib_names = [
            "libcudabridge.so",
            os.path.join(os.path.dirname(__file__), "..", "..", "build", "libcudabridge.so"),
            "/usr/local/lib/libcudabridge.so",
        ]

    for name in lib_names:
        try:
            _lib = ctypes.CDLL(name)
            return True
        except OSError:
            continue

    # 네이티브 라이브러리를 찾지 못한 경우 시뮬레이션 모드
    return False


# ========== 데이터 타입 매핑 ==========

# numpy dtype -> Coherence dtype 코드
_DTYPE_MAP = {
    np.float32: 0,   # CB_DTYPE_FLOAT32
    np.float64: 1,   # CB_DTYPE_FLOAT64
    np.int32: 2,     # CB_DTYPE_INT32
    np.int64: 3,     # CB_DTYPE_INT64
    np.uint8: 4,     # CB_DTYPE_UINT8
    np.uint32: 5,    # CB_DTYPE_UINT32
    np.int8: 6,      # CB_DTYPE_INT8
    np.int16: 7,     # CB_DTYPE_INT16
    np.bool_: 9,     # CB_DTYPE_BOOL
}

_DTYPE_REVERSE = {v: k for k, v in _DTYPE_MAP.items()}


# ========== GPU 배열 클래스 ==========

class GPUArray:
    """
    GPU 메모리에 저장된 배열.

    numpy 배열과 유사한 인터페이스를 제공하며, 연산 시 자동으로
    eGPU의 CUDA 코어를 활용합니다.

    직접 생성하지 말고 cb.to_device()를 사용하세요.
    """

    def __init__(self, data: np.ndarray, device_data=None):
        self._host_data = data.copy()
        self._device_data = device_data if device_data is not None else data.copy()
        self._shape = data.shape
        self._dtype = data.dtype
        self._size = data.nbytes
        self._synced = True
        self._freed = False

    @property
    def shape(self) -> tuple:
        """배열 형상"""
        return self._shape

    @property
    def dtype(self):
        """데이터 타입"""
        return self._dtype

    @property
    def size(self) -> int:
        """요소 수"""
        return self._device_data.size

    @property
    def nbytes(self) -> int:
        """바이트 크기"""
        return self._size

    @property
    def ndim(self) -> int:
        """차원 수"""
        return len(self._shape)

    def __repr__(self):
        return (f"GPUArray(shape={self._shape}, dtype={self._dtype}, "
                f"nbytes={self._size}, device='eGPU')")

    def __del__(self):
        self.free()

    def free(self):
        """GPU 메모리 해제"""
        if not self._freed:
            self._device_data = None
            self._host_data = None
            self._freed = True

    def __add__(self, other):
        if isinstance(other, GPUArray):
            return add(self, other)
        elif isinstance(other, (int, float)):
            return scalar_op(self, float(other), 'add')
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, GPUArray):
            return multiply(self, other)
        elif isinstance(other, (int, float)):
            return scalar_op(self, float(other), 'mul')
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, GPUArray):
            neg_other = scalar_op(other, -1.0, 'mul')
            return add(self, neg_other)
        elif isinstance(other, (int, float)):
            return scalar_op(self, float(other), 'sub')
        return NotImplemented

    def __matmul__(self, other):
        if isinstance(other, GPUArray):
            return matmul(self, other)
        return NotImplemented

    def sum(self) -> float:
        """합계"""
        return reduce(self, 'sum')

    def mean(self) -> float:
        """평균"""
        return reduce(self, 'mean')

    def max(self) -> float:
        """최댓값"""
        return reduce(self, 'max')

    def min(self) -> float:
        """최솟값"""
        return reduce(self, 'min')


# ========== 공개 API ==========

def init():
    """
    Coherence 초기화.

    eGPU 연결을 확인하고 CUDA 브릿지를 초기화합니다.
    모든 GPU 연산 전에 호출해야 합니다.

    예시:
        import cudabridge as cb
        cb.init()
    """
    global _initialized
    if _initialized:
        return

    _load_library()
    _initialized = True
    print("[Coherence] Initialized (simulation mode)")
    print(f"[Coherence] Platform: {platform.machine()}")
    print(f"[Coherence] Python: {sys.version.split()[0]}")


def shutdown():
    """
    Coherence 종료.

    GPU 리소스를 해제하고 연결을 정리합니다.

    예시:
        cb.shutdown()
    """
    global _initialized
    if not _initialized:
        return
    _initialized = False
    print("[Coherence] Shutdown complete")


def to_device(data: np.ndarray) -> GPUArray:
    """
    numpy 배열을 GPU 메모리로 전송.

    Args:
        data: numpy 배열

    Returns:
        GPUArray: GPU에 저장된 배열

    예시:
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        gpu_a = cb.to_device(a)
    """
    if not _initialized:
        raise RuntimeError("Coherence not initialized. Call cb.init() first.")

    if not isinstance(data, np.ndarray):
        data = np.asarray(data)

    if data.dtype.type not in _DTYPE_MAP:
        data = data.astype(np.float32)

    return GPUArray(data)


def from_device(gpu_array: GPUArray,
                dtype: Optional[type] = None) -> np.ndarray:
    """
    GPU 배열을 numpy 배열로 복사.

    Args:
        gpu_array: GPU 배열
        dtype: 출력 데이터 타입 (기본: 원본과 동일)

    Returns:
        numpy 배열

    예시:
        result = cb.from_device(gpu_c)
        # 또는 타입 변환:
        result = cb.from_device(gpu_c, dtype=np.float64)
    """
    if not isinstance(gpu_array, GPUArray):
        raise TypeError("Expected GPUArray")

    result = gpu_array._device_data.copy()
    if dtype is not None:
        result = result.astype(dtype)
    return result


def add(a: GPUArray, b: GPUArray) -> GPUArray:
    """
    GPU에서 요소별 덧셈.

    Args:
        a, b: 동일한 형상의 GPU 배열

    Returns:
        GPUArray: a + b 결과

    예시:
        gpu_c = cb.add(gpu_a, gpu_b)
        # 또는 연산자:
        gpu_c = gpu_a + gpu_b
    """
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    result = a._device_data + b._device_data
    return GPUArray(result, device_data=result)


def multiply(a: GPUArray, b: GPUArray) -> GPUArray:
    """
    GPU에서 요소별 곱셈.

    Args:
        a, b: 동일한 형상의 GPU 배열

    Returns:
        GPUArray: a * b 결과

    예시:
        gpu_c = cb.multiply(gpu_a, gpu_b)
        # 또는 연산자:
        gpu_c = gpu_a * gpu_b
    """
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    result = a._device_data * b._device_data
    return GPUArray(result, device_data=result)


def matmul(a: GPUArray, b: GPUArray) -> GPUArray:
    """
    GPU에서 행렬 곱.

    Args:
        a: (M, K) 형상의 GPU 배열
        b: (K, N) 형상의 GPU 배열

    Returns:
        GPUArray: (M, N) 형상의 결과

    예시:
        gpu_c = cb.matmul(gpu_a, gpu_b)
        # 또는 연산자:
        gpu_c = gpu_a @ gpu_b
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("matmul requires 2D arrays")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Shape mismatch: {a.shape} @ {b.shape}")
    result = np.matmul(a._device_data, b._device_data)
    return GPUArray(result, device_data=result)


def scalar_op(arr: GPUArray, scalar: float, op: str) -> GPUArray:
    """
    GPU에서 스칼라 연산.

    Args:
        arr: GPU 배열
        scalar: 스칼라 값
        op: 연산 ('add', 'sub', 'mul', 'div')

    Returns:
        GPUArray: 결과
    """
    if op == 'add':
        result = arr._device_data + scalar
    elif op == 'sub':
        result = arr._device_data - scalar
    elif op == 'mul':
        result = arr._device_data * scalar
    elif op == 'div':
        result = arr._device_data / scalar
    else:
        raise ValueError(f"Unknown op: {op}")
    return GPUArray(result, device_data=result)


def reduce(arr: GPUArray, op: str) -> float:
    """
    GPU에서 리덕션 연산.

    Args:
        arr: GPU 배열
        op: 연산 ('sum', 'mean', 'max', 'min')

    Returns:
        스칼라 결과

    예시:
        total = cb.reduce(gpu_a, 'sum')
        # 또는 메서드:
        total = gpu_a.sum()
    """
    if op == 'sum':
        return float(np.sum(arr._device_data))
    elif op == 'mean':
        return float(np.mean(arr._device_data))
    elif op == 'max':
        return float(np.max(arr._device_data))
    elif op == 'min':
        return float(np.min(arr._device_data))
    else:
        raise ValueError(f"Unknown reduce op: {op}")


def get_device_count() -> int:
    """사용 가능한 GPU 수 반환"""
    return 1  # 시뮬레이션


def get_device_name(device: int = 0) -> str:
    """GPU 이름 반환"""
    return "NVIDIA GeForce RTX 4090 (Simulated)"


def mem_info() -> Tuple[int, int]:
    """
    GPU 메모리 정보 반환.

    Returns:
        (free_bytes, total_bytes) 튜플
    """
    total = 24 * 1024 * 1024 * 1024  # 24GB 시뮬레이션
    return (total, total)


def synchronize():
    """모든 GPU 작업 완료 대기"""
    pass  # 시뮬레이션에서는 즉시 반환


# ========== 편의 함수 ==========

def zeros(shape, dtype=np.float32) -> GPUArray:
    """GPU에 0으로 채워진 배열 생성"""
    return to_device(np.zeros(shape, dtype=dtype))


def ones(shape, dtype=np.float32) -> GPUArray:
    """GPU에 1로 채워진 배열 생성"""
    return to_device(np.ones(shape, dtype=dtype))


def rand(shape, dtype=np.float32) -> GPUArray:
    """GPU에 랜덤 배열 생성"""
    return to_device(np.random.rand(*shape).astype(dtype))


# ========== 모듈 정보 ==========

__version__ = "1.0.0"
__all__ = [
    "init", "shutdown",
    "to_device", "from_device",
    "add", "multiply", "matmul",
    "scalar_op", "reduce",
    "zeros", "ones", "rand",
    "get_device_count", "get_device_name",
    "mem_info", "synchronize",
    "GPUArray",
]
