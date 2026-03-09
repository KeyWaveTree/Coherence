"""CudaBridge Python API with explicit native/simulation backend separation."""

from __future__ import annotations

import ctypes
import platform
import sys
from typing import Optional, Tuple, Union

import numpy as np

from cudabridge_native import bind_native_api, load_native_library
from gpu_array import GPUArray

_DTYPE_MAP = {
    np.float32: 0,
    np.float64: 1,
    np.int32: 2,
    np.int64: 3,
    np.uint8: 4,
    np.uint32: 5,
    np.int8: 6,
    np.int16: 7,
    np.bool_: 9,
}

_SCALAR_OP_MAP = {"add": 0, "sub": 1, "mul": 2, "div": 3}
_REDUCE_OP_MAP = {"sum": 0, "mean": 1, "max": 2, "min": 3}

_lib = None
_initialized = False
_backend = None


def _require_initialized() -> None:
    if _initialized:
        return
    raise RuntimeError("CudaBridge not initialized. Call cb.init() first.")


def _require_gpu_array(arr: GPUArray) -> None:
    if isinstance(arr, GPUArray):
        return
    raise TypeError("Expected GPUArray")


def _require_matching_arrays(a: GPUArray, b: GPUArray) -> None:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    if a.dtype == b.dtype:
        return
    raise ValueError(f"Dtype mismatch: {a.dtype} vs {b.dtype}")


def _as_numpy(data: Union[np.ndarray, list, tuple]) -> np.ndarray:
    if isinstance(data, np.ndarray):
        arr = data
    else:
        arr = np.asarray(data)

    if arr.dtype.type in _DTYPE_MAP:
        return arr
    return arr.astype(np.float32)


def _dtype_code(dtype: np.dtype) -> int:
    code = _DTYPE_MAP.get(np.dtype(dtype).type)
    if code is not None:
        return code
    raise TypeError(f"Unsupported dtype: {dtype}")


def is_simulation_mode() -> bool:
    if _backend is None:
        return False
    return _backend == "simulation"


def init(allow_simulation: bool = False) -> None:
    global _initialized, _backend, _lib
    if _initialized:
        return

    _lib = load_native_library()
    if _lib is None:
        if not allow_simulation:
            raise RuntimeError("Native CudaBridge library not found; pass allow_simulation=True to run CPU simulation")
        _backend = "simulation"
        _initialized = True
        print("[CudaBridge] Initialized (simulation mode)")
        print(f"[CudaBridge] Platform: {platform.machine()}")
        print(f"[CudaBridge] Python: {sys.version.split()[0]}")
        return

    bind_native_api(_lib)
    rc = _lib.cbpy_init()
    if rc != 0:
        if not allow_simulation:
            raise RuntimeError(f"cbpy_init failed: {rc}")
        _backend = "simulation"
        _initialized = True
        print("[CudaBridge] Native init failed, fallback to simulation mode")
        return

    _backend = "native"
    _initialized = True
    print("[CudaBridge] Initialized (native mode)")
    print(f"[CudaBridge] Platform: {platform.machine()}")
    print(f"[CudaBridge] Python: {sys.version.split()[0]}")


def shutdown() -> None:
    global _initialized, _backend, _lib
    if not _initialized:
        return

    if _backend == "native" and _lib is not None:
        _lib.cbpy_shutdown()

    _initialized = False
    _backend = None
    _lib = None
    print("[CudaBridge] Shutdown complete")


def to_device(data: Union[np.ndarray, list, tuple]) -> GPUArray:
    _require_initialized()
    arr = _as_numpy(data)

    if _backend == "simulation":
        return GPUArray(arr.shape, arr.dtype, "simulation", sim_data=arr.copy(), releaser=free)

    dtype_code = _dtype_code(arr.dtype)
    shape_buffer = (ctypes.c_size_t * arr.ndim)(*arr.shape)
    handle = _lib.cbpy_to_device(
        ctypes.c_void_p(arr.ctypes.data),
        ctypes.c_size_t(arr.size),
        ctypes.c_int(dtype_code),
        ctypes.c_int(arr.ndim),
        shape_buffer,
    )
    if handle:
        return GPUArray(arr.shape, arr.dtype, "native", native_handle=ctypes.c_void_p(handle), releaser=free)
    raise RuntimeError("cbpy_to_device failed")


def from_device(gpu_array: GPUArray, dtype: Optional[type] = None) -> np.ndarray:
    _require_initialized()
    _require_gpu_array(gpu_array)
    gpu_array.ensure_alive()

    out_dtype = np.dtype(dtype if dtype is not None else gpu_array.dtype)

    if gpu_array.backend == "simulation":
        data = np.asarray(gpu_array.sim_data, dtype=out_dtype)
        return data.copy()

    if gpu_array.backend != "native":
        raise RuntimeError(f"Unknown backend: {gpu_array.backend}")

    out = np.empty(gpu_array.shape, dtype=out_dtype)
    rc = _lib.cbpy_from_device(gpu_array.native_handle, ctypes.c_void_p(out.ctypes.data))
    if rc == 0:
        return out
    raise RuntimeError(f"cbpy_from_device failed: {rc}")


def _binary_op(a: GPUArray, b: GPUArray, native_func: str, numpy_func) -> GPUArray:
    _require_initialized()
    _require_gpu_array(a)
    _require_gpu_array(b)
    a.ensure_alive()
    b.ensure_alive()
    _require_matching_arrays(a, b)

    if a.backend != b.backend:
        raise ValueError("Backend mismatch between arrays")

    if a.backend == "simulation":
        return GPUArray(a.shape, a.dtype, "simulation", sim_data=numpy_func(a.sim_data, b.sim_data), releaser=free)

    handle = getattr(_lib, native_func)(a.native_handle, b.native_handle)
    if handle:
        return GPUArray(a.shape, a.dtype, "native", native_handle=ctypes.c_void_p(handle), releaser=free)
    raise RuntimeError(f"{native_func} failed")


def add(a: GPUArray, b: GPUArray) -> GPUArray:
    return _binary_op(a, b, "cbpy_add", np.add)


def multiply(a: GPUArray, b: GPUArray) -> GPUArray:
    return _binary_op(a, b, "cbpy_multiply", np.multiply)


def matmul(a: GPUArray, b: GPUArray) -> GPUArray:
    _require_initialized()
    _require_gpu_array(a)
    _require_gpu_array(b)
    a.ensure_alive()
    b.ensure_alive()

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("matmul requires two 2D arrays")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Shape mismatch for matmul: {a.shape} x {b.shape}")
    if a.dtype != b.dtype:
        raise ValueError(f"Dtype mismatch: {a.dtype} vs {b.dtype}")
    if a.backend != b.backend:
        raise ValueError("Backend mismatch between arrays")

    result_shape = (a.shape[0], b.shape[1])

    if a.backend == "simulation":
        return GPUArray(result_shape, a.dtype, "simulation", sim_data=np.matmul(a.sim_data, b.sim_data), releaser=free)

    handle = _lib.cbpy_matmul(a.native_handle, b.native_handle)
    if handle:
        return GPUArray(result_shape, a.dtype, "native", native_handle=ctypes.c_void_p(handle), releaser=free)
    raise RuntimeError("cbpy_matmul failed")


def scalar_op(arr: GPUArray, scalar: float, op: str) -> GPUArray:
    _require_initialized()
    _require_gpu_array(arr)
    arr.ensure_alive()

    code = _SCALAR_OP_MAP.get(op)
    if code is None:
        raise ValueError(f"Unsupported scalar op: {op}")

    if arr.backend == "simulation":
        if op == "add":
            data = arr.sim_data + scalar
        elif op == "sub":
            data = arr.sim_data - scalar
        elif op == "mul":
            data = arr.sim_data * scalar
        else:
            data = arr.sim_data / scalar
        return GPUArray(arr.shape, arr.dtype, "simulation", sim_data=data, releaser=free)

    handle = _lib.cbpy_scalar_op(arr.native_handle, ctypes.c_double(scalar), ctypes.c_int(code))
    if handle:
        return GPUArray(arr.shape, arr.dtype, "native", native_handle=ctypes.c_void_p(handle), releaser=free)
    raise RuntimeError("cbpy_scalar_op failed")


def reduce(arr: GPUArray, op: str) -> float:
    _require_initialized()
    _require_gpu_array(arr)
    arr.ensure_alive()

    code = _REDUCE_OP_MAP.get(op)
    if code is None:
        raise ValueError(f"Unsupported reduce op: {op}")

    if arr.backend == "simulation":
        if op == "sum":
            return float(np.sum(arr.sim_data))
        if op == "mean":
            return float(np.mean(arr.sim_data))
        if op == "max":
            return float(np.max(arr.sim_data))
        return float(np.min(arr.sim_data))

    return float(_lib.cbpy_reduce(arr.native_handle, ctypes.c_int(code)))


def free(gpu_array: GPUArray) -> None:
    _require_gpu_array(gpu_array)
    try:
        gpu_array.ensure_alive()
    except RuntimeError:
        return

    if gpu_array.backend == "native" and _lib is not None:
        _lib.cbpy_free(gpu_array.native_handle)

    gpu_array.mark_freed()


def get_device_count() -> int:
    _require_initialized()
    return 1 if _backend == "simulation" else 1


def get_device_name(device: int = 0) -> str:
    _require_initialized()
    if device != 0:
        raise ValueError("Only device 0 is currently supported")
    if _backend == "simulation":
        return "Simulation Device"
    raw = _lib.cbpy_device_name()
    if raw:
        return raw.decode("utf-8")
    return "Unknown Device"


def mem_info() -> Tuple[int, int]:
    _require_initialized()
    if _backend == "simulation":
        total = 24 * 1024 * 1024 * 1024
        return total // 2, total

    free_b = ctypes.c_size_t()
    total_b = ctypes.c_size_t()
    rc = _lib.cbpy_mem_info(ctypes.byref(free_b), ctypes.byref(total_b))
    if rc == 0:
        return int(free_b.value), int(total_b.value)
    raise RuntimeError(f"cbpy_mem_info failed: {rc}")


def synchronize() -> None:
    _require_initialized()
    return


def zeros(shape, dtype=np.float32) -> GPUArray:
    return to_device(np.zeros(shape, dtype=dtype))


def ones(shape, dtype=np.float32) -> GPUArray:
    return to_device(np.ones(shape, dtype=dtype))


def rand(shape, dtype=np.float32) -> GPUArray:
    data = np.random.rand(*shape).astype(dtype)
    return to_device(data)
