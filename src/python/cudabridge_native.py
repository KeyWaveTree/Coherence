"""Native ctypes bindings for CudaBridge Python API."""

from __future__ import annotations

import ctypes
import os
import sys
from typing import Optional


def _candidate_libraries() -> list[str]:
    base = os.path.dirname(__file__)
    if sys.platform == "darwin":
        return [
            "libcudabridge.dylib",
            os.path.join(base, "..", "..", "build", "libcudabridge.dylib"),
            "/usr/local/lib/libcudabridge.dylib",
        ]

    return [
        "libcudabridge.so",
        os.path.join(base, "..", "..", "build", "libcudabridge.so"),
        "/usr/local/lib/libcudabridge.so",
    ]


def load_native_library() -> Optional[ctypes.CDLL]:
    for lib_name in _candidate_libraries():
        try:
            return ctypes.CDLL(lib_name)
        except OSError:
            continue

    return None


def bind_native_api(lib: ctypes.CDLL) -> None:
    void_p = ctypes.c_void_p
    size_t = ctypes.c_size_t

    lib.cbpy_init.argtypes = []
    lib.cbpy_init.restype = ctypes.c_int

    lib.cbpy_shutdown.argtypes = []
    lib.cbpy_shutdown.restype = None

    lib.cbpy_to_device.argtypes = [void_p, size_t, ctypes.c_int, ctypes.c_int, ctypes.POINTER(size_t)]
    lib.cbpy_to_device.restype = void_p

    lib.cbpy_from_device.argtypes = [void_p, void_p]
    lib.cbpy_from_device.restype = ctypes.c_int

    lib.cbpy_free.argtypes = [void_p]
    lib.cbpy_free.restype = None

    lib.cbpy_add.argtypes = [void_p, void_p]
    lib.cbpy_add.restype = void_p

    lib.cbpy_multiply.argtypes = [void_p, void_p]
    lib.cbpy_multiply.restype = void_p

    lib.cbpy_matmul.argtypes = [void_p, void_p]
    lib.cbpy_matmul.restype = void_p

    lib.cbpy_scalar_op.argtypes = [void_p, ctypes.c_double, ctypes.c_int]
    lib.cbpy_scalar_op.restype = void_p

    lib.cbpy_reduce.argtypes = [void_p, ctypes.c_int]
    lib.cbpy_reduce.restype = ctypes.c_double

    lib.cbpy_device_name.argtypes = []
    lib.cbpy_device_name.restype = ctypes.c_char_p

    lib.cbpy_mem_info.argtypes = [ctypes.POINTER(size_t), ctypes.POINTER(size_t)]
    lib.cbpy_mem_info.restype = ctypes.c_int

    if hasattr(lib, "cudaBridgeIsSimulationMode"):
        lib.cudaBridgeIsSimulationMode.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.cudaBridgeIsSimulationMode.restype = ctypes.c_int
