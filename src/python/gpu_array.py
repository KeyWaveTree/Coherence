"""GPUArray abstraction for CudaBridge."""

from __future__ import annotations

import ctypes
from typing import Callable, Optional, Tuple

import numpy as np


class GPUArray:
    """GPU array handle backed by either native device memory or simulation ndarray."""

    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: np.dtype,
        backend: str,
        native_handle: Optional[ctypes.c_void_p] = None,
        sim_data: Optional[np.ndarray] = None,
        releaser: Optional[Callable[["GPUArray"], None]] = None,
    ) -> None:
        self._shape = tuple(shape)
        self._dtype = np.dtype(dtype)
        self._backend = backend
        self._native_handle = native_handle
        self._sim_data = sim_data
        self._freed = False
        self._releaser = releaser

    def ensure_alive(self) -> None:
        if not self._freed:
            return
        raise RuntimeError("GPUArray has already been freed")

    @property
    def shape(self) -> Tuple[int, ...]:
        self.ensure_alive()
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        self.ensure_alive()
        return self._dtype

    @property
    def size(self) -> int:
        self.ensure_alive()
        return int(np.prod(self._shape))

    @property
    def nbytes(self) -> int:
        self.ensure_alive()
        return self.size * self._dtype.itemsize

    @property
    def ndim(self) -> int:
        self.ensure_alive()
        return len(self._shape)

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def native_handle(self) -> Optional[ctypes.c_void_p]:
        self.ensure_alive()
        return self._native_handle

    @property
    def sim_data(self) -> Optional[np.ndarray]:
        self.ensure_alive()
        return self._sim_data

    def mark_freed(self) -> None:
        self._native_handle = None
        self._sim_data = None
        self._freed = True


    def free(self) -> None:
        if self._freed:
            return
        if self._releaser is not None:
            self._releaser(self)
            return
        self.mark_freed()

    def __del__(self) -> None:
        self.free()

    def __repr__(self) -> str:
        state = "freed" if self._freed else "alive"
        return f"GPUArray(shape={self._shape}, dtype={self._dtype}, backend='{self._backend}', state='{state}')"
