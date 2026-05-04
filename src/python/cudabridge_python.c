/**
 * Coherence - Python C Extension API Implementation
 *
 * Python 사용자가 별도의 CUDA 라이브러리 없이 기존 Python/numpy 코드와
 * 거의 동일한 구조로 eGPU CUDA 연산을 수행할 수 있게 합니다.
 *
 * 데이터 흐름:
 *   Python numpy array → cbpy_to_device() → GPU VRAM
 *   GPU 연산 (add, multiply, matmul 등)
 *   GPU VRAM → cbpy_from_device() → Python numpy array
 */

#include "cudabridge_python.h"
#include "../logging/cb_log.h"
#include "../egpu/egpu_safety.h"

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 전역 상태 */
static struct {
    int             initialized;
    pthread_mutex_t lock;
    uint64_t        total_allocs;
    uint64_t        total_frees;
    size_t          current_usage;
    size_t          peak_usage;
} g_pyctx = {0};

/* ========== 유틸리티 ========== */

size_t cbpy_dtype_size(int dtype) {
    switch (dtype) {
        case CB_DTYPE_FLOAT32: return 4;
        case CB_DTYPE_FLOAT64: return 8;
        case CB_DTYPE_INT32:   return 4;
        case CB_DTYPE_INT64:   return 8;
        case CB_DTYPE_UINT8:   return 1;
        case CB_DTYPE_UINT32:  return 4;
        case CB_DTYPE_INT8:    return 1;
        case CB_DTYPE_INT16:   return 2;
        case CB_DTYPE_FLOAT16: return 2;
        case CB_DTYPE_BOOL:    return 1;
        default:               return 0;
    }
}

static const char* dtype_name(int dtype) {
    switch (dtype) {
        case CB_DTYPE_FLOAT32: return "float32";
        case CB_DTYPE_FLOAT64: return "float64";
        case CB_DTYPE_INT32:   return "int32";
        case CB_DTYPE_INT64:   return "int64";
        case CB_DTYPE_UINT8:   return "uint8";
        case CB_DTYPE_UINT32:  return "uint32";
        case CB_DTYPE_INT8:    return "int8";
        case CB_DTYPE_INT16:   return "int16";
        case CB_DTYPE_FLOAT16: return "float16";
        case CB_DTYPE_BOOL:    return "bool";
        default:               return "unknown";
    }
}

static uint32_t compute_crc32(const void *data, size_t size) {
    return egpu_compute_crc32(data, size);
}

/* ========== 초기화/종료 ========== */

int cbpy_init(void) {
    if (g_pyctx.initialized) return 0;

    /* 로깅 초기화 (아직 안 되어 있으면) */
    cb_log_init_default();

    CB_LOG_INFO(CB_LOG_CAT_PYTHON, "Python API bridge initializing...");

    pthread_mutex_init(&g_pyctx.lock, NULL);
    g_pyctx.total_allocs = 0;
    g_pyctx.total_frees = 0;
    g_pyctx.current_usage = 0;
    g_pyctx.peak_usage = 0;
    g_pyctx.initialized = 1;

    CB_LOG_INFO(CB_LOG_CAT_PYTHON, "Python API bridge initialized");
    return 0;
}

void cbpy_shutdown(void) {
    if (!g_pyctx.initialized) return;

    CB_LOG_INFO(CB_LOG_CAT_PYTHON,
                "Python API stats: allocs=%lu, frees=%lu, peak=%lu bytes",
                (unsigned long)g_pyctx.total_allocs,
                (unsigned long)g_pyctx.total_frees,
                (unsigned long)g_pyctx.peak_usage);

    if (g_pyctx.total_allocs != g_pyctx.total_frees) {
        CB_LOG_WARN(CB_LOG_CAT_PYTHON,
                    "Memory leak detected: %lu allocations not freed",
                    (unsigned long)(g_pyctx.total_allocs - g_pyctx.total_frees));
    }

    pthread_mutex_destroy(&g_pyctx.lock);
    g_pyctx.initialized = 0;
    CB_LOG_INFO(CB_LOG_CAT_PYTHON, "Python API bridge shutdown");
}

/* ========== 데이터 전송 ========== */

CBPyArray* cbpy_to_device(const void *host_data, size_t elem_count,
                           int dtype, int ndim, const size_t *shape) {
    if (!g_pyctx.initialized) {
        CB_LOG_ERROR(CB_LOG_CAT_PYTHON, "API not initialized");
        return NULL;
    }
    if (!host_data || elem_count == 0) {
        CB_LOG_ERROR(CB_LOG_CAT_PYTHON, "Invalid input: null data or zero count");
        return NULL;
    }
    if (ndim < 1 || ndim > 8) {
        CB_LOG_ERROR(CB_LOG_CAT_PYTHON, "Invalid dimensions: %d (must be 1-8)", ndim);
        return NULL;
    }

    size_t elem_size = cbpy_dtype_size(dtype);
    if (elem_size == 0) {
        CB_LOG_ERROR(CB_LOG_CAT_PYTHON, "Unknown dtype: %d", dtype);
        return NULL;
    }

    size_t total_size = elem_count * elem_size;

    CB_LOG_DEBUG(CB_LOG_CAT_PYTHON,
                 "to_device: %lu elements, dtype=%s, size=%lu bytes",
                 (unsigned long)elem_count, dtype_name(dtype),
                 (unsigned long)total_size);

    CBPyArray *arr = (CBPyArray *)calloc(1, sizeof(CBPyArray));
    if (!arr) {
        CB_LOG_ERROR(CB_LOG_CAT_PYTHON, "Failed to allocate array handle");
        return NULL;
    }

    /*
     * 시뮬레이션 모드: 호스트 메모리를 할당하여 GPU VRAM을 시뮬레이션.
     * 실제 환경에서는 cbMalloc()으로 GPU 메모리를 할당하고
     * cbMemcpy()로 데이터를 H2D 전송합니다.
     */
    arr->device_ptr = malloc(total_size);
    if (!arr->device_ptr) {
        CB_LOG_ERROR(CB_LOG_CAT_PYTHON, "Failed to allocate device memory: %lu bytes",
                     (unsigned long)total_size);
        free(arr);
        return NULL;
    }

    /* 호스트 → 디바이스 복사 (시뮬레이션) */
    memcpy(arr->device_ptr, host_data, total_size);

    arr->size = total_size;
    arr->elem_count = elem_count;
    arr->dtype = dtype;
    arr->ndim = ndim;
    for (int i = 0; i < ndim && i < 8; i++) {
        arr->shape[i] = shape ? shape[i] : elem_count;
    }
    arr->is_synced = 1;
    arr->crc32 = compute_crc32(host_data, total_size);
    arr->host_ptr = NULL;

    /* 통계 업데이트 */
    pthread_mutex_lock(&g_pyctx.lock);
    g_pyctx.total_allocs++;
    g_pyctx.current_usage += total_size;
    if (g_pyctx.current_usage > g_pyctx.peak_usage) {
        g_pyctx.peak_usage = g_pyctx.current_usage;
    }
    pthread_mutex_unlock(&g_pyctx.lock);

    CB_LOG_DEBUG(CB_LOG_CAT_PYTHON, "to_device complete: CRC=0x%08X", arr->crc32);
    return arr;
}

int cbpy_from_device(CBPyArray *arr, void *host_data) {
    if (!arr || !host_data) {
        CB_LOG_ERROR(CB_LOG_CAT_PYTHON, "Invalid arguments to from_device");
        return -1;
    }

    CB_LOG_DEBUG(CB_LOG_CAT_PYTHON,
                 "from_device: %lu bytes, dtype=%s",
                 (unsigned long)arr->size, dtype_name(arr->dtype));

    /* 디바이스 → 호스트 복사 (시뮬레이션) */
    memcpy(host_data, arr->device_ptr, arr->size);

    /* 데이터 무결성 검증 */
    uint32_t new_crc = compute_crc32(host_data, arr->size);
    if (new_crc != arr->crc32 && arr->is_synced) {
        CB_LOG_WARN(CB_LOG_CAT_PYTHON,
                    "Data modified on device: CRC 0x%08X -> 0x%08X",
                    arr->crc32, new_crc);
    }

    return 0;
}

void cbpy_free(CBPyArray *arr) {
    if (!arr) return;

    CB_LOG_DEBUG(CB_LOG_CAT_PYTHON, "free: %lu bytes", (unsigned long)arr->size);

    /* 디바이스 메모리 해제 */
    if (arr->device_ptr) {
        /* 보안: 메모리 제로화 */
        volatile uint8_t *p = (volatile uint8_t *)arr->device_ptr;
        for (size_t i = 0; i < arr->size; i++) {
            p[i] = 0;
        }
        free(arr->device_ptr);
    }
    if (arr->host_ptr) {
        free(arr->host_ptr);
    }

    pthread_mutex_lock(&g_pyctx.lock);
    g_pyctx.total_frees++;
    if (g_pyctx.current_usage >= arr->size) {
        g_pyctx.current_usage -= arr->size;
    }
    pthread_mutex_unlock(&g_pyctx.lock);

    free(arr);
}

/* ========== GPU 연산 (시뮬레이션) ========== */

/*
 * 실제 환경에서는 이 연산들이 CUDA 커널로 GPU에서 실행됩니다.
 * Coherence 드라이버가 연산 데이터를 eGPU로 전달하고 결과를 받아옵니다.
 * 시뮬레이션 모드에서는 CPU에서 동일한 연산을 수행합니다.
 */

CBPyArray* cbpy_add(CBPyArray *a, CBPyArray *b) {
    if (!a || !b) return NULL;
    if (a->elem_count != b->elem_count || a->dtype != b->dtype) {
        CB_LOG_ERROR(CB_LOG_CAT_PYTHON,
                     "Shape/dtype mismatch in add: (%lu, %s) vs (%lu, %s)",
                     (unsigned long)a->elem_count, dtype_name(a->dtype),
                     (unsigned long)b->elem_count, dtype_name(b->dtype));
        return NULL;
    }

    CB_LOG_DEBUG(CB_LOG_CAT_PYTHON, "GPU add: %lu elements, dtype=%s",
                 (unsigned long)a->elem_count, dtype_name(a->dtype));

    /* 결과 배열 생성 */
    size_t total_size = a->size;
    void *result_data = malloc(total_size);
    if (!result_data) return NULL;

    /* 요소별 덧셈 (시뮬레이션 - 실제로는 CUDA 커널 실행) */
    switch (a->dtype) {
        case CB_DTYPE_FLOAT32: {
            const float *pa = (const float *)a->device_ptr;
            const float *pb = (const float *)b->device_ptr;
            float *pr = (float *)result_data;
            for (size_t i = 0; i < a->elem_count; i++) pr[i] = pa[i] + pb[i];
            break;
        }
        case CB_DTYPE_FLOAT64: {
            const double *pa = (const double *)a->device_ptr;
            const double *pb = (const double *)b->device_ptr;
            double *pr = (double *)result_data;
            for (size_t i = 0; i < a->elem_count; i++) pr[i] = pa[i] + pb[i];
            break;
        }
        case CB_DTYPE_INT32: {
            const int32_t *pa = (const int32_t *)a->device_ptr;
            const int32_t *pb = (const int32_t *)b->device_ptr;
            int32_t *pr = (int32_t *)result_data;
            for (size_t i = 0; i < a->elem_count; i++) pr[i] = pa[i] + pb[i];
            break;
        }
        default:
            CB_LOG_ERROR(CB_LOG_CAT_PYTHON, "Unsupported dtype for add: %d", a->dtype);
            free(result_data);
            return NULL;
    }

    CBPyArray *result = cbpy_to_device(result_data, a->elem_count,
                                        a->dtype, a->ndim, a->shape);
    free(result_data);
    if (result) result->is_synced = 0;
    return result;
}

CBPyArray* cbpy_multiply(CBPyArray *a, CBPyArray *b) {
    if (!a || !b) return NULL;
    if (a->elem_count != b->elem_count || a->dtype != b->dtype) {
        CB_LOG_ERROR(CB_LOG_CAT_PYTHON, "Shape/dtype mismatch in multiply");
        return NULL;
    }

    CB_LOG_DEBUG(CB_LOG_CAT_PYTHON, "GPU multiply: %lu elements",
                 (unsigned long)a->elem_count);

    size_t total_size = a->size;
    void *result_data = malloc(total_size);
    if (!result_data) return NULL;

    switch (a->dtype) {
        case CB_DTYPE_FLOAT32: {
            const float *pa = (const float *)a->device_ptr;
            const float *pb = (const float *)b->device_ptr;
            float *pr = (float *)result_data;
            for (size_t i = 0; i < a->elem_count; i++) pr[i] = pa[i] * pb[i];
            break;
        }
        case CB_DTYPE_FLOAT64: {
            const double *pa = (const double *)a->device_ptr;
            const double *pb = (const double *)b->device_ptr;
            double *pr = (double *)result_data;
            for (size_t i = 0; i < a->elem_count; i++) pr[i] = pa[i] * pb[i];
            break;
        }
        case CB_DTYPE_INT32: {
            const int32_t *pa = (const int32_t *)a->device_ptr;
            const int32_t *pb = (const int32_t *)b->device_ptr;
            int32_t *pr = (int32_t *)result_data;
            for (size_t i = 0; i < a->elem_count; i++) pr[i] = pa[i] * pb[i];
            break;
        }
        default:
            free(result_data);
            return NULL;
    }

    CBPyArray *result = cbpy_to_device(result_data, a->elem_count,
                                        a->dtype, a->ndim, a->shape);
    free(result_data);
    if (result) result->is_synced = 0;
    return result;
}

CBPyArray* cbpy_matmul(CBPyArray *a, CBPyArray *b) {
    if (!a || !b) return NULL;
    if (a->ndim != 2 || b->ndim != 2) {
        CB_LOG_ERROR(CB_LOG_CAT_PYTHON, "matmul requires 2D arrays");
        return NULL;
    }
    if (a->shape[1] != b->shape[0]) {
        CB_LOG_ERROR(CB_LOG_CAT_PYTHON,
                     "matmul shape mismatch: (%lu,%lu) x (%lu,%lu)",
                     (unsigned long)a->shape[0], (unsigned long)a->shape[1],
                     (unsigned long)b->shape[0], (unsigned long)b->shape[1]);
        return NULL;
    }

    size_t M = a->shape[0];
    size_t K = a->shape[1];
    size_t N = b->shape[1];

    CB_LOG_DEBUG(CB_LOG_CAT_PYTHON, "GPU matmul: (%lu,%lu) x (%lu,%lu)",
                 (unsigned long)M, (unsigned long)K,
                 (unsigned long)K, (unsigned long)N);

    size_t result_elems = M * N;
    size_t elem_size = cbpy_dtype_size(a->dtype);
    void *result_data = calloc(result_elems, elem_size);
    if (!result_data) return NULL;

    /* 행렬 곱 (시뮬레이션 - 실제로는 cuBLAS 또는 CUDA 커널) */
    if (a->dtype == CB_DTYPE_FLOAT32) {
        const float *pa = (const float *)a->device_ptr;
        const float *pb = (const float *)b->device_ptr;
        float *pr = (float *)result_data;
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                float sum = 0.0f;
                for (size_t k = 0; k < K; k++) {
                    sum += pa[i * K + k] * pb[k * N + j];
                }
                pr[i * N + j] = sum;
            }
        }
    } else if (a->dtype == CB_DTYPE_FLOAT64) {
        const double *pa = (const double *)a->device_ptr;
        const double *pb = (const double *)b->device_ptr;
        double *pr = (double *)result_data;
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                double sum = 0.0;
                for (size_t k = 0; k < K; k++) {
                    sum += pa[i * K + k] * pb[k * N + j];
                }
                pr[i * N + j] = sum;
            }
        }
    } else {
        CB_LOG_ERROR(CB_LOG_CAT_PYTHON, "matmul: unsupported dtype %d", a->dtype);
        free(result_data);
        return NULL;
    }

    size_t result_shape[2] = {M, N};
    CBPyArray *result = cbpy_to_device(result_data, result_elems,
                                        a->dtype, 2, result_shape);
    free(result_data);
    if (result) result->is_synced = 0;
    return result;
}

CBPyArray* cbpy_scalar_op(CBPyArray *arr, double scalar, int op) {
    if (!arr) return NULL;

    CB_LOG_DEBUG(CB_LOG_CAT_PYTHON, "GPU scalar op %d: scalar=%.6f", op, scalar);

    size_t total_size = arr->size;
    void *result_data = malloc(total_size);
    if (!result_data) return NULL;

    if (arr->dtype == CB_DTYPE_FLOAT32) {
        const float *pa = (const float *)arr->device_ptr;
        float *pr = (float *)result_data;
        float s = (float)scalar;
        for (size_t i = 0; i < arr->elem_count; i++) {
            switch (op) {
                case CBPY_OP_ADD: pr[i] = pa[i] + s; break;
                case CBPY_OP_SUB: pr[i] = pa[i] - s; break;
                case CBPY_OP_MUL: pr[i] = pa[i] * s; break;
                case CBPY_OP_DIV: pr[i] = (s != 0.0f) ? pa[i] / s : 0.0f; break;
                default: pr[i] = pa[i];
            }
        }
    } else if (arr->dtype == CB_DTYPE_FLOAT64) {
        const double *pa = (const double *)arr->device_ptr;
        double *pr = (double *)result_data;
        for (size_t i = 0; i < arr->elem_count; i++) {
            switch (op) {
                case CBPY_OP_ADD: pr[i] = pa[i] + scalar; break;
                case CBPY_OP_SUB: pr[i] = pa[i] - scalar; break;
                case CBPY_OP_MUL: pr[i] = pa[i] * scalar; break;
                case CBPY_OP_DIV: pr[i] = (scalar != 0.0) ? pa[i] / scalar : 0.0; break;
                default: pr[i] = pa[i];
            }
        }
    } else {
        free(result_data);
        return NULL;
    }

    CBPyArray *result = cbpy_to_device(result_data, arr->elem_count,
                                        arr->dtype, arr->ndim, arr->shape);
    free(result_data);
    if (result) result->is_synced = 0;
    return result;
}

double cbpy_reduce(CBPyArray *arr, int op) {
    if (!arr) return 0.0;

    CB_LOG_DEBUG(CB_LOG_CAT_PYTHON, "GPU reduce op %d: %lu elements",
                 op, (unsigned long)arr->elem_count);

    double result = 0.0;

    if (arr->dtype == CB_DTYPE_FLOAT32) {
        const float *p = (const float *)arr->device_ptr;
        result = (double)p[0];
        for (size_t i = 1; i < arr->elem_count; i++) {
            switch (op) {
                case CBPY_REDUCE_SUM:
                case CBPY_REDUCE_MEAN:
                    result += (double)p[i];
                    break;
                case CBPY_REDUCE_MAX:
                    if ((double)p[i] > result) { result = (double)p[i]; }
                    break;
                case CBPY_REDUCE_MIN:
                    if ((double)p[i] < result) { result = (double)p[i]; }
                    break;
            }
        }
        if (op == CBPY_REDUCE_MEAN) result /= (double)arr->elem_count;
    } else if (arr->dtype == CB_DTYPE_FLOAT64) {
        const double *p = (const double *)arr->device_ptr;
        result = p[0];
        for (size_t i = 1; i < arr->elem_count; i++) {
            switch (op) {
                case CBPY_REDUCE_SUM:
                case CBPY_REDUCE_MEAN:
                    result += p[i];
                    break;
                case CBPY_REDUCE_MAX:
                    if (p[i] > result) { result = p[i]; }
                    break;
                case CBPY_REDUCE_MIN:
                    if (p[i] < result) { result = p[i]; }
                    break;
            }
        }
        if (op == CBPY_REDUCE_MEAN) result /= (double)arr->elem_count;
    }

    return result;
}

const char* cbpy_device_name(void) {
    static char name[256] = "No device";
    EGPUDeviceInfo info;
    if (egpu_get_device_info(&info) == 0 && info.vendor_id != 0) {
        snprintf(name, sizeof(name), "%s", info.name);
    }
    return name;
}

int cbpy_mem_info(size_t *free_bytes, size_t *total_bytes) {
    /* 시뮬레이션: 24GB VRAM */
    size_t total = (size_t)24 * 1024 * 1024 * 1024;
    if (total_bytes) *total_bytes = total;
    if (free_bytes) *free_bytes = total - g_pyctx.current_usage;
    return 0;
}
