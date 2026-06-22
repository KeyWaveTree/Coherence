/**
 * CudaBridge - Python C Extension API Implementation
 */

#include "cudabridge_python.h"
#include "../userspace/include/cudabridge.h"
#include "../logging/cb_log.h"
#include "../egpu/egpu_safety.h"

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static struct {
    int             initialized;
    pthread_mutex_t lock;
    uint64_t        total_allocs;
    uint64_t        total_frees;
    size_t          current_usage;
    size_t          peak_usage;
} g_pyctx = {0};

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

static uint32_t compute_crc32(const void *data, size_t size) {
    return egpu_compute_crc32(data, size);
}

static int validate_array(const CBPyArray *arr) {
    if (!arr) {
        return -1;
    }

    if (!arr->device_ptr || arr->size == 0) {
        return -1;
    }

    return 0;
}

static CBPyArray* alloc_array_handle(size_t elem_count, size_t total_size, int dtype, int ndim, const size_t *shape) {
    CBPyArray *arr = (CBPyArray *)calloc(1, sizeof(CBPyArray));
    if (!arr) {
        return NULL;
    }

    arr->size = total_size;
    arr->elem_count = elem_count;
    arr->dtype = dtype;
    arr->ndim = ndim;
    for (int i = 0; i < ndim && i < 8; i++) {
        arr->shape[i] = shape ? shape[i] : elem_count;
    }

    return arr;
}

int cbpy_init(void) {
    if (g_pyctx.initialized) {
        return 0;
    }

    cb_log_init_default();

    cbError_t err = cbInit();
    if (err != cbSuccess) {
        CB_LOG_ERROR(CB_LOG_CAT_PYTHON, "cbInit failed: %s", cbGetErrorString(err));
        return -1;
    }

    pthread_mutex_init(&g_pyctx.lock, NULL);
    g_pyctx.total_allocs = 0;
    g_pyctx.total_frees = 0;
    g_pyctx.current_usage = 0;
    g_pyctx.peak_usage = 0;
    g_pyctx.initialized = 1;

    return 0;
}

void cbpy_shutdown(void) {
    if (!g_pyctx.initialized) {
        return;
    }

    cbShutdown();
    pthread_mutex_destroy(&g_pyctx.lock);
    g_pyctx.initialized = 0;
}

CBPyArray* cbpy_to_device(const void *host_data, size_t elem_count,
                           int dtype, int ndim, const size_t *shape) {
    if (!g_pyctx.initialized || !host_data || elem_count == 0) {
        return NULL;
    }

    if (ndim < 1 || ndim > 8) {
        return NULL;
    }

    size_t elem_size = cbpy_dtype_size(dtype);
    if (elem_size == 0) {
        return NULL;
    }

    size_t total_size = elem_count * elem_size;
    CBPyArray *arr = alloc_array_handle(elem_count, total_size, dtype, ndim, shape);
    if (!arr) {
        return NULL;
    }

    cbError_t err = cbMalloc(&arr->device_ptr, total_size);
    if (err != cbSuccess) {
        free(arr);
        return NULL;
    }

    err = cbMemcpy(arr->device_ptr, host_data, total_size, CB_MEMCPY_HOST_TO_DEVICE);
    if (err != cbSuccess) {
        cbFree(arr->device_ptr);
        free(arr);
        return NULL;
    }

    arr->is_synced = 1;
    arr->crc32 = compute_crc32(host_data, total_size);

    pthread_mutex_lock(&g_pyctx.lock);
    g_pyctx.total_allocs++;
    g_pyctx.current_usage += total_size;
    if (g_pyctx.current_usage > g_pyctx.peak_usage) {
        g_pyctx.peak_usage = g_pyctx.current_usage;
    }
    pthread_mutex_unlock(&g_pyctx.lock);

    return arr;
}

int cbpy_from_device(CBPyArray *arr, void *host_data) {
    if (validate_array(arr) != 0 || !host_data) {
        return -1;
    }

    cbError_t err = cbMemcpy(host_data, arr->device_ptr, arr->size, CB_MEMCPY_DEVICE_TO_HOST);
    if (err != cbSuccess) {
        return -1;
    }

    arr->is_synced = 1;
    return 0;
}

void cbpy_free(CBPyArray *arr) {
    if (!arr) {
        return;
    }

    if (arr->device_ptr) {
        cbFree(arr->device_ptr);
        arr->device_ptr = NULL;
    }

    pthread_mutex_lock(&g_pyctx.lock);
    g_pyctx.total_frees++;
    if (g_pyctx.current_usage >= arr->size) {
        g_pyctx.current_usage -= arr->size;
    } else {
        g_pyctx.current_usage = 0;
    }
    pthread_mutex_unlock(&g_pyctx.lock);

    free(arr);
}

static int copy_to_host(CBPyArray *arr, void **host_buf) {
    if (validate_array(arr) != 0 || !host_buf) {
        return -1;
    }

    void *buf = malloc(arr->size);
    if (!buf) {
        return -1;
    }

    cbError_t err = cbMemcpy(buf, arr->device_ptr, arr->size, CB_MEMCPY_DEVICE_TO_HOST);
    if (err != cbSuccess) {
        free(buf);
        return -1;
    }

    *host_buf = buf;
    return 0;
}

static CBPyArray* copy_result_to_device(void *host_buf, size_t elem_count, int dtype,
                                        int ndim, const size_t *shape) {
    CBPyArray *result = cbpy_to_device(host_buf, elem_count, dtype, ndim, shape);
    if (result) {
        result->is_synced = 0;
    }
    return result;
}

static int validate_binary_inputs(CBPyArray *a, CBPyArray *b) {
    if (validate_array(a) != 0 || validate_array(b) != 0) {
        return -1;
    }

    if (a->elem_count != b->elem_count || a->dtype != b->dtype || a->ndim != b->ndim) {
        return -1;
    }

    for (int i = 0; i < a->ndim; i++) {
        if (a->shape[i] == b->shape[i]) {
            continue;
        }

        return -1;
    }

    return 0;
}

static CBPyArray* launch_binary_kernel(CBPyArray *a, CBPyArray *b, const void *kernel_ptr) {
    if (validate_binary_inputs(a, b) != 0 || !kernel_ptr) {
        return NULL;
    }

    CBPyArray *out = alloc_array_handle(a->elem_count, a->size, a->dtype, a->ndim, a->shape);
    if (!out) {
        return NULL;
    }

    cbError_t err = cbMalloc(&out->device_ptr, out->size);
    if (err != cbSuccess) {
        free(out);
        return NULL;
    }

    void *args[] = { &out->device_ptr, &a->device_ptr, &b->device_ptr, &a->elem_count };
    dim3 block = {256, 1, 1};
    dim3 grid = {(unsigned int)((a->elem_count + block.x - 1) / block.x), 1, 1};

    err = cbLaunchKernel(kernel_ptr, grid, block, args, 0, NULL);
    if (err != cbSuccess) {
        cbFree(out->device_ptr);
        free(out);
        return NULL;
    }

    err = cbDeviceSynchronize();
    if (err != cbSuccess) {
        cbFree(out->device_ptr);
        free(out);
        return NULL;
    }

    pthread_mutex_lock(&g_pyctx.lock);
    g_pyctx.total_allocs++;
    g_pyctx.current_usage += out->size;
    if (g_pyctx.current_usage > g_pyctx.peak_usage) {
        g_pyctx.peak_usage = g_pyctx.current_usage;
    }
    pthread_mutex_unlock(&g_pyctx.lock);

    out->is_synced = 0;
    return out;
}

static CBPyArray* binary_host_op(CBPyArray *a, CBPyArray *b, int op) {
    (void)launch_binary_kernel;

    if (validate_binary_inputs(a, b) != 0) {
        return NULL;
    }

    size_t elem_size = cbpy_dtype_size(a->dtype);
    if (elem_size == 0) {
        return NULL;
    }

    void *a_buf = NULL;
    void *b_buf = NULL;
    void *out_buf = malloc(a->size);
    if (!out_buf) {
        return NULL;
    }

    if (copy_to_host(a, &a_buf) != 0 || copy_to_host(b, &b_buf) != 0) {
        free(a_buf);
        free(b_buf);
        free(out_buf);
        return NULL;
    }

#define APPLY_BINARY(T) do { \
        T *ap = (T *)a_buf; \
        T *bp = (T *)b_buf; \
        T *optr = (T *)out_buf; \
        for (size_t i = 0; i < a->elem_count; i++) { \
            optr[i] = (op == CBPY_OP_ADD) ? (T)(ap[i] + bp[i]) : (T)(ap[i] * bp[i]); \
        } \
    } while (0)

    switch (a->dtype) {
        case CB_DTYPE_FLOAT32: APPLY_BINARY(float); break;
        case CB_DTYPE_FLOAT64: APPLY_BINARY(double); break;
        case CB_DTYPE_INT32:   APPLY_BINARY(int32_t); break;
        case CB_DTYPE_INT64:   APPLY_BINARY(int64_t); break;
        case CB_DTYPE_UINT8:   APPLY_BINARY(uint8_t); break;
        case CB_DTYPE_UINT32:  APPLY_BINARY(uint32_t); break;
        case CB_DTYPE_INT8:    APPLY_BINARY(int8_t); break;
        case CB_DTYPE_INT16:   APPLY_BINARY(int16_t); break;
        case CB_DTYPE_BOOL:    APPLY_BINARY(uint8_t); break;
        default:
            free(a_buf);
            free(b_buf);
            free(out_buf);
            return NULL;
    }

#undef APPLY_BINARY

    CBPyArray *result = copy_result_to_device(out_buf, a->elem_count, a->dtype, a->ndim, a->shape);
    free(a_buf);
    free(b_buf);
    free(out_buf);
    return result;
}

CBPyArray* cbpy_add(CBPyArray *a, CBPyArray *b) {
    return binary_host_op(a, b, CBPY_OP_ADD);
}

CBPyArray* cbpy_multiply(CBPyArray *a, CBPyArray *b) {
    return binary_host_op(a, b, CBPY_OP_MUL);
}

CBPyArray* cbpy_matmul(CBPyArray *a, CBPyArray *b) {
    if (validate_array(a) != 0 || validate_array(b) != 0) {
        return NULL;
    }

    if (a->ndim != 2 || b->ndim != 2) {
        return NULL;
    }

    if (a->shape[1] != b->shape[0] || a->dtype != b->dtype) {
        return NULL;
    }

    if (a->dtype != CB_DTYPE_FLOAT32 && a->dtype != CB_DTYPE_FLOAT64) {
        return NULL;
    }

    void *a_buf = NULL;
    void *b_buf = NULL;
    size_t out_shape[2] = {a->shape[0], b->shape[1]};
    size_t out_count = out_shape[0] * out_shape[1];
    size_t out_size = out_count * cbpy_dtype_size(a->dtype);
    void *out_buf = calloc(out_count, cbpy_dtype_size(a->dtype));
    if (!out_buf) {
        return NULL;
    }

    if (copy_to_host(a, &a_buf) != 0 || copy_to_host(b, &b_buf) != 0) {
        free(a_buf);
        free(b_buf);
        free(out_buf);
        return NULL;
    }

    size_t m = a->shape[0];
    size_t k = a->shape[1];
    size_t n = b->shape[1];
    if (a->dtype == CB_DTYPE_FLOAT32) {
        float *ap = (float *)a_buf;
        float *bp = (float *)b_buf;
        float *op = (float *)out_buf;
        for (size_t row = 0; row < m; row++) {
            for (size_t col = 0; col < n; col++) {
                float sum = 0.0f;
                for (size_t inner = 0; inner < k; inner++) {
                    sum += ap[row * k + inner] * bp[inner * n + col];
                }
                op[row * n + col] = sum;
            }
        }
    } else {
        double *ap = (double *)a_buf;
        double *bp = (double *)b_buf;
        double *op = (double *)out_buf;
        for (size_t row = 0; row < m; row++) {
            for (size_t col = 0; col < n; col++) {
                double sum = 0.0;
                for (size_t inner = 0; inner < k; inner++) {
                    sum += ap[row * k + inner] * bp[inner * n + col];
                }
                op[row * n + col] = sum;
            }
        }
    }

    CBPyArray *result = copy_result_to_device(out_buf, out_count, a->dtype, 2, out_shape);
    if (result) {
        result->size = out_size;
    }
    free(a_buf);
    free(b_buf);
    free(out_buf);
    return result;
}

CBPyArray* cbpy_scalar_op(CBPyArray *arr, double scalar, int op) {
    if (validate_array(arr) != 0) {
        return NULL;
    }

    size_t elem_size = cbpy_dtype_size(arr->dtype);
    if (elem_size == 0) {
        return NULL;
    }

    void *host_buf = malloc(arr->size);
    if (!host_buf) {
        return NULL;
    }

    cbError_t err = cbMemcpy(host_buf, arr->device_ptr, arr->size, CB_MEMCPY_DEVICE_TO_HOST);
    if (err != cbSuccess) {
        free(host_buf);
        return NULL;
    }

    if (arr->dtype == CB_DTYPE_FLOAT32) {
        float *p = (float *)host_buf;
        float s = (float)scalar;
        for (size_t i = 0; i < arr->elem_count; i++) {
            if (op == CBPY_OP_ADD) { p[i] += s; continue; }
            if (op == CBPY_OP_SUB) { p[i] -= s; continue; }
            if (op == CBPY_OP_MUL) { p[i] *= s; continue; }
            if (op == CBPY_OP_DIV) { p[i] = (s != 0.0f) ? p[i] / s : 0.0f; }
        }
    } else if (arr->dtype == CB_DTYPE_FLOAT64) {
        double *p = (double *)host_buf;
        for (size_t i = 0; i < arr->elem_count; i++) {
            if (op == CBPY_OP_ADD) { p[i] += scalar; continue; }
            if (op == CBPY_OP_SUB) { p[i] -= scalar; continue; }
            if (op == CBPY_OP_MUL) { p[i] *= scalar; continue; }
            if (op == CBPY_OP_DIV) { p[i] = (scalar != 0.0) ? p[i] / scalar : 0.0; }
        }
    } else {
        free(host_buf);
        return NULL;
    }

    CBPyArray *result = cbpy_to_device(host_buf, arr->elem_count, arr->dtype, arr->ndim, arr->shape);
    free(host_buf);
    return result;
}

double cbpy_reduce(CBPyArray *arr, int op) {
    if (validate_array(arr) != 0) {
        return 0.0;
    }

    void *host_buf = malloc(arr->size);
    if (!host_buf) {
        return 0.0;
    }

    cbError_t err = cbMemcpy(host_buf, arr->device_ptr, arr->size, CB_MEMCPY_DEVICE_TO_HOST);
    if (err != cbSuccess) {
        free(host_buf);
        return 0.0;
    }

    double result = 0.0;
    if (arr->dtype == CB_DTYPE_FLOAT32) {
        float *p = (float *)host_buf;
        result = (double)p[0];
        for (size_t i = 1; i < arr->elem_count; i++) {
            if (op == CBPY_REDUCE_SUM || op == CBPY_REDUCE_MEAN) { result += (double)p[i]; continue; }
            if (op == CBPY_REDUCE_MAX && (double)p[i] > result) { result = (double)p[i]; continue; }
            if (op == CBPY_REDUCE_MIN && (double)p[i] < result) { result = (double)p[i]; }
        }
    } else if (arr->dtype == CB_DTYPE_FLOAT64) {
        double *p = (double *)host_buf;
        result = p[0];
        for (size_t i = 1; i < arr->elem_count; i++) {
            if (op == CBPY_REDUCE_SUM || op == CBPY_REDUCE_MEAN) { result += p[i]; continue; }
            if (op == CBPY_REDUCE_MAX && p[i] > result) { result = p[i]; continue; }
            if (op == CBPY_REDUCE_MIN && p[i] < result) { result = p[i]; }
        }
    }

    if (op == CBPY_REDUCE_MEAN && arr->elem_count > 0) {
        result /= (double)arr->elem_count;
    }

    free(host_buf);
    return result;
}

const char* cbpy_device_name(void) {
    static char name[256] = "No device";

    cbDeviceProp prop;
    cbError_t err = cbGetDeviceProperties(&prop, 0);
    if (err != cbSuccess) {
        return name;
    }

    snprintf(name, sizeof(name), "%s", prop.name);
    return name;
}

int cbpy_mem_info(size_t *free_bytes, size_t *total_bytes) {
    cbError_t err = cbMemGetInfo(free_bytes, total_bytes);
    if (err == cbSuccess) {
        return 0;
    }

    return -1;
}
