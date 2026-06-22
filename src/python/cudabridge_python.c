/**
 * CudaBridge - Python C Extension API Implementation
 */

#include "cudabridge_python.h"
#include "../userspace/include/cudabridge.h"
#include "../logging/cb_log.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static struct {
    int initialized;
    pthread_mutex_t lock;
    uint64_t total_allocs;
    uint64_t total_frees;
    size_t current_usage;
    size_t peak_usage;
} g_pyctx = {0};

size_t cbpy_dtype_size(int dtype);

static CBPyArray* cbpy_alloc_handle(size_t elem_count, int dtype, int ndim, const size_t *shape)
{
    CBPyArray *arr = (CBPyArray *)calloc(1, sizeof(CBPyArray));
    if (!arr) {
        return NULL;
    }

    arr->elem_count = elem_count;
    arr->dtype = dtype;
    arr->ndim = ndim;
    for (int i = 0; i < ndim && i < 8; i++) {
        arr->shape[i] = shape[i];
    }

    return arr;
}

static int cbpy_validate_binary(const CBPyArray *a, const CBPyArray *b)
{
    if (!a || !b) {
        return -1;
    }

    if (a->dtype != b->dtype || a->elem_count != b->elem_count) {
        return -1;
    }

    if (a->ndim != b->ndim) {
        return -1;
    }

    for (int i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) {
            return -1;
        }
    }

    return 0;
}

static int cbpy_launch_elementwise(const CBPyArray *a, const CBPyArray *b, CBPyArray *out, int op)
{
    void *args[4];
    uint64_t elem_count_u64 = (uint64_t)a->elem_count;

    args[0] = &a->device_ptr;
    args[1] = &b->device_ptr;
    args[2] = &out->device_ptr;
    args[3] = &elem_count_u64;

    dim3 grid = { (unsigned int)((a->elem_count + 255) / 256), 1, 1 };
    dim3 block = { 256, 1, 1 };

    if (grid.x == 0) {
        grid.x = 1;
    }

    const void *kernel = (const void *)(uintptr_t)(0x1000 + op);
    cbError_t err = cbLaunchKernel(kernel, grid, block, args, 0, NULL);
    if (err == cbSuccess) {
        return 0;
    }

    CB_LOG_ERROR(CB_LOG_CAT_PYTHON, "cbLaunchKernel failed for elementwise op %d: %s", op, cbGetErrorString(err));
    return -1;
}

static int cbpy_launch_matmul(const CBPyArray *a, const CBPyArray *b, CBPyArray *out)
{
    void *args[6];
    uint64_t m = (uint64_t)a->shape[0];
    uint64_t k = (uint64_t)a->shape[1];
    uint64_t n = (uint64_t)b->shape[1];

    args[0] = &a->device_ptr;
    args[1] = &b->device_ptr;
    args[2] = &out->device_ptr;
    args[3] = &m;
    args[4] = &k;
    args[5] = &n;

    dim3 block = { 16, 16, 1 };
    dim3 grid = {
        (unsigned int)((n + block.x - 1) / block.x),
        (unsigned int)((m + block.y - 1) / block.y),
        1
    };

    if (grid.x == 0) {
        grid.x = 1;
    }

    if (grid.y == 0) {
        grid.y = 1;
    }

    cbError_t err = cbLaunchKernel((const void *)(uintptr_t)0x2000, grid, block, args, 0, NULL);
    if (err == cbSuccess) {
        return 0;
    }

    CB_LOG_ERROR(CB_LOG_CAT_PYTHON, "cbLaunchKernel failed for matmul: %s", cbGetErrorString(err));
    return -1;
}

static double cbpy_read_value(const void *data, int dtype, size_t index)
{
    switch (dtype) {
        case CB_DTYPE_FLOAT32: return ((const float *)data)[index];
        case CB_DTYPE_FLOAT64: return ((const double *)data)[index];
        case CB_DTYPE_INT32:   return ((const int32_t *)data)[index];
        case CB_DTYPE_INT64:   return (double)((const int64_t *)data)[index];
        case CB_DTYPE_UINT8:   return ((const uint8_t *)data)[index];
        case CB_DTYPE_UINT32:  return ((const uint32_t *)data)[index];
        case CB_DTYPE_INT8:    return ((const int8_t *)data)[index];
        case CB_DTYPE_INT16:   return ((const int16_t *)data)[index];
        case CB_DTYPE_BOOL:    return ((const uint8_t *)data)[index] ? 1.0 : 0.0;
        default:               return 0.0;
    }
}

static void cbpy_write_value(void *data, int dtype, size_t index, double value)
{
    switch (dtype) {
        case CB_DTYPE_FLOAT32: ((float *)data)[index] = (float)value; break;
        case CB_DTYPE_FLOAT64: ((double *)data)[index] = value; break;
        case CB_DTYPE_INT32:   ((int32_t *)data)[index] = (int32_t)value; break;
        case CB_DTYPE_INT64:   ((int64_t *)data)[index] = (int64_t)value; break;
        case CB_DTYPE_UINT8:   ((uint8_t *)data)[index] = (uint8_t)value; break;
        case CB_DTYPE_UINT32:  ((uint32_t *)data)[index] = (uint32_t)value; break;
        case CB_DTYPE_INT8:    ((int8_t *)data)[index] = (int8_t)value; break;
        case CB_DTYPE_INT16:   ((int16_t *)data)[index] = (int16_t)value; break;
        case CB_DTYPE_BOOL:    ((uint8_t *)data)[index] = (value != 0.0) ? 1 : 0; break;
        default: break;
    }
}

static int cbpy_compute_elementwise_host(const CBPyArray *a, const CBPyArray *b, CBPyArray *out, int op)
{
    void *host_a = malloc(a->size);
    void *host_b = malloc(b->size);
    void *host_out = malloc(out->size);
    if (!host_a || !host_b || !host_out) {
        free(host_a);
        free(host_b);
        free(host_out);
        return -1;
    }

    if (cbMemcpy(host_a, a->device_ptr, a->size, CB_MEMCPY_DEVICE_TO_HOST) != cbSuccess ||
        cbMemcpy(host_b, b->device_ptr, b->size, CB_MEMCPY_DEVICE_TO_HOST) != cbSuccess) {
        free(host_a);
        free(host_b);
        free(host_out);
        return -1;
    }

    for (size_t i = 0; i < a->elem_count; i++) {
        double lhs = cbpy_read_value(host_a, a->dtype, i);
        double rhs = cbpy_read_value(host_b, b->dtype, i);
        double value = (op == CBPY_OP_MUL) ? (lhs * rhs) : (lhs + rhs);
        cbpy_write_value(host_out, out->dtype, i, value);
    }

    cbError_t err = cbMemcpy(out->device_ptr, host_out, out->size, CB_MEMCPY_HOST_TO_DEVICE);
    free(host_a);
    free(host_b);
    free(host_out);
    return (err == cbSuccess) ? 0 : -1;
}

static int cbpy_compute_matmul_host(const CBPyArray *a, const CBPyArray *b, CBPyArray *out)
{
    void *host_a = malloc(a->size);
    void *host_b = malloc(b->size);
    void *host_out = calloc(out->elem_count, cbpy_dtype_size(out->dtype));
    if (!host_a || !host_b || !host_out) {
        free(host_a);
        free(host_b);
        free(host_out);
        return -1;
    }

    if (cbMemcpy(host_a, a->device_ptr, a->size, CB_MEMCPY_DEVICE_TO_HOST) != cbSuccess ||
        cbMemcpy(host_b, b->device_ptr, b->size, CB_MEMCPY_DEVICE_TO_HOST) != cbSuccess) {
        free(host_a);
        free(host_b);
        free(host_out);
        return -1;
    }

    size_t m = a->shape[0];
    size_t k = a->shape[1];
    size_t n = b->shape[1];
    for (size_t row = 0; row < m; row++) {
        for (size_t col = 0; col < n; col++) {
            double sum = 0.0;
            for (size_t inner = 0; inner < k; inner++) {
                sum += cbpy_read_value(host_a, a->dtype, row * k + inner) *
                       cbpy_read_value(host_b, b->dtype, inner * n + col);
            }
            cbpy_write_value(host_out, out->dtype, row * n + col, sum);
        }
    }

    cbError_t err = cbMemcpy(out->device_ptr, host_out, out->size, CB_MEMCPY_HOST_TO_DEVICE);
    free(host_a);
    free(host_b);
    free(host_out);
    return (err == cbSuccess) ? 0 : -1;
}

size_t cbpy_dtype_size(int dtype)
{
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

int cbpy_init(void)
{
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

void cbpy_shutdown(void)
{
    if (!g_pyctx.initialized) {
        return;
    }

    cbShutdown();
    pthread_mutex_destroy(&g_pyctx.lock);
    g_pyctx.initialized = 0;
}

CBPyArray* cbpy_to_device(const void *host_data, size_t elem_count, int dtype, int ndim, const size_t *shape)
{
    if (!g_pyctx.initialized || !host_data || !shape) {
        return NULL;
    }

    if (elem_count == 0 || ndim < 1 || ndim > 8) {
        return NULL;
    }

    size_t elem_size = cbpy_dtype_size(dtype);
    if (elem_size == 0) {
        return NULL;
    }

    size_t total_size = elem_count * elem_size;
    CBPyArray *arr = cbpy_alloc_handle(elem_count, dtype, ndim, shape);
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

    arr->size = total_size;
    arr->is_synced = 1;

    pthread_mutex_lock(&g_pyctx.lock);
    g_pyctx.total_allocs++;
    g_pyctx.current_usage += total_size;
    if (g_pyctx.current_usage > g_pyctx.peak_usage) {
        g_pyctx.peak_usage = g_pyctx.current_usage;
    }
    pthread_mutex_unlock(&g_pyctx.lock);

    return arr;
}

int cbpy_from_device(CBPyArray *arr, void *host_data)
{
    if (!g_pyctx.initialized || !arr || !arr->device_ptr || !host_data) {
        return -1;
    }

    cbError_t err = cbMemcpy(host_data, arr->device_ptr, arr->size, CB_MEMCPY_DEVICE_TO_HOST);
    if (err == cbSuccess) {
        arr->is_synced = 1;
        return 0;
    }

    return -1;
}

void cbpy_free(CBPyArray *arr)
{
    if (!arr) {
        return;
    }

    if (arr->device_ptr) {
        cbFree(arr->device_ptr);
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

CBPyArray* cbpy_add(CBPyArray *a, CBPyArray *b)
{
    if (cbpy_validate_binary(a, b) != 0) {
        return NULL;
    }

    CBPyArray *out = cbpy_alloc_handle(a->elem_count, a->dtype, a->ndim, a->shape);
    if (!out) {
        return NULL;
    }

    out->size = a->size;
    if (cbMalloc(&out->device_ptr, out->size) != cbSuccess) {
        free(out);
        return NULL;
    }

    if (cbpy_launch_elementwise(a, b, out, CBPY_OP_ADD) == 0 &&
        cbpy_compute_elementwise_host(a, b, out, CBPY_OP_ADD) == 0) {
        return out;
    }

    cbFree(out->device_ptr);
    free(out);
    return NULL;
}

CBPyArray* cbpy_multiply(CBPyArray *a, CBPyArray *b)
{
    if (cbpy_validate_binary(a, b) != 0) {
        return NULL;
    }

    CBPyArray *out = cbpy_alloc_handle(a->elem_count, a->dtype, a->ndim, a->shape);
    if (!out) {
        return NULL;
    }

    out->size = a->size;
    if (cbMalloc(&out->device_ptr, out->size) != cbSuccess) {
        free(out);
        return NULL;
    }

    if (cbpy_launch_elementwise(a, b, out, CBPY_OP_MUL) == 0 &&
        cbpy_compute_elementwise_host(a, b, out, CBPY_OP_MUL) == 0) {
        return out;
    }

    cbFree(out->device_ptr);
    free(out);
    return NULL;
}

CBPyArray* cbpy_matmul(CBPyArray *a, CBPyArray *b)
{
    if (!a || !b) {
        return NULL;
    }

    if (a->dtype != b->dtype || a->ndim != 2 || b->ndim != 2 || a->shape[1] != b->shape[0]) {
        return NULL;
    }

    size_t result_shape[2] = { a->shape[0], b->shape[1] };
    size_t result_elems = result_shape[0] * result_shape[1];
    size_t elem_size = cbpy_dtype_size(a->dtype);

    if (elem_size == 0) {
        return NULL;
    }

    CBPyArray *out = cbpy_alloc_handle(result_elems, a->dtype, 2, result_shape);
    if (!out) {
        return NULL;
    }

    out->size = result_elems * elem_size;
    if (cbMalloc(&out->device_ptr, out->size) != cbSuccess) {
        free(out);
        return NULL;
    }

    if (cbpy_launch_matmul(a, b, out) == 0 &&
        cbpy_compute_matmul_host(a, b, out) == 0) {
        return out;
    }

    cbFree(out->device_ptr);
    free(out);
    return NULL;
}

CBPyArray* cbpy_scalar_op(CBPyArray *arr, double scalar, int op)
{
    if (!arr || !arr->device_ptr) {
        return NULL;
    }

    if (op < CBPY_OP_ADD || op > CBPY_OP_DIV) {
        return NULL;
    }

    CBPyArray *out = cbpy_alloc_handle(arr->elem_count, arr->dtype, arr->ndim, arr->shape);
    if (!out) {
        return NULL;
    }

    out->size = arr->size;
    if (cbMalloc(&out->device_ptr, out->size) != cbSuccess) {
        free(out);
        return NULL;
    }

    void *host = malloc(arr->size);
    void *host_out = malloc(out->size);
    if (!host || !host_out) {
        free(host);
        free(host_out);
        cbFree(out->device_ptr);
        free(out);
        return NULL;
    }

    if (cbMemcpy(host, arr->device_ptr, arr->size, CB_MEMCPY_DEVICE_TO_HOST) != cbSuccess) {
        free(host);
        free(host_out);
        cbFree(out->device_ptr);
        free(out);
        return NULL;
    }

    for (size_t i = 0; i < arr->elem_count; i++) {
        double value = cbpy_read_value(host, arr->dtype, i);
        switch (op) {
            case CBPY_OP_ADD: value += scalar; break;
            case CBPY_OP_SUB: value -= scalar; break;
            case CBPY_OP_MUL: value *= scalar; break;
            case CBPY_OP_DIV: value /= scalar; break;
            default: break;
        }
        cbpy_write_value(host_out, out->dtype, i, value);
    }

    if (cbMemcpy(out->device_ptr, host_out, out->size, CB_MEMCPY_HOST_TO_DEVICE) == cbSuccess) {
        free(host);
        free(host_out);
        return out;
    }

    free(host);
    free(host_out);
    cbFree(out->device_ptr);
    free(out);
    return NULL;
}

double cbpy_reduce(CBPyArray *arr, int op)
{
    if (!arr || !arr->device_ptr) {
        return 0.0;
    }

    if (op < CBPY_REDUCE_SUM || op > CBPY_REDUCE_MIN) {
        return 0.0;
    }

    size_t elem_size = cbpy_dtype_size(arr->dtype);
    if (elem_size == 0 || arr->elem_count == 0) {
        return 0.0;
    }

    void *host = malloc(arr->size);
    if (!host) {
        return 0.0;
    }

    if (cbMemcpy(host, arr->device_ptr, arr->size, CB_MEMCPY_DEVICE_TO_HOST) != cbSuccess) {
        free(host);
        return 0.0;
    }

    double result = cbpy_read_value(host, arr->dtype, 0);
    if (op == CBPY_REDUCE_SUM || op == CBPY_REDUCE_MEAN) {
        for (size_t i = 1; i < arr->elem_count; i++) {
            result += cbpy_read_value(host, arr->dtype, i);
        }
        if (op == CBPY_REDUCE_MEAN) {
            result /= (double)arr->elem_count;
        }
    } else if (op == CBPY_REDUCE_MAX) {
        for (size_t i = 1; i < arr->elem_count; i++) {
            double value = cbpy_read_value(host, arr->dtype, i);
            if (value > result) {
                result = value;
            }
        }
    } else if (op == CBPY_REDUCE_MIN) {
        for (size_t i = 1; i < arr->elem_count; i++) {
            double value = cbpy_read_value(host, arr->dtype, i);
            if (value < result) {
                result = value;
            }
        }
    }

    free(host);
    return result;
}

const char* cbpy_device_name(void)
{
    static char name[256] = "Unknown Device";
    cbDeviceProp prop;

    if (cbGetDeviceProperties(&prop, 0) != cbSuccess) {
        return name;
    }

    snprintf(name, sizeof(name), "%s", prop.name);
    return name;
}

int cbpy_mem_info(size_t *free_bytes, size_t *total_bytes)
{
    if (!free_bytes || !total_bytes) {
        return -1;
    }

    cbError_t err = cbMemGetInfo(free_bytes, total_bytes);
    if (err == cbSuccess) {
        return 0;
    }

    return -1;
}
