#include "cudabridge_python.h"

#include "../userspace/include/cudabridge.h"
#include "../logging/cb_log.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__GNUC__)
#define WEAK_SYMBOL __attribute__((weak))
#else
#define WEAK_SYMBOL
#endif

/* optional kernel entry symbols supplied by runtime/driver integration */
extern void cbpy_kernel_add_f32(void) WEAK_SYMBOL;
extern void cbpy_kernel_add_f64(void) WEAK_SYMBOL;
extern void cbpy_kernel_add_i32(void) WEAK_SYMBOL;
extern void cbpy_kernel_mul_f32(void) WEAK_SYMBOL;
extern void cbpy_kernel_mul_f64(void) WEAK_SYMBOL;
extern void cbpy_kernel_mul_i32(void) WEAK_SYMBOL;
extern void cbpy_kernel_matmul_f32(void) WEAK_SYMBOL;
extern void cbpy_kernel_matmul_f64(void) WEAK_SYMBOL;

static struct {
    int initialized;
    pthread_mutex_t lock;
    uint64_t total_allocs;
    uint64_t total_frees;
    size_t current_usage;
    size_t peak_usage;
} g_pyctx = {0};

size_t cbpy_dtype_size(int dtype) {
    switch (dtype) {
        case CB_DTYPE_FLOAT32: return 4;
        case CB_DTYPE_FLOAT64: return 8;
        case CB_DTYPE_INT32: return 4;
        case CB_DTYPE_INT64: return 8;
        case CB_DTYPE_UINT8: return 1;
        case CB_DTYPE_UINT32: return 4;
        case CB_DTYPE_INT8: return 1;
        case CB_DTYPE_INT16: return 2;
        case CB_DTYPE_FLOAT16: return 2;
        case CB_DTYPE_BOOL: return 1;
        default: return 0;
    }
}

static int ensure_initialized(void) {
    if (g_pyctx.initialized) {
        return 0;
    }
    return -1;
}

static CBPyArray* create_device_array(size_t elem_count, int dtype, int ndim, const size_t *shape) {
    if (elem_count == 0 || ndim < 1 || ndim > 8) {
        return NULL;
    }

    size_t elem_size = cbpy_dtype_size(dtype);
    if (elem_size == 0) {
        return NULL;
    }

    size_t total_size = elem_count * elem_size;
    if (total_size < elem_count) {
        return NULL;
    }

    CBPyArray *arr = calloc(1, sizeof(CBPyArray));
    if (!arr) {
        return NULL;
    }

    cbError_t alloc_err = cbMalloc(&arr->device_ptr, total_size);
    if (alloc_err != cbSuccess) {
        free(arr);
        return NULL;
    }

    arr->size = total_size;
    arr->elem_count = elem_count;
    arr->dtype = dtype;
    arr->ndim = ndim;
    for (int i = 0; i < ndim; i++) {
        arr->shape[i] = shape[i];
    }

    pthread_mutex_lock(&g_pyctx.lock);
    g_pyctx.total_allocs++;
    g_pyctx.current_usage += total_size;
    if (g_pyctx.current_usage > g_pyctx.peak_usage) {
        g_pyctx.peak_usage = g_pyctx.current_usage;
    }
    pthread_mutex_unlock(&g_pyctx.lock);

    return arr;
}

static int validate_binary_args(CBPyArray *a, CBPyArray *b) {
    if (!a || !b) return -1;
    if (a->elem_count != b->elem_count) return -1;
    if (a->dtype != b->dtype) return -1;
    if (a->ndim != b->ndim) return -1;
    for (int i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) return -1;
    }
    return 0;
}

static cbError_t launch_elementwise_kernel(const void *kernel,
                                           void *out_ptr,
                                           void *a_ptr,
                                           void *b_ptr,
                                           size_t elem_count) {
    if (!kernel) {
        return cbErrorUnknown;
    }

    void *args[] = { &out_ptr, &a_ptr, &b_ptr, &elem_count };
    dim3 block = {256, 1, 1};
    dim3 grid = {(unsigned int)((elem_count + 255) / 256), 1, 1};

    cbError_t launch_err = cbLaunchKernel(kernel, grid, block, args, 0, NULL);
    if (launch_err != cbSuccess) {
        return launch_err;
    }

    return cbDeviceSynchronize();
}

static cbError_t launch_matmul_kernel(const void *kernel,
                                      void *out_ptr,
                                      void *a_ptr,
                                      void *b_ptr,
                                      size_t m,
                                      size_t n,
                                      size_t k) {
    if (!kernel) {
        return cbErrorUnknown;
    }

    void *args[] = { &out_ptr, &a_ptr, &b_ptr, &m, &n, &k };
    dim3 block = {16, 16, 1};
    dim3 grid = {(unsigned int)((n + 15) / 16), (unsigned int)((m + 15) / 16), 1};

    cbError_t launch_err = cbLaunchKernel(kernel, grid, block, args, 0, NULL);
    if (launch_err != cbSuccess) {
        return launch_err;
    }

    return cbDeviceSynchronize();
}

int cbpy_init(void) {
    if (g_pyctx.initialized) {
        return 0;
    }

    cb_log_init_default();

    cbError_t init_err = cbInit();
    if (init_err != cbSuccess) {
        CB_LOG_ERROR(CB_LOG_CAT_PYTHON, "cbInit failed: %s", cbGetErrorString(init_err));
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
    if (ensure_initialized() != 0) {
        return NULL;
    }

    if (!host_data || !shape) {
        return NULL;
    }

    CBPyArray *arr = create_device_array(elem_count, dtype, ndim, shape);
    if (!arr) {
        return NULL;
    }

    cbError_t copy_err = cbMemcpy(arr->device_ptr, host_data, arr->size, CB_MEMCPY_HOST_TO_DEVICE);
    if (copy_err == cbSuccess) {
        return arr;
    }

    cbFree(arr->device_ptr);
    free(arr);
    return NULL;
}

int cbpy_from_device(CBPyArray *arr, void *host_data) {
    if (ensure_initialized() != 0) {
        return -1;
    }

    if (!arr || !host_data || !arr->device_ptr || arr->size == 0) {
        return -1;
    }

    cbError_t copy_err = cbMemcpy(host_data, arr->device_ptr, arr->size, CB_MEMCPY_DEVICE_TO_HOST);
    if (copy_err == cbSuccess) {
        return 0;
    }

    return -1;
}

void cbpy_free(CBPyArray *arr) {
    if (!arr) {
        return;
    }

    if (arr->device_ptr) {
        cbFree(arr->device_ptr);
        pthread_mutex_lock(&g_pyctx.lock);
        g_pyctx.total_frees++;
        if (g_pyctx.current_usage >= arr->size) {
            g_pyctx.current_usage -= arr->size;
        }
        pthread_mutex_unlock(&g_pyctx.lock);
    }

    free(arr);
}

CBPyArray* cbpy_add(CBPyArray *a, CBPyArray *b) {
    if (ensure_initialized() != 0) return NULL;
    if (validate_binary_args(a, b) != 0) return NULL;

    CBPyArray *out = create_device_array(a->elem_count, a->dtype, a->ndim, a->shape);
    if (!out) return NULL;

    const void *kernel = NULL;
    if (a->dtype == CB_DTYPE_FLOAT32) kernel = (const void*)cbpy_kernel_add_f32;
    if (a->dtype == CB_DTYPE_FLOAT64) kernel = (const void*)cbpy_kernel_add_f64;
    if (a->dtype == CB_DTYPE_INT32) kernel = (const void*)cbpy_kernel_add_i32;

    cbError_t err = launch_elementwise_kernel(kernel, out->device_ptr, a->device_ptr, b->device_ptr, a->elem_count);
    if (err == cbSuccess) {
        return out;
    }

    cbpy_free(out);
    return NULL;
}

CBPyArray* cbpy_multiply(CBPyArray *a, CBPyArray *b) {
    if (ensure_initialized() != 0) return NULL;
    if (validate_binary_args(a, b) != 0) return NULL;

    CBPyArray *out = create_device_array(a->elem_count, a->dtype, a->ndim, a->shape);
    if (!out) return NULL;

    const void *kernel = NULL;
    if (a->dtype == CB_DTYPE_FLOAT32) kernel = (const void*)cbpy_kernel_mul_f32;
    if (a->dtype == CB_DTYPE_FLOAT64) kernel = (const void*)cbpy_kernel_mul_f64;
    if (a->dtype == CB_DTYPE_INT32) kernel = (const void*)cbpy_kernel_mul_i32;

    cbError_t err = launch_elementwise_kernel(kernel, out->device_ptr, a->device_ptr, b->device_ptr, a->elem_count);
    if (err == cbSuccess) {
        return out;
    }

    cbpy_free(out);
    return NULL;
}

CBPyArray* cbpy_matmul(CBPyArray *a, CBPyArray *b) {
    if (ensure_initialized() != 0) return NULL;
    if (!a || !b) return NULL;
    if (a->dtype != b->dtype) return NULL;
    if (a->ndim != 2 || b->ndim != 2) return NULL;
    if (a->shape[1] != b->shape[0]) return NULL;

    size_t result_shape[2] = {a->shape[0], b->shape[1]};
    size_t result_count = result_shape[0] * result_shape[1];
    CBPyArray *out = create_device_array(result_count, a->dtype, 2, result_shape);
    if (!out) return NULL;

    const void *kernel = NULL;
    if (a->dtype == CB_DTYPE_FLOAT32) kernel = (const void*)cbpy_kernel_matmul_f32;
    if (a->dtype == CB_DTYPE_FLOAT64) kernel = (const void*)cbpy_kernel_matmul_f64;

    cbError_t err = launch_matmul_kernel(kernel,
                                         out->device_ptr,
                                         a->device_ptr,
                                         b->device_ptr,
                                         a->shape[0],
                                         b->shape[1],
                                         a->shape[1]);
    if (err == cbSuccess) {
        return out;
    }

    cbpy_free(out);
    return NULL;
}

static int apply_scalar_op_double(double value, double scalar, int op, double *out) {
    if (!out) return -1;

    switch (op) {
        case CBPY_OP_ADD:
            *out = value + scalar;
            return 0;
        case CBPY_OP_SUB:
            *out = value - scalar;
            return 0;
        case CBPY_OP_MUL:
            *out = value * scalar;
            return 0;
        case CBPY_OP_DIV:
            if (scalar == 0.0) return -1;
            *out = value / scalar;
            return 0;
        default:
            return -1;
    }
}

CBPyArray* cbpy_scalar_op(CBPyArray *arr, double scalar, int op) {
    if (ensure_initialized() != 0) return NULL;
    if (!arr || !arr->device_ptr || arr->size == 0) return NULL;

    void *host_data = malloc(arr->size);
    if (!host_data) return NULL;

    cbError_t copy_err = cbMemcpy(host_data, arr->device_ptr, arr->size, CB_MEMCPY_DEVICE_TO_HOST);
    if (copy_err != cbSuccess) {
        free(host_data);
        return NULL;
    }

#define APPLY_SCALAR_CASE(dtype_const, c_type) \
    case dtype_const: { \
        c_type *p = (c_type *)host_data; \
        for (size_t i = 0; i < arr->elem_count; i++) { \
            double result = 0.0; \
            if (apply_scalar_op_double((double)p[i], scalar, op, &result) != 0) { \
                free(host_data); \
                return NULL; \
            } \
            p[i] = (c_type)result; \
        } \
        break; \
    }

    switch (arr->dtype) {
        APPLY_SCALAR_CASE(CB_DTYPE_FLOAT32, float)
        APPLY_SCALAR_CASE(CB_DTYPE_FLOAT64, double)
        APPLY_SCALAR_CASE(CB_DTYPE_INT32, int32_t)
        APPLY_SCALAR_CASE(CB_DTYPE_INT64, int64_t)
        APPLY_SCALAR_CASE(CB_DTYPE_UINT8, uint8_t)
        APPLY_SCALAR_CASE(CB_DTYPE_UINT32, uint32_t)
        APPLY_SCALAR_CASE(CB_DTYPE_INT8, int8_t)
        APPLY_SCALAR_CASE(CB_DTYPE_INT16, int16_t)
        APPLY_SCALAR_CASE(CB_DTYPE_BOOL, uint8_t)
        default:
            free(host_data);
            return NULL;
    }

#undef APPLY_SCALAR_CASE

    CBPyArray *out = create_device_array(arr->elem_count, arr->dtype, arr->ndim, arr->shape);
    if (!out) {
        free(host_data);
        return NULL;
    }

    copy_err = cbMemcpy(out->device_ptr, host_data, out->size, CB_MEMCPY_HOST_TO_DEVICE);
    free(host_data);
    if (copy_err == cbSuccess) {
        return out;
    }

    cbpy_free(out);
    return NULL;
}

double cbpy_reduce(CBPyArray *arr, int op) {
    if (ensure_initialized() != 0) return 0.0;
    if (!arr || !arr->device_ptr || arr->size == 0) return 0.0;

    void *host_data = malloc(arr->size);
    if (!host_data) return 0.0;

    cbError_t copy_err = cbMemcpy(host_data, arr->device_ptr, arr->size, CB_MEMCPY_DEVICE_TO_HOST);
    if (copy_err != cbSuccess) {
        free(host_data);
        return 0.0;
    }

    double result = 0.0;
    if (arr->dtype == CB_DTYPE_FLOAT32) {
        float *p = (float *)host_data;
        result = p[0];
        for (size_t i = 1; i < arr->elem_count; i++) {
            if (op == CBPY_REDUCE_SUM || op == CBPY_REDUCE_MEAN) result += p[i];
            if (op == CBPY_REDUCE_MAX && p[i] > result) result = p[i];
            if (op == CBPY_REDUCE_MIN && p[i] < result) result = p[i];
        }
    }

    if (op == CBPY_REDUCE_MEAN && arr->elem_count > 0) {
        result /= (double)arr->elem_count;
    }

    free(host_data);
    return result;
}

int cbpy_get_device_count(void) {
    if (ensure_initialized() != 0) {
        return -1;
    }

    int count = 0;
    cbError_t err = cbGetDeviceCount(&count);
    if (err == cbSuccess) {
        return count;
    }

    return -1;
}

const char* cbpy_device_name(void) {
    static char name[256] = "Unknown Device";
    if (ensure_initialized() != 0) {
        return name;
    }

    cbDeviceProp prop;
    memset(&prop, 0, sizeof(prop));

    cbError_t err = cbGetDeviceProperties(&prop, 0);
    if (err != cbSuccess) {
        return name;
    }

    snprintf(name, sizeof(name), "%s", prop.name);
    return name;
}

int cbpy_mem_info(size_t *free_bytes, size_t *total_bytes) {
    if (ensure_initialized() != 0) {
        return -1;
    }

    cbError_t err = cbMemGetInfo(free_bytes, total_bytes);
    if (err == cbSuccess) {
        return 0;
    }

    return -1;
}
