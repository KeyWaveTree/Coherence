#ifndef CUDABRIDGE_PYTHON_H
#define CUDABRIDGE_PYTHON_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CBPyArray {
    void       *device_ptr;
    size_t      size;
    size_t      elem_count;
    int         dtype;
    int         ndim;
    size_t      shape[8];
    uint32_t    crc32;
} CBPyArray;

typedef enum {
    CB_DTYPE_FLOAT32 = 0,
    CB_DTYPE_FLOAT64 = 1,
    CB_DTYPE_INT32   = 2,
    CB_DTYPE_INT64   = 3,
    CB_DTYPE_UINT8   = 4,
    CB_DTYPE_UINT32  = 5,
    CB_DTYPE_INT8    = 6,
    CB_DTYPE_INT16   = 7,
    CB_DTYPE_FLOAT16 = 8,
    CB_DTYPE_BOOL    = 9
} CBDtype;

int cbpy_init(void);
void cbpy_shutdown(void);

CBPyArray* cbpy_to_device(const void *host_data, size_t elem_count,
                           int dtype, int ndim, const size_t *shape);
int cbpy_from_device(CBPyArray *arr, void *host_data);
void cbpy_free(CBPyArray *arr);

CBPyArray* cbpy_add(CBPyArray *a, CBPyArray *b);
CBPyArray* cbpy_multiply(CBPyArray *a, CBPyArray *b);
CBPyArray* cbpy_matmul(CBPyArray *a, CBPyArray *b);
CBPyArray* cbpy_scalar_op(CBPyArray *arr, double scalar, int op);
double cbpy_reduce(CBPyArray *arr, int op);

size_t cbpy_dtype_size(int dtype);
int cbpy_get_device_count(void);
const char* cbpy_device_name(void);
int cbpy_mem_info(size_t *free_bytes, size_t *total_bytes);

#define CBPY_OP_ADD   0
#define CBPY_OP_SUB   1
#define CBPY_OP_MUL   2
#define CBPY_OP_DIV   3

#define CBPY_REDUCE_SUM  0
#define CBPY_REDUCE_MEAN 1
#define CBPY_REDUCE_MAX  2
#define CBPY_REDUCE_MIN  3

#ifdef __cplusplus
}
#endif

#endif
