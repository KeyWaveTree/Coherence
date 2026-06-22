/**
 * CudaBridge - Python C Extension API Header
 *
 * Python에서 CUDA 연산을 기존 코드와 유사한 방식으로 사용할 수 있게
 * 해주는 C 확장 모듈의 헤더입니다.
 *
 * Python 사용 예시:
 *   import cudabridge as cb
 *   cb.init()
 *   gpu_array = cb.to_device(numpy_array)
 *   result = cb.from_device(gpu_array)
 */

#ifndef CUDABRIDGE_PYTHON_H
#define CUDABRIDGE_PYTHON_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Python 바인딩용 간소화된 GPU 배열 핸들 */
typedef struct CBPyArray {
    void       *device_ptr;     /* GPU 메모리 포인터 */
    void       *host_ptr;       /* 호스트 메모리 캐시 (선택적) */
    size_t      size;           /* 바이트 크기 */
    size_t      elem_count;     /* 요소 수 */
    int         dtype;          /* 데이터 타입 코드 */
    int         ndim;           /* 차원 수 */
    size_t      shape[8];       /* 각 차원 크기 (최대 8차원) */
    int         is_synced;      /* 호스트/디바이스 동기화 상태 */
    uint32_t    crc32;          /* 데이터 무결성 CRC */
} CBPyArray;

/* 데이터 타입 코드 (numpy 호환) */
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

/**
 * Python 바인딩 초기화
 */
int cbpy_init(void);

/**
 * Python 바인딩 종료
 */
void cbpy_shutdown(void);

/**
 * 호스트 데이터를 GPU로 전송
 */
CBPyArray* cbpy_to_device(const void *host_data, size_t elem_count,
                           int dtype, int ndim, const size_t *shape);

/**
 * GPU 데이터를 호스트로 복사
 */
int cbpy_from_device(CBPyArray *arr, void *host_data);

/**
 * GPU 배열 해제
 */
void cbpy_free(CBPyArray *arr);

/**
 * GPU에서 요소별 덧셈 실행
 */
CBPyArray* cbpy_add(CBPyArray *a, CBPyArray *b);

/**
 * GPU에서 요소별 곱셈 실행
 */
CBPyArray* cbpy_multiply(CBPyArray *a, CBPyArray *b);

/**
 * GPU에서 행렬 곱 실행
 */
CBPyArray* cbpy_matmul(CBPyArray *a, CBPyArray *b);

/**
 * GPU에서 스칼라 연산 실행
 */
CBPyArray* cbpy_scalar_op(CBPyArray *arr, double scalar, int op);

/**
 * GPU에서 리덕션 (합계, 평균, 최대, 최소)
 */
double cbpy_reduce(CBPyArray *arr, int op);

/**
 * 데이터 타입 크기 반환
 */
size_t cbpy_dtype_size(int dtype);

/**
 * 디바이스 정보 문자열 반환
 */
const char* cbpy_device_name(void);

/**
 * 사용 가능한 GPU 메모리 조회
 */
int cbpy_mem_info(size_t *free_bytes, size_t *total_bytes);

/* 스칼라 연산 코드 */
#define CBPY_OP_ADD   0
#define CBPY_OP_SUB   1
#define CBPY_OP_MUL   2
#define CBPY_OP_DIV   3

/* 리덕션 연산 코드 */
#define CBPY_REDUCE_SUM  0
#define CBPY_REDUCE_MEAN 1
#define CBPY_REDUCE_MAX  2
#define CBPY_REDUCE_MIN  3

#ifdef __cplusplus
}
#endif

#endif /* CUDABRIDGE_PYTHON_H */
