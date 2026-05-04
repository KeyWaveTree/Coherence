/**
 * Coherence - GPU Memory Manager
 *
 * GPU 메모리 할당, 가상 주소 공간 관리, 통합 메모리를 담당합니다.
 */

#ifndef COHERENCE_MEMORY_MANAGER_H
#define COHERENCE_MEMORY_MANAGER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/* 메모리 할당 플래그 */
#define CB_MEM_FLAG_DEVICE      (1 << 0)  /* 디바이스 메모리 */
#define CB_MEM_FLAG_HOST        (1 << 1)  /* 호스트 메모리 */
#define CB_MEM_FLAG_MANAGED     (1 << 2)  /* 통합 메모리 */
#define CB_MEM_FLAG_PINNED      (1 << 3)  /* 페이지 락 */
#define CB_MEM_FLAG_MAPPED      (1 << 4)  /* 호스트 매핑 */
#define CB_MEM_FLAG_WRITE_COMBINED (1 << 5)
#define CB_MEM_FLAG_PORTABLE    (1 << 6)  /* 멀티 디바이스 */

/* 메모리 접근 힌트 */
typedef enum {
    CB_MEM_ADVISE_NONE = 0,
    CB_MEM_ADVISE_SET_READ_MOSTLY,
    CB_MEM_ADVISE_UNSET_READ_MOSTLY,
    CB_MEM_ADVISE_SET_PREFERRED_LOCATION,
    CB_MEM_ADVISE_UNSET_PREFERRED_LOCATION,
    CB_MEM_ADVISE_SET_ACCESSED_BY,
    CB_MEM_ADVISE_UNSET_ACCESSED_BY
} CBMemoryAdvise;

/* 페이지 크기 */
#define CB_PAGE_SIZE_4K         (4 * 1024)
#define CB_PAGE_SIZE_64K        (64 * 1024)
#define CB_PAGE_SIZE_2M         (2 * 1024 * 1024)

/* 메모리 블록 */
typedef struct CBMemoryBlock {
    uint64_t            gpu_addr;       /* GPU 가상 주소 */
    void               *host_addr;      /* 호스트 가상 주소 */
    size_t              size;           /* 크기 */
    uint32_t            flags;          /* 할당 플래그 */
    int                 device_id;      /* 소유 디바이스 */
    bool                is_allocated;   /* 할당 상태 */

    /* 통합 메모리 */
    bool                is_managed;     /* 통합 메모리 여부 */
    int                 preferred_device; /* 선호 디바이스 */

    /* 내부 관리 */
    struct CBMemoryBlock *next;
    struct CBMemoryBlock *prev;
} CBMemoryBlock;

/* 메모리 풀 */
typedef struct CBMemoryPool {
    uint64_t            base_addr;      /* 베이스 주소 */
    size_t              total_size;     /* 전체 크기 */
    size_t              used_size;      /* 사용 중인 크기 */
    size_t              page_size;      /* 페이지 크기 */

    CBMemoryBlock      *free_list;      /* 프리 리스트 */
    CBMemoryBlock      *used_list;      /* 사용 중인 블록 리스트 */

    int                 block_count;    /* 총 블록 수 */
} CBMemoryPool;

/* GPU 메모리 관리자 */
typedef struct CBMemoryManager {
    /* 디바이스 메모리 풀 */
    CBMemoryPool        device_pool;

    /* 호스트 핀드 메모리 풀 */
    CBMemoryPool        pinned_pool;

    /* 통합 메모리 풀 */
    CBMemoryPool        managed_pool;

    /* 페이지 테이블 (GPU 가상 주소 → 물리 주소 매핑) */
    uint64_t           *page_table;
    size_t              page_table_size;

    /* 통계 */
    size_t              total_allocated;
    size_t              peak_allocated;
    uint64_t            alloc_count;
    uint64_t            free_count;

    /* 설정 */
    int                 device_id;
    bool                unified_memory_enabled;
} CBMemoryManager;

/* 함수 선언 */

/**
 * 메모리 관리자 초기화
 */
int cb_memory_manager_init(CBMemoryManager *mgr, int device_id,
                           size_t device_mem_size);

/**
 * 메모리 관리자 종료
 */
void cb_memory_manager_shutdown(CBMemoryManager *mgr);

/**
 * 디바이스 메모리 할당
 */
int cb_malloc_device(CBMemoryManager *mgr, void **ptr, size_t size);

/**
 * 디바이스 메모리 해제
 */
int cb_free_device(CBMemoryManager *mgr, void *ptr);

/**
 * 호스트 핀드 메모리 할당
 */
int cb_malloc_host(CBMemoryManager *mgr, void **ptr, size_t size,
                   uint32_t flags);

/**
 * 호스트 메모리 해제
 */
int cb_free_host(CBMemoryManager *mgr, void *ptr);

/**
 * 통합 메모리 할당
 */
int cb_malloc_managed(CBMemoryManager *mgr, void **ptr, size_t size,
                      uint32_t flags);

/**
 * 통합 메모리 해제
 */
int cb_free_managed(CBMemoryManager *mgr, void *ptr);

/**
 * 메모리 조언 설정
 */
int cb_mem_advise(CBMemoryManager *mgr, void *ptr, size_t size,
                  CBMemoryAdvise advice, int device);

/**
 * 메모리 프리페치
 */
int cb_mem_prefetch(CBMemoryManager *mgr, void *ptr, size_t size,
                    int dst_device);

/**
 * 포인터 속성 쿼리
 */
int cb_pointer_get_attributes(CBMemoryManager *mgr, void *ptr,
                              uint32_t *flags, int *device,
                              void **host_ptr, uint64_t *gpu_addr);

/**
 * 사용 가능한 메모리 쿼리
 */
int cb_mem_get_info(CBMemoryManager *mgr, size_t *free, size_t *total);

/**
 * 메모리 블록 찾기
 */
CBMemoryBlock* cb_find_block(CBMemoryManager *mgr, void *ptr);

/**
 * 메모리 풀 상태 출력 (디버그)
 */
void cb_memory_pool_dump(CBMemoryPool *pool, const char *name);

/**
 * 메모리 관리자 통계 출력
 */
void cb_memory_manager_stats(CBMemoryManager *mgr);

#endif /* COHERENCE_MEMORY_MANAGER_H */
