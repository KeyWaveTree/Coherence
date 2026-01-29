/**
 * CudaBridge - GPU Memory Manager Implementation
 */

#include "memory_manager.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/vm_map.h>
#endif

/* 로깅 */
#define MEM_LOG(fmt, ...) printf("[MEM] " fmt "\n", ##__VA_ARGS__)
#define MEM_ERR(fmt, ...) fprintf(stderr, "[MEM ERROR] " fmt "\n", ##__VA_ARGS__)

#ifdef DEBUG
#define MEM_DBG(fmt, ...) printf("[MEM DEBUG] " fmt "\n", ##__VA_ARGS__)
#else
#define MEM_DBG(fmt, ...)
#endif

/* 에러 코드 */
#define CB_SUCCESS              0
#define CB_ERR_INVALID_PARAM   -1
#define CB_ERR_NO_MEMORY       -2
#define CB_ERR_NOT_FOUND       -3

/* 정렬 매크로 */
#define ALIGN_UP(x, align) (((x) + (align) - 1) & ~((align) - 1))

/* 내부 함수 */
static int pool_init(CBMemoryPool *pool, uint64_t base, size_t size, size_t page_size);
static void pool_shutdown(CBMemoryPool *pool);
static CBMemoryBlock* pool_alloc(CBMemoryPool *pool, size_t size);
static void pool_free(CBMemoryPool *pool, CBMemoryBlock *block);
static CBMemoryBlock* create_block(uint64_t addr, size_t size);

/**
 * 메모리 관리자 초기화
 */
int cb_memory_manager_init(CBMemoryManager *mgr, int device_id,
                           size_t device_mem_size)
{
    if (!mgr) {
        return CB_ERR_INVALID_PARAM;
    }

    memset(mgr, 0, sizeof(CBMemoryManager));
    mgr->device_id = device_id;
    mgr->unified_memory_enabled = true;

    MEM_LOG("Initializing memory manager for device %d (%.2f GB)",
            device_id, (double)device_mem_size / (1024 * 1024 * 1024));

    /* 디바이스 메모리 풀 초기화 */
    /* GPU 가상 주소 공간은 0x200000000부터 시작 */
    int ret = pool_init(&mgr->device_pool,
                       0x200000000ULL + device_id * 0x100000000ULL,
                       device_mem_size,
                       CB_PAGE_SIZE_2M);
    if (ret != CB_SUCCESS) {
        MEM_ERR("Failed to initialize device memory pool");
        return ret;
    }

    /* 호스트 핀드 메모리 풀 (1GB) */
    ret = pool_init(&mgr->pinned_pool, 0, 1ULL * 1024 * 1024 * 1024,
                   CB_PAGE_SIZE_4K);
    if (ret != CB_SUCCESS) {
        pool_shutdown(&mgr->device_pool);
        return ret;
    }

    /* 통합 메모리 풀 (디바이스 메모리의 50%) */
    ret = pool_init(&mgr->managed_pool,
                   0x100000000ULL,
                   device_mem_size / 2,
                   CB_PAGE_SIZE_64K);
    if (ret != CB_SUCCESS) {
        pool_shutdown(&mgr->device_pool);
        pool_shutdown(&mgr->pinned_pool);
        return ret;
    }

    /* 페이지 테이블 할당 */
    size_t num_pages = device_mem_size / CB_PAGE_SIZE_4K;
    mgr->page_table_size = num_pages;
    mgr->page_table = calloc(num_pages, sizeof(uint64_t));
    if (!mgr->page_table) {
        pool_shutdown(&mgr->device_pool);
        pool_shutdown(&mgr->pinned_pool);
        pool_shutdown(&mgr->managed_pool);
        return CB_ERR_NO_MEMORY;
    }

    MEM_LOG("Memory manager initialized");
    MEM_LOG("  Device pool: base=0x%" PRIX64 " size=%.2f GB",
            mgr->device_pool.base_addr,
            (double)mgr->device_pool.total_size / (1024 * 1024 * 1024));

    return CB_SUCCESS;
}

/**
 * 메모리 관리자 종료
 */
void cb_memory_manager_shutdown(CBMemoryManager *mgr)
{
    if (!mgr) return;

    MEM_LOG("Shutting down memory manager...");

    cb_memory_manager_stats(mgr);

    pool_shutdown(&mgr->device_pool);
    pool_shutdown(&mgr->pinned_pool);
    pool_shutdown(&mgr->managed_pool);

    /* 보안: 페이지 테이블 민감 데이터 제거 후 해제 */
    if (mgr->page_table) {
        volatile uint8_t *pt = (volatile uint8_t *)mgr->page_table;
        for (size_t i = 0; i < mgr->page_table_size * sizeof(uint64_t); i++) {
            pt[i] = 0;
        }
        free(mgr->page_table);
    }

    /* 보안: 구조체 민감 데이터 확실히 제거 */
    volatile uint8_t *p = (volatile uint8_t *)mgr;
    for (size_t i = 0; i < sizeof(CBMemoryManager); i++) {
        p[i] = 0;
    }
}

/**
 * 메모리 풀 초기화
 */
static int pool_init(CBMemoryPool *pool, uint64_t base, size_t size,
                     size_t page_size)
{
    memset(pool, 0, sizeof(CBMemoryPool));

    pool->base_addr = base;
    pool->total_size = size;
    pool->page_size = page_size;
    pool->used_size = 0;

    /* 초기 프리 블록 생성 */
    CBMemoryBlock *block = create_block(base, size);
    if (!block) {
        return CB_ERR_NO_MEMORY;
    }

    pool->free_list = block;
    pool->block_count = 1;

    return CB_SUCCESS;
}

/**
 * 메모리 풀 종료
 */
static void pool_shutdown(CBMemoryPool *pool)
{
    /* 프리 리스트 정리 */
    CBMemoryBlock *block = pool->free_list;
    while (block) {
        CBMemoryBlock *next = block->next;
        free(block);
        block = next;
    }

    /* 사용 중인 리스트 정리 (누수 경고) */
    block = pool->used_list;
    while (block) {
        CBMemoryBlock *next = block->next;
        if (block->is_allocated) {
            MEM_ERR("Memory leak: %zu bytes at 0x%" PRIX64,
                    block->size, block->gpu_addr);
        }
        free(block);
        block = next;
    }

    memset(pool, 0, sizeof(CBMemoryPool));
}

/**
 * 블록 생성
 */
static CBMemoryBlock* create_block(uint64_t addr, size_t size)
{
    CBMemoryBlock *block = calloc(1, sizeof(CBMemoryBlock));
    if (!block) return NULL;

    block->gpu_addr = addr;
    block->size = size;
    block->is_allocated = false;

    return block;
}

/**
 * 풀에서 메모리 할당 (First Fit)
 */
static CBMemoryBlock* pool_alloc(CBMemoryPool *pool, size_t size)
{
    /* 페이지 크기로 정렬 (오버플로우 검증) */
    size_t aligned = ALIGN_UP(size, pool->page_size);
    if (aligned < size) {
        return NULL;  /* 오버플로우 */
    }
    size = aligned;

    /* First Fit 검색 */
    CBMemoryBlock *prev = NULL;
    CBMemoryBlock *block = pool->free_list;

    while (block) {
        if (block->size >= size) {
            /* 블록 분할 */
            if (block->size > size + pool->page_size) {
                /* 남은 부분으로 새 블록 생성 */
                CBMemoryBlock *remaining = create_block(
                    block->gpu_addr + size,
                    block->size - size
                );

                if (remaining) {
                    remaining->next = block->next;
                    if (block->next) {
                        block->next->prev = remaining;
                    }
                    block->next = remaining;
                    remaining->prev = block;
                    block->size = size;
                    pool->block_count++;
                }
            }

            /* 프리 리스트에서 제거 */
            if (prev) {
                prev->next = block->next;
            } else {
                pool->free_list = block->next;
            }
            if (block->next) {
                block->next->prev = prev;
            }

            /* 사용 중인 리스트에 추가 */
            block->next = pool->used_list;
            block->prev = NULL;
            if (pool->used_list) {
                pool->used_list->prev = block;
            }
            pool->used_list = block;

            block->is_allocated = true;
            pool->used_size += size;

            return block;
        }

        prev = block;
        block = block->next;
    }

    return NULL;  /* 할당 실패 */
}

/**
 * 풀에 메모리 반환
 */
static void pool_free(CBMemoryPool *pool, CBMemoryBlock *block)
{
    if (!block || !block->is_allocated) return;

    /* 사용 중인 리스트에서 제거 */
    if (block->prev) {
        block->prev->next = block->next;
    } else {
        pool->used_list = block->next;
    }
    if (block->next) {
        block->next->prev = block->prev;
    }

    block->is_allocated = false;
    pool->used_size -= block->size;

    /* 프리 리스트에 주소 순서대로 삽입 */
    CBMemoryBlock *prev = NULL;
    CBMemoryBlock *curr = pool->free_list;

    while (curr && curr->gpu_addr < block->gpu_addr) {
        prev = curr;
        curr = curr->next;
    }

    /* 삽입 */
    block->next = curr;
    block->prev = prev;

    if (prev) {
        prev->next = block;
    } else {
        pool->free_list = block;
    }
    if (curr) {
        curr->prev = block;
    }

    /* 인접 블록과 병합 시도 */
    /* 다음 블록과 병합 */
    if (block->next && !block->next->is_allocated &&
        block->gpu_addr + block->size == block->next->gpu_addr) {
        CBMemoryBlock *next = block->next;
        block->size += next->size;
        block->next = next->next;
        if (next->next) {
            next->next->prev = block;
        }
        free(next);
        pool->block_count--;
    }

    /* 이전 블록과 병합 */
    if (block->prev && !block->prev->is_allocated &&
        block->prev->gpu_addr + block->prev->size == block->gpu_addr) {
        CBMemoryBlock *prev_block = block->prev;
        prev_block->size += block->size;
        prev_block->next = block->next;
        if (block->next) {
            block->next->prev = prev_block;
        }
        free(block);
        pool->block_count--;
    }
}

/**
 * 디바이스 메모리 할당
 */
int cb_malloc_device(CBMemoryManager *mgr, void **ptr, size_t size)
{
    if (!mgr || !ptr || size == 0) {
        return CB_ERR_INVALID_PARAM;
    }

    CBMemoryBlock *block = pool_alloc(&mgr->device_pool, size);
    if (!block) {
        MEM_ERR("Failed to allocate %zu bytes of device memory", size);
        return CB_ERR_NO_MEMORY;
    }

    block->flags = CB_MEM_FLAG_DEVICE;
    block->device_id = mgr->device_id;

    *ptr = (void*)(uintptr_t)block->gpu_addr;

    mgr->total_allocated += block->size;
    if (mgr->total_allocated > mgr->peak_allocated) {
        mgr->peak_allocated = mgr->total_allocated;
    }
    mgr->alloc_count++;

    MEM_DBG("Allocated %zu bytes at 0x%" PRIX64 " (device)", size, block->gpu_addr);

    return CB_SUCCESS;
}

/**
 * 디바이스 메모리 해제
 */
int cb_free_device(CBMemoryManager *mgr, void *ptr)
{
    if (!mgr || !ptr) {
        return CB_ERR_INVALID_PARAM;
    }

    CBMemoryBlock *block = cb_find_block(mgr, ptr);
    if (!block) {
        MEM_ERR("Invalid pointer: %p", ptr);
        return CB_ERR_NOT_FOUND;
    }

    mgr->total_allocated -= block->size;
    mgr->free_count++;

    MEM_DBG("Freed %zu bytes at 0x%" PRIX64 " (device)", block->size, block->gpu_addr);

    pool_free(&mgr->device_pool, block);

    return CB_SUCCESS;
}

/**
 * 호스트 핀드 메모리 할당
 */
int cb_malloc_host(CBMemoryManager *mgr, void **ptr, size_t size, uint32_t flags)
{
    if (!mgr || !ptr || size == 0) {
        return CB_ERR_INVALID_PARAM;
    }

    /* 시스템 메모리 할당 */
    void *host_ptr = NULL;

#ifdef __APPLE__
    /* macOS에서 페이지 락 메모리 할당 */
    mach_vm_address_t addr = 0;
    kern_return_t kr = mach_vm_allocate(mach_task_self(),
                                        &addr,
                                        ALIGN_UP(size, CB_PAGE_SIZE_4K),
                                        VM_FLAGS_ANYWHERE);
    if (kr != KERN_SUCCESS) {
        return CB_ERR_NO_MEMORY;
    }

    /* 페이지 락 (wire) */
    kr = mach_vm_wire(mach_host_self(), mach_task_self(),
                      addr, size, VM_PROT_READ | VM_PROT_WRITE);
    if (kr != KERN_SUCCESS) {
        /* 락 실패해도 계속 진행 (경고만) */
        MEM_DBG("Warning: Failed to wire host memory");
    }

    host_ptr = (void*)addr;
#else
    int ret = posix_memalign(&host_ptr, CB_PAGE_SIZE_4K, size);
    if (ret != 0 || !host_ptr) {
        return CB_ERR_NO_MEMORY;
    }
#endif

    /* 관리 구조체 할당 */
    CBMemoryBlock *block = pool_alloc(&mgr->pinned_pool, size);
    if (!block) {
        /* 풀에 공간이 없으면 직접 관리 */
        block = create_block((uint64_t)(uintptr_t)host_ptr, size);
        if (!block) {
#ifdef __APPLE__
            mach_vm_deallocate(mach_task_self(), (mach_vm_address_t)host_ptr, size);
#else
            free(host_ptr);
#endif
            return CB_ERR_NO_MEMORY;
        }
        block->is_allocated = true;
    }

    block->host_addr = host_ptr;
    block->flags = CB_MEM_FLAG_HOST | CB_MEM_FLAG_PINNED | flags;
    block->device_id = mgr->device_id;

    *ptr = host_ptr;

    mgr->alloc_count++;

    MEM_DBG("Allocated %zu bytes at %p (pinned host)", size, host_ptr);

    return CB_SUCCESS;
}

/**
 * 호스트 메모리 해제
 */
int cb_free_host(CBMemoryManager *mgr, void *ptr)
{
    if (!mgr || !ptr) {
        return CB_ERR_INVALID_PARAM;
    }

    CBMemoryBlock *block = cb_find_block(mgr, ptr);
    size_t size = block ? block->size : 0;

#ifdef __APPLE__
    if (size > 0) {
        mach_vm_deallocate(mach_task_self(), (mach_vm_address_t)ptr, size);
    }
#else
    free(ptr);
#endif

    if (block) {
        mgr->free_count++;
        pool_free(&mgr->pinned_pool, block);
    }

    MEM_DBG("Freed host memory at %p", ptr);

    return CB_SUCCESS;
}

/**
 * 통합 메모리 할당
 */
int cb_malloc_managed(CBMemoryManager *mgr, void **ptr, size_t size,
                      uint32_t flags)
{
    if (!mgr || !ptr || size == 0) {
        return CB_ERR_INVALID_PARAM;
    }

    if (!mgr->unified_memory_enabled) {
        MEM_ERR("Unified memory is not enabled");
        return CB_ERR_INVALID_PARAM;
    }

    CBMemoryBlock *block = pool_alloc(&mgr->managed_pool, size);
    if (!block) {
        return CB_ERR_NO_MEMORY;
    }

    /* 호스트 메모리도 할당 (통합 주소 공간) */
    void *host_ptr = NULL;
    int ret = posix_memalign(&host_ptr, CB_PAGE_SIZE_4K, size);
    if (ret != 0 || !host_ptr) {
        pool_free(&mgr->managed_pool, block);
        return CB_ERR_NO_MEMORY;
    }

    block->host_addr = host_ptr;
    block->flags = CB_MEM_FLAG_MANAGED | flags;
    block->is_managed = true;
    block->device_id = mgr->device_id;
    block->preferred_device = mgr->device_id;

    /* 통합 메모리는 호스트 주소를 반환 */
    *ptr = host_ptr;

    mgr->total_allocated += block->size;
    if (mgr->total_allocated > mgr->peak_allocated) {
        mgr->peak_allocated = mgr->total_allocated;
    }
    mgr->alloc_count++;

    MEM_DBG("Allocated %zu bytes managed memory (host=%p, gpu=0x%" PRIX64 ")",
            size, host_ptr, block->gpu_addr);

    return CB_SUCCESS;
}

/**
 * 통합 메모리 해제
 */
int cb_free_managed(CBMemoryManager *mgr, void *ptr)
{
    if (!mgr || !ptr) {
        return CB_ERR_INVALID_PARAM;
    }

    CBMemoryBlock *block = cb_find_block(mgr, ptr);
    if (!block) {
        return CB_ERR_NOT_FOUND;
    }

    if (block->host_addr) {
        /* 보안: 통합 메모리 데이터 삭제 후 해제 (데이터 유출 방지) */
        volatile uint8_t *bp = (volatile uint8_t *)block->host_addr;
        for (size_t b = 0; b < block->size; b++) {
            bp[b] = 0;
        }
        free(block->host_addr);
    }

    mgr->total_allocated -= block->size;
    mgr->free_count++;

    pool_free(&mgr->managed_pool, block);

    MEM_DBG("Freed managed memory at %p", ptr);

    return CB_SUCCESS;
}

/**
 * 메모리 조언 설정
 */
int cb_mem_advise(CBMemoryManager *mgr, void *ptr, size_t size,
                  CBMemoryAdvise advice, int device)
{
    if (!mgr || !ptr) {
        return CB_ERR_INVALID_PARAM;
    }

    CBMemoryBlock *block = cb_find_block(mgr, ptr);
    if (!block || !block->is_managed) {
        return CB_ERR_NOT_FOUND;
    }

    switch (advice) {
        case CB_MEM_ADVISE_SET_PREFERRED_LOCATION:
            block->preferred_device = device;
            MEM_DBG("Set preferred location to device %d for %p", device, ptr);
            break;

        case CB_MEM_ADVISE_UNSET_PREFERRED_LOCATION:
            block->preferred_device = -1;
            break;

        default:
            MEM_DBG("Memory advise %d for %p", advice, ptr);
    }

    (void)size;  /* 부분 범위는 현재 미지원 */

    return CB_SUCCESS;
}

/**
 * 메모리 프리페치
 */
int cb_mem_prefetch(CBMemoryManager *mgr, void *ptr, size_t size,
                    int dst_device)
{
    if (!mgr || !ptr) {
        return CB_ERR_INVALID_PARAM;
    }

    CBMemoryBlock *block = cb_find_block(mgr, ptr);
    if (!block || !block->is_managed) {
        return CB_ERR_NOT_FOUND;
    }

    MEM_DBG("Prefetch %zu bytes from %p to device %d", size, ptr, dst_device);

    /* 실제로는 비동기 DMA 전송 시작 */

    return CB_SUCCESS;
}

/**
 * 포인터 속성 쿼리
 */
int cb_pointer_get_attributes(CBMemoryManager *mgr, void *ptr,
                              uint32_t *flags, int *device,
                              void **host_ptr, uint64_t *gpu_addr)
{
    if (!mgr || !ptr) {
        return CB_ERR_INVALID_PARAM;
    }

    CBMemoryBlock *block = cb_find_block(mgr, ptr);
    if (!block) {
        return CB_ERR_NOT_FOUND;
    }

    if (flags) *flags = block->flags;
    if (device) *device = block->device_id;
    if (host_ptr) *host_ptr = block->host_addr;
    if (gpu_addr) *gpu_addr = block->gpu_addr;

    return CB_SUCCESS;
}

/**
 * 사용 가능한 메모리 쿼리
 */
int cb_mem_get_info(CBMemoryManager *mgr, size_t *free, size_t *total)
{
    if (!mgr) {
        return CB_ERR_INVALID_PARAM;
    }

    if (total) {
        *total = mgr->device_pool.total_size;
    }
    if (free) {
        *free = mgr->device_pool.total_size - mgr->device_pool.used_size;
    }

    return CB_SUCCESS;
}

/**
 * 메모리 블록 찾기
 */
CBMemoryBlock* cb_find_block(CBMemoryManager *mgr, void *ptr)
{
    if (!mgr || !ptr) return NULL;

    uint64_t addr = (uint64_t)(uintptr_t)ptr;

    /* 디바이스 풀 검색 */
    CBMemoryBlock *block = mgr->device_pool.used_list;
    while (block) {
        if (block->gpu_addr == addr) {
            return block;
        }
        block = block->next;
    }

    /* 핀드 풀 검색 */
    block = mgr->pinned_pool.used_list;
    while (block) {
        if (block->host_addr == ptr) {
            return block;
        }
        block = block->next;
    }

    /* 통합 메모리 풀 검색 */
    block = mgr->managed_pool.used_list;
    while (block) {
        if (block->host_addr == ptr || block->gpu_addr == addr) {
            return block;
        }
        block = block->next;
    }

    return NULL;
}

/**
 * 메모리 풀 상태 출력
 */
void cb_memory_pool_dump(CBMemoryPool *pool, const char *name)
{
    printf("\n=== Memory Pool: %s ===\n", name);
    printf("Base: 0x%" PRIX64 ", Total: %.2f MB, Used: %.2f MB (%.1f%%)\n",
           pool->base_addr,
           (double)pool->total_size / (1024 * 1024),
           (double)pool->used_size / (1024 * 1024),
           pool->total_size > 0 ? 100.0 * pool->used_size / pool->total_size : 0.0);
    printf("Block count: %d\n", pool->block_count);

    printf("\nFree blocks:\n");
    CBMemoryBlock *block = pool->free_list;
    int count = 0;
    while (block && count < 10) {
        printf("  [%d] 0x%" PRIX64 " - 0x%" PRIX64 " (%.2f MB)\n",
               count, block->gpu_addr, block->gpu_addr + block->size,
               (double)block->size / (1024 * 1024));
        block = block->next;
        count++;
    }
    if (block) printf("  ... (more blocks)\n");
}

/**
 * 메모리 관리자 통계 출력
 */
void cb_memory_manager_stats(CBMemoryManager *mgr)
{
    printf("\n=== Memory Manager Statistics ===\n");
    printf("Device: %d\n", mgr->device_id);
    printf("Total allocated: %.2f MB\n",
           (double)mgr->total_allocated / (1024 * 1024));
    printf("Peak allocated: %.2f MB\n",
           (double)mgr->peak_allocated / (1024 * 1024));
    printf("Allocation count: %" PRIu64 "\n", mgr->alloc_count);
    printf("Free count: %" PRIu64 "\n", mgr->free_count);
    printf("Unified memory: %s\n",
           mgr->unified_memory_enabled ? "enabled" : "disabled");
}
