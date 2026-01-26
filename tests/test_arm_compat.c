/**
 * CudaBridge - ARM Compatibility Test
 *
 * ARM64 아키텍처(Apple Silicon 등)에서의 호환성 테스트
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <inttypes.h>

#define TEST_PASS(name) printf("[PASS] %s\n", name)
#define TEST_FAIL(name, msg) do { \
    printf("[FAIL] %s: %s\n", name, msg); \
    failures++; \
} while(0)

static int failures = 0;

/**
 * 아키텍처 감지 테스트
 */
void test_architecture(void)
{
    printf("  Architecture: ");

#if defined(__aarch64__) || defined(__arm64__)
    printf("ARM64 (aarch64)\n");
    TEST_PASS("architecture detection (ARM64)");
#elif defined(__x86_64__) || defined(_M_X64)
    printf("x86_64\n");
    printf("  [WARN] This project is optimized for ARM64/Apple Silicon\n");
    TEST_PASS("architecture detection (x86_64)");
#elif defined(__i386__) || defined(_M_IX86)
    printf("x86 (32-bit)\n");
    TEST_FAIL("architecture", "32-bit architecture not supported");
#else
    printf("Unknown\n");
    TEST_FAIL("architecture", "Unknown architecture");
#endif
}

/**
 * 포인터 크기 테스트
 */
void test_pointer_size(void)
{
    size_t ptr_size = sizeof(void*);
    printf("  Pointer size: %zu bytes\n", ptr_size);

    if (ptr_size != 8) {
        TEST_FAIL("pointer size", "64-bit pointers required");
        return;
    }

    TEST_PASS("pointer size (64-bit)");
}

/**
 * 고정 크기 정수형 테스트
 */
void test_fixed_width_types(void)
{
    bool all_ok = true;

    printf("  Type sizes:\n");
    printf("    uint8_t:  %zu bytes\n", sizeof(uint8_t));
    printf("    uint16_t: %zu bytes\n", sizeof(uint16_t));
    printf("    uint32_t: %zu bytes\n", sizeof(uint32_t));
    printf("    uint64_t: %zu bytes\n", sizeof(uint64_t));
    printf("    size_t:   %zu bytes\n", sizeof(size_t));
    printf("    uintptr_t: %zu bytes\n", sizeof(uintptr_t));

    if (sizeof(uint8_t) != 1) all_ok = false;
    if (sizeof(uint16_t) != 2) all_ok = false;
    if (sizeof(uint32_t) != 4) all_ok = false;
    if (sizeof(uint64_t) != 8) all_ok = false;

    if (!all_ok) {
        TEST_FAIL("fixed width types", "unexpected type sizes");
        return;
    }

    TEST_PASS("fixed width integer types");
}

/**
 * 엔디안 테스트
 */
void test_endianness(void)
{
    union {
        uint32_t u32;
        uint8_t bytes[4];
    } test;

    test.u32 = 0x01020304;

    printf("  Byte order: ");
    if (test.bytes[0] == 0x04) {
        printf("Little-endian\n");
        TEST_PASS("endianness (little-endian)");
    } else if (test.bytes[0] == 0x01) {
        printf("Big-endian\n");
        printf("  [WARN] ARM64 big-endian mode is unusual\n");
        TEST_PASS("endianness (big-endian)");
    } else {
        printf("Unknown\n");
        TEST_FAIL("endianness", "cannot determine byte order");
    }
}

/**
 * 메모리 정렬 테스트
 */
void test_alignment(void)
{
    /* 정렬 테스트용 구조체 */
    struct AlignTest {
        uint8_t a;
        uint64_t b;
        uint8_t c;
        uint32_t d;
    };

    printf("  Struct AlignTest size: %zu bytes\n", sizeof(struct AlignTest));
    printf("  uint64_t alignment: %zu bytes\n", _Alignof(uint64_t));

    /* 동적 할당된 메모리의 정렬 확인 */
    void *ptr = malloc(64);
    if (ptr) {
        uintptr_t addr = (uintptr_t)ptr;
        printf("  malloc() alignment: %zu bytes\n",
               addr - (addr & ~(uintptr_t)0xF));

        /* 16바이트 정렬 확인 (ARM64에서 일반적) */
        if ((addr & 0xF) != 0) {
            printf("  [WARN] malloc not 16-byte aligned\n");
        }
        free(ptr);
    }

    TEST_PASS("memory alignment");
}

/**
 * printf 형식 지정자 테스트 (ARM64 호환성)
 */
void test_printf_formats(void)
{
    uint64_t test_u64 = 0x123456789ABCDEF0ULL;
    int64_t test_i64 = -1234567890123456789LL;
    size_t test_size = 1024 * 1024 * 1024;

    char buffer[256];

    /* PRIu64 테스트 */
    snprintf(buffer, sizeof(buffer), "%" PRIu64, test_u64);
    if (strlen(buffer) == 0) {
        TEST_FAIL("printf PRIu64", "empty output");
        return;
    }
    printf("  PRIu64 output: %s\n", buffer);

    /* PRIx64 테스트 */
    snprintf(buffer, sizeof(buffer), "%" PRIx64, test_u64);
    printf("  PRIx64 output: %s\n", buffer);

    /* PRIX64 테스트 */
    snprintf(buffer, sizeof(buffer), "%" PRIX64, test_u64);
    printf("  PRIX64 output: %s\n", buffer);

    /* PRId64 테스트 */
    snprintf(buffer, sizeof(buffer), "%" PRId64, test_i64);
    printf("  PRId64 output: %s\n", buffer);

    /* size_t 테스트 */
    snprintf(buffer, sizeof(buffer), "%zu", test_size);
    printf("  %%zu output: %s\n", buffer);

    TEST_PASS("printf format specifiers");
}

/**
 * 포인터-정수 변환 테스트
 */
void test_pointer_int_conversion(void)
{
    void *original_ptr = (void*)0x123456789ABCDEF0ULL;

    /* 포인터 -> uintptr_t -> 포인터 */
    uintptr_t as_int = (uintptr_t)original_ptr;
    void *back_to_ptr = (void*)as_int;

    printf("  Original: %p\n", original_ptr);
    printf("  As uintptr_t: 0x%" PRIX64 "\n", (uint64_t)as_int);
    printf("  Back to ptr: %p\n", back_to_ptr);

    if (original_ptr != back_to_ptr) {
        TEST_FAIL("pointer conversion", "roundtrip failed");
        return;
    }

    /* 포인터 -> uint64_t -> 포인터 (uintptr_t 경유) */
    uint64_t as_u64 = (uint64_t)(uintptr_t)original_ptr;
    void *back_from_u64 = (void*)(uintptr_t)as_u64;

    if (original_ptr != back_from_u64) {
        TEST_FAIL("pointer to uint64_t conversion", "roundtrip failed");
        return;
    }

    TEST_PASS("pointer-integer conversion");
}

/**
 * 비트 연산 테스트
 */
void test_bit_operations(void)
{
    uint64_t value = 0x123456789ABCDEF0ULL;

    /* 비트 시프트 */
    uint64_t left_shift = value << 4;
    uint64_t right_shift = value >> 4;

    printf("  Original:    0x%016" PRIX64 "\n", value);
    printf("  Left << 4:   0x%016" PRIX64 "\n", left_shift);
    printf("  Right >> 4:  0x%016" PRIX64 "\n", right_shift);

    if (left_shift != 0x23456789ABCDEF00ULL) {
        TEST_FAIL("bit operations", "left shift failed");
        return;
    }

    if (right_shift != 0x0123456789ABCDEFULL) {
        TEST_FAIL("bit operations", "right shift failed");
        return;
    }

    /* 마스크 연산 */
    uint64_t mask = 0xFF00FF00FF00FF00ULL;
    uint64_t masked = value & mask;
    printf("  Masked:      0x%016" PRIX64 "\n", masked);

    TEST_PASS("bit operations");
}

/**
 * volatile 메모리 접근 테스트 (MMIO 시뮬레이션)
 */
void test_volatile_access(void)
{
    volatile uint32_t mmio_sim[4];

    /* volatile 쓰기 */
    mmio_sim[0] = 0xDEADBEEF;
    mmio_sim[1] = 0xCAFEBABE;
    mmio_sim[2] = 0x12345678;
    mmio_sim[3] = 0x87654321;

    /* volatile 읽기 */
    uint32_t read_back[4];
    for (int i = 0; i < 4; i++) {
        read_back[i] = mmio_sim[i];
    }

    printf("  Volatile write/read:\n");
    for (int i = 0; i < 4; i++) {
        printf("    [%d] 0x%08X\n", i, read_back[i]);
    }

    if (read_back[0] != 0xDEADBEEF ||
        read_back[1] != 0xCAFEBABE ||
        read_back[2] != 0x12345678 ||
        read_back[3] != 0x87654321) {
        TEST_FAIL("volatile access", "read mismatch");
        return;
    }

    TEST_PASS("volatile memory access");
}

/**
 * 메모리 정렬 매크로 테스트
 */
void test_align_macro(void)
{
#define ALIGN_UP(x, align) (((x) + (align) - 1) & ~((align) - 1))

    size_t test_cases[][3] = {
        /* input, alignment, expected */
        {0, 4, 0},
        {1, 4, 4},
        {4, 4, 4},
        {5, 4, 8},
        {1023, 1024, 1024},
        {1024, 1024, 1024},
        {1025, 1024, 2048},
        {0x12345, 0x1000, 0x13000},
    };

    bool all_ok = true;
    printf("  ALIGN_UP tests:\n");

    for (size_t i = 0; i < sizeof(test_cases) / sizeof(test_cases[0]); i++) {
        size_t input = test_cases[i][0];
        size_t align = test_cases[i][1];
        size_t expected = test_cases[i][2];
        size_t result = ALIGN_UP(input, align);

        if (result != expected) {
            printf("    [FAIL] ALIGN_UP(0x%zX, 0x%zX) = 0x%zX (expected 0x%zX)\n",
                   input, align, result, expected);
            all_ok = false;
        }
    }

    if (!all_ok) {
        TEST_FAIL("align macro", "alignment calculation failed");
        return;
    }

    printf("    All alignment tests passed\n");
    TEST_PASS("alignment macro");

#undef ALIGN_UP
}

/**
 * 64비트 주소 테스트
 */
void test_64bit_addresses(void)
{
    /* GPU 메모리 주소 시뮬레이션 (실제 CudaBridge에서 사용하는 값들) */
    uint64_t gpu_base = 0x200000000ULL;      /* 8GB offset */
    uint64_t managed_base = 0x100000000ULL;  /* 4GB offset */
    uint64_t bar_base = 0x80000000ULL;       /* 2GB offset */

    printf("  GPU memory base:     0x%016" PRIX64 "\n", gpu_base);
    printf("  Managed memory base: 0x%016" PRIX64 "\n", managed_base);
    printf("  BAR base:            0x%016" PRIX64 "\n", bar_base);

    /* 주소 연산 테스트 */
    uint64_t offset = 0x1000;
    uint64_t final_addr = gpu_base + offset;

    printf("  GPU + 0x1000:        0x%016" PRIX64 "\n", final_addr);

    if (final_addr != 0x200001000ULL) {
        TEST_FAIL("64-bit address", "address calculation failed");
        return;
    }

    /* 포인터로 변환 테스트 (실제로 접근하지는 않음) */
    void *ptr = (void*)(uintptr_t)gpu_base;
    uint64_t back = (uint64_t)(uintptr_t)ptr;

    if (back != gpu_base) {
        TEST_FAIL("64-bit address", "pointer conversion failed");
        return;
    }

    TEST_PASS("64-bit address operations");
}

/**
 * Apple Silicon 특정 테스트
 */
void test_apple_silicon(void)
{
#ifdef __APPLE__
    printf("  Platform: macOS\n");

    #if defined(__aarch64__) || defined(__arm64__)
        printf("  Apple Silicon detected\n");

        /* macOS 버전 확인 */
        #include <AvailabilityMacros.h>
        printf("  MAC_OS_X_VERSION_MIN_REQUIRED: %d\n", MAC_OS_X_VERSION_MIN_REQUIRED);

        TEST_PASS("Apple Silicon platform");
    #else
        printf("  Intel Mac detected\n");
        printf("  [WARN] CudaBridge is optimized for Apple Silicon\n");
        TEST_PASS("macOS Intel platform");
    #endif
#else
    printf("  Platform: Non-Apple\n");
    printf("  [WARN] CudaBridge is designed for macOS\n");
    TEST_PASS("non-Apple platform");
#endif
}

int main(void)
{
    printf("\n=== CudaBridge ARM Compatibility Tests ===\n\n");

    printf("== Architecture Tests ==\n");
    test_architecture();
    test_pointer_size();
    test_fixed_width_types();
    test_endianness();
    test_alignment();

    printf("\n== Format Specifier Tests ==\n");
    test_printf_formats();

    printf("\n== Data Operation Tests ==\n");
    test_pointer_int_conversion();
    test_bit_operations();
    test_volatile_access();
    test_align_macro();
    test_64bit_addresses();

    printf("\n== Platform Tests ==\n");
    test_apple_silicon();

    printf("\n=== Test Results ===\n");
    if (failures == 0) {
        printf("All ARM compatibility tests passed!\n");
        return 0;
    } else {
        printf("%d test(s) failed\n", failures);
        return 1;
    }
}
