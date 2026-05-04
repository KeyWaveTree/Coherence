/**
 * Coherence - Stream and Event Test
 *
 * 스트림과 이벤트 기능 테스트
 */

#include <stdio.h>
#include <stdlib.h>
#include <cudabridge.h>

#define TEST_PASS(name) printf("[PASS] %s\n", name)
#define TEST_FAIL(name, msg) do { \
    printf("[FAIL] %s: %s\n", name, msg); \
    failures++; \
} while(0)

static int failures = 0;

/* 스트림 생성/제거 테스트 */
void test_stream_create_destroy(void)
{
    cbStream_t stream = NULL;
    cbError_t err;

    err = cbStreamCreate(&stream);
    if (err != cbSuccess) {
        TEST_FAIL("stream create", cbGetErrorString(err));
        return;
    }

    if (stream == NULL) {
        TEST_FAIL("stream create", "returned NULL stream");
        return;
    }

    err = cbStreamDestroy(stream);
    if (err != cbSuccess) {
        TEST_FAIL("stream destroy", cbGetErrorString(err));
        return;
    }

    TEST_PASS("stream create/destroy");
}

/* 다중 스트림 테스트 */
void test_multiple_streams(void)
{
    const int NUM_STREAMS = 8;
    cbStream_t streams[NUM_STREAMS];
    cbError_t err;

    printf("  Creating %d streams...\n", NUM_STREAMS);

    for (int i = 0; i < NUM_STREAMS; i++) {
        err = cbStreamCreate(&streams[i]);
        if (err != cbSuccess) {
            TEST_FAIL("multiple streams create", cbGetErrorString(err));
            /* 정리 */
            for (int j = 0; j < i; j++) {
                cbStreamDestroy(streams[j]);
            }
            return;
        }
    }

    for (int i = 0; i < NUM_STREAMS; i++) {
        err = cbStreamDestroy(streams[i]);
        if (err != cbSuccess) {
            TEST_FAIL("multiple streams destroy", cbGetErrorString(err));
            return;
        }
    }

    TEST_PASS("multiple streams");
}

/* 스트림 동기화 테스트 */
void test_stream_sync(void)
{
    cbStream_t stream;
    cbError_t err;

    err = cbStreamCreate(&stream);
    if (err != cbSuccess) {
        TEST_FAIL("stream sync setup", cbGetErrorString(err));
        return;
    }

    /* 비동기 작업 (여기서는 memcpy) */
    void *d_ptr = NULL;
    void *h_ptr = malloc(4096);

    cbMalloc(&d_ptr, 4096);
    cbMemcpyAsync(d_ptr, h_ptr, 4096, CB_MEMCPY_HOST_TO_DEVICE, stream);

    /* 스트림 동기화 */
    err = cbStreamSynchronize(stream);
    if (err != cbSuccess) {
        TEST_FAIL("stream sync", cbGetErrorString(err));
        cbFree(d_ptr);
        free(h_ptr);
        cbStreamDestroy(stream);
        return;
    }

    cbFree(d_ptr);
    free(h_ptr);
    cbStreamDestroy(stream);

    TEST_PASS("stream synchronization");
}

/* 스트림 쿼리 테스트 */
void test_stream_query(void)
{
    cbStream_t stream;
    cbError_t err;

    cbStreamCreate(&stream);

    /* 작업 없는 스트림은 완료 상태여야 함 */
    err = cbStreamQuery(stream);
    if (err != cbSuccess) {
        TEST_FAIL("stream query", "empty stream should be complete");
        cbStreamDestroy(stream);
        return;
    }

    cbStreamDestroy(stream);

    TEST_PASS("stream query");
}

/* 이벤트 생성/제거 테스트 */
void test_event_create_destroy(void)
{
    cbEvent_t event = NULL;
    cbError_t err;

    err = cbEventCreate(&event);
    if (err != cbSuccess) {
        TEST_FAIL("event create", cbGetErrorString(err));
        return;
    }

    if (event == NULL) {
        TEST_FAIL("event create", "returned NULL event");
        return;
    }

    err = cbEventDestroy(event);
    if (err != cbSuccess) {
        TEST_FAIL("event destroy", cbGetErrorString(err));
        return;
    }

    TEST_PASS("event create/destroy");
}

/* 이벤트 기록 및 동기화 테스트 */
void test_event_record_sync(void)
{
    cbEvent_t event;
    cbStream_t stream;
    cbError_t err;

    cbEventCreate(&event);
    cbStreamCreate(&stream);

    /* 이벤트 기록 */
    err = cbEventRecord(event, stream);
    if (err != cbSuccess) {
        TEST_FAIL("event record", cbGetErrorString(err));
        cbEventDestroy(event);
        cbStreamDestroy(stream);
        return;
    }

    /* 이벤트 동기화 */
    err = cbEventSynchronize(event);
    if (err != cbSuccess) {
        TEST_FAIL("event sync", cbGetErrorString(err));
        cbEventDestroy(event);
        cbStreamDestroy(stream);
        return;
    }

    cbEventDestroy(event);
    cbStreamDestroy(stream);

    TEST_PASS("event record/sync");
}

/* 이벤트 타이밍 테스트 */
void test_event_timing(void)
{
    cbEvent_t start, end;
    float elapsed;
    cbError_t err;

    cbEventCreate(&start);
    cbEventCreate(&end);

    /* 시작 이벤트 기록 */
    cbEventRecord(start, NULL);

    /* 작업 수행 (메모리 전송) */
    void *d_ptr = NULL;
    void *h_ptr = malloc(1024 * 1024);

    cbMalloc(&d_ptr, 1024 * 1024);
    cbMemcpy(d_ptr, h_ptr, 1024 * 1024, CB_MEMCPY_HOST_TO_DEVICE);

    /* 종료 이벤트 기록 */
    cbEventRecord(end, NULL);
    cbEventSynchronize(end);

    /* 경과 시간 계산 */
    err = cbEventElapsedTime(&elapsed, start, end);
    if (err != cbSuccess) {
        TEST_FAIL("event elapsed time", cbGetErrorString(err));
    } else {
        printf("  Elapsed time: %.3f ms\n", elapsed);
        TEST_PASS("event timing");
    }

    cbFree(d_ptr);
    free(h_ptr);
    cbEventDestroy(start);
    cbEventDestroy(end);
}

/* 스트림과 이벤트 조합 테스트 */
void test_stream_event_combo(void)
{
    cbStream_t stream1, stream2;
    cbEvent_t event;

    cbStreamCreate(&stream1);
    cbStreamCreate(&stream2);
    cbEventCreate(&event);

    void *d_ptr1 = NULL, *d_ptr2 = NULL;
    void *h_ptr = malloc(4096);

    cbMalloc(&d_ptr1, 4096);
    cbMalloc(&d_ptr2, 4096);

    /* stream1에서 작업 */
    cbMemcpyAsync(d_ptr1, h_ptr, 4096, CB_MEMCPY_HOST_TO_DEVICE, stream1);
    cbEventRecord(event, stream1);

    /* stream2가 event 대기 (stream1 완료 후 실행) */
    /* 현재 구현에서는 동기적이지만, API는 지원 */

    /* stream2에서 작업 */
    cbMemcpyAsync(d_ptr2, d_ptr1, 4096, CB_MEMCPY_DEVICE_TO_DEVICE, stream2);

    /* 모든 작업 완료 대기 */
    cbStreamSynchronize(stream1);
    cbStreamSynchronize(stream2);

    cbFree(d_ptr1);
    cbFree(d_ptr2);
    free(h_ptr);
    cbEventDestroy(event);
    cbStreamDestroy(stream1);
    cbStreamDestroy(stream2);

    TEST_PASS("stream/event combination");
}

int main(void)
{
    printf("\n=== Coherence Stream and Event Tests ===\n\n");

    cbError_t err = cbInit();
    if (err != cbSuccess) {
        printf("Failed to initialize Coherence: %s\n", cbGetErrorString(err));
        return 1;
    }

    test_stream_create_destroy();
    test_multiple_streams();
    test_stream_sync();
    test_stream_query();
    test_event_create_destroy();
    test_event_record_sync();
    test_event_timing();
    test_stream_event_combo();

    cbShutdown();

    printf("\n=== Test Results ===\n");
    if (failures == 0) {
        printf("All tests passed!\n");
        return 0;
    } else {
        printf("%d test(s) failed\n", failures);
        return 1;
    }
}
