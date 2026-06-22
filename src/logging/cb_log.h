/**
 * CudaBridge - Logging System
 *
 * GPU 드라이버 디버깅을 위한 구조화된 로깅 시스템.
 * 파일/콘솔 출력, 로그 레벨, 로그 로테이션을 지원합니다.
 */

#ifndef CUDABRIDGE_LOG_H
#define CUDABRIDGE_LOG_H

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 로그 레벨 */
typedef enum {
    CB_LOG_TRACE = 0,   /* 상세 추적 정보 */
    CB_LOG_DEBUG = 1,   /* 디버그 정보 */
    CB_LOG_INFO  = 2,   /* 일반 정보 */
    CB_LOG_WARN  = 3,   /* 경고 */
    CB_LOG_ERROR = 4,   /* 오류 */
    CB_LOG_FATAL = 5,   /* 치명적 오류 */
    CB_LOG_OFF   = 6    /* 로깅 비활성화 */
} CBLogLevel;

/* 로그 카테고리 */
typedef enum {
    CB_LOG_CAT_GENERAL  = 0,    /* 일반 */
    CB_LOG_CAT_DRIVER   = 1,    /* GPU 드라이버 */
    CB_LOG_CAT_EGPU     = 2,    /* eGPU 연결 */
    CB_LOG_CAT_MEMORY   = 3,    /* 메모리 관리 */
    CB_LOG_CAT_PCIE     = 4,    /* PCIe 터널 */
    CB_LOG_CAT_USB4     = 5,    /* USB4 컨트롤러 */
    CB_LOG_CAT_CUDA     = 6,    /* CUDA 런타임 */
    CB_LOG_CAT_CLI      = 7,    /* CLI 도구 */
    CB_LOG_CAT_PYTHON   = 8,    /* Python API */
    CB_LOG_CAT_COUNT    = 9
} CBLogCategory;

/* 로그 설정 */
typedef struct CBLogConfig {
    CBLogLevel      min_level;          /* 최소 로그 레벨 */
    int             enable_console;     /* 콘솔 출력 활성화 */
    int             enable_file;        /* 파일 출력 활성화 */
    int             enable_color;       /* ANSI 색상 출력 */
    int             enable_timestamp;   /* 타임스탬프 포함 */
    int             enable_source_loc;  /* 소스 위치 포함 */
    char            log_dir[512];       /* 로그 디렉토리 경로 */
    size_t          max_file_size;      /* 로그 파일 최대 크기 (bytes) */
    int             max_rotated_files;  /* 보관할 로테이션 파일 수 */
} CBLogConfig;

/* 로그 항목 */
typedef struct CBLogEntry {
    CBLogLevel      level;
    CBLogCategory   category;
    struct timespec timestamp;
    const char     *file;
    int             line;
    const char     *func;
    char            message[2048];
} CBLogEntry;

/* 로그 콜백 타입 */
typedef void (*CBLogCallback)(const CBLogEntry *entry, void *user_data);

/**
 * 로깅 시스템 초기화
 */
int cb_log_init(const CBLogConfig *config);

/**
 * 로깅 시스템 종료
 */
void cb_log_shutdown(void);

/**
 * 기본 설정으로 초기화
 */
int cb_log_init_default(void);

/**
 * 로그 레벨 설정
 */
void cb_log_set_level(CBLogLevel level);

/**
 * 카테고리별 로그 레벨 설정
 */
void cb_log_set_category_level(CBLogCategory category, CBLogLevel level);

/**
 * 로그 콜백 등록
 */
void cb_log_set_callback(CBLogCallback callback, void *user_data);

/**
 * 로그 메시지 기록
 */
void cb_log_write(CBLogLevel level, CBLogCategory category,
                  const char *file, int line, const char *func,
                  const char *fmt, ...) __attribute__((format(printf, 6, 7)));

/**
 * 로그 파일 플러시
 */
void cb_log_flush(void);

/**
 * 로그 파일 강제 로테이션
 */
int cb_log_rotate(void);

/**
 * 로그 레벨 이름 반환
 */
const char* cb_log_level_name(CBLogLevel level);

/**
 * 카테고리 이름 반환
 */
const char* cb_log_category_name(CBLogCategory category);

/**
 * 로그 통계 조회
 */
void cb_log_get_stats(uint64_t *total_entries, uint64_t *dropped_entries);

/* 편의 매크로 */
#define CB_LOG_TRACE(cat, fmt, ...) \
    cb_log_write(CB_LOG_TRACE, cat, __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)

#define CB_LOG_DEBUG(cat, fmt, ...) \
    cb_log_write(CB_LOG_DEBUG, cat, __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)

#define CB_LOG_INFO(cat, fmt, ...) \
    cb_log_write(CB_LOG_INFO, cat, __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)

#define CB_LOG_WARN(cat, fmt, ...) \
    cb_log_write(CB_LOG_WARN, cat, __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)

#define CB_LOG_ERROR(cat, fmt, ...) \
    cb_log_write(CB_LOG_ERROR, cat, __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)

#define CB_LOG_FATAL(cat, fmt, ...) \
    cb_log_write(CB_LOG_FATAL, cat, __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)

#ifdef __cplusplus
}
#endif

#endif /* CUDABRIDGE_LOG_H */
