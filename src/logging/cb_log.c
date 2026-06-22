/**
 * CudaBridge - Logging System Implementation
 */

#include "cb_log.h"

#include <errno.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/* 로그 시스템 상태 */
static struct {
    int             initialized;
    CBLogConfig     config;
    FILE           *log_file;
    char            log_file_path[1024];
    pthread_mutex_t lock;
    CBLogCallback   callback;
    void           *callback_data;
    CBLogLevel      category_levels[CB_LOG_CAT_COUNT];
    uint64_t        total_entries;
    uint64_t        dropped_entries;
    size_t          current_file_size;
} g_log = {0};

/* ANSI 색상 코드 */
static const char *level_colors[] = {
    "\033[90m",     /* TRACE: 회색 */
    "\033[36m",     /* DEBUG: 청록 */
    "\033[32m",     /* INFO:  녹색 */
    "\033[33m",     /* WARN:  노란색 */
    "\033[31m",     /* ERROR: 빨간색 */
    "\033[35;1m",   /* FATAL: 보라색 볼드 */
};
static const char *color_reset = "\033[0m";

const char* cb_log_level_name(CBLogLevel level) {
    static const char *names[] = {
        "TRACE", "DEBUG", "INFO", "WARN", "ERROR", "FATAL", "OFF"
    };
    if (level >= 0 && level <= CB_LOG_OFF) return names[level];
    return "UNKNOWN";
}

const char* cb_log_category_name(CBLogCategory category) {
    static const char *names[] = {
        "GENERAL", "DRIVER", "EGPU", "MEMORY",
        "PCIE", "USB4", "CUDA", "CLI", "PYTHON"
    };
    if (category >= 0 && category < CB_LOG_CAT_COUNT) return names[category];
    return "UNKNOWN";
}

static int ensure_log_dir(const char *dir) {
    struct stat st;
    if (stat(dir, &st) == 0) {
        return S_ISDIR(st.st_mode) ? 0 : -1;
    }
    return mkdir(dir, 0755);
}

static int open_log_file(void) {
    time_t now = time(NULL);
    struct tm *tm = localtime(&now);

    snprintf(g_log.log_file_path, sizeof(g_log.log_file_path),
             "%s/cudabridge_%04d%02d%02d_%02d%02d%02d.log",
             g_log.config.log_dir,
             tm->tm_year + 1900, tm->tm_mon + 1, tm->tm_mday,
             tm->tm_hour, tm->tm_min, tm->tm_sec);

    g_log.log_file = fopen(g_log.log_file_path, "a");
    if (!g_log.log_file) return -1;

    g_log.current_file_size = 0;

    /* 헤더 작성 */
    fprintf(g_log.log_file,
            "=== CudaBridge Log Started at %04d-%02d-%02d %02d:%02d:%02d ===\n",
            tm->tm_year + 1900, tm->tm_mon + 1, tm->tm_mday,
            tm->tm_hour, tm->tm_min, tm->tm_sec);
    fflush(g_log.log_file);

    return 0;
}

int cb_log_init(const CBLogConfig *config) {
    if (g_log.initialized) return 0;

    if (!config) return -1;

    memcpy(&g_log.config, config, sizeof(CBLogConfig));
    pthread_mutex_init(&g_log.lock, NULL);

    /* 모든 카테고리를 전역 레벨로 초기화 */
    for (int i = 0; i < CB_LOG_CAT_COUNT; i++) {
        g_log.category_levels[i] = config->min_level;
    }

    if (config->enable_file) {
        if (ensure_log_dir(config->log_dir) != 0) {
            fprintf(stderr, "[CudaBridge] Failed to create log directory: %s\n",
                    config->log_dir);
            return -1;
        }
        if (open_log_file() != 0) {
            fprintf(stderr, "[CudaBridge] Failed to open log file\n");
            return -1;
        }
    }

    g_log.total_entries = 0;
    g_log.dropped_entries = 0;
    g_log.initialized = 1;

    return 0;
}

int cb_log_init_default(void) {
    CBLogConfig config = {
        .min_level = CB_LOG_INFO,
        .enable_console = 1,
        .enable_file = 1,
        .enable_color = isatty(STDERR_FILENO),
        .enable_timestamp = 1,
        .enable_source_loc = 0,
        .max_file_size = 10 * 1024 * 1024,   /* 10MB */
        .max_rotated_files = 5,
    };
    snprintf(config.log_dir, sizeof(config.log_dir), "/tmp/cudabridge_logs");
    return cb_log_init(&config);
}

void cb_log_shutdown(void) {
    if (!g_log.initialized) return;

    pthread_mutex_lock(&g_log.lock);

    if (g_log.log_file) {
        fprintf(g_log.log_file, "=== CudaBridge Log Ended ===\n");
        fclose(g_log.log_file);
        g_log.log_file = NULL;
    }

    g_log.initialized = 0;
    pthread_mutex_unlock(&g_log.lock);
    pthread_mutex_destroy(&g_log.lock);
}

void cb_log_set_level(CBLogLevel level) {
    g_log.config.min_level = level;
    for (int i = 0; i < CB_LOG_CAT_COUNT; i++) {
        g_log.category_levels[i] = level;
    }
}

void cb_log_set_category_level(CBLogCategory category, CBLogLevel level) {
    if (category >= 0 && category < CB_LOG_CAT_COUNT) {
        g_log.category_levels[category] = level;
    }
}

void cb_log_set_callback(CBLogCallback callback, void *user_data) {
    pthread_mutex_lock(&g_log.lock);
    g_log.callback = callback;
    g_log.callback_data = user_data;
    pthread_mutex_unlock(&g_log.lock);
}

int cb_log_rotate(void) {
    if (!g_log.log_file) return -1;

    fclose(g_log.log_file);
    g_log.log_file = NULL;

    /* 기존 파일 로테이션 */
    char old_path[1100], new_path[1100];
    for (int i = g_log.config.max_rotated_files - 1; i > 0; i--) {
        snprintf(old_path, sizeof(old_path), "%s.%d", g_log.log_file_path, i - 1);
        snprintf(new_path, sizeof(new_path), "%s.%d", g_log.log_file_path, i);
        rename(old_path, new_path);
    }
    snprintf(new_path, sizeof(new_path), "%s.0", g_log.log_file_path);
    rename(g_log.log_file_path, new_path);

    return open_log_file();
}

void cb_log_write(CBLogLevel level, CBLogCategory category,
                  const char *file, int line, const char *func,
                  const char *fmt, ...) {
    if (!g_log.initialized) return;
    if (level < g_log.category_levels[category]) return;

    CBLogEntry entry;
    entry.level = level;
    entry.category = category;
    entry.file = file;
    entry.line = line;
    entry.func = func;

    clock_gettime(CLOCK_REALTIME, &entry.timestamp);

    va_list args;
    va_start(args, fmt);
    vsnprintf(entry.message, sizeof(entry.message), fmt, args);
    va_end(args);

    pthread_mutex_lock(&g_log.lock);
    g_log.total_entries++;

    struct tm tm;
    localtime_r(&entry.timestamp.tv_sec, &tm);

    /* 콘솔 출력 */
    if (g_log.config.enable_console && level >= CB_LOG_WARN) {
        if (g_log.config.enable_color) {
            fprintf(stderr, "%s[%s]%s [%-7s] [%-7s] %s",
                    level_colors[level], cb_log_level_name(level), color_reset,
                    cb_log_category_name(category),
                    g_log.config.enable_timestamp ?
                        (char[32]){0} : "",  /* placeholder */
                    entry.message);
        }

        if (g_log.config.enable_timestamp) {
            fprintf(stderr, "%04d-%02d-%02d %02d:%02d:%02d.%03ld ",
                    tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                    tm.tm_hour, tm.tm_min, tm.tm_sec,
                    entry.timestamp.tv_nsec / 1000000);
        }

        if (g_log.config.enable_color) {
            fprintf(stderr, "%s%-5s%s ", level_colors[level],
                    cb_log_level_name(level), color_reset);
        } else {
            fprintf(stderr, "%-5s ", cb_log_level_name(level));
        }

        fprintf(stderr, "[%-7s] %s", cb_log_category_name(category),
                entry.message);

        if (g_log.config.enable_source_loc) {
            fprintf(stderr, " (%s:%d %s)", file, line, func);
        }
        fprintf(stderr, "\n");
    }

    /* 파일 출력 */
    if (g_log.config.enable_file && g_log.log_file) {
        int written = fprintf(g_log.log_file,
                "%04d-%02d-%02d %02d:%02d:%02d.%03ld %-5s [%-7s] %s",
                tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                tm.tm_hour, tm.tm_min, tm.tm_sec,
                entry.timestamp.tv_nsec / 1000000,
                cb_log_level_name(level),
                cb_log_category_name(category),
                entry.message);

        if (g_log.config.enable_source_loc) {
            written += fprintf(g_log.log_file, " (%s:%d %s)", file, line, func);
        }
        written += fprintf(g_log.log_file, "\n");
        fflush(g_log.log_file);

        if (written > 0) {
            g_log.current_file_size += (size_t)written;
        }

        /* 로테이션 체크 */
        if (g_log.current_file_size >= g_log.config.max_file_size) {
            cb_log_rotate();
        }
    }

    /* 콜백 호출 */
    if (g_log.callback) {
        g_log.callback(&entry, g_log.callback_data);
    }

    pthread_mutex_unlock(&g_log.lock);
}

void cb_log_flush(void) {
    if (!g_log.initialized) return;
    pthread_mutex_lock(&g_log.lock);
    if (g_log.log_file) {
        fflush(g_log.log_file);
    }
    pthread_mutex_unlock(&g_log.lock);
}

void cb_log_get_stats(uint64_t *total_entries, uint64_t *dropped_entries) {
    if (total_entries) *total_entries = g_log.total_entries;
    if (dropped_entries) *dropped_entries = g_log.dropped_entries;
}
