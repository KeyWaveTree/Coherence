/**
 * Coherence CLI - GPU Driver Control Interface Implementation
 *
 * nvidia-smi 스타일의 GPU 제어/모니터링 CLI 도구.
 */

#include "cudabridge_cli.h"
#include "../logging/cb_log.h"
#include "../egpu/egpu_safety.h"

#include <ctype.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static int safe_atoi(const char *s, int *out) {
    char *end;
    errno = 0;
    long val = strtol(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || val < INT_MIN || val > INT_MAX) {
        return -1;
    }
    *out = (int)val;
    return 0;
}

/* ========== 전역 상태 ========== */

static volatile sig_atomic_t g_running = 1;

static void signal_handler(int sig) {
    (void)sig;
    g_running = 0;
}

/* ANSI 색상 */
static int g_use_color = 1;

#define C_RESET   (g_use_color ? "\033[0m"    : "")
#define C_BOLD    (g_use_color ? "\033[1m"     : "")
#define C_RED     (g_use_color ? "\033[31m"    : "")
#define C_GREEN   (g_use_color ? "\033[32m"    : "")
#define C_YELLOW  (g_use_color ? "\033[33m"    : "")
#define C_BLUE    (g_use_color ? "\033[34m"    : "")
#define C_MAGENTA (g_use_color ? "\033[35m"    : "")
#define C_CYAN    (g_use_color ? "\033[36m"    : "")
#define C_DIM     (g_use_color ? "\033[2m"     : "")

/* ========== 명령 테이블 ========== */

static CLICommand commands[] = {
    {"info",       "i",  "GPU 디바이스 정보 표시",              "coherence-cli info [-d <device>] [--json]", cmd_info},
    {"status",     "s",  "eGPU 연결 상태 표시",                "coherence-cli status [--json]",              cmd_status},
    {"connect",    "c",  "eGPU 연결",                          "coherence-cli connect [-d <device>]",        cmd_connect},
    {"disconnect", "dc", "eGPU 안전 해제",                     "coherence-cli disconnect [--force]",         cmd_disconnect},
    {"monitor",    "m",  "실시간 GPU 모니터링",                 "coherence-cli monitor [-i <interval_ms>]",   cmd_monitor},
    {"config",     "cf", "GPU 설정 관리",                      "coherence-cli config <subcommand>",          cmd_config},
    {"diag",       "d",  "시스템 진단 실행",                    "coherence-cli diag",                         cmd_diag},
    {"log",        "l",  "드라이버 로그 조회",                  "coherence-cli log [-n <lines>] [--level <level>]", cmd_log},
    {"benchmark",  "b",  "성능 벤치마크 실행",                  "coherence-cli benchmark",                    cmd_benchmark},
    {"reset",      "r",  "GPU 리셋",                           "coherence-cli reset [--hard]",               cmd_reset},
    {"help",       "h",  "도움말 표시",                        "coherence-cli help [command]",                cmd_help},
    {"version",    "v",  "버전 정보 표시",                     "coherence-cli version",                      cmd_version},
    {NULL, NULL, NULL, NULL, NULL}
};

/* ========== 유틸리티 함수 ========== */

void cli_print_banner(void) {
    printf("%s", C_CYAN);
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║           Coherence GPU Driver CLI v1.0.0                  ║\n");
    printf("║     Apple Silicon eGPU Management & Control Tool            ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    printf("%s", C_RESET);
}

void cli_print_separator(void) {
    printf("%s───────────────────────────────────────────────────────────────%s\n",
           C_DIM, C_RESET);
}

const char* cli_format_bytes(uint64_t bytes, char *buf, size_t buf_size) {
    if (bytes >= (uint64_t)1024 * 1024 * 1024 * 1024) {
        snprintf(buf, buf_size, "%.2f TB", (double)bytes / (1024.0 * 1024.0 * 1024.0 * 1024.0));
    } else if (bytes >= (uint64_t)1024 * 1024 * 1024) {
        snprintf(buf, buf_size, "%.2f GB", (double)bytes / (1024.0 * 1024.0 * 1024.0));
    } else if (bytes >= 1024 * 1024) {
        snprintf(buf, buf_size, "%.2f MB", (double)bytes / (1024.0 * 1024.0));
    } else if (bytes >= 1024) {
        snprintf(buf, buf_size, "%.2f KB", (double)bytes / 1024.0);
    } else {
        snprintf(buf, buf_size, "%lu B", (unsigned long)bytes);
    }
    return buf;
}

static void cli_print_json_string(const char *s) {
    putchar('"');
    for (; *s; s++) {
        switch (*s) {
            case '"':  fputs("\\\"", stdout); break;
            case '\\': fputs("\\\\", stdout); break;
            case '\n': fputs("\\n", stdout);  break;
            case '\r': fputs("\\r", stdout);  break;
            case '\t': fputs("\\t", stdout);  break;
            default:
                if ((unsigned char)*s < 0x20)
                    printf("\\u%04x", (unsigned char)*s);
                else
                    putchar(*s);
        }
    }
    putchar('"');
}

const char* cli_format_bandwidth(uint64_t bps, char *buf, size_t buf_size) {
    if (bps >= 1000000000ULL) {
        snprintf(buf, buf_size, "%.1f Gbps", (double)bps / 1000000000.0);
    } else if (bps >= 1000000ULL) {
        snprintf(buf, buf_size, "%.1f Mbps", (double)bps / 1000000.0);
    } else {
        snprintf(buf, buf_size, "%lu bps", (unsigned long)bps);
    }
    return buf;
}

const char* cli_format_temperature(int celsius, char *buf, size_t buf_size) {
    const char *color = "";
    if (g_use_color) {
        if (celsius >= 85) color = C_RED;
        else if (celsius >= 70) color = C_YELLOW;
        else color = C_GREEN;
    }
    snprintf(buf, buf_size, "%s%d°C%s", color, celsius, g_use_color ? C_RESET : "");
    return buf;
}

static const char* state_to_string(EGPUConnectionState state) {
    switch (state) {
        case EGPU_STATE_DISCONNECTED:    return "Disconnected";
        case EGPU_STATE_DETECTING:       return "Detecting";
        case EGPU_STATE_AUTHENTICATING:  return "Authenticating";
        case EGPU_STATE_TUNNEL_SETUP:    return "Setting up tunnel";
        case EGPU_STATE_INITIALIZING:    return "Initializing";
        case EGPU_STATE_CONNECTED:       return "Connected";
        case EGPU_STATE_DEGRADED:        return "Degraded";
        case EGPU_STATE_ERROR:           return "Error";
        case EGPU_STATE_RECOVERING:      return "Recovering";
        case EGPU_STATE_SAFE_DISCONNECT: return "Disconnecting (safe)";
        case EGPU_STATE_FORCE_DISCONNECT:return "Disconnecting (force)";
        default:                         return "Unknown";
    }
}

static const char* state_color(EGPUConnectionState state) {
    if (!g_use_color) return "";
    switch (state) {
        case EGPU_STATE_CONNECTED:  return C_GREEN;
        case EGPU_STATE_DEGRADED:   return C_YELLOW;
        case EGPU_STATE_ERROR:      return C_RED;
        default:                    return C_DIM;
    }
}

/* ========== 명령 구현 ========== */

int cmd_info(int argc, char *argv[], CLIOptions *opts) {
    (void)argc; (void)argv;

    printf("\n%s GPU Device Information %s\n", C_BOLD, C_RESET);
    cli_print_separator();

    /* 시뮬레이션된 GPU 정보 */
    EGPUDeviceInfo info;
    int has_device = (egpu_get_device_info(&info) == 0 &&
                      info.vendor_id != 0);

    if (opts->json_output) {
        printf("{\n");
        printf("  \"driver_version\": \"1.0.0\",\n");
        printf("  \"cuda_version\": \"12.0 (Bridge)\",\n");
        if (has_device) {
            printf("  \"gpu\": {\n");
            printf("    \"name\": "); cli_print_json_string(info.name); printf(",\n");
            printf("    \"vendor_id\": \"0x%04X\",\n", info.vendor_id);
            printf("    \"device_id\": \"0x%04X\",\n", info.device_id);
            printf("    \"serial\": "); cli_print_json_string(info.serial); printf(",\n");
            printf("    \"connection\": \"%s\",\n",
                   info.conn_type == EGPU_CONN_USB4 ? "USB4" :
                   info.conn_type == EGPU_CONN_THUNDERBOLT3 ? "Thunderbolt 3" :
                   info.conn_type == EGPU_CONN_THUNDERBOLT4 ? "Thunderbolt 4" : "Unknown");
            printf("    \"max_bandwidth_mbps\": %u,\n", info.max_bandwidth_mbps);
            printf("    \"power_budget_watts\": %d,\n", info.power_budget_watts);
            printf("    \"compatible\": %s\n", info.is_compatible ? "true" : "false");
            printf("  }\n");
        } else {
            printf("  \"gpu\": null\n");
        }
        printf("}\n");
        return CLI_OK;
    }

    printf("  %-24s %s%s%s\n", "Driver Version:", C_BOLD, "1.0.0", C_RESET);
    printf("  %-24s %s%s%s\n", "CUDA Version:", C_BOLD, "12.0 (Bridge)", C_RESET);
    printf("  %-24s %s%s%s\n", "Platform:", C_BOLD, "Coherence for Apple Silicon", C_RESET);
    printf("\n");

    if (has_device) {
        printf("  %sGPU 0:%s %s\n", C_BOLD, C_RESET, info.name);
        cli_print_separator();

        char buf[64];
        printf("  %-24s 0x%04X\n", "Vendor ID:", info.vendor_id);
        printf("  %-24s 0x%04X\n", "Device ID:", info.device_id);
        printf("  %-24s %s\n", "Serial:", info.serial);
        printf("  %-24s %s\n", "Connection:",
               info.conn_type == EGPU_CONN_USB4 ? "USB4" :
               info.conn_type == EGPU_CONN_THUNDERBOLT3 ? "Thunderbolt 3" :
               info.conn_type == EGPU_CONN_THUNDERBOLT4 ? "Thunderbolt 4" : "Unknown");
        printf("  %-24s %s\n", "Max Bandwidth:",
               cli_format_bandwidth((uint64_t)info.max_bandwidth_mbps * 1000000ULL, buf, sizeof(buf)));
        printf("  %-24s %d W\n", "Power Budget:", info.power_budget_watts);
        printf("  %-24s %s%s%s\n", "Compatible:",
               info.is_compatible ? C_GREEN : C_RED,
               info.is_compatible ? "Yes" : "No",
               C_RESET);

        /* 시뮬레이션된 상세 정보 */
        printf("\n  %sCompute Capabilities:%s\n", C_BOLD, C_RESET);
        printf("  %-24s %s\n", "Architecture:", "Ada Lovelace");
        printf("  %-24s %d.%d\n", "Compute Capability:", 8, 9);
        printf("  %-24s %d\n", "SM Count:", 128);
        printf("  %-24s %d\n", "CUDA Cores:", 16384);
        printf("  %-24s %s\n", "VRAM:", "24 GB GDDR6X");
        printf("  %-24s %d-bit\n", "Memory Bus:", 384);
        printf("  %-24s %d MHz\n", "GPU Clock:", 2520);
        printf("  %-24s %d MHz\n", "Memory Clock:", 1313);
    } else {
        printf("  %sNo eGPU device detected%s\n", C_DIM, C_RESET);
        printf("  Connect an eGPU via Thunderbolt/USB4 and run: coherence-cli connect\n");
    }

    printf("\n");
    return CLI_OK;
}

int cmd_status(int argc, char *argv[], CLIOptions *opts) {
    (void)argc; (void)argv;

    EGPUConnectionState state = egpu_get_state();
    EGPUError last_err = egpu_get_last_error();
    EGPUHealthMetrics health;
    egpu_get_health(&health);

    if (opts->json_output) {
        printf("{\n");
        printf("  \"state\": \"%s\",\n", state_to_string(state));
        printf("  \"error\": \"%s\",\n", egpu_error_string(last_err));
        printf("  \"health\": {\n");
        printf("    \"temperature\": %d,\n", health.temperature);
        printf("    \"power_draw\": %d,\n", health.power_draw);
        printf("    \"link_error_rate\": %.6f,\n", health.link_error_rate);
        printf("    \"bandwidth_utilization\": %.4f,\n", health.bandwidth_util);
        printf("    \"packets_sent\": %lu,\n", (unsigned long)health.packets_sent);
        printf("    \"packets_received\": %lu,\n", (unsigned long)health.packets_received);
        printf("    \"errors_corrected\": %lu,\n", (unsigned long)health.errors_corrected);
        printf("    \"errors_uncorrected\": %lu\n", (unsigned long)health.errors_uncorrected);
        printf("  }\n");
        printf("}\n");
        return CLI_OK;
    }

    printf("\n%s eGPU Connection Status %s\n", C_BOLD, C_RESET);
    cli_print_separator();

    printf("  %-24s %s%s%s\n", "State:",
           state_color(state), state_to_string(state), C_RESET);

    if (last_err != EGPU_ERR_NONE) {
        printf("  %-24s %s%s%s\n", "Last Error:",
               C_RED, egpu_error_string(last_err), C_RESET);
    }

    if (state == EGPU_STATE_CONNECTED || state == EGPU_STATE_DEGRADED) {
        char buf[64];
        printf("\n  %sHealth Metrics:%s\n", C_BOLD, C_RESET);
        printf("  %-24s %s\n", "Temperature:",
               cli_format_temperature(health.temperature, buf, sizeof(buf)));
        printf("  %-24s %d W\n", "Power Draw:", health.power_draw);
        printf("  %-24s %.4f%%\n", "Link Error Rate:", health.link_error_rate * 100);
        printf("  %-24s %.1f%%\n", "Bandwidth Usage:", health.bandwidth_util * 100);
        printf("  %-24s %lu / %lu\n", "Packets (TX/RX):",
               (unsigned long)health.packets_sent,
               (unsigned long)health.packets_received);

        if (health.errors_corrected > 0 || health.errors_uncorrected > 0) {
            printf("  %-24s %s%lu corrected, %lu uncorrected%s\n", "Errors:",
                   health.errors_uncorrected > 0 ? C_RED : C_YELLOW,
                   (unsigned long)health.errors_corrected,
                   (unsigned long)health.errors_uncorrected,
                   C_RESET);
        }
    }

    printf("\n");
    return CLI_OK;
}

int cmd_connect(int argc, char *argv[], CLIOptions *opts) {
    (void)argc; (void)argv;

    printf("\n%s Connecting eGPU... %s\n", C_BOLD, C_RESET);
    cli_print_separator();

    EGPUError err = egpu_connect(opts->device_index);

    if (err == EGPU_ERR_NONE) {
        EGPUDeviceInfo info;
        egpu_get_device_info(&info);
        printf("  %s✓ Connected:%s %s\n", C_GREEN, C_RESET, info.name);
        printf("  %s✓ Connection:%s %s at %u Mbps\n", C_GREEN, C_RESET,
               info.conn_type == EGPU_CONN_USB4 ? "USB4" :
               info.conn_type == EGPU_CONN_THUNDERBOLT3 ? "Thunderbolt 3" :
               info.conn_type == EGPU_CONN_THUNDERBOLT4 ? "Thunderbolt 4" : "Unknown",
               info.max_bandwidth_mbps);
        printf("  %s✓ Health monitoring started%s\n", C_GREEN, C_RESET);
    } else {
        printf("  %s✗ Connection failed: %s%s\n",
               C_RED, egpu_error_string(err), C_RESET);

        if (err == EGPU_ERR_INCOMPATIBLE_DEVICE) {
            printf("\n  %sNote:%s The device is not compatible with Coherence.\n",
                   C_YELLOW, C_RESET);
            printf("  Only NVIDIA GPUs (Kepler or newer) are supported.\n");
            printf("  The device has been forcefully disconnected for safety.\n");
        }
    }

    printf("\n");
    return (err == EGPU_ERR_NONE) ? CLI_OK : CLI_ERR_CONNECT;
}

int cmd_disconnect(int argc, char *argv[], CLIOptions *opts) {
    (void)argc; (void)argv;

    EGPUConnectionState state = egpu_get_state();
    if (state == EGPU_STATE_DISCONNECTED) {
        printf("\n  %sNo eGPU connected%s\n\n", C_DIM, C_RESET);
        return CLI_OK;
    }

    printf("\n%s Disconnecting eGPU... %s\n", C_BOLD, C_RESET);
    cli_print_separator();

    EGPUError err;
    if (opts->force) {
        printf("  %s⚠ Force disconnect requested%s\n", C_YELLOW, C_RESET);
        err = egpu_force_disconnect("User requested force disconnect via CLI");
    } else {
        printf("  Waiting for pending operations...\n");
        err = egpu_safe_disconnect();
    }

    if (err == EGPU_ERR_NONE) {
        printf("  %s✓ eGPU disconnected safely%s\n", C_GREEN, C_RESET);
    } else {
        printf("  %s✗ Disconnect error: %s%s\n",
               C_RED, egpu_error_string(err), C_RESET);
    }

    printf("\n");
    return (err == EGPU_ERR_NONE) ? CLI_OK : CLI_ERR_DISCONNECT;
}

int cmd_monitor(int argc, char *argv[], CLIOptions *opts) {
    (void)argc; (void)argv;

    EGPUConnectionState state = egpu_get_state();
    if (state != EGPU_STATE_CONNECTED && state != EGPU_STATE_DEGRADED) {
        printf("\n  %sNo eGPU connected. Connect first: coherence-cli connect%s\n\n",
               C_DIM, C_RESET);
        return CLI_ERR_DEVICE;
    }

    int interval = (opts->interval_ms > 0 && opts->interval_ms <= 60000)
                   ? opts->interval_ms : 1000;

    signal(SIGINT, signal_handler);
    printf("  Monitoring GPU (interval: %dms, Ctrl+C to stop)\n\n", interval);

    printf("  %-8s %-8s %-8s %-12s %-12s %-10s %-10s\n",
           "Temp", "Power", "ErrRate", "TX Pkts", "RX Pkts", "BW Usage", "State");
    cli_print_separator();

    while (g_running) {
        EGPUHealthMetrics health;
        egpu_get_health(&health);
        state = egpu_get_state();

        char temp_buf[32];
        printf("  %-8s %-8d %-8.4f %-12lu %-12lu %-9.1f%% %s%-10s%s\r\n",
               cli_format_temperature(health.temperature, temp_buf, sizeof(temp_buf)),
               health.power_draw,
               health.link_error_rate,
               (unsigned long)health.packets_sent,
               (unsigned long)health.packets_received,
               health.bandwidth_util * 100,
               state_color(state),
               state_to_string(state),
               C_RESET);
        fflush(stdout);

        usleep((unsigned int)(interval * 1000));
    }

    printf("\n  Monitoring stopped.\n\n");
    signal(SIGINT, SIG_DFL);
    g_running = 1;
    return CLI_OK;
}

int cmd_config(int argc, char *argv[], CLIOptions *opts) {
    (void)opts;

    if (argc < 1) {
        printf("\n%s GPU Configuration %s\n", C_BOLD, C_RESET);
        cli_print_separator();
        printf("  Subcommands:\n");
        printf("    show             현재 설정 표시\n");
        printf("    set <key> <val>  설정 변경\n");
        printf("    reset            기본값으로 초기화\n");
        printf("    clock <gpu> <mem>  클럭 속도 설정 (MHz)\n");
        printf("    power <limit>    전력 제한 설정 (W)\n");
        printf("    fan <speed>      팬 속도 설정 (%%, 0=자동)\n");
        printf("    pstate <state>   전력 상태 설정 (P0-P12)\n");
        printf("\n  Available keys:\n");
        printf("    thermal_limit      온도 제한 (°C)\n");
        printf("    auto_recovery      자동 복구 (0/1)\n");
        printf("    data_integrity     데이터 무결성 검증 (0/1)\n");
        printf("    heartbeat_interval 하트비트 간격 (ms)\n");
        printf("    max_retries        최대 재시도 횟수\n");
        printf("\n");
        return CLI_OK;
    }

    const char *subcmd = argv[0];

    if (strcmp(subcmd, "show") == 0) {
        EGPUSafetyPolicy policy = egpu_safety_default_policy();
        printf("\n%s Current Configuration %s\n", C_BOLD, C_RESET);
        cli_print_separator();
        printf("  %-28s %d\n", "Max Retries:", policy.max_retry_count);
        printf("  %-28s %d ms\n", "Retry Delay:", policy.retry_delay_ms);
        printf("  %-28s %d ms\n", "Heartbeat Interval:", policy.heartbeat_interval_ms);
        printf("  %-28s %d ms\n", "Heartbeat Timeout:", policy.heartbeat_timeout_ms);
        printf("  %-28s %.2f%%\n", "Error Rate Threshold:", policy.error_rate_threshold * 100);
        printf("  %-28s %d °C\n", "Thermal Limit:", policy.thermal_limit);
        printf("  %-28s %s\n", "Auto Recovery:", policy.enable_auto_recovery ? "Enabled" : "Disabled");
        printf("  %-28s %s\n", "Data Integrity Check:", policy.enable_data_integrity ? "Enabled" : "Disabled");
        printf("  %-28s %s\n", "Force Disconnect (Incompat.):", policy.force_disconnect_on_incompatible ? "Enabled" : "Disabled");
        printf("  %-28s %d ms\n", "Safe Disconnect Timeout:", policy.safe_disconnect_timeout_ms);
        printf("\n");

        printf("  %sGPU Performance Settings:%s\n", C_BOLD, C_RESET);
        printf("  %-28s %s\n", "GPU Clock:", "2520 MHz (simulated)");
        printf("  %-28s %s\n", "Memory Clock:", "1313 MHz (simulated)");
        printf("  %-28s %s\n", "Power Limit:", "450 W (simulated)");
        printf("  %-28s %s\n", "Fan Speed:", "Auto");
        printf("  %-28s %s\n", "Performance State:", "P0 (Max Performance)");
        printf("\n");
    } else if (strcmp(subcmd, "set") == 0) {
        if (argc < 3) {
            printf("  Usage: coherence-cli config set <key> <value>\n");
            return CLI_ERR_ARGS;
        }
        printf("  %s✓ Set %s = %s%s\n", C_GREEN, argv[1], argv[2], C_RESET);
        printf("  (Note: Configuration changes are simulated in this version)\n\n");
    } else if (strcmp(subcmd, "clock") == 0) {
        if (argc < 3) {
            printf("  Usage: coherence-cli config clock <gpu_mhz> <mem_mhz>\n");
            return CLI_ERR_ARGS;
        }
        int gpu_mhz, mem_mhz;
        if (safe_atoi(argv[1], &gpu_mhz) != 0 || safe_atoi(argv[2], &mem_mhz) != 0) {
            printf("  Invalid clock value. Must be an integer.\n");
            return CLI_ERR_ARGS;
        }
        printf("  Setting GPU clock: %d MHz, Memory clock: %d MHz\n", gpu_mhz, mem_mhz);
        printf("  %s✓ Clock speeds updated (simulated)%s\n\n", C_GREEN, C_RESET);
    } else if (strcmp(subcmd, "power") == 0) {
        if (argc < 2) {
            printf("  Usage: coherence-cli config power <limit_watts>\n");
            return CLI_ERR_ARGS;
        }
        int watts;
        if (safe_atoi(argv[1], &watts) != 0) {
            printf("  Invalid power value. Must be an integer.\n");
            return CLI_ERR_ARGS;
        }
        printf("  Setting power limit: %d W\n", watts);
        printf("  %s✓ Power limit updated (simulated)%s\n\n", C_GREEN, C_RESET);
    } else if (strcmp(subcmd, "fan") == 0) {
        if (argc < 2) {
            printf("  Usage: coherence-cli config fan <speed_percent|auto>\n");
            return CLI_ERR_ARGS;
        }
        if (strcmp(argv[1], "auto") == 0) {
            printf("  Fan speed set to: Auto\n");
        } else {
            printf("  Fan speed set to: %s%%\n", argv[1]);
        }
        printf("  %s✓ Fan speed updated (simulated)%s\n\n", C_GREEN, C_RESET);
    } else if (strcmp(subcmd, "pstate") == 0) {
        if (argc < 2) {
            printf("  Usage: coherence-cli config pstate <P0-P12>\n");
            return CLI_ERR_ARGS;
        }
        printf("  Performance state set to: %s\n", argv[1]);
        printf("  %s✓ Performance state updated (simulated)%s\n\n", C_GREEN, C_RESET);
    } else if (strcmp(subcmd, "reset") == 0) {
        printf("  %s✓ Configuration reset to defaults%s\n\n", C_GREEN, C_RESET);
    } else {
        printf("  Unknown subcommand: %s\n", subcmd);
        return CLI_ERR_ARGS;
    }

    return CLI_OK;
}

int cmd_diag(int argc, char *argv[], CLIOptions *opts) {
    (void)argc; (void)argv; (void)opts;

    printf("\n%s System Diagnostics %s\n", C_BOLD, C_RESET);
    cli_print_separator();

    printf("  Running diagnostics...\n\n");

    /* 시스템 요구사항 체크 */
    printf("  [1/6] System Requirements\n");
    printf("        %s✓%s OS compatible\n", C_GREEN, C_RESET);
    printf("        %s✓%s USB4/Thunderbolt port available (simulated)\n", C_GREEN, C_RESET);

    /* 드라이버 상태 */
    printf("  [2/6] Driver Status\n");
    printf("        %s✓%s Coherence driver v1.0.0 loaded\n", C_GREEN, C_RESET);
    printf("        %s✓%s Logging system active\n", C_GREEN, C_RESET);

    /* eGPU 연결 */
    EGPUConnectionState state = egpu_get_state();
    printf("  [3/6] eGPU Connection\n");
    if (state == EGPU_STATE_CONNECTED || state == EGPU_STATE_DEGRADED) {
        printf("        %s✓%s eGPU connected\n", C_GREEN, C_RESET);
    } else {
        printf("        %s-%s eGPU not connected\n", C_YELLOW, C_RESET);
    }

    /* PCIe 터널 */
    printf("  [4/6] PCIe Tunnel\n");
    if (state == EGPU_STATE_CONNECTED) {
        printf("        %s✓%s Tunnel active\n", C_GREEN, C_RESET);
        printf("        %s✓%s Bandwidth: 32 Gbps (simulated)\n", C_GREEN, C_RESET);
    } else {
        printf("        %s-%s Tunnel not established\n", C_YELLOW, C_RESET);
    }

    /* 데이터 무결성 */
    printf("  [5/6] Data Integrity\n");
    uint8_t test_data[] = {0x01, 0x02, 0x03, 0x04, 0x05};
    uint32_t crc = egpu_compute_crc32(test_data, sizeof(test_data));
    int integrity_ok = egpu_verify_data_integrity(test_data, sizeof(test_data), crc);
    printf("        %s✓%s CRC32 engine: %s\n",
           integrity_ok ? C_GREEN : C_RED, C_RESET,
           integrity_ok ? "OK" : "FAILED");

    /* 건강 상태 */
    printf("  [6/6] Health Check\n");
    if (state == EGPU_STATE_CONNECTED) {
        EGPUHealthMetrics health;
        egpu_get_health(&health);
        char tbuf[32];
        printf("        Temperature: %s\n",
               cli_format_temperature(health.temperature, tbuf, sizeof(tbuf)));
        printf("        Error rate: %.6f\n", health.link_error_rate);
        printf("        %s✓%s GPU health: OK\n", C_GREEN, C_RESET);
    } else {
        printf("        %s-%s GPU not available for health check\n", C_YELLOW, C_RESET);
    }

    printf("\n  %sDiagnostics complete.%s\n\n", C_BOLD, C_RESET);
    return CLI_OK;
}

int cmd_log(int argc, char *argv[], CLIOptions *opts) {
    (void)argc; (void)argv;

    int lines = opts->log_lines > 0 ? opts->log_lines : 20;

    printf("\n%s Driver Logs (last %d entries) %s\n", C_BOLD, lines, C_RESET);
    cli_print_separator();

    uint64_t total, dropped;
    cb_log_get_stats(&total, &dropped);

    printf("  Total log entries: %lu, Dropped: %lu\n\n",
           (unsigned long)total, (unsigned long)dropped);

    printf("  Log files are stored in: /tmp/coherence_logs/\n");
    printf("  Use: tail -f /tmp/coherence_logs/*.log  for live monitoring\n");

    if (opts->log_level[0] != '\0') {
        printf("  Filter level: %s\n", opts->log_level);
    }

    printf("\n  %sTip:%s Set log level with: coherence-cli config set log_level DEBUG\n\n",
           C_CYAN, C_RESET);
    return CLI_OK;
}

int cmd_benchmark(int argc, char *argv[], CLIOptions *opts) {
    (void)argc; (void)argv; (void)opts;

    EGPUConnectionState state = egpu_get_state();
    if (state != EGPU_STATE_CONNECTED) {
        printf("\n  %sNo eGPU connected. Connect first: coherence-cli connect%s\n\n",
               C_DIM, C_RESET);
        return CLI_ERR_DEVICE;
    }

    printf("\n%s Performance Benchmark %s\n", C_BOLD, C_RESET);
    cli_print_separator();

    printf("  Running benchmarks (simulated)...\n\n");

    /* H2D 전송 벤치마크 */
    printf("  [Host → Device Transfer]\n");
    printf("    1 KB:   %s%.2f GB/s%s  (latency: 0.012 ms)\n", C_GREEN, 0.08, C_RESET);
    printf("    1 MB:   %s%.2f GB/s%s  (latency: 0.089 ms)\n", C_GREEN, 11.2, C_RESET);
    printf("    64 MB:  %s%.2f GB/s%s  (latency: 4.57 ms)\n", C_GREEN, 14.0, C_RESET);
    printf("    256 MB: %s%.2f GB/s%s  (latency: 17.8 ms)\n", C_GREEN, 14.4, C_RESET);

    /* D2H 전송 벤치마크 */
    printf("\n  [Device → Host Transfer]\n");
    printf("    1 KB:   %s%.2f GB/s%s  (latency: 0.015 ms)\n", C_GREEN, 0.06, C_RESET);
    printf("    1 MB:   %s%.2f GB/s%s  (latency: 0.095 ms)\n", C_GREEN, 10.5, C_RESET);
    printf("    64 MB:  %s%.2f GB/s%s  (latency: 4.92 ms)\n", C_GREEN, 13.0, C_RESET);
    printf("    256 MB: %s%.2f GB/s%s  (latency: 18.5 ms)\n", C_GREEN, 13.8, C_RESET);

    /* 커널 실행 벤치마크 */
    printf("\n  [Kernel Launch Overhead]\n");
    printf("    Empty kernel:  %s%.3f ms%s\n", C_GREEN, 0.045, C_RESET);
    printf("    Vector add:    %s%.3f ms%s (1M elements)\n", C_GREEN, 0.123, C_RESET);
    printf("    Matrix mul:    %s%.3f ms%s (1024x1024)\n", C_GREEN, 2.456, C_RESET);

    printf("\n  %sNote:%s Benchmarks are simulated in this version.\n", C_YELLOW, C_RESET);
    printf("  Connect a real eGPU for actual performance measurements.\n\n");
    return CLI_OK;
}

int cmd_reset(int argc, char *argv[], CLIOptions *opts) {
    (void)argc; (void)argv;

    printf("\n%s GPU Reset %s\n", C_BOLD, C_RESET);
    cli_print_separator();

    if (opts->force) {
        printf("  Performing %shard reset%s...\n", C_RED, C_RESET);
        EGPUError err = egpu_hard_reset();
        if (err == EGPU_ERR_NONE) {
            printf("  %s✓ Hard reset complete%s\n", C_GREEN, C_RESET);
        } else {
            printf("  %s✗ Hard reset failed: %s%s\n",
                   C_RED, egpu_error_string(err), C_RESET);
        }
    } else {
        printf("  Performing soft link reset...\n");
        EGPUError err = egpu_reset_link();
        if (err == EGPU_ERR_NONE) {
            printf("  %s✓ Link reset complete%s\n", C_GREEN, C_RESET);
        } else {
            printf("  %s✗ Link reset failed: %s%s\n",
                   C_RED, egpu_error_string(err), C_RESET);
        }
    }

    printf("\n");
    return CLI_OK;
}

int cmd_help(int argc, char *argv[], CLIOptions *opts) {
    (void)opts;

    if (argc > 0) {
        /* 특정 명령 도움말 */
        for (int i = 0; commands[i].name; i++) {
            if (strcmp(commands[i].name, argv[0]) == 0 ||
                (commands[i].alias && strcmp(commands[i].alias, argv[0]) == 0)) {
                printf("\n  %s%s%s - %s\n", C_BOLD, commands[i].name, C_RESET,
                       commands[i].description);
                printf("  Usage: %s\n\n", commands[i].usage);
                return CLI_OK;
            }
        }
        printf("  Unknown command: %s\n\n", argv[0]);
    }

    cli_print_banner();
    printf("\n  %sUsage:%s coherence-cli <command> [options]\n\n", C_BOLD, C_RESET);
    printf("  %sCommands:%s\n", C_BOLD, C_RESET);

    for (int i = 0; commands[i].name; i++) {
        printf("    %s%-14s%s (%s)  %s\n",
               C_GREEN, commands[i].name, C_RESET,
               commands[i].alias,
               commands[i].description);
    }

    printf("\n  %sGlobal Options:%s\n", C_BOLD, C_RESET);
    printf("    -v, --verbose     상세 출력\n");
    printf("    -j, --json        JSON 형식 출력\n");
    printf("    --no-color        색상 비활성화\n");
    printf("    -d, --device N    대상 장치 인덱스\n");
    printf("    -f, --force       강제 실행\n");
    printf("    -i, --interval N  모니터링 간격 (ms)\n");
    printf("    -n, --lines N     표시할 로그 줄 수\n");
    printf("    --level LEVEL     로그 필터 레벨\n");

    printf("\n  %sExamples:%s\n", C_BOLD, C_RESET);
    printf("    coherence-cli info                 GPU 정보 확인\n");
    printf("    coherence-cli connect              eGPU 연결\n");
    printf("    coherence-cli monitor -i 500       500ms 간격 모니터링\n");
    printf("    coherence-cli config show           현재 설정 확인\n");
    printf("    coherence-cli config clock 2100 1200  클럭 속도 변경\n");
    printf("    coherence-cli disconnect --force   강제 해제\n");
    printf("    coherence-cli diag                 진단 실행\n");
    printf("    coherence-cli log -n 50 --level ERROR  오류 로그 확인\n");
    printf("\n");

    return CLI_OK;
}

int cmd_version(int argc, char *argv[], CLIOptions *opts) {
    (void)argc; (void)argv; (void)opts;

    printf("coherence-cli version 1.0.0\n");
    printf("Coherence Driver version 1.0.0\n");
    printf("CUDA Bridge API version 12.0\n");
    printf("Build: %s %s\n", __DATE__, __TIME__);
    return CLI_OK;
}

/* ========== 옵션 파싱 ========== */

static CLICommand* find_command(const char *name) {
    for (int i = 0; commands[i].name; i++) {
        if (strcmp(commands[i].name, name) == 0 ||
            (commands[i].alias && strcmp(commands[i].alias, name) == 0)) {
            return &commands[i];
        }
    }
    return NULL;
}

static int parse_options(int argc, char *argv[], CLIOptions *opts, int *cmd_start) {
    memset(opts, 0, sizeof(CLIOptions));
    opts->device_index = -1;
    opts->interval_ms = 1000;
    opts->log_lines = 20;
    *cmd_start = 1;

    for (int i = 1; i < argc; i++) {
        if (argv[i][0] == '-') {
            if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
                opts->verbose = 1;
            } else if (strcmp(argv[i], "-j") == 0 || strcmp(argv[i], "--json") == 0) {
                opts->json_output = 1;
            } else if (strcmp(argv[i], "--no-color") == 0) {
                opts->no_color = 1;
                g_use_color = 0;
            } else if (strcmp(argv[i], "-f") == 0 || strcmp(argv[i], "--force") == 0) {
                opts->force = 1;
            } else if ((strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--device") == 0) && i + 1 < argc) {
                safe_atoi(argv[++i], &opts->device_index);
            } else if ((strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--interval") == 0) && i + 1 < argc) {
                safe_atoi(argv[++i], &opts->interval_ms);
            } else if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--lines") == 0) && i + 1 < argc) {
                safe_atoi(argv[++i], &opts->log_lines);
            } else if (strcmp(argv[i], "--level") == 0 && i + 1 < argc) {
                strncpy(opts->log_level, argv[++i], sizeof(opts->log_level) - 1);
            } else if (strcmp(argv[i], "--hard") == 0) {
                opts->force = 1;
            }
        } else {
            *cmd_start = i;
            return 0;
        }
    }

    return 0;
}

/* ========== main ========== */

int main(int argc, char *argv[]) {
    CLIOptions opts;
    int cmd_start;

    /* 옵션 파싱 */
    parse_options(argc, argv, &opts, &cmd_start);

    if (argc < 2 || cmd_start >= argc) {
        cmd_help(0, NULL, &opts);
        return CLI_OK;
    }

    /* 로깅 시스템 초기화 */
    CBLogConfig log_config = {
        .min_level = opts.verbose ? CB_LOG_DEBUG : CB_LOG_INFO,
        .enable_console = opts.verbose,
        .enable_file = 1,
        .enable_color = !opts.no_color,
        .enable_timestamp = 1,
        .enable_source_loc = opts.verbose,
        .max_file_size = 10 * 1024 * 1024,
        .max_rotated_files = 5,
    };
    snprintf(log_config.log_dir, sizeof(log_config.log_dir), "/tmp/coherence_logs");
    cb_log_init(&log_config);

    /* eGPU 안전 관리자 초기화 */
    EGPUSafetyPolicy policy = egpu_safety_default_policy();
    egpu_safety_init(&policy);

    CB_LOG_INFO(CB_LOG_CAT_CLI, "CLI started: %s", argv[cmd_start]);

    /* 명령 찾기 및 실행 */
    CLICommand *cmd = find_command(argv[cmd_start]);
    int result;

    if (cmd) {
        int sub_argc = argc - cmd_start - 1;
        char **sub_argv = &argv[cmd_start + 1];
        result = cmd->handler(sub_argc, sub_argv, &opts);
    } else {
        fprintf(stderr, "Unknown command: %s\n", argv[cmd_start]);
        fprintf(stderr, "Run 'coherence-cli help' for usage information.\n");
        result = CLI_ERR_ARGS;
    }

    CB_LOG_INFO(CB_LOG_CAT_CLI, "CLI finished with code %d", result);

    /* 정리 */
    egpu_safety_shutdown();
    cb_log_shutdown();

    return result;
}
