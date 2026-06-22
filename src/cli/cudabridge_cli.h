/**
 * CudaBridge CLI - GPU Driver Control Interface
 *
 * nvidia-smi와 유사한 GPU 제어/모니터링 CLI 도구입니다.
 * GPU 설정, eGPU 연결 관리, 성능 모니터링, 진단 기능을 제공합니다.
 *
 * 주요 명령:
 *   cudabridge-cli info          - GPU 정보 표시
 *   cudabridge-cli status        - 연결 상태 표시
 *   cudabridge-cli connect       - eGPU 연결
 *   cudabridge-cli disconnect    - eGPU 안전 해제
 *   cudabridge-cli monitor       - 실시간 모니터링
 *   cudabridge-cli config        - 설정 관리
 *   cudabridge-cli diag          - 진단 실행
 *   cudabridge-cli log           - 로그 조회
 */

#ifndef CUDABRIDGE_CLI_H
#define CUDABRIDGE_CLI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* CLI 반환 코드 */
#define CLI_OK              0
#define CLI_ERR_ARGS        1
#define CLI_ERR_INIT        2
#define CLI_ERR_DEVICE      3
#define CLI_ERR_CONNECT     4
#define CLI_ERR_DISCONNECT  5
#define CLI_ERR_CONFIG      6
#define CLI_ERR_UNKNOWN     99

/* CLI 옵션 플래그 */
typedef struct CLIOptions {
    int     verbose;        /* 상세 출력 */
    int     json_output;    /* JSON 형식 출력 */
    int     no_color;       /* 색상 비활성화 */
    int     force;          /* 강제 실행 */
    int     device_index;   /* 대상 장치 인덱스 */
    int     interval_ms;    /* 모니터링 간격 */
    int     log_lines;      /* 로그 표시 줄 수 */
    char    log_level[16];  /* 로그 레벨 문자열 */
} CLIOptions;

/* 명령 핸들러 타입 */
typedef int (*CLICommandHandler)(int argc, char *argv[], CLIOptions *opts);

/* 명령 정의 */
typedef struct CLICommand {
    const char         *name;
    const char         *alias;      /* 짧은 별명 */
    const char         *description;
    const char         *usage;
    CLICommandHandler   handler;
} CLICommand;

/* 명령 핸들러들 */
int cmd_info(int argc, char *argv[], CLIOptions *opts);
int cmd_status(int argc, char *argv[], CLIOptions *opts);
int cmd_connect(int argc, char *argv[], CLIOptions *opts);
int cmd_disconnect(int argc, char *argv[], CLIOptions *opts);
int cmd_monitor(int argc, char *argv[], CLIOptions *opts);
int cmd_config(int argc, char *argv[], CLIOptions *opts);
int cmd_diag(int argc, char *argv[], CLIOptions *opts);
int cmd_log(int argc, char *argv[], CLIOptions *opts);
int cmd_benchmark(int argc, char *argv[], CLIOptions *opts);
int cmd_reset(int argc, char *argv[], CLIOptions *opts);
int cmd_help(int argc, char *argv[], CLIOptions *opts);
int cmd_version(int argc, char *argv[], CLIOptions *opts);

/* 유틸리티 함수 */
void cli_print_banner(void);
void cli_print_separator(void);
const char* cli_format_bytes(uint64_t bytes, char *buf, size_t buf_size);
const char* cli_format_bandwidth(uint64_t bps, char *buf, size_t buf_size);
const char* cli_format_temperature(int celsius, char *buf, size_t buf_size);

#ifdef __cplusplus
}
#endif

#endif /* CUDABRIDGE_CLI_H */
