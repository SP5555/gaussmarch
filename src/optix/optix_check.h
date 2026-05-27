#pragma once
#include <optix.h>
#include <cstdio>
#include "../utils/ansi_colors.h"

#define OPTIX_CHECK(call)                                                          \
    do {                                                                           \
        OptixResult res = (call);                                                  \
        if (res != OPTIX_SUCCESS) {                                                \
            fprintf(stderr, ANSI_RED "[OptiX] Error at %s:%d: %s\n" ANSI_RESET,   \
                    __FILE__, __LINE__, optixGetErrorString(res));                 \
            exit(EXIT_FAILURE);                                                    \
        }                                                                          \
    } while (0)

#define OPTIX_CHECK_LOG(call, log, log_size)                                       \
    do {                                                                           \
        OptixResult res = (call);                                                  \
        if (res != OPTIX_SUCCESS) {                                                \
            fprintf(stderr, ANSI_RED "[OptiX] Error at %s:%d: %s\n%s\n" ANSI_RESET,\
                    __FILE__, __LINE__, optixGetErrorString(res), log);            \
            exit(EXIT_FAILURE);                                                    \
        }                                                                          \
    } while (0)
