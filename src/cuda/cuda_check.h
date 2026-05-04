#pragma once
#include <cstdio>
#include "../utils/ansi_colors.h"

#ifdef DEBUG

#define CUDA_WARN(err)                                                            \
    do                                                                            \
    {                                                                             \
        cudaError_t e = (err);                                                    \
        if (e != cudaSuccess)                                                     \
        {                                                                         \
            fprintf(stderr, ANSI_YELLOW "[CUDA] Error at %s:%d: %s\n" ANSI_RESET, \
                    __FILE__, __LINE__, cudaGetErrorString(e));                   \
        }                                                                         \
    } while (0)

#define CUDA_CHECK(err)                                                        \
    do                                                                         \
    {                                                                          \
        cudaError_t e = (err);                                                 \
        if (e != cudaSuccess)                                                  \
        {                                                                      \
            fprintf(stderr, ANSI_RED "[CUDA] Error at %s:%d: %s\n" ANSI_RESET, \
                    __FILE__, __LINE__, cudaGetErrorString(e));                \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CUDA_SYNC_CHECK()               \
    do                                  \
    {                                   \
        cudaDeviceSynchronize();        \
        CUDA_CHECK(cudaGetLastError()); \
    } while (0)

#else

#define CUDA_WARN(err)                                                            \
    do                                                                            \
    {                                                                             \
        cudaError_t e = (err);                                                    \
        if (e != cudaSuccess)                                                     \
            fprintf(stderr, ANSI_YELLOW "[CUDA] Error at %s:%d: %s\n" ANSI_RESET, \
                    __FILE__, __LINE__, cudaGetErrorString(e));                   \
    } while (0)

#define CUDA_CHECK(err)                                                        \
    do                                                                         \
    {                                                                          \
        cudaError_t e = (err);                                                 \
        if (e != cudaSuccess)                                                  \
            fprintf(stderr, ANSI_RED "[CUDA] Error at %s:%d: %s\n" ANSI_RESET, \
                    __FILE__, __LINE__, cudaGetErrorString(e));                \
    } while (0)

// No cudaDeviceSynchronize() in release -- check the async error flag only.
#define CUDA_SYNC_CHECK() CUDA_CHECK(cudaGetLastError())

#endif // DEBUG
