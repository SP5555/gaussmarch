#pragma once
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>

#ifdef DEBUG

#define PRINT_BUILD_INFO()                                                                  \
    do                                                                                      \
    {                                                                                       \
        auto now = std::chrono::system_clock::now();                                        \
        std::time_t now_c = std::chrono::system_clock::to_time_t(now);                      \
        std::cout << "[App] " << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S") \
                  << " - Running in DEBUG mode\n";                                          \
    } while (0)

#else

#define PRINT_BUILD_INFO()                                                                  \
    do                                                                                      \
    {                                                                                       \
        auto now = std::chrono::system_clock::now();                                        \
        std::time_t now_c = std::chrono::system_clock::to_time_t(now);                      \
        std::cout << "[App] " << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S") \
                  << " - Running in RELEASE mode\n";                                        \
    } while (0)

#endif

inline void log_info(const std::string &tag, const std::string &msg, const char *color = "\033[0m")
{
    std::cout << "[" << tag << "] " << color << msg << "\033[0m\n";
}

inline void log_error(const std::string &tag, const std::string &msg)
{
    std::cerr << "[" << tag << "] \033[31m" << msg << "\033[0m\n";
}

inline void log_warning(const std::string &tag, const std::string &msg)
{
    std::cerr << "[" << tag << "] \033[33m" << msg << "\033[0m\n";
}

inline void log_fatal(const std::string &tag, const std::string &msg)
{
    std::cerr << "[" << tag << "] \033[31m" << msg << "\033[0m\n";
    throw std::runtime_error("[" + tag + "] \033[31m" + msg + "\033[0m");
}