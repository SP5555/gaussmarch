cmake_minimum_required(VERSION 3.22)
project(gaussmarch LANGUAGES C CXX CUDA)

# ===== Standards =====
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ===== CUDA Architecture =====
# RTX 4060 Laptop = SM 8.9.
set(CMAKE_CUDA_ARCHITECTURES 89)
set(PARTRACE_PTX_ARCH "sm_89" CACHE STRING "PTX target architecture for gaussmarch.cu")

# ===== OptiX SDK =====
# Pass -DOptiX_INSTALL_DIR=/path/to/OptiX at configure time, or set the
# OPTIX_INSTALL_DIR environment variable. Common install locations are
# checked automatically as a fallback.
set(OptiX_INSTALL_DIR "$ENV{OPTIX_INSTALL_DIR}" CACHE PATH "Root directory of the OptiX SDK installation")

if(NOT OptiX_INSTALL_DIR)
    # Check common default install paths
    foreach(_candidate
        "/opt/optix"
        "/usr/local/optix"
        "$ENV{HOME}/optix"
        "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.1.0"
    )
        if(EXISTS "${_candidate}/include/optix.h")
            set(OptiX_INSTALL_DIR "${_candidate}")
            break()
        endif()
    endforeach()
endif()

set(OptiX_INCLUDE "${OptiX_INSTALL_DIR}/include")
if(NOT EXISTS "${OptiX_INCLUDE}/optix.h")
    message(FATAL_ERROR
        "OptiX SDK not found. Set OptiX_INSTALL_DIR:\n"
        "  cmake .. -DOptiX_INSTALL_DIR=/path/to/OptiX\n"
        "Or set the OPTIX_INSTALL_DIR environment variable.")
endif()
message(STATUS "OptiX include: ${OptiX_INCLUDE}")

# ===== Dependencies =====
find_package(OpenGL REQUIRED)
find_package(CUDAToolkit REQUIRED)

# ===== Third-party libraries =====
# GLFW
if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/glfw/CMakeLists.txt")
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    if(NOT WIN32)
        set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_X11 ON CACHE BOOL "" FORCE)
    endif()
    add_subdirectory(third_party/glfw)
else()
    message(FATAL_ERROR "GLFW submodule not found. Run: git submodule update --init")
endif()

# GLM
if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/glm")
    add_library(glm::glm INTERFACE IMPORTED)
    set_target_properties(glm::glm PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/third_party/glm"
    )
else()
    message(FATAL_ERROR "GLM submodule not found. Run: git submodule update --init")
endif()

# ImGui
if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/imgui")
    set(IMGUI_SOURCES
        third_party/imgui/imgui.cpp
        third_party/imgui/imgui_draw.cpp
        third_party/imgui/imgui_tables.cpp
        third_party/imgui/imgui_widgets.cpp
        third_party/imgui/backends/imgui_impl_glfw.cpp
        third_party/imgui/backends/imgui_impl_opengl3.cpp
    )
    add_library(imgui STATIC ${IMGUI_SOURCES})
    target_include_directories(imgui PUBLIC
        ${CMAKE_SOURCE_DIR}/third_party/imgui
        ${CMAKE_SOURCE_DIR}/third_party/imgui/backends
    )
    target_link_libraries(imgui PRIVATE glfw OpenGL::GL)
else()
    message(FATAL_ERROR "ImGui submodule not found. Run: git submodule update --init")
endif()

# ImPlot
if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/implot")
    set(IMPLOT_SOURCES
        third_party/implot/implot.cpp
        third_party/implot/implot_items.cpp
    )
    add_library(implot STATIC ${IMPLOT_SOURCES})
    target_include_directories(implot PUBLIC
        ${CMAKE_SOURCE_DIR}/third_party/implot
        ${CMAKE_SOURCE_DIR}/third_party/imgui
    )
    target_link_libraries(implot PUBLIC imgui)
else()
    message(FATAL_ERROR "ImPlot submodule not found. Run: git submodule update --init")
endif()

# ===== Compile gaussmarch OptiX programs to PTX =====
set(GAUSSIAN_PTX "${CMAKE_BINARY_DIR}/gaussmarch.ptx")

add_custom_command(
    OUTPUT "${GAUSSIAN_PTX}"
    COMMAND "${CMAKE_CUDA_COMPILER}"
        --ptx
        -arch ${PARTRACE_PTX_ARCH}
        -std c++17
        --use_fast_math
        -I "${OptiX_INCLUDE}"
        -I "${CMAKE_SOURCE_DIR}/src"
        -o "${GAUSSIAN_PTX}"
        "${CMAKE_SOURCE_DIR}/src/optix_programs/gaussmarch.cu"
    DEPENDS
        "${CMAKE_SOURCE_DIR}/src/optix_programs/gaussmarch.cu"
        "${CMAKE_SOURCE_DIR}/src/optix_programs/gaussmarch_params.h"
    COMMENT "Compiling gaussmarch OptiX programs to gaussmarch.ptx"
)
add_custom_target(gaussian_ptx ALL DEPENDS "${GAUSSIAN_PTX}")

# ===== gaussmarch executable =====
set(PARTRACE_SOURCES
    third_party/glad/src/glad.c
    third_party/tinyfiledialogs/tinyfiledialogs.c
    src/input/input.cpp
    src/app/app_base.cpp
    src/app/gaussmarch_app.cpp
    src/io/image_io.cpp
    src/io/veg_ply_io.cpp
    src/camera/arcball_camera.cpp
    src/camera/fly_camera.cpp
    src/pipelines/gaussian_renderer.cpp
    src/ui/transfer_function.cpp
    src/main_viewer.cpp
)

add_executable(gaussmarch ${PARTRACE_SOURCES})
add_dependencies(gaussmarch gaussian_ptx)

target_include_directories(gaussmarch PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/third_party/glad/include
    ${CMAKE_SOURCE_DIR}/third_party/stb
    ${CMAKE_SOURCE_DIR}/third_party/tinyfiledialogs
    ${CMAKE_SOURCE_DIR}/third_party/cxxopts
    ${CMAKE_SOURCE_DIR}/third_party
    ${OptiX_INCLUDE}
)

target_link_libraries(gaussmarch PRIVATE
    OpenGL::GL
    glfw
    glm::glm
    CUDA::cuda_driver
    CUDA::cudart
    imgui
    implot
)

target_compile_definitions(gaussmarch PRIVATE
    GAUSSIAN_PTX_FILE="${GAUSSIAN_PTX}"
    STBN_FILE="${CMAKE_SOURCE_DIR}/data/img/stbn8.png"
    $<$<CONFIG:Debug>:DEBUG>
    # MSVC: suppress min/max macros from <windows.h> that clash with std::max
    $<$<CXX_COMPILER_ID:MSVC>:NOMINMAX>
    # MSVC: expose M_PI and friends from <cmath> (not standard C++, off by default)
    $<$<CXX_COMPILER_ID:MSVC>:_USE_MATH_DEFINES>
)

target_compile_options(gaussmarch PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Debug>:-O0 -g>
    >
)

# ===== Status =====
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Gaussian PTX output: ${GAUSSIAN_PTX}")
if(CMAKE_BUILD_TYPE)
    message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
else()
    message(STATUS "Build type: multi-config (${CMAKE_CONFIGURATION_TYPES})")
endif()
