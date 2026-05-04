cmake_minimum_required(VERSION 3.22)
project(diffsplat LANGUAGES C CXX CUDA)

# ===== Standards =====
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ===== CUDA Architecture =====
# Common values: 75=Turing, 86=Ampere, 89=Ada, 90=Hopper
# Override: cmake -DCMAKE_CUDA_ARCHITECTURES=89 ..
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86 89 90)
endif()

# ===== Dependencies =====
find_package(OpenGL REQUIRED)
find_package(CUDAToolkit REQUIRED)

# ===== Third-party libraries =====
# GLFW
if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/glfw/CMakeLists.txt")
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    if (NOT WIN32)
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

# ===== Shared sources =====
set(CPP_SHARED_SOURCES
    third_party/glad/src/glad.c
    third_party/tinyfiledialogs/tinyfiledialogs.c
    src/input/input.cpp
    src/app/app_base.cpp
    src/io/image_io.cpp
)

set(CUDA_SHARED_SOURCES
    src/layers/gauss_activ_layer.cu
    src/layers/persp_project_layer.cu
    src/layers/gsplat_rasterize_layer.cu
    src/layers/mse_loss_layer.cu
    src/optimizers/adam.cu
)

# ===== Helper: apply common settings to a target =====
function(diffsplat_target_setup TARGET)
    target_include_directories(${TARGET} PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/third_party/glad/include
        ${CMAKE_SOURCE_DIR}/third_party/stb
        ${CMAKE_SOURCE_DIR}/third_party/tinyfiledialogs
        ${CMAKE_SOURCE_DIR}/third_party/cxxopts
    )
    target_link_libraries(${TARGET} PRIVATE
        OpenGL::GL
        glfw
        glm::glm
        CUDA::cuda_driver
        imgui
        implot
    )
    set_target_properties(${TARGET} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_compile_options(${TARGET} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --use_fast_math
            -diag-suppress=20012
            -diag-suppress=20013
            $<$<CONFIG:Debug>:-lineinfo>
        >
    )
    target_compile_options(${TARGET} PRIVATE
        # Optimization flags:
        # Linux: -O3 for release, -O0 -g for debug
        # Windows/MSVC: /O2 for release, /Od /Zi for debug
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
            $<$<CONFIG:Release>:-O3>
            $<$<CONFIG:Debug>:-O0 -g>
        >
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:
            $<$<CONFIG:Release>:/O2>
            $<$<CONFIG:Debug>:/Od /Zi>
        >
    )
    target_compile_definitions(${TARGET} PRIVATE
        $<$<CONFIG:Debug>:DEBUG>
    )
endfunction()

# ===== fitter: differentiable image fitter =====
add_executable(fitter
    src/main_fitter.cpp
    src/app/fitter_app.cpp
    src/utils/gaussian3d_io.cpp
    src/io/ply_io.cpp
    src/pipelines/image_fitter.cpp
    ${CPP_SHARED_SOURCES}
    ${CUDA_SHARED_SOURCES}
)
diffsplat_target_setup(fitter)

# ===== viewer: PLY scene viewer =====
add_executable(viewer
    src/main_viewer.cpp
    src/app/viewer_app.cpp
    src/utils/gaussian3d_io.cpp
    src/io/ply_io.cpp
    src/camera/arcball_camera.cpp
    src/camera/fly_camera.cpp
    src/pipelines/splat_renderer.cpp
    ${CPP_SHARED_SOURCES}
    ${CUDA_SHARED_SOURCES}
)
diffsplat_target_setup(viewer)

# ===== Status =====
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
if (CMAKE_BUILD_TYPE)
    message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
else()
    message(STATUS "Build type: multi-config (${CMAKE_CONFIGURATION_TYPES})")
endif()