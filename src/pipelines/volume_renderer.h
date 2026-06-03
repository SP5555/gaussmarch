#pragma once
#include <string>

#include <optix.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include "../cuda/cuda_buffer.h"
#include "../optix_programs/volmarch_params.h"

class VolumeRenderer {
public:
    VolumeRenderer() = default;
    ~VolumeRenderer();

    void init(OptixDeviceContext ctx, int width, int height);
    void loadVolume(const std::string &path);
    void render(const glm::mat4 &view, const glm::mat4 &proj, const glm::vec3 &cam_pos);
    void resize(int width, int height);

    void setColormap(const float *rgba4, int n);

    const float *getOutput() const { return d_output.ptr; }
    bool isLoaded()          const { return loaded; }
    int  getVoxelCount()     const { return dim_x * dim_y * dim_z; }

    void resetAccum();

    // Public parameters
    float step_size          = 0.01f;
    float shadow_step_size   = 0.02f;
    int   max_depth          = 512;
    float density_scale      = 10.f;  // raw [0,1] values need amplification for visible density
    bool  accum_enabled      = true;
    bool  blue_noise_enabled = false;

    // Light
    float light_azimuth   = 0.1111f;
    float light_elevation = 0.7778f;
    float light_ambient   = 0.4f;
    bool  shadows_enabled = true;

private:
    void buildPipeline();
    void buildSBT();
    void loadSTBN();
    void uploadVolume(const std::string &path);

    OptixDeviceContext optix_ctx = nullptr;
    int width  = 0;
    int height = 0;
    bool loaded = false;

    int dim_x = 0, dim_y = 0, dim_z = 0;
    glm::vec3 scene_min, scene_max;

    // 3D volume texture
    cudaArray_t         d_vol_array  = nullptr;
    cudaTextureObject_t vol_tex      = 0;

    // OptiX pipeline (raygen + miss only, no BVH)
    OptixModule       module    = nullptr;
    OptixProgramGroup pg_raygen = nullptr;
    OptixProgramGroup pg_miss   = nullptr;
    OptixPipeline     pipeline  = nullptr;

    // SBT
    CudaBuffer<unsigned char> d_raygen_record;
    CudaBuffer<unsigned char> d_miss_record;
    OptixShaderBindingTable   sbt = {};

    // Output / accumulation
    CudaBuffer<float>  d_output;
    CudaBuffer<float4> d_accum;
    int accum_id = 1;

    CudaBuffer<VolumeLaunchParams> d_params;

    // STBN blue noise
    cudaArray_t         d_stbn_array = nullptr;
    cudaTextureObject_t stbn_tex     = 0;

    // 1D RGBA colormap texture
    cudaArray_t         d_colormap_array = nullptr;
    cudaTextureObject_t colormap_tex     = 0;

    glm::mat4 prev_view{0.f};
};
