#pragma once
#include <string>
#include <vector>

#include <optix.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include "../cuda/cuda_buffer.h"
#include "../optix_programs/gaussmarch_params.h"

class GaussianRenderer {
public:
    GaussianRenderer() = default;
    ~GaussianRenderer();

    void init(OptixDeviceContext ctx, int width, int height);
    void loadGaussians(const std::string &path, float scene_scale = 1.f);
    void render(const glm::mat4 &view, const glm::mat4 &proj, const glm::vec3 &cam_pos);
    void resize(int width, int height);

    // Called by the TF widget callback; rgba4 is a dense float4 array of `n` entries.
    void setColormap(const float *rgba4, int n);

    const float *getOutput() const { return d_output.ptr; }
    bool isLoaded()          const { return loaded; }
    int  getGaussianCount()  const { return (int)num_gaussians; }

    void resetAccum();
    void rebuildBVH();

    // Public parameters (read/write from viewer app)
    float step_size          = 0.02f;
    float shadow_step_size   = 0.04f;  // default 2x primary; coarser is fine for shadows
    int   max_depth          = 512;
    bool  accum_enabled      = true;
    bool  blue_noise_enabled = false;

    // Light
    float light_azimuth   = 0.1111f;  // [0,1] -> [0, 2*pi]; default 40 deg
    float light_elevation = 0.7778f;  // [0,1] -> [0, pi];   default 50 deg
    float light_ambient   = 0.4f;     // [0,1]; 1.0 = no shadows
    bool  shadows_enabled = true;     // false forces ambient=1.0, skipping shadow rays

private:
    void buildPipeline();
    void buildSBT();
    void buildBVH();
    void loadSTBN();

    OptixDeviceContext optix_ctx = nullptr;
    int width  = 0;
    int height = 0;
    bool loaded = false;

    // Gaussian data
    uint32_t            num_gaussians = 0;
    CudaBuffer<GpuGaussian> d_gaussians;
    glm::vec3           scene_min, scene_max;
    float               scene_diag = 1.f;

    // BVH
    uint32_t                  num_leaves = 0;
    CudaBuffer<unsigned char> d_bvh_output;
    CudaBuffer<OptixAabb>     d_aabbs;
    OptixTraversableHandle    traversable = 0;

    // OptiX pipeline
    OptixModule       module    = nullptr;
    OptixProgramGroup pg_raygen = nullptr;
    OptixProgramGroup pg_miss   = nullptr;
    OptixProgramGroup pg_hit    = nullptr;
    OptixPipeline     pipeline  = nullptr;

    // SBT
    CudaBuffer<unsigned char> d_raygen_record;
    CudaBuffer<unsigned char> d_miss_record;
    CudaBuffer<unsigned char> d_hit_record;
    OptixShaderBindingTable   sbt = {};

    // Output / accumulation
    CudaBuffer<float>  d_output;   // RGB floats, width*height*3
    CudaBuffer<float4> d_accum;    // RGBA accum,  width*height
    int accum_id = 1;

    CudaBuffer<GaussianLaunchParams> d_params;

    // Per-Gaussian AABB half-extents (hx,hy,hz), computed correctly at load time,
    // stored on host so buildBVH can reuse them without recomputing from Sigma^-1.
    std::vector<float3> h_extents;

    // STBN blue noise
    cudaArray_t         d_stbn_array  = nullptr;
    cudaTextureObject_t stbn_tex      = 0;

    // 1D RGBA colormap texture (uploaded via setColormap)
    cudaArray_t         d_colormap_array = nullptr;
    cudaTextureObject_t colormap_tex     = 0;

    glm::mat4 prev_view{0.f};
};
