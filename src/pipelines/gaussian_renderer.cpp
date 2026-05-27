#include "gaussian_renderer.h"

#include <fstream>
#include <stdexcept>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>

#include <optix_stubs.h>
#include <glm/gtc/matrix_inverse.hpp>

#include <stb_image.h>

#include "../optix/optix_check.h"
#include "../cuda/cuda_check.h"
#include "../utils/logs.h"
#include "../io/veg_ply_io.h"

// 2.5 not 3: tighter AABBs = faster BVH traversal; tails beyond 2.5-sigma contribute negligible density.
// Raise toward 3.0 if you see clipping artifacts on very spread-out Gaussians.
static constexpr float AABB_SIGMA_EXTENT = 2.5f;

// ===== Colormap CUDA helpers =====

static cudaTextureObject_t make_colormap_texture(cudaArray_t arr)
{
    cudaResourceDesc res = {};
    res.resType         = cudaResourceTypeArray;
    res.res.array.array = arr;
    cudaTextureDesc td  = {};
    td.addressMode[0]   = cudaAddressModeClamp;
    td.filterMode       = cudaFilterModeLinear;
    td.readMode         = cudaReadModeElementType;
    td.normalizedCoords = 1;
    cudaTextureObject_t obj = 0;
    cudaCreateTextureObject(&obj, &res, &td, nullptr);
    return obj;
}

// ===== SBT record helper =====

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) EmptyRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

// ===== Destructor =====

GaussianRenderer::~GaussianRenderer()
{
    if (colormap_tex)     cudaDestroyTextureObject(colormap_tex);
    if (d_colormap_array) cudaFreeArray(d_colormap_array);
    if (stbn_tex)         cudaDestroyTextureObject(stbn_tex);
    if (d_stbn_array)     cudaFreeArray(d_stbn_array);
    if (pipeline)         optixPipelineDestroy(pipeline);
    if (pg_hit)           optixProgramGroupDestroy(pg_hit);
    if (pg_miss)          optixProgramGroupDestroy(pg_miss);
    if (pg_raygen)        optixProgramGroupDestroy(pg_raygen);
    if (module)           optixModuleDestroy(module);
}

// ===== init =====

void GaussianRenderer::init(OptixDeviceContext ctx, int w, int h)
{
    optix_ctx = ctx;
    width = w; height = h;

    d_output.allocate(w * h * 3);
    d_accum.allocate(w * h);
    d_params.allocate(1);

    buildPipeline();
    loadSTBN();
}

// ===== loadSTBN =====

void GaussianRenderer::loadSTBN()
{
    int w, h, n;
    uint16_t *data = stbi_load_16(STBN_FILE, &w, &h, &n, 1);
    if (!data) {
        log_error("GaussianRenderer", std::string("STBN load failed: ") + stbi_failure_reason());
        return;
    }

    std::vector<float> fdata(w * h);
    constexpr float inv65535 = 1.f / 65535.f;
    for (int i = 0; i < w * h; ++i)
        fdata[i] = data[i] * inv65535;
    stbi_image_free(data);

    cudaChannelFormatDesc fmt = cudaCreateChannelDesc<float>();
    CUDA_CHECK(cudaMallocArray(&d_stbn_array, &fmt, (size_t)w, (size_t)h));
    CUDA_CHECK(cudaMemcpy2DToArray(
        d_stbn_array, 0, 0,
        fdata.data(), w * sizeof(float),
        w * sizeof(float), h,
        cudaMemcpyHostToDevice));

    cudaResourceDesc res = {};
    res.resType         = cudaResourceTypeArray;
    res.res.array.array = d_stbn_array;

    cudaTextureDesc tex = {};
    tex.addressMode[0]   = cudaAddressModeWrap;
    tex.addressMode[1]   = cudaAddressModeWrap;
    tex.filterMode       = cudaFilterModePoint;
    tex.readMode         = cudaReadModeElementType;
    tex.normalizedCoords = 0;

    CUDA_CHECK(cudaCreateTextureObject(&stbn_tex, &res, &tex, nullptr));
    log_info("GaussianRenderer", "STBN loaded: " + std::to_string(w) + "x" + std::to_string(h));
}

// ===== buildPipeline =====

void GaussianRenderer::buildPipeline()
{
    std::ifstream ptx_file(GAUSSIAN_PTX_FILE, std::ios::binary);
    if (!ptx_file)
        throw std::runtime_error(std::string("Cannot open PTX: ") + GAUSSIAN_PTX_FILE);
    std::string ptx((std::istreambuf_iterator<char>(ptx_file)),
                     std::istreambuf_iterator<char>());

    OptixModuleCompileOptions mod_opts = {};
    mod_opts.optLevel   = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    mod_opts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    OptixPipelineCompileOptions pipe_opts = {};
    pipe_opts.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipe_opts.numPayloadValues                 = 3;
    pipe_opts.numAttributeValues               = 2;
    pipe_opts.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
    pipe_opts.pipelineLaunchParamsVariableName = "params";
    pipe_opts.usesPrimitiveTypeFlags           = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

    char log[4096]; size_t log_size = sizeof(log);
    OPTIX_CHECK_LOG(optixModuleCreate(
        optix_ctx, &mod_opts, &pipe_opts,
        ptx.c_str(), ptx.size(), log, &log_size, &module),
        log, log_size);

    OptixProgramGroupOptions pg_opts = {};
    {
        OptixProgramGroupDesc desc = {};
        desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        desc.raygen.module            = module;
        desc.raygen.entryFunctionName = "__raygen__gaussian";
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            optix_ctx, &desc, 1, &pg_opts, log, &log_size, &pg_raygen), log, log_size);
    }
    {
        OptixProgramGroupDesc desc = {};
        desc.kind                       = OPTIX_PROGRAM_GROUP_KIND_MISS;
        desc.miss.module                = module;
        desc.miss.entryFunctionName     = "__miss__gaussian";
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            optix_ctx, &desc, 1, &pg_opts, log, &log_size, &pg_miss), log, log_size);
    }
    {
        OptixProgramGroupDesc desc = {};
        desc.kind                             = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc.hitgroup.moduleCH                = nullptr;
        desc.hitgroup.entryFunctionNameCH     = nullptr;
        desc.hitgroup.moduleIS                = module;
        desc.hitgroup.entryFunctionNameIS     = "__intersection__gaussian";
        desc.hitgroup.moduleAH                = module;
        desc.hitgroup.entryFunctionNameAH     = "__anyhit__gaussian";
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            optix_ctx, &desc, 1, &pg_opts, log, &log_size, &pg_hit), log, log_size);
    }

    OptixProgramGroup pgs[] = {pg_raygen, pg_miss, pg_hit};
    OptixPipelineLinkOptions link_opts = {};
    link_opts.maxTraceDepth = 1;
    OPTIX_CHECK_LOG(optixPipelineCreate(
        optix_ctx, &pipe_opts, &link_opts, pgs, 3, log, &log_size, &pipeline), log, log_size);

    // IS+AH without CH: use fixed stack size to avoid OptiX 8.1 INVALID_VALUE
    OPTIX_CHECK(optixPipelineSetStackSize(pipeline, 0, 0, 4096, 1));

    buildSBT();
    log_info("GaussianRenderer", "OptiX pipeline built");
}

void GaussianRenderer::buildSBT()
{
    auto fill_record = [](CudaBuffer<unsigned char> &buf, OptixProgramGroup pg) {
        buf.allocate(sizeof(EmptyRecord));
        EmptyRecord rec = {};
        OPTIX_CHECK(optixSbtRecordPackHeader(pg, &rec));
        CUDA_CHECK(cudaMemcpy(buf.ptr, &rec, sizeof(rec), cudaMemcpyHostToDevice));
    };

    fill_record(d_raygen_record, pg_raygen);
    fill_record(d_miss_record,   pg_miss);
    fill_record(d_hit_record,    pg_hit);

    sbt = {};
    sbt.raygenRecord                = reinterpret_cast<CUdeviceptr>(d_raygen_record.ptr);
    sbt.missRecordBase              = reinterpret_cast<CUdeviceptr>(d_miss_record.ptr);
    sbt.missRecordStrideInBytes     = sizeof(EmptyRecord);
    sbt.missRecordCount             = 1;
    sbt.hitgroupRecordBase          = reinterpret_cast<CUdeviceptr>(d_hit_record.ptr);
    sbt.hitgroupRecordStrideInBytes = sizeof(EmptyRecord);
    sbt.hitgroupRecordCount         = 1;
}

// ===== loadGaussians =====
// Centers and scales splats so positions fit in [-1, 1] on the largest axis.
// Scale parameters (log-space) are adjusted to match.
static void normalizeScene(std::vector<VegGaussian3D> &splats, float scene_scale)
{
    if (splats.empty()) return;

    float cx = 0, cy = 0, cz = 0;
    for (const auto &g : splats) { cx += g.pos_x; cy += g.pos_y; cz += g.pos_z; }
    float inv_n = 1.f / (float)splats.size();
    cx *= inv_n; cy *= inv_n; cz *= inv_n;

    for (auto &g : splats) { g.pos_x -= cx; g.pos_y -= cy; g.pos_z -= cz; }

    float max_ext = 0.f;
    for (const auto &g : splats)
        max_ext = std::max({max_ext, std::abs(g.pos_x), std::abs(g.pos_y), std::abs(g.pos_z)});

    if (max_ext < 1e-6f) return;

    float linear_scale = scene_scale / max_ext;
    float log_scale    = logf(linear_scale);
    for (auto &g : splats) {
        g.pos_x   *= linear_scale;
        g.pos_y   *= linear_scale;
        g.pos_z   *= linear_scale;
        g.scale_x += log_scale;
        g.scale_y += log_scale;
        g.scale_z += log_scale;
    }
}

// Precomputes Sigma^-1 and AABB_SIGMA_EXTENT AABB half-extents from each Gaussian's scale/rotation.

void GaussianRenderer::loadGaussians(const std::string &path, float scene_scale)
{
    auto splats = VegPLYLoader::load(path);
    normalizeScene(splats, scene_scale);
    num_gaussians = (uint32_t)splats.size();

    std::vector<GpuGaussian> gpu_data(num_gaussians);
    h_extents.resize(num_gaussians);

    // Use union of Gaussian AABBs as scene bounds for marching
    float3 smin = { 1e30f,  1e30f,  1e30f};
    float3 smax = {-1e30f, -1e30f, -1e30f};

    for (uint32_t idx = 0; idx < num_gaussians; ++idx) {
        const VegGaussian3D &v = splats[idx];
        GpuGaussian &g = gpu_data[idx];

        g.mu_x    = v.pos_x;
        g.mu_y    = v.pos_y;
        g.mu_z    = v.pos_z;
        g.opacity = v.opacity;
        g.scalar  = v.scalar;
        g.pad     = 0.f;

        // Scale: stored as log, exponentiate
        float sx = expf(v.scale_x);
        float sy = expf(v.scale_y);
        float sz = expf(v.scale_z);

        // Rotation matrix from unit quaternion (w, x, y, z)
        float qw = v.rot_w, qx = v.rot_x, qy = v.rot_y, qz = v.rot_z;
        // Normalize (should already be unit, but defensive)
        float qlen = sqrtf(qw*qw + qx*qx + qy*qy + qz*qz);
        if (qlen > 0.f) { qw /= qlen; qx /= qlen; qy /= qlen; qz /= qlen; }

        // R (row-major):
        float r[3][3] = {
            {1.f - 2.f*(qy*qy + qz*qz),  2.f*(qx*qy - qz*qw),  2.f*(qx*qz + qy*qw)},
            {       2.f*(qx*qy + qz*qw),  1.f - 2.f*(qx*qx + qz*qz),  2.f*(qy*qz - qx*qw)},
            {       2.f*(qx*qz - qy*qw),  2.f*(qy*qz + qx*qw),  1.f - 2.f*(qx*qx + qy*qy)},
        };

        float s2[3]     = {sx*sx, sy*sy, sz*sz};
        float inv_s2[3] = {1.f/s2[0], 1.f/s2[1], 1.f/s2[2]};

        // Sigma^-1_ij = sum_k  R[i][k] * (1/sk^2) * R[j][k]
        g.s00 = inv_s2[0]*r[0][0]*r[0][0] + inv_s2[1]*r[0][1]*r[0][1] + inv_s2[2]*r[0][2]*r[0][2];
        g.s01 = inv_s2[0]*r[0][0]*r[1][0] + inv_s2[1]*r[0][1]*r[1][1] + inv_s2[2]*r[0][2]*r[1][2];
        g.s02 = inv_s2[0]*r[0][0]*r[2][0] + inv_s2[1]*r[0][1]*r[2][1] + inv_s2[2]*r[0][2]*r[2][2];
        g.s11 = inv_s2[0]*r[1][0]*r[1][0] + inv_s2[1]*r[1][1]*r[1][1] + inv_s2[2]*r[1][2]*r[1][2];
        g.s12 = inv_s2[0]*r[1][0]*r[2][0] + inv_s2[1]*r[1][1]*r[2][1] + inv_s2[2]*r[1][2]*r[2][2];
        g.s22 = inv_s2[0]*r[2][0]*r[2][0] + inv_s2[1]*r[2][1]*r[2][1] + inv_s2[2]*r[2][2]*r[2][2];

        // Sigma_ii (diagonal of Sigma) = sum_k R[i][k]^2 * sk^2  -> AABB half-extents
        float sig00 = s2[0]*r[0][0]*r[0][0] + s2[1]*r[0][1]*r[0][1] + s2[2]*r[0][2]*r[0][2];
        float sig11 = s2[0]*r[1][0]*r[1][0] + s2[1]*r[1][1]*r[1][1] + s2[2]*r[1][2]*r[1][2];
        float sig22 = s2[0]*r[2][0]*r[2][0] + s2[1]*r[2][1]*r[2][1] + s2[2]*r[2][2]*r[2][2];

        float hx = AABB_SIGMA_EXTENT * sqrtf(sig00);
        float hy = AABB_SIGMA_EXTENT * sqrtf(sig11);
        float hz = AABB_SIGMA_EXTENT * sqrtf(sig22);

        h_extents[idx] = {hx, hy, hz};

        smin.x = fminf(smin.x, v.pos_x - hx);
        smin.y = fminf(smin.y, v.pos_y - hy);
        smin.z = fminf(smin.z, v.pos_z - hz);
        smax.x = fmaxf(smax.x, v.pos_x + hx);
        smax.y = fmaxf(smax.y, v.pos_y + hy);
        smax.z = fmaxf(smax.z, v.pos_z + hz);
    }

    scene_min = {smin.x, smin.y, smin.z};
    scene_max = {smax.x, smax.y, smax.z};

    d_gaussians.allocate(num_gaussians);
    CUDA_CHECK(cudaMemcpy(d_gaussians.ptr, gpu_data.data(),
        num_gaussians * sizeof(GpuGaussian), cudaMemcpyHostToDevice));

    buildBVH();
    resetAccum();
    loaded = true;

    scene_diag = glm::length(scene_max - scene_min);

    log_info("GaussianRenderer", "Loaded " + std::to_string(num_gaussians) + " gaussians");
}

// ===== buildBVH =====

void GaussianRenderer::buildBVH()
{
    num_leaves = (num_gaussians + GAUSSIANS_PER_LEAF - 1) / GAUSSIANS_PER_LEAF;

    // Download Gaussian positions (only need mu + extents, but h_extents already on host)
    std::vector<GpuGaussian> h_gaussians(num_gaussians);
    CUDA_CHECK(cudaMemcpy(h_gaussians.data(), d_gaussians.ptr,
        num_gaussians * sizeof(GpuGaussian), cudaMemcpyDeviceToHost));

    std::vector<OptixAabb> aabbs(num_leaves);
    for (uint32_t leaf = 0; leaf < num_leaves; ++leaf) {
        float3 lo = { 1e30f,  1e30f,  1e30f};
        float3 hi = {-1e30f, -1e30f, -1e30f};
        for (uint32_t k = 0; k < GAUSSIANS_PER_LEAF; ++k) {
            uint32_t gid = leaf * GAUSSIANS_PER_LEAF + k;
            if (gid >= num_gaussians) break;
            const GpuGaussian &g = h_gaussians[gid];
            const float3      &e = h_extents[gid];  // AABB_SIGMA_EXTENT extents

            lo.x = fminf(lo.x, g.mu_x - e.x);
            lo.y = fminf(lo.y, g.mu_y - e.y);
            lo.z = fminf(lo.z, g.mu_z - e.z);
            hi.x = fmaxf(hi.x, g.mu_x + e.x);
            hi.y = fmaxf(hi.y, g.mu_y + e.y);
            hi.z = fmaxf(hi.z, g.mu_z + e.z);
        }
        aabbs[leaf] = {lo.x, lo.y, lo.z, hi.x, hi.y, hi.z};
    }

    d_aabbs.allocate(num_leaves);
    CUDA_CHECK(cudaMemcpy(d_aabbs.ptr, aabbs.data(),
        num_leaves * sizeof(OptixAabb), cudaMemcpyHostToDevice));

    OptixAccelBuildOptions accel_opts = {};
    accel_opts.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_opts.operation  = OPTIX_BUILD_OPERATION_BUILD;

    uint32_t flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
    CUdeviceptr d_aabb_ptr = reinterpret_cast<CUdeviceptr>(d_aabbs.ptr);

    OptixBuildInput build_input = {};
    build_input.type                               = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    build_input.customPrimitiveArray.aabbBuffers   = &d_aabb_ptr;
    build_input.customPrimitiveArray.numPrimitives = num_leaves;
    build_input.customPrimitiveArray.flags         = flags;
    build_input.customPrimitiveArray.numSbtRecords = 1;

    OptixAccelBufferSizes sizes = {};
    OPTIX_CHECK(optixAccelComputeMemoryUsage(optix_ctx, &accel_opts, &build_input, 1, &sizes));

    CudaBuffer<unsigned char> d_temp;
    d_temp.allocate(sizes.tempSizeInBytes);
    CudaBuffer<unsigned char> d_uncompacted;
    d_uncompacted.allocate(sizes.outputSizeInBytes);

    CudaBuffer<uint64_t> d_compacted_size;
    d_compacted_size.allocate(1);
    OptixAccelEmitDesc emit_desc = {};
    emit_desc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emit_desc.result = reinterpret_cast<CUdeviceptr>(d_compacted_size.ptr);

    OPTIX_CHECK(optixAccelBuild(
        optix_ctx, 0,
        &accel_opts, &build_input, 1,
        reinterpret_cast<CUdeviceptr>(d_temp.ptr),        sizes.tempSizeInBytes,
        reinterpret_cast<CUdeviceptr>(d_uncompacted.ptr), sizes.outputSizeInBytes,
        &traversable, &emit_desc, 1));

    CUDA_CHECK(cudaDeviceSynchronize());

    uint64_t compacted_size = 0;
    CUDA_CHECK(cudaMemcpy(&compacted_size, d_compacted_size.ptr,
        sizeof(uint64_t), cudaMemcpyDeviceToHost));

    d_bvh_output.allocate(compacted_size);
    OPTIX_CHECK(optixAccelCompact(
        optix_ctx, 0, traversable,
        reinterpret_cast<CUdeviceptr>(d_bvh_output.ptr), compacted_size,
        &traversable));

    CUDA_CHECK(cudaDeviceSynchronize());
    log_info("GaussianRenderer",
        "BVH: " + std::to_string(num_leaves) + " leaves, " +
        std::to_string(compacted_size / 1024) + " KB");
}

// ===== resize =====

void GaussianRenderer::resize(int w, int h)
{
    width = w; height = h;
    d_output.allocate(w * h * 3);
    d_accum.allocate(w * h);
    resetAccum();
}

// ===== resetAccum =====

void GaussianRenderer::resetAccum()
{
    accum_id = 1;
    if (d_accum.ptr) d_accum.zero();
}

// ===== rebuildBVH =====

void GaussianRenderer::rebuildBVH()
{
    if (!loaded) return;
    buildBVH();
    resetAccum();
}

// ===== render =====

void GaussianRenderer::render(const glm::mat4 &view, const glm::mat4 &proj,
                               const glm::vec3 &cam_pos)
{
    if (!loaded) return;

    if (view != prev_view) {
        resetAccum();
        prev_view = view;
    }

    glm::mat4 VP_inv = glm::inverse(proj * view);
    auto unproject = [&](float sx, float sy) -> glm::vec3 {
        float ndc_x =  2.f * sx - 1.f;
        float ndc_y =  1.f - 2.f * sy;
        glm::vec4 clip(ndc_x, ndc_y, 0.f, 1.f);
        glm::vec4 world = VP_inv * clip;
        world /= world.w;
        return glm::vec3(world) - cam_pos;
    };

    glm::vec3 d00 = unproject(0.f, 0.f);
    glm::vec3 d10 = unproject(1.f, 0.f);
    glm::vec3 d01 = unproject(0.f, 1.f);

    // Y-up spherical: az in [0, 2*pi], el in [-pi/2, pi/2]
    float az = light_azimuth * 2.f * PT_PI;
    float el = light_elevation * PT_PI - 0.5f * PT_PI;
    float3 ld = {
        cosf(el) * sinf(az),
        sinf(el),
        cosf(el) * cosf(az),
    };

    GaussianLaunchParams lp = {};
    lp.cam_pos = {cam_pos.x, cam_pos.y, cam_pos.z};
    lp.dir_00  = {d00.x, d00.y, d00.z};
    lp.dir_du  = {d10.x - d00.x, d10.y - d00.y, d10.z - d00.z};
    lp.dir_dv  = {d01.x - d00.x, d01.y - d00.y, d01.z - d00.z};

    lp.traversable    = traversable;
    lp.gaussians      = d_gaussians.ptr;
    lp.num_gaussians  = num_gaussians;

    lp.step_size = step_size;
    lp.max_depth = max_depth;

    lp.colormap  = colormap_tex;

    lp.scene_min = {scene_min.x, scene_min.y, scene_min.z};
    lp.scene_max = {scene_max.x, scene_max.y, scene_max.z};

    lp.light_dir     = ld;
    lp.light_ambient = shadows_enabled ? light_ambient : 1.0f;

    lp.output   = d_output.ptr;
    lp.accum    = d_accum.ptr;
    lp.width    = width;
    lp.height   = height;
    lp.accum_id = accum_enabled ? accum_id : 1;
    lp.frame_id = accum_id;

    lp.stbn_tex       = stbn_tex;
    lp.use_blue_noise = (blue_noise_enabled && stbn_tex != 0) ? 1 : 0;

    CUDA_CHECK(cudaMemcpy(d_params.ptr, &lp, sizeof(lp), cudaMemcpyHostToDevice));

    OPTIX_CHECK(optixLaunch(
        pipeline, 0,
        reinterpret_cast<CUdeviceptr>(d_params.ptr), sizeof(GaussianLaunchParams),
        &sbt,
        (unsigned)width, (unsigned)height, 1));

    CUDA_CHECK(cudaDeviceSynchronize());

    if (accum_enabled) ++accum_id;
}

// ===== setColormap =====

void GaussianRenderer::setColormap(const float *rgba4, int n)
{
    if (colormap_tex)     { cudaDestroyTextureObject(colormap_tex); colormap_tex = 0; }
    if (d_colormap_array) { cudaFreeArray(d_colormap_array); d_colormap_array = nullptr; }

    cudaChannelFormatDesc fmt = cudaCreateChannelDesc<float4>();
    CUDA_CHECK(cudaMallocArray(&d_colormap_array, &fmt, (size_t)n, 0, cudaArrayDefault));
    CUDA_CHECK(cudaMemcpy2DToArray(
        d_colormap_array, 0, 0,
        rgba4, n * sizeof(float4),
        n * sizeof(float4), 1,
        cudaMemcpyHostToDevice));

    colormap_tex = make_colormap_texture(d_colormap_array);
    resetAccum();
}
