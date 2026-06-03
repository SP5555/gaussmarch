#include "volume_renderer.h"

#include <fstream>
#include <stdexcept>
#include <cstring>
#include <cmath>

#include <optix_stubs.h>
#include <stb_image.h>

#include "../optix/optix_check.h"
#include "../cuda/cuda_check.h"
#include "../utils/logs.h"
#include "../io/raw_volume_io.h"

static constexpr float PT_PI = 3.14159265358979323846f;

// ===== Colormap texture helper =====

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

VolumeRenderer::~VolumeRenderer()
{
    if (colormap_tex)     cudaDestroyTextureObject(colormap_tex);
    if (d_colormap_array) cudaFreeArray(d_colormap_array);
    if (stbn_tex)         cudaDestroyTextureObject(stbn_tex);
    if (d_stbn_array)     cudaFreeArray(d_stbn_array);
    if (vol_tex)          cudaDestroyTextureObject(vol_tex);
    if (d_vol_array)      cudaFreeArray(d_vol_array);
    if (pipeline)         optixPipelineDestroy(pipeline);
    if (pg_miss)          optixProgramGroupDestroy(pg_miss);
    if (pg_raygen)        optixProgramGroupDestroy(pg_raygen);
    if (module)           optixModuleDestroy(module);
}

// ===== init =====

void VolumeRenderer::init(OptixDeviceContext ctx, int w, int h)
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

void VolumeRenderer::loadSTBN()
{
    int w, h, n;
    uint8_t *data = stbi_load(STBN_FILE, &w, &h, &n, 1);
    if (!data) {
        log_error("VolumeRenderer", std::string("STBN load failed: ") + stbi_failure_reason());
        return;
    }

    std::vector<float> fdata(w * h);
    constexpr float inv255 = 1.f / 255.f;
    for (int i = 0; i < w * h; ++i)
        fdata[i] = data[i] * inv255;
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
    log_info("VolumeRenderer", "STBN loaded: " + std::to_string(w) + "x" + std::to_string(h));
}

// ===== buildPipeline =====

void VolumeRenderer::buildPipeline()
{
    std::ifstream ptx_file(VOLUME_PTX_FILE, std::ios::binary);
    if (!ptx_file)
        throw std::runtime_error(std::string("Cannot open PTX: ") + VOLUME_PTX_FILE);
    std::string ptx((std::istreambuf_iterator<char>(ptx_file)),
                     std::istreambuf_iterator<char>());

    OptixModuleCompileOptions mod_opts = {};
    mod_opts.optLevel   = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    mod_opts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    OptixPipelineCompileOptions pipe_opts = {};
    pipe_opts.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipe_opts.numPayloadValues                 = 0;
    pipe_opts.numAttributeValues               = 0;
    pipe_opts.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
    pipe_opts.pipelineLaunchParamsVariableName = "params";
    pipe_opts.usesPrimitiveTypeFlags           = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

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
        desc.raygen.entryFunctionName = "__raygen__volume";
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            optix_ctx, &desc, 1, &pg_opts, log, &log_size, &pg_raygen), log, log_size);
    }
    {
        OptixProgramGroupDesc desc = {};
        desc.kind                       = OPTIX_PROGRAM_GROUP_KIND_MISS;
        desc.miss.module                = module;
        desc.miss.entryFunctionName     = "__miss__volume";
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            optix_ctx, &desc, 1, &pg_opts, log, &log_size, &pg_miss), log, log_size);
    }

    OptixProgramGroup pgs[] = {pg_raygen, pg_miss};
    OptixPipelineLinkOptions link_opts = {};
    link_opts.maxTraceDepth = 0;
    OPTIX_CHECK_LOG(optixPipelineCreate(
        optix_ctx, &pipe_opts, &link_opts, pgs, 2, log, &log_size, &pipeline), log, log_size);

    OPTIX_CHECK(optixPipelineSetStackSize(pipeline, 0, 0, 4096, 1));

    buildSBT();
    log_info("VolumeRenderer", "OptiX pipeline built");
}

// ===== buildSBT =====

void VolumeRenderer::buildSBT()
{
    auto fill_record = [](CudaBuffer<unsigned char> &buf, OptixProgramGroup pg) {
        buf.allocate(sizeof(EmptyRecord));
        EmptyRecord rec = {};
        OPTIX_CHECK(optixSbtRecordPackHeader(pg, &rec));
        CUDA_CHECK(cudaMemcpy(buf.ptr, &rec, sizeof(rec), cudaMemcpyHostToDevice));
    };

    fill_record(d_raygen_record, pg_raygen);
    fill_record(d_miss_record,   pg_miss);

    sbt = {};
    sbt.raygenRecord            = reinterpret_cast<CUdeviceptr>(d_raygen_record.ptr);
    sbt.missRecordBase          = reinterpret_cast<CUdeviceptr>(d_miss_record.ptr);
    sbt.missRecordStrideInBytes = sizeof(EmptyRecord);
    sbt.missRecordCount         = 1;
}

// ===== uploadVolume =====

void VolumeRenderer::uploadVolume(const std::string &path)
{
    RawVolume vol = load_raw_volume(path);
    dim_x = vol.dim_x;
    dim_y = vol.dim_y;
    dim_z = vol.dim_z;

    // Scene box: normalize to [-1, 1] on the largest axis
    float max_dim = (float)std::max({dim_x, dim_y, dim_z});
    float sx = (dim_x - 1) / max_dim;
    float sy = (dim_y - 1) / max_dim;
    float sz = (dim_z - 1) / max_dim;
    scene_min = {-sx, -sy, -sz};
    scene_max = { sx,  sy,  sz};

    // Upload to 3D CUDA array with trilinear filtering
    if (vol_tex)     { cudaDestroyTextureObject(vol_tex); vol_tex = 0; }
    if (d_vol_array) { cudaFreeArray(d_vol_array); d_vol_array = nullptr; }

    cudaChannelFormatDesc fmt = cudaCreateChannelDesc<float>();
    cudaExtent extent = make_cudaExtent((size_t)dim_x, (size_t)dim_y, (size_t)dim_z);
    CUDA_CHECK(cudaMalloc3DArray(&d_vol_array, &fmt, extent));

    cudaMemcpy3DParms copy = {};
    copy.srcPtr   = make_cudaPitchedPtr(vol.data.data(),
                                        dim_x * sizeof(float), dim_x, dim_y);
    copy.dstArray = d_vol_array;
    copy.extent   = extent;
    copy.kind     = cudaMemcpyHostToDevice;
    CUDA_CHECK(cudaMemcpy3D(&copy));

    cudaResourceDesc res = {};
    res.resType         = cudaResourceTypeArray;
    res.res.array.array = d_vol_array;

    cudaTextureDesc td = {};
    td.addressMode[0]   = cudaAddressModeClamp;
    td.addressMode[1]   = cudaAddressModeClamp;
    td.addressMode[2]   = cudaAddressModeClamp;
    td.filterMode       = cudaFilterModeLinear;
    td.readMode         = cudaReadModeElementType;
    td.normalizedCoords = 1;

    CUDA_CHECK(cudaCreateTextureObject(&vol_tex, &res, &td, nullptr));
    log_info("VolumeRenderer",
        "Volume loaded: " + std::to_string(dim_x) + "x" +
        std::to_string(dim_y) + "x" + std::to_string(dim_z));
}

// ===== loadVolume =====

void VolumeRenderer::loadVolume(const std::string &path)
{
    uploadVolume(path);
    resetAccum();
    loaded = true;
}

// ===== resize =====

void VolumeRenderer::resize(int w, int h)
{
    width = w; height = h;
    d_output.allocate(w * h * 3);
    d_accum.allocate(w * h);
    resetAccum();
}

// ===== resetAccum =====

void VolumeRenderer::resetAccum()
{
    accum_id = 1;
    if (d_accum.ptr) d_accum.zero();
}

// ===== render =====

void VolumeRenderer::render(const glm::mat4 &view, const glm::mat4 &proj,
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

    float az = light_azimuth * 2.f * PT_PI;
    float el = light_elevation * PT_PI - 0.5f * PT_PI;
    float3 ld = {
        cosf(el) * sinf(az),
        sinf(el),
        cosf(el) * cosf(az),
    };

    VolumeLaunchParams lp = {};
    lp.cam_pos = {cam_pos.x, cam_pos.y, cam_pos.z};
    lp.dir_00  = {d00.x, d00.y, d00.z};
    lp.dir_du  = {d10.x - d00.x, d10.y - d00.y, d10.z - d00.z};
    lp.dir_dv  = {d01.x - d00.x, d01.y - d00.y, d01.z - d00.z};

    lp.vol_tex       = vol_tex;
    lp.step_size     = step_size;
    lp.shadow_step_size = shadow_step_size;
    lp.max_depth     = max_depth;
    lp.density_scale = density_scale;

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
        reinterpret_cast<CUdeviceptr>(d_params.ptr), sizeof(VolumeLaunchParams),
        &sbt,
        (unsigned)width, (unsigned)height, 1));

    CUDA_CHECK(cudaDeviceSynchronize());

    if (accum_enabled) ++accum_id;
}

// ===== setColormap =====

void VolumeRenderer::setColormap(const float *rgba4, int n)
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
