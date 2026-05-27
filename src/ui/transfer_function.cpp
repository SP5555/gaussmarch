#include "transfer_function.h"

#include <cmath>
#include <stdexcept>

#include "imgui.h"
#include "implot.h"

#include "../cuda/cuda_check.h"

// ===== Texture helpers =====

static cudaTextureObject_t make_1d_texture_float4(cudaArray_t arr)
{
    cudaResourceDesc res = {};
    res.resType          = cudaResourceTypeArray;
    res.res.array.array  = arr;
    cudaTextureDesc td   = {};
    td.addressMode[0]    = cudaAddressModeClamp;
    td.filterMode        = cudaFilterModeLinear;
    td.readMode          = cudaReadModeElementType;
    td.normalizedCoords  = 1;
    cudaTextureObject_t obj = 0;
    CUDA_CHECK(cudaCreateTextureObject(&obj, &res, &td, nullptr));
    return obj;
}

static cudaTextureObject_t make_1d_texture_float(cudaArray_t arr)
{
    cudaResourceDesc res = {};
    res.resType          = cudaResourceTypeArray;
    res.res.array.array  = arr;
    cudaTextureDesc td   = {};
    td.addressMode[0]    = cudaAddressModeClamp;
    td.filterMode        = cudaFilterModeLinear;
    td.readMode          = cudaReadModeElementType;
    td.normalizedCoords  = 1;
    cudaTextureObject_t obj = 0;
    CUDA_CHECK(cudaCreateTextureObject(&obj, &res, &td, nullptr));
    return obj;
}

static void destroy_texture(cudaTextureObject_t &obj, cudaArray_t &arr)
{
    if (obj) { cudaDestroyTextureObject(obj); obj = 0; }
    if (arr) { cudaFreeArray(arr); arr = nullptr; }
}

// ===== Gradient editor layout =====

static constexpr float kBarH  = 20.f;
static constexpr float kGap   =  3.f;
static constexpr float kMrkH  = 12.f;
static constexpr float kMrkHW =  6.f;   // marker half-width

static float lerpf(float a, float b, float t) { return a + t * (b - a); }

static void draw_marker(ImDrawList *dl, float cx, float tip_y,
                        ImU32 fill, ImU32 outline)
{
    ImVec2 tip = {cx,          tip_y        };
    ImVec2 bl  = {cx - kMrkHW, tip_y + kMrkH};
    ImVec2 br  = {cx + kMrkHW, tip_y + kMrkH};
    dl->AddTriangleFilled(tip, bl, br, fill);
    dl->AddTriangle      (tip, bl, br, outline);
}

// ===== ColormapTF =====

ColormapTF::ColormapTF()
{
    // Default: blue -> cyan -> green -> yellow -> red (matches aarbfs defaults)
    pts = {
        {0.00f, 0.00f, 0.00f, 1.00f, 1.00f},
        {0.25f, 0.00f, 0.97f, 1.00f, 1.00f},
        {0.50f, 0.00f, 1.00f, 0.01f, 1.00f},
        {0.75f, 1.00f, 1.00f, 0.00f, 1.00f},
        {1.00f, 1.00f, 0.00f, 0.00f, 1.00f},
    };
}

ColormapTF::~ColormapTF() { destroy_texture(tex_obj, tex_array); }

float4 ColormapTF::sample(float t) const
{
    if (pts.empty()) return {0,0,0,0};
    if (t <= pts.front().t) { auto &p = pts.front(); return {p.r,p.g,p.b,p.a}; }
    if (t >= pts.back().t)  { auto &p = pts.back();  return {p.r,p.g,p.b,p.a}; }
    size_t hi = 1;
    while (hi < pts.size() && pts[hi].t < t) ++hi;
    const auto &lo = pts[hi-1], &hp = pts[hi];
    float a = (t - lo.t) / (hp.t - lo.t);
    return {lerpf(lo.r,hp.r,a), lerpf(lo.g,hp.g,a),
            lerpf(lo.b,hp.b,a), lerpf(lo.a,hp.a,a)};
}

void ColormapTF::rebuildTexture()
{
    // Sort locally so texture is always correct even mid-drag
    std::vector<ColormapPoint> sorted = pts;
    std::sort(sorted.begin(), sorted.end(), [](auto &a, auto &b){ return a.t < b.t; });

    destroy_texture(tex_obj, tex_array);
    cudaChannelFormatDesc fmt = cudaCreateChannelDesc<float4>();
    CUDA_CHECK(cudaMallocArray(&tex_array, &fmt, TF_TEXTURE_SIZE, 0, cudaArrayDefault));

    std::vector<float4> data(TF_TEXTURE_SIZE);
    for (int i = 0; i < TF_TEXTURE_SIZE; ++i) {
        float t = (float)i / (TF_TEXTURE_SIZE - 1);
        // Sample sorted copy inline
        float4 c = {};
        if (t <= sorted.front().t) { auto &p=sorted.front(); c={p.r,p.g,p.b,p.a}; }
        else if (t >= sorted.back().t) { auto &p=sorted.back(); c={p.r,p.g,p.b,p.a}; }
        else {
            size_t hi=1; while (hi<sorted.size() && sorted[hi].t<t) ++hi;
            const auto &lo=sorted[hi-1], &hp=sorted[hi];
            float a=(t-lo.t)/(hp.t-lo.t);
            c={lerpf(lo.r,hp.r,a),lerpf(lo.g,hp.g,a),lerpf(lo.b,hp.b,a),lerpf(lo.a,hp.a,a)};
        }
        data[i] = c;
    }

    CUDA_CHECK(cudaMemcpy2DToArray(tex_array, 0, 0,
        data.data(), TF_TEXTURE_SIZE * sizeof(float4),
        TF_TEXTURE_SIZE * sizeof(float4), 1, cudaMemcpyHostToDevice));
    tex_obj = make_1d_texture_float4(tex_array);
    dirty = false;
}

void ColormapTF::upload() { if (dirty) rebuildTexture(); }

bool ColormapTF::draw(const char *label)
{
    bool changed = false;
    ImGui::PushID(label);
    ImDrawList *dl    = ImGui::GetWindowDrawList();
    const float bar_w = ImGui::GetContentRegionAvail().x;
    const ImVec2 bar_tl = ImGui::GetCursorScreenPos();
    const ImVec2 bar_br = {bar_tl.x + bar_w, bar_tl.y + kBarH};
    const float  tip_y  = bar_br.y + kGap;
    const float  mouse_x = ImGui::GetIO().MousePos.x;

    // Sorted copy for gradient bar display (pts may be temporarily unsorted while dragging)
    std::vector<ColormapPoint> disp = pts;
    std::sort(disp.begin(), disp.end(), [](auto &a, auto &b){ return a.t < b.t; });

    // --- Draw gradient bar ---
    {
        auto bar_sample = [&](float t) -> float4 {
            if (t <= disp.front().t) { auto &p=disp.front(); return {p.r,p.g,p.b,p.a}; }
            if (t >= disp.back().t)  { auto &p=disp.back();  return {p.r,p.g,p.b,p.a}; }
            size_t hi=1; while (hi<disp.size() && disp[hi].t<t) ++hi;
            const auto &lo=disp[hi-1], &hp=disp[hi]; float a=(t-lo.t)/(hp.t-lo.t);
            return {lerpf(lo.r,hp.r,a),lerpf(lo.g,hp.g,a),lerpf(lo.b,hp.b,a),lerpf(lo.a,hp.a,a)};
        };
        auto composite = [](float4 c) -> ImU32 {
            const float bg = 0.33f;
            return IM_COL32((int)((c.x*c.w+bg*(1-c.w))*255),
                            (int)((c.y*c.w+bg*(1-c.w))*255),
                            (int)((c.z*c.w+bg*(1-c.w))*255), 255);
        };
        for (int i = 0; i < 128; ++i) {
            float t0=(float)i/128, t1=(float)(i+1)/128;
            ImU32 c0=composite(bar_sample(t0)), c1=composite(bar_sample(t1));
            dl->AddRectFilledMultiColor(
                {bar_tl.x+t0*bar_w,bar_tl.y},{bar_tl.x+t1*bar_w,bar_br.y},
                c0,c1,c1,c0);
        }
        dl->AddRect(bar_tl, bar_br, IM_COL32(80,80,80,200));
    }

    // --- Bar InvisibleButton (click-to-add; also prevents window drag on bar) ---
    ImGui::SetCursorScreenPos(bar_tl);
    ImGui::InvisibleButton("##bar", {bar_w, kBarH});
    bool bar_clicked = ImGui::IsItemClicked(ImGuiMouseButton_Left);

    // --- Per-marker InvisibleButtons ---
    // These are the key fix: ImGui sees an active widget during drag -> window doesn't move.
    int  prev_drag   = drag_idx;
    drag_idx         = -1;
    bool any_deleted = false;

    for (int i = 0; i < (int)pts.size(); ++i) {
        float cx = bar_tl.x + pts[i].t * bar_w;
        ImGui::SetCursorScreenPos({cx - kMrkHW, tip_y});
        ImGui::PushID(i);
        ImGui::InvisibleButton("##m", {kMrkHW * 2.f, kMrkH});

        if (ImGui::IsItemActivated())
            selected = i;

        if (ImGui::IsItemActive()) {
            drag_idx    = i;
            pts[i].t    = std::clamp((mouse_x - bar_tl.x) / bar_w, 0.f, 1.f);
            changed = dirty = true;
        }

        // Right-click to delete (need >= 2 stops minimum)
        if (ImGui::IsItemHovered() &&
            ImGui::IsMouseClicked(ImGuiMouseButton_Right) &&
            (int)pts.size() > 2)
        {
            pts.erase(pts.begin() + i);
            if (selected >= (int)pts.size()) selected = (int)pts.size() - 1;
            any_deleted  = true;
            changed = dirty = true;
            ImGui::PopID();
            break;
        }
        ImGui::PopID();
    }

    // Sort pts when drag ends so sample() stays valid
    if (prev_drag >= 0 && drag_idx < 0 && !any_deleted) {
        ColormapPoint moved = pts[prev_drag];
        std::stable_sort(pts.begin(), pts.end(), [](auto &a, auto &b){ return a.t < b.t; });
        for (int j = 0; j < (int)pts.size(); ++j)
            if (pts[j].t==moved.t && pts[j].r==moved.r && pts[j].g==moved.g) {
                selected = j; break;
            }
    }

    // Add stop on bar click (only when no marker drag is/was active)
    if (bar_clicked && drag_idx < 0 && prev_drag < 0) {
        float t = std::clamp((ImGui::GetIO().MousePos.x - bar_tl.x) / bar_w, 0.f, 1.f);
        float4 c = sample(t);
        pts.push_back({t, c.x, c.y, c.z, c.w});
        std::stable_sort(pts.begin(), pts.end(), [](auto &a, auto &b){ return a.t < b.t; });
        for (int j = 0; j < (int)pts.size(); ++j)
            if (pts[j].t == t) { selected = j; break; }
        changed = dirty = true;
    }

    // --- Draw markers (after interaction so position reflects this frame's drag) ---
    for (int i = 0; i < (int)pts.size(); ++i) {
        float cx      = bar_tl.x + pts[i].t * bar_w;
        bool  sel     = (i == selected);
        bool  hovered = ImGui::IsItemHovered(); // true for last InvisibleButton; not reliable here
        ImU32 fill    = IM_COL32((int)(pts[i].r*255),(int)(pts[i].g*255),(int)(pts[i].b*255),255);
        ImU32 outline = sel ? IM_COL32(255,220,60,255) : IM_COL32(150,150,150,255);
        draw_marker(dl, cx, tip_y, fill, outline);
    }

    // Advance cursor past marker row
    ImGui::SetCursorScreenPos({bar_tl.x, tip_y + kMrkH + 4.f});
    ImGui::Dummy({bar_w, 0.f});

    // --- Selected stop color editor ---
    if (selected >= 0 && selected < (int)pts.size()) {
        ImGui::Spacing();
        auto &p = pts[selected];
        float col[4] = {p.r, p.g, p.b, p.a};
        ImGui::SetNextItemWidth(-1.f);
        if (ImGui::ColorEdit4("##rgba", col, ImGuiColorEditFlags_AlphaBar)) {
            p.r=col[0]; p.g=col[1]; p.b=col[2]; p.a=col[3];
            changed = dirty = true;
        }
    }

    ImGui::Spacing();
    ImGui::TextDisabled("Click bar: add  |  Drag: move  |  Right-click: remove");

    ImGui::PopID();
    return changed;
}

// ===== ScalarTF =====

ScalarTF::ScalarTF(float default_value)
{
    pts = { {0.f, default_value}, {1.f, default_value} };
}

ScalarTF::~ScalarTF() { destroy_texture(tex_obj, tex_array); }

float ScalarTF::sample(float t) const
{
    if (pts.empty()) return 0.f;
    if (t <= pts.front().t) return pts.front().v;
    if (t >= pts.back().t)  return pts.back().v;
    size_t hi = 1;
    while (hi < pts.size() && pts[hi].t < t) ++hi;
    const auto &lo = pts[hi-1], &hp = pts[hi];
    return lerpf(lo.v, hp.v, (t - lo.t) / (hp.t - lo.t));
}

void ScalarTF::rebuildTexture()
{
    std::vector<ScalarPoint> sorted = pts;
    std::sort(sorted.begin(), sorted.end(), [](auto &a, auto &b){ return a.t < b.t; });

    destroy_texture(tex_obj, tex_array);
    cudaChannelFormatDesc fmt = cudaCreateChannelDesc<float>();
    CUDA_CHECK(cudaMallocArray(&tex_array, &fmt, TF_TEXTURE_SIZE, 0, cudaArrayDefault));

    std::vector<float> data(TF_TEXTURE_SIZE);
    for (int i = 0; i < TF_TEXTURE_SIZE; ++i) {
        float t = (float)i / (TF_TEXTURE_SIZE - 1);
        float v = 0.f;
        if      (t <= sorted.front().t) v = sorted.front().v;
        else if (t >= sorted.back().t)  v = sorted.back().v;
        else {
            size_t hi=1; while (hi<sorted.size() && sorted[hi].t<t) ++hi;
            const auto &lo=sorted[hi-1], &hp=sorted[hi];
            v = lerpf(lo.v, hp.v, (t-lo.t)/(hp.t-lo.t));
        }
        data[i] = v;
    }

    CUDA_CHECK(cudaMemcpy2DToArray(tex_array, 0, 0,
        data.data(), TF_TEXTURE_SIZE * sizeof(float),
        TF_TEXTURE_SIZE * sizeof(float), 1, cudaMemcpyHostToDevice));
    tex_obj = make_1d_texture_float(tex_array);
    dirty = false;
}

void ScalarTF::upload() { if (dirty) rebuildTexture(); }

bool ScalarTF::draw(const char *label)
{
    bool changed = false;
    ImGui::PushID(label);
    ImDrawList *dl    = ImGui::GetWindowDrawList();
    const float bar_w = ImGui::GetContentRegionAvail().x;
    const ImVec2 bar_tl = ImGui::GetCursorScreenPos();
    const ImVec2 bar_br = {bar_tl.x + bar_w, bar_tl.y + kBarH};
    const float  tip_y  = bar_br.y + kGap;
    const float  mouse_x = ImGui::GetIO().MousePos.x;

    std::vector<ScalarPoint> disp = pts;
    std::sort(disp.begin(), disp.end(), [](auto &a, auto &b){ return a.t < b.t; });

    // --- Draw gradient bar (grayscale) ---
    {
        auto bar_sample = [&](float t) -> float {
            if (t <= disp.front().t) return disp.front().v;
            if (t >= disp.back().t)  return disp.back().v;
            size_t hi=1; while (hi<disp.size() && disp[hi].t<t) ++hi;
            const auto &lo=disp[hi-1], &hp=disp[hi];
            return lerpf(lo.v, hp.v, (t-lo.t)/(hp.t-lo.t));
        };
        for (int i = 0; i < 128; ++i) {
            float t0=(float)i/128, t1=(float)(i+1)/128;
            auto to_col = [](float v) -> ImU32 {
                int b = (int)std::clamp(v*255.f, 0.f, 255.f);
                return IM_COL32(b,b,b,255);
            };
            dl->AddRectFilledMultiColor(
                {bar_tl.x+t0*bar_w,bar_tl.y},{bar_tl.x+t1*bar_w,bar_br.y},
                to_col(bar_sample(t0)), to_col(bar_sample(t1)),
                to_col(bar_sample(t1)), to_col(bar_sample(t0)));
        }
        dl->AddRect(bar_tl, bar_br, IM_COL32(80,80,80,200));
    }

    // --- Bar InvisibleButton ---
    ImGui::SetCursorScreenPos(bar_tl);
    ImGui::InvisibleButton("##bar", {bar_w, kBarH});
    bool bar_clicked = ImGui::IsItemClicked(ImGuiMouseButton_Left);

    // --- Per-marker InvisibleButtons ---
    int  prev_drag   = drag_idx;
    drag_idx         = -1;
    bool any_deleted = false;

    for (int i = 0; i < (int)pts.size(); ++i) {
        float cx = bar_tl.x + pts[i].t * bar_w;
        ImGui::SetCursorScreenPos({cx - kMrkHW, tip_y});
        ImGui::PushID(i);
        ImGui::InvisibleButton("##m", {kMrkHW * 2.f, kMrkH});

        if (ImGui::IsItemActivated())
            selected = i;

        if (ImGui::IsItemActive()) {
            drag_idx    = i;
            pts[i].t    = std::clamp((mouse_x - bar_tl.x) / bar_w, 0.f, 1.f);
            changed = dirty = true;
        }

        if (ImGui::IsItemHovered() &&
            ImGui::IsMouseClicked(ImGuiMouseButton_Right) &&
            (int)pts.size() > 2)
        {
            pts.erase(pts.begin() + i);
            if (selected >= (int)pts.size()) selected = (int)pts.size() - 1;
            any_deleted  = true;
            changed = dirty = true;
            ImGui::PopID();
            break;
        }
        ImGui::PopID();
    }

    if (prev_drag >= 0 && drag_idx < 0 && !any_deleted) {
        ScalarPoint moved = pts[prev_drag];
        std::stable_sort(pts.begin(), pts.end(), [](auto &a, auto &b){ return a.t < b.t; });
        for (int j = 0; j < (int)pts.size(); ++j)
            if (pts[j].t == moved.t && pts[j].v == moved.v) { selected = j; break; }
    }

    if (bar_clicked && drag_idx < 0 && prev_drag < 0) {
        float t = std::clamp((ImGui::GetIO().MousePos.x - bar_tl.x) / bar_w, 0.f, 1.f);
        float v = sample(t);
        pts.push_back({t, v});
        std::stable_sort(pts.begin(), pts.end(), [](auto &a, auto &b){ return a.t < b.t; });
        for (int j = 0; j < (int)pts.size(); ++j)
            if (pts[j].t == t) { selected = j; break; }
        changed = dirty = true;
    }

    // --- Draw markers ---
    for (int i = 0; i < (int)pts.size(); ++i) {
        float cx  = bar_tl.x + pts[i].t * bar_w;
        bool  sel = (i == selected);
        int   bv  = (int)std::clamp(pts[i].v * 255.f, 0.f, 255.f);
        ImU32 fill    = IM_COL32(bv, bv, bv, 255);
        ImU32 outline = sel ? IM_COL32(255,220,60,255) : IM_COL32(150,150,150,255);
        draw_marker(dl, cx, tip_y, fill, outline);
    }

    ImGui::SetCursorScreenPos({bar_tl.x, tip_y + kMrkH + 4.f});
    ImGui::Dummy({bar_w, 0.f});

    // --- Selected stop value slider ---
    if (selected >= 0 && selected < (int)pts.size()) {
        ImGui::Spacing();
        auto &p = pts[selected];
        ImGui::SetNextItemWidth(-1.f);
        if (ImGui::SliderFloat("##val", &p.v, 0.f, 1.f, "Value: %.3f")) {
            changed = dirty = true;
        }
    }

    ImGui::Spacing();
    ImGui::TextDisabled("Click bar: add  |  Drag: move  |  Right-click: remove");

    ImGui::PopID();
    return changed;
}
