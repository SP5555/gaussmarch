#pragma once

// Spherical harmonic coefficients (Condon-Shortley phase convention).
// Values must match the original 3DGS codebase exactly so PLY files load correctly.

// Degree 0
static constexpr float SH_C0 = 0.28209479177387814f;

// Degree 1
static constexpr float SH_C1 = 0.4886025119029199f;

// Degree 2 (individual scalars -- constexpr arrays are not visible in CUDA device code)
static constexpr float SH_C2_0 =  1.0925484305920792f;
static constexpr float SH_C2_1 = -1.0925484305920792f;
static constexpr float SH_C2_2 =  0.3153915652525200f;
static constexpr float SH_C2_3 = -1.0925484305920792f;
static constexpr float SH_C2_4 =  0.5462742152960396f;

// Degree 3
static constexpr float SH_C3_0 = -0.5900435899266435f;
static constexpr float SH_C3_1 =  2.8906114426405540f;
static constexpr float SH_C3_2 = -0.4570457994644658f;
static constexpr float SH_C3_3 =  0.3731763325901154f;
static constexpr float SH_C3_4 = -0.4570457994644658f;
static constexpr float SH_C3_5 =  1.4453057213202770f;
static constexpr float SH_C3_6 = -0.5900435899266435f;

// Number of higher-order bands (excluding DC) for a given degree.
// degree 0 -> 0, degree 1 -> 3, degree 2 -> 8, degree 3 -> 15
inline int sh_degree_to_bands(int degree)
{
    int n = 0;
    for (int l = 1; l <= degree; l++) n += 2 * l + 1;
    return n;
}
