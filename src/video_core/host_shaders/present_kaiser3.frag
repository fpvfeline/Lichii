// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Kaiser window resampling: a sinc kernel whose lobes are cut off by a Kaiser
// (modified Bessel) window instead of a hard radius. Ported from OptiScaler's
// bcds_kaiser3.hlsl, which is a separable radius-3 kernel (a = 3, beta = 6)
// with 6 taps per axis, from the anchor texel - 2 to + 3.

#version 460 core

layout(location = 0) in vec2 tex_coord;
layout(location = 0) out vec4 frag_color;
layout(binding = 0) uniform sampler2D tex;

#define PI 3.141592653589793
#define KAISER_A 3.0
#define KAISER_BETA 6.0
#define KAISER_TAP_0 -2.0
#define KAISER_TAP_5 3.0

float sinc(float x) {
    return abs(x) < 1e-5 ? 1.0 : sin(PI * x) / (PI * x);
}

// Modified Bessel function of the first kind, order zero.
float i0(float x) {
    float ax = abs(x);
    if (ax < 3.75) {
        float t = x / 3.75;
        float t2 = t * t;
        return 1.0 + t2 * (3.5156229 + t2 * (3.0899424 + t2 * (1.2067492 +
                                                      t2 * (0.2659732 + t2 * (0.0360768 + t2 * 0.0045813)))));
    }

    float t = 3.75 / ax;
    return (exp(ax) / sqrt(ax)) *
        (0.39894228 + t * (0.01328592 + t * (0.00225319 + t * (-0.00157565 +
                                                      t * (0.00916281 + t * (-0.02057706 +
                                                      t * (0.02635537 + t * (-0.01647633 + t * 0.00392377))))))));
}

// I0(beta) is left out of the window: every tap carries it, and the taps are
// divided by their weight sum in main, so it cancels.
float kaiser(float x) {
    float ax = abs(x);
    if (ax >= KAISER_A)
        return 0.0;

    float r = ax / KAISER_A;
    return sinc(x) * i0(KAISER_BETA * sqrt(clamp(1.0 - r * r, 0.0, 1.0)));
}

float kaiser_weight(float sum, float w) {
    return sum != 0.0 ? w / sum : 0.0;
}

ivec2 source_extent;

// Taps are clamped inside the source image rather than left to the sampler's
// border, the sinc's outer lobes otherwise smear the edges of the frame.
vec4 tap(float tx, float ty) {
    ivec2 at = ivec2(clamp(vec2(tx, ty), vec2(0.0), vec2(source_extent - ivec2(1, 1))));
    return texelFetch(tex, at, 0);
}

// One row of the kernel: accumulate its weighted taps and track the color
// range it covers, for the ringing clamp in main. Reads the tap weights and
// the row anchor declared in main.
#define KAISER_ROW(oy, wy)                                                          \
    {                                                                               \
        vec4 c0 = tap(bx + 0.0, by + (oy));                                         \
        vec4 c1 = tap(bx + 1.0, by + (oy));                                         \
        vec4 c2 = tap(bx + 2.0, by + (oy));                                         \
        vec4 c3 = tap(bx + 3.0, by + (oy));                                         \
        vec4 c4 = tap(bx + 4.0, by + (oy));                                         \
        vec4 c5 = tap(bx + 5.0, by + (oy));                                         \
                                                                                    \
        bounds_lo = min(bounds_lo, min(min(min(c0, c1), min(c2, c3)), min(c4, c5))); \
        bounds_hi = max(bounds_hi, max(max(max(c0, c1), max(c2, c3)), max(c4, c5))); \
                                                                                    \
        color += (wy) * (wx0 * c0 + wx1 * c1 + wx2 * c2 + wx3 * c3 + wx4 * c4 + wx5 * c5); \
    }

void main() {
    source_extent = ivec2(textureSize(tex, 0));
    vec2 source_size = vec2(source_extent);

    // Fragments read texel centers, so the tap anchor is the texel left of the
    // sample point and its weights are its sub-texel distance.
    vec2 src_pos = tex_coord * source_size - vec2(0.5);
    vec2 ip = floor(src_pos);
    vec2 f = src_pos - ip;

    float bx = ip.x + KAISER_TAP_0;
    float by = ip.y + KAISER_TAP_0;

    float kx0 = kaiser(KAISER_TAP_0 + 0.0 - f.x);
    float kx1 = kaiser(KAISER_TAP_0 + 1.0 - f.x);
    float kx2 = kaiser(KAISER_TAP_0 + 2.0 - f.x);
    float kx3 = kaiser(KAISER_TAP_0 + 3.0 - f.x);
    float kx4 = kaiser(KAISER_TAP_0 + 4.0 - f.x);
    float kx5 = kaiser(KAISER_TAP_0 + 5.0 - f.x);

    float ky0 = kaiser(KAISER_TAP_0 + 0.0 - f.y);
    float ky1 = kaiser(KAISER_TAP_0 + 1.0 - f.y);
    float ky2 = kaiser(KAISER_TAP_0 + 2.0 - f.y);
    float ky3 = kaiser(KAISER_TAP_0 + 3.0 - f.y);
    float ky4 = kaiser(KAISER_TAP_0 + 4.0 - f.y);
    float ky5 = kaiser(KAISER_TAP_0 + 5.0 - f.y);

    float sum_x = kx0 + kx1 + kx2 + kx3 + kx4 + kx5;
    float sum_y = ky0 + ky1 + ky2 + ky3 + ky4 + ky5;

    float wx0 = kaiser_weight(sum_x, kx0);
    float wx1 = kaiser_weight(sum_x, kx1);
    float wx2 = kaiser_weight(sum_x, kx2);
    float wx3 = kaiser_weight(sum_x, kx3);
    float wx4 = kaiser_weight(sum_x, kx4);
    float wx5 = kaiser_weight(sum_x, kx5);

    float wy0 = kaiser_weight(sum_y, ky0);
    float wy1 = kaiser_weight(sum_y, ky1);
    float wy2 = kaiser_weight(sum_y, ky2);
    float wy3 = kaiser_weight(sum_y, ky3);
    float wy4 = kaiser_weight(sum_y, ky4);
    float wy5 = kaiser_weight(sum_y, ky5);

    vec4 color = vec4(0.0);
    vec4 bounds_lo = vec4(1e30);
    vec4 bounds_hi = vec4(-1e30);

    KAISER_ROW(0.0, wy0);
    KAISER_ROW(1.0, wy1);
    KAISER_ROW(2.0, wy2);
    KAISER_ROW(3.0, wy3);
    KAISER_ROW(4.0, wy4);
    KAISER_ROW(5.0, wy5);

    // Negative sinc lobes can push the result outside the kernel's color range;
    // clamping back in keeps ringing artefacts down.
    frag_color = clamp(color, bounds_lo, bounds_hi);
}
