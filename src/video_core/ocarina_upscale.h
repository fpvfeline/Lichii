// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <string_view>

namespace VideoCommon {

enum class OcarinaUpscaleBackend { Native, Fsr, Dlss };

/// render_scaled: a DLSS render-scale patch shrank the scene to a size FSR 3.1 takes, so FSR 3.1
/// reconstructs it when DLSS can't run.
constexpr OcarinaUpscaleBackend SelectOcarinaUpscaler(bool fsr_enabled, bool dlss_enabled,
                                                    bool dlss_available, bool render_scaled,
                                                    std::string_view override) {
    if (override == "native") {
        return OcarinaUpscaleBackend::Native;
    }
    if (override == "fsr") {
        return OcarinaUpscaleBackend::Fsr;
    }
    if ((dlss_enabled || override == "dlss") && dlss_available) {
        return OcarinaUpscaleBackend::Dlss;
    }
    return fsr_enabled || render_scaled ? OcarinaUpscaleBackend::Fsr
                                        : OcarinaUpscaleBackend::Native;
}

/// Texture LOD bias added to Ocarina's mipmapped samplers while FSR upscales. AMD recommends
/// log2(render / output) - 1: -2.0 for docked 1440p -> 2880p (the upscale target).
/// Keep this in step with vk_rasterizer.cpp's output_extent: it is a per-backend constant, not
/// derived from the extent at runtime.
constexpr float OcarinaFsrLodBias(OcarinaUpscaleBackend backend) {
    return backend == OcarinaUpscaleBackend::Fsr ? -2.0f : 0.0f;
}

/// Whether a sampler with that bias keeps it for an image: only docked (handheld isn't measured),
/// not for images the GPU wrote (render targets would read a mip too sharp), and not once the
/// upscaler failed this session (the game's own pass runs again).
constexpr bool KeepOcarinaFsrLodBias(bool docked, bool upscaler_failed, bool gpu_written) {
    return docked && !upscaler_failed && !gpu_written;
}

} // namespace VideoCommon
