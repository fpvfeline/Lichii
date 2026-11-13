// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Which frame generation runs this session when NVIDIA DLSS Frame Generation is the chosen type.

#pragma once

#include "common/common_types.h"
#include "common/settings_enums.h"

namespace VideoCore {

/// The Legend of Zelda: Ocarina of Time. Its DLSS evaluation hands DLSS Frame Generation the
/// depth, the motion vectors and the camera (vk_ocarina_camera_motion.cpp); no other game does yet.
constexpr u64 OCARINA_PROGRAM_ID = 0x0400486024DEC000ULL;

/// Why NVIDIA DLSS Frame Generation can't run this session.
enum class DlssgBlock : u8 {
    None,
    NoStreamline, ///< Built without NVIDIA Streamline (Windows x64 builds have it)
    DlssOff,      ///< "Use NVIDIA DLSS" is off, and DLSS-G's inputs come from the DLSS evaluation
    NoGameInputs, ///< The game hands no depth, motion vectors and camera
    NotLoaded,    ///< Streamline could not load DLSS-G or turned it off (scRGB output); it logs why
};

struct DlssgSession {
    bool streamline;  ///< Built with NVIDIA Streamline
    bool dlss;        ///< "Use NVIDIA DLSS" is on
    u64 program_id;   ///< The running game
    bool loaded;      ///< Streamline attached DLSS-G and nothing turned it off since
};

[[nodiscard]] constexpr DlssgBlock BlockedDlssg(const DlssgSession& session) {
    if (!session.streamline) {
        return DlssgBlock::NoStreamline;
    }
    if (!session.dlss) {
        return DlssgBlock::DlssOff;
    }
    if (session.program_id != OCARINA_PROGRAM_ID) {
        return DlssgBlock::NoGameInputs;
    }
    if (!session.loaded) {
        return DlssgBlock::NotLoaded;
    }
    return DlssgBlock::None;
}

/// Why AMD FSR 4 Frame Generation can't run this session.
enum class Fsr4Block : u8 {
    None,
    NotOcarina, ///< No other game hands FSR 4 Frame Generation the depth, vectors and camera
    NoRuntime,  ///< AMD's FidelityFX runtime is missing, or offers no FSR 4 generator for the GPU
};

struct Fsr4Session {
    u64 program_id; ///< The running game
    bool runtime;   ///< The FFX runtime is loaded and offers an FSR 4 frame generator
};

[[nodiscard]] constexpr Fsr4Block BlockedFsr4(const Fsr4Session& session) {
    if (session.program_id != OCARINA_PROGRAM_ID) {
        return Fsr4Block::NotOcarina;
    }
    if (!session.runtime) {
        return Fsr4Block::NoRuntime;
    }
    return Fsr4Block::None;
}

/// The frame generation that runs: the chosen type, except that NVIDIA DLSS Frame Generation and
/// AMD FSR 4 Frame Generation give way to the built-in optical flow interpolation (Fsr3) while the
/// one the user chose is blocked.
[[nodiscard]] constexpr Settings::FrameGenType ActiveFrameGenType(Settings::FrameGenType chosen,
                                                                  DlssgBlock dlssg,
                                                                  Fsr4Block fsr4) {
    if (chosen == Settings::FrameGenType::NvidiaDlssG && dlssg != DlssgBlock::None) {
        return Settings::FrameGenType::Fsr3;
    }
    if (chosen == Settings::FrameGenType::Fsr4 && fsr4 != Fsr4Block::None) {
        return Settings::FrameGenType::Fsr3;
    }
    return chosen;
}

} // namespace VideoCore
