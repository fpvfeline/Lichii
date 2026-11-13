// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "video_core/frame_gen_choice.h"

namespace {

using Settings::FrameGenType;
using VideoCore::ActiveFrameGenType;
using VideoCore::BlockedDlssg;
using VideoCore::BlockedFsr4;
using VideoCore::DlssgBlock;
using VideoCore::DlssgSession;
using VideoCore::Fsr4Block;
using VideoCore::Fsr4Session;

constexpr DlssgSession OCARINA{
    .streamline = true,
    .dlss = true,
    .program_id = VideoCore::OCARINA_PROGRAM_ID,
    .loaded = true,
};

constexpr Fsr4Session OCARINA_FSR4{
    .program_id = VideoCore::OCARINA_PROGRAM_ID,
    .runtime = true,
};

} // Anonymous namespace

TEST_CASE("DLSS Frame Generation runs only for a game with inputs, else FSR3", "[video_core]") {
    REQUIRE(BlockedDlssg(OCARINA) == DlssgBlock::None);
    REQUIRE(ActiveFrameGenType(FrameGenType::NvidiaDlssG, DlssgBlock::None) ==
            FrameGenType::NvidiaDlssG);

    DlssgSession fire_emblem = OCARINA;
    fire_emblem.program_id = 0x04007EF01F8AE000ULL;
    fire_emblem.loaded = false;
    REQUIRE(BlockedDlssg(fire_emblem) == DlssgBlock::NoGameInputs);

    DlssgSession no_dlss = OCARINA;
    no_dlss.dlss = false;
    REQUIRE(BlockedDlssg(no_dlss) == DlssgBlock::DlssOff);

    DlssgSession not_loaded = OCARINA;
    not_loaded.loaded = false;
    REQUIRE(BlockedDlssg(not_loaded) == DlssgBlock::NotLoaded);

    DlssgSession linux_build = OCARINA;
    linux_build.streamline = false;
    REQUIRE(BlockedDlssg(linux_build) == DlssgBlock::NoStreamline);

    for (const DlssgBlock block : {DlssgBlock::NoStreamline, DlssgBlock::DlssOff,
                                   DlssgBlock::NoGameInputs, DlssgBlock::NotLoaded}) {
        REQUIRE(ActiveFrameGenType(FrameGenType::NvidiaDlssG, block, Fsr4Block::None) ==
                FrameGenType::Fsr3);
    }
    REQUIRE(ActiveFrameGenType(FrameGenType::Lsfg, DlssgBlock::NotLoaded, Fsr4Block::None) ==
            FrameGenType::Lsfg);
    REQUIRE(ActiveFrameGenType(FrameGenType::Fsr3, DlssgBlock::None, Fsr4Block::None) ==
            FrameGenType::Fsr3);
}

TEST_CASE("FSR 4 Frame Generation runs only for Ocarina with an FFX runtime, else FSR3",
          "[video_core]") {
    REQUIRE(BlockedFsr4(OCARINA_FSR4) == Fsr4Block::None);
    REQUIRE(ActiveFrameGenType(FrameGenType::Fsr4, DlssgBlock::NoStreamline, Fsr4Block::None) ==
            FrameGenType::Fsr4);

    Fsr4Session fire_emblem = OCARINA_FSR4;
    fire_emblem.program_id = 0x04007EF01F8AE000ULL;
    REQUIRE(BlockedFsr4(fire_emblem) == Fsr4Block::NotOcarina);

    Fsr4Session no_runtime = OCARINA_FSR4;
    no_runtime.runtime = false;
    REQUIRE(BlockedFsr4(no_runtime) == Fsr4Block::NoRuntime);

    for (const Fsr4Block block : {Fsr4Block::NotOcarina, Fsr4Block::NoRuntime}) {
        REQUIRE(ActiveFrameGenType(FrameGenType::Fsr4, DlssgBlock::None, block) ==
                FrameGenType::Fsr3);
    }
    REQUIRE(ActiveFrameGenType(FrameGenType::NvidiaDlssG, DlssgBlock::None,
                               Fsr4Block::NoRuntime) == FrameGenType::NvidiaDlssG);
    REQUIRE(ActiveFrameGenType(FrameGenType::Lsfg, DlssgBlock::None,
                               Fsr4Block::NoRuntime) == FrameGenType::Lsfg);
}
