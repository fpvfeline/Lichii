// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// AMD FSR 4 Frame Generation through AMD's FidelityFX runtime (vulkan_common/ffx_runtime.h): the
// frame generator from the same SDK release FSR 4 upscaling comes from (FSR 4.0.2). AMD ships it
// only as a signed dll, so the FFX API calls drive it the way the FSR 4 upscaler does. The
// generated frames are presented by Vulkan::FrameGen, and the frame generator is told to leave
// the swapchain alone, so nothing here depends on AMD's swapchain. Only Ocarina of Time gets this
// path: it is the only game the emulator has depth and motion vectors for, and it is how the
// NVIDIA DLSS Frame Generation path is gated too. When the runtime isn't there or doesn't offer
// frame generation, Vulkan::FrameGen keeps its built-in FSR 3.1 frame generator.

#pragma once

#include <array>
#include <memory>
#include <string>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;
class MemoryAllocator;
class Scheduler;

namespace FfxRuntime {
struct Api;
}

/// The frame inputs FSR 4 Frame Generation needs, which only the Ocarina rasterizer has. It notes
/// them for the frame about to be composited (NoteFsrFrameGenInputs); Vulkan::FrameGen takes them
/// in FsrFrameGen::Process.
struct FsrFrameGenInputs {
    VkImage depth{};
    VkFormat depth_format{VK_FORMAT_UNDEFINED};
    VkImage motion_vectors{};
    VkFormat motion_format{VK_FORMAT_UNDEFINED};
    VkExtent2D render_size{};
    std::array<f32, 16> camera_view_projection{};  ///< The scene's view-projection matrix.
    std::array<f32, 2> jitter{};   ///< The offset the scene was drawn with, in render pixels.
    f32 frame_time_ms{};
    f32 camera_near{0.1f};
    f32 camera_far{25000.0f};
    f32 vertical_fov{};  ///< In radians.
    bool reset{};        ///< The camera jumped: drop the previous frame's history.
};

/// Hands the inputs above to the frame generator. Overwritten by the next frame, so a frame
/// without inputs simply isn't generated.
void NoteFsrFrameGenInputs(const FsrFrameGenInputs& inputs);

class FsrFrameGen {
public:
    /// Whether AMD's runtime offers FSR 4 Frame Generation for the device. Logged once, the answer
    /// is the same for the rest of the session.
    [[nodiscard]] static bool IsAvailable(const Device& device);

    /// Throws vk::Exception when the runtime offers no FSR 4 Frame Generation: the caller keeps
    /// its built-in FSR 3.1 frame generator from then on.
    explicit FsrFrameGen(const Device& device, MemoryAllocator& memory_allocator,
                         VkExtent2D display_extent, VkFormat color_format);
    ~FsrFrameGen();

    /// Records the prepare and the generation workloads for one composited frame. `color` is the
    /// frame as it is presented (the display size); the rest of the inputs were noted for it.
    /// False when there are no inputs for the frame or AMD's generator fails; then the frame is
    /// simply not generated.
    bool Process(const Device& device, Scheduler& scheduler, VkImage color, u32 generations,
                 u64 frame_count);

    /// Copies one of the generated frames into a presented image.
    void Generate(const Device& device, Scheduler& scheduler, VkImage destination,
                  u32 generation) const;

    [[nodiscard]] bool Failed() const noexcept {
        return failed.load();
    }

private:
    void DestroyContext();
    bool CreateContext(VkExtent2D render_size);
    void CreateOutputs();

    const Device& device;
    MemoryAllocator& memory_allocator;
    std::shared_ptr<const FfxRuntime::Api> api;
    u64 provider_id{};
    std::string provider_name;
    VkExtent2D display_extent{};
    VkFormat color_format{VK_FORMAT_UNDEFINED};
    void* context{};  ///< ffxContext
    VkExtent2D render_extent{};
    std::array<vk::Image, 4> output_images{};
    u32 staged_frames{};  ///< The output_images that hold a generated frame.
    std::atomic<bool> failed{};
};

} // namespace Vulkan
