// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// AMD FSR 4 Frame Generation through AMD's FidelityFX runtime. The runtime records into our
// command buffer and reads plain Vulkan images, as the FSR 4 upscaler does: the inputs in
// SHADER_READ_ONLY_OPTIMAL, the generated frames in GENERAL. AMD's frame generator is asked to
// leave the swapchain alone (FFX_FRAMEGENERATION_FLAG_NO_SWAPCHAIN_CONTEXT_NOTIFY) and is given
// no swapchain, because the emulator presents the generated frames itself, through Vulkan::FrameGen
// (Generate copies a generated frame into a presented image).
//
// The prepare workload takes the game's depth and motion vectors, which only the Ocarina rasterizer
// has, and which it notes for the frame about to be composited (NoteFsrFrameGenInputs). Frames
// without inputs are simply not generated, and when AMD's generator fails the built-in FSR 3.1
// frame generator is what Vulkan::FrameGen's own fallback logic already asked for.

#include <algorithm>
#include <array>
#include <cmath>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <boost/container/static_vector.hpp>

#include "common/logging.h"
#include "video_core/renderer_vulkan/present/fsr_frame_gen.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/vulkan_common/ffx_files.h"
#include "video_core/vulkan_common/ffx_runtime.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"

// After vulkan.h, which sets the platform macros the wrapper's Win32 entry points need.
#ifndef _WIN32
#define __declspec(x)
#endif
#include "api/include/vk/ffx_api_vk.h"
// AMD's frame generation header carries its own warning pragmas (FFX_PRAGMA_*), which only MSVC's
// preprocessor passes through; clang turns the unknown ones into errors under -Werror=all.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-pragmas"
#endif
#include "framegeneration/include/ffx_framegeneration.h"
#ifdef __clang__
#pragma clang diagnostic pop
#endif
#ifndef _WIN32
#undef __declspec
#endif

namespace Vulkan {

namespace {

/// ffxDispatchDescFrameGeneration's outputs[], which is also the most frames AMD's generator makes
/// from one frame.
constexpr u32 MAX_GENERATED_FRAMES = 4;
constexpr VkImageSubresourceRange COLOR_RANGE{
    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
    .baseMipLevel = 0,
    .levelCount = 1,
    .baseArrayLayer = 0,
    .layerCount = 1,
};
constexpr VkImageSubresourceRange DEPTH_RANGE{
    .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
    .baseMipLevel = 0,
    .levelCount = 1,
    .baseArrayLayer = 0,
    .layerCount = 1,
};
constexpr VkImageUsageFlags OUTPUT_USAGE =
    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
constexpr VkAccessFlags ATTACHMENT_WRITES =
    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
    VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
constexpr VkAccessFlags SHADER_READ = VK_ACCESS_SHADER_READ_BIT;

FfxApiResource Resource(VkImage image, VkFormat format, VkExtent2D size, u32 usage, u32 state) {
    FfxApiResource resource{};
    resource.resource = reinterpret_cast<void*>(image);
    resource.description.type = FFX_API_RESOURCE_TYPE_TEXTURE2D;
    resource.description.format = ffxApiGetSurfaceFormatVK(format);
    resource.description.width = size.width;
    resource.description.height = size.height;
    resource.description.depth = 1;
    resource.description.mipCount = 1;
    resource.description.flags = 0;
    resource.description.usage = usage;
    resource.state = state;
    return resource;
}

[[nodiscard]] VkDeviceContext MakeDeviceContext(const Device& device) {
    return VkDeviceContext{
        .vkDevice = *device.GetLogical(),
        .vkPhysicalDevice = static_cast<VkPhysicalDevice>(device.GetPhysical()),
        .vkDeviceProcAddr = device.GetDispatchLoader().vkGetDeviceProcAddr,
    };
}

VkImageMemoryBarrier Barrier(VkImage image, VkImageSubresourceRange range, VkAccessFlags src_access,
                             VkAccessFlags dst_access, VkImageLayout old_layout,
                             VkImageLayout new_layout) {
    return VkImageMemoryBarrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = src_access,
        .dstAccessMask = dst_access,
        .oldLayout = old_layout,
        .newLayout = new_layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = range,
    };
}

VkImageCopy MakeCopyRegion(VkExtent2D extent) {
    return VkImageCopy{
        .srcSubresource{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
        .srcOffset = {},
        .dstSubresource{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
        .dstOffset = {},
        .extent = {.width = extent.width, .height = extent.height, .depth = 1},
    };
}

FfxApiRect2D FullRect(VkExtent2D extent) {
    return FfxApiRect2D{
        .left = 0,
        .top = 0,
        .width = static_cast<int32_t>(extent.width),
        .height = static_cast<int32_t>(extent.height),
    };
}

/// The camera's world-space axes and position, from the scene's view-projection matrix. That
/// matrix is row-major, and its first two rows are the camera's right and up scaled by the
/// projection's scales, so normalizing them gives the axes; the forward axis is their cross
/// product, and the position follows from the last row (the view's translation).
struct CameraBasis {
    std::array<f32, 3> position{};
    std::array<f32, 3> up{0.0f, 1.0f, 0.0f};
    std::array<f32, 3> right{1.0f, 0.0f, 0.0f};
    std::array<f32, 3> forward{0.0f, 0.0f, -1.0f};
};

[[nodiscard]] f32 Length(const std::array<f32, 3>& v) {
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

[[nodiscard]] std::array<f32, 3> Normalize(const std::array<f32, 3>& v) {
    const f32 length = Length(v);
    if (length < 1e-8f) {
        return {};
    }
    return {v[0] / length, v[1] / length, v[2] / length};
}

[[nodiscard]] std::array<f32, 3> Cross(const std::array<f32, 3>& a, const std::array<f32, 3>& b) {
    return std::array<f32, 3>{a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                              a[0] * b[1] - a[1] * b[0]};
}

[[nodiscard]] f32 Dot(const std::array<f32, 3>& a, const std::array<f32, 3>& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

[[nodiscard]] CameraBasis MakeCameraBasis(const std::array<f32, 16>& view_projection) {
    const std::array<f32, 3> row_right{view_projection[0], view_projection[1],
                                       view_projection[2]};
    const std::array<f32, 3> row_up{view_projection[4], view_projection[5], view_projection[6]};
    const std::array<f32, 3> row_forward{view_projection[8], view_projection[9],
                                         view_projection[10]};
    const std::array<f32, 3> translation{view_projection[12], view_projection[13],
                                         view_projection[14]};
    const std::array<f32, 3> right = Normalize(row_right);
    const std::array<f32, 3> up = Normalize(row_up);
    if (Length(row_right) < 1e-8f || Length(row_up) < 1e-8f) {
        return {};
    }
    // The scene's forward axis points away from the camera, which is the sign of the matrix's own
    // forward row.
    const std::array<f32, 3> axis = Cross(right, up);
    const bool flipped = Dot(axis, row_forward) < 0.0f;
    const std::array<f32, 3> forward = flipped ? Normalize(axis) : Normalize(Cross(up, right));
    CameraBasis basis;
    basis.right = right;
    basis.up = up;
    basis.forward = forward;
    basis.position = {-Dot(translation, right), -Dot(translation, up), -Dot(translation, forward)};
    return basis;
}

struct StagedInputs {
    std::mutex mutex;
    std::optional<FsrFrameGenInputs> inputs;
};

/// The game inputs the Ocarina rasterizer noted for the frame about to be composited.
StagedInputs& Staged() {
    static StagedInputs staged;
    return staged;
}

struct Provider {
    u64 id{};
    std::string name;
};

/// The FSR 4 frame generator AMD's runtime offers for the device, if it offers one. Asked once per
/// session: the runtime and its files don't change while a game runs.
std::optional<Provider> CachedFrameGenerator(const Device& device) {
    static std::mutex mutex;
    static std::optional<Provider> found;
    static bool asked = false;
    std::scoped_lock lock{mutex};
    if (asked) {
        return found;
    }
    asked = true;
    const auto api = FfxRuntime::Acquire();
    if (!api) {
        LOG_INFO(Render_Vulkan,
                 "FSR 4 FG: no FidelityFX runtime; the built-in FSR 3.1 frame generator stays on");
        return found;
    }
    if (!api->configure) {
        LOG_INFO(Render_Vulkan,
                 "FSR 4 FG: the FidelityFX runtime doesn't export ffxConfigure; the built-in FSR "
                 "3.1 frame generator stays on");
        return found;
    }
    VkDeviceContext vk_context = MakeDeviceContext(device);
    u64 count = 0;
    ffxQueryDescGetVersions versions{};
    versions.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
    versions.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
    versions.device = &vk_context;
    versions.outputCount = &count;
    if (api->query(nullptr, &versions.header) != FFX_API_RETURN_OK || count == 0) {
        LOG_INFO(Render_Vulkan, "FSR 4 FG: the FidelityFX runtime offers no frame generator for "
                                "{}; the built-in FSR 3.1 frame generator stays on",
                 device.GetModelName());
        return found;
    }
    std::vector<u64> ids(count);
    std::vector<const char*> names(count);
    versions.versionIds = ids.data();
    versions.versionNames = names.data();
    if (api->query(nullptr, &versions.header) != FFX_API_RETURN_OK) {
        return found;
    }
    std::string offered;
    std::optional<Provider> frame_generator;
    for (u64 i = 0; i < count; ++i) {
        const std::string name = names[i] ? names[i] : "";
        offered += (offered.empty() ? "" : ", ") + name;
        // Newest first: FSR 4 and anything later than it. The FSR 3.1 generator stays with the
        // emulator's own.
        if (!frame_generator && !name.empty() && name[0] >= '4' && name[0] <= '9') {
            frame_generator = Provider{ids[i], name};
        }
    }
    if (!frame_generator) {
        LOG_INFO(Render_Vulkan,
                 "FSR 4 FG: the FidelityFX runtime offers {} for {} ({}); the built-in FSR 3.1 "
                 "frame generator stays on",
                 count, device.GetModelName(), offered);
        return found;
    }
    LOG_INFO(Render_Vulkan, "FSR 4 FG: AMD's frame generator for {}: {}", device.GetModelName(),
             frame_generator->name);
    found = frame_generator;
    return found;
}

}  // Anonymous namespace

bool FsrFrameGen::IsAvailable(const Device& device) {
    return static_cast<bool>(CachedFrameGenerator(device));
}

void NoteFsrFrameGenInputs(const FsrFrameGenInputs& inputs) {
    std::scoped_lock lock{Staged().mutex};
    Staged().inputs = inputs;
}

FsrFrameGen::FsrFrameGen(const Device& device_, MemoryAllocator& memory_allocator_,
                         VkExtent2D display_extent_, VkFormat color_format_)
    : device{device_}, memory_allocator{memory_allocator_}, api{FfxRuntime::Acquire()},
      display_extent{display_extent_}, color_format{color_format_} {
    const auto provider = CachedFrameGenerator(device_);
    if (!api || !api->configure || !provider) {
        throw vk::Exception(VK_ERROR_FEATURE_NOT_PRESENT);
    }
    provider_id = provider->id;
    provider_name = provider->name;
}

FsrFrameGen::~FsrFrameGen() {
    DestroyContext();
}

void FsrFrameGen::DestroyContext() {
    if (context == nullptr) {
        return;
    }
    // Vulkan::FrameGen finishes the scheduler before it drops a frame generator, so anything
    // recorded against the context has run by now.
    ffxContext created = context;
    api->destroy_context(&created, nullptr);
    context = nullptr;
    render_extent = {};
    staged_frames = 0;
}

bool FsrFrameGen::CreateContext(VkExtent2D render_size) {
    DestroyContext();
    const auto provider = CachedFrameGenerator(device);
    if (!provider) {
        return false;
    }
    ffxCreateBackendVKDesc backend{};
    backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
    backend.header.pNext = nullptr;
    backend.deviceContext = MakeDeviceContext(device);
    ffxOverrideVersion version_override{};
    version_override.header.type = FFX_API_DESC_TYPE_OVERRIDE_VERSION;
    version_override.header.pNext = &backend.header;
    version_override.versionId = provider->id;
    // Linked, or the runtime keeps the frame generator at its old API and refuses the prepare
    // workload below.
    ffxCreateContextDescFrameGenerationVersion api_version{};
    api_version.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION_VERSION;
    api_version.header.pNext = &version_override.header;
    api_version.version = FFX_FRAMEGENERATION_VERSION;

    ffxCreateContextDescFrameGeneration create{};
    create.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
    create.header.pNext = &api_version.header;
    // Ocarina's scene is drawn at the render size and upscaled to the display size, so its motion
    // vectors are at the render size; its depth is a standard, finite device depth and its color
    // is tonemapped LDR.
    create.flags = 0;
    create.displaySize = {display_extent.width, display_extent.height};
    create.maxRenderSize = {render_size.width, render_size.height};
    create.backBufferFormat = ffxApiGetSurfaceFormatVK(color_format);
    if (create.backBufferFormat == FFX_API_SURFACE_FORMAT_UNKNOWN) {
        LOG_ERROR(Render_Vulkan,
                  "FSR 4 FG: the presented color format has no FFX surface format; the built-in "
                  "FSR 3.1 frame generator stays on");
        return false;
    }
    ffxContext created{};
    const ffxReturnCode_t result = api->create_context(&created, &create.header, nullptr);
    if (result != FFX_API_RETURN_OK) {
        LOG_ERROR(Render_Vulkan, "FSR 4 FG: ffxCreateContext {}x{} -> {}x{} failed ({}); using the "
                                 "built-in FSR 3.1 frame generator",
                  render_size.width, render_size.height, display_extent.width, display_extent.height,
                  result);
        return false;
    }
    context = created;
    render_extent = render_size;
    LOG_INFO(Render_Vulkan, "FSR 4 FG: AMD's frame generator {} context {}x{} -> {}x{}", provider_name,
             render_size.width, render_size.height, display_extent.width, display_extent.height);

    ffxConfigureDescFrameGeneration configure{};
    configure.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
    configure.header.pNext = nullptr;
    // The emulator presents the generated frames itself, so AMD's generator gets no swapchain and
    // is told to leave one alone; its callbacks are how a swapchain would be driven.
    configure.swapChain = nullptr;
    configure.presentCallback = nullptr;
    configure.presentCallbackUserContext = nullptr;
    configure.frameGenerationCallback = nullptr;
    configure.frameGenerationCallbackUserContext = nullptr;
    configure.frameGenerationEnabled = true;
    configure.allowAsyncWorkloads = false;
    // Ocarina bakes its HUD into the backbuffer, so there is no hudless image to tell it apart.
    configure.HUDLessColor = {};
    configure.flags = FFX_FRAMEGENERATION_FLAG_NO_SWAPCHAIN_CONTEXT_NOTIFY;
    configure.onlyPresentGenerated = false;
    configure.generationRect = FullRect(display_extent);
    configure.frameID = 0;
    const ffxReturnCode_t configured = api->configure(&created, &configure.header);
    if (configured != FFX_API_RETURN_OK) {
        LOG_ERROR(Render_Vulkan, "FSR 4 FG: ffxConfigure failed ({}); using the built-in FSR 3.1 "
                                 "frame generator",
                  configured);
        api->destroy_context(&created, nullptr);
        context = nullptr;
        render_extent = {};
        return false;
    }
    CreateOutputs();
    return true;
}

void FsrFrameGen::CreateOutputs() {
    for (auto& image : output_images) {
        if (image) {
            continue;
        }
        image = memory_allocator.CreateImage(VkImageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_IMAGE_CREATE_EXTENDED_USAGE_BIT,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = color_format,
            .extent = {display_extent.width, display_extent.height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = OUTPUT_USAGE,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        });
    }
}

bool FsrFrameGen::Process(const Device&, Scheduler& scheduler, VkImage color, u32 generations,
                          u64 frame_count) {
    if (failed.load() || generations == 0) {
        return false;
    }
    std::optional<FsrFrameGenInputs> in;
    {
        std::scoped_lock lock{Staged().mutex};
        in.swap(Staged().inputs);
    }
    if (!in || in->render_size.width == 0 || in->render_size.height == 0) {
        return false;
    }
    const VkExtent2D render_size = in->render_size;
    const u32 generated = std::min(generations, MAX_GENERATED_FRAMES);
    if (context == nullptr || render_size.width != render_extent.width ||
        render_size.height != render_extent.height) {
        if (!CreateContext(render_size)) {
            failed = true;
            return false;
        }
    }
    if (!output_images[0]) {
        failed = true;
        return false;
    }
    const CameraBasis basis = MakeCameraBasis(in->camera_view_projection);

    // The workloads are recorded right away, so whether they went in is known here; when they
    // didn't, nothing is staged for Generate and the frame simply is not generated.
    bool recorded = true;
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([this, in, basis, color, generated, frame_count, render_size,
                      &recorded](vk::CommandBuffer cmdbuf) {
        ffxDispatchDescFrameGenerationPrepareV2 prepare{};
        prepare.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE_V2;
        prepare.header.pNext = nullptr;
        prepare.frameID = frame_count;
        prepare.flags = 0;
        prepare.renderSize = {render_size.width, render_size.height};
        prepare.jitterOffset = {in->jitter[0], in->jitter[1]};
        // Ocarina's motion vectors are already in render pixels.
        prepare.motionVectorScale = {1.0f, 1.0f};
        prepare.frameTimeDelta = in->frame_time_ms;
        prepare.reset = in->reset;
        prepare.cameraNear = in->camera_near;
        prepare.cameraFar = in->camera_far;
        prepare.cameraFovAngleVertical = in->vertical_fov;
        prepare.viewSpaceToMetersFactor = 1.0f;
        prepare.depth = Resource(in->depth, in->depth_format, render_size,
                                 FFX_API_RESOURCE_USAGE_DEPTHTARGET,
                                 FFX_API_RESOURCE_STATE_COMPUTE_READ);
        prepare.motionVectors =
            Resource(in->motion_vectors, in->motion_format, render_size,
                     FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_COMPUTE_READ);
        for (u32 i = 0; i < 3; ++i) {
            prepare.cameraPosition[i] = basis.position[i];
            prepare.cameraUp[i] = basis.up[i];
            prepare.cameraRight[i] = basis.right[i];
            prepare.cameraForward[i] = basis.forward[i];
        }

        boost::container::static_vector<VkImageMemoryBarrier, MAX_GENERATED_FRAMES + 3> to_read{
            Barrier(color, COLOR_RANGE, ATTACHMENT_WRITES, SHADER_READ, VK_IMAGE_LAYOUT_GENERAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
            Barrier(in->depth, DEPTH_RANGE, ATTACHMENT_WRITES, SHADER_READ, VK_IMAGE_LAYOUT_GENERAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
            Barrier(in->motion_vectors, COLOR_RANGE, ATTACHMENT_WRITES, SHADER_READ,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
        };
        for (u32 i = 0; i < generated; ++i) {
            to_read.push_back(
                Barrier(*output_images[i], COLOR_RANGE, SHADER_READ,
                        SHADER_READ | VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL,
                        VK_IMAGE_LAYOUT_GENERAL));
        }
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, {}, {}, to_read);

        ffxContext created = context;
        ffxDispatchDescFrameGenerationPrepareV2 recorded_prepare = prepare;
        recorded_prepare.commandList = static_cast<void*>(*cmdbuf);
        if (const ffxReturnCode_t result = api->dispatch(&created, &recorded_prepare.header);
            result != FFX_API_RETURN_OK) {
            if (!failed.exchange(true)) {
                LOG_ERROR(Render_Vulkan, "FSR 4 FG: ffxDispatch for frame {}'s prepare failed ({}); "
                                         "the frame is not generated",
                          frame_count, result);
            }
            recorded = false;
            return;
        }

        ffxDispatchDescFrameGeneration dispatch{};
        dispatch.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION;
        dispatch.header.pNext = nullptr;
        dispatch.commandList = static_cast<void*>(*cmdbuf);
        // The frame as it is presented, at the display size. AMD's generator interpolates between
        // it and the previous frame's, which it keeps to itself.
        dispatch.presentColor = Resource(color, color_format, display_extent,
                                         FFX_API_RESOURCE_USAGE_READ_ONLY,
                                         FFX_API_RESOURCE_STATE_COMPUTE_READ);
        for (u32 i = 0; i < generated; ++i) {
            dispatch.outputs[i] =
                Resource(*output_images[i], color_format, display_extent, FFX_API_RESOURCE_USAGE_UAV,
                         FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        dispatch.numGeneratedFrames = generated;
        dispatch.reset = in->reset;
        // Ocarina's scene is tonemapped LDR, so the backbuffer reads as sRGB and there is nothing
        // to scale in nits.
        dispatch.backbufferTransferFunction = FFX_API_BACKBUFFER_TRANSFER_FUNCTION_SRGB;
        dispatch.minMaxLuminance[0] = 1.0f;
        dispatch.minMaxLuminance[1] = 1.0f;
        dispatch.generationRect = FullRect(display_extent);
        dispatch.frameID = frame_count;
        if (const ffxReturnCode_t result = api->dispatch(&created, &dispatch.header);
            result != FFX_API_RETURN_OK) {
            if (!failed.exchange(true)) {
                LOG_ERROR(Render_Vulkan, "FSR 4 FG: ffxDispatch for frame {} failed ({}); the frame is "
                                         "not generated",
                          frame_count, result);
            }
            recorded = false;
            return;
        }

        boost::container::static_vector<VkImageMemoryBarrier, MAX_GENERATED_FRAMES + 3> back{
            Barrier(color, COLOR_RANGE, SHADER_READ, ATTACHMENT_WRITES,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
            Barrier(in->depth, DEPTH_RANGE, SHADER_READ, ATTACHMENT_WRITES,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
            Barrier(in->motion_vectors, COLOR_RANGE, SHADER_READ, ATTACHMENT_WRITES,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
        };
        for (u32 i = 0; i < generated; ++i) {
            back.push_back(Barrier(*output_images[i], COLOR_RANGE,
                                   SHADER_READ | VK_ACCESS_SHADER_WRITE_BIT, SHADER_READ,
                                   VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL));
        }
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, {}, {}, back);
    });
    staged_frames = recorded ? generated : 0;
    return recorded;
}

void FsrFrameGen::Generate(const Device&, Scheduler& scheduler, VkImage destination,
                           u32 generation) const {
    if (generation >= staged_frames || !output_images[generation]) {
        return;
    }
    const VkImage source = *output_images[generation];
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([source, destination, extent = display_extent](vk::CommandBuffer cmdbuf) {
        const std::array before{
            Barrier(source, COLOR_RANGE, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL),
            Barrier(destination, COLOR_RANGE, ATTACHMENT_WRITES, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL),
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               0, {}, {}, before);
        cmdbuf.CopyImage(source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, destination,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, MakeCopyRegion(extent));
        const std::array after{
            Barrier(source, COLOR_RANGE, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
            Barrier(destination, COLOR_RANGE, VK_ACCESS_TRANSFER_WRITE_BIT, ATTACHMENT_WRITES,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, {}, {}, after);
    });
}

}  // namespace Vulkan
