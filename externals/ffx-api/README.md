# AMD FidelityFX API headers

The FFX API headers from AMD's FidelityFX SDK 2.4.0, used by
`src/video_core/renderer_vulkan/vk_fsr4_upscaler.cpp` to call AMD's FSR 4 runtime and
`src/video_core/renderer_vulkan/present/fsr_frame_gen.cpp` to call its FSR 4 Frame Generation.

- Source: https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK, tag `v2.4.0`
  (commit `7539ae8b6568ad29ed78a753404351daf3b5cca7`).
- Files, unmodified: `Kits/FidelityFX/api/include/ffx_api.h`, `ffx_api_types.h`,
  `vk/ffx_api_vk.h`, `Kits/FidelityFX/upscalers/include/ffx_upscale.h`,
  `Kits/FidelityFX/framegeneration/include/ffx_framegeneration.h` and
  `ffx_framegeneration_api_types.h`. The SDK's `framegeneration/include/vk/` header is not vendored:
  it only declares the Frame Generation Swapchain entry points, which the emulator doesn't call.
- Licence: MIT (the SDK's `Kits/FidelityFX/docs/license.md` lists these files under its MIT
  terms; the notice is in each header).

AMD's signed runtime DLLs are not part of this repository or of any Lichii package. Users
download them from the same SDK commit in the Graphics settings; see
`src/video_core/vulkan_common/ffx_files.h`.
