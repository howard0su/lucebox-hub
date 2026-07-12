// Runtime helpers for initializing CUDA/HIP ggml backends in one process.

#pragma once

#include "placement_backend.h"

#include "ggml-backend.h"

namespace dflash::common {

PlacementBackend resolve_placement_backend(
    PlacementBackend requested,
    PlacementBackend fallback = compiled_placement_backend());

ggml_backend_t init_placement_backend(
    PlacementBackend backend,
    int gpu,
    const char * role);

}  // namespace dflash::common
