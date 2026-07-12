#include "placement_backend_runtime.h"

#include "ggml-cuda.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <mutex>
#include <string>

namespace dflash::common {

namespace {

std::string lower_name(const char * value) {
    std::string out = value ? value : "";
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return out;
}

bool reg_matches_backend(ggml_backend_reg_t reg, PlacementBackend backend) {
    const std::string name = lower_name(ggml_backend_reg_name(reg));
    switch (backend) {
    case PlacementBackend::Cuda:
        return name == "cuda";
    case PlacementBackend::Hip:
        return name == "hip" || name == "rocm";
    case PlacementBackend::Auto:
        return false;
    }
    return false;
}

void load_dynamic_backends_once() {
    static std::once_flag once;
    std::call_once(once, []() {
        ggml_backend_load_all();
    });
}

ggml_backend_t init_dynamic_backend(PlacementBackend backend, int gpu) {
    load_dynamic_backends_once();
    for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        if (!reg_matches_backend(reg, backend)) {
            continue;
        }
        const size_t ndev = ggml_backend_reg_dev_count(reg);
        if (gpu < 0 || (size_t)gpu >= ndev) {
            std::fprintf(stderr,
                         "[placement] %s backend has %zu devices; gpu=%d is out of range\n",
                         placement_backend_name(backend), ndev, gpu);
            return nullptr;
        }
        return ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, (size_t)gpu), nullptr);
    }
    std::fprintf(stderr,
                 "[placement] %s backend is not registered; put libggml-%s.so next to the server or set the ggml backend search path\n",
                 placement_backend_name(backend),
                 backend == PlacementBackend::Hip ? "hip" : "cuda");
    return nullptr;
}

}  // namespace

PlacementBackend resolve_placement_backend(
    PlacementBackend requested,
    PlacementBackend fallback) {
    return requested == PlacementBackend::Auto ? fallback : requested;
}

ggml_backend_t init_placement_backend(
    PlacementBackend backend,
    int gpu,
    const char * role) {
    backend = resolve_placement_backend(backend);
    gpu = std::max(0, gpu);

    ggml_backend_t result = nullptr;
    if (backend == compiled_placement_backend()) {
        result = ggml_backend_cuda_init(gpu);
    } else {
        result = init_dynamic_backend(backend, gpu);
    }

    if (!result) {
        std::fprintf(stderr, "%s %s init failed gpu=%d\n",
                     role ? role : "backend",
                     placement_backend_name(backend),
                     gpu);
    }
    return result;
}

}  // namespace dflash::common
