// Gemma4 daemon entry point: loads model, creates backend, runs daemon loop.

#include "gemma4_daemon.h"
#include "gemma4_backend.h"
#include "common/daemon_loop.h"

#include <cstdio>

namespace dflash27b {

int run_gemma4_daemon(const Gemma4DaemonArgs & args) {
    Gemma4BackendConfig cfg;
    cfg.model_path = args.model_path;
    cfg.max_ctx    = args.max_ctx;
    cfg.stream_fd  = args.stream_fd;
    cfg.chunk      = args.chunk;
    cfg.gpu        = args.gpu;

    Gemma4Backend backend(cfg);
    if (!backend.init()) {
        std::fprintf(stderr, "[gemma4-daemon] init failed\n");
        return 1;
    }

    DaemonLoopArgs da;
    da.stream_fd = args.stream_fd;
    return run_daemon(backend, da);
}

}  // namespace dflash27b
