// Gemma4 daemon entry point.

#pragma once

#include <string>

namespace dflash27b {

struct Gemma4DaemonArgs {
    const char * model_path = nullptr;
    int          max_ctx    = 8192;
    int          stream_fd  = -1;
    int          chunk      = 512;
    int          gpu        = 0;
};

int run_gemma4_daemon(const Gemma4DaemonArgs & args);

}  // namespace dflash27b
