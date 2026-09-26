#include "core/signals.h"

namespace gai {
namespace signals {

volatile std::sig_atomic_t stop_requested = 0;
static char g_reason[64] = {0};

void request_stop(int sig) {
    if (stop_requested == 0) {
        stop_requested = 1;
        // strncpy is not async-signal-safe; copy the few bytes by hand.
        const char* name = (sig == SIGINT) ? "SIGINT" : "SIGTERM";
        size_t i = 0;
        for (; name[i] && i + 1 < sizeof(g_reason); ++i) g_reason[i] = name[i];
        g_reason[i] = '\0';
    } else {
        // Second signal: exit now. async-signal-safe only (no atexit, no flush).
        std::_Exit(130);
    }
}

void install_stop_handlers() {
    std::signal(SIGINT, request_stop);
    std::signal(SIGTERM, request_stop);
#ifdef SIGBREAK
    std::signal(SIGBREAK, request_stop);
#endif
}

const char* stop_reason() {
    return stop_requested ? g_reason : "";
}

} // namespace signals
} // namespace gai
