#pragma once

#include "core/common.h"

#include <csignal>

namespace gai {

// Graceful stop for a training run.
//
// WHY THIS EXISTS: a Kaggle session ends at its wall-clock limit, and the
// process is killed wherever it happens to be. With save_every = total/10 that
// threw away up to hundreds of completed steps, every single session, with no
// message. A signal handler cannot do real work (no allocations, no locks), so
// it only sets a flag; the training loop checks it at a step boundary, drains
// the writer and returns normally — last.ckpt on disk, exit code 0-ish, and a
// log line that says exactly what happened.
//
// A SECOND signal gives up on the checkpoint and exits immediately: a hung
// save must not turn a 30s exit into an infinite wait.
namespace signals {

extern volatile std::sig_atomic_t stop_requested;   // 0 none, 1 stop, 2 hard exit

// Async-signal-safe: sets the flag (first signal) or _Exit()s (second one).
// Defined in signals.cpp.
void request_stop(int sig);

// Install handlers for SIGINT/SIGTERM (SIGBREAK on Windows). Idempotent.
void install_stop_handlers();

// Reason to show in the log, or "" if no stop was requested.
const char* stop_reason();

} // namespace signals
} // namespace gai
