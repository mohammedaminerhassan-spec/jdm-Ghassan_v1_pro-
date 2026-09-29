#pragma once

#include "core/common.h"

#include <csignal>

namespace gai {

namespace signals {

extern volatile std::sig_atomic_t stop_requested;

void request_stop(int sig);

void install_stop_handlers();

const char* stop_reason();

}
}
