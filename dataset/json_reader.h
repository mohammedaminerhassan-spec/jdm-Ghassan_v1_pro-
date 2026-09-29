#pragma once

#include "core/common.h"
#include "tokenizer/chat_template.h"
#include <functional>
#include <string>
#include <vector>

namespace gai {

struct JsonDoc {
    bool is_chat = false;
    std::string text;
    std::vector<Message> messages;
};

struct JsonReaderOptions {

    std::vector<std::string> text_keys = {
        "text", "content", "sentence", "document", "body",
        "passage", "story", "article", "input"
    };
    size_t max_value_bytes = 1 << 20;
    size_t max_docs = 0;
    bool   verbose = true;
};

using JsonDocCallback = std::function<void(const JsonDoc&)>;

bool doc_from_json_text(const std::string& text, JsonDoc& doc,
                        const JsonReaderOptions& opts = {});

}
