#pragma once

// dataset/json_reader.h — PARQUET-ONLY project: JSON file ingestion REMOVED.
//
// The prebuilt Hermes lake is stored as
// english_parquet/english_chat_part*.parquet +
// english_parquet/english_instruction_part*.parquet and training reads ONLY
// that lake via dataset/parquet_reader.h.
//
// This header keeps the MINIMAL chat-document types + the single
// messages_json parser used by the parquet chat route. File-based JSON
// ingestion (read_json_docs / read_json_dir / load_json_texts /
// inspect_json) was DELETED on purpose: a second ingestion path silently
// diverges (the old "conversations" plural key dropped 100% of Hermes
// docs) and wastes T4 hours. Use the Parquet lake, never JSON files.

#include "core/common.h"
#include "tokenizer/chat_template.h"
#include <functional>
#include <string>
#include <vector>

namespace gai {

// ---------------------------------------------------------------- document
struct JsonDoc {
    bool is_chat = false;
    std::string text;                    // plain-text doc (is_chat == false)
    std::vector<Message> messages;       // chat doc (is_chat == true)
};

struct JsonReaderOptions {
    // Ordered text keys to try for plain-text objects (first hit wins).
    std::vector<std::string> text_keys = {
        "text", "content", "sentence", "document", "body",
        "passage", "story", "article", "input"
    };
    size_t max_value_bytes = 1 << 20;    // 1 MiB: longer strings are truncated
    size_t max_docs = 0;                 // 0 = unlimited
    bool   verbose = true;
};

using JsonDocCallback = std::function<void(const JsonDoc&)>;

// PARQUET-ONLY: file-based JSON ingestion is REMOVED (was read_json_docs /
// read_json_dir / load_json_texts / inspect_json). Any call is a recipe bug:
// The prebuilt lake is consumed with
//   data_pipeline parquet --mode chat --lake english_parquet ...
// Keeping the old path would re-introduce the silent-drop divergence.

// Parses ONE {"messages":[...]} object text (parquet messages_json column)
// into a JsonDoc with the same schema mapping + caps as before. Returns
// false on malformed text (caller counts it as skipped, never fatal).
bool doc_from_json_text(const std::string& text, JsonDoc& doc,
                        const JsonReaderOptions& opts = {});

} // namespace gai
