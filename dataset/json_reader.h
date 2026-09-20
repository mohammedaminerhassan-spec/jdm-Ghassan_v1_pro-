#pragma once

// dataset/json_reader.h — the JSON ingestion path for Ghassan AI training.
//
// This is THE data format of the project: big JSON/JSONL corpora
// (conversations, instructions, plain text) -> .gbin training shards.
// No Parquet, no Python, no external dependencies: pure C++.
//
// Accepted layouts (auto-detected per file, mixed layouts allowed in JSONL):
//   1. JSONL: one JSON object per line (best for huge corpora, streams).
//   2. JSON array of objects:  [{...}, {...}]
//   3. Single JSON object:     {...}
// Accepted object schemas (first match wins, case-sensitive keys):
//   chat:        {"messages":[{"role":"user","content":"..."}, ...]}
//                role aliases: human->user, gpt/assistant/ai->assistant,
//                from/value pairs, "conversation"/"turns" arrays.
//   instruction: {"instruction":"...","input":"...","output":"..."}
//                (+ optional "system")
//   prompt:      {"prompt":"...","completion":"..."}  (also "question"/
//                "answer", "problem"/"solution", "instruction"/"response").
//                NOTE: bare {id,question,answer} shards (e.g. the 86-file
//                English corpus) ALSO match here intentionally and train as
//                user->assistant turns with assistant-only loss. They remain
//                loadable by load_qa_json (dataset/retrieval.h) for the BM25
//                index — the two readers share the layout on purpose.
//   text:        {"text":"..."} (also "content","sentence","document",
//                "body","passage","story","article" — configurable)
// Chat/instruction/prompt docs become SFT documents: the pipeline encodes
// them with ChatTemplate so the loss mask supervises ASSISTANT tokens only.
// Plain-text docs become pretraining documents (mask = supervise all).
//
// Safety: bounded nesting depth, per-value byte cap, whole-file size cap,
// strict UTF-8-preserving string parsing with \uXXXX + surrogate support.
// Malformed lines/objects are skipped with a warning, never fatal.

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

// Reads one file (.json array / object, or .jsonl lines) and calls `cb`
// once per extracted document. Returns total documents delivered.
// A single bad line/object never aborts the file.
size_t read_json_docs(const std::string& path, JsonDocCallback cb,
                      const JsonReaderOptions& opts = {});

// Recursively scans `dir` for *.json / *.jsonl (sorted) and reads them all.
// If `dir` is a single file it is read directly. Returns total documents.
size_t read_json_dir(const std::string& dir, JsonDocCallback cb,
                     const JsonReaderOptions& opts = {});

// PRO-EN: يحلل نص JSON واحد (object) إلى JsonDoc بنفس object_to_doc المستعمل
// لمسار الملفات. يستعمله مسار parquet --mode chat لأعمدة messages_json دون
// تكرار الـparser (نفس السكيما، نفس الحدود، نفس skip الصامت للشاذ).
bool doc_from_json_text(const std::string& text, JsonDoc& doc,
                        const JsonReaderOptions& opts = {});

// Convenience: collect plain-text views (chat docs become "user\nassistant"
// joined text) into a vector. For training shards prefer the callbacks.
std::vector<std::string> load_json_texts(const std::string& path,
                                         const JsonReaderOptions& opts = {});

// Prints schema statistics for a file: format, doc count, schema mix,
// longest values. Used by `data_pipeline json-inspect`.
void inspect_json(const std::string& path);

} // namespace gai
