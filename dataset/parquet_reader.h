#pragma once

// dataset/parquet_reader.h — native Apache Parquet input (THE lake input).
//
// The default build has NO parquet support: rebuild with
// -DGAI_ENABLE_PARQUET=ON plus Apache Arrow C++ to enable:
//   data_pipeline parquet --lake english_parquet --mode chat --domain english_chat ...
// (kaggle/setup.sh --with-parquet installs Arrow and passes the flag.)
// There is no JSON file route: the Hermes lake is the only training input.
//
// Contract (skips, never crashes):
//   * recursive .parquet discovery, sorted (deterministic order)
//   * row-group streaming (bounded RAM, never whole-file materialization)
//   * only UTF-8 string columns are read (String/Binary/LargeString +
//     dictionary-encoded strings, i.e. the supplied lake format).
//     Anything else is skipped with a warning, never fatal.
//   * per-value byte cap + row cap, malformed rows skipped loudly
// Narrow on purpose: this reads OUR lake files, not arbitrary parquet.

#include "core/common.h"
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace gai {

// True only when built with -DGAI_ENABLE_PARQUET=ON and Arrow was found.
// data_pipeline uses it for --probe and for route selection in scripts.
bool parquet_available();

// Sorted recursive discovery of *.parquet under a dir (or the single file).
// Needs no Arrow dependency (filesystem only).
std::vector<std::string> list_parquet_files(const std::string& dir_or_file);

struct ParquetOptions {
    size_t max_rows = 0;                 // 0 = unlimited
    size_t max_value_bytes = 1 << 20;    // 1 MiB: longer strings are truncated
    bool   verbose = true;
};

struct ParquetTableInfo {
    std::string path;
    int64_t rows = 0;
    std::vector<std::string> columns;
};

// One decoded row: column name -> UTF-8 string value (missing/null -> "").
using ParquetRowCallback = std::function<void(const std::map<std::string, std::string>&)>;

// Stream every row of every file through cb. Returns rows delivered.
// Without the Arrow backend this fails LOUDLY (never silently empty).
size_t read_parquet_docs(const std::vector<std::string>& files,
                         ParquetRowCallback cb,
                         const ParquetOptions& opts = {});

// Header-only table peek (rows + column names). Without Arrow: rows = -1.
ParquetTableInfo inspect_parquet(const std::string& path);

} // namespace gai
