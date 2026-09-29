#pragma once

#include "core/common.h"
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace gai {

bool parquet_available();

std::vector<std::string> list_parquet_files(const std::string& dir_or_file);

struct ParquetOptions {
    size_t max_rows = 0;
    size_t max_value_bytes = 1 << 20;
    bool   verbose = true;
};

struct ParquetTableInfo {
    std::string path;
    int64_t rows = 0;
    std::vector<std::string> columns;
};

using ParquetRowCallback = std::function<void(const std::map<std::string, std::string>&)>;

size_t read_parquet_docs(const std::vector<std::string>& files,
                         ParquetRowCallback cb,
                         const ParquetOptions& opts = {});

ParquetTableInfo inspect_parquet(const std::string& path);

}
