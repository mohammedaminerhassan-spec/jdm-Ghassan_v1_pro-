// dataset/parquet_reader.cpp — see parquet_reader.h for the contract.
#include "dataset/parquet_reader.h"

#include <algorithm>
#include <filesystem>
#include <string>

// Third-party headers MUST live at global scope: including Arrow inside
// `namespace gai` (as this file once did) drags <bitset>/<string>/etc. into
// gai:: scope and GCC 11 + Arrow 25 fails with
// "'__throw_out_of_range_fmt' was not declared in this scope" plus cascading
// unused-function errors. Global scope keeps ::arrow / ::parquet / ::std exact.
#ifdef GAI_PARQUET
#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#endif

namespace fs = std::filesystem;

namespace gai {

std::vector<std::string> list_parquet_files(const std::string& dir_or_file) {
    std::vector<std::string> out;
    std::error_code ec;
    if (fs::is_regular_file(dir_or_file, ec)) {
        out.push_back(dir_or_file);
    } else {
        for (const auto& e : fs::recursive_directory_iterator(dir_or_file, ec)) {
            if (!e.is_regular_file()) continue;
            std::string ext = e.path().extension().string();
            for (char& ch : ext) ch = static_cast<char>(::tolower(static_cast<unsigned char>(ch)));
            if (ext == ".parquet") out.push_back(e.path().string());
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

#ifdef GAI_PARQUET

// Decode one chunk to UTF-8 strings. Handles the encodings our factory
// emits (PLAIN + RLE_DICTIONARY over BYTE_ARRAY): plain string/binary
// arrays read directly, dictionary arrays go through compute::Cast to utf8.
// Anything else -> empty column + one warning (loud skip, never fatal).
static bool chunk_to_strings(const arrow::Array& arr, std::vector<std::string>& out,
                             size_t cap, const std::string& file, const std::string& col) {
    const arrow::Array* a = &arr;
    std::shared_ptr<arrow::Array> held;  // keeps cast results alive
    if (a->type_id() == arrow::Type::DICTIONARY) {
        // Our lake files are dictionary-encoded strings (pyarrow default).
        // CastOptions with to_type set is the version-stable spelling
        // (avoids Safe()/Unsafe() factory drift across Arrow releases).
        arrow::compute::CastOptions opts;
        opts.to_type = arrow::utf8();
        auto res = arrow::compute::Cast(arrow::Datum(a), opts);
        if (!res.ok()) {
            log_warn("parquet: cannot decode dictionary column '" + col + "' in " + file);
            return false;
        }
        // Cast of a whole Array yields an Array datum (not chunked).
        held = res.ValueOrDie().make_array();
        a = held.get();
    }
    // NOTE: LargeStringArray is NOT a BinaryArray (int64 offsets), so the
    // two layouts need separate accessors. GetString(int64)->std::string is
    // the long-standing accessor on both classes.
    auto push_capped = [&](std::string s) {
        if (s.size() > cap) s.resize(cap);  // truncate, keep streaming
        out.push_back(std::move(s));
    };
    if (a->type_id() == arrow::Type::STRING || a->type_id() == arrow::Type::BINARY) {
        const auto& bin = static_cast<const arrow::BinaryArray&>(*a);
        out.reserve(out.size() + static_cast<size_t>(bin.length()));
        for (int64_t i = 0; i < bin.length(); ++i)
            push_capped(bin.IsNull(i) ? std::string() : bin.GetString(i));
        return true;
    }
    if (a->type_id() == arrow::Type::LARGE_STRING || a->type_id() == arrow::Type::LARGE_BINARY) {
        const auto& bin = static_cast<const arrow::LargeBinaryArray&>(*a);
        out.reserve(out.size() + static_cast<size_t>(bin.length()));
        for (int64_t i = 0; i < bin.length(); ++i)
            push_capped(bin.IsNull(i) ? std::string() : bin.GetString(i));
        return true;
    }
    log_warn("parquet: column '" + col + "' in " + file + " is not a string column; skipped");
    return false;
}

bool parquet_available() { return true; }

ParquetTableInfo inspect_parquet(const std::string& path) {
    ParquetTableInfo info;
    info.path = path;
    info.rows = -1;
    auto maybe_file = arrow::io::ReadableFile::Open(path);
    if (!maybe_file.ok()) return info;
    std::shared_ptr<arrow::io::RandomAccessFile> file = maybe_file.ValueOrDie();
    // Result-based OpenFile: the Status out-param overload was REMOVED in
    // Arrow 25 (and the old num_rows()/ReadRowGroup-out-param went with it).
    // The Result spelling below compiles on old AND new Arrow alike.
    auto maybe_reader = parquet::arrow::OpenFile(file);
    if (!maybe_reader.ok()) return info;
    std::unique_ptr<parquet::arrow::FileReader> reader = std::move(maybe_reader).ValueOrDie();
    std::shared_ptr<arrow::Schema> schema;
    if (!reader->GetSchema(&schema).ok()) return info;
    for (int i = 0; i < schema->num_fields(); ++i)
        info.columns.push_back(schema->field(i)->name());
    info.rows = reader->parquet_reader()->num_rows();
    return info;
}

size_t read_parquet_docs(const std::vector<std::string>& files,
                         ParquetRowCallback cb,
                         const ParquetOptions& opts) {
    size_t delivered = 0;
    for (const auto& path : files) {
        if (opts.max_rows > 0 && static_cast<size_t>(delivered) >= opts.max_rows) break;
        auto maybe_file = arrow::io::ReadableFile::Open(path);
        if (!maybe_file.ok()) {
            log_warn("parquet: cannot open " + path + "; skipped");
            continue;
        }
        std::shared_ptr<arrow::io::RandomAccessFile> file = maybe_file.ValueOrDie();
        auto maybe_reader = parquet::arrow::OpenFile(file);
        if (!maybe_reader.ok()) {
            log_warn("parquet: cannot read footer of " + path + "; skipped");
            continue;
        }
        std::unique_ptr<parquet::arrow::FileReader> reader = std::move(maybe_reader).ValueOrDie();
        std::shared_ptr<arrow::Schema> schema;
        if (!reader->GetSchema(&schema).ok() || schema->num_fields() == 0) {
            log_warn("parquet: no readable schema in " + path + "; skipped");
            continue;
        }
        std::vector<std::string> cols;
        for (int i = 0; i < schema->num_fields(); ++i) cols.push_back(schema->field(i)->name());
        // Row counts via the embedded ParquetFileReader: FileReader::num_rows()
        // was removed in Arrow 25, this spelling is version-stable.
        const int64_t file_rows = reader->parquet_reader()->num_rows();
        const int file_groups = reader->parquet_reader()->num_row_groups();
        if (opts.verbose) {
            std::string cl;
            for (size_t i = 0; i < cols.size(); ++i) cl += (i ? "," : "") + cols[i];
            log_info(strfmt("[parquet] %s: %lld rows x %d cols (%s)",
                            path.c_str(), static_cast<long long>(file_rows),
                            static_cast<int>(cols.size()), cl.c_str()));
        }
        // Row-group streaming: one group resident at a time (bounded RAM even
        // for the 275k-row QA table).
        for (int rg = 0; rg < file_groups; ++rg) {
            if (opts.max_rows > 0 && static_cast<size_t>(delivered) >= opts.max_rows) break;
            // Result-based ReadRowGroup: the Status out-param overload is
            // deprecated since Arrow 24 (fatal under our -Werror).
            auto maybe_table = reader->ReadRowGroup(rg);
            if (!maybe_table.ok()) {
                log_warn(strfmt("parquet: unreadable row group %d in %s; skipped", rg, path.c_str()));
                continue;
            }
            std::shared_ptr<arrow::Table> table = maybe_table.ValueOrDie();
            if (!table || table->num_rows() == 0) {
                log_warn(strfmt("parquet: unreadable row group %d in %s; skipped", rg, path.c_str()));
                continue;
            }
            // Decode the needed columns once per group, then zip rows.
            std::vector<std::vector<std::string>> col_data(cols.size());
            std::vector<char> col_ok(cols.size(), 0);
            for (size_t c = 0; c < cols.size(); ++c) {
                std::shared_ptr<arrow::ChunkedArray> chunked = table->GetColumnByName(cols[c]);
                if (!chunked) continue;
                bool ok = true;
                for (int ch = 0; ch < chunked->num_chunks(); ++ch) {
                    if (!chunk_to_strings(*chunked->chunk(ch), col_data[c],
                                          opts.max_value_bytes, path, cols[c])) {
                        ok = false;
                        break;
                    }
                }
                col_ok[c] = ok ? 1 : 0;
            }
            const int64_t nrows = table->num_rows();
            for (int64_t r = 0; r < nrows; ++r) {
                if (opts.max_rows > 0 && static_cast<size_t>(delivered) >= opts.max_rows) break;
                std::map<std::string, std::string> row;
                for (size_t c = 0; c < cols.size(); ++c) {
                    if (!col_ok[c]) continue;
                    const auto& v = col_data[c];
                    // A skipped unreadable chunk shortens the column: pad "".
                    row[cols[c]] = (r < static_cast<int64_t>(v.size())) ? v[static_cast<size_t>(r)] : "";
                }
                cb(row);
                ++delivered;
            }
        }
    }
    return delivered;
}

#else  // ---- no Arrow backend: loud stubs (never silent) ----

bool parquet_available() { return false; }

ParquetTableInfo inspect_parquet(const std::string& path) {
    ParquetTableInfo info;
    info.path = path;
    info.rows = -1;  // unknown without Arrow
    return info;
}

size_t read_parquet_docs(const std::vector<std::string>& files,
                         ParquetRowCallback /*cb*/,
                         const ParquetOptions& /*opts*/) {
    (void)files;
    GAI_FAIL("parquet input needs Apache Arrow: rebuild with -DGAI_ENABLE_PARQUET=ON "
             "(kaggle/setup.sh --with-parquet), or use the JSON route "
             "dataset/qa_darija/*.json which needs no extra dependency");
    return 0;
}

#endif  // GAI_PARQUET

} // namespace gai
