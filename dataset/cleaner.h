#pragma once

#include "core/common.h"
#include "tokenizer/normalizer.h"
#include <string>
#include <vector>

namespace gai {

struct PiiReport {
    int emails = 0, phones = 0, urls_with_creds = 0, ibans = 0, cards = 0;
    int api_keys = 0, ids = 0, ips = 0;
    bool any() const { return emails || phones || urls_with_creds || ibans || cards || api_keys || ids || ips; }
    int total() const { return emails + phones + urls_with_creds + ibans + cards + api_keys + ids + ips; }
    std::string summary() const;
};

PiiReport   scan_pii(const std::string& text);
std::string redact_pii(const std::string& text, PiiReport* report = nullptr);

struct CleanConfig {
    bool strip_html          = true;
    bool fix_mojibake        = true;
    bool drop_pii_lines      = true;
    bool redact_instead_drop = false;
    int  min_chars           = 2;
    int  max_chars           = 100000;
    NormalizerConfig normalizer{};
};

struct CleanStats {
    u64 lines_in = 0, lines_out = 0;
    u64 dropped_empty = 0, dropped_short = 0, dropped_long = 0;
    u64 dropped_pii = 0, dropped_encoding = 0, dropped_quality = 0;
    u64 bytes_in = 0, bytes_out = 0;
    std::string summary() const;
};

class Cleaner {
public:
    explicit Cleaner(CleanConfig cfg = {}) : cfg_(cfg), norm_(cfg.normalizer) {}

    bool clean_line(const std::string& in, std::string& out, CleanStats& st) const;

    const CleanConfig& config() const { return cfg_; }

private:
    CleanConfig cfg_;
    Normalizer  norm_;
};

std::string strip_html_tags(const std::string& s);
std::string fix_mojibake(const std::string& s);

struct QualityConfig {
    double max_symbol_ratio    = 0.25;
    double max_digit_ratio     = 0.30;
    double max_upper_ratio     = 0.40;
    double max_repeat_line     = 0.15;
    double max_word_repeat     = 0.15;
    double min_letter_ratio    = 0.45;
    int    min_words           = 2;
    int    max_word_length     = 60;
    bool   require_arabic_or_latin = true;
    bool   reject_ai_disclaimers   = true;
    bool   reject_placeholders     = true;
};

struct QualityVerdict {
    bool accept = true;
    std::string reason;
};

QualityVerdict quality_check(const std::string& text, const QualityConfig& cfg = {});

QualityConfig english_quality_config();
QualityVerdict quality_check_english(const std::string& text, const QualityConfig& cfg = {});

struct ToxicityResult {
    bool  toxic = false;
    int   hits = 0;
    std::vector<std::string> categories;
};

ToxicityResult check_toxicity(const std::string& text);

}
