#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace stlq::app {

struct StoreLifecyclePrepareOptions {
    std::string label;
    std::string dir;
    std::vector<std::string> extra_remove_dirs;

    bool protect_existing_outputs = false;
    bool preserve_existing = false;
    bool verify_hash_when_preserving = false;
    std::string hash_path;
    std::uint64_t expected_hash = 0;
    std::string preserve_log_message;
    std::string hash_mismatch_message;
};

bool PrepareStoreForRebuildOrResume(const StoreLifecyclePrepareOptions& opts,
                                    std::string* error);

bool WriteExpectedStoreHash(const std::string& label,
                            const std::string& hash_path,
                            std::uint64_t expected_hash,
                            std::string* error);

struct StoreReuseCheckOptions {
    std::string label;
    std::string dir;
    std::string hash_path;
    std::uint64_t expected_hash = 0;
    bool eval_only = false;

    bool error_on_non_eval_hash_mismatch = false;
    bool rebuild_on_non_eval_hash_mismatch = true;
    bool allow_missing_hash_reuse = false;

    std::string non_eval_hash_mismatch_error;
    std::string eval_only_hash_mismatch_warning;
};

struct StoreReuseDecision {
    bool reuse = false;
    bool rebuild = true;
    bool hash_present = false;
    bool hash_matches = false;
    bool adopted_previous_hash = false;
    std::uint64_t effective_hash = 0;
};

bool CheckExistingStoreReuse(const StoreReuseCheckOptions& opts,
                             StoreReuseDecision* decision,
                             std::string* error);

struct CodecStoreLifecycleOptions {
    std::string label;
    std::string store_dir;
    std::string codec_meta_path;
    std::string hash_path;
    std::uint64_t expected_hash = 0;

    bool has_codec_store = false;
    bool protect_existing_outputs = false;
    bool has_rebuild_source = false;
    std::string rebuild_source_why_not;
};

struct CodecStoreLifecycleDecision {
    bool hash_present = false;
    bool hash_matches = false;
    bool needs_rebuild = false;
    std::uint64_t previous_hash = 0;
};

bool PrepareCodecStoreForEval(const CodecStoreLifecycleOptions& opts,
                              CodecStoreLifecycleDecision* decision,
                              std::string* error);

}  // namespace stlq::app
