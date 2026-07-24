#include "stlq/pipeline/large_store_lifecycle.h"

#include "stlq/pipeline/large_store_hash.h"
#include "stlq/common/logger.h"

#include <filesystem>

namespace stlq::app {

bool PrepareStoreForRebuildOrResume(const StoreLifecyclePrepareOptions& opts,
                                    std::string* error) {
    const bool have_out_dir = std::filesystem::exists(opts.dir);
    if (opts.protect_existing_outputs && have_out_dir && !opts.preserve_existing) {
        if (error) {
            *error = opts.label + " outputs exist but large.protect_existing_outputs=true; delete manually: " +
                     opts.dir;
        }
        return false;
    }

    if (opts.preserve_existing) {
        if (opts.verify_hash_when_preserving) {
            std::uint64_t prev_hash = 0;
            if (ReadU64File(opts.hash_path, &prev_hash) && prev_hash != opts.expected_hash) {
                if (error) {
                    *error = opts.hash_mismatch_message.empty()
                                 ? (opts.label + " store hash mismatch for checkpoint resume: " + opts.dir)
                                 : opts.hash_mismatch_message;
                }
                return false;
            }
        }
        if (!opts.preserve_log_message.empty()) {
            LogInfo(opts.preserve_log_message);
        }
        return true;
    }

    std::error_code ec;
    std::filesystem::remove_all(opts.dir, ec);
    for (const std::string& extra : opts.extra_remove_dirs) {
        std::filesystem::remove_all(extra, ec);
    }
    return true;
}

bool WriteExpectedStoreHash(const std::string& label,
                            const std::string& hash_path,
                            std::uint64_t expected_hash,
                            std::string* error) {
    if (!WriteU64FileHex(hash_path, expected_hash, error)) {
        if (error && error->empty()) {
            *error = "Failed to write " + label + " hash: " + hash_path;
        }
        return false;
    }
    return true;
}

bool CheckExistingStoreReuse(const StoreReuseCheckOptions& opts,
                             StoreReuseDecision* decision,
                             std::string* error) {
    if (!decision) {
        if (error) *error = "CheckExistingStoreReuse: decision output is null.";
        return false;
    }
    *decision = StoreReuseDecision{};
    decision->effective_hash = opts.expected_hash;

    std::uint64_t prev_hash = 0;
    if (ReadU64File(opts.hash_path, &prev_hash)) {
        decision->hash_present = true;
        decision->hash_matches = (prev_hash == opts.expected_hash);
        if (decision->hash_matches) {
            decision->reuse = true;
            decision->rebuild = false;
            decision->effective_hash = prev_hash;
            decision->adopted_previous_hash = true;
            return true;
        }

        if (opts.eval_only) {
            if (!opts.eval_only_hash_mismatch_warning.empty()) {
                LogWarn(opts.eval_only_hash_mismatch_warning);
            }
            decision->reuse = true;
            decision->rebuild = false;
            decision->effective_hash = prev_hash;
            decision->adopted_previous_hash = true;
            return true;
        }

        if (opts.error_on_non_eval_hash_mismatch) {
            if (error) {
                *error = opts.non_eval_hash_mismatch_error.empty()
                             ? (opts.label + " store hash mismatch: " + opts.dir)
                             : opts.non_eval_hash_mismatch_error;
            }
            return false;
        }

        decision->reuse = !opts.rebuild_on_non_eval_hash_mismatch;
        decision->rebuild = opts.rebuild_on_non_eval_hash_mismatch;
        return true;
    }

    if (opts.allow_missing_hash_reuse) {
        decision->reuse = true;
        decision->rebuild = false;
    }
    return true;
}

bool PrepareCodecStoreForEval(const CodecStoreLifecycleOptions& opts,
                              CodecStoreLifecycleDecision* decision,
                              std::string* error) {
    if (!decision) {
        if (error) *error = "PrepareCodecStoreForEval: decision output is null.";
        return false;
    }
    *decision = CodecStoreLifecycleDecision{};

    std::uint64_t prev_hash = 0;
    if (ReadU64File(opts.hash_path, &prev_hash)) {
        decision->hash_present = true;
        decision->previous_hash = prev_hash;
        decision->hash_matches = (prev_hash == opts.expected_hash);
    }

    if (opts.has_codec_store && decision->hash_matches) {
        return true;
    }

    decision->needs_rebuild = true;
    if (opts.protect_existing_outputs && std::filesystem::exists(opts.codec_meta_path)) {
        if (error) {
            *error = opts.label + " outputs exist but large.protect_existing_outputs=true; delete manually: " +
                     opts.store_dir;
        }
        return false;
    }
    if (!opts.has_rebuild_source) {
        if (error) {
            *error = "Need to (re)build " + opts.label +
                     ", but linkage_list float coeff payload is unavailable (" +
                     opts.rebuild_source_why_not + "). "
                     "Rebuild linkage_list with float coeffs, or point to a linkage_list store that still has *.f32.";
        }
        return false;
    }
    return true;
}

}  // namespace stlq::app
