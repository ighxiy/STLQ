#include "stlq/pipeline/large_store_lifecycle.h"
#include "stlq/pipeline/large_store_hash.h"

#include <filesystem>
#include <fstream>
#include <iostream>

namespace {

void Touch(const std::filesystem::path& p) {
    std::filesystem::create_directories(p.parent_path());
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    out << "x\n";
}

void Check(bool ok, const char* msg) {
    if (!ok) {
        std::cerr << msg << "\n";
        std::exit(1);
    }
}

}  // namespace

int main() {
    namespace fs = std::filesystem;

    const fs::path root = fs::current_path() / "test_large_store_lifecycle_tmp";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root);

    {
        const fs::path dir = root / "protected_store";
        Touch(dir / "meta.bin");

        stlq::app::StoreLifecyclePrepareOptions opts;
        opts.label = "protected_store";
        opts.dir = dir.string();
        opts.protect_existing_outputs = true;
        std::string err;
        const bool ok = stlq::app::PrepareStoreForRebuildOrResume(opts, &err);
        Check(!ok, "protect_existing_outputs should reject existing store");
        Check(err.find("protect_existing_outputs=true") != std::string::npos, "unexpected protect error");
        Check(fs::exists(dir / "meta.bin"), "protected store should remain");
    }

    {
        const fs::path dir = root / "resume_store";
        fs::create_directories(dir);
        const fs::path hash_path = dir / "hash.u64";
        std::string err;
        Check(stlq::app::WriteExpectedStoreHash("resume_store", hash_path.string(), 0x1234ull, &err),
              "failed to write resume hash");

        stlq::app::StoreLifecyclePrepareOptions opts;
        opts.label = "resume_store";
        opts.dir = dir.string();
        opts.preserve_existing = true;
        opts.verify_hash_when_preserving = true;
        opts.hash_path = hash_path.string();
        opts.expected_hash = 0x5678ull;
        opts.hash_mismatch_message = "resume hash mismatch";
        const bool ok = stlq::app::PrepareStoreForRebuildOrResume(opts, &err);
        Check(!ok, "hash mismatch should reject resume");
        Check(err == "resume hash mismatch", "unexpected hash mismatch error");
        Check(fs::exists(hash_path), "resume hash file should remain");
    }

    {
        const fs::path dir = root / "rebuild_store";
        const fs::path tmp = root / "rebuild_tmp";
        Touch(dir / "old.bin");
        Touch(tmp / "old.tmp");

        stlq::app::StoreLifecyclePrepareOptions opts;
        opts.label = "rebuild_store";
        opts.dir = dir.string();
        opts.extra_remove_dirs.push_back(tmp.string());
        std::string err;
        const bool ok = stlq::app::PrepareStoreForRebuildOrResume(opts, &err);
        Check(ok, "rebuild preparation should succeed");
        Check(!fs::exists(dir), "rebuild store dir should be removed");
        Check(!fs::exists(tmp), "extra tmp dir should be removed");
    }

    {
        const fs::path dir = root / "reuse_match";
        fs::create_directories(dir);
        const fs::path hash_path = dir / "hash.u64";
        std::string err;
        Check(stlq::app::WriteExpectedStoreHash("reuse_match", hash_path.string(), 0x9ull, &err),
              "failed to write reuse hash");

        stlq::app::StoreReuseCheckOptions opts;
        opts.label = "reuse_match";
        opts.dir = dir.string();
        opts.hash_path = hash_path.string();
        opts.expected_hash = 0x9ull;
        stlq::app::StoreReuseDecision decision;
        Check(stlq::app::CheckExistingStoreReuse(opts, &decision, &err), "reuse match check failed");
        Check(decision.reuse && !decision.rebuild, "matching hash should reuse");
        Check(decision.adopted_previous_hash && decision.effective_hash == 0x9ull,
              "matching hash should adopt previous hash");
    }

    {
        const fs::path dir = root / "reuse_eval_only";
        fs::create_directories(dir);
        const fs::path hash_path = dir / "hash.u64";
        std::string err;
        Check(stlq::app::WriteExpectedStoreHash("reuse_eval_only", hash_path.string(), 0x10ull, &err),
              "failed to write eval-only hash");

        stlq::app::StoreReuseCheckOptions opts;
        opts.label = "reuse_eval_only";
        opts.dir = dir.string();
        opts.hash_path = hash_path.string();
        opts.expected_hash = 0x20ull;
        opts.eval_only = true;
        opts.eval_only_hash_mismatch_warning = "eval-only warning";
        stlq::app::StoreReuseDecision decision;
        Check(stlq::app::CheckExistingStoreReuse(opts, &decision, &err), "eval-only reuse check failed");
        Check(decision.reuse && !decision.rebuild, "eval-only stale hash should reuse");
        Check(decision.adopted_previous_hash && decision.effective_hash == 0x10ull,
              "eval-only stale hash should adopt previous hash");
    }

    {
        const fs::path dir = root / "reuse_non_eval_error";
        fs::create_directories(dir);
        const fs::path hash_path = dir / "hash.u64";
        std::string err;
        Check(stlq::app::WriteExpectedStoreHash("reuse_non_eval_error", hash_path.string(), 0x1ull, &err),
              "failed to write non-eval hash");

        stlq::app::StoreReuseCheckOptions opts;
        opts.label = "reuse_non_eval_error";
        opts.dir = dir.string();
        opts.hash_path = hash_path.string();
        opts.expected_hash = 0x2ull;
        opts.error_on_non_eval_hash_mismatch = true;
        opts.non_eval_hash_mismatch_error = "non-eval mismatch";
        stlq::app::StoreReuseDecision decision;
        Check(!stlq::app::CheckExistingStoreReuse(opts, &decision, &err),
              "non-eval stale hash should fail");
        Check(err == "non-eval mismatch", "unexpected non-eval mismatch error");
    }

    {
        const fs::path dir = root / "reuse_missing_hash";
        fs::create_directories(dir);
        std::string err;

        stlq::app::StoreReuseCheckOptions opts;
        opts.label = "reuse_missing_hash";
        opts.dir = dir.string();
        opts.hash_path = (dir / "hash.u64").string();
        opts.expected_hash = 0x33ull;
        opts.allow_missing_hash_reuse = true;
        stlq::app::StoreReuseDecision decision;
        Check(stlq::app::CheckExistingStoreReuse(opts, &decision, &err), "missing-hash reuse check failed");
        Check(decision.reuse && !decision.rebuild, "missing hash should reuse when allowed");
        Check(!decision.adopted_previous_hash && decision.effective_hash == 0x33ull,
              "missing hash should keep expected hash");
    }

    {
        const fs::path dir = root / "codec_reuse_match";
        const fs::path meta = dir / "coeff_meta.bin";
        fs::create_directories(dir);
        Touch(meta);
        const fs::path hash_path = dir / "codec_hash.u64";
        std::string err;
        Check(stlq::app::WriteExpectedStoreHash("codec_reuse_match", hash_path.string(), 0x44ull, &err),
              "failed to write codec hash");

        stlq::app::CodecStoreLifecycleOptions opts;
        opts.label = "coeff codec";
        opts.store_dir = dir.string();
        opts.codec_meta_path = meta.string();
        opts.hash_path = hash_path.string();
        opts.expected_hash = 0x44ull;
        opts.has_codec_store = true;
        stlq::app::CodecStoreLifecycleDecision decision;
        Check(stlq::app::PrepareCodecStoreForEval(opts, &decision, &err), "codec reuse check failed");
        Check(!decision.needs_rebuild && decision.hash_matches, "matching codec hash should not rebuild");
    }

    {
        const fs::path dir = root / "codec_protected";
        const fs::path meta = dir / "coeff_meta.bin";
        Touch(meta);

        stlq::app::CodecStoreLifecycleOptions opts;
        opts.label = "coeff codec";
        opts.store_dir = dir.string();
        opts.codec_meta_path = meta.string();
        opts.hash_path = (dir / "codec_hash.u64").string();
        opts.expected_hash = 0x55ull;
        opts.has_codec_store = true;
        opts.protect_existing_outputs = true;
        opts.has_rebuild_source = true;
        stlq::app::CodecStoreLifecycleDecision decision;
        std::string err;
        Check(!stlq::app::PrepareCodecStoreForEval(opts, &decision, &err),
              "protected stale codec should fail");
        Check(err.find("protect_existing_outputs=true") != std::string::npos,
              "unexpected protected codec error");
    }

    {
        const fs::path dir = root / "codec_missing_source";
        fs::create_directories(dir);

        stlq::app::CodecStoreLifecycleOptions opts;
        opts.label = "coeff codec";
        opts.store_dir = dir.string();
        opts.codec_meta_path = (dir / "coeff_meta.bin").string();
        opts.hash_path = (dir / "codec_hash.u64").string();
        opts.expected_hash = 0x66ull;
        opts.rebuild_source_why_not = "test missing f32";
        stlq::app::CodecStoreLifecycleDecision decision;
        std::string err;
        Check(!stlq::app::PrepareCodecStoreForEval(opts, &decision, &err),
              "codec without rebuild source should fail");
        Check(err.find("float coeff payload is unavailable (test missing f32)") != std::string::npos,
              "unexpected missing source error");
    }

    {
        const fs::path dir = root / "codec_rebuild";
        fs::create_directories(dir);

        stlq::app::CodecStoreLifecycleOptions opts;
        opts.label = "coeff codec";
        opts.store_dir = dir.string();
        opts.codec_meta_path = (dir / "coeff_meta.bin").string();
        opts.hash_path = (dir / "codec_hash.u64").string();
        opts.expected_hash = 0x77ull;
        opts.has_rebuild_source = true;
        stlq::app::CodecStoreLifecycleDecision decision;
        std::string err;
        Check(stlq::app::PrepareCodecStoreForEval(opts, &decision, &err),
              "codec rebuild decision should succeed");
        Check(decision.needs_rebuild && !decision.hash_matches,
              "missing codec should request rebuild");
    }

    fs::remove_all(root, ec);
    std::cout << "test_large_store_lifecycle: OK\n";
    return 0;
}
