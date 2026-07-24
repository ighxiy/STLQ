#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "stlq/common/config.h"
#include "stlq/eval/recall_linkage_disk.h"
#include "stlq/io/base_store.h"
#include "stlq/io/linkage_list_store.h"
#include "stlq/common/types.h"

namespace stlq::app {

// ---- Runtime helpers ----

std::string ZeroPad4(int v);
std::filesystem::path NextRuntimeLogArchivePath(const std::filesystem::path& log_dir);

double Median(std::vector<double> v);

// Rotates queries (with a small GEMM warmup) and returns the GEMM wall seconds for the real rotation.
double RotateQueriesForEvalWithWarmup(const stlq::ColMajorMatrix<float>& R,
                                     stlq::ColMajorMatrix<float>* X);

// Meta-only precomp structure (offsets/H) derived from a codebook pack.
stlq::Precomp BuildPrecompMetaOnly(const stlq::CodebookPack& pack);

// Optional crash tracing on Windows (no-op on other platforms).
void InstallCrashTraceIfEnabled();

// ---- BaseBasic checkpoint helpers (resume/truncate) ----

struct BaseBasicCheckpoint {
    static constexpr std::uint32_t kMagic = 0x544b4342u;  // "BCKT" (Base basic ChecKpoinT)
    std::uint32_t magic = kMagic;
    std::uint32_t version = 1;
    std::uint32_t nbucket = 0;
    std::uint32_t reserved = 0;
    std::uint64_t next_id = 0;
    std::vector<std::uint64_t> bucket_sizes;
};

int ComputeNbucketsForBasic(const stlq::io::BaseBasicStoreConfig& store_cfg);
std::filesystem::path BaseBasicCheckpointPath(const std::string& base_basic_dir);

bool ReadBaseBasicCheckpoint(const std::filesystem::path& path,
                            BaseBasicCheckpoint* out,
                            std::string* err);

bool WriteBaseBasicCheckpointAtomic(const std::filesystem::path& path,
                                   const BaseBasicCheckpoint& ck,
                                   std::string* err);

bool TruncateBaseBasicToCheckpoint(const std::string& out_dir,
                                  const stlq::io::BaseBasicStoreConfig& store_cfg,
                                  const BaseBasicCheckpoint& ck,
                                  std::string* err);

struct BaseBasicCkptCtx {
    std::filesystem::path path;
    std::uint32_t nbucket = 0;
};

bool OnCommitBaseBasicCkpt(std::uint64_t next_id,
                          stlq::io::BaseBasicWriter* writer,
                          void* vctx,
                          std::string* err);

// ---- Bit budget logging helpers (app-only diagnostics) ----

struct LinkageLOUDSIndexBitBudget {
    double rank_bits_per_real = 0.0;
    double select_bits_per_real = 0.0;
    double total_bits_per_real = 0.0;
    bool enabled = false;
    std::uint32_t select_stride = 128;
    std::uint32_t rank_words_per_super_log2 = 4;
};

LinkageLOUDSIndexBitBudget LogLinkageBitBudgetSummary(const stlq::Config& config,
                                                 const stlq::io::LinkageListReader& linkage_list);

void ApplyLOUDSIndexBudgetToTiming(const LinkageLOUDSIndexBitBudget& b,
                                  stlq::DiskLinkageEvalSession* session);

}  // namespace stlq::app
