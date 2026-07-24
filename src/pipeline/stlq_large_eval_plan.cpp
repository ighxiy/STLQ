#include "stlq/pipeline/stlq_large_eval_plan.h"

#include "stlq/common/logger.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <sstream>

namespace {

std::vector<int> NormalizeBatchValues(const std::vector<int>& raw, int fallback) {
    std::vector<int> batch;
    batch.reserve(raw.size() + 1);
    for (int v : raw) {
        if (v <= 0) continue;
        if (std::find(batch.begin(), batch.end(), v) == batch.end()) {
            batch.push_back(v);
        }
    }
    if (batch.empty()) {
        batch.push_back(std::max(1, fallback));
    }
    return batch;
}

bool LinkageListF32PayloadPresent(const stlq::io::LinkageListReader& linkage_list,
                                std::string* why_not) {
    if (why_not) why_not->clear();
    if (linkage_list.meta().store_coeffs_f32 == 0) {
        if (why_not) *why_not = "meta.store_coeffs_f32=0";
        return false;
    }
    const std::filesystem::path dir = std::filesystem::path(linkage_list.dir());
    const std::filesystem::path paths[] = {
        dir / "coeffs.f32",
        dir / "a0.f32",
        dir / "virt_coeffs.f32",
        dir / "virt_a0.f32",
    };
    std::string missing;
    for (const auto& p : paths) {
        std::error_code ec;
        const bool ok = std::filesystem::exists(p, ec);
        if (ec || !ok) {
            if (!missing.empty()) missing += ", ";
            missing += p.filename().string();
        }
    }
    if (!missing.empty()) {
        if (why_not) {
            *why_not = "missing f32 payload files: [" + missing + "]";
        }
        return false;
    }
    return true;
}

}  // namespace

namespace stlq {

int PrepareLargeLinkageCoeffEvalPlanStage(const Config& config,
                                        const std::string& linkage_list_dir,
                                        io::LinkageListReader& linkage_list,
                                        LargeLinkageCoeffEvalPlan& coeff_eval_plan,
                                        std::string& error) {
    coeff_eval_plan = LargeLinkageCoeffEvalPlan{};
    coeff_eval_plan.mode = config.eval.linkage_coeff_mode;
    std::transform(coeff_eval_plan.mode.begin(), coeff_eval_plan.mode.end(), coeff_eval_plan.mode.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    // The linkage_list store may intentionally omit float coeffs (store_coeffs_f32=0),
    // especially under disk-min presets like eval_int8_min. In that case, attempting
    // float-coeff recall is invalid and can crash downstream paths.
    bool meta_has_float_coeffs = (linkage_list.meta().store_coeffs_f32 != 0);
    coeff_eval_plan.has_float_coeffs =
        LinkageListF32PayloadPresent(linkage_list, &coeff_eval_plan.float_coeffs_why_not);
    if (meta_has_float_coeffs && !coeff_eval_plan.has_float_coeffs) {
        std::string meta_fix_err;
        if (io::UpdateLinkageListMetaStoreCoeffsF32(linkage_list_dir, /*store_coeffs_f32=*/false, &meta_fix_err)) {
            if (!linkage_list.Open(linkage_list_dir, &error)) {
                LogError(error);
                return 1;
            }
            meta_has_float_coeffs = false;
            coeff_eval_plan.has_float_coeffs = false;
            LogInfo("linkage_list meta.bin was stale after cleanup; updated store_coeffs_f32=0 to match missing *.f32 payload files.");
        } else {
            LogWarn("linkage_list meta indicates float coeffs exist, but f32 payload is not usable: " +
                    coeff_eval_plan.float_coeffs_why_not +
                    ". Also failed to auto-fix meta.bin: " + meta_fix_err);
        }
    }

    coeff_eval_plan.coeff_meta = std::filesystem::path(linkage_list_dir) / "coeff_meta.bin";
    coeff_eval_plan.has_coeff_codec = std::filesystem::exists(coeff_eval_plan.coeff_meta);

    if (coeff_eval_plan.mode == "float") {
        if (!coeff_eval_plan.has_float_coeffs) {
            LogError("eval.linkage.coeff_mode=float but linkage_list float coeff payload is unavailable (" +
                     coeff_eval_plan.float_coeffs_why_not + "). "
                     "Rebuild linkage_list with large.base_linkage_store_coeffs_f32=true, or use eval.linkage.coeff_mode=int8 and ensure coeff codec exists.");
            return 1;
        }
        coeff_eval_plan.run_float = true;
    } else if (coeff_eval_plan.mode == "int8") {
        coeff_eval_plan.run_int8 = true;
    } else if (coeff_eval_plan.mode == "both") {
        coeff_eval_plan.run_float = coeff_eval_plan.has_float_coeffs;
        coeff_eval_plan.run_int8 = true;
    } else {  // "auto" or invalid
        coeff_eval_plan.run_float = coeff_eval_plan.has_float_coeffs;
        coeff_eval_plan.run_int8 = (config.large.linkage_coeff_codec.enabled && coeff_eval_plan.has_coeff_codec);
        if (coeff_eval_plan.mode != "auto") {
            LogWarn("Invalid eval.linkage.coeff_mode=" + config.eval.linkage_coeff_mode +
                    ", fallback to \"auto\".");
        }
    }

    if (!coeff_eval_plan.run_float && !coeff_eval_plan.run_int8) {
        LogError("No valid linkage coeff mode available: linkage_list has no float coeffs and int8 coeff codec is unavailable. "
                 "Enable large.linkage_coeff_codec.enabled=true (and build/rebuild codec), or rebuild linkage_list with float coeffs.");
        return 1;
    }

    if (coeff_eval_plan.mode == "both" && !coeff_eval_plan.has_float_coeffs) {
        LogWarn("eval.linkage.coeff_mode=both requested, but linkage_list float coeff payload is unavailable (" +
                coeff_eval_plan.float_coeffs_why_not + "); running int8 only.");
    }
    return 0;
}

LargeLinkageProbeEvalPlan PrepareLargeLinkageProbeEvalPlanStage(
    const Config& config,
    const std::vector<int>& linkage_nprobe_batch_cfg,
    const std::vector<int>& linkage_ef_search_batch_cfg) {
    LargeLinkageProbeEvalPlan probe_eval_plan;
    probe_eval_plan.linkage_nprobe_batch =
        NormalizeBatchValues(linkage_nprobe_batch_cfg, std::max(1, config.eval.linkage_nprobe));
    probe_eval_plan.linkage_ef_search_batch =
        NormalizeBatchValues(linkage_ef_search_batch_cfg, std::max(1, config.eval.linkage_ivf_hnsw_ef_search));
    probe_eval_plan.hnsw_probe_mode = (config.eval.linkage_ivf_probe_mode == "hnsw");
    if (!probe_eval_plan.hnsw_probe_mode) {
        if (probe_eval_plan.linkage_ef_search_batch.size() > 1) {
            LogInfo("Linkage disk recall ef_search batch ignored because eval.linkage.ivf_probe_mode=" +
                    config.eval.linkage_ivf_probe_mode + " (only active for hnsw).");
        }
        probe_eval_plan.linkage_ef_search_batch.assign(1, std::max(1, config.eval.linkage_ivf_hnsw_ef_search));
    }
    probe_eval_plan.reuse_provider_across_nprobes = (config.eval.linkage_preload_clusters_io_threads >= 0);
    probe_eval_plan.linkage_eval_combo_count =
        probe_eval_plan.linkage_nprobe_batch.size() * probe_eval_plan.linkage_ef_search_batch.size();
    probe_eval_plan.cases.reserve(probe_eval_plan.linkage_eval_combo_count);
    for (int nprobe : probe_eval_plan.linkage_nprobe_batch) {
        for (int ef_search : probe_eval_plan.linkage_ef_search_batch) {
            probe_eval_plan.cases.push_back(LargeLinkageProbeEvalCase{
                .nprobe = std::max(1, nprobe),
                .ef_search = std::max(1, ef_search),
            });
        }
    }
    return probe_eval_plan;
}

int PrepareLargeLinkageDiskEvalPlanStage(const Config& config,
                                       const std::string& linkage_list_dir,
                                       io::LinkageListReader& linkage_list,
                                       const std::vector<int>& linkage_nprobe_batch_cfg,
                                       const std::vector<int>& linkage_ef_search_batch_cfg,
                                       LargeLinkageDiskEvalPlan* plan,
                                       std::string& error) {
    if (!plan) {
        error = "PrepareLargeLinkageDiskEvalPlanStage: output plan is null.";
        return 1;
    }
    *plan = LargeLinkageDiskEvalPlan{};

    const int coeff_stage =
        PrepareLargeLinkageCoeffEvalPlanStage(config, linkage_list_dir, linkage_list, plan->coeff, error);
    if (coeff_stage != 0) {
        return coeff_stage;
    }
    plan->probe =
        PrepareLargeLinkageProbeEvalPlanStage(config, linkage_nprobe_batch_cfg, linkage_ef_search_batch_cfg);

    if (plan->coeff.run_float) {
        plan->coeff_runs.push_back(LargeLinkageCoeffEvalRun{
            .use_coeff_codec = false,
            .label = "float coeffs",
            .archive_label = "float",
        });
    }
    if (plan->coeff.run_int8) {
        plan->coeff_runs.push_back(LargeLinkageCoeffEvalRun{
            .use_coeff_codec = true,
            .label = "int8 coeff codec",
            .archive_label = "int8",
        });
    }
    if (plan->coeff_runs.empty()) {
        error = "PrepareLargeLinkageDiskEvalPlanStage: no coeff eval runs selected.";
        return 1;
    }
    return 0;
}

void LogLargeLinkageProbeEvalPlan(const LargeLinkageProbeEvalPlan& plan) {
    if (plan.linkage_eval_combo_count <= 1) {
        return;
    }
    std::ostringstream oss;
    oss << "Linkage disk recall probe batch: nprobe=[";
    for (std::size_t i = 0; i < plan.linkage_nprobe_batch.size(); ++i) {
        if (i) oss << ", ";
        oss << plan.linkage_nprobe_batch[i];
    }
    oss << "]";
    if (plan.hnsw_probe_mode) {
        oss << " ef_search=[";
        for (std::size_t i = 0; i < plan.linkage_ef_search_batch.size(); ++i) {
            if (i) oss << ", ";
            oss << plan.linkage_ef_search_batch[i];
        }
        oss << "]";
    }
    oss << " reuse_provider_cache=" << (plan.reuse_provider_across_nprobes ? "true" : "false");
    if (!plan.reuse_provider_across_nprobes) {
        oss << " (lazy mode keeps per-config sessions isolated)";
    }
    LogInfo(oss.str());
}

}  // namespace stlq
