#include "stlq/pipeline/stlq_large_eval_plan.h"

#include <cassert>
#include <iostream>
#include <vector>

int main() {
    stlq::Config cfg;
    cfg.eval.linkage_nprobe = 7;
    cfg.eval.linkage_ivf_hnsw_ef_search = 31;
    cfg.eval.linkage_ivf_probe_mode = "hnsw";
    cfg.eval.linkage_preload_clusters_io_threads = 0;

    {
        const std::vector<int> nprobe{4, 4, 0, 8};
        const std::vector<int> ef{16, -1, 32};
        const stlq::LargeLinkageProbeEvalPlan plan =
            stlq::PrepareLargeLinkageProbeEvalPlanStage(cfg, nprobe, ef);
        assert(plan.hnsw_probe_mode);
        assert(plan.reuse_provider_across_nprobes);
        assert((plan.linkage_nprobe_batch == std::vector<int>{4, 8}));
        assert((plan.linkage_ef_search_batch == std::vector<int>{16, 32}));
        assert(plan.linkage_eval_combo_count == 4);
        assert(plan.cases.size() == 4);
        assert(plan.cases[0].nprobe == 4 && plan.cases[0].ef_search == 16);
        assert(plan.cases[1].nprobe == 4 && plan.cases[1].ef_search == 32);
        assert(plan.cases[2].nprobe == 8 && plan.cases[2].ef_search == 16);
        assert(plan.cases[3].nprobe == 8 && plan.cases[3].ef_search == 32);
    }

    {
        cfg.eval.linkage_ivf_probe_mode = "linear";
        cfg.eval.linkage_preload_clusters_io_threads = -1;
        const stlq::LargeLinkageProbeEvalPlan plan =
            stlq::PrepareLargeLinkageProbeEvalPlanStage(cfg, {}, {16, 32});
        assert(!plan.hnsw_probe_mode);
        assert(!plan.reuse_provider_across_nprobes);
        assert((plan.linkage_nprobe_batch == std::vector<int>{7}));
        assert((plan.linkage_ef_search_batch == std::vector<int>{31}));
        assert(plan.linkage_eval_combo_count == 1);
        assert(plan.cases.size() == 1);
        assert(plan.cases[0].nprobe == 7 && plan.cases[0].ef_search == 31);
    }

    std::cout << "test_large_eval_plan: OK\n";
    return 0;
}
