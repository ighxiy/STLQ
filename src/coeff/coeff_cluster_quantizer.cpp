#include "stlq/coeff/coeff_cluster_quantizer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace stlq {

namespace {

inline int QFromBits(int bits) {
  // signed midtread, clip to [-Q, Q]
  return (1 << (bits - 1)) - 1;
}

std::vector<int> ExpandBitsPerLayer(int m, const std::vector<int>& in) {
  if (m <= 0) return {};
  if (in.empty()) {
    // Fallback: 7-bit signed by default (Q=127).
    return std::vector<int>(static_cast<std::size_t>(m), 7);
  }
  if (int(in.size()) == m) return in;

  std::vector<int> out;
  out.reserve(static_cast<std::size_t>(m));

  if (in.size() == 1) {
    out.assign(static_cast<std::size_t>(m), in[0]);
    return out;
  }
  if (in.size() == 2) {
    out.assign(static_cast<std::size_t>(m), in[1]);
    out[0] = in[0];
    return out;
  }

  // Convenience: if shorter than m, extend by repeating the last value.
  // If longer than m, truncate.
  out.insert(out.end(), in.begin(), in.end());
  if (int(out.size()) < m) out.resize(static_cast<std::size_t>(m), out.back());
  if (int(out.size()) > m) out.resize(static_cast<std::size_t>(m));
  return out;
}

std::vector<double> MakePCandidates(const CoeffQuantConfig& cfg, int layer) {
  const bool is_first = (layer == 0);
  const auto& fallback = is_first ? cfg.p_first_candidates : cfg.p_rest_candidates;
  if (!cfg.auto_search_p) return fallback;

  const auto& range = is_first ? cfg.p_first_range : cfg.p_rest_range;
  const double pmin = std::get<0>(range);
  const double pmax = std::get<1>(range);
  const double pstep = std::get<2>(range);
  if (pstep <= 0) throw std::runtime_error("pstep must be >0");
  std::vector<double> out;
  for (double p = pmin; p <= pmax + 1e-12; p += pstep) out.push_back(p);
  return out;
}

inline float QuantileType7Sorted(const std::vector<float>& x_sorted, double p01) {
  const int n = int(x_sorted.size());
  if (n == 0) return 0.0f;
  if (p01 <= 0.0) return x_sorted.front();
  if (p01 >= 1.0) return x_sorted.back();
  const double h = (n - 1) * p01 + 1.0;
  const int j = int(std::floor(h));
  const double g = h - j;
  if (j >= n) return x_sorted.back();
  // j is 1-based in the Julia formula; convert.
  const int i0 = std::max(0, j - 1);
  const int i1 = std::min(n - 1, j);
  return float((1.0 - g) * x_sorted[i0] + g * x_sorted[i1]);
}

inline float WeightedQuantileStepSorted(const std::vector<float>& x_sorted,
                                               const std::vector<float>& cumw_sorted,
                                               double p01) {
  const int n = int(x_sorted.size());
  if (n == 0) return 0.0f;
  const double totalw = cumw_sorted.back();
  if (totalw <= 0.0) return x_sorted.back();
  const double target = p01 * totalw;
  auto it = std::lower_bound(cumw_sorted.begin(), cumw_sorted.end(), float(target));
  int idx = int(it - cumw_sorted.begin());
  if (idx < 0) idx = 0;
  if (idx >= n) idx = n - 1;
  return x_sorted[idx];
}

struct GroupView {
  // indices of in-cluster items belonging to this group (root or linkage)
  std::vector<int> idx;
};

float GetWeightForSample(
    int layer,
    std::uint32_t code1,
    const std::vector<Span<const float>>& norm2_layers) {
  if (norm2_layers.empty()) return 1.0f;
  if (layer < 0 || layer >= int(norm2_layers.size())) return 1.0f;
  const auto tbl = norm2_layers[layer];
  if (tbl.empty()) return 1.0f;
  if (code1 == 0) return 0.0f;
  // norm2 tables are expected to be 1-based: tbl[code1]
  const auto idx = std::size_t(code1);
  if (idx >= tbl.size()) return 1.0f;
  return tbl[idx];
}

void QuantizeOneGroupOneLayer(
    int layer,
    int bits,
    int nc,
    Span<const float> a,
    Span<const std::uint32_t> B,
    const GroupView& g,
    const std::vector<Span<const float>>& norm2_layers,
    const CoeffQuantConfig& cfg,
    Span<std::int8_t> q_out,
    CoeffQuantLayerStats* st) {

  const int Q = QFromBits(bits);
  const auto p_candidates = MakePCandidates(cfg, layer);

  if (g.idx.empty()) {
    st->p_used = 100.0;
    st->clip_ratio = 0.0;
    st->step_delta = 1.0f;
    st->scale = 1.0f;
    return;
  }

  // Collect abs values (and weights if needed) for quantile.
  std::vector<float> absvals;
  std::vector<float> weights;
  absvals.reserve(g.idx.size());
  if (cfg.use_weighted_quantile) weights.reserve(g.idx.size());

  float max_abs_round = 0.0f;
  for (int ii : g.idx) {
    const float aval = a[layer * nc + ii];
    absvals.push_back(std::abs(aval));
    if (cfg.use_weighted_quantile) {
      const std::uint32_t code1 = B.empty() ? 0u : B[layer * nc + ii];
      weights.push_back(GetWeightForSample(layer, code1, norm2_layers));
    }
    const float r = std::abs(std::round(aval));
    if (r > max_abs_round) max_abs_round = r;
  }

  // Sort for quantile computation.
  std::vector<int> perm(absvals.size());
  std::iota(perm.begin(), perm.end(), 0);
  std::sort(perm.begin(), perm.end(), [&](int i, int j) { return absvals[i] < absvals[j]; });

  std::vector<float> abs_sorted(absvals.size());
  for (std::size_t i = 0; i < perm.size(); ++i) abs_sorted[i] = absvals[perm[i]];

  std::vector<float> cumw_sorted;
  if (cfg.use_weighted_quantile) {
    cumw_sorted.resize(weights.size());
    double run = 0.0;
    for (std::size_t i = 0; i < perm.size(); ++i) {
      const float w = weights[perm[i]];
      run += (w > 0.0f ? double(w) : 0.0);
      cumw_sorted[i] = float(run);
    }
  }

  const float max_abs = abs_sorted.back();

  auto get_Amax = [&](double p) -> float {
    if (p >= 100.0) return max_abs;
    const double p01 = p / 100.0;
    if (cfg.use_weighted_quantile) {
      return WeightedQuantileStepSorted(abs_sorted, cumw_sorted, p01);
    }
    return QuantileType7Sorted(abs_sorted, p01);
  };

  // Evaluate candidates and pick best.
  double best_cost = std::numeric_limits<double>::infinity();
  double best_p = 100.0;
  float best_delta = 1.0f;
  float best_scale = 1.0f;
  double best_clip = 0.0;

  for (double p : p_candidates) {
    const float Amax = get_Amax(p);
    if (!cfg.allow_clip && Amax < max_abs_round) {
      continue;
    }
    if (!(Amax > 0.0f)) {
      continue;
    }

    const float delta = Amax / float(std::max(Q, 1));

    // Weighted SSE minimizer in coefficient domain.
    // cost = sum w*a^2 - (sum w*a*q)^2 / (sum w*q^2)
    double Sa2 = 0.0;
    double Saq = 0.0;
    double Sq2 = 0.0;
    int clip_cnt = 0;
    int tot = 0;

    for (int ii : g.idx) {
      const float aval = a[layer * nc + ii];
      int q = int(std::lrint(double(aval) / double(delta)));
      if (q > Q) {
        q = Q;
        ++clip_cnt;
      } else if (q < -Q) {
        q = -Q;
        ++clip_cnt;
      }
      const float w = cfg.use_weighted_quantile ?
          GetWeightForSample(layer, B.empty() ? 0u : B[layer * nc + ii], norm2_layers) :
          1.0f;
      const auto ww = double(w);
      Sa2 += ww * double(aval) * double(aval);
      Saq += ww * double(aval) * double(q);
      Sq2 += ww * double(q) * double(q);
      ++tot;
    }

    if (Sq2 <= 0.0) {
      continue;
    }

    const double cost = Sa2 - (Saq * Saq) / Sq2;
    const double s = cfg.fit_scale ? (Saq / Sq2) : double(delta);
    const double clip_ratio = (tot > 0) ? double(clip_cnt) / double(tot) : 0.0;

    if (cost < best_cost) {
      best_cost = cost;
      best_p = p;
      best_delta = delta;
      best_scale = float(s);
      best_clip = clip_ratio;
    }
  }

  st->p_used = best_p;
  st->clip_ratio = best_clip;
  st->step_delta = best_delta;
  st->scale = best_scale;

  // Emit quantized q for this layer/group into q_out.
  for (int ii : g.idx) {
    const float aval = a[layer * nc + ii];
    int q = int(std::lrint(double(aval) / double(best_delta)));
    if (q > Q) q = Q;
    if (q < -Q) q = -Q;
    q_out[layer * nc + ii] = std::int8_t(q);
  }
}

}  // namespace

std::vector<int> MakeBitsPerLayer(int m, int bits_first, int bits_rest) {
  std::vector<int> bits(m, bits_rest);
  if (m > 0) bits[0] = bits_first;
  return bits;
}

void QuantizeClusterCoeffsSplitRootLinkage(
    int m,
    int nc,
    Span<const float> a,
    Span<const std::uint32_t> B,
    Span<const std::uint8_t> is_linkage,
    const std::vector<Span<const float>>& norm2_root_layers,
    const std::vector<Span<const float>>& norm2_linkage_layers,
    const CoeffQuantConfig& cfg,
    Span<std::int8_t> q_out,
    CoeffQuantGroupResult* result_root,
    CoeffQuantGroupResult* result_linkage) {

  if (m <= 0 || nc < 0) throw std::runtime_error("Invalid m/nc");
  const std::vector<int> bits_per_layer = ExpandBitsPerLayer(m, cfg.bits_per_layer);
  if (a.size() != std::size_t(m) * std::size_t(nc)) throw std::runtime_error("a size mismatch");
  if (!B.empty() && B.size() != std::size_t(m) * std::size_t(nc)) throw std::runtime_error("B size mismatch");
  if (!is_linkage.empty() && is_linkage.size() != std::size_t(nc)) throw std::runtime_error("is_linkage size mismatch");
  if (q_out.size() != std::size_t(m) * std::size_t(nc)) throw std::runtime_error("q_out size mismatch");

  GroupView root, linkage;
  root.idx.reserve(static_cast<std::size_t>(nc));
  linkage.idx.reserve(static_cast<std::size_t>(nc));
  for (int i = 0; i < nc; ++i) {
    const bool ilinkage = (!is_linkage.empty() && is_linkage[static_cast<std::size_t>(i)] != 0);
    (ilinkage ? linkage.idx : root.idx).push_back(i);
  }

  result_root->scales.assign(m, 1.0f);
  result_root->stats.assign(m, CoeffQuantLayerStats{});
  result_linkage->scales.assign(m, 1.0f);
  result_linkage->stats.assign(m, CoeffQuantLayerStats{});

  // Initialize q_out to 0.
  std::fill(q_out.begin(), q_out.end(), std::int8_t(0));

  for (int layer = 0; layer < m; ++layer) {
    const int bits = bits_per_layer[static_cast<std::size_t>(layer)];
    if (bits < 2 || bits > 8) throw std::runtime_error("bits must be in [2,8]");

    QuantizeOneGroupOneLayer(layer, bits, nc, a, B, root, norm2_root_layers, cfg, q_out,
                            &result_root->stats[static_cast<std::size_t>(layer)]);
    result_root->scales[static_cast<std::size_t>(layer)] =
        result_root->stats[static_cast<std::size_t>(layer)].scale;

    QuantizeOneGroupOneLayer(layer, bits, nc, a, B, linkage, norm2_linkage_layers, cfg, q_out,
                            &result_linkage->stats[static_cast<std::size_t>(layer)]);
    result_linkage->scales[static_cast<std::size_t>(layer)] =
        result_linkage->stats[static_cast<std::size_t>(layer)].scale;
  }
}

}  // namespace stlq
