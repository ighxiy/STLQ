#include "stlq/coeff/coeff_cluster_codec.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace stlq {

namespace {

inline int QFromBits(int bits) {
  return (1 << (bits - 1)) - 1;
}

inline std::uint16_t SymFromQ(std::int8_t q, int base) {
  return static_cast<std::uint16_t>(static_cast<int>(q) + base);
}

inline std::int8_t QFromSym(std::uint16_t sym, int base) {
  return std::int8_t(int(sym) - base);
}

struct GroupIndices {
  std::vector<int> root;
  std::vector<int> linkage;
};

GroupIndices SplitRootLinkage(int nc, Span<const std::uint8_t> is_linkage) {
  GroupIndices g;
  g.root.reserve(static_cast<std::size_t>(nc));
  g.linkage.reserve(static_cast<std::size_t>(nc));
  for (int i = 0; i < nc; ++i) {
    const bool ilinkage = (!is_linkage.empty() && is_linkage[static_cast<std::size_t>(i)] != 0);
    (ilinkage ? g.linkage : g.root).push_back(i);
  }
  return g;
}

std::vector<std::uint32_t> BuildFreqAllLayers(
    int m,
    int nc,
    Span<const std::int8_t> q,
    const std::vector<int>& idx,
    int base,
    int S) {
  std::vector<std::uint32_t> freq(static_cast<std::size_t>(S), 0);
  for (int l = 0; l < m; ++l) {
    const std::int8_t* ql = q.data() + std::size_t(l) * nc;
    for (int ii : idx) {
      const std::uint16_t sym = SymFromQ(ql[ii], base);
      freq[static_cast<std::size_t>(sym)] += 1;
    }
  }
  return freq;
}

std::vector<std::uint32_t> BuildFreqOneLayer(
    int nc,
    const std::int8_t* q_layer,
    const std::vector<int>& idx,
    int base,
    int S) {
  (void)nc;
  std::vector<std::uint32_t> freq(static_cast<std::size_t>(S), 0);
  for (int ii : idx) {
    const std::uint16_t sym = SymFromQ(q_layer[ii], base);
    freq[static_cast<std::size_t>(sym)] += 1;
  }
  return freq;
}

std::vector<std::uint8_t> EncodeAllLayers(
    int m,
    int nc,
    Span<const std::int8_t> q,
    const std::vector<int>& idx,
    int base,
    const HuffmanCanonicalModel& model) {
  BitWriter bw;
  bw.Reset();
  for (int l = 0; l < m; ++l) {
    const std::int8_t* ql = q.data() + std::size_t(l) * nc;
    for (int ii : idx) {
      model.EncodeSymbol(bw, SymFromQ(ql[ii], base));
    }
  }
  return bw.Finish();
}

std::vector<std::uint8_t> EncodeOneLayer(
    int nc,
    const std::int8_t* q_layer,
    const std::vector<int>& idx,
    int base,
    const HuffmanCanonicalModel& model) {
  (void)nc;
  BitWriter bw;
  bw.Reset();
  for (int ii : idx) {
    model.EncodeSymbol(bw, SymFromQ(q_layer[ii], base));
  }
  return bw.Finish();
}

void DecodeAllLayers(
    int m,
    int nc,
    Span<std::int8_t> q_out,
    const std::vector<int>& idx,
    int base,
    const HuffmanCanonicalModel& model,
    const std::vector<std::uint8_t>& payload) {
  BitReader br(payload.data(), payload.size());
  for (int l = 0; l < m; ++l) {
    std::int8_t* ql = q_out.data() + std::size_t(l) * nc;
    for (int ii : idx) {
      const std::uint16_t sym = model.DecodeSymbol(br);
      ql[ii] = QFromSym(sym, base);
    }
  }
}

void DecodeOneLayer(
    int nc,
    std::int8_t* q_layer_out,
    const std::vector<int>& idx,
    int base,
    const HuffmanCanonicalModel& model,
    const std::vector<std::uint8_t>& payload) {
  (void)nc;
  BitReader br(payload.data(), payload.size());
  for (int ii : idx) {
    const std::uint16_t sym = model.DecodeSymbol(br);
    q_layer_out[ii] = QFromSym(sym, base);
  }
}

}  // namespace

ClusterCoeffCompressed QuantizeAndCompressClusterCoeffs(
    std::uint32_t cluster_id,
    int m,
    int nc,
    Span<const float> a,
    Span<const std::uint32_t> B,
    Span<const std::uint8_t> is_linkage,
    const std::vector<Span<const float>>& norm2_root_layers,
    const std::vector<Span<const float>>& norm2_linkage_layers,
    const CoeffQuantConfig& qcfg,
    const HuffmanConfig& hcfg,
    Span<std::int8_t> q_tmp,
    IcmIlsRefiner* refiner) {

  if (m <= 0 || nc < 0) throw std::runtime_error("Invalid m/nc");
  if (a.size() != std::size_t(m) * std::size_t(nc)) throw std::runtime_error("a size mismatch");
  if (q_tmp.size() != std::size_t(m) * std::size_t(nc)) throw std::runtime_error("q_tmp size mismatch");

  ClusterCoeffCompressed comp;
  comp.cluster_id = cluster_id;
  comp.n_in_cluster = static_cast<std::uint32_t>(nc);

  // 1) Quantize.
  CoeffQuantGroupResult qr, qc;
  QuantizeClusterCoeffsSplitRootLinkage(
      m, nc, a, B, is_linkage, norm2_root_layers, norm2_linkage_layers, qcfg, q_tmp, &qr, &qc);
  comp.scale_root = qr.scales;
  comp.scale_linkage = qc.scales;

  // Optional refinement hook.
  if (refiner) {
    refiner->RefineCluster(static_cast<int>(cluster_id), m, nc, is_linkage, q_tmp,
                           Span<float>(comp.scale_root.data(), comp.scale_root.size()),
                           Span<float>(comp.scale_linkage.data(), comp.scale_linkage.size()));
  }

  const GroupIndices groups = SplitRootLinkage(nc, is_linkage);
  comp.n_root = static_cast<std::uint32_t>(groups.root.size());
  comp.n_linkage = static_cast<std::uint32_t>(groups.linkage.size());

  // 2) Huffman (lossless).
  int Qmax = 0;
  for (int bits : qcfg.bits_per_layer) Qmax = std::max(Qmax, QFromBits(bits));
  int S = 0;
  if (hcfg.alphabet_size == 256) {
    Qmax = 127;
    S = 256;
  } else {
    S = (hcfg.alphabet_size > 0) ? hcfg.alphabet_size : (2 * Qmax + 1);
  }
  const int base = (S == 256) ? 128 : Qmax;

  auto encode_group = [&](const std::vector<int>& idx,
                          ClusterGroupCompressed* out) {
    out->models.clear();
    out->models_serialized.clear();
    out->payloads.clear();
    out->num_symbols.clear();

    if (idx.empty()) {
      return;
    }

    if (hcfg.granularity == HuffmanGranularity::kPerClusterAllLayers) {
      auto freq = BuildFreqAllLayers(m, nc, q_tmp, idx, base, S);
      auto model = BuildHuffmanCanonicalModel(freq);
      // Serialize just lengths (S bytes) in the real store; keep full model here.
      out->models.push_back(model);
      out->models_serialized.push_back(std::move(model.len));
      out->payloads.push_back(EncodeAllLayers(m, nc, q_tmp, idx, base, out->models.back()));
      out->num_symbols.push_back(static_cast<std::uint32_t>(idx.size() * std::size_t(m)));
    } else {
      out->models.resize(static_cast<std::size_t>(m));
      out->models_serialized.resize(static_cast<std::size_t>(m));
      out->payloads.resize(static_cast<std::size_t>(m));
      out->num_symbols.resize(static_cast<std::size_t>(m));
      for (int l = 0; l < m; ++l) {
        const std::int8_t* ql = q_tmp.data() + std::size_t(l) * nc;
        auto freq = BuildFreqOneLayer(nc, ql, idx, base, S);
        out->models[static_cast<std::size_t>(l)] = BuildHuffmanCanonicalModel(freq);
        out->models_serialized[static_cast<std::size_t>(l)] = out->models[static_cast<std::size_t>(l)].len;
        out->payloads[static_cast<std::size_t>(l)] =
            EncodeOneLayer(nc, ql, idx, base, out->models[static_cast<std::size_t>(l)]);
        out->num_symbols[static_cast<std::size_t>(l)] = static_cast<std::uint32_t>(idx.size());
      }
    }
  };

  encode_group(groups.root, &comp.root);
  encode_group(groups.linkage, &comp.linkage);
  return comp;
}

ClusterCoeffCompressed CompressClusterCoeffsFromQ(
    std::uint32_t cluster_id,
    int m,
    int nc,
    Span<const std::int8_t> q,
    Span<const std::uint8_t> is_linkage,
    const std::vector<float>& scale_root,
    const std::vector<float>& scale_linkage,
    const CoeffQuantConfig& qcfg,
    const HuffmanConfig& hcfg) {
  if (m <= 0 || nc < 0) throw std::runtime_error("Invalid m/nc");
  if (q.size() != std::size_t(m) * std::size_t(nc)) throw std::runtime_error("q size mismatch");
  if (int(scale_root.size()) != m || int(scale_linkage.size()) != m) throw std::runtime_error("scale size mismatch");

  ClusterCoeffCompressed comp;
  comp.cluster_id = cluster_id;
  comp.n_in_cluster = static_cast<std::uint32_t>(nc);
  comp.scale_root = scale_root;
  comp.scale_linkage = scale_linkage;

  const GroupIndices groups = SplitRootLinkage(nc, is_linkage);
  comp.n_root = static_cast<std::uint32_t>(groups.root.size());
  comp.n_linkage = static_cast<std::uint32_t>(groups.linkage.size());

  // Huffman (lossless).
  int Qmax = 0;
  for (int bits : qcfg.bits_per_layer) Qmax = std::max(Qmax, QFromBits(bits));
  int S = 0;
  if (hcfg.alphabet_size == 256) {
    Qmax = 127;
    S = 256;
  } else {
    S = (hcfg.alphabet_size > 0) ? hcfg.alphabet_size : (2 * Qmax + 1);
  }
  const int base = (S == 256) ? 128 : Qmax;

  auto encode_group = [&](const std::vector<int>& idx,
                          ClusterGroupCompressed* out) {
    out->models.clear();
    out->models_serialized.clear();
    out->payloads.clear();
    out->num_symbols.clear();

    if (idx.empty()) {
      return;
    }

    if (hcfg.granularity == HuffmanGranularity::kPerClusterAllLayers) {
      auto freq = BuildFreqAllLayers(m, nc, Span<const std::int8_t>(q.data(), q.size()), idx, base, S);
      auto model = BuildHuffmanCanonicalModel(freq);
      out->models.push_back(model);
      out->models_serialized.push_back(std::move(model.len));
      out->payloads.push_back(EncodeAllLayers(m, nc, Span<const std::int8_t>(q.data(), q.size()),
                                              idx, base, out->models.back()));
      out->num_symbols.push_back(static_cast<std::uint32_t>(idx.size() * std::size_t(m)));
    } else {
      out->models.resize(static_cast<std::size_t>(m));
      out->models_serialized.resize(static_cast<std::size_t>(m));
      out->payloads.resize(static_cast<std::size_t>(m));
      out->num_symbols.resize(static_cast<std::size_t>(m));
      for (int l = 0; l < m; ++l) {
        const std::int8_t* ql = q.data() + std::size_t(l) * nc;
        auto freq = BuildFreqOneLayer(nc, ql, idx, base, S);
        out->models[static_cast<std::size_t>(l)] = BuildHuffmanCanonicalModel(freq);
        out->models_serialized[static_cast<std::size_t>(l)] = out->models[static_cast<std::size_t>(l)].len;
        out->payloads[static_cast<std::size_t>(l)] =
            EncodeOneLayer(nc, ql, idx, base, out->models[static_cast<std::size_t>(l)]);
        out->num_symbols[static_cast<std::size_t>(l)] = static_cast<std::uint32_t>(idx.size());
      }
    }
  };

  encode_group(groups.root, &comp.root);
  encode_group(groups.linkage, &comp.linkage);
  return comp;
}

void DecodeClusterCoeffs(
    const ClusterCoeffCompressed& comp,
    int m,
    int nc,
    Span<const std::uint8_t> is_linkage,
    Span<std::int8_t> q_out) {

  if (q_out.size() != std::size_t(m) * std::size_t(nc)) {
    throw std::runtime_error("DecodeClusterCoeffs: q_out size mismatch");
  }
  std::fill(q_out.begin(), q_out.end(), std::int8_t(0));

  const GroupIndices groups = SplitRootLinkage(nc, is_linkage);

  auto decode_group = [&](const ClusterGroupCompressed& gcomp,
                          const std::vector<int>& idx) {
    if (idx.empty()) return;
    if (gcomp.payloads.empty() || gcomp.models.empty()) return;

    // NOTE: models_serialized stores len only (S bytes). Rebuild model tables here.
    if (gcomp.payloads.size() == 1) {
      HuffmanCanonicalModel model = gcomp.models[0];
      DecodeAllLayers(m, nc, q_out, idx,
                      /*base=*/(model.S == 256 ? 128 : int(model.S / 2)),
                      model, gcomp.payloads[0]);
      return;
    }

    // Per-layer.
    for (int l = 0; l < m; ++l) {
      HuffmanCanonicalModel model = gcomp.models[static_cast<std::size_t>(l)];
      std::int8_t* ql = q_out.data() + std::size_t(l) * nc;
      DecodeOneLayer(nc, ql, idx,
                     /*base=*/(model.S == 256 ? 128 : int(model.S / 2)),
                     model, gcomp.payloads[static_cast<std::size_t>(l)]);
    }
  };

  decode_group(comp.root, groups.root);
  decode_group(comp.linkage, groups.linkage);
}

}  // namespace stlq
