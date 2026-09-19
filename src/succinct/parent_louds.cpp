#include "stlq/succinct/parent_louds.h"

#include <algorithm>
#include <cstring>
#include <limits>

#if defined(_MSC_VER)
#include <intrin.h>
#endif


namespace stlq::succinct {

namespace {
constexpr uint32_t kMagic = 0x53444C50u;  // 'PLDS'
constexpr uint32_t kVersionV1 = 1;

inline uint32_t Ctz64NonZero(std::uint64_t x) {
  // Caller must ensure x != 0.
#if defined(_MSC_VER)
  unsigned long idx = 0;
  _BitScanForward64(&idx, static_cast<unsigned __int64>(x));
  return static_cast<uint32_t>(idx);
#else
  return static_cast<uint32_t>(__builtin_ctzll(x));
#endif
}

#pragma pack(push, 1)
struct Header {
  uint32_t magic;
  uint32_t version;
  uint64_t n_nodes;
  uint64_t n_bits;
  uint32_t select_stride;
  uint32_t reserved0;
  uint64_t n_words;
};
#pragma pack(pop)

inline void AppendBytes(std::vector<uint8_t>& out, const void* data, size_t len) {
  if (len == 0) return;
  const size_t old = out.size();
  out.resize(old + len);
  std::memcpy(out.data() + old, data, len);
}

inline void ReadBytesPtr(const uint8_t* data, size_t len, size_t& off, void* out, size_t n) {
  if (off + n > len) throw std::runtime_error("ParentLOUDS: blob truncated");
  std::memcpy(out, data + off, n);
  off += n;
}

}  // namespace

uint64_t ParentLOUDS::SerializedBytesForNodes(uint32_t n_nodes) {
  const auto n = static_cast<uint64_t>(n_nodes);
  const uint64_t n_bits = 2ull * n + 1ull;
  const uint64_t n_words = (n_bits + 63ull) / 64ull;
  return static_cast<uint64_t>(sizeof(Header)) + n_words * sizeof(uint64_t);
}

std::vector<uint64_t> ParentLOUDS::BuildLoudsWordsFromDegrees(const std::vector<uint32_t>& deg, size_t total_bits) {
  std::vector<uint64_t> words((total_bits + 63) / 64, 0ull);

  size_t pos = 0;  // bit position in [0..total_bits)
  for (size_t u = 0; u < deg.size(); ++u) {
    uint32_t remain = deg[u];
    while (remain > 0) {
      const size_t wi = pos >> 6;
      const auto bi = static_cast<uint32_t>(pos & 63u);
      const uint32_t room = 64u - bi;
      const uint32_t take = (remain < room) ? remain : room;
      if (take == 64u) {
        // bi == 0 in this case.
        words[wi] = ~0ull;
      } else {
        const uint64_t mask = ((take == 64u) ? ~0ull : ((1ull << take) - 1ull)) << bi;
        words[wi] |= mask;
      }
      pos += static_cast<size_t>(take);
      remain -= take;
    }
    // write 0 delimiter (bit already 0)
    ++pos;
  }
  if (pos != total_bits) {
    throw std::runtime_error("ParentLOUDS: internal error building LOUDS bitvector");
  }
  return words;
}

void ParentLOUDS::BuildFromParent1Based(const uint32_t* parent_1based,
                                        size_t n,
                                        uint32_t select_stride,
                                        uint32_t rank_words_per_super_log2,
                                        bool build_indices) {
  if (n == 0) {
    // deg(super)=0 => bitvector is single '0'
    n_ = 0;
    std::vector<uint64_t> words(1, 0ull);
    bv_.BuildFromWords(std::move(words), 1u, select_stride, rank_words_per_super_log2, build_indices);
    return;
  }

  // degrees for nodes: 0 is super-root, 1..n correspond to input nodes in stored order.
  std::vector<uint32_t> deg(n + 1, 0);
  for (size_t i = 0; i < n; ++i) {
    const uint32_t p = parent_1based[i];
    if (p > n) {
      throw std::runtime_error("ParentLOUDS: parent_1based out of range");
    }
    // p == 0 means root under super-root
    deg[p] += 1;
  }

  // LOUDS length: 2n+1 bits (edges=n, nodes=n+1)
  const size_t total_bits = 2 * n + 1;
  std::vector<uint64_t> words = BuildLoudsWordsFromDegrees(deg, total_bits);

  n_ = n;
  bv_.BuildFromWords(std::move(words),
                     static_cast<uint32_t>(total_bits),
                     select_stride,
                     rank_words_per_super_log2,
                     build_indices);
}

void ParentLOUDS::DecodeParent1Based(std::vector<uint32_t>* out, size_t n_out) const {
  if (!out) {
    throw std::runtime_error("ParentLOUDS: DecodeParent1Based: null out");
  }
  if (n_out > n_) {
    throw std::runtime_error("ParentLOUDS: DecodeParent1Based: n_out exceeds n_nodes");
  }
  out->resize(n_out);
  if (n_out == 0) {
    return;
  }

  // Full decode without rank/select:
  // In unary LOUDS, each node emits `deg` ones followed by a zero. Therefore,
  // degree(node_i) = zero_pos[i] - zero_pos[i-1] - 1 (with zero_pos[-1] = -1).
  //
  // We only need the positions of zeros, which we can enumerate word-wise.
  const auto& words = bv_.words();
  const auto n_bits = static_cast<size_t>(bv_.n_bits());
  if (n_bits != (2 * n_ + 1)) {
    throw std::runtime_error("ParentLOUDS: invalid LOUDS bitvector length");
  }

  int64_t prev_zero = -1;
  uint32_t node_id = 0;      // [0..n_] (0 is super-root)
  uint32_t next_child = 1;   // node ids of children in [1..n_]
  const auto n_nodes = static_cast<uint32_t>(n_);
  const size_t n_words = words.size();

  for (size_t wi = 0; wi < n_words && node_id <= n_nodes; ++wi) {
    uint64_t w = words[wi];
    uint64_t mask = ~0ull;
    const size_t base = wi * 64ull;
    if (base + 64ull > n_bits) {
      const auto tail = static_cast<uint32_t>(n_bits - base);
      mask = (tail == 64u) ? ~0ull : ((1ull << tail) - 1ull);
      w &= mask;
    }
    uint64_t z = (~w) & mask;  // positions of zeros within this word
    while (z != 0ull && node_id <= n_nodes) {
      const uint32_t b = Ctz64NonZero(z);
      const auto zpos = static_cast<int64_t>(base + static_cast<size_t>(b));
      const int64_t deg = zpos - prev_zero - 1;
      if (deg < 0) {
        throw std::runtime_error("ParentLOUDS: decode failed (negative degree)");
      }
      for (int64_t k = 0; k < deg; ++k) {
        if (next_child > n_nodes) break;
        if (static_cast<size_t>(next_child - 1u) < n_out) {
          (*out)[static_cast<size_t>(next_child - 1u)] = node_id;
        }
        ++next_child;
      }
      prev_zero = zpos;
      ++node_id;
      z &= (z - 1ull);
    }
  }

#ifndef NDEBUG
  if (node_id != n_nodes + 1u) {
    throw std::runtime_error("ParentLOUDS: decode failed (zero count mismatch)");
  }
  if (next_child != n_nodes + 1u) {
    throw std::runtime_error("ParentLOUDS: decode failed (child count mismatch)");
  }
#endif
}

ParentLOUDS::SequentialParentDecoder::SequentialParentDecoder(const ParentLOUDS& louds)
    : words_(&louds.bitvector().words()) {}

void ParentLOUDS::SequentialParentDecoder::Skip(size_t n) {
  while (n != 0) {
    while (one_positions_ == 0) {
      word_base_ = static_cast<uint64_t>(next_word_) * 64u;
      one_positions_ = (*words_)[next_word_++];
    }
    const uint32_t available = detail::Popcount64(one_positions_);
    const size_t consumed = std::min(n, static_cast<size_t>(available));
    if (consumed == available) {
      one_positions_ = 0;
    } else {
      for (size_t child = 0; child < consumed; ++child) {
        one_positions_ &= one_positions_ - 1u;
      }
    }
    n -= consumed;
    emitted_children_ += static_cast<uint32_t>(consumed);
  }
}

void ParentLOUDS::SequentialParentDecoder::InitializeAfterValidatedRootPrefix(
    size_t root_count) {
  // A validated canonical stream starts with root_count one bits and the
  // implicit-super-root zero delimiter. Position at the first linked child
  // without enumerating that known prefix.
  const uint64_t next_bit = static_cast<uint64_t>(root_count) + 1u;
  const size_t word = static_cast<size_t>(next_bit >> 6u);
  const uint32_t offset = static_cast<uint32_t>(next_bit & 63u);
  word_base_ = static_cast<uint64_t>(word) * 64u;
  one_positions_ = (*words_)[word] & (~uint64_t{0} << offset);
  next_word_ = word + 1u;
  emitted_children_ = static_cast<uint32_t>(root_count);
}

uint32_t ParentLOUDS::Parent1BasedOf(size_t pos) const {
  if (pos >= n_) {
    throw std::runtime_error("ParentLOUDS: pos out of range");
  }
  // node ids in LOUDS: 0=super-root, 1..n are input nodes.
  const size_t v = pos + 1;  // 1..n
  const size_t one_pos = bv_.Select1(v);  // position of v-th '1'
  const size_t parent_node = bv_.Rank0(one_pos);  // returns parent node id in [0..n]
  return static_cast<uint32_t>(parent_node);
}

std::vector<uint8_t> ParentLOUDS::Serialize() const {
  Header h{};
  h.magic = kMagic;
  h.version = kVersionV1;
  h.n_nodes = static_cast<uint64_t>(n_);
  h.n_bits = static_cast<uint64_t>(bv_.n_bits());
  h.select_stride = bv_.select_stride();
  h.reserved0 = 0;
  h.n_words = static_cast<uint64_t>(bv_.words().size());

  std::vector<uint8_t> out;
  out.reserve(sizeof(Header) + bv_.words().size() * sizeof(uint64_t));
  AppendBytes(out, &h, sizeof(h));
  AppendBytes(out, bv_.words().data(), bv_.words().size() * sizeof(uint64_t));
  return out;
}

void ParentLOUDS::Deserialize(const uint8_t* data,
                              size_t len,
                              uint32_t rank_words_per_super_log2,
                              bool build_indices,
                              uint32_t select_stride_override) {
  if (len < sizeof(Header)) throw std::runtime_error("ParentLOUDS: blob too small");
  size_t off = 0;
  Header h1{};
  ReadBytesPtr(data, len, off, &h1, sizeof(h1));

  if (h1.magic != kMagic) throw std::runtime_error("ParentLOUDS: bad magic");
  if (h1.version == kVersionV1) {
    const auto n = static_cast<size_t>(h1.n_nodes);
    const auto n_bits = static_cast<size_t>(h1.n_bits);
    const auto n_words = static_cast<size_t>(h1.n_words);

    if (n_bits != (2 * n + 1)) {
      throw std::runtime_error("ParentLOUDS: invalid n_bits (expected 2n+1)");
    }
    if (n_words != (n_bits + 63) / 64) {
      throw std::runtime_error("ParentLOUDS: invalid n_words");
    }
    const size_t bytes_words = n_words * sizeof(uint64_t);
    if (off + bytes_words != len) {
      throw std::runtime_error("ParentLOUDS: blob size mismatch");
    }

    std::vector<uint64_t> words(n_words);
    std::memcpy(words.data(), data + off, bytes_words);
    const uint32_t select_stride =
        (select_stride_override > 0) ? select_stride_override : static_cast<uint32_t>(h1.select_stride);

    n_ = n;
    bv_.BuildFromWords(std::move(words),
                       static_cast<uint32_t>(n_bits),
                       select_stride,
                       rank_words_per_super_log2,
                       build_indices);
    return;
  }

  throw std::runtime_error("ParentLOUDS: unsupported version");
}

} // namespace stlq::succinct
