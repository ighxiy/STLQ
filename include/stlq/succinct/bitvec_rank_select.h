#pragma once

#include <cstdint>
#include <vector>
#include <stdexcept>
#include <limits>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace stlq {
namespace succinct {

namespace detail {
inline uint32_t Popcount64(std::uint64_t x) {
#if defined(_MSC_VER)
  return static_cast<uint32_t>(__popcnt64(static_cast<unsigned __int64>(x)));
#else
  return static_cast<uint32_t>(__builtin_popcountll(x));
#endif
}

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
}  // namespace detail

// A compact bitvector with O(1) rank1 and O(1) select1 (constant bounded by select sampling stride).
//
// Storage:
// - bits stored in uint64_t words.
// - rank directory: superblocks + per-word offsets.
// - select1 sampling: position of every S-th 1.
//
// All indices are 0-based bit positions.
class RankSelectBitVector {
 public:
  RankSelectBitVector() = default;

  void Reset() {
    n_bits_ = 0;
    words_.clear();
    super_rank_.clear();
    word_rank_in_super_bits_.clear();
    select1_samples_.clear();
    ones_ = 0;
    select_stride_ = 128;
    words_per_super_log2_ = 4;
    words_per_super_ = 1u << words_per_super_log2_;
    word_rank_bits_ = 0;
  }

  void BuildFromWords(std::vector<uint64_t> words,
                      uint32_t n_bits,
                      uint32_t select_stride = 128,
                      uint32_t words_per_super_log2 = 4,
                      bool build_indices = true) {
    if (n_bits == 0) {
      Reset();
      return;
    }
    if (select_stride == 0) {
      throw std::invalid_argument("select_stride must be > 0");
    }
    if (words_per_super_log2 == 0 || words_per_super_log2 > 10) {
      throw std::invalid_argument("words_per_super_log2 must be in [1..10]");
    }
    n_bits_ = n_bits;
    words_ = std::move(words);
    select_stride_ = select_stride;
    words_per_super_log2_ = words_per_super_log2;
    words_per_super_ = 1u << words_per_super_log2_;
    super_rank_.clear();
    word_rank_in_super_bits_.clear();
    select1_samples_.clear();
    ones_ = 0;
    word_rank_bits_ = 0;
    if (build_indices) {
      BuildRankIndex_();
      BuildSelect1Samples_();
    }
  }

  uint32_t n_bits() const { return n_bits_; }
  uint64_t ones() const { return ones_; }
  uint32_t select_stride() const { return select_stride_; }
  uint32_t words_per_super_log2() const { return words_per_super_log2_; }

  const std::vector<uint64_t>& words() const { return words_; }

  // Return number of 1s in B[0..pos] inclusive.
  uint64_t Rank1(uint32_t pos) const {
    if (pos >= n_bits_) {
      throw std::out_of_range("Rank1 pos out of range");
    }
    if (super_rank_.empty() || word_rank_in_super_bits_.empty()) {
      throw std::runtime_error("Rank1 requires rank index (build_indices=true)");
    }
    const uint32_t word_idx = pos >> 6;
    const uint32_t bit_in_word = pos & 63u;
    const uint32_t super_idx = word_idx >> words_per_super_log2_;

    const uint64_t base =
      static_cast<uint64_t>(super_rank_[super_idx]) + static_cast<uint64_t>(WordRankInSuperAt_(word_idx));
    const uint64_t w = words_[word_idx];
    const uint64_t mask = (bit_in_word == 63u) ? std::numeric_limits<uint64_t>::max() : ((1ULL << (bit_in_word + 1)) - 1ULL);
    return base + static_cast<uint64_t>(detail::Popcount64(w & mask));
  }

  // Return number of 0s in B[0..pos] inclusive.
  uint64_t Rank0(uint32_t pos) const {
    return static_cast<uint64_t>(pos) + 1ULL - Rank1(pos);
  }

  // Return the 0-based bit position of the k-th 1 (1-indexed k).
  uint32_t Select1(uint64_t k) const {
    if (select1_samples_.empty()) {
      throw std::runtime_error("Select1 requires select index (build_indices=true)");
    }
    if (k == 0 || k > ones_) {
      throw std::out_of_range("Select1 k out of range");
    }
    const uint64_t bucket = (k - 1) / static_cast<uint64_t>(select_stride_);
    const uint64_t base_rank = bucket * static_cast<uint64_t>(select_stride_);

    uint32_t pos = select1_samples_[static_cast<size_t>(bucket)];  // position of (base_rank+1)-th 1
    uint64_t remaining = k - base_rank;  // include the 1 at pos

    // Scan starting at pos.
    uint32_t word_idx = pos >> 6;
    uint32_t bit_in_word = pos & 63u;

    // First word: mask out bits before pos.
    uint64_t w = words_[word_idx] & (~0ULL << bit_in_word);
    uint32_t pc = detail::Popcount64(w);
    if (remaining <= pc) {
      return static_cast<uint32_t>(word_idx * 64u + Select1InWord_(w, static_cast<uint32_t>(remaining)));
    }
    remaining -= pc;
    word_idx++;

    // Subsequent words.
    const uint32_t n_words = static_cast<uint32_t>(words_.size());
    for (; word_idx < n_words; ++word_idx) {
      w = words_[word_idx];
      pc = detail::Popcount64(w);
      if (remaining <= pc) {
        return static_cast<uint32_t>(word_idx * 64u + Select1InWord_(w, static_cast<uint32_t>(remaining)));
      }
      remaining -= pc;
    }
    // Should never reach here if indices are correct.
    throw std::runtime_error("Select1 internal error");
  }

 private:
  uint32_t n_bits_ = 0;
  std::vector<uint64_t> words_;

  // rank directory
  std::vector<uint32_t> super_rank_;          // cumulative ones at each superblock start
  std::vector<uint8_t> word_rank_in_super_bits_;  // bit-packed cumulative ones within superblock up to this word
  uint8_t word_rank_bits_ = 0;

  // select sampling
  std::vector<uint32_t> select1_samples_;
  uint64_t ones_ = 0;
  uint32_t select_stride_ = 128;
  uint32_t words_per_super_log2_ = 4;  // 16 words => 1024 bits
  uint32_t words_per_super_ = 16;

  static uint32_t Select1InWord_(uint64_t w, uint32_t k) {
    // Return bit index in [0..63] of k-th 1 in w (1-indexed).
    // k is guaranteed to be <= popcount(w).
    while (true) {
      const uint32_t tz = detail::Ctz64NonZero(w);
      if (--k == 0) {
        return tz;
      }
      w &= (w - 1);
    }
  }

  static uint8_t BitsForWordRankInSuper_(uint32_t words_per_super) {
    if (words_per_super <= 1) {
      return 1;
    }
    const uint32_t max_within = 64u * (words_per_super - 1u);
    uint8_t bits = 0;
    uint32_t x = max_within;
    while (x > 0u) {
      ++bits;
      x >>= 1u;
    }
    return (bits == 0) ? uint8_t(1) : bits;
  }

  void SetWordRankInSuperAt_(uint32_t idx, uint32_t value) {
    const uint64_t bit_off = static_cast<uint64_t>(idx) * static_cast<uint64_t>(word_rank_bits_);
    const uint32_t byte_off = static_cast<uint32_t>(bit_off >> 3);
    const uint32_t bit_in_byte = static_cast<uint32_t>(bit_off & 7u);
    const uint32_t need_bits = static_cast<uint32_t>(word_rank_bits_) + bit_in_byte;
    const uint32_t need_bytes = (need_bits + 7u) >> 3;
    uint32_t chunk = 0;
    for (uint32_t i = 0; i < need_bytes; ++i) {
      chunk |= static_cast<uint32_t>(word_rank_in_super_bits_[static_cast<size_t>(byte_off + i)]) << (8u * i);
    }
    const uint32_t mask = (word_rank_bits_ >= 32) ? 0xffffffffu : ((1u << word_rank_bits_) - 1u);
    chunk &= ~(mask << bit_in_byte);
    chunk |= (value & mask) << bit_in_byte;
    for (uint32_t i = 0; i < need_bytes; ++i) {
      word_rank_in_super_bits_[static_cast<size_t>(byte_off + i)] = static_cast<uint8_t>((chunk >> (8u * i)) & 0xffu);
    }
  }

  uint32_t WordRankInSuperAt_(uint32_t idx) const {
    const uint64_t bit_off = static_cast<uint64_t>(idx) * static_cast<uint64_t>(word_rank_bits_);
    const uint32_t byte_off = static_cast<uint32_t>(bit_off >> 3);
    const uint32_t bit_in_byte = static_cast<uint32_t>(bit_off & 7u);
    const uint32_t need_bits = static_cast<uint32_t>(word_rank_bits_) + bit_in_byte;
    const uint32_t need_bytes = (need_bits + 7u) >> 3;
    uint32_t chunk = 0;
    for (uint32_t i = 0; i < need_bytes; ++i) {
      chunk |= static_cast<uint32_t>(word_rank_in_super_bits_[static_cast<size_t>(byte_off + i)]) << (8u * i);
    }
    const uint32_t mask = (word_rank_bits_ >= 32) ? 0xffffffffu : ((1u << word_rank_bits_) - 1u);
    return (chunk >> bit_in_byte) & mask;
  }

  void BuildRankIndex_() {
    const uint32_t n_words = static_cast<uint32_t>(words_.size());
    const uint32_t n_supers = (n_words + words_per_super_ - 1) / words_per_super_;

    super_rank_.assign(n_supers + 1, 0);
    word_rank_bits_ = BitsForWordRankInSuper_(words_per_super_);
    const uint64_t total_rank_bits = static_cast<uint64_t>(n_words) * static_cast<uint64_t>(word_rank_bits_);
    word_rank_in_super_bits_.assign(static_cast<size_t>((total_rank_bits + 7u) >> 3), 0);

    uint64_t total = 0;
    for (uint32_t s = 0; s < n_supers; ++s) {
      super_rank_[s] = static_cast<uint32_t>(total);
      uint64_t within = 0;
      const uint32_t start = s * words_per_super_;
      const uint32_t end = (start + words_per_super_ < n_words) ? (start + words_per_super_) : n_words;
      for (uint32_t w = start; w < end; ++w) {
        SetWordRankInSuperAt_(w, static_cast<uint32_t>(within));
        within += static_cast<uint64_t>(detail::Popcount64(words_[w]));
      }
      total += within;
    }
    super_rank_[n_supers] = static_cast<uint32_t>(total);
    ones_ = total;
  }

  void BuildSelect1Samples_() {
    select1_samples_.clear();
    if (ones_ == 0) {
      return;
    }
    const uint64_t stride = static_cast<uint64_t>(select_stride_);
    const uint64_t buckets = (ones_ + stride - 1) / stride;
    select1_samples_.assign(static_cast<size_t>(buckets), 0);

    uint64_t next_rank = 1;  // we store position of rank 1, 1+stride, 1+2*stride, ...
    uint64_t bucket = 0;

    const uint32_t n_words = static_cast<uint32_t>(words_.size());
    uint64_t seen = 0;
    for (uint32_t wi = 0; wi < n_words && bucket < buckets; ++wi) {
      uint64_t w = words_[wi];
      while (w != 0 && bucket < buckets) {
        const uint32_t tz = detail::Ctz64NonZero(w);
        ++seen;
        if (seen == next_rank) {
          select1_samples_[static_cast<size_t>(bucket)] = wi * 64u + tz;
          ++bucket;
          next_rank = 1 + bucket * stride;
        }
        w &= (w - 1);
      }
    }
    if (bucket != buckets) {
      throw std::runtime_error("BuildSelect1Samples failed");
    }
  }
};

}  // namespace succinct
}  // namespace stlq
