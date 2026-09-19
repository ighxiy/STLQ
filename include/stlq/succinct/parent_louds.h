#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <stdexcept>

#include "stlq/succinct/bitvec_rank_select.h"

namespace stlq {
namespace succinct {

// ParentLOUDS: succinct parent representation for a per-cluster forest.
//
// Input parent convention (matches your eval pipeline):
//   - parent_1based[i] == 0 means root (super-root).
//   - otherwise parent_1based[i] == parent_pos + 1, where parent_pos is 0-based
//     index of the parent within THIS cluster's node sequence.
//
// NOTE: The cluster's node sequence MUST be BFS-compatible (level order) for the
// augmented tree with an implicit super-root (id=0). This implies:
//   - all roots (parent_1based==0) must appear first in the node sequence, before
//     any non-root nodes.
// Otherwise LOUDS parent queries will not match the original parent array.
class ParentLOUDS {
 public:
  class SequentialParentDecoder {
   public:
    explicit SequentialParentDecoder(const ParentLOUDS& louds);

    uint32_t NextParent1Based();
    void Skip(size_t n);
    void InitializeAfterValidatedRootPrefix(size_t root_count);

   private:
    const std::vector<uint64_t>* words_ = nullptr;
    size_t next_word_ = 0;
    uint64_t one_positions_ = 0;
    uint64_t word_base_ = 0;
    uint32_t emitted_children_ = 0;
  };

  ParentLOUDS() = default;

  size_t n_nodes() const { return n_; }                 // excluding super-root
  size_t bit_length() const { return static_cast<size_t>(bv_.n_bits()); }
  uint32_t select_stride() const { return bv_.select_stride(); }

  // Deterministic serialized size for planning positioned writes.
  // - Serialization is header(v1) + LOUDS words (rank/select indices are NOT serialized).
  static uint64_t SerializedBytesForNodes(uint32_t n_nodes);

  // Build LOUDS from a BFS-compatible parent array.
  void BuildFromParent1Based(const uint32_t* parent_1based,
                             size_t n,
                             uint32_t select_stride = 128,
                             uint32_t rank_words_per_super_log2 = 4,
                             bool build_indices = false);
  inline void BuildFromParent1Based(const std::vector<uint32_t>& parent_1based, uint32_t select_stride = 128) {
    BuildFromParent1Based(parent_1based.data(), parent_1based.size(), select_stride, 4, false);
  }

  // O(1) parent query. Returns in the same 1-based convention as the input.
  // - Returns 0 for root.
  // - Otherwise returns (parent_pos + 1).
  uint32_t Parent1BasedOf(size_t pos0) const;

  // Decode parent pointers into `out` for old_id positions [0, n_out).
  void DecodeParent1Based(std::vector<uint32_t>* out, size_t n_out) const;

  SequentialParentDecoder MakeSequentialParentDecoder() const { return SequentialParentDecoder(*this); }

  // Serialize/deserialize.
  // Format: Header(v1) + LOUDS words. Rank/select indices are rebuilt on load if requested.
  std::vector<uint8_t> Serialize() const;
  void Deserialize(const uint8_t* data,
                   size_t len,
                   uint32_t rank_words_per_super_log2 = 4,
                   bool build_indices = false,
                   uint32_t select_stride_override = 0);
  inline void Deserialize(const std::vector<uint8_t>& blob,
                          uint32_t rank_words_per_super_log2 = 4,
                          bool build_indices = false,
                          uint32_t select_stride_override = 0) {
    Deserialize(blob.data(), blob.size(), rank_words_per_super_log2, build_indices, select_stride_override);
  }

  const RankSelectBitVector& bitvector() const { return bv_; }

 private:
  size_t n_ = 0;  // number of nodes excluding super-root
  RankSelectBitVector bv_;

  // Build LOUDS bitvector from degrees in node order [super-root, node1..nodeN].
  static std::vector<uint64_t> BuildLoudsWordsFromDegrees(const std::vector<uint32_t>& deg, size_t total_bits);
};

}  // namespace succinct
}  // namespace stlq

#if defined(_MSC_VER)
#define STLQ_PARENT_LOUDS_ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define STLQ_PARENT_LOUDS_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define STLQ_PARENT_LOUDS_ALWAYS_INLINE inline
#endif

namespace stlq::succinct {

STLQ_PARENT_LOUDS_ALWAYS_INLINE
uint32_t ParentLOUDS::SequentialParentDecoder::NextParent1Based() {
  // Every one bit emits one child. If child j is zero-based and its one bit
  // is at zero-based position p, then its one-based parent is p - j, the
  // number of zero delimiters preceding that bit. The caller consumes only
  // known node ranges from an already validated canonical stream.
  while (one_positions_ == 0) {
    word_base_ = static_cast<uint64_t>(next_word_) * 64u;
    one_positions_ = (*words_)[next_word_++];
  }
  const uint32_t bit = detail::Ctz64NonZero(one_positions_);
  one_positions_ &= one_positions_ - 1u;
  return static_cast<uint32_t>(
      word_base_ + bit - emitted_children_++);
}

}  // namespace stlq::succinct

#undef STLQ_PARENT_LOUDS_ALWAYS_INLINE
