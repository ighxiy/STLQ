#pragma once

#include <algorithm>
#include <cstdint>
#include <queue>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "stlq/coeff/bit_io.h"

namespace stlq {

struct HuffmanCanonicalModel {
  // Fixed alphabet size S for symbols in [0..S).
  std::uint16_t S = 0;

  // Per-symbol code lengths in bits. 0 means symbol is unused.
  std::vector<std::uint8_t> len;

  // Derived for encoding (valid only after BuildDerivedTables()).
  std::vector<std::uint32_t> code;

  // Derived for canonical decoding.
  std::uint8_t max_len = 0;
  std::vector<std::uint32_t> bl_count;    // [1..max_len]
  std::vector<std::uint32_t> first_code;  // [1..max_len]
  std::vector<std::uint32_t> first_sym;   // [1..max_len] position into syms_sorted
  std::vector<std::uint16_t> syms_sorted; // symbols sorted by (len, sym)

  void Clear() {
    S = 0;
    len.clear();
    code.clear();
    max_len = 0;
    bl_count.clear();
    first_code.clear();
    first_sym.clear();
    syms_sorted.clear();
  }

  // Build encoding + decoding tables from `len`.
  // Throws if `len` is invalid.
  void BuildDerivedTables();

  inline void EncodeSymbol(BitWriter& bw, std::uint16_t sym) const {
    const std::uint8_t L = len[sym];
    if (L == 0) {
      throw std::runtime_error("Huffman EncodeSymbol: symbol has zero length");
    }
    bw.WriteBits(code[sym], L);
  }

  std::uint16_t DecodeSymbol(BitReader& br) const;
};

// Compute Huffman code lengths for a fixed alphabet [0..S).
//
// - `freq` is length S; freq[s]==0 means symbol is unused.
// - If only one symbol is used, it gets length 1.
// - Unused symbols keep length 0.
std::vector<std::uint8_t> HuffmanCodeLengths(const std::vector<std::uint32_t>& freq);

// Build canonical model from frequencies.
HuffmanCanonicalModel BuildHuffmanCanonicalModel(const std::vector<std::uint32_t>& freq);

}  // namespace stlq
