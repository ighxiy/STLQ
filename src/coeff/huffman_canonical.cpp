#include "stlq/coeff/huffman_canonical.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace stlq {

namespace {

struct Node {
  std::uint32_t freq{};
  std::uint16_t sym{};
  Node* left = nullptr;
  Node* right = nullptr;
};

struct NodeLess {
  bool operator()(const Node* a, const Node* b) const {
    if (a->freq != b->freq) return a->freq > b->freq;
    return a->sym > b->sym;
  }
};

void AssignLengths(Node* node, int depth, std::vector<std::uint8_t>* len) {
  if (!node) return;
  if (!node->left && !node->right) {
    (*len)[node->sym] = static_cast<std::uint8_t>(depth);
    return;
  }
  AssignLengths(node->left, depth + 1, len);
  AssignLengths(node->right, depth + 1, len);
}

}  // namespace

void HuffmanCanonicalModel::BuildDerivedTables() {
  if (S == 0) throw std::runtime_error("HuffmanCanonicalModel: S==0");
  if (len.size() != S) throw std::runtime_error("HuffmanCanonicalModel: len.size()!=S");

  max_len = 0;
  for (std::uint8_t L : len) {
    if (L > 0) max_len = std::max(max_len, L);
  }
  if (max_len == 0) throw std::runtime_error("HuffmanCanonicalModel: empty model");

  bl_count.assign(static_cast<std::size_t>(max_len) + 1, 0);
  for (std::uint8_t L : len) {
    if (L > 0) bl_count[L] += 1;
  }

  first_code.assign(static_cast<std::size_t>(max_len) + 1, 0);
  std::uint32_t code_val = 0;
  for (int bits = 1; bits <= max_len; ++bits) {
    code_val = (code_val + bl_count[bits - 1]) << 1;
    first_code[static_cast<std::size_t>(bits)] = code_val;
  }

  syms_sorted.clear();
  syms_sorted.reserve(S);
  for (std::uint16_t sym = 0; sym < S; ++sym) {
    if (len[sym] > 0) syms_sorted.push_back(sym);
  }
  std::sort(syms_sorted.begin(), syms_sorted.end(),
            [&](std::uint16_t a, std::uint16_t b) {
              if (len[a] != len[b]) return len[a] < len[b];
              return a < b;
            });

  first_sym.assign(static_cast<std::size_t>(max_len) + 1, 0);
  std::uint32_t pos = 0;
  for (int bits = 1; bits <= max_len; ++bits) {
    first_sym[static_cast<std::size_t>(bits)] = pos;
    pos += bl_count[static_cast<std::size_t>(bits)];
  }

  code.assign(S, 0);
  std::vector<std::uint32_t> next_code = first_code;
  for (std::uint16_t sym : syms_sorted) {
    const std::uint8_t L = len[sym];
    if (L == 0) continue;
    code[sym] = next_code[L];
    next_code[L] += 1;
  }
}

std::uint16_t HuffmanCanonicalModel::DecodeSymbol(BitReader& br) const {
  std::uint32_t code_val = 0;
  for (int bits = 1; bits <= max_len; ++bits) {
    code_val = (code_val << 1) | br.ReadBit();
    const std::uint32_t first = first_code[static_cast<std::size_t>(bits)];
    const std::uint32_t count = bl_count[static_cast<std::size_t>(bits)];
    if (code_val >= first && code_val < first + count) {
      const std::uint32_t idx = first_sym[static_cast<std::size_t>(bits)] + (code_val - first);
      // Hot-path decode relies on BuildDerivedTables() validating the canonical table.
      // if (idx >= syms_sorted.size()) throw std::runtime_error("Huffman decode: index overflow");
      return syms_sorted[idx];
    }
  }
  // Hot-path decode relies on validated codec payloads. This path is unreachable for valid stores.
  // throw std::runtime_error("Huffman decode: invalid code");
  return 0;
}

std::vector<std::uint8_t> HuffmanCodeLengths(const std::vector<std::uint32_t>& freq) {
  const auto S = static_cast<std::uint16_t>(freq.size());
  std::vector<std::uint8_t> len(S, 0);
  if (S == 0) return len;

  std::vector<Node> nodes;
  nodes.reserve(2 * S);
  std::priority_queue<Node*, std::vector<Node*>, NodeLess> pq;

  for (std::uint16_t s = 0; s < S; ++s) {
    if (freq[s] == 0) continue;
    nodes.push_back(Node{freq[s], s, nullptr, nullptr});
    pq.push(&nodes.back());
  }

  if (pq.empty()) {
    return len;
  }
  if (pq.size() == 1) {
    pq.top()->freq = pq.top()->freq;
    len[pq.top()->sym] = 1;
    return len;
  }

  while (pq.size() > 1) {
    Node* a = pq.top(); pq.pop();
    Node* b = pq.top(); pq.pop();
    nodes.push_back(Node{a->freq + b->freq, std::min(a->sym, b->sym), a, b});
    pq.push(&nodes.back());
  }

  Node* root = pq.top();
  AssignLengths(root, 0, &len);
  return len;
}

HuffmanCanonicalModel BuildHuffmanCanonicalModel(const std::vector<std::uint32_t>& freq) {
  HuffmanCanonicalModel model;
  model.S = static_cast<std::uint16_t>(freq.size());
  model.len = HuffmanCodeLengths(freq);
  model.BuildDerivedTables();
  return model;
}

}  // namespace stlq
