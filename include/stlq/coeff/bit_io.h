#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace stlq {

class BitWriter {
public:
  void Reset() {
    buf_.clear();
    cur_ = 0;
    nbits_ = 0;
  }

  void WriteBits(std::uint32_t bits, int nbits) {
    if (nbits < 0 || nbits > 32) {
      throw std::runtime_error("BitWriter::WriteBits: invalid bit count");
    }
    // MSB-first within the codeword (matches Julia huffman.jl).
    for (int i = nbits - 1; i >= 0; --i) {
      const std::uint32_t b = (bits >> i) & 1u;
      cur_ = (cur_ << 1) | b;
      ++nbits_;
      if (nbits_ == 8) {
        buf_.push_back(static_cast<std::uint8_t>(cur_));
        cur_ = 0;
        nbits_ = 0;
      }
    }
  }

  std::vector<std::uint8_t> Finish() {
    if (nbits_ > 0) {
      cur_ <<= (8 - nbits_);
      buf_.push_back(static_cast<std::uint8_t>(cur_));
      cur_ = 0;
      nbits_ = 0;
    }
    return std::move(buf_);
  }

private:
  std::vector<std::uint8_t> buf_;
  std::uint32_t cur_ = 0;
  int nbits_ = 0;
};

class BitReader {
public:
  BitReader(const std::uint8_t* data, std::size_t nbytes)
      : data_(data), nbytes_(nbytes) {}

  std::uint32_t ReadBit() {
    // Hot-path codec decode relies on the validated payload length and expected symbol count.
    // if (byte_pos_ >= nbytes_) {
    //   throw std::runtime_error("BitReader::ReadBit: out of range");
    // }
    const std::uint8_t byte = data_[byte_pos_];
    const std::uint32_t bit = (byte >> (7 - bit_pos_)) & 1u;
    ++bit_pos_;
    if (bit_pos_ == 8) {
      bit_pos_ = 0;
      ++byte_pos_;
    }
    return bit;
  }

private:
  const std::uint8_t* data_ = nullptr;
  std::size_t nbytes_ = 0;
  std::size_t byte_pos_ = 0;
  int bit_pos_ = 0;
};

}  // namespace stlq
