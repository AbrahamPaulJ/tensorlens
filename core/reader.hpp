// Bounds-checked little-endian readers shared by the format parsers.
#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>

namespace tl {

// Thrown by the readers on any out-of-bounds or malformed access; parsers catch it at the
// top level and turn it into ParseResult{ok=false}.
struct ParseError {
  const char *what;
};

class Bytes {
 public:
  Bytes(const uint8_t *p, size_t n) : p_(p), n_(n) {}
  size_t size() const { return n_; }
  const uint8_t *data() const { return p_; }

  void need(uint64_t off, uint64_t len) const {
    if (off > n_ || len > n_ - off) throw ParseError{"read out of bounds"};
  }
  template <class T>
  T get(uint64_t off) const {
    need(off, sizeof(T));
    T v;
    std::memcpy(&v, p_ + off, sizeof(T));
    return v;  // all supported hosts are little-endian
  }
  std::string_view view(uint64_t off, uint64_t len) const {
    need(off, len);
    return {reinterpret_cast<const char *>(p_ + off), static_cast<size_t>(len)};
  }

 private:
  const uint8_t *p_;
  size_t n_;
};

inline bool printable_ascii(std::string_view s) {
  for (unsigned char c : s)
    if (c < 0x20 && c != '\n' && c != '\r' && c != '\t') return false;
  return true;
}

}  // namespace tl
