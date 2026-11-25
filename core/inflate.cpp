// gzip stream decoding (models are often shipped compressed inside binaries, e.g. a .tar.gz of
// ONNX files embedded by a Rust include_bytes!).
#include <cstring>

#include "inflate.hpp"
#include "miniz.h"

namespace tl {

bool gunzip_at(const uint8_t *p, size_t avail, std::vector<uint8_t> &out, size_t &consumed, size_t cap) {
  out.clear();
  consumed = 0;
  if (avail < 18 || p[0] != 0x1f || p[1] != 0x8b || p[2] != 8) return false;
  const uint8_t flg = p[3];
  if (flg & 0xe0) return false;  // reserved bits must be zero
  size_t pos = 10;
  if (flg & 4) {  // FEXTRA
    if (pos + 2 > avail) return false;
    size_t xlen = p[pos] | (p[pos + 1] << 8);
    pos += 2 + xlen;
  }
  auto skip_cstr = [&]() {
    while (pos < avail && p[pos]) pos++;
    pos++;
  };
  if (flg & 8) skip_cstr();   // FNAME
  if (flg & 16) skip_cstr();  // FCOMMENT
  if (flg & 2) pos += 2;      // FHCRC
  if (pos >= avail) return false;

  tinfl_decompressor d;
  tinfl_init(&d);
  size_t in_pos = pos;
  out.resize(64 * 1024);
  size_t out_pos = 0;
  for (;;) {
    size_t in_size = avail - in_pos;
    size_t out_size = out.size() - out_pos;
    tinfl_status st = tinfl_decompress(&d, p + in_pos, &in_size, out.data(), out.data() + out_pos, &out_size,
                                       TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF);
    in_pos += in_size;
    out_pos += out_size;
    if (st == TINFL_STATUS_DONE) break;
    if (st != TINFL_STATUS_HAS_MORE_OUTPUT) {
      out.clear();
      return false;
    }
    if (out.size() >= cap) {
      out.clear();
      return false;
    }
    out.resize(std::min(cap, out.size() * 2));
  }
  out.resize(out_pos);
  consumed = std::min(avail, in_pos + 8);  // CRC32 + ISIZE trailer
  return out_pos > 0;
}

std::string tar_member_name(const uint8_t *data, size_t size, uint64_t offset) {
  // A tar member's data starts on a 512-byte boundary, right after its 512-byte header.
  if (offset < 512 || offset % 512 || offset > size) return {};
  const uint8_t *h = data + offset - 512;
  if (std::memcmp(h + 257, "ustar", 5) != 0) return {};
  std::string prefix(reinterpret_cast<const char *>(h + 345), strnlen(reinterpret_cast<const char *>(h + 345), 155));
  std::string name(reinterpret_cast<const char *>(h), strnlen(reinterpret_cast<const char *>(h), 100));
  return prefix.empty() ? name : prefix + "/" + name;
}

}  // namespace tl
