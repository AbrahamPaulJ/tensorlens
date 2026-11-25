#include "scan.hpp"

#include <algorithm>
#include <cstring>

#include "inflate.hpp"

namespace tl {
namespace {

// Exception-free pre-check, so the full parser (which reports errors by throwing) only runs on
// candidates that already look like a ModelProto. Code sections are full of 0x08 bytes.
bool plausible_onnx_start(const uint8_t *p, size_t avail) {
  // ModelProto starts with ir_version (field 1, varint 1..12).
  if (avail < 16 || p[0] != 0x08 || p[1] < 1 || p[1] > 12) return false;
  size_t i = 2;
  auto varint = [&](uint64_t &v) {
    v = 0;
    for (int shift = 0; shift < 64 && i < avail; shift += 7) {
      uint8_t c = p[i++];
      v |= uint64_t(c & 0x7f) << shift;
      if (!(c & 0x80)) return true;
    }
    return false;
  };
  // The following fields must be known ModelProto fields with the right wire type; string
  // fields must be printable. Check up to three of them, stopping at the graph.
  for (int k = 0; k < 3 && i < avail; k++) {
    uint8_t key = p[i++];
    uint64_t len = 0;
    switch (key) {
      case 0x12: case 0x1a: case 0x22: case 0x32:  // producer_name/version, domain, doc_string
        if (!varint(len) || len > 4096 || len > avail - i) return false;
        for (uint64_t j = 0; j < len; j++)
          if (p[i + j] < 0x20 && p[i + j] != '\n' && p[i + j] != '\t' && p[i + j] != '\r') return false;
        i += len;
        break;
      case 0x28:  // model_version
        if (!varint(len)) return false;
        break;
      case 0x3a:  // graph: a message that must fit
        if (!varint(len) || len < 8 || len > avail - i) return false;
        return p[i] == 0x0a || p[i] == 0x12 || p[i] == 0x2a || p[i] == 0x5a;  // node, name, initializer, input
      case 0x42:  // opset_import
        if (!varint(len) || len < 2 || len > 256 || len > avail - i) return false;
        i += len;
        break;
      default: return false;
    }
  }
  return true;
}

bool plausible_flatbuffer_root(const uint8_t *p, size_t avail) {
  if (avail < 16) return false;
  uint32_t root;
  std::memcpy(&root, p, 4);
  if (root < 8 || root % 4 || root + 4 > avail) return false;
  int32_t soff;
  std::memcpy(&soff, p + root, 4);
  int64_t vt = int64_t(root) - soff;
  if (vt < 0 || uint64_t(vt) + 4 > avail) return false;
  uint16_t vts;
  std::memcpy(&vts, p + vt, 2);
  return vts >= 4 && vts < 512 && !(vts & 1);
}

}  // namespace

std::vector<Hit> scan(const uint8_t *p, size_t n, const ScanOptions &opt) {
  std::vector<Hit> hits;
  uint64_t next_progress = 16u << 20;
  size_t i = 0;
  while (i + 8 <= n) {
    if (opt.progress && i >= next_progress) {
      if (!opt.progress(i, n)) break;
      next_progress += 16u << 20;
    }
    const uint8_t c = p[i];
    ParseResult pr;
    uint64_t start = i;
    bool tried = false;
    // TFLite: "TFL3" file identifier lives at offset 4 of the model.
    if (opt.tflite && c == 'T' && i >= 4 && p[i + 1] == 'F' && p[i + 2] == 'L' && p[i + 3] == '3') {
      start = i - 4;
      if (plausible_flatbuffer_root(p + start, n - start)) {
        pr = parse_tflite(p + start, n - start);
        tried = true;
      }
    } else if (opt.executorch && c == 'E' && i >= 4 && p[i + 1] == 'T' && p[i + 2] >= '0' && p[i + 2] <= '9' &&
               p[i + 3] >= '0' && p[i + 3] <= '9') {
      start = i - 4;
      if (plausible_flatbuffer_root(p + start, n - start)) {
        pr = parse_executorch(p + start, n - start);
        tried = true;
      }
    } else if (opt.ncnn && c == '7' && std::memcmp(p + i, "7767517", 7) == 0) {
      pr = parse_ncnn_param(p + i, n - i);
      tried = true;
    } else if (opt.onnx && c == 0x08 && plausible_onnx_start(p + i, n - i)) {
      pr = parse_onnx(p + i, n - i);
      tried = true;
    } else if (opt.gzip && c == 0x1f && p[i + 1] == 0x8b && p[i + 2] == 8) {
      auto data = std::make_shared<std::vector<uint8_t>>();
      size_t consumed = 0;
      if (gunzip_at(p + i, n - i, *data, consumed)) {
        ScanOptions inner = opt;
        inner.gzip = false;  // one level is enough for real binaries and bounds the work
        inner.progress = nullptr;
        auto inner_hits = scan(data->data(), data->size(), inner);
        std::shared_ptr<const std::vector<uint8_t>> shared = data;
        for (Hit &ih : inner_hits) {
          Hit h = std::move(ih);
          h.compressed = true;
          h.inner_offset = h.offset;
          h.offset = i;
          std::string member = tar_member_name(shared->data(), shared->size(), h.inner_offset);
          h.container = "gzip stream (" + format_bytes(consumed) + " compressed, " + format_bytes(shared->size()) +
                        " inflated)" + (member.empty() ? "" : ", tar member " + member);
          h.payload = shared;
          hits.push_back(std::move(h));
        }
        i += std::max<size_t>(consumed, 1);
        continue;
      }
    }
    if (tried && pr.ok) {
      // A hit must not start inside the previous hit (e.g. TFL3 inside an ONNX initializer).
      if (!hits.empty() && !hits.back().compressed && start < hits.back().offset + hits.back().model.size) {
        i++;
        continue;
      }
      Hit h;
      h.offset = start;
      h.model = std::move(pr.model);
      uint64_t skip = std::max<uint64_t>(h.model.size, 8);
      hits.push_back(std::move(h));
      i = static_cast<size_t>(std::min<uint64_t>(start + skip, n));
      continue;
    }
    i++;
  }
  return hits;
}

ParseResult parse_any(const uint8_t *p, size_t n) {
  if (n >= 8 && std::memcmp(p + 4, "TFL3", 4) == 0) return parse_tflite(p, n);
  if (n >= 8 && std::memcmp(p, "7767517", 7) == 0) return parse_ncnn_param(p, n);
  if (n >= 8 && p[4] == 'E' && p[5] == 'T') return parse_executorch(p, n);
  ParseResult r = parse_onnx(p, n);
  if (!r.ok) r.error = "not a TFLite, ONNX, ncnn or ExecuTorch model (" + r.error + ")";
  return r;
}

}  // namespace tl
