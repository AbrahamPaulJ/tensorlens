// Find ML models embedded anywhere in a byte buffer, including inside gzip streams.
#pragma once

#include <functional>
#include <memory>

#include "model.hpp"

namespace tl {

struct Hit {
  uint64_t offset = 0;   // offset of the model (or of the gzip stream holding it) in the scanned buffer
  Model model;
  // Set when the model sits inside a compressed stream at `offset`:
  bool compressed = false;
  uint64_t inner_offset = 0;   // offset of the model inside the decompressed data
  std::string container;       // e.g. "gzip (tar member: enc.onnx)"
  std::shared_ptr<const std::vector<uint8_t>> payload;  // decompressed bytes (shared by hits)

  // The model's bytes: points into the scanned buffer, or into the payload when compressed.
  const uint8_t *bytes(const uint8_t *scanned) const {
    return compressed ? payload->data() + inner_offset : scanned + offset;
  }
};

struct ScanOptions {
  bool tflite = true;
  bool onnx = true;
  bool ncnn = true;
  bool executorch = true;
  bool gzip = true;            // look inside gzip streams (one level deep)
  // Called every ~16 MB with the current offset; return false to cancel.
  std::function<bool(uint64_t done, uint64_t total)> progress;
};

std::vector<Hit> scan(const uint8_t *p, size_t n, const ScanOptions &opt = {});

// Parse a buffer that is known to be exactly one model file (any supported format).
ParseResult parse_any(const uint8_t *p, size_t n);

}  // namespace tl
