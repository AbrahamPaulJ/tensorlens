// Find ML models embedded anywhere in a byte buffer.
#pragma once

#include <functional>

#include "model.hpp"

namespace tl {

struct Hit {
  uint64_t offset = 0;   // offset of the model's first byte in the scanned buffer
  Model model;
};

struct ScanOptions {
  bool tflite = true;
  bool onnx = true;
  bool ncnn = true;
  bool executorch = true;
  // Called every ~16 MB with the current offset; return false to cancel.
  std::function<bool(uint64_t done, uint64_t total)> progress;
};

std::vector<Hit> scan(const uint8_t *p, size_t n, const ScanOptions &opt = {});

// Parse a buffer that is known to be exactly one model file (any supported format).
ParseResult parse_any(const uint8_t *p, size_t n);

}  // namespace tl
