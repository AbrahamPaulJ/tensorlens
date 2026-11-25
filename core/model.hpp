// TensorLens core: a format-neutral description of an ML model found in a binary.
// No IDA dependency; used by the IDA plugin, the loader and the command-line scanner.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace tl {

enum class Format { TFLite, ONNX, NCNN, ExecuTorch };

const char *format_name(Format f);

constexpr uint64_t kNoData = UINT64_MAX;

struct Tensor {
  std::string name;
  std::vector<int64_t> shape;     // -1 = dynamic dimension
  std::string dtype;              // "FLOAT32", "INT8", ...
  bool constant = false;          // has data in the model (weights, biases, constants)
  uint64_t data_offset = kNoData; // relative to the start of the model
  uint64_t data_size = 0;
  std::string quant;              // e.g. "scale=0.0078 zp=128" or "per-channel(32)"
};

struct Node {
  std::string op;                 // "CONV_2D", "Conv", "Convolution", ...
  std::string name;               // node name when the format stores one
  std::string domain;             // ONNX domain, "custom" for TFLite custom ops
  int builtin_code = -1;          // TFLite builtin operator code, -1 otherwise
  int op_version = 0;             // TFLite operator version / ONNX opset of its domain
  bool custom = false;            // TFLite custom op or non-standard ONNX domain
  std::vector<int> inputs;        // tensor indices, -1 = omitted optional input
  std::vector<int> outputs;
};

struct Graph {
  std::string name;
  std::vector<Tensor> tensors;
  std::vector<Node> nodes;
  std::vector<int> inputs;
  std::vector<int> outputs;
};

struct Model {
  Format format = Format::TFLite;
  uint64_t size = 0;              // bytes occupied by the model in the scanned buffer
  std::string producer;           // "TOCO Converted.", "pytorch 2.3", ...
  std::string version;            // schema / IR / opset summary
  std::vector<std::pair<std::string, std::string>> metadata;
  std::vector<Graph> graphs;
  std::vector<std::string> warnings;
  bool detect_only = false;       // format recognised, graph not decoded

  uint64_t weight_bytes() const;
  size_t node_count() const;
  // Distinct operator names with use counts, most used first.
  std::vector<std::pair<std::string, int>> op_histogram() const;
  // Custom / non-standard operators (worth a reviewer's attention).
  std::vector<std::string> custom_ops() const;
  std::string summary() const;    // one line: "TFLite v3, 64 ops (CONV_2D x27 ...), 3.4 MB weights"
};

// Parse results carry the model or a reason it was rejected.
struct ParseResult {
  bool ok = false;
  Model model;
  std::string error;
};

// Parsers. Each takes a pointer to the first byte of the candidate model and the number of
// bytes available after it (the model may be shorter; its real size is reported in model.size).
// All reads are bounds-checked: hostile or truncated input yields ok=false, never a crash.
ParseResult parse_tflite(const uint8_t *p, size_t avail);
ParseResult parse_onnx(const uint8_t *p, size_t avail);
ParseResult parse_ncnn_param(const uint8_t *p, size_t avail);
ParseResult parse_executorch(const uint8_t *p, size_t avail);

// Human-readable helpers.
std::string format_bytes(uint64_t n);
std::string shape_str(const std::vector<int64_t> &shape);
const char *tflite_op_name(int code);       // nullptr if unknown
const char *tflite_type_name(int code);     // nullptr if unknown
// Operators implemented by a TFLite kernel source file ("conv.cc" -> "CONV_2D"), or nullptr.
const char *tflite_kernel_file_ops(const std::string &basename);

}  // namespace tl
