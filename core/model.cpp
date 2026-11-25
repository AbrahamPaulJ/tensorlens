#include "model.hpp"

#include <algorithm>
#include <cstdio>
#include <map>

#include "generated_tables.inc"

namespace tl {

const char *format_name(Format f) {
  switch (f) {
    case Format::TFLite: return "TFLite";
    case Format::ONNX: return "ONNX";
    case Format::NCNN: return "ncnn";
    case Format::ExecuTorch: return "ExecuTorch";
  }
  return "?";
}

uint64_t Model::weight_bytes() const {
  uint64_t n = 0;
  for (const Graph &g : graphs)
    for (const Tensor &t : g.tensors)
      if (t.constant) n += t.data_size;
  return n;
}

size_t Model::node_count() const {
  size_t n = 0;
  for (const Graph &g : graphs) n += g.nodes.size();
  return n;
}

std::vector<std::pair<std::string, int>> Model::op_histogram() const {
  std::map<std::string, int> h;
  for (const Graph &g : graphs)
    for (const Node &n : g.nodes) h[n.op]++;
  std::vector<std::pair<std::string, int>> v(h.begin(), h.end());
  std::stable_sort(v.begin(), v.end(), [](auto &a, auto &b) { return a.second > b.second; });
  return v;
}

std::vector<std::string> Model::custom_ops() const {
  std::vector<std::string> out;
  for (const Graph &g : graphs)
    for (const Node &n : g.nodes)
      if (n.custom) {
        std::string k = n.domain.empty() || n.domain == "custom" ? n.op : n.domain + "::" + n.op;
        if (std::find(out.begin(), out.end(), k) == out.end()) out.push_back(k);
      }
  return out;
}

std::string Model::summary() const {
  std::string s = format_name(format);
  if (detect_only) return s + " (" + version + ", graph not decoded)";
  s += " " + version + ", " + std::to_string(node_count()) + " ops";
  auto h = op_histogram();
  if (!h.empty()) {
    s += " (";
    for (size_t i = 0; i < h.size() && i < 3; i++) {
      if (i) s += ", ";
      s += h[i].first + " x" + std::to_string(h[i].second);
    }
    if (h.size() > 3) s += ", ...";
    s += ")";
  }
  uint64_t w = weight_bytes();
  if (w) s += ", " + format_bytes(w) + " weights";
  return s;
}

std::string format_bytes(uint64_t n) {
  char buf[32];
  if (n < 1024) std::snprintf(buf, sizeof buf, "%llu B", (unsigned long long)n);
  else if (n < (1u << 20)) std::snprintf(buf, sizeof buf, "%.1f KB", n / 1024.0);
  else if (n < (1ull << 30)) std::snprintf(buf, sizeof buf, "%.1f MB", n / 1048576.0);
  else std::snprintf(buf, sizeof buf, "%.2f GB", n / 1073741824.0);
  return buf;
}

std::string shape_str(const std::vector<int64_t> &shape) {
  std::string s = "[";
  for (size_t i = 0; i < shape.size(); i++) {
    if (i) s += ",";
    s += shape[i] < 0 ? "?" : std::to_string(shape[i]);
  }
  return s + "]";
}

const char *tflite_op_name(int code) {
  for (const auto &o : gen::kBuiltinOps)
    if (o.code == code) return o.name;
  return nullptr;
}

const char *tflite_type_name(int code) {
  for (const auto &o : gen::kTensorTypes)
    if (o.code == code) return o.name;
  return nullptr;
}

const char *tflite_kernel_file_ops(const std::string &basename) {
  for (const auto &k : gen::kKernelFiles)
    if (basename == k.file) return k.ops;
  return nullptr;
}

}  // namespace tl
