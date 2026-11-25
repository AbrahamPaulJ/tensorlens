// Core tests: real TFLite/ONNX fixtures from the TensorFlow and ONNX repositories (Apache-2.0),
// embedding/scanning, false-positive rate on noise, and mutation fuzzing for crash safety.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "../core/report.hpp"
#include "../core/scan.hpp"

static int g_fail = 0, g_pass = 0;
#define CHECK(cond)                                                       \
  do {                                                                    \
    if (cond) g_pass++;                                                   \
    else { g_fail++; std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); } \
  } while (0)

static std::string g_data;

static std::vector<uint8_t> load(const std::string &name) {
  std::ifstream f(g_data + "/" + name, std::ios::binary);
  std::vector<uint8_t> v((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (v.empty()) std::printf("missing fixture %s\n", name.c_str());
  return v;
}

static int count_op(const tl::Model &m, const std::string &op) {
  for (auto &e : m.op_histogram()) if (e.first == op) return e.second;
  return 0;
}

static void test_tflite_fixtures() {
  auto add = load("tflite_add.bin");
  auto r = tl::parse_tflite(add.data(), add.size());
  CHECK(r.ok);
  CHECK(count_op(r.model, "ADD") >= 1);
  CHECK(r.model.size <= add.size() && r.model.size + 16 >= add.size());
  CHECK(!r.model.graphs.empty() && !r.model.graphs[0].inputs.empty());

  auto q = load("tflite_add_quantized_int8.bin");
  r = tl::parse_tflite(q.data(), q.size());
  CHECK(r.ok);
  bool int8 = false, quant = false;
  for (auto &t : r.model.graphs[0].tensors) { int8 |= t.dtype == "INT8"; quant |= !t.quant.empty(); }
  CHECK(int8);
  CHECK(quant);

  auto c = load("tflite_custom_sinh.bin");
  r = tl::parse_tflite(c.data(), c.size());
  CHECK(r.ok);
  auto custom = r.model.custom_ops();
  CHECK(custom.size() == 1 && custom[0] == "Sinh");

  auto md = load("tflite_with_metadata.bin");
  r = tl::parse_tflite(md.data(), md.size());
  CHECK(r.ok && !r.model.metadata.empty());

  auto sg = load("tflite_2_subgraphs.bin");
  r = tl::parse_tflite(sg.data(), sg.size());
  CHECK(r.ok && r.model.graphs.size() == 2);

  auto ms = load("tflite_multi_signatures.bin");
  r = tl::parse_tflite(ms.data(), ms.size());
  int sigs = 0;
  for (auto &kv : r.model.metadata) sigs += kv.first == "signature";
  CHECK(r.ok && sigs >= 2);

  auto mr = load("tflite_test_min_runtime.bin");
  r = tl::parse_tflite(mr.data(), mr.size());
  bool minrt = false;
  for (auto &kv : r.model.metadata) minrt |= kv.first == "min_runtime_version" && !kv.second.empty();
  CHECK(r.ok && minrt);

  auto broken = load("tflite_test_model_broken.bin");  // must not crash; may parse or be rejected
  (void)tl::parse_tflite(broken.data(), broken.size());
  CHECK(true);
}

static void test_onnx_fixtures() {
  auto s = load("onnx_light_shufflenet.onnx");
  auto r = tl::parse_onnx(s.data(), s.size());
  CHECK(r.ok);
  CHECK(r.model.size == s.size());
  CHECK(count_op(r.model, "Conv") > 10);
  CHECK(r.model.producer.find("onnx-caffe2") == 0);
  CHECK(!r.model.graphs[0].inputs.empty() && !r.model.graphs[0].outputs.empty());
  CHECK(r.model.version.find("opset") != std::string::npos);

  for (const char *f : {"onnx_light_resnet50.onnx", "onnx_light_inception_v1.onnx", "onnx_expand_shape.onnx"}) {
    auto b = load(f);
    auto rr = tl::parse_onnx(b.data(), b.size());
    CHECK(rr.ok && rr.model.size == b.size() && rr.model.node_count() > 0);
  }
}

static void test_ncnn() {
  auto b = load("ncnn_tiny.param");
  auto r = tl::parse_ncnn_param(b.data(), b.size());
  CHECK(r.ok);
  CHECK(r.model.node_count() == 6);
  CHECK(count_op(r.model, "Convolution") == 1);
  CHECK(r.model.graphs[0].inputs.size() == 1 && r.model.graphs[0].outputs.size() == 1);
}

// Models embedded between noise must be found at the exact offset with the exact size.
static void test_embedding() {
  std::mt19937 rng(1234);
  std::vector<std::string> files = {"tflite_add.bin", "onnx_light_shufflenet.onnx", "tflite_custom_sinh.bin",
                                    "ncnn_tiny.param", "onnx_light_resnet50.onnx"};
  std::vector<uint8_t> blob;
  std::vector<std::pair<uint64_t, uint64_t>> where;
  for (auto &f : files) {
    for (int i = 0; i < 4096 + int(rng() % 4096); i++) blob.push_back(uint8_t(rng()));
    while (blob.size() % 16) blob.push_back(0);
    auto m = load(f);
    where.push_back({blob.size(), m.size()});
    blob.insert(blob.end(), m.begin(), m.end());
  }
  for (int i = 0; i < 4096; i++) blob.push_back(uint8_t(rng()));
  auto hits = tl::scan(blob.data(), blob.size());
  CHECK(hits.size() == files.size());
  for (size_t i = 0; i < hits.size() && i < where.size(); i++) {
    CHECK(hits[i].offset == where[i].first);
    // TFLite sizes round up to 16 within padding; ncnn stops at the first non-text byte.
    CHECK(hits[i].model.size >= where[i].second - 16 && hits[i].model.size <= where[i].second + 16);
  }
}

// 64 MB of random bytes and 64 MB of x86-like low-entropy data should produce no hits.
static void test_false_positives() {
  std::mt19937_64 rng(42);
  std::vector<uint8_t> noise(64u << 20);
  for (size_t i = 0; i + 8 <= noise.size(); i += 8) {
    uint64_t v = rng();
    std::memcpy(&noise[i], &v, 8);
  }
  auto hits = tl::scan(noise.data(), noise.size());
  CHECK(hits.empty());
  // Low-entropy data rich in 0x08 / small values, the worst case for the ONNX heuristic.
  std::vector<uint8_t> low(64u << 20);
  for (size_t i = 0; i < low.size(); i++) low[i] = uint8_t(rng() % 16 == 0 ? 0x08 : rng() % 0x48);
  hits = tl::scan(low.data(), low.size());
  if (!hits.empty()) std::printf("  low-entropy false positives: %zu (first: %s)\n", hits.size(), hits[0].model.summary().c_str());
  CHECK(hits.empty());
}

// Truncate and mutate fixtures; parsers must reject or accept, never crash or hang.
static void test_fuzz() {
  std::mt19937 rng(7);
  int parsed = 0;
  for (const char *f : {"tflite_add.bin", "tflite_multi_add.bin", "tflite_custom_sinh.bin", "onnx_light_shufflenet.onnx",
                        "ncnn_tiny.param", "tflite_with_metadata.bin"}) {
    auto base = load(f);
    for (size_t n = 0; n <= base.size(); n += (base.size() > 2000 ? 97 : 1)) {
      auto r = tl::parse_any(base.data(), n);
      parsed += r.ok;
    }
    const int iters = base.size() > 4096 ? 400 : 3000;  // big fixtures: fewer, still 400 mutants
    for (int it = 0; it < iters; it++) {
      auto m = base;
      int flips = 1 + rng() % 8;
      for (int k = 0; k < flips; k++) m[rng() % m.size()] = uint8_t(rng());
      auto r = tl::parse_any(m.data(), m.size());
      parsed += r.ok;
      auto hits = tl::scan(m.data(), m.size());
      (void)hits;
    }
  }
  std::printf("  fuzz: %d mutated inputs still parsed\n", parsed);
  CHECK(true);
}

// A .tar.gz of models embedded in a binary (as Rust include_bytes! or resource sections do).
static void test_gzip_tar() {
  std::mt19937 rng(99);
  std::vector<uint8_t> blob(3000);
  for (auto &b : blob) b = uint8_t(rng());
  auto gz = load("models.tar.gz");
  uint64_t at = blob.size();
  blob.insert(blob.end(), gz.begin(), gz.end());
  for (int i = 0; i < 3000; i++) blob.push_back(uint8_t(rng()));
  auto hits = tl::scan(blob.data(), blob.size());
  CHECK(hits.size() == 3);
  int named = 0;
  for (auto &h : hits) {
    CHECK(h.compressed && h.offset == at && h.payload);
    named += h.container.find("tar member models/") != std::string::npos;
    // The model bytes reachable through the hit must re-parse on their own.
    CHECK(tl::parse_any(h.bytes(blob.data()), h.model.size).ok);
  }
  CHECK(named == 3);
}

static void test_report() {
  auto b = load("tflite_custom_sinh.bin");
  tl::Report rep;
  rep.title = "t";
  rep.subject = "unit \"test\"";
  tl::ReportModel rm;
  rm.where = "x.so";
  rm.model = tl::parse_tflite(b.data(), b.size()).model;
  rep.models.push_back(rm);
  rep.sections.push_back({"Kernels", {"a", "b"}, {{"1", "<2>"}}});
  std::string j = tl::to_json(rep, true);
  CHECK(j.find("\"Sinh\"") != std::string::npos);
  CHECK(j.find("unit \\\"test\\\"") != std::string::npos);
  std::string h = tl::to_html(rep);
  CHECK(h.find("&lt;2&gt;") != std::string::npos);
  CHECK(h.find("Custom operators") != std::string::npos);
}

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  g_data = argc > 1 ? argv[1] : "tests/data";
  struct T { const char *name; void (*fn)(); } tests[] = {
      {"tflite_fixtures", test_tflite_fixtures}, {"onnx_fixtures", test_onnx_fixtures}, {"ncnn", test_ncnn},
      {"embedding", test_embedding},             {"false_positives", test_false_positives},
      {"fuzz", test_fuzz},                       {"gzip_tar", test_gzip_tar},                       {"report", test_report}};
  for (auto &t : tests) {
    auto t0 = std::chrono::steady_clock::now();
    t.fn();
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("[%s] %.0f ms\n", t.name, ms);
  }
  std::printf("%d passed, %d failed\n", g_pass, g_fail);
  return g_fail ? 1 : 0;
}
