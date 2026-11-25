// ncnn: text .param graphs ("7767517" magic) and ExecuTorch .pte detection.
#include <cstdio>
#include <sstream>
#include <unordered_map>

#include "model.hpp"
#include "reader.hpp"

namespace tl {

ParseResult parse_ncnn_param(const uint8_t *p, size_t avail) {
  ParseResult r;
  try {
    static const char kMagic[] = "7767517";
    if (avail < 16 || std::memcmp(p, kMagic, 7) != 0 || (p[7] != '\n' && p[7] != '\r'))
      throw ParseError{"no ncnn magic"};
    // The text ends at the first NUL or non-text byte.
    size_t end = 0;
    while (end < avail && end < (64u << 20)) {
      unsigned char c = p[end];
      if (c == 0 || (c < 0x20 && c != '\n' && c != '\r' && c != '\t') || c >= 0x7f) break;
      end++;
    }
    std::istringstream in(std::string(reinterpret_cast<const char *>(p), end));
    std::string magic;
    int layers = 0, blobs = 0;
    in >> magic >> layers >> blobs;
    if (!in || layers <= 0 || layers > 100000 || blobs <= 0 || blobs > 1000000)
      throw ParseError{"bad ncnn header"};
    Model &m = r.model;
    m.format = Format::NCNN;
    m.version = "param 7767517";
    Graph g;
    std::unordered_map<std::string, int> idx;
    auto blob = [&](const std::string &name) {
      auto it = idx.find(name);
      if (it != idx.end()) return it->second;
      Tensor t;
      t.name = name;
      g.tensors.push_back(t);
      int id = static_cast<int>(g.tensors.size() - 1);
      idx.emplace(name, id);
      return id;
    };
    std::string line;
    std::getline(in, line);  // rest of header line
    int parsed = 0;
    while (parsed < layers && std::getline(in, line)) {
      if (line.empty() || line == "\r") continue;
      std::istringstream ls(line);
      Node n;
      int ni = 0, no = 0;
      ls >> n.op >> n.name >> ni >> no;
      if (!ls || ni < 0 || no < 0 || ni > 4096 || no > 4096) throw ParseError{"bad ncnn layer line"};
      for (int i = 0; i < ni; i++) { std::string s; ls >> s; n.inputs.push_back(blob(s)); }
      for (int i = 0; i < no; i++) { std::string s; ls >> s; n.outputs.push_back(blob(s)); }
      if (!ls) throw ParseError{"truncated ncnn layer line"};
      if (n.op == "Input") {
        for (int o : n.outputs) g.inputs.push_back(o);
      }
      g.nodes.push_back(std::move(n));
      parsed++;
    }
    if (parsed != layers) throw ParseError{"fewer layers than declared"};
    // Outputs: blobs produced but never consumed.
    std::vector<int> consumed(g.tensors.size(), 0);
    for (const Node &n : g.nodes) for (int i : n.inputs) if (i >= 0) consumed[i]++;
    for (const Node &n : g.nodes) for (int o : n.outputs) if (o >= 0 && !consumed[o]) g.outputs.push_back(o);
    m.size = static_cast<uint64_t>(in.tellg() < 0 ? end : static_cast<size_t>(in.tellg()));
    m.warnings.push_back("weights live in a separate .bin file");
    m.graphs.push_back(std::move(g));
    r.ok = true;
  } catch (const ParseError &e) {
    r.error = e.what;
  }
  return r;
}

// ExecuTorch programs are flatbuffers with file identifier "ET12" (or later "ETxx").
// We recognise and size-bound them but do not decode the graph yet.
ParseResult parse_executorch(const uint8_t *p, size_t avail) {
  ParseResult r;
  if (avail < 32 || p[4] != 'E' || p[5] != 'T' || p[6] < '0' || p[6] > '9' || p[7] < '0' || p[7] > '9') {
    r.error = "no ET identifier";
    return r;
  }
  uint32_t root;
  std::memcpy(&root, p, 4);
  if (root < 8 || root > avail) {
    r.error = "bad root";
    return r;
  }
  r.ok = true;
  r.model.format = Format::ExecuTorch;
  r.model.detect_only = true;
  r.model.version = std::string("program ") + char(p[4]) + char(p[5]) + char(p[6]) + char(p[7]);
  r.model.size = 0;  // unknown; reported as such
  r.model.warnings.push_back("ExecuTorch graph decoding not implemented yet");
  return r;
}

}  // namespace tl
