// ONNX ModelProto reader: a minimal protobuf wire-format decoder plus the handful of ONNX
// messages needed to rebuild the graph. Field numbers follow onnx/onnx.proto (Apache-2.0).
#include <algorithm>
#include <cstdio>
#include <functional>
#include <map>
#include <unordered_map>

#include "model.hpp"
#include "reader.hpp"

namespace tl {
namespace {

enum Wire { VARINT = 0, I64 = 1, LEN = 2, I32 = 5 };

struct Field {
  uint32_t num = 0;
  int wire = 0;
  uint64_t value = 0;   // varint / fixed value
  uint64_t off = 0;     // LEN: payload offset (absolute)
  uint64_t len = 0;     // LEN: payload length
};

class Pb {
 public:
  Pb(const Bytes &b, uint64_t begin, uint64_t end) : b_(b), pos_(begin), end_(end) {}
  bool done() const { return pos_ >= end_; }
  uint64_t pos() const { return pos_; }

  uint64_t varint() {
    uint64_t v = 0;
    for (int shift = 0; shift < 64; shift += 7) {
      if (pos_ >= end_) throw ParseError{"truncated varint"};
      uint8_t c = b_.get<uint8_t>(pos_++);
      v |= uint64_t(c & 0x7f) << shift;
      if (!(c & 0x80)) return v;
    }
    throw ParseError{"varint too long"};
  }
  Field next() {
    Field f;
    uint64_t key = varint();
    f.num = static_cast<uint32_t>(key >> 3);
    f.wire = static_cast<int>(key & 7);
    if (f.num == 0) throw ParseError{"field number 0"};
    switch (f.wire) {
      case VARINT: f.value = varint(); break;
      case I64: need(8); f.value = b_.get<uint64_t>(pos_); pos_ += 8; break;
      case I32: need(4); f.value = b_.get<uint32_t>(pos_); pos_ += 4; break;
      case LEN:
        f.len = varint();
        need(f.len);
        f.off = pos_;
        pos_ += f.len;
        break;
      default: throw ParseError{"unsupported wire type"};
    }
    return f;
  }

 private:
  void need(uint64_t n) {
    if (n > end_ - pos_) throw ParseError{"field past end"};
  }
  const Bytes &b_;
  uint64_t pos_, end_;
};

// Iterate the fields of a sub-message.
void each(const Bytes &b, const Field &f, const std::function<void(const Field &)> &fn) {
  if (f.wire != LEN) throw ParseError{"expected message"};
  Pb pb(b, f.off, f.off + f.len);
  while (!pb.done()) fn(pb.next());
}

std::string str(const Bytes &b, const Field &f) {
  if (f.wire != LEN) throw ParseError{"expected string"};
  return std::string(b.view(f.off, f.len));
}

// ONNX TensorProto.DataType names.
const char *onnx_dtype(uint64_t t) {
  static const char *names[] = {"UNDEFINED", "FLOAT32", "UINT8", "INT8", "UINT16", "INT16", "INT32",
                                "INT64", "STRING", "BOOL", "FLOAT16", "FLOAT64", "UINT32", "UINT64",
                                "COMPLEX64", "COMPLEX128", "BFLOAT16", "FLOAT8E4M3FN", "FLOAT8E4M3FNUZ",
                                "FLOAT8E5M2", "FLOAT8E5M2FNUZ", "UINT4", "INT4", "FLOAT4E2M1"};
  return t < sizeof(names) / sizeof(names[0]) ? names[t] : "UNKNOWN";
}

struct ValueInfo { std::string name, dtype; std::vector<int64_t> shape; };

ValueInfo parse_value_info(const Bytes &b, const Field &vi) {
  ValueInfo out;
  each(b, vi, [&](const Field &f) {
    if (f.num == 1) out.name = str(b, f);
    if (f.num == 2 && f.wire == LEN)  // TypeProto
      each(b, f, [&](const Field &tf) {
        if (tf.num != 1 || tf.wire != LEN) return;  // tensor_type only
        each(b, tf, [&](const Field &tt) {
          if (tt.num == 1 && tt.wire == VARINT) out.dtype = onnx_dtype(tt.value);
          if (tt.num == 2 && tt.wire == LEN)  // TensorShapeProto
            each(b, tt, [&](const Field &dim) {
              if (dim.num != 1 || dim.wire != LEN) return;
              int64_t v = -1;
              each(b, dim, [&](const Field &d) {
                if (d.num == 1 && d.wire == VARINT) v = static_cast<int64_t>(d.value);
              });
              out.shape.push_back(v);
            });
        });
      });
  });
  return out;
}

struct Graphish {
  Graph g;
  std::unordered_map<std::string, int> index;
  int tensor(const std::string &name) {
    if (name.empty()) return -1;
    auto it = index.find(name);
    if (it != index.end()) return it->second;
    Tensor t;
    t.name = name;
    g.tensors.push_back(t);
    int id = static_cast<int>(g.tensors.size() - 1);
    index.emplace(name, id);
    return id;
  }
};

void parse_initializer(const Bytes &b, const Field &tp, Graphish &gg, uint64_t &weights, bool &external) {
  std::string name, dtype;
  std::vector<int64_t> dims;
  uint64_t off = kNoData, size = 0;
  each(b, tp, [&](const Field &f) {
    switch (f.num) {
      case 1:
        if (f.wire == VARINT) dims.push_back(static_cast<int64_t>(f.value));
        else if (f.wire == LEN) {  // packed dims
          Pb pb(b, f.off, f.off + f.len);
          while (!pb.done()) dims.push_back(static_cast<int64_t>(pb.varint()));
        }
        break;
      case 2: if (f.wire == VARINT) dtype = onnx_dtype(f.value); break;
      case 8: name = str(b, f); break;
      case 9: if (f.wire == LEN) { off = f.off; size = f.len; } break;          // raw_data
      case 4: case 5: case 6: case 7: case 10: case 11:                         // typed data
        if (f.wire == LEN && off == kNoData) { off = f.off; size = f.len; }
        break;
      case 14: if (f.wire == VARINT && f.value == 1) external = true; break;    // EXTERNAL
      default: break;
    }
  });
  int id = gg.tensor(name);
  if (id < 0) throw ParseError{"initializer without a name"};
  Tensor &t = gg.g.tensors[id];
  t.shape = dims;
  t.dtype = dtype;
  t.constant = true;
  t.data_offset = off;
  t.data_size = size;
  weights += size;
}

Graph parse_graph(const Bytes &b, const Field &gf, std::vector<std::string> &warnings, uint64_t &weights) {
  Graphish gg;
  std::vector<ValueInfo> inputs, outputs, infos;
  struct RawNode { Node n; std::vector<std::string> in, out; };
  std::vector<RawNode> nodes;
  bool external = false, subgraphs = false;
  each(b, gf, [&](const Field &f) {
    switch (f.num) {
      case 1: {  // NodeProto
        RawNode rn;
        each(b, f, [&](const Field &nf) {
          switch (nf.num) {
            case 1: rn.in.push_back(str(b, nf)); break;
            case 2: rn.out.push_back(str(b, nf)); break;
            case 3: rn.n.name = str(b, nf); break;
            case 4: rn.n.op = str(b, nf); break;
            case 7: rn.n.domain = str(b, nf); break;
            case 5:  // AttributeProto: nested graphs (If/Loop/Scan bodies) and tensor values
              each(b, nf, [&](const Field &af) {
                if (af.num == 6 || af.num == 11) subgraphs = true;
                if (af.num == 5 && af.wire == LEN)  // t: TensorProto, e.g. a Constant node's value
                  each(b, af, [&](const Field &tf) {
                    if ((tf.num == 9 || (tf.num >= 4 && tf.num <= 7) || tf.num == 10 || tf.num == 11) && tf.wire == LEN)
                      weights += tf.len;
                  });
              });
              break;
            default: break;
          }
        });
        if (rn.n.op.empty()) throw ParseError{"node without op_type"};
        nodes.push_back(std::move(rn));
        break;
      }
      case 2: gg.g.name = str(b, f); break;
      case 5: parse_initializer(b, f, gg, weights, external); break;
      case 11: inputs.push_back(parse_value_info(b, f)); break;
      case 12: outputs.push_back(parse_value_info(b, f)); break;
      case 13: infos.push_back(parse_value_info(b, f)); break;
      default: break;
    }
  });
  auto apply = [&](const ValueInfo &vi) {
    int id = gg.tensor(vi.name);
    if (id < 0) return id;
    Tensor &t = gg.g.tensors[id];
    if (t.dtype.empty()) t.dtype = vi.dtype;
    if (t.shape.empty()) t.shape = vi.shape;
    return id;
  };
  for (const ValueInfo &vi : inputs) {
    int id = apply(vi);
    if (id >= 0 && !gg.g.tensors[id].constant) gg.g.inputs.push_back(id);
  }
  for (const ValueInfo &vi : outputs) gg.g.outputs.push_back(apply(vi));
  for (const ValueInfo &vi : infos) apply(vi);
  for (RawNode &rn : nodes) {
    for (const std::string &s : rn.in) rn.n.inputs.push_back(gg.tensor(s));
    for (const std::string &s : rn.out) rn.n.outputs.push_back(gg.tensor(s));
    const std::string &d = rn.n.domain;
    rn.n.custom = !(d.empty() || d == "ai.onnx" || d == "ai.onnx.ml");
    gg.g.nodes.push_back(std::move(rn.n));
  }
  if (external) warnings.push_back("weights stored in external data files");
  if (subgraphs) warnings.push_back("control-flow subgraphs (If/Loop/Scan) not expanded");
  return std::move(gg.g);
}

}  // namespace

ParseResult parse_onnx(const uint8_t *p, size_t avail) {
  ParseResult r;
  try {
    Bytes b(p, avail);
    Pb pb(b, 0, avail);
    Model &m = r.model;
    m.format = Format::ONNX;
    uint64_t ir = 0, weights = 0, end = 0;
    bool have_graph = false;
    Field graph;
    std::map<std::string, uint64_t> opsets;
    std::string producer, producer_ver;
    bool first = true;
    while (!pb.done()) {
      uint64_t before = pb.pos();
      Field f;
      try {
        f = pb.next();
      } catch (const ParseError &) {
        if (first) throw;
        break;  // trailing bytes after an embedded model
      }
      // Fields ModelProto defines, with their expected wire types; anything else ends the model.
      static const std::map<uint32_t, int> known = {{1, VARINT}, {2, LEN}, {3, LEN}, {4, LEN}, {5, VARINT},
                                                    {6, LEN},    {7, LEN}, {8, LEN}, {14, LEN}, {20, LEN},
                                                    {25, LEN}};
      auto it = known.find(f.num);
      if (it == known.end() || it->second != f.wire) {
        if (first) throw ParseError{"not an ONNX ModelProto"};
        (void)before;
        break;
      }
      if (first && f.num != 1) throw ParseError{"ModelProto must start with ir_version"};
      first = false;
      switch (f.num) {
        case 1: ir = f.value; break;
        case 2: producer = str(b, f); break;
        case 3: producer_ver = str(b, f); break;
        case 7: graph = f; have_graph = true; break;
        case 8: {
          std::string dom; uint64_t ver = 0;
          each(b, f, [&](const Field &o) {
            if (o.num == 1) dom = str(b, o);
            if (o.num == 2 && o.wire == VARINT) ver = o.value;
          });
          opsets[dom.empty() ? "ai.onnx" : dom] = ver;
          break;
        }
        case 14: {
          std::string k, v;
          each(b, f, [&](const Field &kv) {
            if (kv.num == 1) k = str(b, kv);
            if (kv.num == 2) v = str(b, kv);
          });
          if (v.size() > 120) v = v.substr(0, 117) + "...";
          m.metadata.emplace_back(k, v);
          break;
        }
        default: break;
      }
      end = pb.pos();
    }
    if (ir < 1 || ir > 20) throw ParseError{"implausible ir_version"};
    if (!have_graph) throw ParseError{"no graph"};
    if (!printable_ascii(producer) || !printable_ascii(producer_ver)) throw ParseError{"binary producer name"};
    uint64_t init_weights = 0;
    m.graphs.push_back(parse_graph(b, graph, m.warnings, init_weights));
    // parse_graph counts initializer bytes (also recorded per tensor) and attribute tensors.
    uint64_t tensor_bytes = 0;
    for (const Tensor &t : m.graphs[0].tensors) if (t.constant) tensor_bytes += t.data_size;
    m.attr_weight_bytes = init_weights > tensor_bytes ? init_weights - tensor_bytes : 0;
    (void)weights;
    if (m.graphs[0].nodes.empty()) throw ParseError{"graph has no nodes"};
    for (const Node &n : m.graphs[0].nodes)
      if (!printable_ascii(n.op) || n.op.size() > 64) throw ParseError{"implausible op_type"};
    m.producer = producer + (producer_ver.empty() ? "" : " " + producer_ver);
    m.version = "IR v" + std::to_string(ir);
    for (auto &[dom, ver] : opsets) m.version += ", " + dom + " opset " + std::to_string(ver);
    auto it = opsets.find("ai.onnx");
    for (Node &n : m.graphs[0].nodes)
      if (!n.custom && it != opsets.end()) n.op_version = static_cast<int>(it->second);
    m.size = end;
    r.ok = true;
  } catch (const ParseError &e) {
    r.ok = false;
    r.error = e.what;
  } catch (const std::bad_alloc &) {
    r.ok = false;
    r.error = "out of memory";
  }
  return r;
}

}  // namespace tl
