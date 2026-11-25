// TFLite flatbuffer reader. Implements just the parts of the FlatBuffers binary format the
// TFLite schema uses, with every offset checked, and records the furthest byte touched so a
// model embedded in a larger binary gets an exact size.
#include <algorithm>
#include <cstdio>

#include "model.hpp"
#include "reader.hpp"

namespace tl {
namespace {

constexpr size_t kMaxItems = 1u << 20;  // sanity cap on vector lengths

struct Fb {
  Bytes b;
  uint64_t end = 0;  // extent of the flatbuffer seen so far
  void touch(uint64_t off, uint64_t len) {
    b.need(off, len);
    end = std::max(end, off + len);
  }
  uint32_t u32(uint64_t off) { touch(off, 4); return b.get<uint32_t>(off); }
};

// A table: position of its start, plus its vtable.
struct Table {
  Fb *fb = nullptr;
  uint64_t pos = 0;
  uint64_t vt = 0;
  uint16_t vt_size = 0;

  static Table at(Fb &fb, uint64_t pos) {
    Table t;
    t.fb = &fb;
    t.pos = pos;
    fb.touch(pos, 4);
    int32_t soff = fb.b.get<int32_t>(pos);
    int64_t vt = static_cast<int64_t>(pos) - soff;
    if (vt < 0) throw ParseError{"vtable before buffer"};
    t.vt = static_cast<uint64_t>(vt);
    fb.touch(t.vt, 4);
    t.vt_size = fb.b.get<uint16_t>(t.vt);
    uint16_t tbl_size = fb.b.get<uint16_t>(t.vt + 2);
    if (t.vt_size < 4 || (t.vt_size & 1)) throw ParseError{"bad vtable"};
    fb.touch(t.vt, t.vt_size);
    fb.touch(pos, tbl_size ? tbl_size : 4);
    return t;
  }
  // Absolute offset of field `idx`, or 0 when absent.
  uint64_t field(int idx) const {
    uint64_t slot = 4 + 2ull * idx;
    if (slot + 2 > vt_size) return 0;
    uint16_t off = fb->b.get<uint16_t>(vt + slot);
    return off ? pos + off : 0;
  }
  template <class T>
  T scalar(int idx, T def) const {
    uint64_t f = field(idx);
    if (!f) return def;
    fb->touch(f, sizeof(T));
    return fb->b.get<T>(f);
  }
  uint64_t deref(int idx) const {  // follow a uoffset field; 0 when absent
    uint64_t f = field(idx);
    if (!f) return 0;
    return f + fb->u32(f);
  }
  bool has(int idx) const { return field(idx) != 0; }
  std::string str(int idx) const {
    uint64_t s = deref(idx);
    if (!s) return {};
    uint32_t len = fb->u32(s);
    fb->touch(s + 4, uint64_t(len) + 1);
    return std::string(fb->b.view(s + 4, len));
  }
  // Vector: returns element count and the offset of element 0.
  bool vec(int idx, uint32_t &count, uint64_t &first, size_t elem) const {
    uint64_t v = deref(idx);
    if (!v) { count = 0; first = 0; return false; }
    count = fb->u32(v);
    // Byte vectors are weight buffers and can be any size (touch() checks them against the data);
    // the cap guards loops over tables and scalars.
    if (elem > 1 && count > kMaxItems) throw ParseError{"vector too long"};
    first = v + 4;
    fb->touch(first, uint64_t(count) * elem);
    return true;
  }
  template <class T>
  std::vector<T> vec_scalar(int idx) const {
    uint32_t n; uint64_t first;
    std::vector<T> out;
    if (!vec(idx, n, first, sizeof(T))) return out;
    out.reserve(n);
    for (uint32_t i = 0; i < n; i++) out.push_back(fb->b.get<T>(first + uint64_t(i) * sizeof(T)));
    return out;
  }
  std::vector<Table> vec_table(int idx) const {
    uint32_t n; uint64_t first;
    std::vector<Table> out;
    if (!vec(idx, n, first, 4)) return out;
    out.reserve(n);
    for (uint32_t i = 0; i < n; i++) {
      uint64_t e = first + 4ull * i;
      out.push_back(Table::at(*fb, e + fb->u32(e)));
    }
    return out;
  }
  Table sub(int idx) const {
    uint64_t s = deref(idx);
    if (!s) return Table{};
    return Table::at(*fb, s);
  }
  bool valid() const { return fb != nullptr; }
};

std::string quant_str(const Table &q, size_t &channels) {
  channels = 0;
  if (!q.valid()) return {};
  auto scale = q.vec_scalar<float>(2);
  auto zp = q.vec_scalar<int64_t>(3);
  if (scale.empty()) return {};
  channels = scale.size();
  char buf[96];
  if (scale.size() == 1) {
    std::snprintf(buf, sizeof buf, "scale=%g zp=%lld", scale[0], zp.empty() ? 0LL : (long long)zp[0]);
  } else {
    std::snprintf(buf, sizeof buf, "per-channel(%zu) axis=%d", scale.size(), q.scalar<int32_t>(6, 0));
  }
  return buf;
}

}  // namespace

ParseResult parse_tflite(const uint8_t *p, size_t avail) {
  ParseResult r;
  try {
    Fb fb{Bytes(p, avail)};
    if (avail < 16 || std::memcmp(p + 4, "TFL3", 4) != 0) throw ParseError{"no TFL3 identifier"};
    uint32_t root = fb.u32(0);
    fb.touch(0, 8);
    if (root < 8) throw ParseError{"bad root offset"};
    Table m = Table::at(fb, root);

    Model &mod = r.model;
    mod.format = Format::TFLite;
    uint32_t schema = m.scalar<uint32_t>(0, 0);
    if (schema > 10) throw ParseError{"implausible schema version"};
    mod.version = "schema v" + std::to_string(schema);
    mod.producer = m.str(3);  // description, e.g. "MLIR Converted."

    // Operator codes.
    struct Code { int builtin; std::string custom; int version; };
    std::vector<Code> codes;
    for (const Table &oc : m.vec_table(1)) {
      int8_t dep = oc.scalar<int8_t>(0, 0);
      int32_t bc = oc.scalar<int32_t>(3, 0);
      // builtin_code supersedes deprecated_builtin_code once codes passed 127.
      int code = std::max<int>(dep, bc);
      codes.push_back({code, oc.str(1), oc.scalar<int32_t>(2, 1)});
    }

    // Buffers: remember where each buffer's bytes live (inline data or offset/size).
    struct Buf { uint64_t off = kNoData, size = 0; };
    std::vector<Buf> bufs;
    for (const Table &b : m.vec_table(4)) {
      Buf bb;
      uint32_t n; uint64_t first;
      if (b.vec(0, n, first, 1) && n) { bb.off = first; bb.size = n; }
      uint64_t eoff = b.scalar<uint64_t>(1, 0), esz = b.scalar<uint64_t>(2, 0);
      if (eoff > 1 && esz) {  // large-model layout: data appended after the flatbuffer
        if (eoff > avail || esz > avail - eoff) throw ParseError{"buffer outside input"};
        bb.off = eoff; bb.size = esz;
        fb.end = std::max(fb.end, eoff + esz);
      }
      bufs.push_back(bb);
    }

    // Metadata entries (name -> buffer); keep short printable ones such as min_runtime_version.
    for (const Table &md : m.vec_table(6)) {
      std::string name = md.str(0);
      uint32_t bi = md.scalar<uint32_t>(1, 0);
      std::string val;
      if (bi < bufs.size() && bufs[bi].off != kNoData) {
        uint64_t n = std::min<uint64_t>(bufs[bi].size, 64);
        std::string_view v = fb.b.view(bufs[bi].off, n);
        std::string s(v.substr(0, v.find('\0')));
        val = printable_ascii(s) ? s : "<" + format_bytes(bufs[bi].size) + " binary>";
      }
      mod.metadata.emplace_back(name, val);
    }
    for (const Table &sd : m.vec_table(7)) mod.metadata.emplace_back("signature", sd.str(2));

    auto subgraphs = m.vec_table(2);
    if (subgraphs.empty()) throw ParseError{"no subgraphs"};
    for (const Table &sg : subgraphs) {
      Graph g;
      g.name = sg.str(4);
      for (const Table &t : sg.vec_table(0)) {
        Tensor tt;
        tt.name = t.str(3);
        for (int32_t d : t.vec_scalar<int32_t>(0)) tt.shape.push_back(d);
        auto sig = t.vec_scalar<int32_t>(7);  // shape_signature carries -1 for dynamic dims
        if (sig.size() == tt.shape.size())
          for (size_t i = 0; i < sig.size(); i++) if (sig[i] < 0) tt.shape[i] = -1;
        int type = t.scalar<int8_t>(1, 0);
        const char *tn = tflite_type_name(type);
        tt.dtype = tn ? tn : "TYPE_" + std::to_string(type);
        uint32_t bi = t.scalar<uint32_t>(2, 0);
        if (bi > 0 && bi < bufs.size() && bufs[bi].off != kNoData) {
          tt.constant = true;
          tt.data_offset = bufs[bi].off;
          tt.data_size = bufs[bi].size;
        }
        size_t ch;
        tt.quant = quant_str(t.sub(4), ch);
        g.tensors.push_back(std::move(tt));
      }
      for (int32_t i : sg.vec_scalar<int32_t>(1)) g.inputs.push_back(i);
      for (int32_t i : sg.vec_scalar<int32_t>(2)) g.outputs.push_back(i);
      const int nt = static_cast<int>(g.tensors.size());
      auto fix = [nt](int32_t i) { return (i >= 0 && i < nt) ? i : -1; };
      for (int &i : g.inputs) i = fix(i);
      for (int &i : g.outputs) i = fix(i);
      for (const Table &op : sg.vec_table(3)) {
        Node n;
        uint32_t oi = op.scalar<uint32_t>(0, 0);
        if (oi >= codes.size()) throw ParseError{"opcode index out of range"};
        const Code &c = codes[oi];
        n.op_version = c.version;
        if (c.builtin == 32 /* CUSTOM */ || !c.custom.empty()) {
          n.custom = true;
          n.domain = "custom";
          n.op = c.custom.empty() ? "CUSTOM" : c.custom;
          n.builtin_code = 32;
        } else {
          const char *on = tflite_op_name(c.builtin);
          n.op = on ? on : "BUILTIN_" + std::to_string(c.builtin);
          n.builtin_code = c.builtin;
        }
        for (int32_t i : op.vec_scalar<int32_t>(1)) n.inputs.push_back(fix(i));
        for (int32_t i : op.vec_scalar<int32_t>(2)) n.outputs.push_back(fix(i));
        // Touch option tables / custom options so the size estimate covers them.
        if (op.has(4)) op.sub(4);
        uint32_t cn; uint64_t cf;
        op.vec(5, cn, cf, 1);
        if (op.has(12)) op.sub(12);
        g.nodes.push_back(std::move(n));
      }
      mod.graphs.push_back(std::move(g));
    }
    mod.size = fb.end;
    // FlatBuffers pad to 4/8/16 bytes; round the size up when padding is present in the input.
    uint64_t aligned = (mod.size + 15) & ~uint64_t(15);
    if (aligned <= avail) {
      bool zeros = true;
      for (uint64_t i = mod.size; i < aligned; i++) zeros &= p[i] == 0;
      if (zeros) mod.size = aligned;
    }
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
