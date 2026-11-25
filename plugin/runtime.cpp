// Recover the TFLite runtime inside a native library: which functions are the Prepare/Eval/Init/
// Free kernels of which operator, where their TfLiteRegistration tables live, and which
// delegates and other ML runtimes are present.
//
// Three independent sources of evidence, combined:
//   1. Symbols: tflite::ops::builtin::Register_CONV_2D() and friends, when the build kept them.
//   2. __FILE__ strings: every TF_LITE_ENSURE in a kernel leaves "tensorflow/lite/kernels/conv.cc"
//      in .rodata, referenced from that kernel's functions. Survives stripping.
//   3. The op resolver: BuiltinOpResolver's constructor calls AddBuiltin(op, Register_X(), min, max)
//      a few hundred times in a row. Reading each call's arguments gives op code -> registration,
//      which names kernels exactly even in fully stripped builds with no strings.
#include <algorithm>
#include <cctype>
#include <cstring>

#include "state.hpp"

#include <bytes.hpp>
#include <funcs.hpp>
#include <kernwin.hpp>
#include <lines.hpp>
#include <name.hpp>
#include <segment.hpp>
#include <strlist.hpp>
#include <ua.hpp>
#include <xref.hpp>

namespace tl::ida {
namespace {

struct Seg { ea_t start, end; bool exec; };

struct Ctx {
  State &st;
  std::vector<Seg> segs;
  int ptr = 8;
  std::map<ea_t, Kernel> tables;  // keyed by registration table address

  explicit Ctx(State &s) : st(s) {
    ptr = inf_is_64bit() ? 8 : 4;
    for (ea_t s = get_first_segment_ea(); s != BADADDR; s = get_next_segment_ea(s)) {
      segment_info_t si;
      if (!get_segment_info(&si, s)) continue;
      segs.push_back({si.start_ea, si.end_ea, (si.get_perm() & SEGPERM_EXEC) != 0 || si.get_type() == SEG_CODE});
    }
  }
  const Seg *seg_of(ea_t ea) const {
    for (const Seg &s : segs)
      if (ea >= s.start && ea < s.end) return &s;
    return nullptr;
  }
  bool is_exec(ea_t ea) const {
    const Seg *s = seg_of(ea);
    return s != nullptr && s->exec;
  }
  ea_t read_ptr(ea_t ea) const { return ptr == 8 ? ea_t(get_qword(ea)) : ea_t(get_dword(ea)); }
};

std::string lower(std::string s) {
  for (char &c : s) c = char(std::tolower((unsigned char)c));
  return s;
}

// Fold variant registrations (CONV_2D_UINT8, ADD_REF, TANH_GENERIC_OPT, ...) into the base op.
std::string base_op(const std::string &name, int *code) {
  static const char *suffixes[] = {"_GENERIC_OPT_UINT8", "_GENERIC_OPT", "_FIXED_POINT_OPT", "_MULTITHREADED_OPT",
                                   "_CBLAS_OPT", "_NEON_OPT", "_REF", "_OPT", "_UINT8", "_INT8", "_INT16", "_FLOAT"};
  // Kernel families whose Register_ function names differ from the operator they implement.
  static const std::map<std::string, std::string> aliases = {
      {"AVERAGE_POOL", "AVERAGE_POOL_2D"}, {"MAX_POOL", "MAX_POOL_2D"}, {"L2_POOL", "L2_POOL_2D"},
      {"CONVOLUTION", "CONV_2D"}, {"DEPTHWISE_CONVOLUTION", "DEPTHWISE_CONV_2D"}, {"DEPTHWISE_CONV", "DEPTHWISE_CONV_2D"},
      {"L2NORM", "L2_NORMALIZATION"}, {"LOCAL_RESPONSE_NORM", "LOCAL_RESPONSE_NORMALIZATION"},
      {"TRANSPOSECONV", "TRANSPOSE_CONV"}, {"CONV", "CONV_2D"}};
  auto lookup = [](const std::string &n0) {
    auto a = aliases.find(n0);
    const std::string &n = a == aliases.end() ? n0 : a->second;
    for (int c = 0; c < 512; c++) {
      const char *on = tflite_op_name(c);
      if (on && n == on) return c;
    }
    return -1;
  };
  int c = lookup(name);
  if (c >= 0) { if (code) *code = c; return tflite_op_name(c); }
  for (const char *suf : suffixes) {
    size_t n = strlen(suf);
    if (name.size() > n && name.compare(name.size() - n, n, suf) == 0) {
      std::string b = name.substr(0, name.size() - n);
      c = lookup(b);
      if (c >= 0) { if (code) *code = c; return tflite_op_name(c); }
    }
  }
  if (code) *code = -1;
  return name;
}

// Validate a TfLiteRegistration at `t`:
//   { init, free, prepare, invoke, profiling_string, int32 builtin_code, const char *custom_name, int version, ... }
bool read_registration(const Ctx &c, ea_t t, Kernel &k) {
  if (t == BADADDR || t % c.ptr || !is_loaded(t) || !is_loaded(t + 8 * c.ptr - 1)) return false;
  if (c.is_exec(t)) return false;  // tables live in data
  ea_t slot[5];
  for (int i = 0; i < 5; i++) slot[i] = c.read_ptr(t + i * c.ptr);
  if (slot[3] == 0 || !c.is_exec(slot[3])) return false;  // invoke is mandatory
  for (int i : {0, 1, 2, 4})
    if (slot[i] != 0 && !c.is_exec(slot[i])) return false;
  int32 builtin = int32(get_dword(t + 5 * c.ptr));
  if (builtin < 0 || builtin > 1000) return false;
  ea_t custom_name = c.read_ptr(t + 6 * c.ptr);
  if (custom_name != 0 && (c.is_exec(custom_name) || !is_loaded(custom_name))) return false;
  int32 version = int32(get_dword(t + 7 * c.ptr));
  if (version < 0 || version > 100) return false;
  k.table = t;
  k.init = slot[0] ? slot[0] : BADADDR;
  k.free = slot[1] ? slot[1] : BADADDR;
  k.prepare = slot[2] ? slot[2] : BADADDR;
  k.invoke = slot[3];
  if (custom_name != 0) {
    qstring s;
    if (get_strlit_contents(&s, custom_name, 64, STRTYPE_C) > 0) {
      k.op = s.c_str();
      k.custom = true;
    }
  }
  return true;
}

Kernel *add_table(Ctx &c, ea_t t) {
  auto it = c.tables.find(t);
  if (it != c.tables.end()) return &it->second;
  Kernel k;
  if (!read_registration(c, t, k)) return nullptr;
  return &c.tables.emplace(t, k).first->second;
}

template <class Fn>
void for_each_head(ea_t func, Fn fn) {
  rangeset_t rs;
  if (get_func_ranges_ea(&rs, func) == BADADDR) return;
  for (size_t i = 0; i < rs.nranges(); i++) {
    const range_t &r = rs.getrange(int(i));
    for (ea_t ea = r.start_ea; ea != BADADDR && ea < r.end_ea; ea = next_head(ea, r.end_ea)) fn(ea);
  }
}

// Registration tables a (small) function points at: Register_X() is `return &static_registration;`.
// Follow PLT stubs and other jump thunks to the function they forward to. Exported functions in
// a shared library are usually called through the PLT, even from inside the same library.
ea_t resolve_thunk(const Ctx &c, ea_t f) {
  for (int depth = 0; depth < 4 && f != BADADDR; depth++) {
    // .plt.sec stubs start with endbr64 (CET): skip landing pads and padding.
    ea_t j = f;
    std::string mn;
    for (int k = 0; k < 3; k++) {
      insn_t insn;
      if (decode_insn(&insn, j) <= 0) { mn.clear(); break; }
      qstring mq;
      print_insn_mnem(&mq, j);
      mn = lower(mq.c_str());
      if (mn != "endbr64" && mn != "endbr32" && mn != "nop" && mn != "bti") break;
      j += insn.size;
    }
    if (mn != "jmp" && mn != "b" && mn != "br" && mn != "bx") break;
    ea_t next = BADADDR;
    xrefblk_t xb;
    for (bool ok = xb.first_from(j, XREF_CODE | XREF_NOFLOW); ok; ok = xb.next_from())
      if (xb.iscode) { next = xb.to; break; }
    if (next == BADADDR && xb.first_from(j, XREF_DATA)) {
      ea_t slot = xb.to;
      ea_t v = c.read_ptr(slot);
      next = c.is_exec(v) ? v : BADADDR;
      if (next == BADADDR) {
        // GOT slot points at an extern: find the local definition with the same name.
        qstring n;
        if (get_ea_name(&n, v != 0 && is_loaded(v) ? v : slot) > 0) {
          std::string s = n.c_str();
          for (const char *pre : {"__imp_", "__imp__", "."})
            if (s.rfind(pre, 0) == 0) s = s.substr(strlen(pre));
          ea_t local = get_name_ea(BADADDR, s.c_str());
          if (local != BADADDR && c.is_exec(local) && local != f) next = local;
        }
      }
    }
    if (next == BADADDR || next == f) break;
    if (c.seg_of(next) && !c.is_exec(next)) break;
    f = next;
  }
  return f;
}

std::vector<ea_t> tables_referenced_by(Ctx &c, ea_t func, size_t max_size) {
  std::vector<ea_t> out;
  func = resolve_thunk(c, func);
  if (calc_func_size_ea(func) > max_size) return out;
  for_each_head(func, [&](ea_t ea) {
    xrefblk_t xb;
    for (bool ok = xb.first_from(ea, XREF_DATA); ok; ok = xb.next_from())
      if (add_table(c, xb.to) != nullptr) out.push_back(xb.to);
  });
  return out;
}

// --- 1. symbols ---------------------------------------------------------------------------
void from_symbols(Ctx &c) {
  size_t n = get_nlist_size();
  for (size_t i = 0; i < n; i++) {
    const char *nm = get_nlist_name(i);
    if (nm == nullptr) continue;
    const char *r = strstr(nm, "Register_");
    if (r == nullptr || strstr(nm, "tflite") == nullptr) continue;
    std::string op;
    const char *p = r + 9;
    for (; *p && (std::isupper((unsigned char)*p) || std::isdigit((unsigned char)*p) || *p == '_'); p++) op += *p;
    // Itanium mangling ends the nested name with 'E' before the parameter list: Register_ABSEv.
    if (op.size() > 1 && op.back() == 'E' && *p == 'v') op.pop_back();
    while (!op.empty() && op.back() == '_') op.pop_back();
    if (op.empty()) continue;
    ea_t fn = get_nlist_ea(i);
    if (!c.is_exec(fn)) continue;
    for (ea_t t : tables_referenced_by(c, fn, 256)) {
      Kernel *k = add_table(c, t);
      if (k == nullptr || (!k->op.empty() && k->evidence.find("resolver") != std::string::npos)) continue;
      int code;
      std::string b = base_op(op, &code);
      if (k->op.empty() || !k->custom) {
        k->op = b;
        k->builtin_code = code;
      }
      k->register_fn = fn;
      k->evidence = "symbol Register_" + op;
      if (b != op) k->evidence += " (variant)";
    }
  }
}

// --- 2. __FILE__ strings -------------------------------------------------------------------
void from_strings(Ctx &c) {
  build_strlist();
  size_t n = get_strlist_qty();
  for (size_t i = 0; i < n; i++) {
    string_info_t si;
    if (!get_strlist_item(&si, i) || si.length < 12) continue;
    qstring s;
    if (get_strlit_contents(&s, si.ea, si.length, si.type) <= 0) continue;
    std::string str = s.c_str();
    std::replace(str.begin(), str.end(), '\\', '/');
    std::string file, kind;
    size_t p;
    if ((p = str.find("tensorflow/lite/kernels/")) != std::string::npos) {
      c.st.runtime.tflite = true;
      file = str.substr(p + 24);
      kind = "tflite";
    } else if ((p = str.find("tensorflow/lite/delegates/")) != std::string::npos) {
      c.st.runtime.tflite = true;
      std::string d = str.substr(p + 26);
      c.st.runtime.delegates.insert(d.substr(0, d.find('/')));
      continue;
    } else if ((p = str.find("onnxruntime/core/providers/")) != std::string::npos) {
      c.st.runtime.onnxruntime = true;
      std::string prov = str.substr(p + 27);
      prov = prov.substr(0, prov.find('/'));
      if (prov != "cpu") c.st.runtime.delegates.insert("onnxruntime:" + prov);
      file = "onnxruntime/" + str.substr(p + 27);
      kind = "ort";
    } else if (str.find("tensorflow/lite/") != std::string::npos || str.find("TfLite") != std::string::npos) {
      c.st.runtime.tflite = true;
      continue;
    } else if (str.find("ncnn") != std::string::npos && str.find("layer") != std::string::npos) {
      c.st.runtime.ncnn = true;
      continue;
    } else {
      continue;
    }
    // Keep only the source file name ("conv.cc", "nn/conv.cc" for ORT).
    size_t end = file.find_first_of(" :\"'\n");
    if (end != std::string::npos) file = file.substr(0, end);
    if (file.size() < 3 || file.find(".c") == std::string::npos) continue;
    xrefblk_t xb;
    for (bool ok = xb.first_to(si.ea, XREF_DATA); ok; ok = xb.next_to()) {
      ea_t f = get_func_start(xb.from);
      if (f != BADADDR) c.st.runtime.func_files[f].insert(file);
    }
  }
  // Tables reachable from functions that reference a kernel file: look for data slots that
  // point at the function and validate a registration around them.
  for (const auto &[f, files] : c.st.runtime.func_files) {
    if (files.begin()->rfind("onnxruntime/", 0) == 0) continue;
    xrefblk_t xb;
    for (bool ok = xb.first_to(f, XREF_DATA); ok; ok = xb.next_to()) {
      if (c.is_exec(xb.from)) continue;
      for (int slot = 0; slot < 4; slot++) {
        if (xb.from < ea_t(slot * c.ptr)) break;
        if (add_table(c, xb.from - slot * c.ptr) != nullptr) break;
      }
    }
  }
}

// --- 3. the op resolver --------------------------------------------------------------------
// BuiltinOpResolver's constructor is a long straight run of
//     AddBuiltin(this, BuiltinOperator_X, Register_X(), min_version, max_version);
// Compilers load the operator code in several ways (mov esi, imm / push imm; pop rsi / a copy of
// a constant kept in a callee-saved register / mov w1, #imm), so a small register-constant
// tracker walks the function once and reads the argument registers at every call.

struct Abi {
  std::string op_reg, reg_reg, ret;  // canonical names: AddBuiltin's op and registration args
  std::set<std::string> volatile_regs;
  bool ok = false;
};

Abi current_abi() {
  Abi a;
  if (PH.id == PLFM_386 && inf_is_64bit()) {
    a.ret = "rax";
    if (inf_get_filetype() == f_PE) {
      a.op_reg = "rdx"; a.reg_reg = "r8";
      a.volatile_regs = {"rax", "rcx", "rdx", "r8", "r9", "r10", "r11"};
    } else {
      a.op_reg = "rsi"; a.reg_reg = "rdx";
      a.volatile_regs = {"rax", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11"};
    }
    a.ok = true;
  } else if (PH.id == PLFM_ARM && inf_is_64bit()) {
    a.ret = "x0"; a.op_reg = "x1"; a.reg_reg = "x2";
    for (int i = 0; i <= 18; i++) a.volatile_regs.insert("x" + std::to_string(i));
    a.ok = true;
  } else if (PH.id == PLFM_ARM) {
    a.ret = "r0"; a.op_reg = "r1"; a.reg_reg = "r2";
    a.volatile_regs = {"r0", "r1", "r2", "r3", "r12"};
    a.ok = true;
  }
  return a;  // 32-bit x86 passes arguments on the stack: not handled
}

// esi/si/sil -> rsi, r12d -> r12, w1 -> x1; anything else lower-cased.
std::string canon_reg(std::string r) {
  r = lower(r);
  static const char *fam[][5] = {{"rax", "eax", "ax", "al", "ah"}, {"rcx", "ecx", "cx", "cl", "ch"},
                                 {"rdx", "edx", "dx", "dl", "dh"}, {"rbx", "ebx", "bx", "bl", "bh"},
                                 {"rsi", "esi", "si", "sil", ""},  {"rdi", "edi", "di", "dil", ""},
                                 {"rbp", "ebp", "bp", "bpl", ""},  {"rsp", "esp", "sp", "spl", ""}};
  for (auto &f : fam)
    for (const char *n : f)
      if (*n && r == n) return f[0];
  if (r.size() >= 2 && r[0] == 'r' && std::isdigit((unsigned char)r[1])) {
    while (!r.empty() && (r.back() == 'd' || r.back() == 'w' || r.back() == 'b')) r.pop_back();
    return r;
  }
  if (r.size() >= 2 && r[0] == 'w' && std::isdigit((unsigned char)r[1])) return "x" + r.substr(1);
  return r;
}

struct Val {
  enum Kind { Unknown, Const, Addr, Ret } kind = Unknown;
  uval_t v = 0;  // constant / address / called function
};

std::string opnd_reg(ea_t ea, int n) {
  qstring r;
  print_operand(&r, ea, n);
  tag_remove(&r);
  return canon_reg(r.c_str());
}

ea_t call_target(ea_t ea) {
  xrefblk_t xb;
  for (bool ok = xb.first_from(ea, XREF_CODE | XREF_NOFLOW); ok; ok = xb.next_from())
    if (xb.iscode) return xb.to;
  return BADADDR;
}

struct Decoded {
  ea_t site;
  int code;            // builtin op code, or -1 for a custom op
  std::string custom;  // custom op name
  ea_t table;
};

// Walk `func` once; for every call to one of `targets`, decode (op, registration).
std::vector<Decoded> decode_calls(Ctx &c, const Abi &abi, ea_t func, const std::set<ea_t> &targets, int *total) {
  std::vector<Decoded> out;
  std::map<std::string, Val> regs;
  bool have_push = false;
  uval_t pushed = 0;
  *total = 0;
  for_each_head(func, [&](ea_t ea) {
    insn_t insn;
    if (decode_insn(&insn, ea) <= 0) return;
    qstring mq;
    print_insn_mnem(&mq, ea);
    std::string mn = lower(mq.c_str());
    if (is_call_insn(insn)) {
      ea_t t = call_target(ea);
      if (t != BADADDR && targets.count(t)) {
        (*total)++;
        Val op = regs[abi.op_reg], rg = regs[abi.reg_reg];
        std::vector<ea_t> tables;
        if (rg.kind == Val::Ret) tables = tables_referenced_by(c, ea_t(rg.v), 128);
        else if (rg.kind == Val::Addr && add_table(c, ea_t(rg.v)) != nullptr) tables.push_back(ea_t(rg.v));
        if (tables.size() == 1) {
          if (op.kind == Val::Const && op.v < 512 && tflite_op_name(int(op.v))) {
            out.push_back({ea, int(op.v), "", tables[0]});
          } else if (op.kind == Val::Addr) {
            qstring s;
            if (get_strlit_contents(&s, ea_t(op.v), 64, STRTYPE_C) > 1) out.push_back({ea, -1, s.c_str(), tables[0]});
          }
        }
      }
      for (const std::string &r : abi.volatile_regs) regs.erase(r);
      if (t != BADADDR) regs[abi.ret] = {Val::Ret, uval_t(t)};
      have_push = false;
      return;
    }
    if (mn == "push") {
      have_push = insn.ops[0].type == o_imm;
      pushed = insn.ops[0].value;
      return;
    }
    if (insn.ops[0].type != o_reg) {
      have_push = false;
      return;
    }
    std::string dst = opnd_reg(ea, 0);
    if (mn == "pop") {
      regs[dst] = have_push ? Val{Val::Const, pushed} : Val{};
      have_push = false;
      return;
    }
    have_push = false;
    if (!(insn.get_canon_feature(PH) & CF_CHG1)) return;
    // Address loads: lea / adrp+add / ldr =label, all of which IDA resolves to a data xref.
    xrefblk_t xb;
    if (xb.first_from(ea, XREF_DATA) && (mn == "lea" || mn == "add" || mn == "adr" || mn == "adrl" || mn == "ldr")) {
      regs[dst] = {Val::Addr, uval_t(xb.to)};
      return;
    }
    if ((mn == "mov" || mn == "movz" || mn == "movw" || mn == "movs") && insn.ops[1].type == o_imm) {
      regs[dst] = {Val::Const, insn.ops[1].value};
      return;
    }
    if ((mn == "mov" || mn == "movs") && insn.ops[1].type == o_reg) {
      auto it = regs.find(opnd_reg(ea, 1));
      regs[dst] = it == regs.end() ? Val{} : it->second;
      return;
    }
    if ((mn == "xor" || mn == "eor") && insn.ops[1].type == o_reg && opnd_reg(ea, 1) == dst) {
      regs[dst] = {Val::Const, 0};
      return;
    }
    regs.erase(dst);
  });
  return out;
}

void from_resolver(Ctx &c) {
  Abi abi = current_abi();
  if (!abi.ok) {
    c.st.runtime.notes.push_back("Op resolver decoding is not implemented for this processor.");
    return;
  }
  // Candidate (caller -> callees) where the caller calls a callee 20+ times. AddBuiltin usually
  // has two overloads (one version, or a version range), so a caller can have several.
  std::map<ea_t, std::set<ea_t>> cands;
  for (ea_t f = get_next_func_ea(0); f != BADADDR; f = get_next_func_ea(f)) {
    std::map<ea_t, int> per_caller;
    xrefblk_t xb;
    for (bool ok = xb.first_to(f, XREF_CODE); ok; ok = xb.next_to()) {
      if (xb.type != fl_CN && xb.type != fl_CF) continue;
      ea_t caller = get_func_start(xb.from);
      if (caller != BADADDR) per_caller[caller]++;
    }
    for (auto &[caller, n] : per_caller)
      if (n >= 20) cands[caller].insert(f);
  }
  for (auto &[caller, callees] : cands) {
    int total = 0;
    auto dec = decode_calls(c, abi, caller, callees, &total);
    // Accept only callers that look like a resolver: most calls decode to (valid op, registration).
    if (dec.size() < 10 || dec.size() * 2 < size_t(total)) {
      if (getenv("TENSORLENS_DEBUG") && total >= 20)
        msg("TensorLens: debug: %a makes %d calls to %zu hot callee(s), %zu decoded\n", caller, total, callees.size(),
            dec.size());
      continue;
    }
    std::set<ea_t> used_callees;
    for (const Decoded &d : dec) {
      Kernel *k = add_table(c, d.table);
      if (k == nullptr) continue;
      used_callees.insert(call_target(d.site));
      if (d.code < 0) {
        k->op = d.custom;
        k->custom = true;
        k->builtin_code = -1;
        k->evidence = "resolver AddCustom(\"" + d.custom + "\") at " + hexea(d.site);
        continue;
      }
      std::string name = tflite_op_name(d.code);
      // Self-check: when symbols named this registration too, count whether the two agree.
      if (k->evidence.rfind("symbol ", 0) == 0) {
        if (k->op == name) {
          c.st.runtime.resolver_agree++;
        } else {
          c.st.runtime.resolver_disagree++;
          msg("TensorLens: resolver says %s, symbol says %s for %a\n", name.c_str(), k->op.c_str(), k->table);
        }
      }
      k->op = name;
      k->builtin_code = d.code;
      k->custom = false;
      k->evidence = "resolver AddBuiltin(" + name + "=" + std::to_string(d.code) + ") at " + hexea(d.site);
    }
    c.st.runtime.tflite = true;
    if (c.st.runtime.resolver == BADADDR) {
      c.st.runtime.resolver = caller;
      c.st.runtime.add_builtin = *used_callees.begin();
    }
    c.st.runtime.resolver_calls += total;
    c.st.runtime.resolver_decoded += int(dec.size());
    msg("TensorLens: op resolver at %a: %d calls to AddBuiltin/AddCustom, %zu decoded\n", caller, total, dec.size());
  }
}

// Kernel source file of a registration, from the strings its functions reference.
std::string file_of(const Ctx &c, const Kernel &k) {
  std::map<std::string, int> votes;
  for (ea_t f : {k.init, k.free, k.prepare, k.invoke}) {
    if (f == BADADDR) continue;
    auto it = c.st.runtime.func_files.find(f);
    if (it != c.st.runtime.func_files.end())
      for (const std::string &s : it->second) votes[s]++;
  }
  std::string best;
  int bv = 0;
  for (auto &[f, v] : votes)
    if (v > bv) { best = f; bv = v; }
  return best;
}

std::string sanitize(std::string s) {
  for (char &ch : s)
    if (!std::isalnum((unsigned char)ch) && ch != '_') ch = '_';
  return s;
}

std::string kernel_label(const Kernel &k) {
  if (!k.op.empty()) return k.op;
  if (!k.source_file.empty()) return k.source_file.substr(0, k.source_file.find('.')) + " (op unknown)";
  return "(op unknown)";
}

// Functions can be shared between registrations (one Prepare for RELU, RELU6 and TANH; one Eval
// for the reference and optimised variants). Such functions get a name from the shared source
// file and a comment listing every operator that uses them.
using Users = std::map<ea_t, std::vector<const Kernel *>>;

Users kernel_users(const State &st) {
  Users users;
  for (const Kernel &k : st.kernels)
    for (ea_t f : {k.init, k.free, k.prepare, k.invoke})
      if (f != BADADDR) users[f].push_back(&k);
  return users;
}

void name_function(State &st, ea_t ea, const char *role, const Kernel &k, const Users &users, std::set<ea_t> &done) {
  if (ea == BADADDR || !done.insert(ea).second) return;
  if (!is_code(get_flags(ea))) create_insn(ea);
  if (get_func_start(ea) != ea) add_func(ea);
  std::set<std::string> ops;
  std::set<std::string> files;
  auto it = users.find(ea);
  if (it != users.end())
    for (const Kernel *u : it->second) {
      ops.insert(kernel_label(*u));
      if (!u->source_file.empty()) files.insert(u->source_file);
    }
  std::string name;
  if (ops.size() > 1) {
    std::string stem = files.size() == 1 ? files.begin()->substr(0, files.begin()->find('.')) : *ops.begin();
    name = "tflite_" + sanitize(stem) + "_" + role + "_shared";
  } else if (!k.op.empty()) {
    name = std::string(k.custom ? "tflite_custom_" : "tflite_") + sanitize(k.op) + "_" + role;
    if (k.variant) name += "_v" + std::to_string(k.variant + 1);
  } else if (!k.source_file.empty()) {
    name = "tflite_" + sanitize(k.source_file.substr(0, k.source_file.find('.'))) + "_" + role;
  } else {
    name = std::string("tflite_kernel_") + role;
  }
  if (rename_if_dummy(ea, name)) st.runtime.named_functions++;

  std::string cmt = std::string("TensorLens: TFLite kernel ") + role + " for ";
  size_t i = 0;
  for (const std::string &o : ops) cmt += (i++ ? ", " : "") + o;
  if (!k.source_file.empty()) cmt += "\nsource: tensorflow/lite/kernels/" + k.source_file;
  if (!k.evidence.empty()) cmt += "\nidentified by: " + k.evidence;
  size_t nuses = 0;
  std::string where;
  for (const std::string &o : ops) {
    auto uses = st.nodes_using(o);
    nuses += uses.size();
    for (size_t j = 0; j < uses.size() && where.size() < 80; j++)
      where += " #" + std::to_string(uses[j].first) + "/node" + std::to_string(uses[j].second);
  }
  if (nuses) cmt += "\nruns " + std::to_string(nuses) + " node(s) of the models in this binary:" + where;
  qstring old;
  if (get_func_cmt_ea(&old, ea, false) <= 0 || strncmp(old.c_str(), "TensorLens", 10) == 0)
    set_func_cmt_ea(ea, cmt.c_str(), false);
}

void name_and_type(State &st, Kernel &k, const Users &users, std::set<ea_t> &done) {
  name_function(st, k.init, "Init", k, users, done);
  name_function(st, k.free, "Free", k, users, done);
  name_function(st, k.prepare, "Prepare", k, users, done);
  name_function(st, k.invoke, "Eval", k, users, done);
  std::string base = k.op.empty() ? "tflite_" + sanitize(k.source_file.substr(0, k.source_file.find('.'))) + "_kernel"
                                  : std::string(k.custom ? "tflite_custom_" : "tflite_") + sanitize(k.op);
  if (k.variant) base += "_v" + std::to_string(k.variant + 1);
  rename_if_dummy(k.table, base + "_registration");
  if (k.register_fn != BADADDR) rename_if_dummy(k.register_fn, "tflite_Register_" + sanitize(k.op.empty() ? base : k.op));
  if (apply_kernel_types(k)) st.runtime.typed_functions++;
  apply_registration_type(k.table);
}

}  // namespace

void recover_runtime(State &st) {
  show_wait_box("TensorLens: recovering the ML runtime...");
  st.kernels.clear();
  st.runtime = RuntimeInfo();
  Ctx c(st);
  replace_wait_box("TensorLens: reading strings");
  from_strings(c);
  replace_wait_box("TensorLens: reading symbols");
  from_symbols(c);
  replace_wait_box("TensorLens: decoding the op resolver");
  from_resolver(c);

  for (auto &[t, k] : c.tables) {
    k.source_file = file_of(c, k);
    if (k.op.empty() && !k.source_file.empty()) {
      const char *ops = tflite_kernel_file_ops(k.source_file);
      if (ops && !strchr(ops, ' ')) {  // the file implements exactly one op
        k.op = ops;
        base_op(k.op, &k.builtin_code);
        k.evidence = "source file " + k.source_file + " implements only " + k.op;
      }
    }
    if (k.evidence.empty() && !k.source_file.empty()) k.evidence = "source file " + k.source_file;
    // A registration with no op and no file is not distinguishable from other function tables.
    if (k.op.empty() && k.source_file.empty()) continue;
    st.kernels.push_back(k);
  }
  // Number variants of the same op (reference vs optimised kernels).
  std::map<std::string, int> seen;
  std::stable_sort(st.kernels.begin(), st.kernels.end(), [](const Kernel &a, const Kernel &b) {
    if (a.op != b.op) return a.op < b.op;
    return a.table < b.table;
  });
  for (Kernel &k : st.kernels)
    if (!k.op.empty()) k.variant = seen[k.op]++;

  if (!st.kernels.empty() || st.runtime.tflite) ensure_tflite_types();
  replace_wait_box("TensorLens: naming %zu kernels", st.kernels.size());
  Users users = kernel_users(st);
  std::set<ea_t> done;
  for (Kernel &k : st.kernels) name_and_type(st, k, users, done);
  hide_wait_box();
  st.recovered = true;
  int ops = 0;
  for (auto &[op, n] : seen) ops++;
  msg("TensorLens: %zu TFLite registrations (%d distinct ops), %d functions named, %d typed; delegates:", st.kernels.size(),
      ops, st.runtime.named_functions, st.runtime.typed_functions);
  for (const std::string &d : st.runtime.delegates) msg(" %s", d.c_str());
  msg("%s\n", st.runtime.delegates.empty() ? " none" : "");
  if (st.runtime.resolver_agree + st.runtime.resolver_disagree)
    msg("TensorLens: resolver decoding vs symbols: %d agree, %d disagree\n", st.runtime.resolver_agree,
        st.runtime.resolver_disagree);
  if (st.runtime.onnxruntime) {
    size_t n = 0;
    for (auto &[f, files] : st.runtime.func_files)
      if (files.begin()->rfind("onnxruntime/", 0) == 0) n++;
    msg("TensorLens: ONNX Runtime present; %zu functions reference ORT kernel sources\n", n);
  }
}

void annotate_kernels(State &st) {
  Users users = kernel_users(st);
  std::set<ea_t> done;
  for (Kernel &k : st.kernels) {
    name_function(st, k.init, "Init", k, users, done);
    name_function(st, k.free, "Free", k, users, done);
    name_function(st, k.prepare, "Prepare", k, users, done);
    name_function(st, k.invoke, "Eval", k, users, done);
  }
}

}  // namespace tl::ida
