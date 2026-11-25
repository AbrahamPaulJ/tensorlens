// TensorLens IDA plugin: shared state for one database.
#pragma once

// Standard and core headers first: IDA's pro.h redefines some C library names.
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "report.hpp"
#include "scan.hpp"

#include <ida.hpp>
#include <idp.hpp>

namespace tl::ida {

// A model found in the database (or in the input file outside any segment).
struct FoundModel {
  ea_t ea = BADADDR;              // start of the model (or of the gzip stream holding it)
  uint64_t file_offset = 0;       // offset in the input file when found there
  bool file_only = false;         // bytes exist in the input file but are not loaded
  bool external = false;          // added from another file (the APK or assets beside the library)
  std::string segment;
  Hit hit;                        // model + compressed-stream info
  std::vector<uint8_t> bytes;     // standalone copy of the model's bytes (for export)
  eavec_t referenced_by;          // functions that reference the model's address
  std::string label() const;      // "#0 TFLite (31 ops)"
};

// One TfLiteRegistration recovered from the binary.
struct Kernel {
  std::string op;                 // "CONV_2D", custom op name, or "" when unknown
  int builtin_code = -1;
  bool custom = false;
  ea_t table = BADADDR;           // the TfLiteRegistration struct
  ea_t init = BADADDR, free = BADADDR, prepare = BADADDR, invoke = BADADDR;
  ea_t register_fn = BADADDR;     // Register_XXX() returning &table, if found
  std::string source_file;        // "conv.cc" from __FILE__ strings
  std::string evidence;           // how the op was identified
  int variant = 0;                // 0 = first registration for this op, 1.. = other variants
};

struct RuntimeInfo {
  bool tflite = false;
  bool onnxruntime = false;
  bool ncnn = false;
  std::set<std::string> delegates;      // "gpu", "nnapi", "xnnpack", "hexagon", ...
  std::map<ea_t, std::set<std::string>> func_files;  // function -> kernel source files it references
  std::vector<std::string> notes;
  ea_t resolver = BADADDR;              // BuiltinOpResolver constructor, if found
  ea_t add_builtin = BADADDR;           // the AddBuiltin callee
  int named_functions = 0;
  int typed_functions = 0;
  int resolver_calls = 0, resolver_decoded = 0;
  int resolver_agree = 0, resolver_disagree = 0;  // resolver decoding checked against symbols
};

struct State {
  std::vector<FoundModel> models;
  std::vector<Kernel> kernels;
  RuntimeInfo runtime;
  bool scanned = false;
  bool recovered = false;

  // Kernels implementing an operator name, best (non-variant) first.
  std::vector<const Kernel *> kernels_for(const std::string &op) const;
  // Model nodes (model index, node index) that run an operator.
  std::vector<std::pair<int, int>> nodes_using(const std::string &op) const;
  Report make_report() const;
};

// idb_models.cpp
void scan_database(State &st, bool include_input_file);
void annotate_models(State &st);
int add_external_models(State &st, const std::string &path);  // -1 if unreadable

// runtime.cpp
void recover_runtime(State &st);
void annotate_kernels(State &st);  // refresh kernel comments after models change

// typelib.cpp
bool ensure_tflite_types();
bool apply_kernel_types(const Kernel &k);
bool apply_registration_type(ea_t table);

// views.cpp
void show_models(State &st);
void show_kernels(State &st);
void show_graph(State &st, int model_index);
bool export_model(const FoundModel &m, const char *path);

// Helpers shared by the files above.
std::string hexea(ea_t ea);
bool rename_if_dummy(ea_t ea, const std::string &name);  // never overwrites user or symbol names

}  // namespace tl::ida
