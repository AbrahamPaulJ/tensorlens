// TFLite C API types (tensorflow/lite/core/c/common.h, Apache-2.0), declared into the database's
// local types and applied to recovered kernels so pseudocode reads
//     context->tensors[node->inputs->data[0]].dims->data[1]
// instead of pointer arithmetic on unknown structs.
#include "state.hpp"

#include <bytes.hpp>
#include <kernwin.hpp>
#include <nalt.hpp>
#include <typeinf.hpp>

namespace tl::ida {
namespace {

const char kDecls[] = R"(
typedef struct TfLiteContext TfLiteContext;
typedef struct TfLiteNode TfLiteNode;
typedef struct TfLiteTensor TfLiteTensor;
typedef struct TfLiteDelegate TfLiteDelegate;
typedef struct TfLiteRegistration TfLiteRegistration;

typedef enum TfLiteStatus {
  kTfLiteOk = 0, kTfLiteError = 1, kTfLiteDelegateError = 2, kTfLiteApplicationError = 3,
  kTfLiteDelegateDataNotFound = 4, kTfLiteDelegateDataWriteError = 5, kTfLiteDelegateDataReadError = 6,
  kTfLiteUnresolvedOps = 7, kTfLiteCancelled = 8, kTfLiteOutputShapeNotKnown = 9
} TfLiteStatus;

typedef enum TfLiteType {
  kTfLiteNoType = 0, kTfLiteFloat32 = 1, kTfLiteInt32 = 2, kTfLiteUInt8 = 3, kTfLiteInt64 = 4,
  kTfLiteString = 5, kTfLiteBool = 6, kTfLiteInt16 = 7, kTfLiteComplex64 = 8, kTfLiteInt8 = 9,
  kTfLiteFloat16 = 10, kTfLiteFloat64 = 11, kTfLiteComplex128 = 12, kTfLiteUInt64 = 13,
  kTfLiteResource = 14, kTfLiteVariant = 15, kTfLiteUInt32 = 16, kTfLiteUInt16 = 17, kTfLiteInt4 = 18,
  kTfLiteBFloat16 = 19
} TfLiteType;

typedef enum TfLiteAllocationType {
  kTfLiteMemNone = 0, kTfLiteMmapRo, kTfLiteArenaRw, kTfLiteArenaRwPersistent, kTfLiteDynamic,
  kTfLitePersistentRo, kTfLiteCustom, kTfLiteVariantObject, kTfLiteNonCpu
} TfLiteAllocationType;

typedef enum TfLiteQuantizationType {
  kTfLiteNoQuantization = 0, kTfLiteAffineQuantization = 1
} TfLiteQuantizationType;

typedef struct TfLiteIntArray { int size; int data[]; } TfLiteIntArray;
typedef struct TfLiteFloatArray { int size; float data[]; } TfLiteFloatArray;

typedef struct TfLiteQuantizationParams { float scale; int zero_point; } TfLiteQuantizationParams;
typedef struct TfLiteAffineQuantization {
  TfLiteFloatArray *scale; TfLiteIntArray *zero_point; int quantized_dimension;
} TfLiteAffineQuantization;
typedef struct TfLiteQuantization { TfLiteQuantizationType type; void *params; } TfLiteQuantization;

typedef union TfLitePtrUnion {
  int *i32; unsigned int *u32; __int64 *i64; unsigned __int64 *u64; float *f; unsigned short *f16;
  double *f64; char *raw; const char *raw_const; unsigned char *uint8; bool *b; short *i16;
  unsigned short *ui16; signed char *int8; void *data;
} TfLitePtrUnion;

struct TfLiteTensor {
  TfLiteType type;
  TfLitePtrUnion data;
  TfLiteIntArray *dims;
  TfLiteQuantizationParams params;
  TfLiteAllocationType allocation_type;
  size_t bytes;
  const void *allocation;
  const char *name;
  TfLiteDelegate *delegate;
  int buffer_handle;
  bool data_is_stale;
  bool is_variable;
  TfLiteQuantization quantization;
  void *sparsity;
  const TfLiteIntArray *dims_signature;
};

struct TfLiteNode {
  TfLiteIntArray *inputs;
  TfLiteIntArray *outputs;
  TfLiteIntArray *intermediates;
  TfLiteIntArray *temporaries;
  void *user_data;
  void *builtin_data;
  const void *custom_initial_data;
  int custom_initial_data_size;
  TfLiteDelegate *delegate;
  bool might_have_side_effect;
};

struct TfLiteContext {
  size_t tensors_size;
  TfLiteStatus (*GetExecutionPlan)(TfLiteContext *context, TfLiteIntArray **execution_plan);
  TfLiteTensor *tensors;
  void *impl_;
  TfLiteStatus (*ResizeTensor)(TfLiteContext *context, TfLiteTensor *tensor, TfLiteIntArray *new_size);
  void (*ReportError)(TfLiteContext *context, const char *msg, ...);
  TfLiteStatus (*AddTensors)(TfLiteContext *context, int tensors_to_add, int *first_new_tensor_index);
  TfLiteStatus (*GetNodeAndRegistration)(TfLiteContext *context, int node_index, TfLiteNode **node,
                                         TfLiteRegistration **registration);
  void *ReplaceNodeSubsetsWithDelegateKernels;
  int recommended_num_threads;
  void *GetExternalContext;
  void *SetExternalContext;
  bool allow_fp32_relax_to_fp16;
  void *profiler;
  void *(*AllocatePersistentBuffer)(TfLiteContext *ctx, size_t bytes);
  TfLiteStatus (*AllocateBufferForEval)(TfLiteContext *ctx, size_t bytes, void **ptr);
  TfLiteStatus (*RequestScratchBufferInArena)(TfLiteContext *ctx, size_t bytes, int *buffer_idx);
  void *(*GetScratchBuffer)(TfLiteContext *ctx, int buffer_idx);
  TfLiteStatus (*ResizeTensorExplicit)(TfLiteContext *ctx, TfLiteTensor *tensor, int dims, const int *shape);
  void *PreviewDelegatePartitioning;
  TfLiteTensor *(*GetTensor)(const TfLiteContext *context, int tensor_idx);
  void *(*GetEvalTensor)(const TfLiteContext *context, int tensor_idx);
  TfLiteStatus (*GetModelMetadata)(const TfLiteContext *context, const char *name, const char **ptr, size_t *bytes);
  TfLiteStatus (*AcquireSubgraphContext)(TfLiteContext *context, int subgraph_index, TfLiteContext **acquired_context);
  TfLiteStatus (*ReleaseSubgraphContext)(TfLiteContext *context, int subgraph_index);
};

struct TfLiteRegistration {
  void *(*init)(TfLiteContext *context, const char *buffer, size_t length);
  void (*free)(TfLiteContext *context, void *buffer);
  TfLiteStatus (*prepare)(TfLiteContext *context, TfLiteNode *node);
  TfLiteStatus (*invoke)(TfLiteContext *context, TfLiteNode *node);
  const char *(*profiling_string)(const TfLiteContext *context, const TfLiteNode *node);
  int builtin_code;
  const char *custom_name;
  int version;
  void *registration_external;
  void *async_kernel;
  unsigned __int64 inplace_operator;
};
)";

bool g_declared = false;

bool apply_decl_if_untyped(ea_t ea, const char *decl) {
  if (ea == BADADDR) return false;
  tinfo_t cur;
  if (get_tinfo(&cur, ea)) return false;  // keep types from symbols, the user or a previous run
  return apply_cdecl(nullptr, ea, decl, TINFO_DEFINITE);
}

}  // namespace

bool ensure_tflite_types() {
  if (g_declared) {
    tinfo_t t;
    if (t.get_named_type(nullptr, "TfLiteRegistration")) return true;
  }
  // size_t is pointer-sized; most loaders already define it, HTI_DCL tolerates the repeat.
  qstring decls;
  decls.sprnt("typedef unsigned %s size_t;\n", inf_is_64bit() ? "__int64" : "int");
  decls.append(kDecls);
  int errors = parse_decls(nullptr, decls.c_str(), msg, HTI_DCL | HTI_NWR);
  g_declared = true;
  if (errors) msg("TensorLens: %d error(s) while declaring TFLite types\n", errors);
  return errors == 0;
}

bool apply_kernel_types(const Kernel &k) {
  bool any = false;
  any |= apply_decl_if_untyped(k.init, "void *tl_init(TfLiteContext *context, const char *buffer, size_t length);");
  any |= apply_decl_if_untyped(k.free, "void tl_free(TfLiteContext *context, void *buffer);");
  any |= apply_decl_if_untyped(k.prepare, "TfLiteStatus tl_prepare(TfLiteContext *context, TfLiteNode *node);");
  any |= apply_decl_if_untyped(k.invoke, "TfLiteStatus tl_invoke(TfLiteContext *context, TfLiteNode *node);");
  return any;
}

bool apply_registration_type(ea_t table) {
  if (table == BADADDR) return false;
  tinfo_t cur;
  if (get_tinfo(&cur, table)) return false;
  tinfo_t t;
  if (!t.get_named_type(nullptr, "TfLiteRegistration")) return false;
  return apply_tinfo(table, t, TINFO_DEFINITE);
}

}  // namespace tl::ida
