// Choosers (models, kernels) and the model graph viewer.
#include <algorithm>
#include <cstdio>

#include "state.hpp"

#include <bytes.hpp>
#include <funcs.hpp>
#include <graph.hpp>
#include <kernwin.hpp>
#include <lines.hpp>
#include <moves.hpp>
#include <name.hpp>
#include <netnode.hpp>

namespace tl::ida {
namespace {

std::string io_text(const Graph &g, const std::vector<int> &ids, size_t max = 3) {
  std::string s;
  size_t n = 0;
  for (int id : ids) {
    if (id < 0 || id >= int(g.tensors.size())) continue;
    if (n++ == max) { s += ", ..."; break; }
    const Tensor &t = g.tensors[id];
    if (!s.empty()) s += ", ";
    s += t.dtype + shape_str(t.shape);
  }
  return s;
}

std::string fname(ea_t ea) {
  if (ea == BADADDR) return "";
  qstring n;
  if (get_ea_name(&n, ea, GN_SHORT | GN_DEMANGLED) <= 0) return hexea(ea);
  return n.c_str();
}

// --- models chooser -------------------------------------------------------------------------
struct ModelsChooser : public chooser_t {
  State &st;
  static const int widths_[];
  static const char *const header_[];
  explicit ModelsChooser(State &s)
      : chooser_t(CH_KEEP | CH_CAN_REFRESH, 9, widths_, header_, "TensorLens: models"), st(s) {}
  const void *get_obj_id(size_t *len) const override {
    *len = strlen(title);
    return title;
  }
  size_t idaapi get_count() const override { return st.models.size(); }
  void idaapi get_row(qstrvec_t *cols, int *, chooser_item_attrs_t *attrs, size_t n) const override {
    const FoundModel &m = st.models[n];
    const Model &mod = m.hit.model;
    (*cols)[0] = m.external ? "file" : m.file_only ? ("file+" + hexea(m.file_offset)).c_str() : hexea(m.ea).c_str();
    (*cols)[1] = format_name(mod.format);
    (*cols)[2].sprnt("%zu", mod.node_count());
    std::string custom;
    for (const std::string &c : mod.custom_ops()) custom += (custom.empty() ? "" : ", ") + c;
    (*cols)[3] = custom.c_str();
    std::string where = m.segment;
    if (m.hit.compressed) where += ", " + m.hit.container;
    if (!m.referenced_by.empty()) where += ", used by " + fname(m.referenced_by[0]);
    (*cols)[4] = where.c_str();
    (*cols)[5] = format_bytes(mod.weight_bytes()).c_str();
    if (!mod.graphs.empty()) {
      (*cols)[6] = io_text(mod.graphs[0], mod.graphs[0].inputs).c_str();
      (*cols)[7] = io_text(mod.graphs[0], mod.graphs[0].outputs).c_str();
    }
    auto h = mod.op_histogram();
    std::string top;
    for (size_t i = 0; i < h.size() && i < 3; i++) top += (i ? ", " : "") + h[i].first + " x" + std::to_string(h[i].second);
    (*cols)[8] = top.c_str();
    if (!custom.empty()) attrs->color = 0xC8C8FF;  // custom ops: native code worth a look
  }
  cbret_t idaapi enter(size_t n) override {
    if (n < st.models.size()) show_graph(st, int(n));
    return cbret_t();
  }
};
const int ModelsChooser::widths_[] = {CHCOL_HEX | 10, 6, 5, 24, 44, 8, 18, 18, 36};
const char *const ModelsChooser::header_[] = {"Address", "Format", "Ops", "Custom ops", "Where",
                                              "Weights", "Inputs", "Outputs", "Most used ops"};

// --- kernels chooser ------------------------------------------------------------------------
struct KernelsChooser : public chooser_t {
  State &st;
  static const int widths_[];
  static const char *const header_[];
  explicit KernelsChooser(State &s)
      : chooser_t(CH_KEEP | CH_CAN_REFRESH, 8, widths_, header_, "TensorLens: TFLite kernels"), st(s) {}
  const void *get_obj_id(size_t *len) const override {
    *len = strlen(title);
    return title;
  }
  // Kernels that run the models found come first, most used first; the rest alphabetically.
  mutable std::vector<size_t> order;
  size_t idaapi get_count() const override {
    std::vector<size_t> uses(st.kernels.size());
    order.resize(st.kernels.size());
    for (size_t i = 0; i < order.size(); i++) {
      order[i] = i;
      uses[i] = st.kernels[i].variant ? 0 : st.nodes_using(st.kernels[i].op).size();
    }
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return uses[a] > uses[b]; });
    return order.size();
  }
  void idaapi get_row(qstrvec_t *cols, int *, chooser_item_attrs_t *attrs, size_t n) const override {
    if (n >= order.size()) return;
    const Kernel &k = st.kernels[order[n]];
    std::string op = k.op.empty() ? "?" : k.op;
    if (k.variant) op += " (variant " + std::to_string(k.variant + 1) + ")";
    auto uses = st.nodes_using(k.op);
    (*cols)[0] = op.c_str();
    (*cols)[1].sprnt("%zu", uses.size());
    (*cols)[2] = fname(k.invoke).c_str();
    (*cols)[3] = fname(k.prepare).c_str();
    (*cols)[4] = k.evidence.c_str();
    (*cols)[5] = k.source_file.c_str();
    (*cols)[6] = hexea(k.table).c_str();
    (*cols)[7] = k.custom ? "custom" : (k.builtin_code >= 0 ? std::to_string(k.builtin_code).c_str() : "");
    if (k.custom) attrs->color = 0xC8C8FF;
    else if (!uses.empty() && k.variant == 0) attrs->color = 0xC8FFC8;  // runs in a model found here
  }
  cbret_t idaapi enter(size_t n) override {
    if (n < order.size()) jumpto(st.kernels[order[n]].invoke);
    return cbret_t();
  }
};
const int KernelsChooser::widths_[] = {18, 5, 24, 24, 46, 14, CHCOL_HEX | 10, 5};
const char *const KernelsChooser::header_[] = {"Operator", "Nodes", "Eval", "Prepare", "Identified by",
                                               "Source file", "Registration", "Code"};

// --- graph viewer ---------------------------------------------------------------------------
struct GraphData {
  State *st = nullptr;
  int model = 0;
  int graph = 0;
  std::vector<std::string> text;     // colored node text
  std::vector<bgcolor_t> color;
  std::vector<int> node_op;          // node -> op index, or -1 for an input/output node
  std::vector<int> node_tensor;      // node -> tensor index for input/output nodes
  struct E { int a, b; };
  std::vector<E> edges;
  graph_viewer_t *gv = nullptr;
  std::string title;

  void build();
  std::string hint(int node) const;
  ea_t target(int node) const;
};

void GraphData::build() {
  const FoundModel &fm = st->models[model];
  const Graph &g = fm.hit.model.graphs[graph];
  const size_t kMax = 3000;
  text.clear(); color.clear(); node_op.clear(); node_tensor.clear(); edges.clear();
  std::vector<int> producer(g.tensors.size(), -1), in_node(g.tensors.size(), -1);
  size_t nops = std::min(g.nodes.size(), kMax);
  for (size_t i = 0; i < nops; i++)
    for (int o : g.nodes[i].outputs)
      if (o >= 0) producer[o] = int(i);
  // Op nodes.
  for (size_t i = 0; i < nops; i++) {
    const Node &n = g.nodes[i];
    std::string t = std::string(SCOLOR_ON SCOLOR_MACRO) + n.op + SCOLOR_OFF SCOLOR_MACRO;
    if (!n.name.empty() && n.name.size() < 48) t += std::string("  " SCOLOR_ON SCOLOR_AUTOCMT) + n.name + SCOLOR_OFF SCOLOR_AUTOCMT;
    std::string outs = io_text(g, n.outputs, 2);
    if (!outs.empty()) t += std::string("\n" SCOLOR_ON SCOLOR_DNUM) + outs + SCOLOR_OFF SCOLOR_DNUM;
    uint64_t w = 0;
    for (int in : n.inputs)
      if (in >= 0 && g.tensors[in].constant) w += g.tensors[in].data_size;
    if (w) t += "  weights " + format_bytes(w);
    auto ks = st->kernels_for(n.op);
    bgcolor_t c = DEFCOLOR;
    if (!ks.empty()) {
      t += std::string("\n" SCOLOR_ON SCOLOR_CNAME) + "-> " + fname(ks[0]->invoke) + SCOLOR_OFF SCOLOR_CNAME;
      c = 0xD8F5D8;
    }
    if (n.custom) c = 0xD0D0FF;
    text.push_back(t);
    color.push_back(c);
    node_op.push_back(int(i));
    node_tensor.push_back(-1);
  }
  // Graph inputs and outputs.
  auto io_node = [&](int tid, const char *kind) {
    const Tensor &t = g.tensors[tid];
    std::string s = std::string(SCOLOR_ON SCOLOR_KEYWORD) + kind + SCOLOR_OFF SCOLOR_KEYWORD + " " + t.name + "\n" +
                    SCOLOR_ON SCOLOR_DNUM + t.dtype + shape_str(t.shape) + SCOLOR_OFF SCOLOR_DNUM;
    text.push_back(s);
    color.push_back(0xF0E6D2);
    node_op.push_back(-1);
    node_tensor.push_back(tid);
    return int(text.size() - 1);
  };
  for (int tid : g.inputs)
    if (tid >= 0) in_node[tid] = io_node(tid, "input");
  for (size_t i = 0; i < nops; i++)
    for (int in : g.nodes[i].inputs) {
      if (in < 0) continue;
      if (producer[in] >= 0) edges.push_back({producer[in], int(i)});
      else if (in_node[in] >= 0) edges.push_back({in_node[in], int(i)});
    }
  for (int tid : g.outputs) {
    if (tid < 0) continue;
    int on = io_node(tid, "output");
    if (producer[tid] >= 0) edges.push_back({producer[tid], on});
  }
  if (g.nodes.size() > kMax) msg("TensorLens: graph truncated to the first %zu of %zu ops\n", kMax, g.nodes.size());
}

std::string GraphData::hint(int node) const {
  const FoundModel &fm = st->models[model];
  const Graph &g = fm.hit.model.graphs[graph];
  if (node < 0 || node >= int(node_op.size())) return {};
  auto tensor_line = [&](int id) {
    if (id < 0) return std::string("  (none)\n");
    const Tensor &t = g.tensors[id];
    std::string s = "  " + t.name + "  " + t.dtype + shape_str(t.shape);
    if (t.constant) s += "  const " + format_bytes(t.data_size);
    if (!t.quant.empty()) s += "  " + t.quant;
    return s + "\n";
  };
  if (node_op[node] < 0) return "tensor\n" + tensor_line(node_tensor[node]);
  const Node &n = g.nodes[node_op[node]];
  std::string s = n.op + (n.custom ? "  (custom op)" : "") + "\n";
  if (n.op_version) s += "version " + std::to_string(n.op_version) + "\n";
  s += "inputs:\n";
  for (int i : n.inputs) s += tensor_line(i);
  s += "outputs:\n";
  for (int o : n.outputs) s += tensor_line(o);
  auto ks = st->kernels_for(n.op);
  if (!ks.empty()) {
    s += "kernel:\n";
    if (ks[0]->prepare != BADADDR) s += "  Prepare " + fname(ks[0]->prepare) + "\n";
    s += "  Eval    " + fname(ks[0]->invoke) + "\n(double-click to jump)";
  }
  return s;
}

ea_t GraphData::target(int node) const {
  const FoundModel &fm = st->models[model];
  const Graph &g = fm.hit.model.graphs[graph];
  if (node < 0 || node >= int(node_op.size())) return BADADDR;
  if (node_op[node] >= 0) {
    auto ks = st->kernels_for(g.nodes[node_op[node]].op);
    if (!ks.empty()) return ks[0]->invoke;
    // No kernel known: jump to the op's first constant input in the database.
    for (int in : g.nodes[node_op[node]].inputs)
      if (in >= 0 && g.tensors[in].constant && !fm.hit.compressed && fm.ea != BADADDR)
        return fm.ea + ea_t(g.tensors[in].data_offset);
  }
  return fm.ea;
}

ssize_t idaapi graph_cb(void *ud, int code, va_list va) {
  GraphData *d = static_cast<GraphData *>(ud);
  switch (code) {
    case grcode_user_refresh: {
      interactive_graph_t *g = va_arg(va, interactive_graph_t *);
      g->clear();
      g->resize(int(d->text.size()));
      for (const auto &e : d->edges) g->add_edge(e.a, e.b, nullptr);
      return true;
    }
    case grcode_user_text: {
      va_arg(va, interactive_graph_t *);
      int node = va_arg(va, int);
      const char **text = va_arg(va, const char **);
      bgcolor_t *bg = va_arg(va, bgcolor_t *);
      if (node < 0 || node >= int(d->text.size())) return false;
      *text = d->text[node].c_str();
      if (bg != nullptr) *bg = d->color[node];
      return true;
    }
    case grcode_user_hint: {
      va_arg(va, interactive_graph_t *);
      int node = va_arg(va, int);
      va_arg(va, int);
      va_arg(va, int);
      char **hint = va_arg(va, char **);
      if (node < 0) return false;
      *hint = qstrdup(d->hint(node).c_str());
      return true;
    }
    case grcode_dblclicked: {
      va_arg(va, graph_viewer_t *);
      selection_item_t *s = va_arg(va, selection_item_t *);
      if (s != nullptr && s->is_node) {
        ea_t ea = d->target(s->node);
        if (ea != BADADDR) jumpto(ea);
      }
      return true;  // handled
    }
    default:
      break;
  }
  return 0;
}

std::vector<std::unique_ptr<GraphData>> g_graphs;

}  // namespace

void show_models(State &st) {
  static ModelsChooser *ch = nullptr;
  if (ch == nullptr || &ch->st != &st) ch = new ModelsChooser(st);  // chooser_t owns itself once shown
  ch->choose();
}

void show_kernels(State &st) {
  static KernelsChooser *ch = nullptr;
  if (ch == nullptr || &ch->st != &st) ch = new KernelsChooser(st);
  ch->choose();
}

void show_graph(State &st, int model_index) {
  if (model_index < 0 || model_index >= int(st.models.size())) return;
  const FoundModel &fm = st.models[model_index];
  if (fm.hit.model.graphs.empty()) {
    info("This %s model was recognised but its graph is not decoded.", format_name(fm.hit.model.format));
    return;
  }
  int sg = 0;
  if (fm.hit.model.graphs.size() > 1) {
    sval_t v = 0;
    qstring prompt;
    prompt.sprnt("The model has %zu subgraphs. Which one (0 = main)?", fm.hit.model.graphs.size());
    if (!ask_long(&v, prompt.c_str())) return;
    sg = int(std::clamp<sval_t>(v, 0, sval_t(fm.hit.model.graphs.size() - 1)));
  }
  qstring title;
  title.sprnt("TensorLens graph #%d (%s, subgraph %d)", model_index, format_name(fm.hit.model.format), sg);
  if (TWidget *w = find_widget(title.c_str())) {
    activate_widget(w, true);
    return;
  }
  auto d = std::make_unique<GraphData>();
  d->st = &st;
  d->model = model_index;
  d->graph = sg;
  d->title = title.c_str();
  d->build();
  netnode id;
  qstring nn;
  nn.sprnt("$ tensorlens graph %d.%d", model_index, sg);
  id.create(nn.c_str());
  d->gv = create_graph_viewer(title.c_str(), id, graph_cb, d.get(), 0);
  if (d->gv == nullptr) return;
  display_widget(d->gv, WOPN_DP_TAB);
  if (d->text.size() <= 25) {
    viewer_fit_window(d->gv);
  } else {
    // Big graphs are unreadable when fitted: open at 100% on the first custom op (native code in
    // the model's data path, the node worth a look), or else on the model's input.
    const Graph &g = fm.hit.model.graphs[sg];
    int focus = -1;
    for (size_t i = 0; i < d->node_op.size() && focus < 0; i++)
      if (d->node_op[i] >= 0 && g.nodes[d->node_op[i]].custom) focus = int(i);
    for (size_t i = 0; i < d->node_op.size() && focus < 0; i++)
      if (d->node_op[i] < 0) focus = int(i);
    graph_location_info_t gli;
    gli.zoom = 1.0;
    viewer_set_gli(d->gv, &gli, 0);
    viewer_center_on(d->gv, focus < 0 ? 0 : focus);
  }
  g_graphs.push_back(std::move(d));
}

bool export_model(const FoundModel &m, const char *path) {
  if (m.bytes.empty()) return false;
  FILE *f = qfopen(path, "wb");
  if (f == nullptr) return false;
  bool ok = qfwrite(f, m.bytes.data(), m.bytes.size()) == ssize_t(m.bytes.size());
  qfclose(f);
  return ok;
}

}  // namespace tl::ida
