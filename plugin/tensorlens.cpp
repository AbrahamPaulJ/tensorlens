// TensorLens: find, read and connect the ML models and ML runtimes inside a binary.
//
// Menu: Edit > Plugins > TensorLens (Ctrl+Alt+M) runs everything; Search and View menu entries
// open the individual pieces. Headless use: ida -A -Otensorlens:<report.json> <file> writes a
// JSON and an HTML report after auto-analysis and exits.
#include <algorithm>
#include <cstdio>
#include <string>

#include "state.hpp"

#include <auto.hpp>
#include <bytes.hpp>
#include <diskio.hpp>
#include <funcs.hpp>
#include <kernwin.hpp>
#include <loader.hpp>
#include <name.hpp>

namespace tl::ida {

std::vector<const Kernel *> State::kernels_for(const std::string &op) const {
  std::vector<const Kernel *> out;
  if (op.empty()) return out;
  for (const Kernel &k : kernels)
    if (k.op == op) out.push_back(&k);
  std::stable_sort(out.begin(), out.end(), [](const Kernel *a, const Kernel *b) { return a->variant < b->variant; });
  return out;
}

std::vector<std::pair<int, int>> State::nodes_using(const std::string &op) const {
  std::vector<std::pair<int, int>> out;
  if (op.empty()) return out;
  for (size_t m = 0; m < models.size(); m++)
    for (const Graph &g : models[m].hit.model.graphs)
      for (size_t n = 0; n < g.nodes.size(); n++)
        if (g.nodes[n].op == op) out.push_back({int(m), int(n)});
  return out;
}

static std::string ea_name(ea_t ea) {
  if (ea == BADADDR) return "";
  qstring n;
  if (get_ea_name(&n, ea, GN_SHORT | GN_DEMANGLED) > 0) return std::string(n.c_str()) + " @" + hexea(ea);
  return hexea(ea);
}

Report State::make_report() const {
  Report r;
  r.title = "TensorLens report";
  char path[QMAXPATH];
  r.subject = get_input_file_path(path, sizeof path) > 0 ? path : "database";
  for (const FoundModel &m : models) {
    ReportModel rm;
    rm.where = m.segment;
    rm.offset = m.file_only ? m.file_offset : m.ea;
    rm.offset_is_ea = !m.file_only;
    rm.container = m.hit.container;
    rm.inner_offset = m.hit.inner_offset;
    rm.model = m.hit.model;
    r.models.push_back(std::move(rm));
  }
  // Where models are loaded from.
  ReportSection loads{"Code referencing the models (load sites)", {"Model", "Function"}, {}};
  for (size_t i = 0; i < models.size(); i++)
    for (ea_t f : models[i].referenced_by) loads.rows.push_back({"#" + std::to_string(i), ea_name(f)});
  if (!loads.rows.empty()) r.sections.push_back(loads);

  if (!kernels.empty()) {
    ReportSection ks{"TFLite kernels recovered",
                     {"Operator", "Eval", "Prepare", "Init", "Registration", "Source file", "Used by", "Identified by"},
                     {}};
    for (const Kernel &k : kernels) {
      std::string op = k.op.empty() ? "?" : k.op;
      if (k.custom) op += " (custom)";
      if (k.variant) op += " (variant " + std::to_string(k.variant + 1) + ")";
      ks.rows.push_back({op, ea_name(k.invoke), ea_name(k.prepare), ea_name(k.init), hexea(k.table), k.source_file,
                         std::to_string(nodes_using(k.op).size()) + " node(s)", k.evidence});
    }
    r.sections.push_back(ks);
  }
  // Ops used by the models that have no kernel in this binary: they run elsewhere
  // (a delegate, another library) or are custom ops registered at run time.
  if (!models.empty() && !kernels.empty()) {
    ReportSection missing{"Operators used by the models with no kernel in this binary", {"Operator", "Nodes"}, {}};
    std::map<std::string, int> miss;
    for (const FoundModel &m : models)
      if (m.hit.model.format == Format::TFLite)
        for (const Graph &g : m.hit.model.graphs)
          for (const Node &n : g.nodes)
            if (kernels_for(n.op).empty()) miss[n.op]++;
    for (auto &[op, c] : miss) missing.rows.push_back({op, std::to_string(c)});
    if (!missing.rows.empty()) r.sections.push_back(missing);
  }
  if (runtime.tflite) r.notes.push_back("TensorFlow Lite runtime present.");
  if (runtime.onnxruntime) r.notes.push_back("ONNX Runtime present.");
  if (runtime.ncnn) r.notes.push_back("ncnn present.");
  if (runtime.resolver != BADADDR)
    r.notes.push_back("Op resolver " + ea_name(runtime.resolver) + " registers kernels through " +
                      ea_name(runtime.add_builtin) + ".");
  for (const std::string &d : runtime.delegates) r.notes.push_back("Delegate / execution provider: " + d);
  if (!kernels.empty())
    r.notes.push_back(std::to_string(runtime.named_functions) + " functions named, " +
                      std::to_string(runtime.typed_functions) + " kernels given TFLite types.");
  if (runtime.resolver_agree + runtime.resolver_disagree)
    r.notes.push_back("Op resolver decoding checked against symbols: " + std::to_string(runtime.resolver_agree) +
                      " agree, " + std::to_string(runtime.resolver_disagree) + " disagree.");
  return r;
}

namespace {

bool write_text(const std::string &path, const std::string &text) {
  FILE *f = qfopen(path.c_str(), "wb");
  if (f == nullptr) return false;
  bool ok = qfwrite(f, text.data(), text.size()) == ssize_t(text.size());
  qfclose(f);
  return ok;
}

struct plugin_ctx_t;

struct ah_t : public action_handler_t {
  plugin_ctx_t &plg;
  int what;
  ah_t(plugin_ctx_t &p, int w) : plg(p), what(w) {}
  int idaapi activate(action_activation_ctx_t *ctx) override;
  action_state_t idaapi update(action_update_ctx_t *ctx) override;
};

enum { A_RUN, A_MODELS, A_KERNELS, A_REPORT, A_EXPORT, A_GRAPH, A_ADD, A_COUNT };

struct plugin_ctx_t : public plugmod_t, public event_listener_t {
  State st;
  std::string auto_report;  // headless mode: write the report here and exit
  ah_t handlers[A_COUNT] = {{*this, A_RUN},    {*this, A_MODELS}, {*this, A_KERNELS},
                            {*this, A_REPORT}, {*this, A_EXPORT}, {*this, A_GRAPH}, {*this, A_ADD}};

  plugin_ctx_t() {
    static const struct { const char *name, *label, *hotkey, *menu; } acts[A_COUNT] = {
        {"tensorlens:run", "TensorLens: analyse ML models and runtime", "Ctrl+Alt+M", "Edit/Plugins/"},
        {"tensorlens:models", "TensorLens: models", nullptr, "View/Open subviews/"},
        {"tensorlens:kernels", "TensorLens: TFLite kernels", nullptr, "View/Open subviews/"},
        {"tensorlens:report", "TensorLens: export report (HTML + JSON)...", nullptr, "File/Produce file/"},
        {"tensorlens:export", "Export model file...", nullptr, nullptr},
        {"tensorlens:graph", "Show model graph", nullptr, nullptr},
        {"tensorlens:add", "TensorLens: add models from a file, APK or AAR...", nullptr, "File/Load file/"},
    };
    for (int i = 0; i < A_COUNT; i++) {
      action_desc_t d = ACTION_DESC_LITERAL_PLUGMOD(acts[i].name, acts[i].label, &handlers[i], this, acts[i].hotkey,
                                                    nullptr, -1);
      register_action(d);
      if (acts[i].menu != nullptr) attach_action_to_menu(acts[i].menu, acts[i].name, SETMENU_APP);
    }
    hook_event_listener(HT_UI, this);
    const char *opt = get_plugin_options("tensorlens");
    if (opt != nullptr && *opt) auto_report = opt;
  }
  ~plugin_ctx_t() override {
    for (const char *n : {"tensorlens:run", "tensorlens:models", "tensorlens:kernels", "tensorlens:report",
                          "tensorlens:export", "tensorlens:graph", "tensorlens:add"})
      unregister_action(n);
  }

  void analyse(bool show) {
    if (!st.scanned) {
      scan_database(st, true);
      annotate_models(st);
    }
    if (!st.recovered) recover_runtime(st);
    if (!show) return;
    if (!st.kernels.empty()) show_kernels(st);
    show_models(st);
    if (st.models.empty() && st.kernels.empty())
      info("TensorLens found no ML models and no TFLite kernels in this database.");
  }

  bool idaapi run(size_t) override {
    st.scanned = st.recovered = false;  // re-run on explicit request
    analyse(true);
    return true;
  }

  void export_report() {
    char *p = ask_file(true, "*.html", "Save TensorLens report (a .json is written next to it)");
    if (p == nullptr) return;
    std::string html = p;
    std::string json = html.substr(0, html.rfind('.')) + ".json";
    Report r = st.make_report();
    if (write_text(html, to_html(r)) && write_text(json, to_json(r, true)))
      msg("TensorLens: report written to %s and %s\n", html.c_str(), json.c_str());
    else
      warning("TensorLens: could not write the report");
  }

  ssize_t idaapi on_event(ssize_t code, va_list va) override {
    if (code == ui_finish_populating_widget_popup) {
      TWidget *w = va_arg(va, TWidget *);
      TPopupMenu *popup = va_arg(va, TPopupMenu *);
      qstring title;
      if (get_widget_title(&title, w) && title == "TensorLens: models") {
        attach_action_to_popup(w, popup, "tensorlens:graph");
        attach_action_to_popup(w, popup, "tensorlens:export");
      }
    } else if (code == ui_ready_to_run && !auto_report.empty()) {
      run_headless();
    }
    return 0;
  }

  // Models shipped beside the library (the APK's assets) are where its kernels get used.
  void add_models(const std::string &path) {
    if (add_external_models(st, path) < 0) {
      warning("TensorLens: cannot read %s", path.c_str());
      return;
    }
    if (st.recovered) annotate_kernels(st);
  }

  void run_headless() {
    auto_wait();
    analyse(false);
    if (const char *extra = getenv("TENSORLENS_MODELS")) {  // ';'-separated files to add
      std::string all = extra;
      for (size_t a = 0; a < all.size();) {
        size_t b = all.find(';', a);
        if (b == std::string::npos) b = all.size();
        if (b > a) add_models(all.substr(a, b - a));
        a = b + 1;
      }
    }
    Report r = st.make_report();
    std::string base = auto_report.substr(0, auto_report.rfind('.'));
    bool ok = write_text(auto_report, to_json(r, false)) && write_text(base + ".html", to_html(r));
    msg("TensorLens: headless report %s: %s\n", ok ? "written" : "FAILED", auto_report.c_str());
    if (const char *shot = getenv("TENSORLENS_SHOW")) {  // demo / screenshot mode: keep IDA open
      (void)shot;
      if (!st.kernels.empty()) show_kernels(st);
      show_models(st);
      if (!st.models.empty()) show_graph(st, 0);
      return;
    }
    qexit(ok ? 0 : 1);
  }
};

int idaapi ah_t::activate(action_activation_ctx_t *ctx) {
  State &st = plg.st;
  switch (what) {
    case A_RUN: plg.run(0); break;
    case A_MODELS: plg.analyse(false); show_models(st); break;
    case A_KERNELS: plg.analyse(false); show_kernels(st); break;
    case A_REPORT: plg.analyse(false); plg.export_report(); break;
    case A_ADD: {
      plg.analyse(false);
      char *p = ask_file(false, "*.apk;*.aab;*.aar;*.zip;*.tflite;*.onnx;*.param;*.pte;*.task;*.*",
                         "Model file, or an APK/AAR/zip holding models");
      if (p == nullptr) break;
      plg.add_models(p);
      show_models(st);
      break;
    }
    case A_EXPORT:
    case A_GRAPH: {
      if (ctx->chooser_selection.empty()) break;
      size_t n = ctx->chooser_selection[0];
      if (n >= st.models.size()) break;
      if (what == A_GRAPH) {
        show_graph(st, int(n));
        break;
      }
      const FoundModel &m = st.models[n];
      if (m.bytes.empty()) {
        warning("This model's size is unknown, so it cannot be exported.");
        break;
      }
      static const char *ext[] = {"tflite", "onnx", "param", "pte"};
      qstring filter;
      filter.sprnt("*.%s", ext[int(m.hit.model.format)]);
      char *p = ask_file(true, filter.c_str(), "Export model (open it in Netron or IDA)");
      if (p != nullptr && export_model(m, p)) msg("TensorLens: model #%zu saved to %s\n", n, p);
      break;
    }
  }
  return 1;
}

action_state_t idaapi ah_t::update(action_update_ctx_t *ctx) {
  if (what == A_EXPORT || what == A_GRAPH) return ctx->widget_type == BWN_CHOOSER ? AST_ENABLE_FOR_WIDGET : AST_DISABLE_FOR_WIDGET;
  return AST_ENABLE_ALWAYS;
}

plugmod_t *idaapi init() {
  return new plugin_ctx_t;
}

}  // namespace
}  // namespace tl::ida

plugin_t PLUGIN = {
    IDP_INTERFACE_VERSION,
    PLUGIN_MULTI,
    tl::ida::init,
    nullptr,
    nullptr,
    "TensorLens: find ML models in binaries and recover the TFLite runtime that runs them",
    "TensorLens finds TFLite, ONNX, ncnn and ExecuTorch models embedded in the database (also inside\n"
    "gzip streams), shows their graphs, and names the TFLite kernels in the binary, linking each\n"
    "graph node to the code that runs it.",
    "TensorLens",
    "Ctrl+Alt+M",
};
