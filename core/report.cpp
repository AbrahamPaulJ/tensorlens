#include "report.hpp"

#include <cstdio>

namespace tl {

std::string json_escape(const std::string &s) {
  std::string o;
  o.reserve(s.size() + 2);
  for (unsigned char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      default:
        if (c < 0x20) {
          char b[8];
          std::snprintf(b, sizeof b, "\\u%04x", c);
          o += b;
        } else {
          o += static_cast<char>(c);
        }
    }
  }
  return o;
}

std::string html_escape(const std::string &s) {
  std::string o;
  for (char c : s) {
    switch (c) {
      case '&': o += "&amp;"; break;
      case '<': o += "&lt;"; break;
      case '>': o += "&gt;"; break;
      case '"': o += "&quot;"; break;
      default: o += c;
    }
  }
  return o;
}

namespace {

std::string q(const std::string &s) { return "\"" + json_escape(s) + "\""; }

std::string hex(uint64_t v) {
  char b[24];
  std::snprintf(b, sizeof b, "0x%llx", (unsigned long long)v);
  return b;
}

std::string tensor_json(const Tensor &t) {
  std::string s = "{\"name\":" + q(t.name) + ",\"dtype\":" + q(t.dtype) + ",\"shape\":[";
  for (size_t i = 0; i < t.shape.size(); i++) s += (i ? "," : "") + std::to_string(t.shape[i]);
  s += "]";
  if (t.constant) s += ",\"bytes\":" + std::to_string(t.data_size);
  if (!t.quant.empty()) s += ",\"quant\":" + q(t.quant);
  return s + "}";
}

std::string io_json(const Graph &g, const std::vector<int> &ids) {
  std::string s = "[";
  bool first = true;
  for (int id : ids) {
    if (id < 0 || id >= (int)g.tensors.size()) continue;
    s += (first ? "" : ",") + tensor_json(g.tensors[id]);
    first = false;
  }
  return s + "]";
}

}  // namespace

std::string to_json(const Report &r, bool include_tensors) {
  std::string s = "{\n  \"tool\": \"TensorLens\",\n  \"subject\": " + q(r.subject) + ",\n  \"models\": [";
  for (size_t i = 0; i < r.models.size(); i++) {
    const ReportModel &rm = r.models[i];
    const Model &m = rm.model;
    s += i ? ",\n    {" : "\n    {";
    s += "\"where\":" + q(rm.where) + ",\"" + (rm.offset_is_ea ? "address" : "offset") + "\":" + q(hex(rm.offset));
    if (!rm.container.empty())
      s += ",\"container\":" + q(rm.container) + ",\"inner_offset\":" + q(hex(rm.inner_offset));
    s += ",\"format\":" + q(format_name(m.format)) + ",\"size\":" + std::to_string(m.size);
    s += ",\"version\":" + q(m.version) + ",\"producer\":" + q(m.producer);
    s += ",\"ops\":" + std::to_string(m.node_count()) + ",\"weight_bytes\":" + std::to_string(m.weight_bytes());
    s += ",\"op_histogram\":{";
    auto h = m.op_histogram();
    for (size_t k = 0; k < h.size(); k++) s += (k ? "," : "") + q(h[k].first) + ":" + std::to_string(h[k].second);
    s += "},\"custom_ops\":[";
    auto c = m.custom_ops();
    for (size_t k = 0; k < c.size(); k++) s += (k ? "," : "") + q(c[k]);
    s += "],\"metadata\":{";
    for (size_t k = 0; k < m.metadata.size(); k++)
      s += (k ? "," : "") + q(m.metadata[k].first) + ":" + q(m.metadata[k].second);
    s += "}";
    if (!m.graphs.empty()) {
      const Graph &g = m.graphs[0];
      s += ",\"inputs\":" + io_json(g, g.inputs) + ",\"outputs\":" + io_json(g, g.outputs);
      s += ",\"subgraphs\":" + std::to_string(m.graphs.size());
      if (include_tensors) {
        s += ",\"nodes\":[";
        for (size_t k = 0; k < g.nodes.size(); k++) {
          const Node &n = g.nodes[k];
          s += (k ? "," : "") + std::string("{\"op\":") + q(n.op) + ",\"inputs\":[";
          for (size_t j = 0; j < n.inputs.size(); j++) s += (j ? "," : "") + std::to_string(n.inputs[j]);
          s += "],\"outputs\":[";
          for (size_t j = 0; j < n.outputs.size(); j++) s += (j ? "," : "") + std::to_string(n.outputs[j]);
          s += "]}";
        }
        s += "]";
      }
    }
    s += ",\"warnings\":[";
    for (size_t k = 0; k < m.warnings.size(); k++) s += (k ? "," : "") + q(m.warnings[k]);
    s += "]}";
  }
  s += r.models.empty() ? "],\n" : "\n  ],\n";
  s += "  \"sections\": [";
  for (size_t i = 0; i < r.sections.size(); i++) {
    const ReportSection &sec = r.sections[i];
    s += (i ? ",\n    {" : "\n    {") + std::string("\"title\":") + q(sec.title) + ",\"columns\":[";
    for (size_t k = 0; k < sec.columns.size(); k++) s += (k ? "," : "") + q(sec.columns[k]);
    s += "],\"rows\":[";
    for (size_t k = 0; k < sec.rows.size(); k++) {
      s += k ? ",[" : "[";
      for (size_t j = 0; j < sec.rows[k].size(); j++) s += (j ? "," : "") + q(sec.rows[k][j]);
      s += "]";
    }
    s += "]}";
  }
  s += r.sections.empty() ? "],\n" : "\n  ],\n";
  s += "  \"notes\": [";
  for (size_t i = 0; i < r.notes.size(); i++) s += (i ? "," : "") + q(r.notes[i]);
  s += "]\n}\n";
  return s;
}

std::string to_html(const Report &r) {
  std::string s =
      "<!doctype html><html><head><meta charset=\"utf-8\"><title>" + html_escape(r.title) +
      "</title><style>"
      "body{font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;margin:2rem auto;max-width:1100px;"
      "padding:0 1rem;color:#1b1f24;background:#fff}"
      "@media(prefers-color-scheme:dark){body{color:#e6e6e6;background:#16181c}th{background:#23262c!important}"
      "td,th{border-color:#333!important}.card{border-color:#333!important}code{background:#23262c!important}}"
      "h1{font-size:1.5rem;margin:0 0 .2rem}h2{font-size:1.15rem;margin:2rem 0 .5rem}"
      ".sub{opacity:.7;margin-bottom:1.5rem}.card{border:1px solid #d8dde3;border-radius:8px;padding:1rem 1.2rem;"
      "margin:1rem 0}table{border-collapse:collapse;width:100%;font-size:13px}"
      "td,th{border:1px solid #d8dde3;padding:4px 8px;text-align:left;vertical-align:top}th{background:#f3f5f7}"
      "code{background:#f3f5f7;padding:1px 4px;border-radius:4px}.warn{color:#b35900}.flag{color:#c62828;"
      "font-weight:600}.kv{display:grid;grid-template-columns:max-content 1fr;gap:2px 14px}"
      "</style></head><body>";
  s += "<h1>" + html_escape(r.title) + "</h1><div class=\"sub\">" + html_escape(r.subject) + "</div>";
  s += "<h2>Models (" + std::to_string(r.models.size()) + ")</h2>";
  for (const ReportModel &rm : r.models) {
    const Model &m = rm.model;
    char off[32];
    std::snprintf(off, sizeof off, "0x%llx", (unsigned long long)rm.offset);
    s += "<div class=\"card\"><b>" + html_escape(format_name(m.format)) + "</b> in <code>" +
         html_escape(rm.where) + "</code> at <code>" + off + "</code><div class=\"kv\" style=\"margin-top:.6rem\">";
    auto kv = [&](const std::string &k, const std::string &v) {
      if (!v.empty()) s += "<div>" + html_escape(k) + "</div><div>" + html_escape(v) + "</div>";
    };
    if (!rm.container.empty()) {
      char io[32];
      std::snprintf(io, sizeof io, "0x%llx", (unsigned long long)rm.inner_offset);
      kv("Inside", rm.container + " at inflated offset " + io);
    }
    kv("Size", m.size ? format_bytes(m.size) : "unknown");
    kv("Version", m.version);
    kv("Producer", m.producer);
    kv("Operators", std::to_string(m.node_count()));
    kv("Weights", format_bytes(m.weight_bytes()));
    if (!m.graphs.empty()) {
      const Graph &g = m.graphs[0];
      auto io = [&](const std::vector<int> &ids) {
        std::string t;
        for (int id : ids)
          if (id >= 0 && id < (int)g.tensors.size())
            t += (t.empty() ? "" : "; ") + g.tensors[id].name + " " + g.tensors[id].dtype +
                 shape_str(g.tensors[id].shape);
        return t;
      };
      kv("Inputs", io(g.inputs));
      kv("Outputs", io(g.outputs));
    }
    for (const auto &md : m.metadata) kv("meta: " + md.first, md.second);
    s += "</div>";
    auto custom = m.custom_ops();
    if (!custom.empty()) {
      s += "<p class=\"flag\">Custom operators (native code to review): ";
      for (size_t i = 0; i < custom.size(); i++) s += (i ? ", " : "") + html_escape(custom[i]);
      s += "</p>";
    }
    for (const std::string &w : m.warnings) s += "<p class=\"warn\">" + html_escape(w) + "</p>";
    auto h = m.op_histogram();
    if (!h.empty()) {
      s += "<table><tr><th>Operator</th><th>Count</th></tr>";
      for (const auto &e : h) s += "<tr><td>" + html_escape(e.first) + "</td><td>" + std::to_string(e.second) + "</td></tr>";
      s += "</table>";
    }
    s += "</div>";
  }
  for (const ReportSection &sec : r.sections) {
    s += "<h2>" + html_escape(sec.title) + " (" + std::to_string(sec.rows.size()) + ")</h2><table><tr>";
    for (const std::string &c : sec.columns) s += "<th>" + html_escape(c) + "</th>";
    s += "</tr>";
    for (const auto &row : sec.rows) {
      s += "<tr>";
      for (const std::string &c : row) s += "<td>" + html_escape(c) + "</td>";
      s += "</tr>";
    }
    s += "</table>";
  }
  if (!r.notes.empty()) {
    s += "<h2>Notes</h2><ul>";
    for (const std::string &n : r.notes) s += "<li>" + html_escape(n) + "</li>";
    s += "</ul>";
  }
  s += "<p class=\"sub\" style=\"margin-top:2rem\">Generated by TensorLens.</p></body></html>\n";
  return s;
}

}  // namespace tl
