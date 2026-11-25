// tensorlens-scan: find ML models in files, APKs, AARs, wheels and other zip containers.
//
//   tensorlens-scan [options] <file>...
//     --json FILE     write a JSON report
//     --html FILE     write an HTML report
//     --extract DIR   save every model found as a standalone file (open it in Netron or IDA)
//     --tensors       include the node list in the JSON report
//     -q              quiet: no per-model lines on stdout
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../core/report.hpp"
#include "../core/container.hpp"
#include "../core/scan.hpp"

namespace {

struct Ctx {
  tl::Report report;
  std::string extract_dir;
  bool quiet = false;
  int extracted = 0;
};

const char *ext_for(tl::Format f) {
  switch (f) {
    case tl::Format::TFLite: return ".tflite";
    case tl::Format::ONNX: return ".onnx";
    case tl::Format::NCNN: return ".param";
    case tl::Format::ExecuTorch: return ".pte";
  }
  return ".bin";
}

void extract(Ctx &c, const std::string &where, const tl::Hit &h, const uint8_t *base) {
  if (c.extract_dir.empty() || !h.model.size) return;
  std::string name = where;
  for (char &ch : name)
    if (!isalnum(static_cast<unsigned char>(ch)) && ch != '.' && ch != '-' && ch != '_') ch = '_';
  if (name.size() > 120) name = name.substr(name.size() - 120);
  char off[32];
  std::snprintf(off, sizeof off, "@%llx", (unsigned long long)h.offset);
  std::string path = c.extract_dir + "/" + name + off + ext_for(h.model.format);
  if (h.compressed) {
    std::snprintf(off, sizeof off, "@%llx+%llx", (unsigned long long)h.offset, (unsigned long long)h.inner_offset);
    path = c.extract_dir + "/" + name + off + ext_for(h.model.format);
  }
  std::ofstream o(path, std::ios::binary);
  o.write(reinterpret_cast<const char *>(h.bytes(base)), static_cast<std::streamsize>(h.model.size));
  if (o) c.extracted++;
}

void scan_buffer(Ctx &c, const std::string &where, const std::vector<uint8_t> &buf) {
  tl::scan_tree(
      where, buf,
      [&](const std::string &w, tl::Hit &h, const uint8_t *base) {
        if (!c.quiet) {
          std::printf("%s @0x%llx  %s  (%s)\n", w.c_str(), (unsigned long long)h.offset, h.model.summary().c_str(),
                      h.model.size ? tl::format_bytes(h.model.size).c_str() : "size unknown");
          if (h.compressed)
            std::printf("    inside %s, at +0x%llx\n", h.container.c_str(), (unsigned long long)h.inner_offset);
          for (const std::string &op : h.model.custom_ops()) std::printf("    custom op: %s\n", op.c_str());
        }
        extract(c, w, h, base);
        tl::ReportModel rm;
        rm.where = w;
        rm.offset = h.offset;
        rm.container = h.container;
        rm.inner_offset = h.inner_offset;
        rm.model = std::move(h.model);
        c.report.models.push_back(std::move(rm));
      },
      [&](const std::string &note) { c.report.notes.push_back(note); });
}

bool write_text(const std::string &path, const std::string &text) {
  std::ofstream o(path, std::ios::binary);
  o << text;
  return static_cast<bool>(o);
}

int usage() {
  std::fprintf(stderr,
               "usage: tensorlens-scan [--json FILE] [--html FILE] [--extract DIR] [--tensors] [-q] <file>...\n"
               "Finds TFLite, ONNX, ncnn and ExecuTorch models in binaries and zip containers (APK, AAR, JAR, wheel).\n");
  return 2;
}

}  // namespace

int main(int argc, char **argv) {
  Ctx c;
  std::string json, html;
  bool tensors = false;
  std::vector<std::string> files;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto val = [&](std::string &dst) {
      if (i + 1 >= argc) return false;
      dst = argv[++i];
      return true;
    };
    if (a == "--json") { if (!val(json)) return usage(); }
    else if (a == "--html") { if (!val(html)) return usage(); }
    else if (a == "--extract") { if (!val(c.extract_dir)) return usage(); }
    else if (a == "--tensors") tensors = true;
    else if (a == "-q") c.quiet = true;
    else if (a == "-h" || a == "--help") return usage();
    else files.push_back(a);
  }
  if (files.empty()) return usage();
  c.report.title = "TensorLens scan";
  for (const std::string &f : files) {
    std::vector<uint8_t> buf;
    if (!tl::read_whole_file(f, buf)) {
      std::fprintf(stderr, "cannot read %s\n", f.c_str());
      c.report.notes.push_back(f + ": cannot read");
      continue;
    }
    c.report.subject += (c.report.subject.empty() ? "" : ", ") + f;
    scan_buffer(c, f, buf);
  }
  if (!c.quiet || (json.empty() && html.empty()))
    std::printf("%zu model(s) found%s\n", c.report.models.size(),
                c.extracted ? (", " + std::to_string(c.extracted) + " extracted").c_str() : "");
  if (!json.empty() && !write_text(json, tl::to_json(c.report, tensors))) return 1;
  if (!html.empty() && !write_text(html, tl::to_html(c.report))) return 1;
  return 0;
}
