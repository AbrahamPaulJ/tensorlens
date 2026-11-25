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
#include "../core/scan.hpp"
#include "miniz.h"

namespace {

constexpr uint64_t kMaxEntry = 1ull << 30;  // skip archive entries over 1 GB

struct Ctx {
  tl::Report report;
  std::string extract_dir;
  bool quiet = false;
  int extracted = 0;
};

bool read_file(const std::string &path, std::vector<uint8_t> &out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  f.seekg(0, std::ios::end);
  std::streamoff n = f.tellg();
  if (n < 0) return false;
  f.seekg(0);
  out.resize(static_cast<size_t>(n));
  return n == 0 || static_cast<bool>(f.read(reinterpret_cast<char *>(out.data()), n));
}

bool is_zip(const std::vector<uint8_t> &b) { return b.size() > 4 && b[0] == 'P' && b[1] == 'K' && b[2] == 3 && b[3] == 4; }

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
  std::ofstream o(path, std::ios::binary);
  o.write(reinterpret_cast<const char *>(base + h.offset), static_cast<std::streamsize>(h.model.size));
  if (o) c.extracted++;
}

void scan_buffer(Ctx &c, const std::string &where, const std::vector<uint8_t> &buf, int depth);

void scan_zip(Ctx &c, const std::string &where, const std::vector<uint8_t> &buf, int depth) {
  mz_zip_archive zip;
  std::memset(&zip, 0, sizeof zip);
  if (!mz_zip_reader_init_mem(&zip, buf.data(), buf.size(), 0)) {
    c.report.notes.push_back(where + ": not a readable zip archive");
    return;
  }
  mz_uint n = mz_zip_reader_get_num_files(&zip);
  for (mz_uint i = 0; i < n; i++) {
    mz_zip_archive_file_stat st;
    if (!mz_zip_reader_file_stat(&zip, i, &st) || st.m_is_directory) continue;
    if (st.m_uncomp_size > kMaxEntry) {
      c.report.notes.push_back(where + "!" + st.m_filename + ": skipped (over 1 GB)");
      continue;
    }
    std::vector<uint8_t> data(static_cast<size_t>(st.m_uncomp_size));
    if (!data.empty() && !mz_zip_reader_extract_to_mem(&zip, i, data.data(), data.size(), 0)) {
      c.report.notes.push_back(where + "!" + st.m_filename + ": could not decompress");
      continue;
    }
    scan_buffer(c, where + "!" + st.m_filename, data, depth + 1);
  }
  mz_zip_reader_end(&zip);
}

void scan_buffer(Ctx &c, const std::string &where, const std::vector<uint8_t> &buf, int depth) {
  if (is_zip(buf) && depth < 3) {
    scan_zip(c, where, buf, depth);
    return;
  }
  auto hits = tl::scan(buf.data(), buf.size());
  for (tl::Hit &h : hits) {
    if (!c.quiet) {
      std::printf("%s @0x%llx  %s  (%s)\n", where.c_str(), (unsigned long long)h.offset, h.model.summary().c_str(),
                  h.model.size ? tl::format_bytes(h.model.size).c_str() : "size unknown");
      for (const std::string &op : h.model.custom_ops()) std::printf("    custom op: %s\n", op.c_str());
    }
    extract(c, where, h, buf.data());
    tl::ReportModel rm;
    rm.where = where;
    rm.offset = h.offset;
    rm.model = std::move(h.model);
    c.report.models.push_back(std::move(rm));
  }
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
    if (!read_file(f, buf)) {
      std::fprintf(stderr, "cannot read %s\n", f.c_str());
      c.report.notes.push_back(f + ": cannot read");
      continue;
    }
    c.report.subject += (c.report.subject.empty() ? "" : ", ") + f;
    scan_buffer(c, f, buf, 0);
  }
  if (!c.quiet || (json.empty() && html.empty()))
    std::printf("%zu model(s) found%s\n", c.report.models.size(),
                c.extracted ? (", " + std::to_string(c.extracted) + " extracted").c_str() : "");
  if (!json.empty() && !write_text(json, tl::to_json(c.report, tensors))) return 1;
  if (!html.empty() && !write_text(html, tl::to_html(c.report))) return 1;
  return 0;
}
