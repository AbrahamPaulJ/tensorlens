#include "container.hpp"

#include <cstring>
#include <fstream>

#include "miniz.h"

namespace tl {
namespace {

constexpr uint64_t kMaxEntry = 1ull << 30;  // skip archive entries over 1 GB

void scan_at(const std::string &where, const std::vector<uint8_t> &buf, const HitFn &on_hit, const NoteFn &on_note,
             int depth, int max_depth) {
  if (is_zip(buf.data(), buf.size()) && depth < max_depth) {
    mz_zip_archive zip;
    std::memset(&zip, 0, sizeof zip);
    if (!mz_zip_reader_init_mem(&zip, buf.data(), buf.size(), 0)) {
      if (on_note) on_note(where + ": not a readable zip archive");
      return;
    }
    mz_uint n = mz_zip_reader_get_num_files(&zip);
    for (mz_uint i = 0; i < n; i++) {
      mz_zip_archive_file_stat st;
      if (!mz_zip_reader_file_stat(&zip, i, &st) || st.m_is_directory) continue;
      std::string name = where + "!" + st.m_filename;
      if (st.m_uncomp_size > kMaxEntry) {
        if (on_note) on_note(name + ": skipped (over 1 GB)");
        continue;
      }
      std::vector<uint8_t> data(static_cast<size_t>(st.m_uncomp_size));
      if (!data.empty() && !mz_zip_reader_extract_to_mem(&zip, i, data.data(), data.size(), 0)) {
        if (on_note) on_note(name + ": could not decompress");
        continue;
      }
      scan_at(name, data, on_hit, on_note, depth + 1, max_depth);
    }
    mz_zip_reader_end(&zip);
    return;
  }
  auto hits = scan(buf.data(), buf.size());
  for (Hit &h : hits) on_hit(where, h, buf.data());
}

}  // namespace

bool is_zip(const uint8_t *p, size_t n) { return n > 4 && p[0] == 'P' && p[1] == 'K' && p[2] == 3 && p[3] == 4; }

bool read_whole_file(const std::string &path, std::vector<uint8_t> &out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  f.seekg(0, std::ios::end);
  std::streamoff n = f.tellg();
  if (n < 0) return false;
  f.seekg(0);
  out.resize(static_cast<size_t>(n));
  return n == 0 || static_cast<bool>(f.read(reinterpret_cast<char *>(out.data()), n));
}

void scan_tree(const std::string &where, const std::vector<uint8_t> &buf, const HitFn &on_hit, const NoteFn &on_note,
               int max_depth) {
  scan_at(where, buf, on_hit, on_note, 0, max_depth);
}

}  // namespace tl
