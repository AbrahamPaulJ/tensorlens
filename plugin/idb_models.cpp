// Find models in the loaded segments (and optionally in the parts of the input file that are not
// loaded), then label them and record which code references them.
#include <algorithm>
#include <cstdio>
#include <fstream>

#include "container.hpp"
#include "state.hpp"

#include <bytes.hpp>
#include <kernwin.hpp>
#include <loader.hpp>
#include <name.hpp>
#include <nalt.hpp>
#include <segment.hpp>
#include <xref.hpp>
#include <funcs.hpp>

namespace tl::ida {

std::string hexea(ea_t ea) {
  char b[32];
  qsnprintf(b, sizeof b, "0x%llX", (unsigned long long)ea);
  return b;
}

bool rename_if_dummy(ea_t ea, const std::string &name) {
  flags64_t f = get_flags(ea);
  if (has_user_name(f)) return false;
  qstring cur;
  if (get_ea_name(&cur, ea) > 0 && !has_dummy_name(f) && has_any_name(f)) return false;
  return set_name(ea, name.c_str(), SN_NOWARN | SN_NOCHECK | SN_FORCE);
}

std::string FoundModel::label() const {
  char b[160];
  qsnprintf(b, sizeof b, "%s, %zu ops", format_name(hit.model.format), hit.model.node_count());
  return b;
}

namespace {

void add_hits(State &st, std::vector<Hit> &&hits, const uint8_t *base, ea_t seg_start, const std::string &seg,
              bool file_only, uint64_t file_base) {
  for (Hit &h : hits) {
    FoundModel m;
    m.segment = seg;
    m.file_only = file_only;
    if (file_only) {
      m.file_offset = file_base + h.offset;
      m.ea = BADADDR;
    } else {
      m.ea = seg_start + h.offset;
    }
    if (h.model.size) {
      const uint8_t *p = h.bytes(base);
      m.bytes.assign(p, p + h.model.size);
    }
    m.hit = std::move(h);
    st.models.push_back(std::move(m));
  }
}

bool read_file(const char *path, std::vector<uint8_t> &out) {
  FILE *f = qfopen(path, "rb");
  if (f == nullptr) return false;
  qfseek(f, 0, SEEK_END);
  int64 n = qftell(f);
  qfseek(f, 0, SEEK_SET);
  if (n <= 0 || n > (int64(2) << 30)) {
    qfclose(f);
    return false;
  }
  out.resize(size_t(n));
  bool ok = qfread(f, out.data(), out.size()) == ssize_t(out.size());
  qfclose(f);
  return ok;
}

}  // namespace

void scan_database(State &st, bool include_input_file) {
  // Keep models added from other files; rescan the database's own.
  st.models.erase(std::remove_if(st.models.begin(), st.models.end(), [](const FoundModel &m) { return !m.external; }),
                  st.models.end());
  show_wait_box("TensorLens: scanning for ML models...");
  std::set<ea_t> found_eas;
  for (ea_t s = get_first_segment_ea(); s != BADADDR; s = get_next_segment_ea(s)) {
    segment_info_t si;
    if (!get_segment_info(&si, s, GSI_NAME)) continue;
    if (si.get_type() == SEG_XTRN || si.get_type() == SEG_BSS) continue;
    asize_t size = si.end_ea - si.start_ea;
    if (size < 16 || size > (asize_t(2) << 30)) continue;
    std::vector<uint8_t> buf(static_cast<size_t>(size));
    ssize_t got = get_bytes(buf.data(), buf.size(), si.start_ea, GMB_READALL);
    if (got <= 0) continue;
    buf.resize(size_t(got));
    replace_wait_box("TensorLens: scanning %s (%s)", si.get_name(), format_bytes(buf.size()).c_str());
    ScanOptions opt;
    opt.progress = [](uint64_t, uint64_t) { return !user_cancelled(); };
    auto hits = scan(buf.data(), buf.size(), opt);
    for (const Hit &h : hits) found_eas.insert(si.start_ea + h.offset);
    add_hits(st, std::move(hits), buf.data(), si.start_ea, si.get_name(), false, 0);
    if (user_cancelled()) break;
  }

  // Bytes of the input file that never made it into a segment (PE overlays, unmapped ELF
  // sections, resources appended after the image) can hold models too.
  if (include_input_file && !user_cancelled()) {
    char path[QMAXPATH];
    std::vector<uint8_t> file;
    if (get_input_file_path(path, sizeof path) > 0 && read_file(path, file)) {
      replace_wait_box("TensorLens: scanning the input file (%s)", format_bytes(file.size()).c_str());
      auto hits = scan(file.data(), file.size());
      std::vector<Hit> extra;
      for (Hit &h : hits) {
        ea_t ea = get_fileregion_ea(qoff64_t(h.offset));
        if (ea != BADADDR && found_eas.count(ea)) continue;  // already found in a segment
        if (ea != BADADDR && is_loaded(ea)) {
          // Mapped but missed above (e.g. a segment we skipped): record it at its address.
          std::vector<Hit> one;
          uint64_t off = h.offset;
          one.push_back(std::move(h));
          one.back().offset = 0;
          add_hits(st, std::move(one), file.data() + off, ea, "file", false, 0);
          continue;
        }
        extra.push_back(std::move(h));
      }
      add_hits(st, std::move(extra), file.data(), BADADDR, "input file (not loaded)", true, 0);
    }
  }
  hide_wait_box();

  // Code that references a model (or the gzip stream holding it) is where it gets loaded.
  for (FoundModel &m : st.models) {
    if (m.ea == BADADDR) continue;
    std::set<ea_t> funcs;
    xrefblk_t xb;
    for (bool ok = xb.first_to(m.ea, XREF_DATA); ok; ok = xb.next_to()) {
      ea_t f = get_func_start(xb.from);
      if (f != BADADDR) funcs.insert(f);
    }
    for (ea_t f : funcs) m.referenced_by.push_back(f);
  }
  st.scanned = true;
  msg("TensorLens: %zu model(s) found\n", st.models.size());
}

int add_external_models(State &st, const std::string &path) {
  std::vector<uint8_t> buf;
  if (!read_whole_file(path, buf)) return -1;
  std::string shown = path;
  size_t slash = shown.find_last_of("/\\");
  if (slash != std::string::npos) shown = shown.substr(slash + 1);
  int added = 0;
  show_wait_box("TensorLens: scanning %s", shown.c_str());
  scan_tree(shown, buf, [&](const std::string &where, Hit &h, const uint8_t *base) {
    FoundModel m;
    m.segment = where;
    m.external = true;
    m.file_only = true;
    m.file_offset = h.offset;
    if (h.model.size) {
      const uint8_t *p = h.bytes(base);
      m.bytes.assign(p, p + h.model.size);
    }
    m.hit = std::move(h);
    st.models.push_back(std::move(m));
    added++;
  });
  hide_wait_box();
  msg("TensorLens: %d model(s) added from %s\n", added, path.c_str());
  return added;
}

void annotate_models(State &st) {
  int n = 0;
  std::set<ea_t> done;
  for (FoundModel &m : st.models) {
    int idx = n++;
    if (m.ea == BADADDR || done.count(m.ea)) continue;
    done.insert(m.ea);
    const Model &mod = m.hit.model;
    qstring name;
    name.sprnt("tl_%s_model_%d", m.hit.compressed ? "gzip" : format_name(mod.format), idx);
    for (char &c : name) c = (c >= 'A' && c <= 'Z') ? char(c - 'A' + 'a') : c;
    if (!is_head(get_flags(m.ea))) continue;  // inside another item: leave the user's layout alone
    rename_if_dummy(m.ea, name.c_str());
    std::string cmt = "TensorLens: " + mod.summary();
    if (m.hit.compressed) cmt += "\n  inside " + m.hit.container;
    if (mod.size) cmt += "\n  model size " + format_bytes(mod.size);
    auto custom = mod.custom_ops();
    if (!custom.empty()) {
      cmt += "\n  custom ops:";
      for (const std::string &c : custom) cmt += " " + c;
    }
    qstring old;
    if (get_cmt(&old, m.ea, false) <= 0 || strstr(old.c_str(), "TensorLens:") == nullptr) set_cmt(m.ea, cmt.c_str(), false);
  }
}

}  // namespace tl::ida
