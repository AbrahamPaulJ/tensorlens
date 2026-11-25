// JSON and HTML reports, shared by the command-line scanner and the IDA plugin.
#pragma once

#include <string>
#include <vector>

#include "model.hpp"

namespace tl {

struct ReportModel {
  std::string where;      // file, APK entry or IDA segment the model was found in
  uint64_t offset = 0;    // offset within `where` (or address in IDA)
  bool offset_is_ea = false;
  Model model;
};

// Free-form findings, e.g. TFLite kernels recovered in a native library.
struct ReportSection {
  std::string title;
  std::vector<std::string> columns;
  std::vector<std::vector<std::string>> rows;
};

struct Report {
  std::string title;
  std::string subject;    // what was analysed
  std::vector<ReportModel> models;
  std::vector<ReportSection> sections;
  std::vector<std::string> notes;
};

std::string to_json(const Report &r, bool include_tensors = false);
std::string to_html(const Report &r);
std::string json_escape(const std::string &s);
std::string html_escape(const std::string &s);

}  // namespace tl
