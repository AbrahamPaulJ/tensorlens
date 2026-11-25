// Scan a file that may be a container (APK, AAR, JAR, wheel, MediaPipe .task: all zip) for models.
#pragma once
#include <functional>
#include <string>
#include <vector>

#include "scan.hpp"

namespace tl {

// Called for every model found. `where` is "file!entry!inner-entry"; `base` is the buffer the
// hit's offsets refer to (valid only during the call).
using HitFn = std::function<void(const std::string &where, Hit &hit, const uint8_t *base)>;
using NoteFn = std::function<void(const std::string &note)>;

bool read_whole_file(const std::string &path, std::vector<uint8_t> &out);
bool is_zip(const uint8_t *p, size_t n);

// Scans `buf`; recurses into zip archives up to `max_depth` levels.
void scan_tree(const std::string &where, const std::vector<uint8_t> &buf, const HitFn &on_hit,
               const NoteFn &on_note = nullptr, int max_depth = 3);

}  // namespace tl
