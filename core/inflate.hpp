#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tl {

// Decode a gzip member starting at p. On success `out` holds the data and `consumed` the number of
// compressed bytes (header and trailer included). Fails cleanly on anything that is not gzip or
// that would inflate beyond `cap` bytes.
bool gunzip_at(const uint8_t *p, size_t avail, std::vector<uint8_t> &out, size_t &consumed,
               size_t cap = 512u << 20);

// If `offset` in a decompressed tar archive is the start of a member's data, return its name.
std::string tar_member_name(const uint8_t *data, size_t size, uint64_t offset);

}  // namespace tl
