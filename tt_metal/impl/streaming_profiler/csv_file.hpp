// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdio>
#include <memory>
#include <string>
#include <string_view>

#include <tt_stl/assert.hpp>

namespace tt::tt_metal::streaming_profiler {

// A CSV file the consumers write their header into once.
class CsvFile {
public:
    CsvFile(const std::string& path, std::string_view what) : path_(path), file_(std::fopen(path.c_str(), "w")) {
        TT_FATAL(file_ != nullptr, "streaming profiler: cannot open {} for the {}", path, what);
    }
    const std::string& path() const { return path_; }
    template <typename WriteHeader>
    std::FILE* begin(WriteHeader&& write_header) {
        if (!header_written_) {
            write_header(file_.get());
            header_written_ = true;
        }
        return file_.get();
    }

private:
    struct Close {
        void operator()(std::FILE* file) const { std::fclose(file); }
    };
    std::string path_;
    std::unique_ptr<std::FILE, Close> file_;
    bool header_written_ = false;
};

}  // namespace tt::tt_metal::streaming_profiler
