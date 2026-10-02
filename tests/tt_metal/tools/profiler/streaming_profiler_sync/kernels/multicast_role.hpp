// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// A source's value is the NoC it multicasts on.
enum class MulticastRole : std::uint32_t { Noc0Source, Noc1Source, Receiver };
