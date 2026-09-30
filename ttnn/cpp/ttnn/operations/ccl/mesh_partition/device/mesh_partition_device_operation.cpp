// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

#include "ttnn/tensor/types.hpp"
#include "mesh_partition_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "cpp/ttnn/operations/data_movement/common/common.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::ccl {

namespace detail {
uint32_t get_cluster_axis_size(const ttnn::Tensor& input_tensor, const std::optional<uint32_t>& cluster_axis) {
    auto* mesh_device = input_tensor.device();
    const auto& mesh_view = mesh_device->get_view();
    return cluster_axis.has_value() ? ((cluster_axis.value() == 0) ? mesh_view.num_rows() : mesh_view.num_cols())
                                    : mesh_view.num_devices();
}
}  // namespace detail

void MeshPartitionDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto input_tensor = tensor_args.input_tensor;
    uint32_t rank = input_tensor.logical_shape().rank();
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    if (tensor_args.optional_output_tensor.has_value()) {
        TT_FATAL(
            tensor_args.optional_output_tensor.value().tensor_spec() == output_spec,
            "Output tensor spec must match computed output spec");
    }
    const auto& input_shape = input_tensor.logical_shape();
    const auto& output_shape = output_spec.padded_shape();
    const auto& output_padded_shape = output_spec.padded_shape();

    TT_FATAL(
        !(operation_attributes.cluster_axis.has_value() && operation_attributes.cluster_axis.value() > 1),
        "Only support cluster axis of None, 0 or 1");

    TT_FATAL(operation_attributes.dim < rank, "dim must be less than the rank of the input tensor");

    const uint32_t cluster_axis_size = detail::get_cluster_axis_size(input_tensor, operation_attributes.cluster_axis);

    TT_FATAL(
        cluster_axis_size > 1,
        "Partition has only been tested with mesh axis size > 1, but has {} devices",
        cluster_axis_size);
    TT_FATAL(
        input_shape[operation_attributes.dim] % cluster_axis_size == 0,
        "input shape {} must be divisible by cluster axis size {}",
        input_tensor.logical_shape(),
        cluster_axis_size);

    if (input_tensor.layout() == ttnn::TILE_LAYOUT) {
        TT_FATAL(
            output_shape[operation_attributes.dim] == output_padded_shape[operation_attributes.dim],
            "for tiled inputs, partitioning along dim {} should not create padding in output shape {}",
            operation_attributes.dim,
            output_shape);
    }
}

void MeshPartitionDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {}

MeshPartitionDeviceOperation::spec_return_value_t MeshPartitionDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto input_tensor = tensor_args.input_tensor;
    auto output_shape = input_tensor.logical_shape();

    const uint32_t cluster_axis_size = detail::get_cluster_axis_size(input_tensor, operation_attributes.cluster_axis);

    output_shape[operation_attributes.dim] = output_shape[operation_attributes.dim] / cluster_axis_size;
    return {tt::tt_metal::TensorSpec(
        Shape(output_shape),
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(),
            tt::tt_metal::PageConfig(input_tensor.layout()),
            operation_attributes.output_mem_config))};
}

MeshPartitionDeviceOperation::tensor_return_value_t MeshPartitionDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor.value();
    }

    auto output_spec = compute_output_specs(operation_attributes, tensor_args);

    auto tensor = create_device_tensor(output_spec, tensor_args.input_tensor.device());
    return tensor;
}

std::vector<tt::tt_metal::TensorTopology> MeshPartitionDeviceOperation::compute_output_topologies(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using Placement = tt::tt_metal::distributed::MeshMapperConfig::Placement;
    using Shard = tt::tt_metal::distributed::MeshMapperConfig::Shard;
    using Replicate = tt::tt_metal::distributed::MeshMapperConfig::Replicate;

    // Every device on the partitioned axis ends up with a distinct slice of `dim`, so the output is Shard{dim} on
    // that axis (the reduce_scatter contract, as reduce_scatter_minimal_async labels it) and unchanged on the other
    // axes -- with one constraint: a tensor dim must never be sharded on two mesh axes, because the N-D composer
    // (concat_ndim) requires unique dims. When the honest result has no TensorTopology spelling the hook returns {}
    // so launch() keeps the union default (the input's label) and logs a warning. The hook is self-contained on
    // purpose: the shared CCL topology helper's reduce_scatter rules describe a SUM across the mesh axis, which is
    // not what a partition does, so mesh_partition does not route through it. `dim` is already normalised to
    // [0, rank) by ttnn::prim::mesh_partition.
    // This runs before validation, so it must not dereference the optional cluster_axis unchecked.
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& input_topology = input_tensor.tensor_topology();
    const auto& input_placements = input_topology.placements();
    const auto& distribution_shape = input_topology.distribution_shape();
    const auto& logical_shape = input_tensor.logical_shape();
    const int rank = static_cast<int>(logical_shape.rank());
    const Shard shard_placement{static_cast<int>(operation_attributes.dim)};

    // Mappers store Shard::dim as given (possibly negative), so compare normalised. Out-of-range dims are left behind
    // by rank-changing ops (#52331) and count as not sharding `dim`, as in all_gather.
    const auto shards_dim = [&](const Placement& placement) {
        const auto* shard = std::get_if<Shard>(&placement);
        if (shard == nullptr || shard->dim >= rank || shard->dim < -rank) {
            return false;
        }
        return logical_shape.get_normalized_index(shard->dim) == operation_attributes.dim;
    };
    const auto axis_size = [&](size_t axis) -> uint32_t {
        return axis < distribution_shape.dims() ? distribution_shape[static_cast<int>(axis)] : 1;
    };
    // Every labelled device holds a distinct slice of `dim`, enumerated in the row-major order the coordinates already
    // record (get_linearized_index = row * cols + col in the program factory): the collapsed 1-D label that
    // ShardTensorToMesh(dim) produces (row-major hierarchical sharding).
    const auto collapsed_label = [&]() {
        return tt::tt_metal::TensorTopology(
            tt::tt_metal::distributed::MeshShape(static_cast<uint32_t>(distribution_shape.mesh_size())),
            {shard_placement},
            input_topology.mesh_coords());
    };
    const auto fallback = [&](const char* reason) {
        // cluster_axis is logged as -1 for a whole-mesh (nullopt) partition.
        const int cluster_axis_or_whole_mesh =
            operation_attributes.cluster_axis.has_value() ? static_cast<int>(*operation_attributes.cluster_axis) : -1;
        log_warning(
            tt::LogOp,
            "mesh_partition(dim={}, cluster_axis={}) on an input distributed over {}: {}; the output keeps the input's "
            "TensorTopology, which does not describe the partitioned result",
            operation_attributes.dim,
            cluster_axis_or_whole_mesh,
            distribution_shape,
            reason);
        return std::vector<tt::tt_metal::TensorTopology>{};
    };

    if (!operation_attributes.cluster_axis.has_value()) {
        // Whole-mesh partition: device k (row-major) gets slice k of `dim`. An N-D label would need Shard{dim} on
        // every axis, so the label is the collapsed one. It is exact for an input that is Replicate on every
        // non-trivial axis; for an input already Shard{dim} on a non-trivial axis it overwrites that Shard{dim} and
        // describes the output bytes (the reduce_scatter_minimal_async stance: the collective's placement replaces
        // what the partitioned axis held). Any other Shard is still present on the device next to the new slice, and
        // no label can state both.
        for (size_t axis = 0; axis < input_placements.size(); ++axis) {
            if (axis_size(axis) > 1 && std::holds_alternative<Shard>(input_placements[axis]) &&
                !shards_dim(input_placements[axis])) {
                return fallback("a whole-mesh partition of a tensor sharded on another dim is not expressible");
            }
        }
        return {collapsed_label()};
    }

    const size_t cluster_axis = operation_attributes.cluster_axis.value();
    if (input_placements.size() > 1) {
        // N-D label over the device mesh (ShardTensor2dMesh and friends).
        if (cluster_axis >= input_placements.size()) {
            return {};  // validation rejects this cluster_axis right after the hook
        }
        auto output_placements = input_placements;
        output_placements[cluster_axis] = shard_placement;
        // An outer Shard{dim} axis composes with the partitioned axis into row-major hierarchical sharding only if the
        // partitioned axis held the full extent of `dim` (Replicate) or its own slice of it (Shard{dim}).
        const Placement& partitioned_placement = input_placements[cluster_axis];
        const bool partitioned_axis_composes =
            std::holds_alternative<Replicate>(partitioned_placement) || shards_dim(partitioned_placement);
        for (size_t axis = 0; axis < output_placements.size(); ++axis) {
            if (axis == cluster_axis || !shards_dim(input_placements[axis])) {
                continue;
            }
            if (axis_size(axis) == 1) {
                // One chunk along a size-1 axis is the whole extent: Replicate is exact.
                output_placements[axis] = Replicate{};
                continue;
            }
            // Another non-trivial axis shards `dim`. Outer axis (coarse slices) then partitioned axis (fine slices),
            // with every remaining axis trivial, is exactly row-major hierarchical sharding -> collapsed label. An
            // inner axis (fine before coarse, i.e. column-major), a partitioned axis that held a different Shard, or a
            // third non-trivial axis (the collapsed label would over-claim N distinct slices) is not expressible.
            bool other_axes_trivial = true;
            for (size_t other = 0; other < output_placements.size(); ++other) {
                if (other != axis && other != cluster_axis && axis_size(other) > 1) {
                    other_axes_trivial = false;
                }
            }
            if (axis < cluster_axis && partitioned_axis_composes && other_axes_trivial) {
                return {collapsed_label()};
            }
            return fallback("partitioning a dim that another mesh axis already shards is not expressible");
        }
        return {tt::tt_metal::TensorTopology(
            distribution_shape, std::move(output_placements), input_topology.mesh_coords())};
    }

    if (input_placements.size() != 1) {
        return {};  // malformed label without placements: leave the union default in place
    }

    // Collapsed 1-D label ({N},[placement]) from the default mappers (ReplicateTensorToMesh / ShardTensorToMesh).
    // Both rules below need the label to cover the mesh: a fewer-shards or sub-mesh label says nothing about where
    // its N devices sit on the mesh axes.
    const auto num_label_devices = distribution_shape.mesh_size();
    const auto& mesh_shape = input_tensor.device()->get_view().shape();
    const bool label_covers_mesh = num_label_devices == mesh_shape.mesh_size();
    const uint32_t cluster_axis_size = detail::get_cluster_axis_size(input_tensor, operation_attributes.cluster_axis);

    // The collapsed axis IS the partitioned axis (cluster_axis=1 on a 1xN ring): every device holds a distinct slice,
    // so [Shard{dim}] over the same coordinates is the honest label rather than leaving the input's Replicate in place
    // (which the serialiser would dedup). A collapsed Shard{k} is overwritten the same way (the
    // reduce_scatter_minimal_async stance: the partitioned axis takes the collective's placement); the whole-mesh
    // rule above has no cluster axis to identify with the label's axis, so it falls back for such an input instead.
    // The covers-mesh guard keeps a fewer-shards label whose N happens to equal the axis size on a multi-axis mesh
    // out of this branch.
    if (label_covers_mesh && num_label_devices == cluster_axis_size) {
        return {tt::tt_metal::TensorTopology(distribution_shape, {shard_placement}, input_topology.mesh_coords())};
    }

    // Replicated over the whole mesh but partitioned along one of several axes: only an N-D label over the device
    // mesh can say "Shard{dim} here, Replicate there". The collapsed coordinates already enumerate that mesh
    // row-major, so they carry over unchanged.
    if (label_covers_mesh && cluster_axis < mesh_shape.dims() &&
        std::holds_alternative<Replicate>(input_placements[0])) {
        ttsl::SmallVector<Placement> output_placements(mesh_shape.dims(), Replicate{});
        output_placements[cluster_axis] = shard_placement;
        return {tt::tt_metal::TensorTopology(mesh_shape, std::move(output_placements), input_topology.mesh_coords())};
    }

    // A collapsed Shard label partitioned along one axis of a multi-axis mesh, or a label that does not cover the
    // mesh: no exact expression in either form; keep the union default (input label), which is what this op
    // returned for every input before it had a topology hook.
    return {};
}

}  // namespace ttnn::operations::ccl

namespace ttnn::prim {
ttnn::Tensor mesh_partition(
    const ttnn::Tensor& input_tensor,
    int32_t dim,
    std::optional<uint32_t> cluster_axis,
    const ttnn::MemoryConfig& memory_config,
    const std::optional<ttnn::Tensor>& optional_output_tensor) {
    using OperationType = ttnn::operations::ccl::MeshPartitionDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .dim = (dim < 0 ? uint32_t(input_tensor.logical_shape().rank() + dim) : (uint32_t)dim),
            .cluster_axis = cluster_axis,
            .output_mem_config = memory_config,
        },
        OperationType::tensor_args_t{.input_tensor = input_tensor, .optional_output_tensor = optional_output_tensor});
}
}  // namespace ttnn::prim
