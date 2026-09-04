#include "internal.h"
#include "tensor_jit.h"

#include <stdlib.h>

#define FUSION_COUNT_MASK UINT32_C(0x0fffffff)
#define FUSION_MAGIC UINT32_C(0x46555331)

typedef struct {
    uint32 magic;
    uint32 count;
    extent parameter_bytes;
} fusion_header;

typedef struct {
    uint32 operation;
    uint32 reserved;
    extent input_count;
    extent parameter_bytes;
} fusion_stage;

static const fusion_header *read_program(const vtensor *node, const fusion_stage **stages, const uint8 **parameters) {
    if (!node || !node->edge || !node->edge->parameters || node->edge->op.parameter_bytes < sizeof(fusion_header))
        return NULL;
    const fusion_header *header = node->edge->parameters;
    extent records = (extent)header->count * sizeof(fusion_stage);
    extent expected = sizeof(*header) + records + header->parameter_bytes;
    if (header->magic != FUSION_MAGIC || header->count != (node->edge->op.op & FUSION_COUNT_MASK) ||
        header->count < 2 || expected != node->edge->op.parameter_bytes) return NULL;
    *stages = (const fusion_stage *)(header + 1);
    *parameters = (const uint8 *)(*stages + header->count);
    return header;
}

static extent maximum_inputs(const fusion_stage *stages, uint32 count) {
    extent maximum = 0;
    for (uint32 i = 0; i < count; ++i)
        if (stages[i].input_count > maximum) maximum = stages[i].input_count;
    return maximum;
}

static tensor *temporary_for(const vtensor *node) {
    return tensor_alloc(node->rank, node->shape, node->dtype, true);
}

static tensor metadata_from(const vtensor *node) {
    tensor value = {0};
    value.shape = node->shape;
    value.rank = node->rank;
    value.dtype = node->dtype;
    value.size = 1;
    value.is_contiguous = true;
    for (extent i = 0; i < node->rank; ++i) value.size *= node->shape[i];
    return value;
}

boolean exec_prepare_fused(vtensor *node) {
    const fusion_stage *stages = NULL;
    const uint8 *parameters = NULL;
    const fusion_header *header = read_program(node, &stages, &parameters);
    (void)parameters;
    if (!header) return true;
    tensor output = metadata_from(node);
    tensor *external = node->num_parents ? malloc(node->num_parents * sizeof(*external)) : NULL;
    extent capacity = maximum_inputs(stages, header->count);
    const tensor **stage_inputs = capacity ? malloc(capacity * sizeof(*stage_inputs)) : NULL;
    if ((node->num_parents && !external) || (capacity && !stage_inputs)) {
        free(external);
        free(stage_inputs);
        return true;
    }
    for (extent i = 0; i < node->num_parents; ++i) external[i] = metadata_from(node->parents[i]);
    extent cursor = 0;
    boolean failed = false;
    for (uint32 i = 0; i < header->count && !failed; ++i) {
        extent first = i ? 1 : 0;
        if (i) stage_inputs[0] = &output;
        for (extent j = first; j < stages[i].input_count; ++j) {
            if (cursor == node->num_parents) {
                failed = true;
                break;
            }
            stage_inputs[j] = &external[cursor++];
        }
        if (!failed) {
            jit_cache_key_t key = tensor_jit_key_create(stages[i].operation, &output, stage_inputs,
                                                        stages[i].input_count, get_device());
            failed = tensor_jit_prepare(key, tensor_get_jit_cache());
        }
    }
    if (cursor != node->num_parents) failed = true;
    free(external);
    free(stage_inputs);
    return failed;
}

boolean exec_execute_fused(vtensor *node, const tensor *const *inputs) {
    const fusion_stage *stages = NULL;
    const uint8 *parameters = NULL;
    const fusion_header *header = read_program(node, &stages, &parameters);
    if (!header) return true;
    extent expected_inputs = stages[0].input_count;
    for (uint32 i = 1; i < header->count; ++i) {
        if (!stages[i].input_count) return true;
        expected_inputs += stages[i].input_count - 1;
    }
    if (expected_inputs != node->num_parents) return true;

    tensor *temporary[2] = {temporary_for(node), header->count > 2 ? temporary_for(node) : NULL};
    extent input_capacity = maximum_inputs(stages, header->count);
    const tensor **stage_inputs = input_capacity ? malloc(input_capacity * sizeof(*stage_inputs)) : NULL;
    if (!temporary[0] || temporary[0]->error || (temporary[1] && temporary[1]->error) ||
        (input_capacity && !stage_inputs)) {
        tensor_free(temporary[0]);
        tensor_free(temporary[1]);
        free(stage_inputs);
        return true;
    }

    extent external = 0;
    extent parameter_offset = 0;
    boolean failed = false;
    for (uint32 i = 0; i < header->count && !failed; ++i) {
        extent first = i ? 1 : 0;
        if (i) stage_inputs[0] = temporary[(i - 1) & 1u];
        for (extent j = first; j < stages[i].input_count; ++j) stage_inputs[j] = inputs[external++];
        tensor *stage_output = i + 1 == header->count ? node->phy_tensor : temporary[i & 1u];
        failed = exec_dispatch_operation(stages[i].operation, stage_output, stage_inputs, stages[i].input_count,
                                         parameters + parameter_offset, stages[i].parameter_bytes);
        parameter_offset += stages[i].parameter_bytes;
    }
    if (external != node->num_parents || parameter_offset != header->parameter_bytes) failed = true;
    tensor_free(temporary[0]);
    tensor_free(temporary[1]);
    free(stage_inputs);
    return failed;
}
