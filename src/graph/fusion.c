#include "tensor_graph.h"

#include <stdlib.h>
#include <string.h>

#define FUSION_TAG UINT32_C(0xf0000000)
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

boolean tensor_jit_is_pointwise(uint32 operation);

static boolean is_fused(uint32 operation) {
    return (operation & FUSION_TAG) == FUSION_TAG && (operation & FUSION_COUNT_MASK) >= 2;
}

static boolean same_layout(const vtensor *left, const vtensor *right) {
    return left && right && left->dtype == right->dtype && left->rank == right->rank && left->shape && right->shape &&
           !memcmp(left->shape, right->shape, left->rank * sizeof(*left->shape));
}

static const fusion_header *fusion_program(const vtensor *node) {
    if (!node || !node->edge || !is_fused(node->edge->op.op) || !node->edge->parameters ||
        node->edge->op.parameter_bytes < sizeof(fusion_header)) return NULL;
    const fusion_header *header = node->edge->parameters;
    extent records = (extent)header->count * sizeof(fusion_stage);
    extent expected = sizeof(*header) + records + header->parameter_bytes;
    if (header->magic != FUSION_MAGIC || header->count != (node->edge->op.op & FUSION_COUNT_MASK) ||
        expected != node->edge->op.parameter_bytes) return NULL;
    return header;
}

static boolean can_fuse(const vtensor *output, const vtensor *producer) {
    if (!output || !producer || !output->edge || !producer->edge || output->edge->kind != IR_NODE ||
        producer->edge->kind != IR_NODE || !output->num_parents || output->parents[0] != producer ||
        output->state != UNMAT || producer->state != UNMAT || producer->num_live_children != 1 ||
        producer->num_live_static_ref || !tensor_jit_is_pointwise(output->edge->op.op) ||
        !same_layout(output, producer)) return false;
    return is_fused(producer->edge->op.op) ? fusion_program(producer) != NULL
                                           : tensor_jit_is_pointwise(producer->edge->op.op);
}

static boolean build_program(const vtensor *producer, const vtensor *output, void **program_out, extent *bytes_out,
                             uint32 *count_out) {
    const fusion_header *old = fusion_program(producer);
    uint32 old_count = old ? old->count : 1;
    if (old_count == FUSION_COUNT_MASK) return true;
    uint32 count = old_count + 1;
    extent old_parameters = old ? old->parameter_bytes : producer->edge->op.parameter_bytes;
    extent parameters = old_parameters + output->edge->op.parameter_bytes;
    extent bytes = sizeof(fusion_header) + (extent)count * sizeof(fusion_stage) + parameters;
    fusion_header *program = malloc(bytes);
    if (!program) return true;
    *program = (fusion_header){.magic = FUSION_MAGIC, .count = count, .parameter_bytes = parameters};
    fusion_stage *stages = (fusion_stage *)(program + 1);
    uint8 *parameter_data = (uint8 *)(stages + count);

    if (old) {
        const fusion_stage *old_stages = (const fusion_stage *)(old + 1);
        const uint8 *old_data = (const uint8 *)(old_stages + old_count);
        memcpy(stages, old_stages, (extent)old_count * sizeof(*stages));
        if (old_parameters) memcpy(parameter_data, old_data, old_parameters);
    } else {
        stages[0] = (fusion_stage){
            .operation = producer->edge->op.op,
            .input_count = producer->num_parents,
            .parameter_bytes = producer->edge->op.parameter_bytes
        };
        if (old_parameters) memcpy(parameter_data, producer->edge->parameters, old_parameters);
    }
    stages[old_count] = (fusion_stage){
        .operation = output->edge->op.op,
        .input_count = output->num_parents,
        .parameter_bytes = output->edge->op.parameter_bytes
    };
    if (output->edge->op.parameter_bytes)
        memcpy(parameter_data + old_parameters, output->edge->parameters, output->edge->op.parameter_bytes);
    *program_out = program;
    *bytes_out = bytes;
    *count_out = count;
    return false;
}

static boolean rewrite(vtensor *output, vtensor *producer) {
    extent parent_count = producer->num_parents + output->num_parents - 1;
    vtensor **parents = parent_count ? malloc(parent_count * sizeof(*parents)) : NULL;
    void *program = NULL;
    extent program_bytes = 0;
    uint32 count = 0;
    if ((parent_count && !parents) || build_program(producer, output, &program, &program_bytes, &count)) {
        free(parents);
        return true;
    }
    extent cursor = 0;
    for (extent i = 0; i < producer->num_parents; ++i) parents[cursor++] = producer->parents[i];
    for (extent i = 1; i < output->num_parents; ++i) parents[cursor++] = output->parents[i];

    if (producer->num_live_children) producer->num_live_children--;
    free(output->parents);
    free(output->edge->parameters);
    output->parents = parents;
    output->num_parents = parent_count;
    output->edge->parameters = program;
    output->edge->op = (op_t){
        .op = FUSION_TAG | count,
        .input_count = parent_count,
        .parameter_bytes = program_bytes
    };
    return false;
}

boolean graph_fusion_rewrite(vtensor *output) {
    if (!output || !output->num_parents) return false;
    vtensor *producer = output->parents[0];
    return can_fuse(output, producer) ? rewrite(output, producer) : false;
}
