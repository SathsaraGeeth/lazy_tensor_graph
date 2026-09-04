/* 
 * user/include/tensor/tensor_graph.h
 *
 * Copyright (C) 2026 Sathsara Geeth
 *
 */

/*
 * Version 1.0
 *
 * Version History
 *
 * Version | Description
 * --------+-----------------------------------------
 * 1.0     | Initial implementation
 */

/*
 * Comments:
 * 1. Lazy
 * 2. FROZEN
 * 3. The graph's structure is csr + csr_reversed
 *    this comes free when the user call the lazy methods
 *    as those are the parameters essentially
 */

#ifndef TENSOR_TENSOR_GRAPH_H
#define TENSOR_TENSOR_GRAPH_H

#include "tensor_core.h"

typedef struct vtensor vtensor;

typedef enum {
    UNMAT,
    MAT
} node_state_t;

typedef enum {
    ALLOC,
    IR_NODE
} edge_kind_t;

typedef struct {
    uint32 op;
    extent input_count;
    extent parameter_bytes;
} op_t;

typedef struct {
    extent    *shape;
    extent    rank;
    dtype_t   dtype;
    boolean   init_with_data;
    boolean   init_static_data;
    mem_block *data_block;
} tensor_init_t;

typedef struct {
    edge_kind_t   kind;
    tensor_init_t *init;
    op_t          op;
    void          *parameters;
    ker_t         kernel;
} edge_t;

struct vtensor {
    tensor       *phy_tensor;
    edge_t       *edge;
    vtensor      **parents;
    extent       num_parents;
    extent       num_live_children;
    extent       num_live_static_ref;
    node_state_t state;
    extent       rank;
    extent       *shape;
    dtype_t      dtype;
    vtensor      *registry_next;
};

vtensor *tensor_lazy_alloc     (extent rank, const extent *shape, dtype_t dtype, extent num_static_ref,
                                boolean init_with_data, boolean init_data_static, dptr *data);
boolean tensor_lazy_op_dispatch(op_t op, vtensor *output, const vtensor **inputs, const void *parameters);
dptr    *tensor_lazy_view_to   (const vtensor *tensor);
boolean tensor_lazy_view_from  (vtensor *tensor, dptr *data);
boolean tensor_lazy_free       (vtensor *tensor);

#endif /* TENSOR_TENSOR_GRAPH_H */
