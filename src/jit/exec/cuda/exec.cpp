#include "internal.h"
#include "../../gen/cuda/ptx.h"
#include <cuda.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

struct mirror {
    tensor *value{};
    mem_block *block{};
    extent *shape{};
    stride *strides{};
};

template <typename Value>
Value *managed(extent count = 1) {
    CUdeviceptr pointer = 0;
    if (!count || cuMemAllocManaged(&pointer, count * sizeof(Value), CU_MEM_ATTACH_GLOBAL) != CUDA_SUCCESS) return nullptr;
    std::memset(reinterpret_cast<void *>(pointer), 0, count * sizeof(Value));
    return reinterpret_cast<Value *>(pointer);
}

void release(void *pointer) {
    if (pointer) cuMemFree(reinterpret_cast<CUdeviceptr>(pointer));
}

mirror mirror_tensor(const tensor *source) {
    mirror result;
    if (!source || !source->data) return result;
    result.value = managed<tensor>();
    result.block = managed<mem_block>();
    result.shape = managed<extent>(source->rank);
    result.strides = managed<stride>(source->rank);
    if (!result.value || !result.block || !result.shape || !result.strides) return result;
    *result.block = *source->data;
    std::memcpy(result.shape, source->shape, source->rank * sizeof(extent));
    std::memcpy(result.strides, source->strides, source->rank * sizeof(stride));
    *result.value = *source;
    result.value->data = result.block;
    result.value->shape = result.shape;
    result.value->strides = result.strides;
    return result;
}

void release(mirror &value) {
    release(value.value);
    release(value.block);
    release(value.shape);
    release(value.strides);
    value = {};
}

boolean prepare_queue(const exec_work *items, extent count, std::vector<vtensor *> &kernels) {
    for (extent i = 0; i < count; ++i) {
        vtensor *node = items[i].node;
        if (items[i].kind == EXEC_ALLOC && cuda_allocate(node)) return true;
        if (items[i].kind == EXEC_INIT && cuda_initialize(node)) return true;
        if (items[i].kind == EXEC_KERNEL) kernels.push_back(node);
    }
    return kernels.empty();
}

boolean launch(const std::vector<vtensor *> &nodes, const char *ptx, const char *kernel_name) {
    extent count = nodes.size();
    std::vector<mirror> outputs(count);
    std::vector<std::vector<mirror>> inputs(count);
    tensor **device_outputs = managed<tensor *>(count);
    tensor ***device_inputs = managed<tensor **>(count);
    extent *input_counts = managed<extent>(count);
    void **parameters = managed<void *>(count);
    extent *parameter_bytes = managed<extent>(count);
    uint32 *status = managed<uint32>();
    std::vector<void *> allocations;
    if (!device_outputs || !device_inputs || !input_counts || !parameters || !parameter_bytes || !status) return true;

    boolean failed = false;
    for (extent i = 0; i < count && !failed; ++i) {
        outputs[i] = mirror_tensor(nodes[i]->phy_tensor);
        if (!outputs[i].value) {
            failed = true;
            break;
        }
        device_outputs[i] = outputs[i].value;
        input_counts[i] = nodes[i]->num_parents;
        tensor **stage_inputs = managed<tensor *>(input_counts[i]);
        device_inputs[i] = stage_inputs;
        if (input_counts[i] && !stage_inputs) {
            failed = true;
            break;
        }
        inputs[i].resize(input_counts[i]);
        for (extent j = 0; j < input_counts[i]; ++j) {
            inputs[i][j] = mirror_tensor(nodes[i]->parents[j]->phy_tensor);
            if (!inputs[i][j].value) {
                failed = true;
                break;
            }
            stage_inputs[j] = inputs[i][j].value;
        }
        parameter_bytes[i] = nodes[i]->edge->op.parameter_bytes;
        if (parameter_bytes[i]) {
            uint8 *copy = managed<uint8>(parameter_bytes[i]);
            if (!copy) {
                failed = true;
                break;
            }
            std::memcpy(copy, nodes[i]->edge->parameters, parameter_bytes[i]);
            parameters[i] = copy;
            allocations.push_back(copy);
        }
    }

    CUmodule module = nullptr;
    CUfunction function = nullptr;
    if (!failed && (cuModuleLoadData(&module, ptx) != CUDA_SUCCESS ||
                    cuModuleGetFunction(&function, module, kernel_name) != CUDA_SUCCESS)) failed = true;
    void *arguments[] = {&device_outputs, &device_inputs, &input_counts, &parameters, &parameter_bytes, &status};
    int cooperative = 0;
    int multiprocessors = 0;
    int blocks_per_multiprocessor = 0;
    CUdevice device = 0;
    const uint32 threads = 256;
    if (!failed && (cuCtxGetDevice(&device) != CUDA_SUCCESS ||
                    cuDeviceGetAttribute(&cooperative, CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH, device) != CUDA_SUCCESS ||
                    cuDeviceGetAttribute(&multiprocessors, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device) != CUDA_SUCCESS ||
                    cuOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_multiprocessor, function, threads, 0) !=
                        CUDA_SUCCESS || !cooperative || !multiprocessors || !blocks_per_multiprocessor)) failed = true;
    extent work = 1;
    for (const vtensor *node : nodes) work = std::max(work, node->phy_tensor->size);
    uint32 needed_blocks = static_cast<uint32>((work + threads - 1) / threads);
    uint32 resident_blocks = static_cast<uint32>(multiprocessors * blocks_per_multiprocessor);
    uint32 blocks = std::max(UINT32_C(1), std::min(needed_blocks, resident_blocks));
    if (!failed) {
        CUresult launch_result = cuLaunchCooperativeKernel(
            function, blocks, 1, 1, threads, 1, 1, 0, nullptr, arguments);
        CUresult synchronize_result = launch_result == CUDA_SUCCESS
                                          ? cuCtxSynchronize()
                                          : CUDA_SUCCESS;
        if (launch_result != CUDA_SUCCESS ||
            synchronize_result != CUDA_SUCCESS || *status)
            failed = true;
    }
    if (module) cuModuleUnload(module);
    for (extent i = 0; i < count; ++i) {
        for (mirror &input : inputs[i]) release(input);
        release(device_inputs[i]);
        release(outputs[i]);
        if (!failed) nodes[i]->state = MAT;
    }
    for (void *allocation : allocations) release(allocation);
    release(device_outputs);
    release(device_inputs);
    release(input_counts);
    release(parameters);
    release(parameter_bytes);
    release(status);
    return failed;
}

} // namespace

extern "C" boolean tensor_exec_submit(const void *opaque_items, extent item_count) {
    const auto *items = static_cast<const exec_work *>(opaque_items);
    if (item_count && !items) return true;
    std::vector<vtensor *> kernels;
    if (prepare_queue(items, item_count, kernels)) return true;
    std::vector<const vtensor *> immutable(kernels.begin(), kernels.end());
    char *ptx = nullptr;
    char *kernel_name = nullptr;
    boolean failed = tensor_cuda_ptx(immutable.data(), immutable.size(), &ptx, &kernel_name);
    if (!failed) failed = launch(kernels, ptx, kernel_name);
    tensor_cuda_ptx_free(ptx);
    tensor_cuda_ptx_free(kernel_name);
    return failed;
}
