/*
 * user/src/tensor/src/jit/gen/cpu/fusion/fusion.cpp
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
 * 1. Generates JIT kernels for fused ones
 *    the .op->parser-->SSA scalar(as .bc) is
 *    per .op/kernel they don't aware about fusion
 *    as it is runtime decision made by the graph/
 *    and graph/fusion.c do the fusion thingy
 *    transparenlty to whole library and to the user - it 
 *    rewrites the metadata of the nodes/vtensors so
 *    the graph now is fused and still safe
 *    it chnages the op (and other metas but not the concern of this
 *    file) this file basically parse that new op and
 *    figure out how to generate a fused kernel with
 *    just one outer loop so all the data remains in cpu
 *    regs no round trips to stack
 */

#include "fusion.h"
#include "../orc_jit.h"

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Verifier.h>
#include <algorithm>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

using namespace llvm;
using namespace llvm::orc;

extern "C" boolean tensor_jit_bitcode_find(uint32 op, dtype_t output_dtype, dtype_t input_dtype,
                                            const char **name, const uint8 **bitcode, extent *bitcode_size);
extern "C" uint64 tensor_device_jit_compatibility_hash(device dev);

namespace {

struct fusion_header {
    uint32 magic;
    uint32 count;
    extent parameter_bytes;
};

struct fusion_stage {
    uint32 operation;
    uint32 reserved;
    extent input_count;
    extent parameter_bytes;
};

struct fused_signature {
    dtype_t dtype{};
    std::vector<extent> shape;
};

struct fused_key {
    uint64 compatibility{};
    std::vector<uint8> program;
    fused_signature output;
    std::vector<fused_signature> inputs;
};

enum class fused_state { compiling, ready, failed };

struct fused_entry {
    fused_key key;
    jit_ker_t kernel{};
    fused_state state{fused_state::compiling};
    std::condition_variable ready;
};

std::mutex fused_lock;
std::vector<std::shared_ptr<fused_entry>> fused_entries;
fused_signature signature_from(const tensor *value) {
    fused_signature signature;
    if (!value) return signature;
    signature.dtype = value->dtype;
    if (value->rank && value->shape) signature.shape.assign(value->shape, value->shape + value->rank);
    return signature;
}

bool signature_equal(const fused_signature &left, const fused_signature &right) {
    return left.dtype == right.dtype && left.shape == right.shape;
}

bool key_equal(const fused_key &left, const fused_key &right) {
    if (left.compatibility != right.compatibility || left.program != right.program ||
        !signature_equal(left.output, right.output) || left.inputs.size() != right.inputs.size()) return false;
    for (size_t i = 0; i < left.inputs.size(); ++i)
        if (!signature_equal(left.inputs[i], right.inputs[i])) return false;
    return true;
}

uint64 hash_fused_key(const fused_key &key) {
    uint64 hash = UINT64_C(1469598103934665603);
    hash = tensor_cpu_hash_bytes(hash, &key.compatibility, sizeof(key.compatibility));
    if (!key.program.empty()) hash = tensor_cpu_hash_bytes(hash, key.program.data(), key.program.size());
    hash = tensor_cpu_hash_bytes(hash, &key.output.dtype, sizeof(key.output.dtype));
    extent rank = key.output.shape.size();
    hash = tensor_cpu_hash_bytes(hash, &rank, sizeof(rank));
    if (rank) hash = tensor_cpu_hash_bytes(hash, key.output.shape.data(), rank * sizeof(extent));
    extent input_count = key.inputs.size();
    hash = tensor_cpu_hash_bytes(hash, &input_count, sizeof(input_count));
    for (const fused_signature &input : key.inputs) {
        hash = tensor_cpu_hash_bytes(hash, &input.dtype, sizeof(input.dtype));
        rank = input.shape.size();
        hash = tensor_cpu_hash_bytes(hash, &rank, sizeof(rank));
        if (rank) hash = tensor_cpu_hash_bytes(hash, input.shape.data(), rank * sizeof(extent));
    }
    return hash;
}

bool make_fused_key(fused_key &key, const void *program, extent program_bytes, const tensor *output,
                    const tensor *const *inputs, extent input_count) {
    if (!program || !program_bytes || !output || (input_count && !inputs)) return false;
    key.compatibility = tensor_device_jit_compatibility_hash(CPU);
    key.program.assign(static_cast<const uint8 *>(program), static_cast<const uint8 *>(program) + program_bytes);
    key.output = signature_from(output);
    if (key.output.shape.size() != output->rank) return false;
    key.inputs.reserve(input_count);
    for (extent i = 0; i < input_count; ++i) {
        if (!inputs[i]) return false;
        key.inputs.push_back(signature_from(inputs[i]));
        if (key.inputs.back().shape.size() != inputs[i]->rank) return false;
    }
    return true;
}

const fusion_header *read_fusion(const fused_key &key, const fusion_stage **stages, const uint8 **parameters) {
    constexpr uint32 magic = UINT32_C(0x46555331);
    if (key.program.size() < sizeof(fusion_header)) return nullptr;
    const auto *header = reinterpret_cast<const fusion_header *>(key.program.data());
    if (header->magic != magic || header->count < 2 || header->count > UINT32_C(0x0fffffff)) return nullptr;
    extent record_bytes = static_cast<extent>(header->count) * sizeof(fusion_stage);
    if (record_bytes / sizeof(fusion_stage) != header->count) return nullptr;
    extent expected = sizeof(*header) + record_bytes;
    if (expected < record_bytes || expected + header->parameter_bytes < expected ||
        expected + header->parameter_bytes != key.program.size()) return nullptr;
    *stages = reinterpret_cast<const fusion_stage *>(header + 1);
    *parameters = reinterpret_cast<const uint8 *>(*stages + header->count);
    return header;
}

std::string fused_object_identifier(const fused_key &key) {
    const fusion_stage *stages = nullptr;
    const uint8 *parameters = nullptr;
    const fusion_header *header = read_fusion(key, &stages, &parameters);
    (void)parameters;
    if (!header || !stages[0].input_count || key.inputs.empty()) return {};
    const char *name = nullptr;
    const uint8 *bitcode = nullptr;
    extent bitcode_size = 0;
    if (tensor_jit_bitcode_find(stages[0].operation, key.output.dtype, key.inputs[0].dtype, &name, &bitcode,
                                &bitcode_size) || !bitcode || !bitcode_size) return {};
    uint64 revision = 1;
    uint64 hash = tensor_cpu_hash_bytes(hash_fused_key(key), bitcode, bitcode_size);
    hash = tensor_cpu_hash_bytes(hash, &revision, sizeof(revision));
    return "tensor-fused-" + std::to_string(hash);
}

size_t dtype_bytes(dtype_t dtype) {
    switch (dtype) {
        case INT8:
        case UINT8: return 1;
        case INT16:
        case UINT16: return 2;
        case INT32:
        case UINT32:
        case REAL32: return 4;
        case INT64:
        case UINT64:
        case REAL64: return 8;
        default: return 0;
    }
}

Type *scalar_type(LLVMContext &context, dtype_t dtype) {
    switch (dtype) {
        case INT8:
        case UINT8: return Type::getInt8Ty(context);
        case INT16:
        case UINT16: return Type::getInt16Ty(context);
        case INT32:
        case UINT32: return Type::getInt32Ty(context);
        case INT64:
        case UINT64: return Type::getInt64Ty(context);
        case REAL32: return Type::getFloatTy(context);
        case REAL64: return Type::getDoubleTy(context);
        default: return nullptr;
    }
}

Value *tensor_data(IRBuilder<> &builder, StructType *tensor_type, Value *descriptor) {
    Type *pointer = builder.getPtrTy();
    Value *block_address = builder.CreateStructGEP(tensor_type, descriptor, 0);
    Value *block = builder.CreateLoad(pointer, block_address);
    return builder.CreateLoad(pointer, block);
}

Value *array_element(IRBuilder<> &builder, Value *array, uint64 index) {
    return builder.CreateGEP(builder.getPtrTy(), array, builder.getInt64(index));
}

bool same_point_shape(const fused_signature &input, const fused_signature &output) {
    return input.shape == output.shape;
}

Function *build_fused_wrapper(Module &module, const fused_key &key, const fusion_header &header,
                              const fusion_stage *stages, const std::vector<std::string> &point_names,
                              const std::string &symbol) {
    LLVMContext &context = module.getContext();
    StructType *tensor_type = StructType::getTypeByName(context, "struct.tensor");
    Function *prototype = module.getFunction(point_names.front().substr(0, point_names.front().size() - 6));
    Type *value_type = scalar_type(context, key.output.dtype);
    size_t output_bytes = dtype_bytes(key.output.dtype);
    if (!tensor_type || !prototype || !value_type || !output_bytes) return nullptr;

    std::vector<Function *> points;
    extent maximum_inputs = 0;
    extent expected_external = 0;
    for (uint32 i = 0; i < header.count; ++i) {
        Function *point = module.getFunction(point_names[i]);
        if (!point || point->arg_size() != 7 || !point->getReturnType()->isIntegerTy(1) ||
            (i && !stages[i].input_count)) return nullptr;
        point->setLinkage(GlobalValue::InternalLinkage);
        point->addFnAttr(Attribute::AlwaysInline);
        points.push_back(point);
        maximum_inputs = std::max(maximum_inputs, stages[i].input_count);
        expected_external += stages[i].input_count - (i ? 1 : 0);
    }
    if (expected_external != key.inputs.size()) return nullptr;

    Function *wrapper = Function::Create(prototype->getFunctionType(), GlobalValue::ExternalLinkage, symbol, module);
    auto argument = wrapper->arg_begin();
    Value *output = &*argument++;
    Value *external_tensors = &*argument++;
    Value *input_count = &*argument++;
    Value *program = &*argument++;
    Value *program_bytes = &*argument;
    output->setName("output");
    external_tensors->setName("inputs");
    program->setName("fusion.program");

    BasicBlock *entry = BasicBlock::Create(context, "entry", wrapper);
    BasicBlock *setup = BasicBlock::Create(context, "setup", wrapper);
    BasicBlock *loop = BasicBlock::Create(context, "loop", wrapper);
    BasicBlock *done = BasicBlock::Create(context, "done", wrapper);
    BasicBlock *failed = BasicBlock::Create(context, "failed", wrapper);
    IRBuilder<> builder(entry);
    Value *stage_tensors = builder.CreateAlloca(builder.getPtrTy(), builder.getInt64(maximum_inputs));
    Value *stage_inputs = builder.CreateAlloca(builder.getPtrTy(), builder.getInt64(maximum_inputs));
    Value *stage_outputs = builder.CreateAlloca(builder.getPtrTy(), builder.getInt64(1));
    AllocaInst *temporaries[2] = {builder.CreateAlloca(value_type), builder.CreateAlloca(value_type)};
    builder.CreateStore(Constant::getNullValue(value_type), temporaries[0]);
    builder.CreateStore(Constant::getNullValue(value_type), temporaries[1]);
    Value *valid_count = builder.CreateICmpEQ(input_count, builder.getInt64(key.inputs.size()));
    Value *valid_bytes = builder.CreateICmpEQ(program_bytes, builder.getInt64(key.program.size()));
    builder.CreateCondBr(builder.CreateAnd(valid_count, valid_bytes), setup, failed);

    builder.SetInsertPoint(setup);
    Value *size_address = builder.CreateStructGEP(tensor_type, output, 5);
    Value *point_count = builder.CreateLoad(builder.getInt64Ty(), size_address, "point.count");
    builder.CreateCondBr(builder.CreateICmpEQ(point_count, builder.getInt64(0)), done, loop);

    builder.SetInsertPoint(loop);
    PHINode *index = builder.CreatePHI(builder.getInt64Ty(), 2, "point.index");
    index->addIncoming(builder.getInt64(0), setup);
    Value *output_data = tensor_data(builder, tensor_type, output);
    Value *output_offset = builder.CreateMul(index, builder.getInt64(output_bytes));
    Value *final_output = builder.CreateGEP(builder.getInt8Ty(), output_data, output_offset);
    extent external_cursor = 0;
    extent parameter_offset = sizeof(fusion_header) + static_cast<extent>(header.count) * sizeof(fusion_stage);

    for (uint32 stage_index = 0; stage_index < header.count; ++stage_index) {
        extent first_external = stage_index ? 1 : 0;
        if (stage_index) {
            builder.CreateStore(output, array_element(builder, stage_tensors, 0));
            builder.CreateStore(temporaries[(stage_index - 1) & 1u], array_element(builder, stage_inputs, 0));
        }
        for (extent input_index = first_external; input_index < stages[stage_index].input_count; ++input_index) {
            const fused_signature &signature = key.inputs[external_cursor];
            Value *descriptor_address = array_element(builder, external_tensors, external_cursor);
            Value *descriptor = builder.CreateLoad(builder.getPtrTy(), descriptor_address);
            Value *raw = tensor_data(builder, tensor_type, descriptor);
            if (same_point_shape(signature, key.output)) {
                size_t bytes = dtype_bytes(signature.dtype);
                if (!bytes) return nullptr;
                Value *offset = builder.CreateMul(index, builder.getInt64(bytes));
                raw = builder.CreateGEP(builder.getInt8Ty(), raw, offset);
            }
            builder.CreateStore(descriptor, array_element(builder, stage_tensors, input_index));
            builder.CreateStore(raw, array_element(builder, stage_inputs, input_index));
            ++external_cursor;
        }
        Value *stage_output = stage_index + 1 == header.count ? final_output : temporaries[stage_index & 1u];
        builder.CreateStore(stage_output, stage_outputs);
        Value *stage_parameters = builder.CreateGEP(builder.getInt8Ty(), program, builder.getInt64(parameter_offset));
        Value *call_failed = builder.CreateCall(points[stage_index], {
            output, stage_tensors, stage_outputs, stage_inputs, stage_parameters,
            ConstantPointerNull::get(builder.getPtrTy()), builder.getInt64(0)
        });
        BasicBlock *continued = BasicBlock::Create(context, "stage.next", wrapper);
        builder.CreateCondBr(call_failed, failed, continued);
        builder.SetInsertPoint(continued);
        parameter_offset += stages[stage_index].parameter_bytes;
    }
    if (external_cursor != key.inputs.size() || parameter_offset != key.program.size()) return nullptr;
    Value *next = builder.CreateAdd(index, builder.getInt64(1), "point.next");
    builder.CreateCondBr(builder.CreateICmpEQ(next, point_count), done, loop);
    index->addIncoming(next, builder.GetInsertBlock());

    builder.SetInsertPoint(done);
    builder.CreateRet(ConstantInt::getFalse(context));
    builder.SetInsertPoint(failed);
    builder.CreateRet(ConstantInt::getTrue(context));
    return wrapper;
}

jit_ker_t compile_fused(LLJIT *engine, const fused_key &key) {
    const fusion_stage *stages = nullptr;
    const uint8 *parameters = nullptr;
    const fusion_header *header = read_fusion(key, &stages, &parameters);
    (void)parameters;
    if (!header) return nullptr;
    const uint8 *bitcode = nullptr;
    extent bitcode_size = 0;
    std::vector<std::string> point_names;
    point_names.reserve(header->count);
    extent external_cursor = 0;
    for (uint32 i = 0; i < header->count; ++i) {
        if (!stages[i].input_count) return nullptr;
        dtype_t input_dtype = i ? key.output.dtype : key.inputs[external_cursor].dtype;
        const char *name = nullptr;
        const uint8 *stage_bitcode = nullptr;
        extent stage_bitcode_size = 0;
        if (tensor_jit_bitcode_find(stages[i].operation, key.output.dtype, input_dtype, &name, &stage_bitcode,
                                    &stage_bitcode_size) || !name || !stage_bitcode || !stage_bitcode_size) return nullptr;
        if (!bitcode) {
            bitcode = stage_bitcode;
            bitcode_size = stage_bitcode_size;
        } else if (bitcode != stage_bitcode || bitcode_size != stage_bitcode_size) {
            return nullptr;
        }
        point_names.emplace_back(std::string(name) + "_point");
        external_cursor += stages[i].input_count - (i ? 1 : 0);
        if (external_cursor > key.inputs.size()) return nullptr;
    }
    if (external_cursor != key.inputs.size()) return nullptr;

    auto context = std::make_unique<LLVMContext>();
    std::unique_ptr<Module> module = tensor_cpu_parse_module("tensor.fused", bitcode, bitcode_size, *context);
    if (!module) return nullptr;
    uint64 hash = hash_fused_key(key);
    std::string symbol = "tensor_fused_" + std::to_string(hash);
    Function *kernel = build_fused_wrapper(*module, key, *header, stages, point_names, symbol);
    if (!kernel) return nullptr;
    for (const std::string &name : point_names) {
        Function *point = module->getFunction(name);
        if (point) point->setLinkage(GlobalValue::InternalLinkage);
    }
    tensor_cpu_remove_unused_exports(*module, *kernel);
    module->setModuleIdentifier(fused_object_identifier(key));
    module->setDataLayout(engine->getDataLayout());
    module->setTargetTriple(engine->getTargetTriple().str());
    if (verifyModule(*module, &errs())) return nullptr;
    ThreadSafeModule thread_safe_module(std::move(module), std::move(context));
    if (!tensor_cpu_prepare_module(thread_safe_module, symbol)) return nullptr;
    if (Error error = engine->addIRModule(std::move(thread_safe_module))) {
        tensor_cpu_report_error(std::move(error));
        return nullptr;
    }
    auto address = engine->lookup(symbol);
    if (!address) {
        tensor_cpu_report_error(address.takeError());
        return nullptr;
    }
    return address->toPtr<jit_ker_t>();
}

std::shared_ptr<fused_entry> find_fused_entry(const fused_key &key) {
    for (const auto &entry : fused_entries)
        if (key_equal(entry->key, key)) return entry;
    return nullptr;
}

jit_ker_t get_fused_kernel(fused_key key) {
    const fusion_stage *stages = nullptr;
    const uint8 *parameters = nullptr;
    const fusion_header *header = read_fusion(key, &stages, &parameters);
    (void)stages;
    (void)parameters;
    if (!header) return nullptr;
    std::shared_ptr<fused_entry> entry;
    {
        std::unique_lock<std::mutex> lock(fused_lock);
        entry = find_fused_entry(key);
        if (entry) {
            while (entry->state == fused_state::compiling) entry->ready.wait(lock);
            return entry->state == fused_state::ready ? entry->kernel : nullptr;
        }
        entry = std::make_shared<fused_entry>();
        entry->key = std::move(key);
        fused_entries.push_back(entry);
    }
    LLJIT *engine = tensor_cpu_jit_engine();
    jit_ker_t kernel = engine ? compile_fused(engine, entry->key) : nullptr;
    {
        std::lock_guard<std::mutex> lock(fused_lock);
        entry->kernel = kernel;
        entry->state = kernel ? fused_state::ready : fused_state::failed;
    }
    entry->ready.notify_all();
    return kernel;
}



} /* namespace */

extern "C" jit_ker_t tensor_cpu_fused_get(const void *program, extent program_bytes, const tensor *output,
                                            const tensor *const *inputs, extent input_count) {
    fused_key key;
    if (!make_fused_key(key, program, program_bytes, output, inputs, input_count)) return nullptr;
    return get_fused_kernel(std::move(key));
}
