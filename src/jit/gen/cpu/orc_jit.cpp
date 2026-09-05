/*
 * user/src/tensor/src/cpu/orc_jit.cpp
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
 *  1. .op--parser-->SSA scalar(as .bc)->
 *     graph.c prefetcher or serialized file->parallelizer->vectorizer->
 *     this_file->jit_cache
 *  2. Similar to generic/orc_jit.cpp but a little more involved
 */

#include "tensor_jit.h"
#include "device.h"
#include "orc_jit.h"
#include "abstract_parallel/abstract_parallel.h"
#include "abstract_vec/abstract_vec.h"

#include <llvm/ADT/StringMap.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/ExecutionEngine/ObjectCache.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/TargetParser/Host.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

using namespace llvm;
using namespace llvm::orc;


/*
 * Generated at library build time.
 */

extern "C" boolean tensor_jit_bitcode_find(
    uint32 op,
    dtype_t output_dtype,
    dtype_t input_dtype,
    const char **name,
    const uint8 **bitcode,
    extent *bitcode_size
);

extern "C" uint64   tensor_device_jit_compatibility_hash(
    device dev
);
extern "C" boolean tensor_jit_is_pointwise(uint32 operation);
extern "C" uint64 tensor_profile_operation_count(
    uint32 operation, const void *output, const void *const *inputs,
    uint64 input_count, const void *parameters);


namespace {

std::unique_ptr<LLJIT> jit;
std::optional<JITTargetMachineBuilder> target;

std::once_flag target_once;
std::mutex init_lock;

class PersistentObjectCache final : public ObjectCache {
    std::mutex lock;
    std::unordered_map<std::string, std::string> objects;
public:
    void notifyObjectCompiled(const Module *module, MemoryBufferRef object) override {
        std::lock_guard<std::mutex> guard(lock);
        objects[module->getModuleIdentifier()] = object.getBuffer().str();
    }
    std::unique_ptr<MemoryBuffer> getObject(const Module *) override { return nullptr; }
    boolean copy(const std::string &identifier, uint8 **object, extent *size) {
        std::lock_guard<std::mutex> guard(lock);
        auto found = objects.find(identifier);
        if (found == objects.end()) return false;
        *size = found->second.size();
        *object = static_cast<uint8 *>(std::malloc(*size));
        if (!*object) { *size = 0; return false; }
        std::memcpy(*object, found->second.data(), *size);
        return true;
    }
} object_cache;

void report_error(Error error) {
    if (!error)
        return;

    std::string message;
    raw_string_ostream stream(message);

    logAllUnhandledErrors(std::move(error), stream);

    std::fprintf(stderr, "tensor CPU JIT: %s\n", message.c_str());
}

void initialize_llvm_target() {
    std::call_once(target_once, [] {
        InitializeNativeTarget();
        InitializeNativeTargetAsmPrinter();
        InitializeNativeTargetAsmParser();
    });
}

std::unique_ptr<LLJIT> create_jit() {
    initialize_llvm_target();

    auto detected = JITTargetMachineBuilder::detectHost();

    if (!detected) {
        report_error(detected.takeError());
        return nullptr;
    }

    target = std::move(*detected);

    LLJITBuilder builder;

    builder.setJITTargetMachineBuilder(*target);
    builder.setNumCompileThreads(std::max(1u, std::thread::hardware_concurrency()));
    builder.setCompileFunctionCreator([](JITTargetMachineBuilder machine) {
        return std::make_unique<ConcurrentIRCompiler>(std::move(machine), &object_cache);
    });
    auto result = builder.create();

    if (!result) {
        report_error(result.takeError());
        target.reset();
        return nullptr;
    }

    std::unique_ptr<LLJIT> engine = std::move(*result);
    SymbolMap runtime_symbols;

    auto add_runtime_symbol = [&](const char *name, auto address) {
        runtime_symbols[engine->mangleAndIntern(name)] = ExecutorSymbolDef(
            ExecutorAddr::fromPtr(address), JITSymbolFlags::Exported);
    };

    add_runtime_symbol("tensor_cpu_parallel_run", &tensor_cpu_parallel_run);
    add_runtime_symbol("tensor_cpu_parallel_index", &tensor_cpu_parallel_index);
    add_runtime_symbol("tensor_cpu_parallel_count", &tensor_cpu_parallel_count);
    add_runtime_symbol(
        "tensor_profile_operation_count", &tensor_profile_operation_count);

    if (Error error = engine->getMainJITDylib().define(
            absoluteSymbols(std::move(runtime_symbols)))) {
        report_error(std::move(error));
        target.reset();
        return nullptr;
    }

    return engine;
}

LLJIT *get_jit() {
    if (jit)
        return jit.get();

    std::lock_guard<std::mutex> guard(init_lock);

    if (!jit)
        jit = create_jit();

    return jit.get();
}


uint64 hash_bytes(uint64 hash, const void *data, size_t size) {
    const uint8 *bytes = static_cast<const uint8 *>(data);

    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= UINT64_C(1099511628211);
    }

    return hash;
}

uint64 hash_string(uint64 hash, const std::string &value) {
    return hash_bytes(hash, value.data(), value.size());
}

uint64 hash_signature(uint64 hash, const jit_tensor_sig_t *sig) {
    hash = hash_bytes(hash, &sig->dtype, sizeof(sig->dtype));
    hash = hash_bytes(hash, &sig->rank, sizeof(sig->rank));

    if (sig->rank)
        hash = hash_bytes(hash, sig->shape, sig->rank * sizeof(extent));

    return hash;
}

uint64 hash_key(const jit_cache_key_t *key) {
    uint64 hash = UINT64_C(1469598103934665603);

    hash = hash_bytes(hash, &key->op, sizeof(key->op));
    hash = hash_bytes(hash, &key->dev, sizeof(key->dev));
    hash = hash_bytes(hash, &key->compatibility_hash, sizeof(key->compatibility_hash));
    hash = hash_signature(hash, &key->output);
    hash = hash_bytes(hash, &key->num_inputs, sizeof(key->num_inputs));

    for (extent i = 0; i < key->num_inputs; ++i)
        hash = hash_signature(hash, &key->inputs[i]);

    return hash;
}

std::string module_identifier(const jit_cache_key_t *key, const uint8 *bitcode, extent bitcode_size) {
    uint64 revision = 4;
    uint64 hash = hash_bytes(hash_key(key), bitcode, bitcode_size);
    hash = hash_bytes(hash, &revision, sizeof(revision));
    return "tensor-jit-" + std::to_string(hash);
}

boolean find_bitcode(const jit_cache_key_t *key, const char **name,
                     const uint8 **bitcode, extent *bitcode_size) {

    dtype_t input_dtype = key->num_inputs ? key->inputs[0].dtype : key->output.dtype;

    return tensor_jit_bitcode_find(
        key->op,
        key->output.dtype,
        input_dtype,
        name,
        bitcode,
        bitcode_size);
}


std::unique_ptr<Module> parse_module(
    const char *name,
    const uint8 *bitcode,
    extent bitcode_size,
    LLVMContext &context) {

    MemoryBufferRef buffer(
        StringRef(reinterpret_cast<const char *>(bitcode), bitcode_size),
        name);

    auto parsed = parseBitcodeFile(buffer, context);

    if (!parsed) {
        report_error(parsed.takeError());
        return nullptr;
    }

    return std::move(*parsed);
}


void remove_unused_exports(Module &module, Function &kernel) {
    for (auto iterator = module.begin(); iterator != module.end();) {
        Function &candidate = *iterator++;

        if (&candidate == &kernel)
            continue;

        if (candidate.isDeclaration())
            continue;

        if (candidate.hasLocalLinkage())
            continue;

        candidate.eraseFromParent();
    }
}

boolean prepare_module(
    ThreadSafeModule &module,
    const std::string &symbol,
    uint32 operation) {

    if (tensor_jit_is_pointwise(operation))
        tensor_cpu_abstract_parallelize(module, symbol, operation);

    if (!target)
        return false;

    return tensor_cpu_abstract_vectorize(module, *target);
}


jit_ker_t compile_bitcode(
    LLJIT *engine,
    const char *name,
    const uint8 *bitcode,
    extent bitcode_size,
    const jit_cache_key_t *key, uint8 **object, extent *object_size) {

    auto context = std::make_unique<LLVMContext>();

    std::unique_ptr<Module> module = parse_module(name, bitcode, bitcode_size, *context);

    if (!module)
        return nullptr;

    Function *kernel = module->getFunction(name);

    if (!kernel)
        return nullptr;

    std::string identifier = module_identifier(key, bitcode, bitcode_size);
    std::string symbol = "tensor_jit_" + std::to_string(hash_key(key));

    remove_unused_exports(*module, *kernel);

    kernel->setName(symbol);
    module->setModuleIdentifier(identifier);

    module->setDataLayout(engine->getDataLayout());
    module->setTargetTriple(engine->getTargetTriple().str());

    ThreadSafeModule thread_safe_module(std::move(module), std::move(context));

    if (!prepare_module(thread_safe_module, symbol, key->op))
        return nullptr;

    if (Error error = engine->addIRModule(std::move(thread_safe_module))) {
        report_error(std::move(error));
        return nullptr;
    }

    auto address = engine->lookup(symbol);

    if (!address) {
        report_error(address.takeError());
        return nullptr;
    }

    if (!object_cache.copy(identifier, object, object_size)) return nullptr;
    return address->toPtr<jit_ker_t>();
}

std::string compatibility_string() {
    std::string value = sys::getDefaultTargetTriple();

    value += ":";
    value += sys::getHostCPUName().str();
    value += ":llvm-" LLVM_VERSION_STRING ":tensor-codegen-12";

    StringMap<bool> features;

    if (sys::getHostCPUFeatures(features)) {
        for (const auto &feature : features) {
            if (!feature.second)
                continue;

            value += ":";
            value += feature.first().str();
        }
    }

    return value;
}

} /* namespace */


LLJIT *tensor_cpu_jit_engine() {
    return get_jit();
}

uint64 tensor_cpu_hash_bytes(uint64 hash, const void *data, size_t size) {
    return hash_bytes(hash, data, size);
}

std::unique_ptr<Module> tensor_cpu_parse_module(const char *name, const uint8 *bitcode, extent bitcode_size,
                                                LLVMContext &context) {
    return parse_module(name, bitcode, bitcode_size, context);
}

void tensor_cpu_remove_unused_exports(Module &module, Function &kernel) {
    remove_unused_exports(module, kernel);
}

bool tensor_cpu_prepare_module(ThreadSafeModule &module, const std::string &symbol, uint32 operation) {
    return prepare_module(module, symbol, operation);
}

void tensor_cpu_report_error(Error error) {
    report_error(std::move(error));
}

/*
 * CPU backend ABI
 */

extern "C" jit_ker_t tensor_device_jit_compile(device dev, const jit_cache_key_t *key,
                                                uint8 **object, extent *object_size) {
    if (!key) return nullptr;

    if (dev != CPU)
        return nullptr;

    if (key->dev != CPU)
        return nullptr;

    if (key->compatibility_hash != tensor_device_jit_compatibility_hash(dev))
        return nullptr;

    LLJIT *engine = get_jit();

    if (!engine || !target)
        return nullptr;

    const char *name = nullptr;
    const uint8 *bitcode = nullptr;
    extent bitcode_size = 0;

    if (find_bitcode(key, &name, &bitcode, &bitcode_size))
        return nullptr;

    if (!name || !bitcode || !bitcode_size)
        return nullptr;

    return compile_bitcode(engine, name, bitcode, bitcode_size, key, object, object_size);
}

extern "C" jit_ker_t tensor_device_jit_load(device dev, const jit_cache_key_t *key,
                                             const uint8 *object, extent object_size) {
    LLJIT *engine = get_jit();
    if (dev != CPU || !key || key->dev != CPU || !engine || !object || !object_size) return nullptr;
    auto buffer = MemoryBuffer::getMemBufferCopy(
        StringRef(reinterpret_cast<const char *>(object), object_size));
    if (Error error = engine->addObjectFile(std::move(buffer))) {
        report_error(std::move(error));
        return nullptr;
    }
    auto address = engine->lookup("tensor_jit_" + std::to_string(hash_key(key)));
    if (!address) { report_error(address.takeError()); return nullptr; }
    return address->toPtr<jit_ker_t>();
}

extern "C" boolean tensor_device_jit_supported(device dev, const jit_cache_key_t *key) {
    if (dev != CPU || !key || key->dev != CPU) return false;
    const char *name = nullptr;
    const uint8 *bitcode = nullptr;
    extent bitcode_size = 0;
    return !find_bitcode(key, &name, &bitcode, &bitcode_size)
           && name && bitcode && bitcode_size;
}

extern "C" uint64 tensor_device_jit_compatibility_hash(device dev) {
    if (dev != CPU)
        return 0;

    static const uint64 hash = hash_string(
        UINT64_C(1469598103934665603), compatibility_string());
    return hash;
}
