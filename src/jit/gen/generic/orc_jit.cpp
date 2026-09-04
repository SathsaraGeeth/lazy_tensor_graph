/*
 * user/src/tensor/src/jit/backend/generic/jit.cpp
 *
 * Copyright (C) 2026 Sathsara Geeth
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
 * 1. Reference: https://llvm.org/docs/ORCv2.html
 * 2. Dont uses the LLVM ORC's laziness here use eagerly
 *    as the library anyway request the compilation 
 *    only if it is sure that it is needed, and it compiles
 *    and put to the cache, little earlier
 * 3. Thread safe, separate contexts are used as recommended.
 *    the CACHE_COMPILING lock based is solves a different
 *    concurrency problem; the same key is requested by multiple threads
 *    so both are needed, not redundant
 * 4. TODO: look at llvm/examples/SpeculativeJIT
 */



#include "tensor_jit.h"
#include "device.h"

#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
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
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

using namespace llvm;
using namespace llvm::orc;


/*
 * Generated at library build time by the op parser.
 */
extern "C" boolean tensor_jit_bitcode_find(
    uint32      op,
    dtype_t     output_dtype,
    dtype_t     input_dtype,
    const char  **name,
    const uint8 **bitcode,
    extent      *bitcode_size);
extern "C" uint64 tensor_device_jit_compatibility_hash(device dev);



namespace {

std::unique_ptr<LLJIT> jit;
std::once_flag target_once;
std::mutex jit_lock;

class PersistentObjectCache final : public ObjectCache {
    std::mutex lock;
    std::unordered_map<std::string, std::string> objects;
public:
    void notifyObjectCompiled(const Module *module, MemoryBufferRef object) override {
        std::lock_guard<std::mutex> guard(lock);
        objects[module->getModuleIdentifier()] = object.getBuffer().str();
    }

    std::unique_ptr<MemoryBuffer> getObject(const Module *) override {
        return nullptr;
    }

    boolean copy(const std::string &identifier, uint8 **object, extent *size) {
        if (!object || !size) return false;
        std::lock_guard<std::mutex> guard(lock);
        auto found = objects.find(identifier);
        if (found == objects.end()) return false;
        *size = found->second.size();
        *object = static_cast<uint8 *>(std::malloc(*size));
        if (!*object) {
            *size = 0;
            return false;
        }
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

    std::fprintf(stderr, "tensor JIT: %s\n", message.c_str());
}


void initialize_target() {
    std::call_once(target_once, [] {
        InitializeNativeTarget();
        InitializeNativeTargetAsmPrinter();
        InitializeNativeTargetAsmParser();
    });
}


std::unique_ptr<LLJIT> create_jit() {
    initialize_target();

    LLJITBuilder builder;

    builder.setNumCompileThreads(std::max(1u, std::thread::hardware_concurrency()));
    builder.setCompileFunctionCreator([](JITTargetMachineBuilder machine) {
        return std::make_unique<ConcurrentIRCompiler>(std::move(machine), &object_cache);
    });
    auto result = builder.create();

    if (!result) {
        report_error(result.takeError());
        return nullptr;
    }

    return std::move(*result);
}


LLJIT *get_jit() {
    if (jit)
        return jit.get();

    std::lock_guard<std::mutex> guard(jit_lock);

    if (!jit)
        jit = create_jit();

    return jit.get();
}


/*
 * Key hashing
 * - hashing is not for the library but
 *   for the LLVM ORC to make unique names
 */

uint64 hash_data(uint64 hash, const void *data, size_t size) {
    const uint8 *bytes = static_cast<const uint8 *>(data);

    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= UINT64_C(1099511628211);
    }

    return hash;
}

uint64 hash_tensor_signature(uint64 hash, const jit_tensor_sig_t *sig) {
    hash = hash_data(hash, &sig->dtype, sizeof(sig->dtype));
    hash = hash_data(hash, &sig->rank, sizeof(sig->rank));

    if (sig->rank)
        hash = hash_data(hash, sig->shape, sig->rank * sizeof(extent));

    return hash;
}

uint64 hash_key(const jit_cache_key_t *key) {
    uint64 hash = UINT64_C(1469598103934665603);

    hash = hash_data(hash, &key->op, sizeof(key->op));
    hash = hash_data(hash, &key->dev, sizeof(key->dev));
    hash = hash_data(hash, &key->compatibility_hash, sizeof(key->compatibility_hash));
    hash = hash_tensor_signature(hash, &key->output);

    hash = hash_data(hash, &key->num_inputs, sizeof(key->num_inputs));

    for (extent i = 0; i < key->num_inputs; ++i)
        hash = hash_tensor_signature(hash, &key->inputs[i]);

    return hash;
}

std::string module_identifier(const jit_cache_key_t *key, const uint8 *bitcode, extent bitcode_size) {
    uint64 revision = 1;
    uint64 hash = hash_data(hash_key(key), bitcode, bitcode_size);
    hash = hash_data(hash, &revision, sizeof(revision));
    return "tensor-jit-" + std::to_string(hash);
}


/*
 * Bitcode lookup
 */

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


/*
 * Module preparation
 */

std::unique_ptr<Module> parse_module(const char *name, const uint8 *bitcode,
                                     extent bitcode_size, LLVMContext &context) {

    MemoryBufferRef buffer(StringRef(reinterpret_cast<const char *>(bitcode), bitcode_size), name);

    auto parsed = parseBitcodeFile(buffer, context);

    if (!parsed) {
        report_error(parsed.takeError());
        return nullptr;
    }

    return std::move(*parsed);
}


Function *find_kernel_function(Module *module, const char *name) {
    if (!module || !name)
        return nullptr;

    return module->getFunction(name);
}


void remove_unused_exports(Module *module, Function *kernel) {
    for (auto iterator = module->begin(); iterator != module->end();) {
        Function &candidate = *iterator++;

        if (&candidate == kernel)
            continue;

        if (candidate.isDeclaration())
            continue;

        if (candidate.hasLocalLinkage())
            continue;

        candidate.eraseFromParent();
    }
}


void configure_module(Module *module, LLJIT *engine) {
    module->setDataLayout(engine->getDataLayout());
    module->setTargetTriple(engine->getTargetTriple().str());
}


/*
 * ORC compilation
 */

jit_ker_t compile_module(LLJIT *engine, const char *name, const uint8 *bitcode, extent bitcode_size,
                         const jit_cache_key_t *key, uint8 **object, extent *object_size) {

    auto context = std::make_unique<LLVMContext>();

    std::unique_ptr<Module> module = parse_module(name, bitcode, bitcode_size, *context);

    if (!module)
        return nullptr;

    Function *kernel = find_kernel_function(module.get(), name);

    if (!kernel)
        return nullptr;

    std::string identifier = module_identifier(key, bitcode, bitcode_size);
    std::string symbol = "tensor_jit_" + std::to_string(hash_key(key));

    remove_unused_exports(module.get(), kernel);

    kernel->setName(symbol);
    module->setModuleIdentifier(identifier);

    configure_module(module.get(), engine);

    ThreadSafeModule thread_safe_module(std::move(module), std::move(context));

    if (Error error = engine->addIRModule(std::move(thread_safe_module))) {
        report_error(std::move(error));
        return nullptr;
    }

    auto address = engine->lookup(symbol);

    if (!address) {
        report_error(address.takeError());
        return nullptr;
    }

    if (!object_cache.copy(identifier, object, object_size))
        return nullptr;

    return address->toPtr<jit_ker_t>();
}

} /* namespace */


/*
 * Generic device JIT ABI
 */

extern "C" jit_ker_t tensor_device_jit_compile(device dev, const jit_cache_key_t *key,
                                                uint8 **object, extent *object_size) {
    if (!key || !object || !object_size) return nullptr;

    *object = nullptr;
    *object_size = 0;

    if (dev != GENERIC && dev != CPU)
        return nullptr;

    LLJIT *engine = get_jit();

    if (!engine)
        return nullptr;

    const char *name     = nullptr;
    const uint8 *bitcode = nullptr;
    extent bitcode_size  = 0;

    if (find_bitcode(key, &name, &bitcode, &bitcode_size))
        return nullptr;

    if (!name || !bitcode || !bitcode_size)
        return nullptr;

    return compile_module(engine, name, bitcode, bitcode_size, key, object, object_size);
}

extern "C" jit_ker_t tensor_device_jit_load(device dev, const jit_cache_key_t *key,
                                             const uint8 *object, extent object_size) {
    LLJIT *engine = get_jit();
    if ((dev != GENERIC && dev != CPU) || !key || key->dev != dev ||
        !engine || !object || !object_size)
        return nullptr;

    auto buffer = MemoryBuffer::getMemBufferCopy(
        StringRef(reinterpret_cast<const char *>(object), object_size));

    if (Error error = engine->addObjectFile(std::move(buffer))) {
        report_error(std::move(error));
        return nullptr;
    }

    auto address = engine->lookup("tensor_jit_" + std::to_string(hash_key(key)));
    if (!address) {
        report_error(address.takeError());
        return nullptr;
    }

    return address->toPtr<jit_ker_t>();
}

extern "C" boolean tensor_device_jit_supported(device dev, const jit_cache_key_t *key) {
    if ((dev != GENERIC && dev != CPU) || !key) return false;
    const char *name = nullptr;
    const uint8 *bitcode = nullptr;
    extent bitcode_size = 0;
    return !find_bitcode(key, &name, &bitcode, &bitcode_size)
           && name && bitcode && bitcode_size;
}

extern "C" uint64 tensor_device_jit_compatibility_hash(device dev) {
    if (dev != GENERIC && dev != CPU)
        return 0;

    static const uint64 hash = [] {
        std::string target = sys::getDefaultTargetTriple();
        target += ":";
        target += sys::getHostCPUName().str();
        target += ":llvm-" LLVM_VERSION_STRING ":tensor-codegen-1";

        uint64 value = UINT64_C(1469598103934665603);
        for (unsigned char byte : target) {
            value ^= byte;
            value *= UINT64_C(1099511628211);
        }
        return value;
    }();
    return hash;
}
