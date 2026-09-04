#include "ptx.h"
#include "parallel.h"
#include "tensor_jit.h"

#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>

using namespace llvm;

extern "C" boolean tensor_jit_bitcode_find(uint32 op, dtype_t output_dtype, dtype_t input_dtype,
                                            const char **name, const uint8 **bitcode, extent *bitcode_size);

namespace {

Value *load_pointer(IRBuilder<> &builder, Value *array, uint64 index) {
    Value *address = builder.CreateGEP(builder.getPtrTy(), array, builder.getInt64(index));
    return builder.CreateLoad(builder.getPtrTy(), address);
}

Function *build_kernel(Module &module, const vtensor *const *nodes, extent count, const std::string &name) {
    LLVMContext &context = module.getContext();
    Type *pointer = PointerType::getUnqual(context);
    FunctionType *type = FunctionType::get(Type::getVoidTy(context),
                                           {pointer, pointer, pointer, pointer, pointer, pointer}, false);
    Function *kernel = Function::Create(type, GlobalValue::ExternalLinkage, name, module);
    kernel->setCallingConv(CallingConv::PTX_Kernel);
    auto argument = kernel->arg_begin();
    Value *outputs = &*argument++;
    Value *inputs = &*argument++;
    Value *counts = &*argument++;
    Value *parameters = &*argument++;
    Value *parameter_bytes = &*argument++;
    Value *status = &*argument;
    BasicBlock *entry = BasicBlock::Create(context, "entry", kernel);
    IRBuilder<> builder(entry);
    std::unordered_map<Function *, boolean> lowered;
    for (extent i = 0; i < count; ++i) {
        const vtensor *node = nodes[i];
        dtype_t input_dtype = node->num_parents ? node->parents[0]->dtype : node->dtype;
        const char *operation_name = nullptr;
        const uint8 *unused = nullptr;
        extent unused_size = 0;
        if (tensor_jit_bitcode_find(node->edge->op.op, node->dtype, input_dtype, &operation_name, &unused,
                                    &unused_size) || !operation_name) return nullptr;
        Function *operation = module.getFunction(operation_name);
        if (!operation || operation->arg_size() != 5) return nullptr;
        operation->setLinkage(GlobalValue::InternalLinkage);
        operation->addFnAttr(Attribute::AlwaysInline);
        auto found = lowered.find(operation);
        boolean parallel = found == lowered.end() ? tensor_cuda_parallelize(*operation) : found->second;
        lowered.emplace(operation, parallel);
        Value *input_count_address = builder.CreateGEP(builder.getInt64Ty(), counts, builder.getInt64(i));
        Value *parameter_bytes_address = builder.CreateGEP(builder.getInt64Ty(), parameter_bytes, builder.getInt64(i));
        SmallVector<Value *, 5> arguments = {load_pointer(builder, outputs, i), load_pointer(builder, inputs, i),
                                             builder.CreateLoad(builder.getInt64Ty(), input_count_address),
                                             load_pointer(builder, parameters, i),
                                             builder.CreateLoad(builder.getInt64Ty(), parameter_bytes_address)};
        Value *result = nullptr;
        if (parallel) {
            result = builder.CreateCall(operation, arguments);
        } else {
            BasicBlock *serial = BasicBlock::Create(context, "stage.serial", kernel);
            BasicBlock *joined = BasicBlock::Create(context, "stage.join", kernel);
            BasicBlock *prior = builder.GetInsertBlock();
            builder.CreateCondBr(builder.CreateICmpEQ(tensor_cuda_global_index(builder), builder.getInt32(0)), serial,
                                 joined);
            builder.SetInsertPoint(serial);
            Value *serial_result = builder.CreateCall(operation, arguments);
            builder.CreateBr(joined);
            builder.SetInsertPoint(joined);
            PHINode *selected = builder.CreatePHI(builder.getInt1Ty(), 2);
            selected->addIncoming(builder.getFalse(), prior);
            selected->addIncoming(serial_result, serial);
            result = selected;
        }
        builder.CreateAtomicRMW(AtomicRMWInst::Or, status, builder.CreateZExt(result, builder.getInt32Ty()), MaybeAlign(),
                                AtomicOrdering::Monotonic);
        if (i + 1 < count) tensor_cuda_grid_sync(builder);
    }
    builder.CreateRetVoid();
    NamedMDNode *annotations = module.getOrInsertNamedMetadata("nvvm.annotations");
    annotations->addOperand(MDNode::get(context, {ValueAsMetadata::get(kernel), MDString::get(context, "kernel"),
                                                   ConstantAsMetadata::get(builder.getInt32(1))}));
    return kernel;
}

std::unique_ptr<Module> parse_operations(LLVMContext &context, const vtensor *node) {
    const char *name = nullptr;
    const uint8 *bitcode = nullptr;
    extent bitcode_size = 0;
    dtype_t input_dtype = node->num_parents ? node->parents[0]->dtype : node->dtype;
    if (tensor_jit_bitcode_find(node->edge->op.op, node->dtype, input_dtype, &name, &bitcode, &bitcode_size) ||
        !bitcode || !bitcode_size) return nullptr;
    StringRef bytes(reinterpret_cast<const char *>(bitcode), bitcode_size);
    auto parsed = parseBitcodeFile(MemoryBufferRef(bytes, "tensor.cuda.subgraph"), context);
    if (!parsed) {
        consumeError(parsed.takeError());
        return nullptr;
    }
    return std::move(*parsed);
}

} // namespace

extern "C" boolean tensor_cuda_ptx(const vtensor *const *nodes, extent count, char **ptx, char **kernel_name) {
    if (!nodes || !count || !ptx || !kernel_name) return true;
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();
    LLVMContext context;
    std::unique_ptr<Module> module = parse_operations(context, nodes[0]);
    if (!module) return true;
    std::string symbol = "tensor_cuda_subgraph";
    if (!build_kernel(*module, nodes, count, symbol)) return true;
    module->setTargetTriple("nvptx64-nvidia-cuda");
    std::string error;
    const Target *target = TargetRegistry::lookupTarget(module->getTargetTriple(), error);
    if (!target) return true;
    TargetOptions options;
    std::unique_ptr<TargetMachine> machine(target->createTargetMachine(module->getTargetTriple(), "sm_52", "+ptx60",
                                                                       options, Reloc::PIC_));
    if (!machine) return true;
    module->setDataLayout(machine->createDataLayout());
    SmallVector<char, 0> assembly;
    raw_svector_ostream stream(assembly);
    legacy::PassManager passes;
    if (machine->addPassesToEmitFile(passes, stream, nullptr, CodeGenFileType::AssemblyFile)) return true;
    passes.run(*module);
    *ptx = static_cast<char *>(std::malloc(assembly.size() + 1));
    *kernel_name = static_cast<char *>(std::malloc(symbol.size() + 1));
    if (!*ptx || !*kernel_name) {
        std::free(*ptx);
        std::free(*kernel_name);
        return true;
    }
    std::memcpy(*ptx, assembly.data(), assembly.size());
    (*ptx)[assembly.size()] = '\0';
    std::memcpy(*kernel_name, symbol.c_str(), symbol.size() + 1);
    return false;
}

extern "C" void tensor_cuda_ptx_free(char *text) {
    std::free(text);
}
