#include "parallel.h"

#include <llvm/Analysis/LoopInfo.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/IntrinsicsNVPTX.h>

using namespace llvm;

namespace {

Value *register_value(IRBuilder<> &builder, Intrinsic::ID id, const Twine &name) {
    Function *intrinsic = Intrinsic::getDeclaration(builder.GetInsertBlock()->getModule(), id);
    return builder.CreateCall(intrinsic, {}, name);
}

Value *thread_index(IRBuilder<> &builder) {
    return register_value(builder, Intrinsic::nvvm_read_ptx_sreg_tid_x, "cuda.thread");
}

Value *block_index(IRBuilder<> &builder) {
    return register_value(builder, Intrinsic::nvvm_read_ptx_sreg_ctaid_x, "cuda.block");
}

Value *block_size(IRBuilder<> &builder) {
    return register_value(builder, Intrinsic::nvvm_read_ptx_sreg_ntid_x, "cuda.block.size");
}

Value *grid_size(IRBuilder<> &builder) {
    return register_value(builder, Intrinsic::nvvm_read_ptx_sreg_nctaid_x, "cuda.grid.size");
}

void block_sync(IRBuilder<> &builder) {
    Function *barrier = Intrinsic::getDeclaration(builder.GetInsertBlock()->getModule(), Intrinsic::nvvm_barrier0);
    builder.CreateCall(barrier);
}

void device_fence(IRBuilder<> &builder) {
    Function *fence = Intrinsic::getDeclaration(builder.GetInsertBlock()->getModule(), Intrinsic::nvvm_membar_gl);
    builder.CreateCall(fence);
}

Loop *work_loop(LoopInfo &loops) {
    Loop *result = nullptr;
    for (Loop *loop : loops) {
        if (result) return nullptr;
        result = loop;
    }
    return result;
}

Value *loop_limit(Loop &loop, PHINode &index, BasicBlock &latch) {
    ICmpInst *compare = loop.getLatchCmpInst();
    if (!compare || compare->getPredicate() != ICmpInst::ICMP_EQ) return nullptr;
    auto *next = dyn_cast_or_null<BinaryOperator>(index.getIncomingValueForBlock(&latch));
    if (!next || next->getOpcode() != Instruction::Add) return nullptr;
    if (compare->getOperand(0) == next) return compare->getOperand(1);
    if (compare->getOperand(1) == next) return compare->getOperand(0);
    return nullptr;
}

} // namespace

Value *tensor_cuda_global_index(IRBuilder<> &builder) {
    Value *block = builder.CreateMul(block_index(builder), block_size(builder), "cuda.block.start");
    return builder.CreateAdd(block, thread_index(builder), "cuda.global.thread");
}

bool tensor_cuda_parallelize(Function &function) {
    DominatorTree dominators(function);
    LoopInfo loops(dominators);
    Loop *loop = work_loop(loops);
    if (!loop || !loop->isLoopSimplifyForm()) return false;
    PHINode *index = loop->getCanonicalInductionVariable();
    BasicBlock *preheader = loop->getLoopPreheader();
    BasicBlock *latch = loop->getLoopLatch();
    ICmpInst *compare = loop->getLatchCmpInst();
    if (!index || !preheader || !latch || !compare || !index->getType()->isIntegerTy(64)) return false;
    Value *limit = loop_limit(*loop, *index, *latch);
    auto *increment = dyn_cast<BinaryOperator>(index->getIncomingValueForBlock(latch));
    if (!limit || !increment) return false;

    IRBuilder<> builder(preheader->getTerminator());
    Value *first = builder.CreateZExt(tensor_cuda_global_index(builder), builder.getInt64Ty());
    Value *blocks = builder.CreateZExt(grid_size(builder), builder.getInt64Ty());
    Value *threads = builder.CreateZExt(block_size(builder), builder.getInt64Ty());
    Value *stride = builder.CreateMul(blocks, threads, "cuda.grid.stride");
    index->setIncomingValueForBlock(preheader, first);
    if (increment->getOperand(0) == index) increment->setOperand(1, stride);
    else if (increment->getOperand(1) == index) increment->setOperand(0, stride);
    else return false;

    BasicBlock *empty = BasicBlock::Create(function.getContext(), "cuda.empty", &function);
    IRBuilder<>(empty).CreateRet(ConstantInt::getFalse(function.getContext()));
    preheader->getTerminator()->eraseFromParent();
    IRBuilder<> guard(preheader);
    guard.CreateCondBr(guard.CreateICmpULT(first, limit), loop->getHeader(), empty);
    return true;
}

void tensor_cuda_grid_sync(IRBuilder<> &builder) {
    Module *module = builder.GetInsertBlock()->getModule();
    LLVMContext &context = module->getContext();
    IntegerType *word = builder.getInt32Ty();
    auto global = [&](StringRef name) {
        GlobalVariable *value = module->getGlobalVariable(name, true);
        if (value) return value;
        return new GlobalVariable(*module, word, false, GlobalValue::InternalLinkage, ConstantInt::get(word, 0), name,
                                  nullptr, GlobalValue::NotThreadLocal, 1);
    };
    GlobalVariable *arrived = global("tensor.cuda.barrier.arrived");
    GlobalVariable *phase = global("tensor.cuda.barrier.phase");
    device_fence(builder);
    block_sync(builder);

    Function *kernel = builder.GetInsertBlock()->getParent();
    BasicBlock *leader = BasicBlock::Create(context, "grid.sync.leader", kernel);
    BasicBlock *wait = BasicBlock::Create(context, "grid.sync.wait", kernel);
    BasicBlock *last = BasicBlock::Create(context, "grid.sync.last", kernel);
    BasicBlock *spin = BasicBlock::Create(context, "grid.sync.spin", kernel);
    BasicBlock *done = BasicBlock::Create(context, "grid.sync.done", kernel);
    builder.CreateCondBr(builder.CreateICmpEQ(thread_index(builder), builder.getInt32(0)), leader, done);

    builder.SetInsertPoint(leader);
    LoadInst *old_phase = builder.CreateLoad(word, phase);
    old_phase->setVolatile(true);
    AtomicRMWInst *ticket = builder.CreateAtomicRMW(AtomicRMWInst::Add, arrived, builder.getInt32(1), MaybeAlign(),
                                                    AtomicOrdering::Monotonic);
    Value *final_ticket = builder.CreateSub(grid_size(builder), builder.getInt32(1));
    builder.CreateCondBr(builder.CreateICmpEQ(ticket, final_ticket), last, wait);

    builder.SetInsertPoint(last);
    StoreInst *reset = builder.CreateStore(builder.getInt32(0), arrived);
    reset->setVolatile(true);
    device_fence(builder);
    builder.CreateAtomicRMW(AtomicRMWInst::Xor, phase, builder.getInt32(1), MaybeAlign(),
                            AtomicOrdering::Monotonic);
    builder.CreateBr(done);

    builder.SetInsertPoint(wait);
    builder.CreateBr(spin);
    builder.SetInsertPoint(spin);
    LoadInst *current = builder.CreateLoad(word, phase);
    current->setVolatile(true);
    builder.CreateCondBr(builder.CreateICmpEQ(current, old_phase), spin, done);

    builder.SetInsertPoint(done);
    block_sync(builder);
}
