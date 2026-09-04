/*
 * user/src/tensor/src/jit/gen/cpu/abstract_parallel/abstract_parallel.cpp
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
 * 1. Reference: https://llvm.org/doxygen/classllvm_1_1Loop.html
 * 2. TODO: look at here 
 *    https://llvm.org/devmtg/2018-04/slides/Finkel-Representing%20Parallelism%20Within%20LLVM.pdf
 * 3. Here currently the llvm production autovectorization is used
 *    - loop_vecotrizer: widen the loop
 *    - SLP vectorizer:  combine similar independent scalar instructions
 * 4. The kernel generator is expected to has one outer loop whose 
 *    iterations are independent
 *    then this partition that loop across the thread pool and
 *    modify the original kernel into a worker function (and rename)
 *    then use the original name for the dispatcher of the worker dispatcher
 *    e.g.
 *           bool matmul(...) {
 *              for (i = 0; i < M; ++i)
 *                   ...
 *           }
 *           into->
 *           bool matmul_worker(...) {
 *               start = M * worker_id / worker_count;
 *               end   = M * (worker_id + 1) / worker_count;
 *
 *               for (i = start; i < end; ++i)
 *                    ...
 *            }
 *           bool matmul(...) {
 *              return parallel_run(matmul_worker, ...);
 *           }
 * 
 * 5. .op--parser-->SSA scalar(as .bc)->
 *     graph.c prefethcer or serilized file->this_file->vectorizer->orc_jit.cpp->jit_cache
 * 6. The order matter parrallize->vectorize
 * 7. This make a thread pool one subgraph's work queue
 *    so thread init dont happen per kernel calls
 */

#include "abstract_parallel.h"

#include <llvm/Analysis/LoopInfo.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Metadata.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Transforms/Utils/LCSSA.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/LoopSimplify.h>

#include <pthread.h>
#include <sched.h>

#include <condition_variable>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <cstdio>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <time.h>

using namespace llvm;
using namespace llvm::orc;

static uint64_t tensor_probe_time_ns(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC_RAW, &now);
    return static_cast<uint64_t>(now.tv_sec) * UINT64_C(1000000000) + now.tv_nsec;
}

namespace {

/*
 * JIT kernel ABI.
 */
using kernel_t = bool (*)(
    void *output,
    const void *const *inputs,
    uint64_t input_count,
    const void *parameters,
    uint64_t parameter_bytes);

extern "C" uint64_t tensor_profile_operation_count(
    uint32_t operation, const void *output, const void *const *inputs,
    uint64_t input_count, const void *parameters);


struct pool_job {
    kernel_t kernel{};

    void *output{};
    const void *const *inputs{};
    uint64_t input_count{};

    const void *parameters{};
    uint64_t parameter_bytes{};

    uint64_t generation{};

    unsigned remaining{};

    bool failed{};
    bool stopping{};
};


struct worker_arg {
    unsigned index{};
};


std::mutex pool_lock;

std::mutex dispatch_lock;

std::condition_variable work_ready;
std::condition_variable work_done;

pool_job job;

std::vector<pthread_t> workers;
std::vector<worker_arg> worker_args;
std::once_flag pool_once;
std::atomic<uint64_t> job_math_max;

thread_local unsigned worker_index = 0;
thread_local unsigned worker_count = 1;


void *worker_main(void *opaque) {
    const unsigned index = static_cast<worker_arg *>(opaque)->index;

    uint64_t seen_generation = 0;

    std::unique_lock<std::mutex> lock(pool_lock);

    for (;;) {
        work_ready.wait(lock, [&] {
            return job.stopping || job.generation != seen_generation;
        });

        if (job.stopping)
            return nullptr;

        seen_generation = job.generation;

        pool_job local = job;

        lock.unlock();

        worker_index = index;
        worker_count = static_cast<unsigned>(workers.size() + 1);

        const uint64_t math_started = [] {
            struct timespec now;
            clock_gettime(CLOCK_MONOTONIC_RAW, &now);
            return static_cast<uint64_t>(now.tv_sec) * UINT64_C(1000000000) + now.tv_nsec;
        }();
        const bool failed = local.kernel(
            local.output,
            local.inputs,
            local.input_count,
            local.parameters,
            local.parameter_bytes);
        const uint64_t math_finished = [] {
            struct timespec now;
            clock_gettime(CLOCK_MONOTONIC_RAW, &now);
            return static_cast<uint64_t>(now.tv_sec) * UINT64_C(1000000000) + now.tv_nsec;
        }();
        const uint64_t math_time = math_finished - math_started;
        uint64_t maximum = job_math_max.load(std::memory_order_relaxed);
        while (maximum < math_time &&
               !job_math_max.compare_exchange_weak(
                   maximum, math_time, std::memory_order_relaxed)) {}

        lock.lock();

        job.failed |= failed;

        if (--job.remaining == 0)
            work_done.notify_one();
    }
}


unsigned environment_worker_count() {
    const char *value = std::getenv("TENSOR_NUM_WORKERS");

    if (!value || !*value)
        return 0;

    char *end = nullptr;

    const unsigned long requested = std::strtoul(value, &end, 10);

    if (end == value || *end || !requested)
        return 0;

    return static_cast<unsigned>(requested);
}


unsigned physical_core_count() {
    cpu_set_t available;

    CPU_ZERO(&available);

    if (sched_getaffinity(0, sizeof(available), &available))
        return 1;

    std::set<std::pair<int, int>> physical_cores;

    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &available))
            continue;

        const std::string topology =
            "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";

        std::ifstream package_file(topology + "physical_package_id");
        std::ifstream core_file(topology + "core_id");

        int package = -1;
        int core = -1;

        if (package_file >> package && core_file >> core)
            physical_cores.emplace(package, core);
    }

    if (physical_cores.empty())
        return 1;

    return static_cast<unsigned>(physical_cores.size());
}


unsigned configured_worker_count() {
    const unsigned configured = environment_worker_count();

    return configured ? configured : physical_core_count();
}

uint64_t minimum_operations_per_worker() {
    static const uint64_t minimum = [] {
        constexpr uint64_t default_minimum = UINT64_C(16384);
        const char *value = std::getenv("TENSOR_MIN_OPS_PER_WORKER");

        if (!value || !*value)
            return default_minimum;

        char *end = nullptr;
        const unsigned long long requested = std::strtoull(value, &end, 10);

        return end != value && !*end ? static_cast<uint64_t>(requested)
                                    : default_minimum;
    }();
    return minimum;
}


void shutdown_pool() {
    {
        std::lock_guard<std::mutex> lock(pool_lock);

        job.stopping = true;
        ++job.generation;
    }

    work_ready.notify_all();

    for (pthread_t worker : workers)
        pthread_join(worker, nullptr);

    workers.clear();
    worker_args.clear();
}


void initialize_pool() {
    const unsigned requested = configured_worker_count();

    const unsigned background = requested > 1 ? requested - 1 : 0;
    workers.reserve(background);
    worker_args.resize(background);

    for (unsigned i = 0; i < background; ++i) {
        worker_args[i].index = i + 1;

        pthread_t worker;

        if (pthread_create(&worker, nullptr, worker_main, &worker_args[i]))
            break;

        workers.push_back(worker);
    }


    if (!workers.empty())
        std::atexit(shutdown_pool);
}


bool parallel_run_impl(kernel_t kernel, void *output, const void *const *inputs,
                       uint64_t input_count, const void *parameters,
                       uint64_t parameter_bytes, uint64_t operations) {
    const uint64_t started = tensor_probe_time_ns();
    tensor_cpu_pool_begin();
    const uint64_t pool_ready = tensor_probe_time_ns();

    const uint64_t minimum = minimum_operations_per_worker();
    const uint64_t active_workers = workers.size() + 1;
    const bool below_threshold = minimum &&
        (active_workers > UINT64_MAX / minimum || operations < minimum * active_workers);
    if (active_workers <= 1 || below_threshold) {
        worker_index = 0;
        worker_count = 1;

        const bool failed = kernel(
            output,
            inputs,
            input_count,
            parameters,
            parameter_bytes);
        if (std::getenv("TENSOR_TIMING_PROBES"))
            std::fprintf(stderr, "[tensor probe] parallel_run serial pool=%.3f us total=%.3f us\n",
                         (pool_ready - started) / 1000.0,
                         (tensor_probe_time_ns() - started) / 1000.0);
        return failed;
    }

    const uint64_t before_lock = tensor_probe_time_ns();
    std::lock_guard<std::mutex> dispatch(dispatch_lock);

    std::unique_lock<std::mutex> lock(pool_lock);
    const uint64_t locked = tensor_probe_time_ns();

    job.kernel = kernel;

    job.output = output;
    job.inputs = inputs;
    job.input_count = input_count;

    job.parameters = parameters;
    job.parameter_bytes = parameter_bytes;

    job.failed = false;
    job.remaining = static_cast<unsigned>(workers.size());
    job_math_max.store(0, std::memory_order_relaxed);

    ++job.generation;

    const uint64_t published = tensor_probe_time_ns();
    work_ready.notify_all();

    lock.unlock();

    worker_index = 0;
    worker_count = static_cast<unsigned>(active_workers);
    const uint64_t caller_math_started = tensor_probe_time_ns();
    const bool caller_failed = kernel(
        output, inputs, input_count, parameters, parameter_bytes);
    const uint64_t caller_math_time = tensor_probe_time_ns() - caller_math_started;
    uint64_t maximum = job_math_max.load(std::memory_order_relaxed);
    while (maximum < caller_math_time &&
           !job_math_max.compare_exchange_weak(
               maximum, caller_math_time, std::memory_order_relaxed)) {}

    lock.lock();
    job.failed |= caller_failed;

    if (job.remaining)
        work_done.wait(lock, [] { return job.remaining == 0; });

    const uint64_t completed = tensor_probe_time_ns();
    static std::atomic<unsigned> dispatch_samples;
    const unsigned sample = dispatch_samples.fetch_add(1, std::memory_order_relaxed);
    if (std::getenv("TENSOR_TIMING_PROBES") && sample < 32)
        std::fprintf(stderr,
                     "[tensor probe] parallel_run pool=%.3f lock=%.3f publish=%.3f math_wall=%.3f coordination=%.3f total=%.3f us\n",
                     (pool_ready - started) / 1000.0,
                     (locked - before_lock) / 1000.0,
                     (published - locked) / 1000.0,
                     job_math_max.load(std::memory_order_relaxed) / 1000.0,
                     ((completed - published) - job_math_max.load(std::memory_order_relaxed)) / 1000.0,
                     (completed - started) / 1000.0);

    return job.failed;
}

extern "C" void tensor_cpu_pool_begin(void) {
    std::call_once(pool_once, initialize_pool);
}

Loop *find_work_loop(LoopInfo &loops) {
    Loop *work_loop = nullptr;

    for (Loop *loop : loops) {

        if (work_loop)
            return nullptr;

        work_loop = loop;
    }

    return work_loop;
}


bool eligible_loop(Loop *loop) {
    if (!loop)
        return false;

    if (!loop->isLoopSimplifyForm())
        return false;

    if (!loop->getCanonicalInductionVariable())
        return false;

    if (!loop->getLoopPreheader())
        return false;

    if (!loop->getLoopLatch())
        return false;

    if (!loop->getLatchCmpInst())
        return false;

    return true;
}


Value *loop_limit(
    Loop *loop,
    PHINode *index,
    BasicBlock *latch) {

    ICmpInst *compare = loop->getLatchCmpInst();

    if (!compare || compare->getPredicate() != ICmpInst::ICMP_EQ)
        return nullptr;

    Value *increment = index->getIncomingValueForBlock(latch);

    auto *next = dyn_cast_or_null<BinaryOperator>(increment);

    if (!next || next->getOpcode() != Instruction::Add)
        return nullptr;

    if (compare->getOperand(0) == next)
        return compare->getOperand(1);

    if (compare->getOperand(1) == next)
        return compare->getOperand(0);

    return nullptr;
}


void replace_loop_limit(
    ICmpInst *compare,
    Value *old_limit,
    Value *new_limit) {

    if (compare->getOperand(0) == old_limit)
        compare->setOperand(0, new_limit);
    else
        compare->setOperand(1, new_limit);
}


bool partition_outer_loop(Function &function) {
    DominatorTree dominators(function);
    LoopInfo loops(dominators);

    Loop *loop = find_work_loop(loops);

    if (!eligible_loop(loop))
        return false;

    PHINode *index = loop->getCanonicalInductionVariable();
    BasicBlock *preheader = loop->getLoopPreheader();
    BasicBlock *latch = loop->getLoopLatch();
    ICmpInst *compare = loop->getLatchCmpInst();

    LLVMContext &context = function.getContext();
    Metadata *vectorize[] = {
        nullptr,
        MDNode::get(context, {
            MDString::get(context, "llvm.loop.vectorize.enable"),
            ConstantAsMetadata::get(ConstantInt::getTrue(context))
        })
    };
    MDNode *loop_id = MDNode::getDistinct(context, vectorize);
    loop_id->replaceOperandWith(0, loop_id);
    latch->getTerminator()->setMetadata(LLVMContext::MD_loop, loop_id);

    Value *limit = loop_limit(loop, index, latch);

    if (!limit || !limit->getType()->isIntegerTy(64))
        return false;


    /*
     * Partition [0, limit);
     *
     *     start = limit * worker_id       / worker_count
     *     end   = limit * (worker_id + 1) / worker_count
     */
    IRBuilder<> builder(preheader->getTerminator());

    FunctionType *query_type = FunctionType::get(
        builder.getInt64Ty(),
        false);

    Module *module = function.getParent();
    FunctionCallee index_query = module->getOrInsertFunction(
        "tensor_cpu_parallel_index", query_type);
    FunctionCallee count_query = module->getOrInsertFunction(
        "tensor_cpu_parallel_count", query_type);

    Value *thread = builder.CreateCall(
        index_query, {}, "parallel.thread");

    Value *count = builder.CreateCall(
        count_query, {}, "parallel.count");

    Value *start = builder.CreateUDiv(
        builder.CreateMul(limit, thread),
        count,
        "parallel.start");

    Value *next_thread = builder.CreateAdd(
        thread,
        builder.getInt64(1));

    Value *end = builder.CreateUDiv(
        builder.CreateMul(limit, next_thread),
        count,
        "parallel.end");


    index->setIncomingValueForBlock(preheader, start);


    replace_loop_limit(compare, limit, end);


    /*
     * worker_count > iteration_count
     * skip the loop for those workers
     */
    BasicBlock *header = loop->getHeader();

    BasicBlock *empty = BasicBlock::Create(
        function.getContext(),
        "parallel.empty",
        &function);

    IRBuilder<> empty_builder(empty);

    empty_builder.CreateRet(
        ConstantInt::getFalse(function.getContext()));


    preheader->getTerminator()->eraseFromParent();

    IRBuilder<> preheader_builder(preheader);

    Value *has_work = preheader_builder.CreateICmpULT(
        start,
        end,
        "parallel.has_work");

    preheader_builder.CreateCondBr(
        has_work,
        header,
        empty);

    return true;
}


bool valid_kernel_abi(Function &kernel) {
    if (kernel.arg_size() != 5)
        return false;

    if (!kernel.getReturnType()->isIntegerTy(1))
        return false;

    return true;
}


bool make_wrapper(Module &module, Function &worker, Function &serial,
                  uint32_t operation) {

    if (!valid_kernel_abi(worker))
        return false;

    const std::string kernel_name = worker.getName().str();

    worker.setName(kernel_name + ".worker");


    Function *wrapper = Function::Create(
        worker.getFunctionType(),
        worker.getLinkage(),
        kernel_name,
        module);

    wrapper->copyAttributesFrom(&worker);


    BasicBlock *entry = BasicBlock::Create(
        module.getContext(),
        "entry",
        wrapper);

    IRBuilder<> builder(entry);


    SmallVector<Value *, 5> arguments;

    for (Argument &argument : wrapper->args())
        arguments.push_back(&argument);


    SmallVector<Value *, 7> call_arguments;

    call_arguments.push_back(&worker);
    call_arguments.append(arguments);

    FunctionType *operation_count_type = FunctionType::get(
        builder.getInt64Ty(),
        {
            builder.getInt32Ty(),
            arguments[0]->getType(),
            arguments[1]->getType(),
            builder.getInt64Ty(),
            arguments[3]->getType()
        },
        false);
    FunctionCallee operation_count = module.getOrInsertFunction(
        "tensor_profile_operation_count", operation_count_type);
    Value *operations = builder.CreateCall(
        operation_count,
        {
            builder.getInt32(operation),
            arguments[0],
            arguments[1],
            arguments[2],
            arguments[3]
        },
        "algorithm.operations");
    call_arguments.push_back(operations);

    BasicBlock *serial_block = BasicBlock::Create(
        module.getContext(), "serial", wrapper);
    BasicBlock *parallel_block = BasicBlock::Create(
        module.getContext(), "parallel", wrapper);
    BasicBlock *exit = BasicBlock::Create(
        module.getContext(), "exit", wrapper);

    constexpr uint64_t direct_threshold = UINT64_C(131072);
    Value *small = builder.CreateICmpULT(
        operations, builder.getInt64(direct_threshold), "algorithm.small");
    builder.CreateCondBr(small, serial_block, parallel_block);

    IRBuilder<> serial_builder(serial_block);
    Value *serial_result = serial_builder.CreateCall(&serial, arguments);
    serial_builder.CreateBr(exit);


    FunctionType *runner_type = FunctionType::get(
        builder.getInt1Ty(),
        {
            worker.getType(),
            arguments[0]->getType(),
            arguments[1]->getType(),
            builder.getInt64Ty(),
            arguments[3]->getType(),
            builder.getInt64Ty(),
            builder.getInt64Ty()
        },
        false);


    FunctionCallee runner = module.getOrInsertFunction(
        "tensor_cpu_parallel_run", runner_type);
    IRBuilder<> parallel_builder(parallel_block);
    Value *parallel_result = parallel_builder.CreateCall(runner, call_arguments);
    parallel_builder.CreateBr(exit);

    IRBuilder<> exit_builder(exit);
    PHINode *result = exit_builder.CreatePHI(builder.getInt1Ty(), 2);
    result->addIncoming(serial_result, serial_block);
    result->addIncoming(parallel_result, parallel_block);
    exit_builder.CreateRet(result);

    return true;
}

} /* namespace */


extern "C" bool tensor_cpu_parallel_run(
    tensor_cpu_parallel_kernel_t kernel,
    void *output,
    const void *const *inputs,
    uint64_t input_count,
    const void *parameters,
    uint64_t parameter_bytes,
    uint64_t operations) {

    return parallel_run_impl(
        kernel, output, inputs, input_count, parameters,
        parameter_bytes, operations);
}


extern "C" uint64_t tensor_cpu_parallel_index(void) {
    return worker_index;
}


extern "C" uint64_t tensor_cpu_parallel_count(void) {
    return worker_count;
}


bool tensor_cpu_abstract_parallelize(
    ThreadSafeModule &thread_safe_module,
    const std::string &kernel_name,
    uint32_t operation) {

    bool changed = false;

    thread_safe_module.withModuleDo([&](Module &module) {
        Function *kernel = module.getFunction(kernel_name);

        if (!kernel)
            return;

        PassBuilder passes;
        FunctionAnalysisManager analyses;
        passes.registerFunctionAnalyses(analyses);
        FunctionPassManager canonicalize;
        canonicalize.addPass(LoopSimplifyPass());
        canonicalize.addPass(LCSSAPass());
        canonicalize.run(*kernel, analyses);

        ValueToValueMapTy mapping;
        Function *serial = CloneFunction(kernel, mapping);
        serial->setName(kernel_name + ".serial");
        serial->setLinkage(GlobalValue::InternalLinkage);

        if (!partition_outer_loop(*kernel)) {
            serial->eraseFromParent();
            return;
        }

        if (!make_wrapper(module, *kernel, *serial, operation)) {
            serial->eraseFromParent();
            return;
        }

        changed = true;
    });

    return changed;
}
