/*
 * user/src/tensor/src/jit/gen/cpu/abstract_vec/abstract_vec.cpp
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
 * 1. Reference: https://llvm.org/docs/Vectorizers.html
 * 2. TODO: look at here https://llvm.org/docs/SandboxVectorizer.html
 *          it can evaluate the profitability then either commit or
 *          rollback the vecotrization
 * 3. Here currently the llvm production autovectorization is used
 *    - loop_vecotrizer: widen the loop
 *    - SLP vectorizer:  combine similar independent scalar instructions
 * 4. The kernel generator is expected to emit portable scalar SSA
 *    provided that llvm choose the simd_width and instructions
 * 5. .op--parser-->SSA scalar(as .bc)->
 *     graph.c prefethcer or serilized file->parrallizer->this_file->orc_jit.cpp->jit_cache
 * 6.  The order matter parrallize->vectorize
 */

#include "abstract_vec.h"

#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Target/TargetMachine.h>

using namespace llvm;
using namespace llvm::orc;


bool tensor_cpu_abstract_vectorize(
    ThreadSafeModule &thread_safe_module,
    JITTargetMachineBuilder &target) {

    auto machine = target.createTargetMachine();

    if (!machine) {
        consumeError(machine.takeError());
        return false;
    }

    thread_safe_module.withModuleDo([&](Module &module) {
        for (Function &function : module) {
            if (function.isDeclaration()) continue;
            function.removeFnAttr("target-cpu");
            function.removeFnAttr("target-features");
            function.addFnAttr("target-cpu", (*machine)->getTargetCPU());
            function.addFnAttr("target-features", (*machine)->getTargetFeatureString());
            function.addFnAttr("prefer-vector-width", "256");
        }

        PipelineTuningOptions tuning;

        tuning.LoopVectorization = true;
        tuning.SLPVectorization = true;
        tuning.LoopUnrolling = true;

        PassBuilder passes(machine->get(), tuning);

        LoopAnalysisManager loops;
        FunctionAnalysisManager functions;
        CGSCCAnalysisManager cgscc;
        ModuleAnalysisManager modules;

        passes.registerModuleAnalyses(modules);
        passes.registerCGSCCAnalyses(cgscc);
        passes.registerFunctionAnalyses(functions);
        passes.registerLoopAnalyses(loops);
        passes.crossRegisterProxies(loops, functions, cgscc, modules);

        ModulePassManager pipeline = passes.buildPerModuleDefaultPipeline(OptimizationLevel::O3);

        pipeline.run(module, modules);
    });

    return true;
}
