/*
 * user/src/tensor/src/cpu/orc_jit.h
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
 *     graph.c prefethcer or serilized file->parallelizer->vectorizer->
 *     this_file->jit_cache
 *  2. Similar to the generic/orc_jit.cpp but little more involved
 */


#ifndef TENSOR_CPU_ORC_JIT_H
#define TENSOR_CPU_ORC_JIT_H

#include "tensor_jit.h"

#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>

#include <memory>
#include <string>

llvm::orc::LLJIT *tensor_cpu_jit_engine();
uint64 tensor_cpu_hash_bytes(uint64 hash, const void *data, size_t size);
std::unique_ptr<llvm::Module> tensor_cpu_parse_module(const char *name, const uint8 *bitcode, extent bitcode_size,
                                                       llvm::LLVMContext &context);
void tensor_cpu_remove_unused_exports(llvm::Module &module, llvm::Function &kernel);
bool tensor_cpu_prepare_module(llvm::orc::ThreadSafeModule &module, const std::string &symbol,
                               uint32 operation = 0);
void tensor_cpu_report_error(llvm::Error error);
#endif
