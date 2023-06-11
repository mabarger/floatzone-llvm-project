//===-- HelloWorld.cpp - Example Transformations --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/HelloWorld.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Analysis/DomPrinter.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
// post-pass-optimization test
#include <llvm/Transforms/InstCombine/InstCombine.h>

using namespace llvm;

/// Returns true if the string is in the value of the MODe environment
/// variable. Just used for selectively disabling some functionality.
/// e.g. `MODE=noload clang ...` for not instrumenting loads.
static bool hasMode(llvm::StringRef m) {
  return llvm::StringRef(getenv("FLOATZONE_MODE")).contains(m);
}

namespace {

  /// Class for storing all the injected checks in our function.
  struct CheckInjector {

    /// The current function we're working on.
    Function &func;
    /// The post-dominator tree of our function.
    PostDominatorTree postDom;
    /// The dominator tree of our function.
    DominatorTree dom;
    /// The list of instructions that do something 'unsafe' such as calling
    /// external code that might mess with our memory regions.
    llvm::SmallPtrSet<Instruction *, 32> unsafeInsts;
    /// Returns the
    Module &getModule() const {
      return *func.getParent();
    }

    /// Returns true if there is potentially a call executed between
    /// A and B.
    bool isDangerousCallBetween(Instruction &A, Instruction &B) const {
      llvm::SmallPtrSet<BasicBlock *, 32> unsafeBlocks;
      for (Instruction *I : unsafeInsts) {
        BasicBlock *BB = I->getParent();
        if (BB == A.getParent() && I->comesBefore(&A))
          continue;
        if (BB == B.getParent() && B.comesBefore(I))
          continue;
        unsafeBlocks.insert(BB);
      }
      if (unsafeBlocks.contains(A.getParent()) || unsafeBlocks.contains(B.getParent()))
          return true;
      // Is there any path A -> B with unsafe block
      // same as: Is there no path A -> B without unsafe block
      return !isPotentiallyReachable(&A, &B, &unsafeBlocks, &dom);
    }

    /// Utility class representing a single injected memory check. E.g.
    /// float *load = PTR;
    /// *load + fadd(MAGIC)
    /// int *originalLoadStore = origPTR;
    struct Check {
      /// The injected floating point load instruction. `load` in the example
      /// above.
      LoadInst *load = nullptr;
      /// The injected floating point add instruction.
      /// The `+` in the example above.
      CallInst *add = nullptr;
      /// The float pointer that is being loaded. `PTR` in the example above.
      Value *ptr = nullptr;
      /// The instrumented instruction in the original LLVM IR.
      /// `originalLoadStore` in the example above.
      Instruction *instrumentedInst = nullptr;
      /// The original (non-float) pointer that is being loaded.
      /// `origPTR` in the example above.
      Value *origPtr = nullptr;
      /// The underlying element type of the non-float pointer.
      /// `int` in the example above.
      Type *ptrElemType = nullptr;
      /// Whether this check was deleted as part of an optimization.
      /// True -> not deletedy, false -> deleted.
      bool valid = true;

      BranchInst *jump = nullptr;
      Value *cmp = nullptr;
      Value *casted = nullptr;


      /// Return the stripped pointer value. Removes LLVM sugar from it.
      Value *strippedPtr() const {
        return ptr->stripPointerCastsAndAliases();
      }

      /// Determines this check and the other given check check the same
      /// memory (same contents and address).
      bool checkSameMemoryToOtherCheck(CheckInjector &C, AAResults &AA, const Check &other) const {
        // if AA thinks they are not aliasing then we're not checking the
        // same memory.
        if (!AA.isMustAlias(strippedPtr(), other.strippedPtr()))
          return false;
        // if there is any call between this check and the other that might
        // change the contents (or allocate/free the memory) then the
        // two checks are not checking the same content.
        if (C.isDangerousCallBetween(*load, *other.load))
          return false;

        return true;
      }

      /// Removes this check and its associated injected code from the IR.
      void remove() {
        if (jump) {
          //BasicBlock *then = jump->getSuccessor(1);
          //jump->setSuccessor(1, jump->getSuccessor(0));
          //then->eraseFromParent();
          //jump->eraseFromParent();
          jump->setCondition(ConstantInt::getNullValue(IntegerType::get(jump->getParent()->getParent()->getContext(), 1U)));
        }
        if (Instruction *I = dyn_cast_or_null<Instruction>(cmp))
          I->eraseFromParent();
        if (add)
          add->eraseFromParent();
        if (casted)
          if (Instruction *I = dyn_cast_or_null<Instruction>(casted))
            I->eraseFromParent();
        if (load)
          load->eraseFromParent();
      }

    };


    std::vector<Check> checks;

    /// Create a list of instructions that potentially mess with memory
    /// (allocate things or write to memory).
    void createUnsafeInstList() {
      for (BasicBlock &BB : func) {
        for (Instruction &I : BB) {
          // Ignore LLVM annotation functions and debug info functions as
          // they do nothing with memory.
          if (I.isLifetimeStartOrEnd())
            continue;
          if (I.isDebugOrPseudoInst())
            continue;

          // Check if the instruction is any function call (direct or
          // indirect).
          CallBase *call = dyn_cast<CallBase>(&I);
          if (!call)
            continue;
          // For now treat all function calls that might have side effects
          // as messing with memory.
          if (!call->mayHaveSideEffects())
            continue;
          unsafeInsts.insert(&I);
        }
      }
    }

    explicit CheckInjector(Function &func)
      : func(func), postDom(func), dom(func) {
      createUnsafeInstList();
    }

    /// Finds and removes redundant memory checks.
    void optimize(AAResults &AA, TargetLibraryInfo &TLI) {

      // Go over every injected check. If the checked pointer is known to be
      // dereferencable at the given instruction, then remove the check.
      // FIXME: "dereferencable" is not really a clear definition, so LLVM
      // might return true here for some weird memory accesses.
      for (Check &check : checks) {
        if (isDereferenceablePointer(check.origPtr, check.ptrElemType,
                                     func.getParent()->getDataLayout(),
                                     check.instrumentedInst, &dom, &TLI))
          check.valid = false;
      }

      // Go over every check again and filter out redundant checks.
      // E.g.
      //   int i = array[1];
      //   array[1] = i + 1; // memory access here is redundant.
      for (Check &check : checks) {
        if (!check.valid)
          continue;

        // Compare the current check against each other check.
        for (Check &otherCheck : checks) {
          // Make sure we never compare a check against itself and ignore
          // all checks marked as being removed (valid == false).
          if (!otherCheck.valid || &otherCheck == &check)
            continue;

          // If both checks point to different memory then they can't be
          // redundant.
          if (!check.checkSameMemoryToOtherCheck(*this, AA, otherCheck))
            continue;

          // If one check post-dominates the other check (and they point to
          // the same memory as checked above), then then we can remove one
          // of the checks.
          if (postDom.dominates(otherCheck.load, check.load))
            otherCheck.valid = false;
        }
      }

      // Now go over all invalid checks and remove their code from the IR.
      for (Check &check : checks)
        if (!check.valid)
          check.remove();
    }

    /// Injects the actual access check via a float load + fadd instruction.
    void addAccessCheckFloat(Instruction &I, Value &origPtr,
                        const Align dataAlignment,
                        Type *ptrElemType,
                        bool insertAfter) {
      LLVMContext &C = func.getContext();
      // This class just injects instructions for us into the LLVM IR.
      IRBuilder<> builder(C);
      if (insertAfter) {
        builder.SetInsertPoint(I.getParent(), std::next(I.getIterator()));
      } else
        builder.SetInsertPoint(&I);

      // The alignment of the 'float' type.
      const Align floatAlign = Align(4);
      // The alignment assumed for the 'float' load.
      Align effectiveAlignment = dataAlignment;
      Value *ptr = &origPtr;

      // If we want forcibly align the load to float, then update the pointer
      // now.
      if (hasMode("align")) {

        const DataLayout &DL = func.getParent()->getDataLayout();

        // check its not scalable (weird dynamic size??)
        TypeSize BaseSize = DL.getTypeSizeInBits(ptrElemType);
        assert(!BaseSize.isScalable());

        IntegerType *t = builder.getIntPtrTy(DL);
        Value *ptrInt = builder.CreatePtrToInt(ptr, t);
        TypeSize size = DL.getTypeStoreSize(ptrElemType);
        Value *off = ConstantInt::get(C, APInt(t->getBitWidth(), size-1));
        Value *neg3 = ConstantInt::get(C, APInt(t->getBitWidth(), (~3ULL)));
        Value *plus2 = ConstantInt::get(C, APInt(t->getBitWidth(), (2ULL)));
        Value *plus4 = ConstantInt::get(C, APInt(t->getBitWidth(), (4ULL)));
        Value *plus6 = ConstantInt::get(C, APInt(t->getBitWidth(), (6ULL)));
        Value *sub2 = ConstantInt::get(C, APInt(t->getBitWidth(), (-2LL)));
        Value *aligned_load = NULL;

        if (size == 1 || size == 2 || size == 4) {
            if (dataAlignment == 4) {
                //Do nothing
                aligned_load = ptrInt;
            } else if (size == 1) {
                //addr & (~0x3ULL);
                aligned_load = builder.CreateAnd(ptrInt, neg3);
            } else if (dataAlignment == 2) {
				if (size == 2) {
                    //addr - 2
					aligned_load = builder.CreateAdd(ptrInt, sub2);
				}
                else if (size == 4) {
                    //addr + 2
					aligned_load = builder.CreateAdd(ptrInt, plus2);
                }
			} else {
                //(addr + size - 1) & (~3ULL)
                aligned_load = builder.CreateAnd(builder.CreateAdd(ptrInt, off), neg3);
            }
        }
        else if (size == 8) {
            if (dataAlignment == 4 || dataAlignment == 8) {
                //addr + 4
	            aligned_load = builder.CreateAdd(ptrInt, plus4);
            }
            else if(dataAlignment == 2) {
                //addr + 6
	            aligned_load = builder.CreateAdd(ptrInt, plus6);
            } else {
                //(addr + size - 1) & (~3ULL)
                aligned_load = builder.CreateAnd(builder.CreateAdd(ptrInt, off), neg3);
            }
        }
        else {
            //(addr + size - 1) & (~3ULL)
            aligned_load = builder.CreateAnd(builder.CreateAdd(ptrInt, off), neg3);
        }

        ptr = builder.CreateIntToPtr(aligned_load, ptr->getType());
        effectiveAlignment = floatAlign;
      }

      // Get the type of 'float'.
      Type *floatT = llvm::Type::getFloatTy(C);
      // Get the type of 'float *'.
      Type *floatPtrT = PointerType::get(floatT, 0);
      // Cast our original pointer to 'float *'.
      Value *floatAddr = builder.CreateBitOrPointerCast(ptr, floatPtrT);
      // Create a load on our 'float *' pointer.
      // We now have 'float *floatzone_load = (float*)ptr'.
      //LoadInst *loaded = builder.CreateAlignedLoad(floatT, floatAddr,
      //                                             effectiveAlignment,
      //                                             "floatzone_load");

#if 0
      InlineAsm *IA = InlineAsm::get(
                           FunctionType::get(llvm::Type::getVoidTy(C), {floatPtrT}, false),
                           StringRef("lea ($0), %r8"),
                           StringRef("r,~{r8},~{dirflag},~{fpsr},~{flags}"),
                           /*hasSideEffects=*/true,
                           /*isAlignStack*/false,
                           InlineAsm::AD_ATT,
                           /*canThrow*/false);
      //TODO Maybe better use original pointer for perf and exception handler?
      std::vector<llvm::Value*> Args = { floatAddr };
      builder.CreateCall(IA, Args, SmallVector<llvm::OperandBundleDef, 1>()); 
#endif
      // Create our magic value that we add to the loaded float.
      const fltSemantics & floatSem = floatT->getFltSemantics();
      APFloat addV(floatSem, APInt(/*bit size*/32UL,
                                   /*magic value*/0xf3000000ULL));

      // proper: load the constant value. todo: pin to xmmY
      //Value *AddVal = ConstantFP::get(floatT, addV);
      // test: use undef value to operate on garbage state register
      //Value *AddVal = UndefValue::get(floatT);

      // Now create the actual floating point add operation.
      CallInst *add = nullptr;
      // if MODE=noadd then we don't emit the add. This is just for benchmarking
      //if (hasMode("noadd"))
      //  loaded->setVolatile(true);
      //else {
        // Create a special floating point add instruction. The 'constrained'
        // is that LLVM knows this has a side effect of raising an exception.
#if 0
        add = builder.CreateConstrainedFPBinOp(Intrinsic::experimental_constrained_fadd, loaded, AddVal);
#else
      InlineAsm *IA = InlineAsm::get(
                           FunctionType::get(llvm::Type::getVoidTy(C), {floatPtrT, floatT}, false),
                           StringRef("addss ($0), $1"),
                           StringRef("r,x,~{dirflag},~{fpsr},~{flags}"),
                           /*hasSideEffects=*/true,
                           /*isAlignStack*/false,
                           InlineAsm::AD_ATT,
                           /*canThrow*/false);
      //TODO Maybe better use original pointer for perf and exception handler?
      std::vector<llvm::Value*> Args = { floatAddr, ConstantFP::get(floatT, addV) };
      add = builder.CreateCall(IA, Args);
#endif
      //}
      // Add the final check to the list of added checks so we can optimize
      // it later.
      checks.push_back(Check{nullptr, add, &origPtr, &I, &origPtr, ptrElemType});
      return;
    }

    /// Returns the error handling function.
    /// Only used for the slow load/compare check from redpool. Currently
    /// returns strncpy as a dummy function.
    Function *findErrorHandlerFunc() {
#if 1
      for (Function &f : getModule().getFunctionList())
        if (f.getName() == "free")
          return &f;

      Type *int64T = IntegerType::get(getModule().getContext(), 64U);
      //Type *int8T = IntegerType::get(getModule().getContext(), 32U);
      Type *charPtr = PointerType::get(int64T, 0);

      std::vector<Type*> args = {charPtr};
      FunctionType *FT = FunctionType::get(llvm::Type::getVoidTy(getModule().getContext()), args, false);

      Function *F = Function::Create(FT, Function::WeakAnyLinkage, "free", getModule());
#else
      for (Function &f : getModule().getFunctionList())
        if (f.getName() == "exit") {
          return &f;
        }

      Type *int32T = IntegerType::get(getModule().getContext(), 32U);

      std::vector<Type*> args = {int32T};
      FunctionType *FT = FunctionType::get(llvm::Type::getVoidTy(getModule().getContext()), args, false);

      Function *F = Function::Create(FT, Function::WeakAnyLinkage, "exit", getModule());
#endif
      return F;
    }

    /// This implementes the slow load and compare check that redpool does.
    /// Only here for benchmarking reasons.
    void addAccessCheckCmp(Instruction &I, Value &origPtr,
                        const Align dataAlignment,
                        Type *ptrElemType,
                        bool insertAfter) {

      LLVMContext &C = func.getContext();
      IRBuilder<> builder(C);
      const DataLayout &DL = func.getParent()->getDataLayout();

      //builder.SetInsertPoint(&I);
      if (insertAfter) {
        builder.SetInsertPoint(I.getParent(), std::next(I.getIterator()));
      } else {
        builder.SetInsertPoint(&I);
      }

      Value *ptr = &origPtr;
    
      Type *int1T = IntegerType::get(C, 8U);
      Type *int2T = IntegerType::get(C, 16U);
      Type *int4T = IntegerType::get(C, 32U);
      Type *int8T = IntegerType::get(C, 64U);

      Type *int1PtrT = PointerType::get(int1T, 0);
      Type *int2PtrT = PointerType::get(int2T, 0);
      Type *int4PtrT = PointerType::get(int4T, 0);
      Type *int8PtrT = PointerType::get(int8T, 0);

      TypeSize size = DL.getTypeStoreSize(ptrElemType);

      Value *floatAddr = nullptr;
      LoadInst *loaded = nullptr;
      Value *magic = nullptr;
      Value *casted = nullptr;

      //if(size == 1) {
      //  floatAddr = builder.CreateBitOrPointerCast(ptr, int1PtrT);
      //  loaded = builder.CreateAlignedLoad(int1T, floatAddr,
      //                                             Align(1),
      //                                             "floatzone_load");
      //  magic = ConstantInt::get(int1T, APInt(8, 0xdf));
      //} else if(size == 2) {
      //  floatAddr = builder.CreateBitOrPointerCast(ptr, int2PtrT);
      //  loaded = builder.CreateAlignedLoad(int2T, floatAddr,
      //                                             Align(1),
      //                                             "floatzone_load");
      //  magic = ConstantInt::get(int2T, APInt(16, 0xdfdf));
      //} else if(size == 4) {
      //floatAddr = builder.CreateBitOrPointerCast(ptr, int4PtrT);
      //loaded = builder.CreateAlignedLoad(int4T, floatAddr,
      //                                           Align(1),
      //                                           "floatzone_load");
      //magic = ConstantInt::get(int4T, APInt(32, 0x8b8b8b8b));
      //} else { 
      //  floatAddr = builder.CreateBitOrPointerCast(ptr, int8PtrT);
      //  loaded = builder.CreateAlignedLoad(int8T, floatAddr,
      //                                             Align(1),
      //                                             "floatzone_load");
      //  magic = ConstantInt::get(int8T, APInt(64, 0xdfdfdfdfdfdfdfdf));
      //}
      
      magic = ConstantInt::get(int1T, APInt(8, 0xdf));
      if(size == 1) {
        floatAddr = builder.CreateBitOrPointerCast(ptr, int1PtrT);
        loaded = builder.CreateAlignedLoad(int1T, floatAddr,
                                                   Align(1),
                                                   "floatzone_load");
        casted = nullptr;
      } else if(size == 2) {
        floatAddr = builder.CreateBitOrPointerCast(ptr, int2PtrT);
        loaded = builder.CreateAlignedLoad(int2T, floatAddr,
                                                   Align(1),
                                                   "floatzone_load");
        casted = builder.CreateTrunc(loaded, int1T);
      } else if(size == 4) {
        floatAddr = builder.CreateBitOrPointerCast(ptr, int4PtrT);
        loaded = builder.CreateAlignedLoad(int4T, floatAddr,
                                                   Align(1),
                                                   "floatzone_load");
        casted = builder.CreateTrunc(loaded, int1T);
      } else { 
        floatAddr = builder.CreateBitOrPointerCast(ptr, int8PtrT);
        loaded = builder.CreateAlignedLoad(int8T, floatAddr,
                                                   Align(1),
                                                   "floatzone_load");
        casted = builder.CreateTrunc(loaded, int1T);
      }

      Value *cmp = nullptr;
      if(casted)
        cmp = builder.CreateICmp(CmpInst::Predicate::ICMP_EQ, casted, magic);
      else
        cmp = builder.CreateICmp(CmpInst::Predicate::ICMP_EQ, loaded, magic);


      Instruction *split = &I;
      if (insertAfter){
        split = &*std::next(cast<Instruction>(cmp)->getIterator());
      }
      Instruction *endOfThen = SplitBlockAndInsertIfThen(cmp, split, /*unreachable*/false);
      builder.SetInsertPoint(endOfThen);
      if (hasMode("ud2")) {

        InlineAsm *IA = InlineAsm::get(
                             FunctionType::get(llvm::Type::getVoidTy(C), {}, false),
                             StringRef("nop"),
                             StringRef("~{dirflag},~{fpsr},~{flags}"),
                             /*hasSideEffects=*/true,
                             /*isAlignStack*/false,
                             InlineAsm::AD_ATT,
                             /*canThrow*/false);
        builder.CreateCall(IA, {}); 
      } 
      else {
        //builder.CreateAlloca(int8T);
        //std::vector<llvm::Value*> Args = {ConstantInt::get(int8T, APInt(64, 0))};
        std::vector<llvm::Value*> Args = {Constant::getNullValue(int8PtrT)};
        builder.CreateCall(findErrorHandlerFunc(), Args);
        //builder.CreateCall(findErrorHandlerFunc(),
        //                   {Constant::getNullValue(charPtr),
        //                    Constant::getNullValue(charPtr),
        //                    ConstantInt::get(int64T, APInt(64, 0))});
      }
      //checks.push_back(Check{loaded, nullptr, &origPtr, &I, &origPtr, ptrElemType});
      Check check{loaded, nullptr, &origPtr, &I, &origPtr, ptrElemType};
      check.cmp = cmp;                                                  
      check.casted = casted;
      check.jump = cast<BranchInst>(loaded->getParent()->getTerminator());    
      checks.push_back(check);                                          

      return;
    }

    /// Adds an access check for the given load/store instruction.
    /// \param I The load store instruction.
    /// \param origPtr The pointer the load/store is accessing.
    /// \param dataAlignment The alignment of the loaded/stored type.
    /// \param ptrElemType The underlying type of the load/store.
    /// \param insertAfter True if the instrumentation should be injected
    ///                    after the memory operation, false otherwise.
    void addAccessCheck(Instruction &I, Value &origPtr,
                        const Align dataAlignment,
                        Type *ptrElemType,
                        bool insertAfter) {
      // Do access check either with our magic float check or via a simple
      // load and compare (which is the slow thing redpool does).
      if (hasMode("compare")) {
         addAccessCheckCmp(I, origPtr, dataAlignment, ptrElemType, insertAfter);
         return;
      }
      addAccessCheckFloat(I, origPtr, dataAlignment, ptrElemType, insertAfter);
    }
  };

}

// Entry point of our implementation. This is executed on every function F.
PreservedAnalyses HelloWorldPass::run(Function &F,
                                      FunctionAnalysisManager &AM) {
#ifdef DEFAULTCLANG
  return PreservedAnalyses::none();
#endif

  if (!hasMode("oldversion"))
    return PreservedAnalyses::none();

  // LLVM can analyse some stuff for us, so let's collect some info.

  // Alias information.
  AAResults &AA = AM.getResult<AAManager>(F);
  // Information about the C library of our target (e.g., what does memcpy
  // do and that stuff).
  TargetLibraryInfo &TLI = AM.getResult<TargetLibraryAnalysis>(F);
  // Information about the loops in our function.
  LoopAnalysis::Result &loops = AM.getResult<LoopAnalysis>(F);

  auto isInLoop = [&loops](const Instruction &I) {
    for (const Loop *l : loops) {
      if (l->contains(&I)) {
        return true;
      }
    }
    return false;
  };

  // Collect all the loads and stores in our function that we need to
  // instrument.
  SmallPtrSet<StoreInst*, 16> storeToInstrument;
  SmallPtrSet<LoadInst*, 16> loadToInstrument;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      // if MODE=noloop then we don't instrument load/store in a loop.
      // This is just for estimating how much of the overhead could be reduced
      // by optimizing loops better.
      if (hasMode("noloop") && isInLoop(I))
        continue;

      if (StoreInst *Store = dyn_cast<StoreInst>(&I))
        storeToInstrument.insert(Store);
      if (LoadInst *Load = dyn_cast<LoadInst>(&I))
        loadToInstrument.insert(Load);
    }
  }

  // Now inject all the instrumentation for our loads and stores.
  CheckInjector check(F);
  if (!hasMode("nostore"))
    for (StoreInst *Store : storeToInstrument)
      check.addAccessCheck(*Store, *Store->getPointerOperand(),
                           Store->getAlign(),
                           Store->getValueOperand()->getType(),
                           /*after=*/false);

  if (!hasMode("noload"))
    for (LoadInst *Load : loadToInstrument)
      check.addAccessCheck(*Load, *Load->getPointerOperand(),
                           Load->getAlign(), Load->getType(),
                           /*after=*/true);

  // Optimize the injected instrumentation by removing redundant checks.
  if (!hasMode("noopt")) {
    check.optimize(AA, TLI);
    // try applying arithmetic optimization in case our code is unoptimized :(
    AM.clear();
    InstCombinePass().run(F, AM);
    AM.clear();
    SimplifyCFGPass().run(F, AM);
  }

  // This just tells LLVM that it can't reuse any analysis results after
  // this pass. Don't change this ever unless you care about compilation times
  // (which we don't care about...).
  return PreservedAnalyses::none();
}
