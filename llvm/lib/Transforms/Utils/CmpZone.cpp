//===-- CmpZone.cpp - Comparisons Instrumentation --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/CmpZone.h"
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
#include "llvm/Analysis/MemoryBuiltins.h"

using namespace llvm;

Type *int1T; Type *int2T; Type *int4T; Type *int8T;
Type *int1PtrT; Type *int2PtrT; Type *int4PtrT; Type *int8PtrT;

Function *fnfree;

FunctionCallee CmpzoneMemmove, CmpzoneMemcpy, CmpzoneMemset;

static bool hasMode(llvm::StringRef m) {
  return llvm::StringRef(getenv("FLOATZONE_MODE")).contains(m);
}

class InterestingMemoryOperand {
public:
  Use *PtrUse;
  bool IsWrite;
  Type *OpType;
  uint64_t TypeSize;
  MaybeAlign Alignment;
  // The mask Value, if we're looking at a masked load/store.
  Value *MaybeMask;

  InterestingMemoryOperand(Instruction *I, unsigned OperandNo, bool IsWrite,
                           class Type *OpType, MaybeAlign Alignment,
                           Value *MaybeMask = nullptr)
      : IsWrite(IsWrite), OpType(OpType), Alignment(Alignment),
        MaybeMask(MaybeMask) {
    const DataLayout &DL = I->getModule()->getDataLayout();
    TypeSize = DL.getTypeStoreSizeInBits(OpType);
    PtrUse = &I->getOperandUse(OperandNo);
  }

  Instruction *getInsn() { return cast<Instruction>(PtrUse->getUser()); }

  Value *getPtr() { return PtrUse->get(); }

  void print(raw_ostream &O) {
    O << "Mem" << (IsWrite ? "Write(" : "Read(");
    O << "inst={ " << *getInsn() << " }";
    O << " size={ " << TypeSize << " }";
    O << " align=" << Alignment.valueOrOne().value();
    O << ")";
  }
  const std::string toString() {
    std::string s;
    raw_string_ostream ss(s);
    print(ss);
    return s;
  }
};

Function *findErrorHandlerFunc(Module &M) {
  for (Function &ff : M.getFunctionList())
    if (ff.getName() == "free")
      return &ff;
  
  Type *int64T = IntegerType::get(M.getContext(), 64U);
  Type *ptrT = PointerType::get(int64T, 0);

  std::vector<Type*> args = {ptrT};
  FunctionType *FT = FunctionType::get(llvm::Type::getVoidTy(M.getContext()), args, false);
  Function *free = Function::Create(FT, Function::WeakAnyLinkage, "free", M);
  return free;
}

void insertCmpOneByte(Instruction &I, Value &addr, bool before, Type* ptrType) {
  Function *F = I.getParent()->getParent();
  Module *M = F->getParent();
  const DataLayout &DL = M->getDataLayout();
  LLVMContext &C = F->getContext();
  IRBuilder<> builder(C);
  if(before){
    builder.SetInsertPoint(&I);
  }
  else{
    builder.SetInsertPoint(I.getParent(), std::next(I.getIterator()));
  }

  Value *magic = ConstantInt::get(int1T, APInt(8, 0x8b));
  TypeSize size = DL.getTypeStoreSize(ptrType);
  
  // Do an N-byte cmp, but truncate it to 1-byte cmp
  Value *casted = nullptr;
  Value *target = nullptr;
  if(size == 1) {
    casted = builder.CreateBitOrPointerCast(&addr, int1PtrT);
    target = builder.CreateLoad(int1T, casted, "cmp_zone_load");
  } 
  else if(size == 2) {
    casted = builder.CreateBitOrPointerCast(&addr, int2PtrT);
    target = builder.CreateLoad(int2T, casted, "cmp_zone_load");
    target = builder.CreateTrunc(target, int1T);
  } 
  else if(size == 4) {
    casted = builder.CreateBitOrPointerCast(&addr, int4PtrT);
    target = builder.CreateLoad(int4T, casted, "cmp_zone_load");
    target = builder.CreateTrunc(target, int1T);
  } 
  else { 
    casted = builder.CreateBitOrPointerCast(&addr, int8PtrT);
    target = builder.CreateLoad(int8T, casted, "cmp_zone_load");
    target = builder.CreateTrunc(target, int1T);
  }

  Value *cmp = builder.CreateICmp(CmpInst::Predicate::ICMP_EQ, target, magic);

  Instruction *split = &I;
  if(!before){
    split = &*std::next(cast<Instruction>(cmp)->getIterator());
  }

  // if-then branch goes to error handling placeholder
  Instruction *endOfThen = SplitBlockAndInsertIfThen(cmp, split, /*unreachable*/false);
  builder.SetInsertPoint(endOfThen);
  // currently implemented as free(NULL) which is a NOP
  std::vector<llvm::Value*> Args = {Constant::getNullValue(int8PtrT)};
  builder.CreateCall(fnfree, Args);
}

void insertCmpFourBytes(Instruction &I, Value &addr, bool before, Type* ptrType) {
  Function *F = I.getParent()->getParent();
  Module *M = F->getParent();
  const DataLayout &DL = M->getDataLayout();
  LLVMContext &C = F->getContext();
  IRBuilder<> builder(C);
  if(before){
    builder.SetInsertPoint(&I);
  }
  else{
    builder.SetInsertPoint(I.getParent(), std::next(I.getIterator()));
  }

  Value *magic = ConstantInt::get(int4T, APInt(32, 0x8b8b8b8b));
  TypeSize size = DL.getTypeStoreSize(ptrType);

  // Do a 4-byte cmp
  Value *casted = nullptr;
  Value *target = nullptr;
  if(size <= 4){
    casted = builder.CreateBitOrPointerCast(&addr, int4PtrT);
    target = builder.CreateLoad(int4T, casted, "cmp_zone_load");
  }
  else{
    casted = builder.CreateBitOrPointerCast(&addr, int8PtrT);
    target = builder.CreateLoad(int8T, casted, "cmp_zone_load");
    // trunc to stimulate re-using registers (e.g.: load RBX -> cmp EBX)
    target = builder.CreateTrunc(target, int4T);
  }

  Value *cmp = builder.CreateICmp(CmpInst::Predicate::ICMP_EQ, target, magic);

  Instruction *split = &I;
  if(!before){
    split = &*std::next(cast<Instruction>(cmp)->getIterator());
  }

  // if-then branch goes to error handling placeholder
  Instruction *endOfThen = SplitBlockAndInsertIfThen(cmp, split, /*unreachable*/false);
  builder.SetInsertPoint(endOfThen);
  // currently implemented as free(NULL) which is a NOP
  std::vector<llvm::Value*> Args = {Constant::getNullValue(int8PtrT)};
  builder.CreateCall(fnfree, Args);
}

void insertCmpNBytes(Instruction &I, Value &addr, bool before, Type* ptrType) {
  Function *F = I.getParent()->getParent();
  Module *M = F->getParent();
  const DataLayout &DL = M->getDataLayout();
  LLVMContext &C = F->getContext();
  IRBuilder<> builder(C);
  if(before){
    builder.SetInsertPoint(&I);
  }
  else{
    builder.SetInsertPoint(I.getParent(), std::next(I.getIterator()));
  }

  Value *magic = nullptr;
  TypeSize size = DL.getTypeStoreSize(ptrType);

  // Do an N-byte cmp, that matches the access (as done in the LBC paper)
  Value *casted = nullptr;
  Value *target = nullptr;
  if(size == 1) {
    casted = builder.CreateBitOrPointerCast(&addr, int1PtrT);
    target = builder.CreateLoad(int1T, casted, "cmp_zone_load");
    magic = ConstantInt::get(int1T, APInt(8, 0x8b));
  } 
  else if(size == 2) {
    casted = builder.CreateBitOrPointerCast(&addr, int2PtrT);
    target = builder.CreateLoad(int2T, casted, "cmp_zone_load");
    magic = ConstantInt::get(int2T, APInt(16, 0x8b8b));
  } 
  else if(size == 4) {
    casted = builder.CreateBitOrPointerCast(&addr, int4PtrT);
    target = builder.CreateLoad(int4T, casted, "cmp_zone_load");
    magic = ConstantInt::get(int4T, APInt(32, 0x8b8b8b8b));
  } 
  else { 
    casted = builder.CreateBitOrPointerCast(&addr, int8PtrT);
    target = builder.CreateLoad(int8T, casted, "cmp_zone_load");
    magic = ConstantInt::get(int8T, APInt(64, 0x8b8b8b8b8b8b8b8b));
  }

  Value *cmp = builder.CreateICmp(CmpInst::Predicate::ICMP_EQ, target, magic);

  Instruction *split = &I;
  if(!before){
    split = &*std::next(cast<Instruction>(cmp)->getIterator());
  }

  // if-then branch goes to error handling placeholder
  Instruction *endOfThen = SplitBlockAndInsertIfThen(cmp, split, /*unreachable*/false);
  builder.SetInsertPoint(endOfThen);
  // currently implemented as free(NULL) which is a NOP
  std::vector<llvm::Value*> Args = {Constant::getNullValue(int8PtrT)};
  builder.CreateCall(fnfree, Args);
}

void insertCmpNBytesMax4(Instruction &I, Value &addr, bool before, Type* ptrType) {
  Function *F = I.getParent()->getParent();
  Module *M = F->getParent();
  const DataLayout &DL = M->getDataLayout();
  LLVMContext &C = F->getContext();
  IRBuilder<> builder(C);
  if(before){
    builder.SetInsertPoint(&I);
  }
  else{
    builder.SetInsertPoint(I.getParent(), std::next(I.getIterator()));
  }

  Value *magic = nullptr;
  TypeSize size = DL.getTypeStoreSize(ptrType);

  // Do an N-byte cmp, that matches the access (as done in the LBC paper)
  Value *casted = nullptr;
  Value *target = nullptr;
  if(size == 1) {
    casted = builder.CreateBitOrPointerCast(&addr, int1PtrT);
    target = builder.CreateLoad(int1T, casted, "cmp_zone_load");
    magic = ConstantInt::get(int1T, APInt(8, 0x8b));
  } 
  else if(size == 2) {
    casted = builder.CreateBitOrPointerCast(&addr, int2PtrT);
    target = builder.CreateLoad(int2T, casted, "cmp_zone_load");
    magic = ConstantInt::get(int2T, APInt(16, 0x8b8b));
  } 
  else if(size == 4) {
    casted = builder.CreateBitOrPointerCast(&addr, int4PtrT);
    target = builder.CreateLoad(int4T, casted, "cmp_zone_load");
    magic = ConstantInt::get(int4T, APInt(32, 0x8b8b8b8b));
  }
  else {
    casted = builder.CreateBitOrPointerCast(&addr, int8PtrT);
    target = builder.CreateLoad(int8T, casted, "cmp_zone_load");
    // trunc to stimulate re-using registers (e.g.: load RBX -> cmp EBX)
    target = builder.CreateTrunc(target, int4T);
    magic = ConstantInt::get(int4T, APInt(32, 0x8b8b8b8b));
  }

  Value *cmp = builder.CreateICmp(CmpInst::Predicate::ICMP_EQ, target, magic);

  Instruction *split = &I;
  if(!before){
    split = &*std::next(cast<Instruction>(cmp)->getIterator());
  }

  // if-then branch goes to error handling placeholder
  Instruction *endOfThen = SplitBlockAndInsertIfThen(cmp, split, /*unreachable*/false);
  builder.SetInsertPoint(endOfThen);
  // currently implemented as free(NULL) which is a NOP
  std::vector<llvm::Value*> Args = {Constant::getNullValue(int8PtrT)};
  builder.CreateCall(fnfree, Args);
}

void getInstructions(Instruction &I, SmallVectorImpl<InterestingMemoryOperand> &Interesting) {

  if (I.hasMetadata("nosanitize")) {
    return;
  }

  if (LoadInst *LI = dyn_cast<LoadInst>(&I)) {
    Interesting.emplace_back(&I, LI->getPointerOperandIndex(), false,
                                      LI->getType(), LI->getAlign());
  }
  if (StoreInst *SI = dyn_cast<StoreInst>(&I)) {
    Interesting.emplace_back(&I, SI->getPointerOperandIndex(), true,
                                      SI->getValueOperand()->getType(), SI->getAlign());
  }
  //TODO Missing atomic mem operations: AtomicRMWInst, AtomicCmpXchgInst, llvm.masked.load, llvm.masked.store
}


static const llvm::DenseSet<llvm::StringRef> functionsToIntercept = {
  "memcpy",
  "memset",
  "memmove",
  "strlen",
  "strnlen",
  "strcat",
  "strncat",
  "strcpy",
  "strncpy",
  "wcscpy",
  "printf",
  "snprintf",
  "puts",
};
static const std::string interceptedFunctionPrefix = "cmp_";

static void maybeReplaceWithInterceptedFunction(llvm::Module &M, CallInst &i) {
  Function *callTarget = i.getCalledFunction();
  // Indirect call.
  if (!callTarget)
    return;

  // Filter out functions not intended to be intercepted.
  std::string name = i.getCalledFunction()->getName().str();
  if (!functionsToIntercept.contains(name))
    return;


  std::string replacementName = interceptedFunctionPrefix + name;
  FunctionCallee replacement = M.getOrInsertFunction(replacementName,
                                                     callTarget->getFunctionType());

  {
    IRBuilder<> builder(&i);
    std::vector<Value *> args;
    for (Value *arg : i.operands())
      args.push_back(arg);
    CallInst *replacedCall = builder.CreateCall(replacement, args);
    assert(replacedCall);
    i.replaceAllUsesWith(replacedCall);
    i.eraseFromParent();
  }

}

static void replaceMemIntrinsics(Function &F, MemIntrinsic* MI){
  Module &M = *(F.getParent());
  int LongSize = M.getDataLayout().getPointerSizeInBits();
  IntegerType* IntptrTy = Type::getIntNTy(M.getContext(), LongSize);
  IRBuilder<> IRB(MI);
  
  if (isa<MemTransferInst>(MI)) {
    IRB.CreateCall(
      isa<MemMoveInst>(MI) ? CmpzoneMemmove : CmpzoneMemcpy,
      {IRB.CreatePointerCast(MI->getOperand(0), IRB.getInt8PtrTy()),
      IRB.CreatePointerCast(MI->getOperand(1), IRB.getInt8PtrTy()),
      IRB.CreateIntCast(MI->getOperand(2), IntptrTy, false)});
  } else if (isa<MemSetInst>(MI)) {
    IRB.CreateCall(
      CmpzoneMemset,
      {IRB.CreatePointerCast(MI->getOperand(0), IRB.getInt8PtrTy()),
      IRB.CreateIntCast(MI->getOperand(1), IRB.getInt32Ty(), false),
      IRB.CreateIntCast(MI->getOperand(2), IntptrTy, false)});
  } else {
    llvm_unreachable("Neither MemSet nor MemTransfer?");
  }
  MI->eraseFromParent();
}

void CmpZonePass::runOnFunc(Function &F, FunctionAnalysisManager &AM) {

  // errs() << "[CmpZone] running on Function: " << F.getName() << "\n";

  // The load/store pointers to be instrumented
  SmallVector<InterestingMemoryOperand, 16> OperandsToInstrument;
  // Temp vector to track only instrumenting every address once per basic block
  SmallPtrSet<Value *, 16> TempsToInstrument;

  for (BasicBlock &BB : F) {
    TempsToInstrument.clear();

    for (Instruction &I : BB) {
      // load + store
      SmallVector<InterestingMemoryOperand, 1> InterestingOperands;
      getInstructions(I, InterestingOperands);

      if (!InterestingOperands.empty()) {
        for (auto &Operand : InterestingOperands) {
          Value *Ptr = Operand.getPtr();
          if (!TempsToInstrument.insert(Ptr).second){
            continue; // We've seen this temp in the current BB.
          }
          OperandsToInstrument.push_back(Operand);
        }
      }
    }
  }
  
  for (auto &MO : OperandsToInstrument){
    // Mode #1:    1-byte cmp
    if(hasMode("1")){
      insertCmpOneByte(*MO.getInsn(), *MO.getPtr(), MO.IsWrite, MO.OpType);
    }
    // Mode #2:    4-byte cmp
    else if(hasMode("4")){
      insertCmpFourBytes(*MO.getInsn(), *MO.getPtr(), MO.IsWrite, MO.OpType);
    }
    // Mode #3:    N-byte cmp -- where N == access size
    else if(hasMode("N")){
      insertCmpNBytes(*MO.getInsn(), *MO.getPtr(), MO.IsWrite, MO.OpType);
    }
    // Mode #4:    N-byte cmp with max N == 4
    else if(hasMode("max")){
      insertCmpNBytesMax4(*MO.getInsn(), *MO.getPtr(), MO.IsWrite, MO.OpType);
    }
  }

  // Run some optimization passes to simplify instrumentation: only if not -O0
  if(!F.hasOptNone()){
    AM.clear();
    InstCombinePass().run(F, AM);
    AM.clear();
    SimplifyCFGPass().run(F, AM);
  }

  SmallVector<MemIntrinsic *, 16> intrinToInstrument;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      // skip our own mem* calls
      if (I.hasMetadata("floatzone")) continue;
      if(MemIntrinsic *MI = dyn_cast<MemIntrinsic>(&I)){
        intrinToInstrument.push_back(MI);
      }
    }
  }

  // make sure to change mem intrinsics to direct calls
  for(auto *memInst : intrinToInstrument){
    replaceMemIntrinsics(F, memInst);
  }

  std::vector<CallInst *> calls;
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      if (CallInst *C = llvm::dyn_cast<CallInst>(&I))
        calls.push_back(C);

  for (CallInst *C : calls)
    maybeReplaceWithInterceptedFunction(*F.getParent(), *C);


  return;
}


PreservedAnalyses CmpZonePass::run(Module &M, ModuleAnalysisManager &MAM){
#ifdef DEFAULTCLANG
  return PreservedAnalyses::none();
#endif

  if (!hasMode("cmp_zone") || hasMode("float"))
    return PreservedAnalyses::none();

  // Code is based on AddressSanitizer and ASan--
  LLVMContext &C = M.getContext();
  IRBuilder<> builder(C);
  int LongSize = M.getDataLayout().getPointerSizeInBits();
  IntegerType* IntptrTy = Type::getIntNTy(C, LongSize);
  CmpzoneMemmove = M.getOrInsertFunction(interceptedFunctionPrefix + "memmove",
                                       builder.getInt8PtrTy(), builder.getInt8PtrTy(),
                                       builder.getInt8PtrTy(), IntptrTy);
  CmpzoneMemcpy = M.getOrInsertFunction(interceptedFunctionPrefix +"memcpy",
                                      builder.getInt8PtrTy(), builder.getInt8PtrTy(),
                                      builder.getInt8PtrTy(), IntptrTy);
  CmpzoneMemset = M.getOrInsertFunction(interceptedFunctionPrefix +"memset",
                                      builder.getInt8PtrTy(), builder.getInt8PtrTy(),
                                      builder.getInt32Ty(), IntptrTy);

  int1T = IntegerType::get(C, 8U);
  int2T = IntegerType::get(C, 16U);
  int4T = IntegerType::get(C, 32U);
  int8T = IntegerType::get(C, 64U);

  int1PtrT = PointerType::get(int1T, 0);
  int2PtrT = PointerType::get(int2T, 0);
  int4PtrT = PointerType::get(int4T, 0);
  int8PtrT = PointerType::get(int8T, 0);

  fnfree = findErrorHandlerFunc(M);

  auto &FAM = MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

  for (Function &F : M){
    if (!F.isDeclaration())
      runOnFunc(F, FAM);
  }

  return PreservedAnalyses::none();
}