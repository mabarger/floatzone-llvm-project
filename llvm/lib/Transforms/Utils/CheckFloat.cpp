#include "llvm/Transforms/Utils/CheckFloat.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Analysis/DomPrinter.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Utils/SlimasanProject.h"
#include "llvm/ADT/Statistic.h"
#include <map>
#include <unordered_set>
#include <assert.h>
#include <fstream>

using namespace llvm;

// Mem family functions (to avoid calling LLVM intrinsics)
FunctionCallee CheckfloatMemmove, CheckfloatMemcpy, CheckfloatMemset;

// returns true if the string is in the value of the mode environment
// variable. just used for selectively disabling some functionality.
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

/// \param I The insertion point
/// \param addr The pointer the load/store is accessing
/// \param before Insert before or after the insertion point
/// \param ptrType Underlying type of the load/store access
static void insertFloatzoneCheck(Instruction &I, Value &addr, bool before, Type* ptrType) {
  Function *F = I.getParent()->getParent();
  Module *M = F->getParent();
  LLVMContext &C = F->getContext();
  IRBuilder<> builder(C);
  if(before){
    builder.SetInsertPoint(&I);
  }
  else{
    builder.SetInsertPoint(I.getParent(), std::next(I.getIterator()));
  }

  const DataLayout &DL = M->getDataLayout();
  IntegerType *t = builder.getIntPtrTy(DL);
  // Get the access size of the memory operation
  TypeSize size = DL.getTypeStoreSize(ptrType);
  // The pointer we want to check
  Value* load = &addr;
  
  if(hasMode("just_size")){
    Value *off_size = ConstantInt::get(C, APInt(t->getBitWidth(), size-1));
    // Offset the to-be-checked address by the access size
    std::vector<Value *> indizes = {off_size};
    load = builder.CreateInBoundsGEP(builder.getInt8Ty(), load, indizes);
  }

  // Get the type of 'float'.
  Type *floatT = llvm::Type::getFloatTy(C);
  // Get the type of 'float *'.
  Type *floatPtrT = PointerType::get(floatT, 0);
  // Cast our original pointer to 'float *'.
  Value *floatAddr = builder.CreateBitOrPointerCast(load, floatPtrT);

  // Create our magic value that we add to the loaded float.
  const fltSemantics & floatSem = floatT->getFltSemantics();
  APFloat addV(floatSem, APInt(32UL, /*magic value*/ 0x0b8b8b8aULL));
  Value *addVal = ConstantFP::get(floatT, addV);

  // Now create the actual floating point add operation (asm embedded).
  InlineAsm *IA = InlineAsm::get(
	      				   FunctionType::get(llvm::Type::getVoidTy(C), {floatPtrT, floatT}, false),
	      				   StringRef("vaddss $0, $1, %xmm15"),
	      				   StringRef("p,v,~{xmm15},~{dirflag},~{fpsr},~{flags}"),
	      				   /*hasSideEffects=*/ true,
	      				   /*isAlignStack*/ false,
	      				   InlineAsm::AD_ATT,
	      				   /*canThrow*/ false);
  std::vector<llvm::Value*> args = { floatAddr, addVal };
  builder.CreateCall(IA, args);
}

static void getInterestingMemoryOperands(Instruction &I, SmallVectorImpl<InterestingMemoryOperand> &Interesting) {

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
static void replaceMemIntrinsics(Function &F, MemIntrinsic* MI){
  Module &M = *(F.getParent());
  int LongSize = M.getDataLayout().getPointerSizeInBits();
  IntegerType* IntptrTy = Type::getIntNTy(M.getContext(), LongSize);
  IRBuilder<> IRB(MI);
  
  if (isa<MemTransferInst>(MI)) {
    IRB.CreateCall(
      isa<MemMoveInst>(MI) ? CheckfloatMemmove : CheckfloatMemcpy,
      {IRB.CreatePointerCast(MI->getOperand(0), IRB.getInt8PtrTy()),
      IRB.CreatePointerCast(MI->getOperand(1), IRB.getInt8PtrTy()),
      IRB.CreateIntCast(MI->getOperand(2), IntptrTy, false)});
  } else if (isa<MemSetInst>(MI)) {
    IRB.CreateCall(
      CheckfloatMemset,
      {IRB.CreatePointerCast(MI->getOperand(0), IRB.getInt8PtrTy()),
      IRB.CreateIntCast(MI->getOperand(1), IRB.getInt32Ty(), false),
      IRB.CreateIntCast(MI->getOperand(2), IntptrTy, false)});
  } else {
    llvm_unreachable("Neither MemSet nor MemTransfer?");
  }
  MI->eraseFromParent();
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

// Entry point of our implementation. This is executed on every function F.
void CheckFloatPass::runOnFunc(Function &F, FunctionAnalysisManager &AM) {
  if (F.getName().startswith(interceptedFunctionPrefix))
    return;

  // Code is based on AddressSanitizer and ASan--

  // The load/store pointers to be instrumented
  SmallVector<InterestingMemoryOperand, 16> OperandsToInstrument;
  // Temp vector to track only instrumenting every address once per basic block
  SmallPtrSet<Value *, 16> TempsToInstrument;

  for (BasicBlock &BB : F) {
    TempsToInstrument.clear();

    for (Instruction &I : BB) {
      // load + store
      SmallVector<InterestingMemoryOperand, 1> InterestingOperands;
      getInterestingMemoryOperands(I, InterestingOperands);

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

  // Instrument remaining load/store targets with checks
  for (auto &MO : OperandsToInstrument){
    // insertion) writes: before, reads: after
    insertFloatzoneCheck(*MO.getInsn(), *MO.getPtr(), MO.IsWrite, MO.OpType);
  }

  // Run some optimization passes to simplify instrumentation: only if not -O0
  if(!F.hasOptNone()){
    AM.clear();
    InstCombinePass().run(F, AM);
    AM.clear();
    SimplifyCFGPass().run(F, AM);
  }

  // We need to replace meminstrinsics after running instcombine, otherwise it gets reverted
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
  // Replace mem intrinsics with calls (to avoid un-instrumented LLVM calls)
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

static void createGlobalMemFamilyPointers(Module &M){
  LLVMContext &C = M.getContext();
  IRBuilder<> builder(C);
  int LongSize = M.getDataLayout().getPointerSizeInBits();
  IntegerType* IntptrTy = Type::getIntNTy(C, LongSize);
  CheckfloatMemmove = M.getOrInsertFunction(interceptedFunctionPrefix + "memmove",
                                       builder.getInt8PtrTy(), builder.getInt8PtrTy(),
                                       builder.getInt8PtrTy(), IntptrTy);
  CheckfloatMemcpy = M.getOrInsertFunction(interceptedFunctionPrefix +"memcpy",
                                      builder.getInt8PtrTy(), builder.getInt8PtrTy(),
                                      builder.getInt8PtrTy(), IntptrTy);
  CheckfloatMemset = M.getOrInsertFunction(interceptedFunctionPrefix +"memset",
                                      builder.getInt8PtrTy(), builder.getInt8PtrTy(),
                                      builder.getInt32Ty(), IntptrTy);
}

PreservedAnalyses CheckFloatPass::run(Module &M, ModuleAnalysisManager &MAM){
  if (!hasMode("cmp_zone") || !hasMode("float"))
    return PreservedAnalyses::none();

#ifdef DEFAULTCLANG
    return PreservedAnalyses::none();
#endif

  auto &FAM = MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

  // Create global mem* family pointers
  createGlobalMemFamilyPointers(M);

  for (Function &F : M){
    if (!F.isDeclaration())
      runOnFunc(F, FAM);
  }

  return PreservedAnalyses::none();
}

