#include "llvm/Transforms/Utils/FloatZone.h"
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

// MACROS
#define FLOATZONE_DEBUG 0
#define USE_BUILTINS 0
#define DISABLE_OPT_FOR_REZZAN 0

// Redzone size in bytes.
constexpr unsigned RedzoneSize = 16;
static GlobalVariable* RedzoneArray = nullptr;

// Redzone poison value array
#define FLOAT_BYTE_ARRAY  0x8b, 0x8b, 0x8b, 0x8b, \
                          0x8b, 0x8b, 0x8b, 0x8b, \
                          0x8b, 0x8b, 0x8b, 0x8b, \
                          0x8b, 0x8b, 0x8b, 0x8b

// Declare SafeStack function to call
llvm::DenseSet<AllocaInst *> getUnsafeAlloca(Function &F, ScalarEvolution &SE);

#if USE_BUILTINS == 1
// Library function to check poison on mem family builtins
FunctionCallee CheckPoison;
#else
// Mem family functions (to avoid calling LLVM intrinsics)
FunctionCallee FloatzoneMemmove, FloatzoneMemcpy, FloatzoneMemset;
#endif

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


namespace {
/// Represents an Alloca with redzone instrumentation afterwards.
class InstrumentedAlloca {
  Function *F = nullptr;
  /// The instrumented alloca that has space for the redzone.
  AllocaInst *Alloca = nullptr;
  /// The element value in the array of Alloca where the redzone starts.
  Value *RedzoneStartElem = nullptr;
  /// Instructions where unpoison instrumentation was added before.
  std::vector<Instruction *> unpoisonPoints;
public:
  bool poisonedByStart = false;
  InstrumentedAlloca(Function &F, AllocaInst *A, Value *RedZoneStart, Value *ObjStart)
    : F(&F), Alloca(A), RedzoneStartElem(RedZoneStart) {
    assert(A && RedZoneStart);
  }

  AllocaInst* getAlloca(){
    return Alloca;
  }

  [[nodiscard]]
  Value* getOverflowStart(IRBuilder<>& Builder, LLVMContext &C){
    // RedzoneStartElem == old size of array in bytes
    Value *oldSize = RedzoneStartElem;
    Value* offsetToOverflow = Builder.CreateAdd(ConstantInt::get(oldSize->getType(), 16), oldSize);
    std::vector<Value *> indizes = {offsetToOverflow};
    return Builder.CreateInBoundsGEP(Alloca->getAllocatedType(), Alloca, indizes);
  }

  /// Return true if the given Value is the respective alloca.
  [[nodiscard]]
  bool isAlloca(Value *A) const {
    return Alloca == A->stripPointerCastsAndAliases();
  }

  /// Return pointer to where redzone starts.
  [[nodiscard]]
  Value *getStartOfZone(IRBuilder<>& Builder, LLVMContext &C) {
    std::vector<Value *> Indizes = {
      RedzoneStartElem,
    };
    return Builder.CreateInBoundsGEP(Alloca->getAllocatedType(), Alloca, Indizes);
  }

  /// Fills redzone with the given byte (i8) value.
  void fillRedzone(Instruction *When, bool redzone) {
    // We can only touch the alloca when it dominates our use. Otherwise
    // this violates SSA.
    DominatorTree DT(*F);
    if (!DT.dominates(Alloca, When))
      return;

    assert(When);
    LLVMContext &C = F->getContext();

    IRBuilder<> Builder(When);
    Constant *Size = ConstantInt::get(Type::getInt64Ty(C), RedzoneSize);
    Constant *Null = ConstantInt::get(Type::getInt8Ty(C), 0x0);
    Instruction *meta;

    if(hasMode("double_sided")){
      Value *overflowStart = getOverflowStart(Builder, C);
#if 0
      // inline asm to memcpy redzones (llvm calls were loading the global twice while this is not necessary)
      if(redzone){
        InlineAsm *IA = InlineAsm::get(
                  FunctionType::get(llvm::Type::getVoidTy(C), {RedzoneArray->getType(), getAlloca()->getType(), overflowStart->getType()}, false),
                  StringRef("movaps $0, %xmm0\nmovaps %xmm0, $1\nmovups %xmm0, $2\n"),
                  StringRef("p,p,p,~{xmm0},~{dirflag},~{fpsr},~{flags}"),
                  /*hasSideEffects=*/ true,
                  /*isAlignStack*/ false,
                  InlineAsm::AD_ATT,
                  /*canThrow*/ false);
          std::vector<llvm::Value*> args = { RedzoneArray, getAlloca(), overflowStart };
          Builder.CreateCall(IA, args);
      }
      else{
        Builder.CreateMemSetInline(getAlloca(), MaybeAlign(), Null, Size, /*isVolatile=*/true);
        Builder.CreateMemSetInline(overflowStart, MaybeAlign(), Null, Size, /*isVolatile=*/true);
      }
#else
      if(redzone) {
        // under
        meta = Builder.CreateMemCpyInline(getAlloca(), Align(16), RedzoneArray, Align(16), Size, false);
        meta->setMetadata(F->getParent()->getMDKindID("floatzone"), llvm::MDNode::get(C, None));
        // over
        meta = Builder.CreateMemCpyInline(overflowStart, MaybeAlign(), RedzoneArray, Align(16), Size, false);
        meta->setMetadata(F->getParent()->getMDKindID("floatzone"), llvm::MDNode::get(C, None));
      } else {
        // under
        meta = Builder.CreateMemSetInline(getAlloca(), Align(16), Null, Size, /*isVolatile=*/true);
        meta->setMetadata(F->getParent()->getMDKindID("floatzone"), llvm::MDNode::get(C, None));
        // over
        meta = Builder.CreateMemSetInline(overflowStart, MaybeAlign(), Null, Size, /*isVolatile=*/true);
        meta->setMetadata(F->getParent()->getMDKindID("floatzone"), llvm::MDNode::get(C, None));
      }
#endif      
    }
    else{
      Value *startOfZone = getStartOfZone(Builder, C);
      Value *castedStart = Builder.CreateBitCast(startOfZone, Type::getInt8PtrTy(C));

      if(redzone){
        // apply redzone 
        meta = Builder.CreateMemCpyInline(castedStart, MaybeAlign(), RedzoneArray, Align(16), Size, true);
        meta->setMetadata(F->getParent()->getMDKindID("floatzone"), llvm::MDNode::get(C, None));
      }
      else{
        // remove redzone
        meta = Builder.CreateMemSetInline(castedStart, MaybeAlign(), Null, Size, /*isVolatile=*/true);
        meta->setMetadata(F->getParent()->getMDKindID("floatzone"), llvm::MDNode::get(C, None));
      }
    }

  }

  /// Add poison to redzone *after* the given instruction 'When'
  void poisonRedzone(Instruction *When) {
    //LLVMContext &C = F->getContext();

    // Fill redzone with magic value after the alloca.
    fillRedzone(When->getNextNode(), true);
  }

  /// Remove poison from redzone *before* instruction 'When'.
  void unpoisonRedzone(Instruction *When) {
    unpoisonPoints.push_back(When);
    //LLVMContext &C = F->getContext();

    // Clear the redzone
    fillRedzone(When, false);
  }

  /// Returns true if the alloca was already unpoisoned at Instruction 'When'.
  [[nodiscard]]
  bool alreadyUnpoisoned(Instruction *When, const DominatorTree &F) const {
    for (Instruction *I : unpoisonPoints)
      if (F.dominates(I, When))
        return true;
    return false;
  }
};
}

/// \param I The insertion point
/// \param addr The pointer the load/store is accessing
/// \param before Insert before or after the insertion point
/// \param ptrType Underlying type of the load/store access
void insertFloatzoneCheck(Instruction &I, Value &addr, bool before, Type* ptrType) {
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

  // TODO: Work with other access sizes
  if (size != 8) {
    return;
  }

  // The pointer we want to check
  Value* load = &addr;
  
  if(hasMode("just_size")){
    Value *off_size = ConstantInt::get(C, APInt(t->getBitWidth(), size-1));
#if 1
    // Offset the to-be-checked address by the access size
    std::vector<Value *> indizes = {off_size};
    load = builder.CreateInBoundsGEP(builder.getInt8Ty(), load, indizes);
#else // slow add
    // Cast original pointer to int
    Value *ptrInt = builder.CreatePtrToInt(load, t);
    // Offset the to-be-checked address by the access size
    load = builder.CreateAdd(ptrInt, off_size);
#endif
  }

  // Prepare assembly string
  std::string asm_string = R"(
    // Prepare indirect jumps
    leaq spec_signal(%rip), %r15
    leaq 1f(%rip), %r14

    // Compare loaded value against reference
    movabsq $$0x8b8b8b8b, %r13
    cmpq %r13, $0

    // Jump based on comparison result
    cmovne %r14, %r15
    jmp *%r15

    // Exit label
    1:
  )";

  // Build inline assembly
  InlineAsm *IA = InlineAsm::get(FunctionType::get(Type::getVoidTy(C), false), asm_string, "r", true);

  // Insert the inline assembly
  builder.CreateCall(IA, {&I});
}

void sequentialExecuteOptimizationPostDom(Function &F, SmallVector<InterestingMemoryOperand, 16> &OperandsToInstrument,
                                          AliasAnalysis *AA) {

  auto PDT = PostDominatorTree();
  PDT.recalculate(F);
  std::map<Value *, std::set<InterestingMemoryOperand *>> AddrToInstructions;

  // pre-processing
  // group instructions that access the same address (alias considered)
  for (InterestingMemoryOperand &Operand : OperandsToInstrument) {
    if (Value *Addr = Operand.getPtr()) {

      if (AddrToInstructions.find(Addr) == AddrToInstructions.end()) {

        bool aliasFound = false;
        //handle the possibility of alias
        for (auto item : AddrToInstructions) {
          if (AA->isMustAlias(item.first, Addr)) {
            aliasFound = true;
            AddrToInstructions[item.first].insert(&Operand);
            break;
          }
        }
        //found an alias, done
        if (aliasFound) continue;

        //never appeared in the map, so add a slot
        AddrToInstructions.insert(std::pair<Value *, std::set<InterestingMemoryOperand *>>(Addr, std::set<InterestingMemoryOperand *>()));
      }
      //add the inst to the target slot (either the newly created one or an existing one)
      AddrToInstructions[Addr].insert(&Operand);
    }
  }

  std::set<Instruction *> deleted;

  for (auto item : AddrToInstructions) {
    for (auto inst1 : item.second) {
      //well, the instruction has been deleted, so who cares
      if (deleted.find(inst1->getInsn()) != deleted.end())
        continue;

      for (auto inst2 : item.second) {
        //avoid checking itself
        if (inst1->getInsn() == inst2->getInsn() || deleted.find(inst2->getInsn()) != deleted.end())
          continue; 

        if (PDT.dominates(inst1->getInsn()->getParent(), inst2->getInsn()->getParent())){
          deleted.insert(inst2->getInsn());
        }
      }
    }
  }
  //Let's only keep the non-deleted ones
  SmallVector<InterestingMemoryOperand, 16> SEOTempToInstrument(OperandsToInstrument);
  OperandsToInstrument.clear();

  for (auto item: SEOTempToInstrument) {
    if (deleted.find(item.getInsn()) == deleted.end())
      OperandsToInstrument.push_back(item);
  }
}

bool isPostDominatWrapper(Instruction *InstStart, Instruction *TargetInst, llvm::PostDominatorTree &PDT) {
  
  BasicBlock *StartBB = InstStart->getParent();
  BasicBlock *TargetBB = TargetInst->getParent();
  if (StartBB == TargetBB) {
    for (auto &itrInst : *StartBB) {
      if (&itrInst == InstStart) {
        return false;
      }
      if (&itrInst == TargetInst) {
        return true;
      }
    }
  }
  return PDT.dominates(StartBB, TargetBB);
}

bool ConservativeCallIntrinsicCheck(Instruction *InstStart, Instruction *InstEnd, std::set<Instruction *> &callIntrinsicSet,
                                    llvm::DominatorTree &DT, llvm::PostDominatorTree &PDT) {

  for (auto TargetInst : callIntrinsicSet) {
    // InstStart -> TargetInst -> InstEnd && InstStart !PostDominat TargetInst
    if (isPotentiallyReachable(InstStart, TargetInst) && isPotentiallyReachable(TargetInst, InstEnd) && !isPostDominatWrapper(InstStart, TargetInst, PDT)) {
      return false;
    }
  }
  return true;
}

void ConservativeCallIntrinsicCollect(Function &F, std::set<Instruction *> &callIntrinsicSet) {

  for (auto &BB : F) {
    for (auto &Inst : BB) {
      // Here we check if current instruction is call instruction
      if (isa<CallInst>(&Inst)) {
        callIntrinsicSet.insert(&Inst);
        continue;
      }
      IntrinsicInst *II = dyn_cast<IntrinsicInst>(&Inst);
      // Here we check if Intrinsic ID is lifetime_end
      if (II && II->getIntrinsicID() == Intrinsic::lifetime_end) {
        callIntrinsicSet.insert(&Inst);
        continue;
      }
    }
  }
}

void sequentialExecuteOptimization(Function &F, SmallVector<InterestingMemoryOperand, 16> &OperandsToInstrument,
                                   AliasAnalysis *AA) {

  auto DT = DominatorTree(F);
  std::map<Value *, std::set<InterestingMemoryOperand *>> AddrToInstructions;
  auto PDT = PostDominatorTree();
  PDT.recalculate(F);

  // pre-processing
  // group instructions that access the same address (alias considered)
  for (InterestingMemoryOperand &Operand : OperandsToInstrument) {
    if (Value *Addr = Operand.getPtr()) {

      if (AddrToInstructions.find(Addr) == AddrToInstructions.end()) {

        bool aliasFound = false;
        //handle the possibility of alias
        for (auto item : AddrToInstructions) {
          if (AA->isMustAlias(item.first, Addr)) {
            aliasFound = true;
            AddrToInstructions[item.first].insert(&Operand);
            break;
          }
        }
        //found an alias, done
        if (aliasFound) continue;

        //never appeared in the map, so add a slot
        AddrToInstructions.insert(std::pair<Value *, std::set<InterestingMemoryOperand *>>(Addr, std::set<InterestingMemoryOperand *>()));
      }
      //add the inst to the target slot (either the newly created one or an existing one)
      AddrToInstructions[Addr].insert(&Operand);
    }
  }

  std::set<Instruction *> deleted;

  std::set<Instruction *> callIntrinsicSet; 

  ConservativeCallIntrinsicCollect(F, callIntrinsicSet);

  for (auto item : AddrToInstructions) {
    for (auto inst1 : item.second) {
      //well, the instruction has been deleted, so who cares
      if (deleted.find(inst1->getInsn()) != deleted.end())
        continue;

      for (auto inst2 : item.second) {
        //avoid checking itself
        if (inst1->getInsn() == inst2->getInsn() || deleted.find(inst2->getInsn()) != deleted.end())
          continue; 

        if (DT.dominates(inst1->getInsn(), inst2->getInsn()) && ConservativeCallIntrinsicCheck(inst1->getInsn(), inst2->getInsn(), callIntrinsicSet, DT, PDT)){
          deleted.insert(inst2->getInsn());
        }
      }
    }
  }
  //Let's only keep the non-deleted ones
  SmallVector<InterestingMemoryOperand, 16> SEOTempToInstrument(OperandsToInstrument);
  OperandsToInstrument.clear();

  for (auto item: SEOTempToInstrument) {
    if (deleted.find(item.getInsn()) == deleted.end())
      OperandsToInstrument.push_back(item);
  }
}

// isSafeAccess returns true if Addr is always inbounds with respect to its
// base object. For example, it is a field access or an array access with
// constant inbounds index.
bool isSafeAccess(ObjectSizeOffsetVisitor &ObjSizeVis,
                                    Value *Addr, uint64_t TypeSize) {
  SizeOffsetType SizeOffset = ObjSizeVis.compute(Addr);
  if (!ObjSizeVis.bothKnown(SizeOffset)) return false;
  uint64_t Size = SizeOffset.first.getZExtValue();
  int64_t Offset = SizeOffset.second.getSExtValue();
  // Three checks are required to ensure safety:
  // . Offset >= 0  (since the offset is given from the base ptr)
  // . Size >= Offset  (unsigned)
  // . Size - Offset >= NeededSize  (unsigned)
  return Offset >= 0 && Size >= uint64_t(Offset) &&
         Size - uint64_t(Offset) >= TypeSize / 8;
}

//ASAN--: Removing Unsatisfiable Checks
bool isSafeAccessBoost(ObjectSizeOffsetVisitor &ObjSizeVis, Instruction *IndexInst, Value *Addr, Function *F) {
  auto DT = DominatorTree(*F);
  if (GetElementPtrInst *Gep_Inst = dyn_cast<GetElementPtrInst>(Addr)) {
    for (auto& Index : make_range(Gep_Inst->idx_begin(), Gep_Inst->idx_end())) {
      for (User *U : Index->users()) {    
        if (CmpInst *i_cmp = dyn_cast<CmpInst>(U)) {
          if (DT.dominates(i_cmp, IndexInst)) {
            if (Index == i_cmp->getOperand(0) && isa<ConstantData>(i_cmp->getOperand(1))) {
              auto IndexSize = i_cmp->getOperand(1);
              auto ConstantSize = dyn_cast<ConstantInt>(IndexSize);
              int64_t MaxOffset = ConstantSize->getSExtValue();
              auto type = Gep_Inst->getPointerOperandType();

              if (isa<PointerType>(type)) {
                auto pttpee = Gep_Inst->getSourceElementType();
                if (isa<ArrayType>(pttpee)) {
                  auto ObjSize = pttpee->getArrayNumElements();
                  return static_cast<int64_t>(ObjSize) >= MaxOffset;
                }
              }
              if (isa<ArrayType>(type)) {
                auto ObjSize = type->getArrayNumElements();
                return static_cast<int64_t>(ObjSize) >= MaxOffset;
              }
            }

            if (Index == i_cmp->getOperand(1) && isa<ConstantData>(i_cmp->getOperand(0))) {
              auto IndexSize = i_cmp->getOperand(0);
              auto ConstantSize = dyn_cast<ConstantInt>(IndexSize);
              int64_t MaxOffset = ConstantSize->getSExtValue();
              auto type = Gep_Inst->getPointerOperandType();

              if (isa<PointerType>(type)) {
                auto pttpee = Gep_Inst->getSourceElementType();
                if (isa<ArrayType>(pttpee)) {
                  auto ObjSize = pttpee->getArrayNumElements();
                  return static_cast<int64_t>(ObjSize) >= MaxOffset;
                }
              }
              if (isa<ArrayType>(type)) {
                auto ObjSize = type->getArrayNumElements();
                return static_cast<int64_t>(ObjSize) >= MaxOffset;
              }
            }
          }
        }
      } 
    }
  }
  return false;
}

void unsatChecksOptimization(ObjectSizeOffsetVisitor &ObjSizeVis, SmallVector<InterestingMemoryOperand, 16> &OperandsToInstrument)
{
  SmallVector<InterestingMemoryOperand, 16> TempToInstrument(OperandsToInstrument);
  TempToInstrument.clear();
  bool toKeep;

  for (InterestingMemoryOperand &O : OperandsToInstrument) {
    Value *Addr = O.getPtr();
#if 0 // disable buggy optimization
    Instruction *Insn = O.getInsn();
#endif
    toKeep = true;

    //If global variable
    if(dyn_cast<GlobalVariable>(getUnderlyingObject(Addr))) {
      if(isSafeAccess(ObjSizeVis, Addr, O.TypeSize)) {
        // errs() << "GLOB_IS_SAFE_ACC " << O.toString() << "\n";
        toKeep = false;
      }
#if 0 // disable buggy optimization
      if(isSafeAccessBoost(ObjSizeVis, Insn, Addr, Insn->getFunction())) {
        // errs() << "GLOB_IS_SAFE_ACC_BOOST " << O.toString() << "\n";
        toKeep = false;
      }
#endif
    }

    //If stack variable
    if (isa<AllocaInst>(getUnderlyingObject(Addr))) {
      if(isSafeAccess(ObjSizeVis, Addr, O.TypeSize)) {
        // errs() << "STACK_IS_SAFE_ACC " << O.toString() << "\n";
        toKeep = false;
      }
#if 0 // disable buggy optimization
      if(isSafeAccessBoost(ObjSizeVis, Insn, Addr, Insn->getFunction())) {
        // errs() << "STACK_IS_SAFE_ACC_BOOST " << O.toString() << "\n";
        toKeep = false;
      }
#endif
    }

    if(toKeep) {
      TempToInstrument.push_back(O);
    }
  }

  //Delete the instructions to avoid
  OperandsToInstrument.clear();
  for (auto item: TempToInstrument) {
    OperandsToInstrument.push_back(item);
  }
}

enum addrType loopOptimizationCategorise(Function &F, Loop *L, InterestingMemoryOperand Oper, ScalarEvolution *SE) {

  std::vector<Value *> backs;
  std::vector<Value *> processedAddr;

  if (Value* addr = Oper.getPtr()) {
    btraceInLoop(addr, backs, L);
    return checkAddrType(addr, backs, processedAddr, SE, L);
  }
  return UNKNOWN; 
}

void LoopInvariantOptimizationFloatzone(Loop *L, std::set<Instruction *> &optimized, 
                        Function &F, InterestingMemoryOperand &Oper, ObjectSizeOffsetVisitor &ObjSizeVis, bool UseCalls) {
  auto DT = DominatorTree(F);
  auto ExitBB = L->getExitBlock();
  Value *addr = Oper.getPtr();
  uint64_t TypeSize = Oper.TypeSize;
  if (!addr) { return; }
  if (!ExitBB) { return; }

  auto exitInst = (*ExitBB).getFirstNonPHI();

  // If the instruction dominates the Loop Exit:
  if (DT.dominates(Oper.getInsn(), ExitBB)) {
    // If Load
    if (LoadInst *LI = dyn_cast<LoadInst>(Oper.getInsn())) {
      // Same as ASan-- : insert single float check right at the exit
      // errs() << "Dominate+Load: instrument target " << *Oper.getInsn() << " before exit: " << *exitInst << "\n";
      insertFloatzoneCheck(*exitInst, *LI->getPointerOperand(), true, LI->getType());
      optimized.insert(Oper.getInsn());
      return;
    }
    // If Store
    else if (StoreInst *SI = dyn_cast<StoreInst>(Oper.getInsn())) {
      // Insert single float check right before the loop starts
      // if the preheader is not known, the predecessor is likely an indirect branch
      // right now we do not optimize these cases
      BasicBlock* preheader = L->getLoopPreheader();
      if(preheader != nullptr){
        Instruction* preheaderExit = preheader->getTerminator();
        // The preheader can be NULL if the block is not well formed
        if(preheaderExit != nullptr){
          // the store ptr needs to dominate the preheader exit, otherwise it may not have a value to check at that time
          if(DT.dominates(SI->getPointerOperand(), preheaderExit) ){
            // errs() << "Dominate+Store: instrument target " << *Oper.getInsn() << " before loop-entry jump: " << *preheaderExit << "\n";
            insertFloatzoneCheck(*preheaderExit, *SI->getPointerOperand(), true, SI->getValueOperand()->getType());
            optimized.insert(Oper.getInsn());
          }
        }
      }

    }
  }
  // If NOT dominates:
  else{
    // If Load
    if (LoadInst *LI = dyn_cast<LoadInst>(Oper.getInsn())) {
      // Same as ASan-- : insert single float check after the loop wrapped in Tracer condition (if-executed)
      Module *M = F.getParent();
      int LongSize = M->getDataLayout().getPointerSizeInBits();
      LLVMContext *C = &(M->getContext());
      Type *IntptrTy = Type::getIntNTy(*C, LongSize);
      
      // Create local variable Tracer, and assign 0 as initial value
      IRBuilder<> IRBinit(F.getEntryBlock().getFirstNonPHI());
      Value *Tracer = IRBinit.CreateAlloca(IntptrTy, nullptr, "Tracer");
      IRBinit.CreateStore(ConstantInt::get(IntptrTy, 0), Tracer);

      // Assign memory access address to the Tracer
      IRBuilder<> IRBassign(Oper.getInsn());
      Value *AddrCast = IRBassign.CreatePointerCast(addr, IntptrTy);
      IRBassign.CreateStore(AddrCast, Tracer);

      // Check the Tracer value to decide add ASan check or not.
      IRBuilder<> IRBcheck(exitInst);
      Value *LITracer = IRBcheck.CreateLoad(IntptrTy, Tracer);
      Value *Cmp = IRBcheck.CreateICmpNE(LITracer, ConstantInt::get(IntptrTy, 0));
      Instruction *CheckTerm = SplitBlockAndInsertIfThen(Cmp, exitInst, false);

      // Check if the tracer pointer is a global variable that is proven to be safe
      if (isa<GlobalVariable>(getUnderlyingObject(LITracer)) &&
          isSafeAccess(ObjSizeVis, LITracer, TypeSize)) {
        optimized.insert(Oper.getInsn());
        return;
      }
      // A direct inbounds access to a stack variable is always valid.
      if (isa<AllocaInst>(getUnderlyingObject(LITracer)) &&
          isSafeAccess(ObjSizeVis, LITracer, TypeSize)) {
        optimized.insert(Oper.getInsn());
        return;
      }

      // Instrument: before 'CheckTerm', address 'LITracer'
      // errs() << "Non-DOM+Load: instrument target " << *LITracer << " before check: " << *CheckTerm << "\n";
      insertFloatzoneCheck(*CheckTerm, *LITracer, true, LI->getType());

      // Condition to handle current loop is outer most loop
      if (L->getParentLoop() == nullptr) {
        optimized.insert(Oper.getInsn());
        return;
      }

      // If current loop is inner loop, we will re-init the tracer to 0
      IRBuilder<> IRBreInit(exitInst);
      IRBreInit.CreateStore(ConstantInt::get(IntptrTy, 0), Tracer);
      optimized.insert(Oper.getInsn());
      return;
    }
    // If Store
    // Skip. FloatZone limitation (would require evaluating all conditions)
  }
}

void loopOptimization(Function &F, SmallVector<InterestingMemoryOperand, 16> &OperandsToInstrument, LoopInfo *LI, ScalarEvolution *SE, ObjectSizeOffsetVisitor &ObjSizeVis, bool UseCalls) {
  std::set<Instruction *> optimized; 
  for (auto Oper : OperandsToInstrument) {
    // Check if current instruction is inside loop
    if (Loop *L = LI->getLoopFor(Oper.getInsn()->getParent())) {
      // We will categorise the type of optimization
      if (loopOptimizationCategorise(F, L, Oper, SE) == IBIO) {
        LoopInvariantOptimizationFloatzone(L, optimized, F, Oper, ObjSizeVis, UseCalls);
      }
    } 
  }

  // errs() << "Original Size: " << OperandsToInstrument.size() << "\n";
  SmallVector<InterestingMemoryOperand, 16> LOTempToInstrument(OperandsToInstrument);
  OperandsToInstrument.clear();

  // errs() << "Remove Size: " << optimized.size() << "\n";

  for (auto item: LOTempToInstrument) {
    if (optimized.find(item.getInsn()) == optimized.end())
      OperandsToInstrument.push_back(item);
  }
  // errs() << "After Size: " << OperandsToInstrument.size() << "\n";
}

void slimAsanOptimization(Function &F, SmallVector<InterestingMemoryOperand, 16> &OperandsToInstrument,
                          AliasAnalysis *AA, LoopInfo *LI, ScalarEvolution *SE,
                          ObjectSizeOffsetVisitor &ObjSizeVis, bool UseCalls)
{
  //For better comparison with rezzan, we disable optimizations not present in rezzan
  #if DISABLE_OPT_FOR_REZZAN == 0
  // ASAN--: Removing Recurring Checks
  sequentialExecuteOptimizationPostDom(F, OperandsToInstrument, AA);
  sequentialExecuteOptimization(F, OperandsToInstrument, AA);
  #endif

  // ASAN--: Removing Unsatisfiable Checks
  unsatChecksOptimization(ObjSizeVis, OperandsToInstrument);

  // ASAN--: Optimizing Neighbor Checks -> not applicable to floatzone
  #if DISABLE_OPT_FOR_REZZAN == 0
  // ASAN--: Optimizing Checks in Loops
  loopOptimization(F, OperandsToInstrument, LI, SE, ObjSizeVis, UseCalls);
  #endif
}


/// Replace all allocas in the given Function with allocas that have redzones.
static std::vector<InstrumentedAlloca> createInstrumentedAllocas(Function &F, const llvm::DenseSet<AllocaInst *> &unsafe) {
  const DataLayout &DL = F.getParent()->getDataLayout();

  std::vector<AllocaInst *> allocas;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (auto *alloca = dyn_cast<AllocaInst>(&I)){
        if (hasMode("disable_safe_stack") || unsafe.contains(alloca)){
          allocas.push_back(alloca);
        }
      }
    }
  }

  std::vector<InstrumentedAlloca> result;
  Type *Int64Ty = Type::getInt64Ty(F.getParent()->getContext());

  for (AllocaInst *A : allocas) {
    if(hasMode("double_sided")){
      /* Rezzan code base */
      IRBuilder<> builder(A);
      Value *numElem = A->getArraySize(); // number of elements allocated
      Type *Ty = A->getAllocatedType(); // type of element allocated
      if(Ty->isEmptyTy()) continue; // [0 x i8]

      const DataLayout &DL = F.getParent()->getDataLayout();
      Value *oldSize = builder.CreateMul(numElem, builder.getInt64(DL.getTypeAllocSize(Ty))); // oldsize = numelem * sizeof(elem)
      Value *newSize = builder.CreateAdd(oldSize, builder.getInt64(RedzoneSize*2)); // +16 underflow +16 overflow
      AllocaInst *Instrumented = builder.CreateAlloca(builder.getInt8Ty(), newSize);
      Instrumented->setAlignment(Align(16));

      // %12 = getelementptr inbounds i8, i8* %11, i64 16
      std::vector<Value *> indizes = {ConstantInt::get(Int64Ty, 16)};
      Value *AllocBaseOffset = builder.CreateInBoundsGEP(Instrumented->getAllocatedType(), Instrumented, indizes);
      Value *AllocBasePtr = builder.CreateBitCast(AllocBaseOffset, A->getType()); // convert the pointer to the original pointer

      A->replaceAllUsesWith(AllocBasePtr);
      A->eraseFromParent();
      
      result.emplace_back(F, Instrumented, oldSize, AllocBasePtr);
    }
    else{
      IRBuilder<> builder(A->getNextNode());

      // Find the type T we're allocating.
      unsigned TypeSize = DL.getTypeSizeInBits(A->getAllocatedType()) / 8U;
      if (!TypeSize) {
        llvm::errs() << *A->getAllocatedType();
        if(A->getAllocatedType()->isEmptyTy()){ // [0 x i8]
          continue;
        }
      }
      assert(TypeSize);
      // Find out how many N of T we need for the redzone.
      // E.g., T = int16 = 2 Byte. So RedzoneElems = 16 / 2 = 8.
      unsigned RedzoneElems = RedzoneSize / TypeSize;
      // Round up elements if we can't divide it.
      // E.g., T = 3 Byte. So RedzoneElems = 16 / 3 = 5 (but we need 6 elements).
      if (RedzoneSize % TypeSize != 0)
        ++RedzoneElems;

      // Create an add: OldSize + RedzoneElems.
      Value *arraySize = A->getArraySize();
      IntegerType *sizeType = cast<IntegerType>(arraySize->getType());
      ConstantInt *redzonePadding = ConstantInt::get(sizeType, RedzoneElems);
      Value *newArraySize = builder.CreateAdd(arraySize, redzonePadding);

      // Create a new alloca with redzone to replace the old one.
      AllocaInst *Instrumented = builder.CreateAlloca(A->getAllocatedType(),
                                                      A->getAddressSpace(),
                                                      newArraySize);
      Instrumented->setAlignment(A->getAlign());
      A->replaceAllUsesWith(Instrumented);
      A->eraseFromParent();

      result.emplace_back(F, Instrumented, arraySize, nullptr);
    }
  }

  return result;
}
  
void insertStackRedzones(Function &F, FunctionAnalysisManager &AM){

  // Call into SafeStack to identify trivially safe stack variables
  TargetLibraryInfo &TLI = AM.getResult<TargetLibraryAnalysis>(F);
  DominatorTree DT(F);
  AssumptionCache AC(F);
  LoopInfo LI(DT);
  ScalarEvolution SE(F, TLI, AC, DT, LI);
  llvm::DenseSet<AllocaInst *> unsafe = getUnsafeAlloca(F, SE);

  // Add redzone to allocas
  std::vector<InstrumentedAlloca> allocas = createInstrumentedAllocas(F, unsafe);
  // Helper to find InstrumentedAlloca object for LLVM AllocaInst.
  auto getInstrumentedOrNull = [&allocas](Value *I) {
    InstrumentedAlloca* Error = nullptr;
    if (!I)
      return Error;
    I = I->stripPointerCastsAndAliases();
    I = getUnderlyingObject(I);

    for (InstrumentedAlloca &A : allocas)
      if (A.isAlloca(I))
        return &A;

    return Error;
  };

  // Additional instrumentation for lifetime starts/ends, returns, setjmp, exceptions
  // to make sure there are no old redzone remnants on the stack that would later cause false positives
  std::vector<IntrinsicInst *> starts;
  std::vector<IntrinsicInst *> ends;
  std::vector<ReturnInst *> rets;
  std::vector<CallInst *> setjmps;
  std::vector<Instruction *> throws;

  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
        if (II->getIntrinsicID() == Intrinsic::lifetime_end)
          ends.push_back(II);
        if (II->getIntrinsicID() == Intrinsic::lifetime_start){
          starts.push_back(II);
          }
      }

      if (auto *R = dyn_cast<ReturnInst>(&I))
        rets.push_back(R);

      if (auto *CI = dyn_cast<CallInst>(&I)){
          if(Function *Callee = CI->getCalledFunction()){
            if(Callee->getName() == "_setjmp" || Callee->getName() == "__sigsetjmp"){
              setjmps.push_back(CI);
            }
          }
      }
      else if (auto *LP = dyn_cast<LandingPadInst>(&I)) {
          throws.push_back(LP);
      }
    }
  }

  // Poison the respective alloca after each lifetime start.
  for (Instruction *I : starts) {
    if (InstrumentedAlloca *A = getInstrumentedOrNull(I->getOperand(1))){
      A->poisonRedzone(/*When=*/I);
      A->poisonedByStart = true;
      }
  }

  // Unpoison the respective alloca after each lifetime end.
  for (Instruction *I : ends) {
    if (InstrumentedAlloca *A = getInstrumentedOrNull(I->getOperand(1)))
      A->unpoisonRedzone(/*When=*/I);
  }

  // Some allocas will not have life time markers. Let's poison them right after
  // the alloca instruction
  for(InstrumentedAlloca &A : allocas){
    if(!A.poisonedByStart){
      bool dominatesAllRets = true;
      DominatorTree DT(F);
      for(ReturnInst *ret : rets) {
        if (!DT.dominates(A.getAlloca(), ret)) {
          // We have found an Alloca that does not dominates all rets.
          // TODO for the time being we do not instrument this corner case
          dominatesAllRets = false;
          break;
        }
      }
      if(dominatesAllRets) {
        A.poisonRedzone(A.getAlloca());
      }
    }
  }

  DominatorTree domTree(F);

  // Check every return if there is an alloca that hasn't been unpoisoned
  // because of a previous lifetime.end.
  for (Instruction *Ret : rets) {
    for (InstrumentedAlloca &A : allocas) {
      // Check if a previous unpoison already happened.
      // E.g.:
      //   lifetime.end(alloca) <- poison removed here
      //   ret <- no need to poison again here
      if (A.alreadyUnpoisoned(Ret, domTree))
        continue;

      A.unpoisonRedzone(/*When=*/Ret);
    }
  }

  // %1 = tail call i64 asm "mov %rsp, $0 ", "=r,~{dirflag},~{fpsr},~{flags}"() #2, !srcloc !5
  const std::string &getRSP = "mov %rsp, $0 ";
  const std::string &getRSPConstraints = "=r,~{dirflag},~{fpsr},~{flags}";

  // Check every setjmp/exception landing pad: call restore stack func
  LLVMContext &C = F.getContext();
  Module *M = F.getParent();
  FunctionType* clearStackTy = FunctionType::get(Type::getVoidTy(C), {Type::getInt64Ty(C)}, false);
  FunctionCallee clearStack = M->getOrInsertFunction("clear_stack_on_jump", clearStackTy);
  auto *f = dyn_cast_or_null<Function>(clearStack.getCallee());
  f->setLinkage(GlobalValue::ExternalWeakLinkage);
  f->setUnnamedAddr(GlobalValue::UnnamedAddr::None);
  for (CallInst* jmp : setjmps) {
    // insert after jmp: call
    // check if return value of setjmp is nonzero (otherwise its the setting of env)
    IRBuilder<> builder(jmp->getNextNode());
    Value* cmp = builder.CreateICmp(CmpInst::Predicate::ICMP_NE, jmp, ConstantInt::getNullValue(Type::getInt32Ty(C)));
    Instruction* split = &*std::next(cast<Instruction>(cmp)->getIterator());

    // insert the call to clear stack only if the cmp result is true (i.e. setjmp != 0)
    // where != 0 implies the setjmp comes from a longjmp
    Instruction *endOfThen = SplitBlockAndInsertIfThen(cmp, split, /*unreachable*/false);
    builder.SetInsertPoint(endOfThen);

    // get RSP
    FunctionType* getRegAsmFuncTy = FunctionType::get(Type::getInt64Ty(C), {}, false);
    InlineAsm *getRSPAsm = InlineAsm::get(getRegAsmFuncTy, getRSP, getRSPConstraints, false);
    Value *rsp = builder.CreateCall(getRSPAsm, {}, "getRSPAsm");

    builder.CreateCall(clearStack, {rsp});
  }

  for (Instruction* t : throws) {
    IRBuilder<> builder(t->getNextNode());

    FunctionType* getRegAsmFuncTy = FunctionType::get(Type::getInt64Ty(C), {}, false);
    InlineAsm *getRSPAsm = InlineAsm::get(getRegAsmFuncTy, getRSP, getRSPConstraints, false);
    Value *rsp = builder.CreateCall(getRSPAsm, {}, "getRSPAsm");

    builder.CreateCall(clearStack, {rsp});
  }
}

void getInterestingMemoryOperands(Instruction &I, SmallVectorImpl<InterestingMemoryOperand> &Interesting) {

  if (I.hasMetadata("nosanitize")) {
    return;
  }

  if (LoadInst *LI = dyn_cast<LoadInst>(&I)) {
    Interesting.emplace_back(&I, LI->getPointerOperandIndex(), false,
                                      LI->getType(), LI->getAlign());
  }
  //TODO Missing atomic mem operations: AtomicRMWInst, AtomicCmpXchgInst, llvm.masked.load, llvm.masked.store
}

#if USE_BUILTINS == 1
bool isBuiltinMemTarget(Intrinsic::ID id){
  switch(id){
    case Intrinsic::memcpy:
    case Intrinsic::memcpy_inline:
    case Intrinsic::memmove:
    case Intrinsic::memset:
    case Intrinsic::memset_inline:
      return true;
    default: 
      return false;
  }
  return false;
}

void instrumentMemIntrinsics(Function &F, MemIntrinsic* MI){
  Module &M = *(F.getParent());
  int LongSize = M.getDataLayout().getPointerSizeInBits();
  IntegerType* IntptrTy = Type::getIntNTy(M.getContext(), LongSize);
  IRBuilder<> IRB(MI);

  if(isBuiltinMemTarget(MI->getIntrinsicID())){
    // void* memset(void *str, int c, size_t n)
    if (isa<MemSetInst>(MI)) { // includes inline
      // -> check_poison_visible(str, n)
      IRB.CreateCall(CheckPoison, {
        IRB.CreatePointerCast(MI->getOperand(0), IRB.getInt8PtrTy()),
        IRB.CreateIntCast(MI->getOperand(2), IntptrTy, false)
      });
    }
    // void *memcpy(void *dest, const void * src, size_t n)
    else if (isa<MemCpyInst>(MI)) { // includes inline
      // -> check_poison_visible(src, n)
      IRB.CreateCall(CheckPoison, {
        IRB.CreatePointerCast(MI->getOperand(1), IRB.getInt8PtrTy()),
        IRB.CreateIntCast(MI->getOperand(2), IntptrTy, false)
      });

      // -> check_poison_visible(dest, n)
      IRB.CreateCall(CheckPoison, {
        IRB.CreatePointerCast(MI->getOperand(0), IRB.getInt8PtrTy()),
        IRB.CreateIntCast(MI->getOperand(2), IntptrTy, false)
      });
    }
    // void* memmove(void *str1, const void *str2, size_t n)
    else if (isa<MemMoveInst>(MI)) {
      // -> check_poison_visible(str2, n)
      IRB.CreateCall(CheckPoison, {
        IRB.CreatePointerCast(MI->getOperand(1), IRB.getInt8PtrTy()),
        IRB.CreateIntCast(MI->getOperand(2), IntptrTy, false)
      });

      // -> check_poison_visible(str1, n)
      IRB.CreateCall(CheckPoison, {
        IRB.CreatePointerCast(MI->getOperand(0), IRB.getInt8PtrTy()),
        IRB.CreateIntCast(MI->getOperand(2), IntptrTy, false)
      });
    }
  }
}
#else

void replaceMemIntrinsics(Function &F, MemIntrinsic* MI){
  Module &M = *(F.getParent());
  int LongSize = M.getDataLayout().getPointerSizeInBits();
  IntegerType* IntptrTy = Type::getIntNTy(M.getContext(), LongSize);
  IRBuilder<> IRB(MI);
  
  if (isa<MemTransferInst>(MI)) {
    IRB.CreateCall(
      isa<MemMoveInst>(MI) ? FloatzoneMemmove : FloatzoneMemcpy,
      {IRB.CreatePointerCast(MI->getOperand(0), IRB.getInt8PtrTy()),
      IRB.CreatePointerCast(MI->getOperand(1), IRB.getInt8PtrTy()),
      IRB.CreateIntCast(MI->getOperand(2), IntptrTy, false)});
  } else if (isa<MemSetInst>(MI)) {
    IRB.CreateCall(
      FloatzoneMemset,
      {IRB.CreatePointerCast(MI->getOperand(0), IRB.getInt8PtrTy()),
      IRB.CreateIntCast(MI->getOperand(1), IRB.getInt32Ty(), false),
      IRB.CreateIntCast(MI->getOperand(2), IntptrTy, false)});
  } else {
    llvm_unreachable("Neither MemSet nor MemTransfer?");
  }
  MI->eraseFromParent();
}
#endif

void floatzoneUnsatOptimization(Function &F, SmallVector<InterestingMemoryOperand, 16> &OperandsToInstrument, 
                                DominatorTree &DT, AssumptionCache &AC, TargetLibraryInfo &TLI) {

  SmallVector<InterestingMemoryOperand, 16> TempToInstrument(OperandsToInstrument);
  TempToInstrument.clear();

  // if the pointer is not known to be dereferenceable -> we instrument it
  for (InterestingMemoryOperand &Oper : OperandsToInstrument) {
    Instruction *I = Oper.getInsn();
    if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
      if(!isDereferenceablePointer(LI->getPointerOperand(), LI->getType(), F.getParent()->getDataLayout(), I, &DT, &TLI)){
        TempToInstrument.push_back(Oper);
      }
    }
    else if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
      if(!isDereferenceablePointer(SI->getPointerOperand(), SI->getValueOperand()->getType(), F.getParent()->getDataLayout(), I, &DT, &TLI)){
        TempToInstrument.push_back(Oper);
      }
    }
  }

  // Delete the instructions to avoid
  OperandsToInstrument.clear();
  for (auto item: TempToInstrument) {
    OperandsToInstrument.push_back(item);
  }
}

const llvm::DenseSet<llvm::StringRef> functionsToIntercept = {
  "memcpy",
  "memset",
  "memmove",
  "strcmp",
  "strncmp",
  "memcmp",
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
const std::string interceptedFunctionPrefix = "floatzone_";

void maybeReplaceWithInterceptedFunction(llvm::Module &M, CallInst &i) {
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
void FloatZonePass::runOnFunc(Function &F, FunctionAnalysisManager &AM) {
  if (F.getName().startswith(interceptedFunctionPrefix))
    return;
#if FLOATZONE_DEBUG == 1
  errs()<<"Instrumenting "<<F.getName()<<"\n";
#endif

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

  #if FLOATZONE_DEBUG == 1
  errs()<<"-----------------BEFORE "<<OperandsToInstrument.size()<<"--------------------\n";
  for (InterestingMemoryOperand &MO : OperandsToInstrument) {
    errs() << MO.toString() << "\n";
  } 
  #endif

  //Collect all the analysis we can
  AAResults &AA = AM.getResult<AAManager>(F);
  TargetLibraryInfo &TLI = AM.getResult<TargetLibraryAnalysis>(F);
  DominatorTree DT(F);
  AssumptionCache AC(F);
  LoopInfo LI(DT);
  ScalarEvolution SE(F, TLI, AC, DT, LI);
  const DataLayout &DL = F.getParent()->getDataLayout();
  ObjectSizeOpts ObjSizeOpts;
  ObjSizeOpts.RoundToAlign = true;
  ObjectSizeOffsetVisitor ObjSizeVis(DL, &TLI, F.getContext(), ObjSizeOpts);

  // Instrument remaining load/store targets with checks
  for (auto &MO : OperandsToInstrument){
    // insertion) writes: before, reads: after
    insertFloatzoneCheck(*MO.getInsn(), *MO.getPtr(), MO.IsWrite, MO.OpType);
  }

  // Instrument stack variables with redzones
  insertStackRedzones(F, AM);

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
#if USE_BUILTINS == 1
    instrumentMemIntrinsics(F, memInst);
#else
    replaceMemIntrinsics(F, memInst);
#endif
  }

  #if FLOATZONE_DEBUG == 1
  errs()<<"-----------------AFTER "<<OperandsToInstrument.size()<<"--------------------\n";
  for (InterestingMemoryOperand &MO : OperandsToInstrument) {
    errs() << MO.toString() << "\n";
  } 
  #endif


  std::vector<CallInst *> calls;
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      if (CallInst *C = llvm::dyn_cast<CallInst>(&I))
        calls.push_back(C);

  for (CallInst *C : calls)
    maybeReplaceWithInterceptedFunction(*F.getParent(), *C);

  return;
}

void createGlobalMemFamilyPointers(Module &M){
  LLVMContext &C = M.getContext();
  IRBuilder<> builder(C);
  int LongSize = M.getDataLayout().getPointerSizeInBits();
  IntegerType* IntptrTy = Type::getIntNTy(C, LongSize);
#if USE_BUILTINS == 1
  FunctionType* CheckPoisonTy = FunctionType::get(Type::getVoidTy(C), {builder.getInt8PtrTy(), IntptrTy}, false);
  CheckPoison = M.getOrInsertFunction("check_poison_visible", CheckPoisonTy);
  auto *f = dyn_cast_or_null<Function>(CheckPoison.getCallee());
  f->setLinkage(GlobalValue::ExternalWeakLinkage);
  f->setUnnamedAddr(GlobalValue::UnnamedAddr::None);
#else
  FloatzoneMemmove = M.getOrInsertFunction(interceptedFunctionPrefix + "memmove",
                                       builder.getInt8PtrTy(), builder.getInt8PtrTy(),
                                       builder.getInt8PtrTy(), IntptrTy);
  FloatzoneMemcpy = M.getOrInsertFunction(interceptedFunctionPrefix +"memcpy",
                                      builder.getInt8PtrTy(), builder.getInt8PtrTy(),
                                      builder.getInt8PtrTy(), IntptrTy);
  FloatzoneMemset = M.getOrInsertFunction(interceptedFunctionPrefix +"memset",
                                      builder.getInt8PtrTy(), builder.getInt8PtrTy(),
                                      builder.getInt32Ty(), IntptrTy);
#endif
}

void createGlobalRedzoneArray(Module &M)
{
  LLVMContext &CTX = M.getContext();
  const uint8_t magic[16] = { FLOAT_BYTE_ARRAY };
  StringRef magicStr = StringRef((const char*)magic, RedzoneSize);
  llvm::Constant *floatarray = llvm::ConstantDataArray::getString(CTX, magicStr, false);

  // insert the global redzone value
  RedzoneArray = dyn_cast_or_null<GlobalVariable>(
      M.getOrInsertGlobal("__floatzone_redzone", floatarray->getType())
  );

  RedzoneArray->setLinkage(GlobalValue::PrivateLinkage);
  RedzoneArray->setAlignment(Align(16));
  RedzoneArray->setConstant(true);
  RedzoneArray->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);
  RedzoneArray->setInitializer(floatarray);
}

Type * wrapGlobalPackedType(GlobalVariable *G)
{
  LLVMContext &CTX = G->getContext();
  Type* Int8Ty = Type::getInt8Ty(CTX);
  Type* GlobalRedzoneTy = ArrayType::get(Int8Ty, RedzoneSize);

  // Retrieve the allocated type
  Type *Ty = G->getValueType();
  if (!Ty) return nullptr;

  // Create a new literal unpacked struct with the original type at index 0,
  // and the redzone at index 1 (offset size of original type)
  StructType *wrappedTy = StructType::get(CTX, {Ty, GlobalRedzoneTy}, false);
  assert(wrappedTy);

  return wrappedTy;
}

void insertGlobalsRedzones(Module &M){
  LLVMContext &CTX = M.getContext();
  Type* Int8Ty = Type::getInt8Ty(CTX);

  // Get initialiser for redzone
  std::vector<Constant*> psn(RedzoneSize-1, ConstantInt::get(Int8Ty, 0x8b, false));
  psn.insert(psn.begin(), ConstantInt::get(Int8Ty, 0x8b, false));
  Constant *redInit = ConstantArray::get(
      ArrayType::get(Int8Ty, RedzoneSize),
      psn
  ); assert(redInit);

  Constant *zero = ConstantInt::get(Type::getInt32Ty(M.getContext()), 0, false);
  Constant *indices[] = {zero, zero};

  // Create a quick list so that we're not modifying the list we're iterating over
  std::unordered_set<GlobalVariable*> globals;

  for (auto &G : M.getGlobalList()) {
      // Double check these aren't functions but variables
      if (isa<Function>(G)) continue;
      if (isa<GlobalIFunc>(G)) continue;

      // Make sure this global has a form of an initialiser and isn't a declaration
      if (!(G.hasInitializer())) continue;
      if (G.isExternallyInitialized()) continue;

      // Check again that this global isn't a declaration or other weird global
      if (G.isDeclaration()) continue;
      if (G.isDeclarationForLinker()) continue;
      if (G.hasSection()) continue;

      // Skip builtins
      if (G.hasName()) {
          if (G.getName().startswith(".llvm")) {
              continue;
          }
      }

      // TODO: Check validity hereof
      // Check if the types are valid
      Type *valTy = dyn_cast<Type>(G.getValueType());
      Type *ptrTy = dyn_cast<PointerType>(G.getType());
      if ((!valTy) || (!ptrTy)) continue;
      if (!(valTy->isSized())) continue;

      // We only want to wrap global variables local to this module
      switch(G.getLinkage()) {
          case GlobalValue::ExternalLinkage:
          case GlobalValue::InternalLinkage:
          case GlobalValue::PrivateLinkage:
          case GlobalValue::WeakAnyLinkage:
          case GlobalValue::WeakODRLinkage:
          {
              globals.insert(&G);
              break;
          }
          default:
          {
              break;
          }
      }
  }

  // For each global in the list we stored, wrap it in a new type with a redzone appended
  std::unordered_set<GlobalVariable *> deleted;

  for (auto *G : globals) {
      if (!G) continue;

      // Create a new name based on the old name
      StringRef gName = (G->hasName()) ? (G->getName()) : StringRef("");
      std::string newName = (gName.str()).append(".wrapped");

      // Get the original types (type of G is always a pointer but need it for casting later)
      //Type *origTy = dyn_cast<Type>(G->getValueType());
      PointerType *origTyPtr = dyn_cast<PointerType>(G->getType());
      assert(origTyPtr);

      // See if a wrapper exists for this type and otherwise make one
      Type *wrapperTy = wrapGlobalPackedType(G);
      if (!wrapperTy) {
          globals.erase(G);
          continue;
      }

      // Sanity checks
      assert(wrapperTy);
      assert(dyn_cast<StructType>(wrapperTy));
      assert(wrapperTy->getNumContainedTypes() == 2);
      //assert(wrapperTy->getContainedType(0) == origTy);

      // Get original initialiser
      Constant *origInit = G->getInitializer();
      assert(origInit);

      // Create a constant struct that initialises the original allocation to its original
      // initialiser and sets the redzone to zeros.
      Constant *wrapInit = ConstantStruct::get(
            dyn_cast<StructType>(wrapperTy), {
                origInit, redInit
            }
        );

      GlobalVariable *newG = new GlobalVariable(
          M, wrapperTy, G->isConstant(), G->getLinkage(),
          wrapInit, newName,
          G, G->getThreadLocalMode(),
          origTyPtr->getAddressSpace()
      ); assert(newG);

      // Copy some metadata and other information explicitly
      newG->copyAttributesFrom(G);
      newG->copyMetadata(G, 0);
      newG->setAlignment(MaybeAlign());

      // Sanity checks
      assert(newG->getValueType());
      assert(newG->getValueType() == wrapperTy);
      assert(newG->getValueType()->getNumContainedTypes() == 2);
      //assert(newG->getValueType()->getContainedType(0) == origTy);

      // Create a pointer (of the correct type) to the original object
      Constant *origGlobalPtr = ConstantExpr::getInBoundsGetElementPtr(
          wrapperTy, newG, indices
      ); assert(origGlobalPtr);

      // Replace all uses in the module with the pointer to the zeroth index of the new global
      G->replaceAllUsesWith(origGlobalPtr);

      // For external references to this global, also modify their references to the new global
      switch (G->getLinkage())
      {
          case llvm::GlobalValue::ExternalLinkage:
          case llvm::GlobalValue::WeakAnyLinkage:
          case llvm::GlobalValue::WeakODRLinkage:
          {
              // Update name to the new global
              std::string Asm(".globl ");
              Asm += G->getName();
              Asm += '\n';
              Asm += ".set ";
              Asm += G->getName();
              Asm += ", ";
              Asm += newG->getName();
              M.appendModuleInlineAsm(Asm);
              break;
          }
          default:
              break;
      }

      // Mark this global as should-be-deleted
      deleted.insert(G);
  }

  // Delete separately to avoid causing issues with iterating over the list you're modifying
  for (auto *G : globals) {
      if (!G)
          continue;
      G->eraseFromParent();
  }
}

/*
void createGlobalSignalSnippet(Module &M) {
  LLVMContext &C = M.getContext();
  IRBuilder<> builder(C);

  // Prepare assembly string
  std::string asm_string = R"(
  signal:
    // Trigger signal with division
    divsd %xmm0, %xmm0
    j signal
  )";
}
*/


PreservedAnalyses FloatZonePass::run(Module &M, ModuleAnalysisManager &MAM){
  if (!hasMode("floatzone"))
    return PreservedAnalyses::none();

#ifdef DEFAULTCLANG
    return PreservedAnalyses::none();
#endif

  auto &FAM = MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

  // Create global signal snippet

  // Create global mem* family pointers
  createGlobalMemFamilyPointers(M);

  // Instrument global variables with redzones
  insertGlobalsRedzones(M);

  // Create global redzone array (used for setting redzone bytes)
  createGlobalRedzoneArray(M);
  for (Function &F : M){
    if (!F.isDeclaration() && !F.hasFnAttribute(llvm::Attribute::DisableSanitizerInstrumentation)){
      runOnFunc(F, FAM);
    }
  }

  return PreservedAnalyses::none();
}

