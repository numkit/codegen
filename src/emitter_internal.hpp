// codegen/src/emitter_internal.hpp — shared internal declarations across emitter_*.cpp.
#pragma once

#include <numkit/codegen/emitter.hpp>
#include <numkit/codegen/indexing.hpp>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <iterator>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace numkit::codegen {
std::string cppArrayElemType(ValueType dtype);
}  // namespace numkit::codegen

namespace numkit::codegen::detail {

constexpr int kMaxFixpoint = 16;  // matches inference.cpp; lattice is shallow

// Per-array metadata for an array parameter or the single output array.
// 1-D vector orientation. The RawBuffer ABI erases it (a vector is just
// ptr+len), but the compile-time type knows it — used to fold size(vec, dim).
enum class VecOrient { Unknown, Row, Col };

struct ArrayInfo {
    ValueType   dtype;
    std::string lenVar;            // 1-D: the length EXPRESSION — `<name>_len`
                                   // for a param/output, `<name>.size()` for a local
    bool        isOutput = false;  // the function's caller-allocated out-param
    bool        is2D     = false;  // a 2-D matrix (column-major storage)
    VecOrient   orient   = VecOrient::Unknown;  // 1-D only (!is2D && !isND)
    std::string rowsVar, colsVar;  // 2-D: dim companions (`<name>_rows/_cols`)
    bool        isLocal  = false;  // an owned `std::vector` local (not a buffer ptr)
    std::string dataExpr;          // the element-pointer EXPRESSION — `<name>` for a
                                   // param/output buffer, `<name>.data()` for a local
    bool        isND     = false;  // a rank-N (N>=3) array (column-major flat storage)
    std::vector<std::string> ndDims;  // when isND: per-dim size EXPRESSIONS (rank = size())
    bool        ndRuntimeLocal = false;  // an N-D LOCAL whose dims are runtime
    bool        isRefParam = false;  // a `std::vector<T>&` reference out-param (do not hoist)
};

struct OptFacts {
    std::set<const ASTNode *>              promotedLoops;  // FOR nodes -> 0-based counter
    std::map<const ASTNode *, std::string> loopBound;      // FOR node -> C++ size_t bound
};

struct CallSite {
    const ASTNode             *def = nullptr;
    std::vector<InferredType>  argTypes;
    std::string                mangled;
    std::vector<ParamSpec>     extraSeed = {};
};

struct ProgramEmitCtx {
    const FunctionTable   *funcs = nullptr;
    std::vector<CallSite>  pending;
    std::set<std::string>  seen;
};

struct OneFn {
    std::string signature;
    std::string definition;  // signature + " { ... }"
};

bool isCppKeyword(const std::string &name);
std::string cppIdent(const std::string &name);
std::string cppGlobalIdent(const std::string &name);

std::string zeroLiteral(ValueType dtype);
const char *cppScalarBinOp(const std::string &op);
[[noreturn]] void unsupported(const std::string &what);
std::string emitBinOpJoin(const std::string &op, const std::string &l, const std::string &r);
std::string emitUnOpJoin(const std::string &op, const std::string &a);

bool is2DMatrixType(const InferredType &t);
bool isUnboxableScalarType(const InferredType &t);
VecOrient orientOf(const InferredType &t);
std::string structFieldLocal(const ASTNode &fa);
bool extractStructFieldChain(const ASTNode &fa, std::string &rootOut,
                             std::vector<std::string> &chainOut);
void setNestedStructField(StructLayout &layout, const std::vector<std::string> &chain,
                          std::size_t idx, const InferredType &leafType);
void syncIntermediateStructLocals(const std::string &prefix, const StructLayout &layout,
                                  TypeEnv &env);
bool isBufferArrayType(const InferredType &t);
bool isByValueReturnArrayType(const InferredType &t);
std::string cppObjectType(int classId, const ClassRegistry *classes);
OptFacts analyzeOptimizations(const ASTNode &body,
                              const std::unordered_map<std::string, ArrayInfo> &arrays);

std::string escapeBase(const std::string &base);
std::string typeCode(const InferredType &t);
std::string cEscape(const std::string &s, bool &ok);
std::string mangle(const std::string &base, const std::vector<InferredType> &args);
std::string companion(const std::string &base, const std::string &suffix);
std::vector<ArgInfo> toArgInfos(const std::vector<InferredType> &types);

const char *unaryMathStd(const std::string &name);
const char *opsTranscendentalFn(const std::string &name);
const char *binaryMathStd(const std::string &name);

extern const char *kPrelude;
std::string bridgePrelude(const std::string &runtimeHeader);
OneFn emitOneFunction(const ASTNode &funcDef, const std::vector<ParamSpec> &params,
                      const TransferRegistry &reg, ProgramEmitCtx *ctx,
                      const std::string &cppName, const ClassRegistry *classes,
                      const std::vector<ParamSpec> &extraSeed = {}, bool bridge = false,
                      bool opsKernels = false, bool interprocByValueReturn = false);

class Emitter {
public:
    Emitter(TypeEnv types, const TransferRegistry &reg,
            std::unordered_map<std::string, ArrayInfo> arrays, OptFacts opt,
            ProgramEmitCtx *ctx = nullptr, const ClassRegistry *classes = nullptr,
            bool bridge = false, bool opsKernels = false)
        : types_(std::move(types)), reg_(reg), arrays_(std::move(arrays)),
          opt_(std::move(opt)), ctx_(ctx), classes_(classes), bridge_(bridge),
          opsKernels_(opsKernels)
    {}

    // Tell the emitter how an early `return` statement should lower (it matches
    // the end-of-function return chosen in emitOneFunction).
    void setReturnInfo(bool returnsValue, std::string name, bool dynamic)
    {
        returnsValue_  = returnsValue;
        returnName_    = std::move(name);
        returnDynamic_ = dynamic;
    }

    void setEpilogueAssigns(std::vector<std::pair<std::string, std::string>> assigns)
    {
        epilogueAssigns_ = std::move(assigns);
    }

    void emitEpilogueAssigns()
    {
        for (const auto &[dst, src] : epilogueAssigns_)
            line(dst + " = " + src + ";");
    }

    void emitRawLine(const std::string &s) { line(s); }

    // Record a local that was hoisted as Dynamic (numkit_val); see dynamicLocals_.
    void addDynamicLocal(const std::string &n) { dynamicLocals_.insert(n); }

    // Hoist a local declaration at function entry (scalar or object).
    void hoistLocal(const std::string &name, const InferredType &t)
    {
        const std::string cname = cppIdent(name);
        if (t.isObject()) {
            const ClassInfo *ci = classes_ ? classes_->byId(t.classId) : nullptr;
            if (!ci) unsupported("object local '" + name + "' with no class registry");
            const std::string clsName = cppGlobalIdent(ci->name);
            if (ci->isHandle)
                line("numkit_rt::handle<" + clsName + "> " + cname + ";");  // null until assigned
            else
                line(clsName + " " + cname + "{};");                    // default-constructed
            return;
        }
        // Dynamic tier (DESIGN.md §10 C1): an un-typeable local is a boxed
        // runtime value (null until assigned). Reachable only under bridge_
        // (the caller's refusal check gates it); cppScalarType(Dynamic) throws.
        if (t.isDynamic()) {
            line("numkit_rt::val " + cname + ";");
            return;
        }
        line(cppScalarType(t.dtype) + " " + cname + " = " + zeroLiteral(t.dtype) + ";");
    }

    // Declare every owned-vector array local (deterministic order). Their
    // ArrayInfo carries isLocal; storage is a runtime-sized std::vector filled
    // by a later assignment (zeros/ones or a bridged array call).
    void hoistArrayLocals()
    {
        std::map<std::string, ValueType> ordered;
        for (const auto &[name, ai] : arrays_)
            if (ai.isLocal) ordered.emplace(name, ai.dtype);
        for (const auto &[name, dtype] : ordered)
            if (!arrays_.at(name).isRefParam)
                line("std::vector<" + cppArrayElemType(dtype) + "> " + cppIdent(name) + ";");
        // Runtime-dim N-D locals also need their size_t dim companions declared
        // (set at the zeros/ones assignment; read by indexN / size / numel).
        for (const auto &[name, dtype] : ordered) {
            const ArrayInfo &ai = arrays_.at(name);
            if (!ai.ndRuntimeLocal) continue;
            std::string decl = "std::size_t ";
            for (std::size_t d = 0; d < ai.ndDims.size(); ++d)
                decl += (d ? ", " : "") + ai.ndDims[d] + " = 0";
            line(decl + ";");
        }
    }

    void emitStmt(const ASTNode &s);
    void emitReturnScalar(const std::string &name) { line("return " + cppIdent(name) + ";"); }
    // Dynamic tier (DESIGN.md §10 C1): a boxed (numkit_val) return transfers handle
    // ownership OUT to the caller (val::take nulls the local so its dtor is a no-op).
    void emitReturnDynamic(const std::string &name) { line("return " + cppIdent(name) + ".take();"); }

    const std::string &out() const { return out_; }

private:
    void line(const std::string &s) { out_ += std::string(indent_ * 4, ' ') + s + "\n"; }
    void open(const std::string &head) { line(head + " {"); ++indent_; }
    void close() { --indent_; line("}"); }

    std::string emitExpr(const ASTNode &e);
    // Dynamic tier (DESIGN.md §10 C1): emit `e` as an `numkit_rt::val` C++
    // expression (a boxed runtime value), for an operand/result the inference
    // could not type. Operators dispatch to the runtime (binop/unop); an
    // un-typeable builtin result stays boxed (call_dyn). Only reachable under
    // bridge_ (it needs the C-ABI). The sound fallback to the typed emitExpr.
    std::string emitDynamicExpr(const ASTNode &e);
    // A control-flow condition (if / while): a typed condition is the bare C++
    // boolean; a Dynamic condition is evaluated in the Value tier and reduced to
    // the reference engine's truthiness (non-empty AND every element non-zero). DESIGN.md §10 C1.
    std::string emitCondition(const ASTNode &c);
    std::string emitBuiltinCall(const std::string &name, const ASTNode &call);
    std::string emitUserCall(const std::string &name, const ASTNode &call);
    std::string emitMethodCall(const ASTNode &call);
    std::string emitConstruct(const std::string &name, const ASTNode &call);
    void        appendCallArg(const ASTNode &arg, std::vector<InferredType> &argTypes,
                              std::string &argList);
    void        appendStructCallArgs(const std::string &prefix, const StructLayout &layout,
                                     std::string &argList);
    void        appendStructOutArgs(const std::string &prefix, const StructLayout &layout,
                                    std::string &argList);
    void        collectStructOutRefArrays(const std::string &prefix, const StructLayout &layout,
                                          std::set<std::string> &out) const;
    void        prepareCallInputAliases(const ASTNode &call, std::size_t firstArgIdx,
                                        const std::set<std::string> &outRefArrays,
                                        std::vector<std::string> &aliasDecls);
    void        appendIgnoreOutArg(const std::string &tmpName, const InferredType &ot,
                                   std::vector<std::string> &ignoreDecls, std::string &argList);
    void        emitStructCopy(const std::string &dstPrefix, const std::string &srcPrefix,
                               const StructLayout &layout);
    std::string emitMultiCallExpr(const std::string &mangled, const std::string &argList,
                                  const std::vector<InferredType> &outs);
    std::string emitIndexRead(const std::string &base, const ASTNode &call);
    // True if `e` is a pure ELEMENTWISE expression — literals, scalar vars,
    // whole arrays, and only elementwise ops (+ - .* ./ .\ .^, unary +/-).
    // Collects the distinct whole-array variable names into `arrays`. (Matrix
    // ops *,/,^ and calls/indexing make it false.)
    bool        collectElementwise(const ASTNode &e, std::set<std::string> &arrays) const;
    // The size_t expression for axis k (0-based) of array `a`: N-D companion /
    // 2-D rows,cols / 1-D orientation (row = 1 x len, col = len x 1). Axes
    // beyond the rank, or an orientation-unknown vector, yield "1" -- callers
    // that must distinguish unknown orientation guard on a.orient first.
    std::string dimExpr(const ArrayInfo &a, std::size_t k) const;
    void        emitAssign(const ASTNode &s);
    void        emitAssignIdentifierTail(const ASTNode &lhs, const ASTNode &rhs,
                                         const std::string &name,
                                         const std::string &structWriteFld);
    void        emitMultiAssign(const ASTNode &s);
    void        emitIndexWrite(const ASTNode &lhsCall, const ASTNode &rhs);
    void        emitFor(const ASTNode &s);
    void        emitWhile(const ASTNode &s);
    void        emitIf(const ASTNode &s);
    void        emitSwitch(const ASTNode &s);
    void        emitTry(const ASTNode &s);

    bool isArrayVar(const std::string &n) const { return arrays_.count(n) != 0; }

    std::string resolveArrayVar(const ASTNode &n) const
    {
        if (n.type == NodeType::IDENTIFIER && isArrayVar(n.strValue))
            return n.strValue;
        if ((n.type == NodeType::FIELD_ACCESS || n.type == NodeType::DYNAMIC_FIELD_ACCESS)
            && !n.children.empty() && n.children[0]) {
            const AbstractValue base = inferExpr(*n.children[0], types_, reg_, classes_);
            if (!base.type.isObject()) {
                const std::string fld = structFieldLocal(n);
                if (!fld.empty() && isArrayVar(fld)) return fld;
            }
        }
        return "";
    }

    void recordStructFieldWrite(const ASTNode &lhs, const std::string &fld,
                                const InferredType &ft)
    {
        types_.set(fld, {ft, ConstVal::unknown()});
        std::string              rootVar;
        std::vector<std::string> chain;
        if (!extractStructFieldChain(lhs, rootVar, chain)) return;
        const AbstractValue cur =
            types_.has(rootVar) ? types_.get(rootVar) : AbstractValue::dynamic();
        auto layout = std::make_shared<StructLayout>();
        if (cur.type.isStruct() && cur.type.structLayout)
            layout->fields = cur.type.structLayout->fields;
        setNestedStructField(*layout, chain, 0, ft);
        syncIntermediateStructLocals("_numkit_fld_" + rootVar, *layout, types_);
        types_.set(rootVar, {InferredType::structOf(std::move(layout)), ConstVal::unknown()});
    }

    // The AbstractValue for an array variable (non-scalar -> buffer).
    AbstractValue arrayValue(const std::string &n) const
    {
        return {InferredType::concrete(arrays_.at(n).dtype, Shape::rowVector()),
                ConstVal::unknown()};
    }

    std::string                                      out_;
    int                                              indent_           = 1;
    int                                              switchCounter_    = 0;  // unique switch-selector temp ids
    int                                              multiCallCounter_ = 0;  // unique multi-output expr temp ids
    int                                              aliasCounter_     = 0;  // unique call input-alias temp ids
    std::unordered_map<std::string, std::pair<std::string, std::string>> callAliasInputs_;
    TypeEnv                                          types_;
    const TransferRegistry                          &reg_;
    std::unordered_map<std::string, ArrayInfo>       arrays_;
    OptFacts                                         opt_;
    // Non-null inside a promoted clean-index loop: the 0-based size_t
    // counter that replaced the 1-based double loop variable.
    const std::string                               *promotedCounter_ = nullptr;
    std::vector<std::string>                         endStack_;  // `end` extents (active index ctx)
    // Non-empty inside an elementwise-array fill loop: the 0-based size_t
    // index, so a bare whole-array `x` emits the element `x[<idx>]`.
    std::string                                       elementCtx_;
    std::string                                       elementRows_, elementCols_;
    std::unordered_map<std::string, std::string>      elementIdxMap_;
    // The function's RETURN ABI (set by emitOneFunction before the body is
    // emitted), so an early `return` lowers to the SAME form as the
    // end-of-function return: a value output (`return name;`, or `.take()` if
    // boxed), or a bare `return;` for a void / out-param / multi-ref function.
    bool                                             returnsValue_  = false;
    std::string                                      returnName_;
    bool                                             returnDynamic_ = false;
    std::vector<std::pair<std::string, std::string>> epilogueAssigns_;
    // Locals hoisted as Dynamic (numkit_val) — arising from markAssignedDynamic
    // constructs (e.g. try/catch bodies). Every assignment to one must box (there
    // is no numkit_val = <typed>), and it stays Dynamic so later reads route through
    // the runtime.
    std::set<std::string>                            dynamicLocals_;
    // Non-null when emitting as part of a multi-function program: routes
    // user-function calls and accumulates the specialisations to emit.
    ProgramEmitCtx                                  *ctx_ = nullptr;
    // Non-null when classes are in play: resolves object field access,
    // value-vs-handle representation, and constructor calls.
    const ClassRegistry                             *classes_ = nullptr;
    // Bridged emission (DESIGN.md §6a): when true, a call the emitter cannot
    // lower but whose result inference proves scalar is emitted as a C-ABI
    // call (numkit_rt::bridge_scalar) instead of throwing.
    bool                                             bridge_ = false;
    // Ops-kernel lowering: when true, a heavy array op with a matching ops
    // kernel (matmul, …) emits a numkit::ops:: call instead of an inline loop.
    bool                                             opsKernels_ = false;
};

} // namespace numkit::codegen::detail
