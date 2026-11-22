// codegen/src/emitter_internal.hpp — shared internal declarations across emitter_*.cpp.
#pragma once

#include <numkit/codegen/emitter.hpp>
#include <numkit/codegen/indexing.hpp>

#include <keel/algo.h>
#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/hash_map.h>
#include <keel/hash_set.h>
#include <keel/ref.h>
#include <keel/smart_ptr.h>
#include <keel/string.h>

#include <charconv>
#include <cstddef>
#include <stdexcept>

namespace numkit::codegen {
keel::String cppArrayElemType(ValueType dtype);
}  // namespace numkit::codegen

namespace numkit::codegen::detail {

constexpr keel::i32 kMaxFixpoint = 16;  // matches inference.cpp; lattice is shallow

// ── String-assembly helpers (keel::String-backed, emitter-internal only) ─────
inline keel::String strCat2(keel::StringRef a, keel::StringRef b)
{
    keel::String out = keel::String::with_capacity(a.size() + b.size());
    out.append(a);
    out.append(b);
    return out;
}

inline keel::String operator+(const keel::String &a, const keel::String &b) { return strCat2(a, b); }
inline keel::String operator+(const keel::String &a, const char *b) { return strCat2(a, b); }
inline keel::String operator+(const char *a, const keel::String &b) { return strCat2(a, b); }
inline keel::String operator+(const keel::String &a, keel::StringRef b) { return strCat2(a, b); }
inline keel::String operator+(keel::StringRef a, const keel::String &b) { return strCat2(a, b); }
inline keel::String operator+(keel::StringRef a, keel::StringRef b) { return strCat2(a, b); }
inline keel::String operator+(keel::StringRef a, const char *b) { return strCat2(a, b); }
inline keel::String operator+(const char *a, keel::StringRef b) { return strCat2(a, b); }
inline keel::String operator+(const keel::String &a, char b) { return strCat2(a, keel::StringRef(&b, 1)); }
inline keel::String operator+(char a, const keel::String &b) { return strCat2(keel::StringRef(&a, 1), b); }
inline keel::String operator+(keel::StringRef a, char b) { return strCat2(a, keel::StringRef(&b, 1)); }
inline keel::String operator+(char a, keel::StringRef b) { return strCat2(keel::StringRef(&a, 1), b); }

inline keel::String &operator+=(keel::String &a, keel::StringRef b)
{
    a.append(b);
    return a;
}

inline keel::String &operator+=(keel::String &a, char b)
{
    a.push_back(b);
    return a;
}

template <typename T>
inline keel::String toStr(T v)
{
    return keel::format("{}", v);
}

inline keel::String repeatChar(keel::usize n, char ch)
{
    keel::String out;
    out.resize(n, ch);
    return out;
}

inline keel::Array<keel::String> sortedKeys(const keel::HashSet<keel::String> &s)
{
    keel::Array<keel::String> out;
    out.reserve(s.size());
    for (const auto &k : s) out.push_back(k.clone());
    keel::sort(out);
    return out;
}

inline keel::String firstSortedKey(const keel::HashSet<keel::String> &s)
{
    keel::String best;
    bool         have = false;
    for (const auto &k : s) {
        if (!have || k < best) {
            best = k.clone();
            have = true;
        }
    }
    return best;
}

struct StrPair {
    keel::String first;
    keel::String second;

    StrPair() noexcept = default;
    StrPair(keel::String &&a, keel::String &&b) noexcept
        : first(keel::move(a)), second(keel::move(b))
    {}
    StrPair(keel::StringRef a, keel::StringRef b)
        : first(keel::String::from(a)), second(keel::String::from(b))
    {}
    StrPair(const StrPair &o) : first(o.first.clone()), second(o.second.clone()) {}
    StrPair &operator=(const StrPair &o)
    {
        if (this != &o) {
            first  = o.first.clone();
            second = o.second.clone();
        }
        return *this;
    }
    StrPair(StrPair &&) noexcept = default;
    StrPair &operator=(StrPair &&) noexcept = default;
    ~StrPair() noexcept = default;

    StrPair clone() const { return *this; }
};

// Per-array metadata for an array parameter or the single output array.
// 1-D vector orientation. The RawBuffer ABI erases it (a vector is just
// ptr+len), but the compile-time type knows it — used to fold size(vec, dim).
enum class VecOrient { Unknown, Row, Col };

struct ArrayInfo {
    ValueType                 dtype          = ValueType::DOUBLE;
    keel::String              lenVar;          // 1-D: the length EXPRESSION — `<name>_len`
                                               // for a param/output, `<name>.size()` for a local
    bool                      isOutput       = false;  // the function's caller-allocated out-param
    bool                      is2D           = false;  // a 2-D matrix (column-major storage)
    VecOrient                 orient         = VecOrient::Unknown;  // 1-D only (!is2D && !isND)
    keel::String              rowsVar, colsVar;  // 2-D: dim companions (`<name>_rows/_cols`)
    bool                      isLocal        = false;  // an owned `std::vector` local (not a buffer ptr)
    keel::String              dataExpr;        // the element-pointer EXPRESSION — `<name>` for a
                                               // param/output buffer, `<name>.data()` for a local
    bool                      isND           = false;  // a rank-N (N>=3) array (column-major flat storage)
    keel::Array<keel::String> ndDims;          // when isND: per-dim size EXPRESSIONS (rank = size())
    bool                      ndRuntimeLocal = false;  // an N-D LOCAL whose dims are runtime
    bool                      isRefParam     = false;  // a `std::vector<T>&` reference out-param (do not hoist)

    ArrayInfo() noexcept = default;
    ArrayInfo(const ArrayInfo &o)
        : dtype(o.dtype), lenVar(o.lenVar.clone()), isOutput(o.isOutput), is2D(o.is2D),
          orient(o.orient), rowsVar(o.rowsVar.clone()), colsVar(o.colsVar.clone()),
          isLocal(o.isLocal), dataExpr(o.dataExpr.clone()), isND(o.isND),
          ndDims(o.ndDims.clone()), ndRuntimeLocal(o.ndRuntimeLocal), isRefParam(o.isRefParam)
    {}
    ArrayInfo &operator=(const ArrayInfo &o)
    {
        if (this != &o) {
            dtype          = o.dtype;
            lenVar         = o.lenVar.clone();
            isOutput       = o.isOutput;
            is2D           = o.is2D;
            orient         = o.orient;
            rowsVar        = o.rowsVar.clone();
            colsVar        = o.colsVar.clone();
            isLocal        = o.isLocal;
            dataExpr       = o.dataExpr.clone();
            isND           = o.isND;
            ndDims         = o.ndDims.clone();
            ndRuntimeLocal = o.ndRuntimeLocal;
            isRefParam     = o.isRefParam;
        }
        return *this;
    }
    ArrayInfo(ArrayInfo &&) noexcept = default;
    ArrayInfo &operator=(ArrayInfo &&) noexcept = default;
    ~ArrayInfo() noexcept = default;

    ArrayInfo clone() const { return *this; }
};

struct OptFacts {
    keel::HashSet<const ASTNode *>               promotedLoops;  // FOR nodes -> 0-based counter
    keel::HashMap<const ASTNode *, keel::String> loopBound;      // FOR node -> C++ size_t bound

    OptFacts() noexcept = default;
    OptFacts(const OptFacts &o)
        : promotedLoops(o.promotedLoops.clone()), loopBound(o.loopBound.clone())
    {}
    OptFacts &operator=(const OptFacts &o)
    {
        if (this != &o) {
            promotedLoops = o.promotedLoops.clone();
            loopBound     = o.loopBound.clone();
        }
        return *this;
    }
    OptFacts(OptFacts &&) noexcept = default;
    OptFacts &operator=(OptFacts &&) noexcept = default;
    ~OptFacts() noexcept = default;

    OptFacts clone() const { return *this; }
};

struct CallSite {
    const ASTNode             *def = nullptr;
    keel::Array<InferredType>  argTypes;
    keel::String               mangled;
    keel::Array<ParamSpec>     extraSeed;

    CallSite() noexcept = default;
    CallSite(const ASTNode *d, keel::Array<InferredType> args, keel::String m,
             keel::Array<ParamSpec> extra = {}) noexcept
        : def(d), argTypes(keel::move(args)), mangled(keel::move(m)), extraSeed(keel::move(extra))
    {}
    CallSite(const CallSite &o)
        : def(o.def), argTypes(o.argTypes.clone()), mangled(o.mangled.clone()),
          extraSeed(o.extraSeed.clone())
    {}
    CallSite &operator=(const CallSite &o)
    {
        if (this != &o) {
            def       = o.def;
            argTypes  = o.argTypes.clone();
            mangled   = o.mangled.clone();
            extraSeed = o.extraSeed.clone();
        }
        return *this;
    }
    CallSite(CallSite &&) noexcept = default;
    CallSite &operator=(CallSite &&) noexcept = default;
    ~CallSite() noexcept = default;

    CallSite clone() const { return *this; }
};

struct ProgramEmitCtx {
    const FunctionTable          *funcs = nullptr;
    keel::Array<CallSite>         pending;
    keel::HashSet<keel::String>   seen;
};

struct OneFn {
    keel::String signature;
    keel::String definition;  // signature + " { ... }"

    OneFn() noexcept = default;
    OneFn(keel::String sig, keel::String def) noexcept
        : signature(keel::move(sig)), definition(keel::move(def))
    {}
    OneFn(const OneFn &o) : signature(o.signature.clone()), definition(o.definition.clone()) {}
    OneFn &operator=(const OneFn &o)
    {
        if (this != &o) {
            signature  = o.signature.clone();
            definition = o.definition.clone();
        }
        return *this;
    }
    OneFn(OneFn &&) noexcept = default;
    OneFn &operator=(OneFn &&) noexcept = default;
    ~OneFn() noexcept = default;
};

bool         isCppKeyword(keel::StringRef name);
keel::String cppIdent(keel::StringRef name);
keel::String cppGlobalIdent(keel::StringRef name);

keel::String zeroLiteral(ValueType dtype);
const char  *cppScalarBinOp(keel::StringRef op);
[[noreturn]] void unsupported(keel::StringRef what);
keel::String emitBinOpJoin(keel::StringRef op, keel::StringRef l, keel::StringRef r);
keel::String emitUnOpJoin(keel::StringRef op, keel::StringRef a);

bool         is2DMatrixType(const InferredType &t);
bool         isUnboxableScalarType(const InferredType &t);
VecOrient    orientOf(const InferredType &t);
keel::String structFieldLocal(const ASTNode &fa);
bool         extractStructFieldChain(const ASTNode &fa, keel::String &rootOut,
                                     keel::Array<keel::String> &chainOut);
void         setNestedStructField(StructLayout &layout, const keel::Array<keel::String> &chain,
                                  keel::usize idx, const InferredType &leafType);
void         syncIntermediateStructLocals(keel::StringRef prefix, const StructLayout &layout,
                                          TypeEnv &env);
bool         isBufferArrayType(const InferredType &t);
bool         isByValueReturnArrayType(const InferredType &t);
keel::String cppObjectType(keel::i32 classId, const ClassRegistry *classes);
OptFacts     analyzeOptimizations(const ASTNode &body,
                                  const keel::HashMap<keel::String, ArrayInfo> &arrays);

keel::String        escapeBase(keel::StringRef base);
keel::String        typeCode(const InferredType &t);
keel::String        cEscape(keel::StringRef s, bool &ok);
keel::String        mangle(keel::StringRef base, keel::ArrayRef<InferredType> args);
keel::String        companion(keel::StringRef base, keel::StringRef suffix);
keel::Array<ArgInfo> toArgInfos(keel::ArrayRef<InferredType> types);

const char *unaryMathStd(keel::StringRef name);
const char *opsTranscendentalFn(keel::StringRef name);
const char *binaryMathStd(keel::StringRef name);

extern const char *kPrelude;
keel::String bridgePrelude(keel::StringRef runtimeHeader);
OneFn        emitOneFunction(const ASTNode &funcDef, keel::ArrayRef<ParamSpec> params,
                             const TransferRegistry &reg, ProgramEmitCtx *ctx,
                             keel::StringRef cppName, const ClassRegistry *classes,
                             keel::ArrayRef<ParamSpec> extraSeed = {}, bool bridge = false,
                             bool opsKernels = false, bool interprocByValueReturn = false);

class Emitter {
public:
    Emitter(TypeEnv types, const TransferRegistry &reg,
            keel::HashMap<keel::String, ArrayInfo> arrays, OptFacts opt,
            ProgramEmitCtx *ctx = nullptr, const ClassRegistry *classes = nullptr,
            bool bridge = false, bool opsKernels = false)
        : types_(keel::move(types)), reg_(reg), arrays_(keel::move(arrays)),
          opt_(keel::move(opt)), ctx_(ctx), classes_(classes), bridge_(bridge),
          opsKernels_(opsKernels)
    {}

    // Tell the emitter how an early `return` statement should lower (it matches
    // the end-of-function return chosen in emitOneFunction).
    void setReturnInfo(bool returnsValue, keel::StringRef name, bool dynamic)
    {
        returnsValue_  = returnsValue;
        returnName_    = keel::String::from(name);
        returnDynamic_ = dynamic;
    }

    void setEpilogueAssigns(keel::Array<StrPair> assigns)
    {
        epilogueAssigns_ = keel::move(assigns);
    }

    void emitEpilogueAssigns()
    {
        for (const auto &p : epilogueAssigns_)
            line(p.first + " = " + p.second + ";");
    }

    void emitRawLine(keel::StringRef s) { line(s); }

    // Record a local that was hoisted as Dynamic (numkit_val); see dynamicLocals_.
    void addDynamicLocal(keel::StringRef n) { dynamicLocals_.insert(keel::String::from(n)); }

    // Hoist a local declaration at function entry (scalar or object).
    void hoistLocal(keel::StringRef name, const InferredType &t)
    {
        const keel::String cname = cppIdent(name);
        if (t.isObject()) {
            const ClassInfo *ci = classes_ ? classes_->byId(t.classId) : nullptr;
            if (!ci) unsupported("object local '" + name + "' with no class registry");
            const keel::String clsName = cppGlobalIdent(ci->name);
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
        keel::Array<keel::String> ordered;
        for (auto [name, ai] : arrays_)
            if (ai.isLocal) ordered.push_back(name.clone());
        keel::sort(ordered);
        for (const auto &name : ordered) {
            const ArrayInfo &ai = arrayAt(name);
            if (!ai.isRefParam)
                line("std::vector<" + cppArrayElemType(ai.dtype) + "> " + cppIdent(name) + ";");
        }
        // Runtime-dim N-D locals also need their size_t dim companions declared
        // (set at the zeros/ones assignment; read by indexN / size / numel).
        for (const auto &name : ordered) {
            const ArrayInfo &ai = arrayAt(name);
            if (!ai.ndRuntimeLocal) continue;
            keel::String decl = keel::String::from("std::size_t ");
            for (keel::usize d = 0; d < ai.ndDims.size(); ++d)
                decl += (d ? ", " : "") + ai.ndDims[d] + " = 0";
            line(decl + ";");
        }
    }

    void emitStmt(const ASTNode &s);
    void emitReturnScalar(keel::StringRef name) { line("return " + cppIdent(name) + ";"); }
    // Dynamic tier (DESIGN.md §10 C1): a boxed (numkit_val) return transfers handle
    // ownership OUT to the caller (val::take nulls the local so its dtor is a no-op).
    void emitReturnDynamic(keel::StringRef name) { line("return " + cppIdent(name) + ".take();"); }

    const keel::String &out() const { return out_; }

private:
    void line(keel::StringRef s)
    {
        for (keel::i32 i = 0; i < indent_ * 4; ++i) out_.push_back(' ');
        out_.append(s);
        out_.push_back('\n');
    }
    void open(keel::StringRef head) { line(head + " {"); ++indent_; }
    void close() { --indent_; line("}"); }

    keel::String emitExpr(const ASTNode &e);
    // Dynamic tier (DESIGN.md §10 C1): emit `e` as an `numkit_rt::val` C++
    // expression (a boxed runtime value), for an operand/result the inference
    // could not type. Operators dispatch to the runtime (binop/unop); an
    // un-typeable builtin result stays boxed (call_dyn). Only reachable under
    // bridge_ (it needs the C-ABI). The sound fallback to the typed emitExpr.
    keel::String emitDynamicExpr(const ASTNode &e);
    // A control-flow condition (if / while): a typed condition is the bare C++
    // boolean; a Dynamic condition is evaluated in the Value tier and reduced to
    // the reference engine's truthiness (non-empty AND every element non-zero). DESIGN.md §10 C1.
    keel::String emitCondition(const ASTNode &c);
    keel::String emitBuiltinCall(keel::StringRef name, const ASTNode &call);
    keel::String emitUserCall(keel::StringRef name, const ASTNode &call);
    keel::String emitMethodCall(const ASTNode &call);
    keel::String emitConstruct(keel::StringRef name, const ASTNode &call);
    void         appendCallArg(const ASTNode &arg, keel::Array<InferredType> &argTypes,
                               keel::String &argList);
    void         appendStructCallArgs(keel::StringRef prefix, const StructLayout &layout,
                                      keel::String &argList);
    void         appendStructOutArgs(keel::StringRef prefix, const StructLayout &layout,
                                     keel::String &argList);
    void         collectStructOutRefArrays(keel::StringRef prefix, const StructLayout &layout,
                                           keel::HashSet<keel::String> &out) const;
    void         prepareCallInputAliases(const ASTNode &call, keel::usize firstArgIdx,
                                         const keel::HashSet<keel::String> &outRefArrays,
                                         keel::Array<keel::String> &aliasDecls);
    void         appendIgnoreOutArg(keel::StringRef tmpName, const InferredType &ot,
                                    keel::Array<keel::String> &ignoreDecls, keel::String &argList);
    void         emitStructCopy(keel::StringRef dstPrefix, keel::StringRef srcPrefix,
                                const StructLayout &layout);
    keel::String emitMultiCallExpr(keel::StringRef mangled, keel::StringRef argList,
                                   keel::ArrayRef<InferredType> outs);
    keel::String emitIndexRead(keel::StringRef base, const ASTNode &call);
    // True if `e` is a pure ELEMENTWISE expression — literals, scalar vars,
    // whole arrays, and only elementwise ops (+ - .* ./ .\ .^, unary +/-).
    // Collects the distinct whole-array variable names into `arrays`. (Matrix
    // ops *,/,^ and calls/indexing make it false.)
    bool         collectElementwise(const ASTNode &e, keel::HashSet<keel::String> &arrays) const;
    // The size_t expression for axis k (0-based) of array `a`: N-D companion /
    // 2-D rows,cols / 1-D orientation (row = 1 x len, col = len x 1). Axes
    // beyond the rank, or an orientation-unknown vector, yield "1" -- callers
    // that must distinguish unknown orientation guard on a.orient first.
    keel::String dimExpr(const ArrayInfo &a, keel::usize k) const;
    void         emitAssign(const ASTNode &s);
    void         emitAssignIdentifierTail(const ASTNode &lhs, const ASTNode &rhs,
                                          keel::StringRef name,
                                          keel::StringRef structWriteFld);
    void         emitMultiAssign(const ASTNode &s);
    void         emitIndexWrite(const ASTNode &lhsCall, const ASTNode &rhs);
    void         emitFor(const ASTNode &s);
    void         emitWhile(const ASTNode &s);
    void         emitIf(const ASTNode &s);
    void         emitSwitch(const ASTNode &s);
    void         emitTry(const ASTNode &s);

    bool isArrayVar(keel::StringRef n) const { return arrays_.contains(n); }

    ArrayInfo &arrayAt(keel::StringRef n)
    {
        ArrayInfo *p = arrays_.get(n);
        if (!p) unsupported("internal: missing array metadata for '" + n + "'");
        return *p;
    }

    const ArrayInfo &arrayAt(keel::StringRef n) const
    {
        const ArrayInfo *p = arrays_.get(n);
        if (!p) unsupported("internal: missing array metadata for '" + n + "'");
        return *p;
    }

    keel::String resolveArrayVar(const ASTNode &n) const
    {
        if (n.type == NodeType::IDENTIFIER && isArrayVar(n.strValue))
            return n.strValue.clone();
        if ((n.type == NodeType::FIELD_ACCESS || n.type == NodeType::DYNAMIC_FIELD_ACCESS)
            && !n.children.empty() && n.children[0]) {
            const AbstractValue base = inferExpr(*n.children[0], types_, reg_, classes_);
            if (!base.type.isObject()) {
                keel::String fld = structFieldLocal(n);
                if (!fld.empty() && isArrayVar(fld)) return fld;
            }
        }
        return {};
    }

    void recordStructFieldWrite(const ASTNode &lhs, keel::StringRef fld,
                                const InferredType &ft)
    {
        types_.set(fld, {ft, ConstVal::unknown()});
        keel::String              rootVar;
        keel::Array<keel::String> chain;
        if (!extractStructFieldChain(lhs, rootVar, chain)) return;
        const AbstractValue cur =
            types_.has(rootVar) ? types_.get(rootVar) : AbstractValue::dynamic();
        auto layout = keel::make_arc<StructLayout>();
        if (cur.type.isStruct() && cur.type.structLayout)
            layout->fields = cur.type.structLayout->fields.clone();
        setNestedStructField(*layout, chain, 0, ft);
        syncIntermediateStructLocals("_numkit_fld_" + rootVar, *layout, types_);
        types_.set(rootVar, {InferredType::structOf(keel::move(layout)), ConstVal::unknown()});
    }

    // The AbstractValue for an array variable (non-scalar -> buffer).
    AbstractValue arrayValue(keel::StringRef n) const
    {
        return {InferredType::concrete(arrayAt(n).dtype, Shape::rowVector()),
                ConstVal::unknown()};
    }

    keel::String                                out_;
    keel::i32                                         indent_           = 1;
    keel::i32                                         switchCounter_    = 0;  // unique switch-selector temp ids
    keel::i32                                         multiCallCounter_ = 0;  // unique multi-output expr temp ids
    keel::i32                                         aliasCounter_     = 0;  // unique call input-alias temp ids
    keel::HashMap<keel::String, StrPair>        callAliasInputs_;
    TypeEnv                                     types_;
    const TransferRegistry                     &reg_;
    keel::HashMap<keel::String, ArrayInfo>      arrays_;
    OptFacts                                    opt_;
    // Non-null inside a promoted clean-index loop: the 0-based size_t
    // counter that replaced the 1-based double loop variable.
    const keel::String                         *promotedCounter_ = nullptr;
    keel::Array<keel::String>                   endStack_;  // `end` extents (active index ctx)
    // Non-empty inside an elementwise-array fill loop: the 0-based size_t
    // index, so a bare whole-array `x` emits the element `x[<idx>]`.
    keel::String                                elementCtx_;
    keel::String                                elementRows_, elementCols_;
    keel::HashMap<keel::String, keel::String>   elementIdxMap_;
    // The function's RETURN ABI (set by emitOneFunction before the body is
    // emitted), so an early `return` lowers to the SAME form as the
    // end-of-function return: a value output (`return name;`, or `.take()` if
    // boxed), or a bare `return;` for a void / out-param / multi-ref function.
    bool                                        returnsValue_  = false;
    keel::String                                returnName_;
    bool                                        returnDynamic_ = false;
    keel::Array<StrPair>                        epilogueAssigns_;
    // Locals hoisted as Dynamic (numkit_val) — arising from markAssignedDynamic
    // constructs (e.g. try/catch bodies). Every assignment to one must box (there
    // is no numkit_val = <typed>), and it stays Dynamic so later reads route through
    // the runtime.
    keel::HashSet<keel::String>                 dynamicLocals_;
    // Non-null when emitting as part of a multi-function program: routes
    // user-function calls and accumulates the specialisations to emit.
    ProgramEmitCtx                             *ctx_ = nullptr;
    // Non-null when classes are in play: resolves object field access,
    // value-vs-handle representation, and constructor calls.
    const ClassRegistry                        *classes_ = nullptr;
    // Bridged emission (DESIGN.md §6a): when true, a call the emitter cannot
    // lower but whose result inference proves scalar is emitted as a C-ABI
    // call (numkit_rt::bridge_scalar) instead of throwing.
    bool                                        bridge_ = false;
    // Ops-kernel lowering: when true, a heavy array op with a matching ops
    // kernel (matmul, …) emits a numkit::ops:: call instead of an inline loop.
    bool                                        opsKernels_ = false;
};

} // namespace numkit::codegen::detail
