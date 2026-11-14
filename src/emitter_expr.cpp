// codegen/src/emitter_expr.cpp — expression, dynamic, builtin/user/method calls, elementwise.

#include "emitter_internal.hpp"

namespace numkit::codegen::detail {

// MATLAB unary-math name -> std:: name. Restricted to functions that BOTH
// have a registered transfer (so the result types to a concrete scalar and
// this lowering is actually reachable) AND have an exact std equivalent.
// Builtins outside this set (sqrt/log/asin/… not yet typed; sign no clean
// std form) hit the explicit boundary in emitBuiltinCall and throw — never
// silently wrong, never dead-but-unreachable speculation.
const char *unaryMathStd(const std::string &name)
{
    static const std::unordered_map<std::string, const char *> kMap = {
        {"sin", "sin"},     {"cos", "cos"},   {"tan", "tan"},   {"atan", "atan"},
        {"sinh", "sinh"},   {"cosh", "cosh"}, {"tanh", "tanh"}, {"exp", "exp"},
        {"floor", "floor"}, {"ceil", "ceil"}, {"round", "round"},
        {"abs", "abs"},     {"fix", "trunc"},
        // total-on-ℝ, real-only (transfer refuses complex) — see
        // realOnlyMathUnaryTransfer
        {"asinh", "asinh"}, {"erf", "erf"}, {"erfc", "erfc"}, {"expm1", "expm1"},
        {"gammaln", "lgamma"}};  // nk gammaln(x) == std::lgamma(x) (bit-identical)
    const auto it = kMap.find(name);
    return it == kMap.end() ? nullptr : it->second;
}

// MATLAB unary-math name -> the numkit::ops:: SIMD transcendental facade entry
// (kernels.hpp), for the opt-in ops-kernel tier. Restricted to the real-total
// transcendentals fusedTransAffine handles (no complex-domain decline) that the
// codegen also lowers inline — so routing here is a faithful, faster swap
// (inline std::<fn> does not auto-vectorise on MSVC). floor/ceil/round/abs/fix/
// erf/erfc are NOT here (cheap or no SIMD kernel) -> they stay inline.
const char *opsTranscendentalFn(const std::string &name)
{
    static const std::unordered_map<std::string, const char *> kMap = {
        {"sin", "sinDouble"},     {"cos", "cosDouble"},   {"tan", "tanDouble"},
        {"atan", "atanDouble"},   {"sinh", "sinhDouble"}, {"cosh", "coshDouble"},
        {"tanh", "tanhDouble"},   {"exp", "expDouble"},   {"asinh", "asinhDouble"},
        {"expm1", "expm1Double"}};
    const auto it = kMap.find(name);
    return it == kMap.end() ? nullptr : it->second;
}

// MATLAB binary-math name -> std:: name. Total on ℝ², typed scalar->scalar by
// realBinaryMathTransfer (complex/array -> Dynamic, so this only fires for a
// real scalar result).
const char *binaryMathStd(const std::string &name)
{
    static const std::unordered_map<std::string, const char *> kMap = {
        {"atan2", "atan2"}, {"hypot", "hypot"},
        {"rem", "fmod"},   // nk rem(a,b) == std::fmod(a,b) (bit-identical)
        // MATLAB's 2-arg max/min ignore NaN (the result is the non-NaN operand), which is
        // bit-identical to std::fmax/std::fmin. The 1-arg reduction max(x)/min(x) is a
        // different arity (handled at the statement level), so it never reaches this 2-arg
        // path. maxMinTransfer types the 2-arg form (real DOUBLE) for the elementwise fill.
        {"max", "fmax"}, {"min", "fmin"}};
    const auto it = kMap.find(name);
    return it == kMap.end() ? nullptr : it->second;
}

std::string Emitter::emitExpr(const ASTNode &e)
{
    switch (e.type) {
    case NodeType::NUMBER_LITERAL:
        return formatDoubleLiteral(e.numValue);
    case NodeType::IMAG_LITERAL:
        return "std::complex<double>(0.0, " + formatDoubleLiteral(e.numValue) + ")";
    case NodeType::BOOL_LITERAL:
        return e.boolValue ? "true" : "false";
    case NodeType::STRING_LITERAL:
        // A 1x1 char literal ('a') is a uint16 code-unit scalar. A multi-char
        // literal is a char ARRAY -> handled at the statement level (emitAssign);
        // it cannot be a C++ scalar expression, so refuse it here.
        if (e.strValue.size() != 1)
            unsupported("multi-char string literal in expression position (char array)");
        return "std::uint16_t("
               + std::to_string(static_cast<unsigned>(static_cast<unsigned char>(e.strValue[0])))
               + ")";
    case NodeType::IDENTIFIER:
        if (isArrayVar(e.strValue)) {
            // Inside an elementwise-array fill loop a whole array means its
            // current element; in any other (scalar) context it is an error.
            if (auto it = elementIdxMap_.find(e.strValue); it != elementIdxMap_.end())
                return arrays_.at(e.strValue).dataExpr + "[" + it->second + "]";
            if (!elementCtx_.empty())
                return arrays_.at(e.strValue).dataExpr + "[" + elementCtx_ + "]";
            unsupported("bare array identifier '" + e.strValue + "' in scalar context");
        }
        return cppIdent(e.strValue);
    case NodeType::END_VAL:
        // `end` inside an index -> the current indexed array's extent (pushed by
        // emitIndexRead as a 1-D length). MATLAB `end` is a scalar; emit it as a
        // double so it composes in index arithmetic (end-1, end/2, ...). Outside an
        // index context the stack is empty -> refuse (sound; never a wrong value).
        if (endStack_.empty())
            unsupported("'end' outside an indexing context");
        return "static_cast<double>(" + endStack_.back() + ")";
    case NodeType::BINARY_OP:
        if (e.children.size() != 2) unsupported("binary op arity");
        return emitBinOpJoin(e.strValue, emitExpr(*e.children[0]),
                             emitExpr(*e.children[1]));
    case NodeType::UNARY_OP:
        if (e.children.size() != 1) unsupported("unary op arity");
        if (e.strValue == "'" || e.strValue == ".'") {
            // Transpose in expression position is only the SCALAR case here
            // (identity, except ctranspose ' conjugates a complex scalar). A
            // whole-vector/matrix transpose is a statement-level producer
            // (emitAssign); transposing a non-scalar sub-expression is refused.
            const AbstractValue a = inferExpr(*e.children[0], types_, reg_, classes_);
            if (!a.type.shape.isScalar())
                unsupported("transpose of a non-scalar sub-expression");
            const std::string inner = emitExpr(*e.children[0]);
            return (e.strValue == "'" && a.type.dtype == ValueType::COMPLEX)
                       ? "std::conj(" + inner + ")"
                       : inner;
        }
        return emitUnOpJoin(e.strValue, emitExpr(*e.children[0]));
    case NodeType::CALL: {
        if (e.children.empty()) unsupported("empty call");
        const ASTNode &callee = *e.children[0];
        if (callee.type == NodeType::FIELD_ACCESS
            || callee.type == NodeType::DYNAMIC_FIELD_ACCESS) {
            const std::string fld = resolveArrayVar(callee);
            if (!fld.empty()) return emitIndexRead(fld, e);
            if (callee.type == NodeType::FIELD_ACCESS && classes_)
                return emitMethodCall(e);  // obj.method(args)
        }
        if (callee.type != NodeType::IDENTIFIER)
            unsupported("non-identifier callee");
        if (isArrayVar(callee.strValue))
            return emitIndexRead(callee.strValue, e);
        // A class name (not a variable in scope) -> construction.
        if (classes_ && !types_.has(callee.strValue) && classes_->has(callee.strValue))
            return emitConstruct(callee.strValue, e);
        // A user function (not a variable in scope) takes priority over a
        // same-named builtin (the reference engine's path shadowing) when emitting a program.
        if (ctx_ && ctx_->funcs && !types_.has(callee.strValue)
            && ctx_->funcs->has(callee.strValue))
            return emitUserCall(callee.strValue, e);
        return emitBuiltinCall(callee.strValue, e);
    }
    case NodeType::DYNAMIC_FIELD_ACCESS:  // s.('a') : a CONSTANT name folds via structFieldLocal
    case NodeType::FIELD_ACCESS: {  // obj.field read -> obj.field / obj->field
        if (e.children.empty()) unsupported("field access arity");
        // s(i).a : struct-ARRAY field read (SoA, G2.5) -> index the field-array _numkit_fld_s_a at
        // i (s is not a value; only its field-arrays exist). Matched before inferring the base.
        if (e.children[0]->type == NodeType::CALL && !e.children[0]->children.empty()
            && e.children[0]->children[0]->type == NodeType::IDENTIFIER) {
            const std::string sv  = e.children[0]->children[0]->strValue;
            const std::string fld = "_numkit_fld_" + sv + "_" + e.strValue;
            if (isArrayVar(fld) && !types_.has(sv))
                return emitIndexRead(fld, *e.children[0]);  // _numkit_fld_s_a indexed by s(i)'s subscript
        }
        const AbstractValue base = inferExpr(*e.children[0], types_, reg_, classes_);
        // Plain struct: the synthesized field-local (field-flattening), generalised
        // to a NESTED chain s.a.b via the chain helper. The immediate base being
        // non-object gates struct-vs-object at every level.
        if (!base.type.isObject()) {
            const std::string fld = structFieldLocal(e);
            if (!fld.empty()) {
                if (isArrayVar(fld)) {
                    if (auto it = elementIdxMap_.find(fld); it != elementIdxMap_.end())
                        return arrays_.at(fld).dataExpr + "[" + it->second + "]";
                    if (!elementCtx_.empty())
                        return arrays_.at(fld).dataExpr + "[" + elementCtx_ + "]";
                }
                return fld;
            }
        }
        if (!base.type.isObject() || !classes_)
            unsupported("field access on a non-object value");
        const ClassInfo *ci = classes_->byId(base.type.classId);
        if (!ci || !ci->field(e.strValue))
            unsupported("unknown field '" + e.strValue + "'");
        return emitExpr(*e.children[0]) + (ci->isHandle ? "->" : ".") + cppIdent(e.strValue);
    }
    default:
        unsupported("expression node kind");
    }
}

// Collect identifiers referenced in `n` (value / callee position) into `out`, for
// the anonymous-function capture check (G5b). A FIELD_ACCESS contributes only its
// base (the field NAME, in strValue, is not a variable). A nested ANON_FUNC sets
// `nestedAnon` (refused — its own param scope is not modeled here). Everything else
// recurses into all children (so a CALL callee and a DYNAMIC_FIELD_ACCESS name expr
// are both collected, which is what the capture check wants).
static void collectAnonRefIds(const ASTNode &n, std::set<std::string> &out, bool &nestedAnon)
{
    switch (n.type) {
    case NodeType::ANON_FUNC:
        nestedAnon = true;
        return;
    case NodeType::IDENTIFIER:
        out.insert(n.strValue);
        return;
    case NodeType::FIELD_ACCESS:  // s.field — the field name is not a variable
        if (!n.children.empty() && n.children[0])
            collectAnonRefIds(*n.children[0], out, nestedAnon);
        return;
    default:
        for (const auto &c : n.children)
            if (c) collectAnonRefIds(*c, out, nestedAnon);
        return;
    }
}

// Dynamic tier (DESIGN.md §10 C1): emit `e` as an numkit_rt::val expression. An
// operand the inference DID type is boxed at the boundary (val::scalar); a
// Dynamic operand is already a val (binds by const-ref into binop/unop); an
// un-typeable builtin result stays boxed (call_dyn). v1 covers real-double
// scalars + Dynamic values — a complex/array operand, a Dynamic call argument,
// or short-circuit && / || is an explicit boundary (A3/A4/A5), never wrong code.
std::string Emitter::emitDynamicExpr(const ASTNode &e)
{
    switch (e.type) {
    case NodeType::NUMBER_LITERAL:
        return "numkit_rt::val::scalar(" + formatDoubleLiteral(e.numValue) + ")";
    case NodeType::IDENTIFIER: {
        if (isArrayVar(e.strValue)) {
            // Box a typed array at the boundary (shape-preserving, column-major).
            const ArrayInfo &ai = arrays_.at(e.strValue);
            if (ai.dtype == ValueType::DOUBLE) {
                if (ai.is2D)
                    return "numkit_rt::val::matrix(" + ai.dataExpr + ", " + ai.rowsVar + ", "
                           + ai.colsVar + ")";
                if (ai.isND) {
                    std::string dims;
                    for (std::size_t k = 0; k < ai.ndDims.size(); ++k)
                        dims += (k ? ", " : "") + ai.ndDims[k];
                    return "numkit_rt::val::array_nd(" + ai.dataExpr + ", {" + dims + "})";
                }
                return "numkit_rt::val::array(" + ai.dataExpr + ", " + ai.lenVar + ")";  // 1-D
            }
            if (ai.dtype == ValueType::COMPLEX) {
                if (ai.is2D)
                    return "numkit_rt::val::complex_matrix(" + ai.dataExpr + ", " + ai.rowsVar + ", "
                           + ai.colsVar + ")";
                if (ai.isND) {
                    std::string dims;
                    for (std::size_t k = 0; k < ai.ndDims.size(); ++k)
                        dims += (k ? ", " : "") + ai.ndDims[k];
                    return "numkit_rt::val::complex_array_nd(" + ai.dataExpr + ", {" + dims + "})";
                }
                return "numkit_rt::val::complex_array(" + ai.dataExpr + ", " + ai.lenVar + ")";  // 1-D complex
            }
            unsupported("Dynamic tier: array operand '" + e.strValue + "' unsupported dtype (v1)");
        }
        if (dynamicLocals_.count(e.strValue))
            return cppIdent(e.strValue);
        const AbstractValue av = inferExpr(e, types_, reg_, classes_);
        if (av.type.isDynamic())
            return cppIdent(e.strValue);  // an existing Dynamic local (already an numkit_rt::val)
        if (av.type.dtype == ValueType::DOUBLE && av.type.shape.isScalar())
            return "numkit_rt::val::scalar(" + cppIdent(e.strValue) + ")";  // box a typed double scalar
        unsupported("Dynamic tier: non-double-scalar operand '" + e.strValue + "' (v1)");
    }
    case NodeType::BINARY_OP:
        if (e.children.size() != 2) unsupported("Dynamic tier: binary op arity");
        if (e.strValue == "&&" || e.strValue == "||")
            unsupported("Dynamic tier: short-circuit && / || operand (v1, A5)");
        return "numkit_rt::binop(\"" + e.strValue + "\", " + emitDynamicExpr(*e.children[0]) + ", "
               + emitDynamicExpr(*e.children[1]) + ")";
    case NodeType::UNARY_OP:
        if (e.children.size() != 1) unsupported("Dynamic tier: unary op arity");
        if (e.strValue == "'" || e.strValue == ".'")
            unsupported("Dynamic tier: transpose operand (v1)");
        return "numkit_rt::unop(\"" + e.strValue + "\", " + emitDynamicExpr(*e.children[0]) + ")";
    case NodeType::CALL: {
        // An un-typeable builtin result, kept boxed. A plain builtin name; each
        // argument is a real-double scalar, a Dynamic value (A3), or a 1-D double
        // array (A4) — emitDynamicExpr is the gatekeeper that boxes/refuses each.
        // Index reads and method calls are not this path.
        if (e.children.empty()) unsupported("Dynamic tier: empty call");
        const ASTNode &callee = *e.children[0];
        if (callee.type != NodeType::IDENTIFIER)
            unsupported("Dynamic tier: non-identifier callee");
        // Indexing a Dynamic value: z(i) where z is a Dynamic VARIABLE in scope
        // (not a builtin, not a typed array). `z(i)` is index/call-ambiguous —
        // the runtime (numkit_index) resolves it. v1: a single subscript.
        if ((types_.has(callee.strValue)
             && inferExpr(callee, types_, reg_, classes_).type.isDynamic())
            || dynamicLocals_.count(callee.strValue)) {
            if (e.children.size() < 2)
                unsupported("Dynamic tier: index with no subscript");
            std::string subs;
            for (std::size_t k = 1; k < e.children.size(); ++k)
                subs += (k > 1 ? ", " : "") + emitDynamicExpr(*e.children[k]);
            return "numkit_rt::index_dyn(" + cppIdent(callee.strValue) + ", {" + subs + "})";
        }
        if (isArrayVar(callee.strValue))
            unsupported("Dynamic tier: index read '" + callee.strValue + "' (v1, A4)");
        // A boxed-result call to a CO-COMPILED user function (in this program's
        // function table) cannot use the call_dyn-by-NAME ABI: that resolves
        // against the runtime registry, which holds builtins / engine functions —
        // NOT the compiled specialisations emitted alongside this code. Emitting
        // call_dyn here would look up a name that is not the compiled spec, i.e.
        // miscompile in a standalone artifact. Refuse instead (refuse-not-
        // miscompile). This is the path a recursive call takes (the monomorphiser
        // breaks recursion to Dynamic), so recursion cleanly refuses under the
        // bridge rather than miscompiling. (An EXTERNAL callee — not in the table —
        // keeps call_dyn below: it genuinely resolves in the runtime/engine.) A
        // future enhancement can call the compiled spec directly and box its result.
        if (ctx_ && ctx_->funcs && !types_.has(callee.strValue)
            && ctx_->funcs->has(callee.strValue))
            unsupported("Dynamic tier: boxed result of a co-compiled user call '"
                        + callee.strValue + "' (e.g. recursion) — the compiled "
                        "specialisation is not resolvable by name in the runtime registry (v1)");
        bool allDoubleScalar = true;
        for (std::size_t i = 1; i < e.children.size(); ++i) {
            if (e.children[i]->type == NodeType::IDENTIFIER
                && dynamicLocals_.count(e.children[i]->strValue)) {
                allDoubleScalar = false;
                continue;
            }
            const AbstractValue av = inferExpr(*e.children[i], types_, reg_, classes_);
            if (!(av.type.isConcrete() && av.type.dtype == ValueType::DOUBLE
                  && av.type.shape.isScalar()))
                allDoubleScalar = false;  // Dynamic / array arg -> call_dynv (emitDynamicExpr boxes)
        }
        std::string args;
        for (std::size_t i = 1; i < e.children.size(); ++i)
            args += (i > 1 ? ", " : "")
                    + (allDoubleScalar ? emitExpr(*e.children[i]) : emitDynamicExpr(*e.children[i]));
        // All real-double scalars -> call_dyn (boxes doubles directly, fewer
        // allocations); any Dynamic arg -> call_dynv (each arg is a boxed val).
        return allDoubleScalar
                   ? ("numkit_rt::call_dyn(\"" + callee.strValue + "\", {" + args + "})")
                   : ("numkit_rt::call_dynv(\"" + callee.strValue + "\", {" + args + "})");
    }
    case NodeType::STRING_LITERAL:
    case NodeType::DQSTRING_LITERAL: {
        // a char/string literal as a Dynamic value -> box via numkit_box_string (printable
        // ASCII only; cEscape refuses a non-printable byte, sound).
        bool              ok  = false;
        const std::string esc = cEscape(e.strValue, ok);
        if (!ok) unsupported("Dynamic tier: char-literal with a non-printable byte (v1)");
        return "numkit_rt::val::string(\"" + esc + "\")";
    }
    case NodeType::CELL_LITERAL: {
        // a brace literal {...} -> a Dynamic cell. Each element is boxed by
        // emitDynamicExpr (scalar / Dynamic / array / string), so the whole tier
        // is reused with no per-element type tracking. The structure MIRRORS the
        // interpreter's execCellLiteral: empty {} -> 0x0; children[0] not a BLOCK
        // -> a 1xN row of direct elements; else each child is a row BLOCK (a 2-D
        // {a,b; c,d} builds an RxC cell, column-major; a ragged literal -- a MATLAB
        // error -- refuses to interpreted fallback).
        if (e.children.empty()) return "numkit_rt::val::cell({})";  // {}
        auto boxRow = [&](const std::vector<ASTNodePtr> &els, std::string &out) {
            for (const auto &el : els) {
                if (!el) unsupported("Dynamic tier: null cell element (v1)");
                out += (out.empty() ? "" : ", ") + emitDynamicExpr(*el);
            }
        };
        if (e.children[0]->type != NodeType::BLOCK) {  // children are elements directly (1 row)
            std::string elems;
            boxRow(e.children, elems);
            return "numkit_rt::val::cell({" + elems + "})";
        }
        // row-block structure: validate rectangular, then emit ROW-MAJOR.
        const std::size_t R = e.children.size();
        const std::size_t C = e.children[0]->children.size();
        for (const auto &row : e.children)
            if (!row || row->type != NodeType::BLOCK || row->children.size() != C)
                unsupported("Dynamic tier: ragged / malformed cell literal (v1)");
        std::string elems;
        for (const auto &row : e.children) boxRow(row->children, elems);
        if (R == 1) return "numkit_rt::val::cell({" + elems + "})";  // 1 x C
        return "numkit_rt::val::cell_2d({" + elems + "}, " + std::to_string(R) + ", "
               + std::to_string(C) + ")";
    }
    case NodeType::CELL_INDEX: {
        // c{i} or c{i,j} content extraction. v1: one or two unboxed-scalar
        // subscripts (a CSL c{:} / c{vec} is multi-value -> refused -> interpreted
        // fallback). The base is boxed (a Dynamic cell local).
        const std::size_t nsub = e.children.empty() ? 0 : e.children.size() - 1;
        if (nsub != 1 && nsub != 2)
            unsupported("Dynamic tier: cell content index supports 1 or 2 subscripts (v1)");
        for (std::size_t k = 1; k <= nsub; ++k)
            if (!isUnboxableScalarType(inferExpr(*e.children[k], types_, reg_, classes_).type))
                unsupported("Dynamic tier: cell content index must be unboxed scalar(s) (v1)");
        const std::string base = emitDynamicExpr(*e.children[0]);
        if (nsub == 1)
            return "numkit_rt::cell_get(" + base + ", " + emitExpr(*e.children[1]) + ")";
        return "numkit_rt::cell_get_2d(" + base + ", " + emitExpr(*e.children[1]) + ", "
               + emitExpr(*e.children[2]) + ")";
    }
    case NodeType::ANON_FUNC: {
        // @funcName -> a named function handle (G5a): NO body child. The name
        // resolves in the runtime registry; a co-compiled user spec is NOT there.
        if (e.children.empty()) {
            if (e.strValue.empty())
                unsupported("Dynamic tier: anonymous function handle with no name");
            if (ctx_ && ctx_->funcs && ctx_->funcs->has(e.strValue))
                unsupported("Dynamic tier: @" + e.strValue + " names a co-compiled user function "
                            "(not resolvable by name in the runtime registry) (v1)");
            return "numkit_rt::make_handle(\"" + e.strValue + "\")";
        }
        // @(params)expr -> a CAPTURE-FREE closure (G5b). The parser stored the
        // func2str-normalized source in strValue; create it via the interpreter
        // (make_closure -> numkit_eval), which owns anon-fn semantics. SOUND ONLY when
        // the body captures nothing the runtime workspace can't bind: REFUSE if any
        // free identifier is an enclosing local (types_/arrays_) or a co-compiled
        // user function (ctx_->funcs) -- those are not in the runtime workspace, so
        // numkit_eval would resolve them differently than the interpreter would capture
        // them. A nested anonymous function is also refused (v1). Each such case
        // falls back to the fully-interpreted tier (which captures correctly).
        std::set<std::string> freeIds;
        bool                  nestedAnon = false;
        collectAnonRefIds(*e.children[0], freeIds, nestedAnon);
        if (nestedAnon)
            unsupported("Dynamic tier: nested anonymous function (v1)");
        const std::set<std::string> params(e.paramNames.begin(), e.paramNames.end());
        // Free identifiers (excluding the anon's own params) split three ways: a
        // co-compiled USER fn reference -> REFUSE (the runtime cannot resolve the
        // compiled spec by name inside the closure); a captured enclosing LOCAL ->
        // bind it by value (G5c); a builtin / global -> resolved by the runtime at
        // call time (nothing to capture). v1 captures UNBOXABLE-SCALAR locals only
        // (an array / Dynamic capture refuses -> interpreted fallback, sound).
        std::vector<std::string> captures;
        for (const std::string &id : freeIds) {
            if (params.count(id)) continue;  // bound by the anon's own params
            if (ctx_ && ctx_->funcs && ctx_->funcs->has(id))
                unsupported("Dynamic tier: closure references a co-compiled user function '" + id
                            + "' (not resolvable by name in the runtime) (v1)");
            if (isArrayVar(id))
                unsupported("Dynamic tier: closure captures a non-scalar local '" + id + "' (v1)");
            if (types_.has(id)) {
                if (!isUnboxableScalarType(types_.get(id).type))
                    unsupported("Dynamic tier: closure captures a non-scalar local '" + id
                                + "' (v1)");
                captures.push_back(id);  // an unboxable-scalar captured local
            }
            // else: a builtin / global -> resolved at call time (no capture).
        }
        if (e.strValue.empty())
            unsupported("Dynamic tier: anonymous function with no reconstructed source");
        bool              ok  = false;
        const std::string esc = cEscape(e.strValue, ok);
        if (!ok)
            unsupported("Dynamic tier: anonymous function source has a non-printable byte (v1)");
        if (captures.empty())
            return "numkit_rt::make_closure(\"" + esc + "\")";  // capture-free (G5b)
        // capturing (G5c): box each captured scalar local + bind it by name.
        std::string names, vals;
        for (const std::string &id : captures) {
            names += (names.empty() ? "" : ", ") + ("\"" + id + "\"");
            vals  += (vals.empty() ? "" : ", ") + ("numkit_rt::val::scalar(" + cppIdent(id) + ")");
        }
        return "numkit_rt::make_closure_captured(\"" + esc + "\", {" + names + "}, {" + vals + "})";
    }
    case NodeType::MATRIX_LITERAL: {
        if (e.children.empty())
            return "numkit_rt::val::matrix(nullptr, 0, 0)";
        const std::size_t R = e.children.size();
        const std::size_t C = e.children[0] ? e.children[0]->children.size() : 0;
        if (C == 0)
            return "numkit_rt::val::matrix(nullptr, 0, 0)";
        bool rectDoubleScalar = true;
        for (const auto &rowN : e.children) {
            if (!rowN || rowN->children.size() != C) {
                rectDoubleScalar = false;
                break;
            }
            for (const auto &el : rowN->children) {
                if (!el) unsupported("Dynamic tier: null matrix element");
                const AbstractValue ev = inferExpr(*el, types_, reg_, classes_);
                if (!ev.type.isConcrete() || !ev.type.shape.isScalar()
                    || ev.type.dtype != ValueType::DOUBLE) {
                    rectDoubleScalar = false;
                    break;
                }
            }
            if (!rectDoubleScalar) break;
        }
        if (!rectDoubleScalar) {
            std::string rowVals;
            for (const auto &rowN : e.children) {
                if (!rowN || rowN->children.empty())
                    unsupported("Dynamic tier: empty matrix row");
                std::string rowExpr;
                if (rowN->children.size() == 1) {
                    rowExpr = emitDynamicExpr(*rowN->children[0]);
                } else {
                    std::string cols;
                    for (const auto &el : rowN->children) {
                        if (!el) unsupported("Dynamic tier: null matrix element");
                        cols += (cols.empty() ? "" : ", ") + emitDynamicExpr(*el);
                    }
                    rowExpr = "numkit_rt::call_dynv(\"horzcat\", {" + cols + "})";
                }
                rowVals += (rowVals.empty() ? "" : ", ") + rowExpr;
            }
            if (e.children.size() == 1) return rowVals;
            return "numkit_rt::call_dynv(\"vertcat\", {" + rowVals + "})";
        }
        std::string elems;
        for (std::size_t c = 0; c < C; ++c) {
            for (std::size_t r = 0; r < R; ++r) {
                if (!elems.empty()) elems += ", ";
                elems += emitExpr(*e.children[r]->children[c]);
            }
        }
        return "([&]() { const double _numkit_m[] = {" + elems
               + "}; return numkit_rt::val::matrix(_numkit_m, " + std::to_string(R) + ", "
               + std::to_string(C) + "); }())";
    }
    default:
        unsupported("Dynamic tier: expression node kind");
    }
}

std::string Emitter::emitCondition(const ASTNode &c)
{
    const AbstractValue cv = inferExpr(c, types_, reg_, classes_);
    // Dynamic tier (DESIGN.md §10 C1): a Dynamic condition reduces to MATLAB
    // truthiness via the runtime — the non-poisoning sink that lets an
    // un-typeable value steer control flow while the branches stay typed.
    if (bridge_ && cv.type.isDynamic())
        return emitDynamicExpr(c) + ".truth()";
    return emitExpr(c);
}

std::string Emitter::emitBuiltinCall(const std::string &name, const ASTNode &call)
{
    const std::size_t nargs = call.children.size() - 1;

    // numel / length / ndims of an array variable (or struct array field). 1-D: the length
    // companion. 2-D: numel = rows*cols; length = max(rows,cols). N-D: product / max of dims.
    if ((name == "numel" || name == "length" || name == "ndims") && nargs == 1) {
        const std::string arrVar = resolveArrayVar(*call.children[1]);
        if (!arrVar.empty()) {
            const ArrayInfo &ai = arrays_.at(arrVar);
            if (name == "ndims")
                return "static_cast<double>("
                       + std::to_string(ai.isND ? ai.ndDims.size() : std::size_t{2}) + ")";
            if (ai.isND) {
                std::string e = ai.ndDims[0];
                for (std::size_t i = 1; i < ai.ndDims.size(); ++i)
                    e = name == "numel" ? (e + " * " + ai.ndDims[i])
                                        : ("(" + e + " > " + ai.ndDims[i] + " ? " + e + " : "
                                           + ai.ndDims[i] + ")");  // length = max dim
                return "static_cast<double>(" + e + ")";
            }
            if (!ai.is2D)
                return "static_cast<double>(" + ai.lenVar + ")";
            if (name == "numel")
                return "static_cast<double>(" + ai.rowsVar + " * " + ai.colsVar + ")";
            return "static_cast<double>(" + ai.rowsVar + " > " + ai.colsVar + " ? " + ai.rowsVar
                   + " : " + ai.colsVar + ")";  // length = max(rows, cols)
        }
    }

    // Type/shape query predicates -> a LOGICAL scalar. Pure expressions (no loop /
    // no bridge) -> they compose in any context, incl. an `if` condition.
    //   isempty(x)  : numel == 0   (a scalar is never empty   -> "false")
    //   isscalar(x) : numel == 1   (a scalar has one element  -> "true")
    // An array var lowers to a numel compare (1-D length, 2-D rows*cols, N-D
    // product of dims); a scalar operand to a compile-time constant. A non-array /
    // non-scalar (Dynamic) arg falls through to the boundary (refuse, never
    // miscompile).
    if ((name == "isempty" || name == "isscalar") && nargs == 1) {
        const ASTNode    &arg    = *call.children[1];
        const char       *cmp    = name == "isempty" ? ") == 0)" : ") == 1)";
        const std::string arrVar = resolveArrayVar(arg);
        if (!arrVar.empty()) {
            const ArrayInfo &ai = arrays_.at(arrVar);
            std::string      n  = ai.lenVar;
            if (ai.isND) {
                n = ai.ndDims[0];
                for (std::size_t i = 1; i < ai.ndDims.size(); ++i) n += " * " + ai.ndDims[i];
            } else if (ai.is2D) {
                n = ai.rowsVar + " * " + ai.colsVar;
            }
            return "((" + n + cmp;
        }
        if (inferExpr(arg, types_, reg_, classes_).type.shape.kind == ShapeKind::Scalar)
            return name == "isempty" ? "false" : "true";
    }

    // isreal(x): true iff x has no imaginary part. A codegen value's complexity is
    // fixed by its STATIC dtype, so this is a compile-time constant -- a COMPLEX
    // operand -> "false", any other concrete dtype -> "true". A Dynamic arg falls
    // through to the boundary.
    if (name == "isreal" && nargs == 1) {
        const InferredType at = inferExpr(*call.children[1], types_, reg_, classes_).type;
        if (at.isConcrete())
            return at.dtype == ValueType::COMPLEX ? "false" : "true";
    }

    // isrow / iscolumn / isvector: 2-D orientation predicates -> a LOGICAL scalar.
    if ((name == "isrow" || name == "iscolumn" || name == "isvector") && nargs == 1) {
        const ASTNode    &arg    = *call.children[1];
        const std::string arrVar = resolveArrayVar(arg);
        if (!arrVar.empty()) {
            const ArrayInfo &ai = arrays_.at(arrVar);
            if (ai.isND)
                return "false";  // ndims >= 3 -> not a row / column / vector
            if (ai.is2D) {
                if (name == "isrow")    return "((" + ai.rowsVar + ") == 1)";
                if (name == "iscolumn") return "((" + ai.colsVar + ") == 1)";
                return "((" + ai.rowsVar + ") == 1 || (" + ai.colsVar + ") == 1)";
            }
            if (name == "isvector")
                return "true";  // a 1-D buffer is always a vector
            if (ai.orient == VecOrient::Row)
                return name == "isrow" ? std::string("true") : "(" + ai.lenVar + " == 1)";
            if (ai.orient == VecOrient::Col)
                return name == "iscolumn" ? std::string("true") : "(" + ai.lenVar + " == 1)";
            // orientation erased -> can't decide isrow/iscolumn -> fall through (boundary)
        } else if (inferExpr(arg, types_, reg_, classes_).type.shape.kind == ShapeKind::Scalar) {
            return "true";  // a 1x1 scalar is a row, a column, and a vector
        }
    }

    // dtype-classification predicates -> a LOGICAL scalar. A codegen value's dtype
    // is STATIC, so each is a compile-time constant from the operand's dtype:
    //   isnumeric : float (double/single) | integer (int*/uint*) | complex
    //   isfloat   : double | single | complex
    //   isinteger : int* | uint*
    //   ischar    : char        islogical : logical
    // A Dynamic arg falls through to the boundary (refuse, never miscompile).
    if ((name == "isnumeric" || name == "isfloat" || name == "isinteger"
         || name == "ischar" || name == "islogical")
        && nargs == 1) {
        const InferredType at = inferExpr(*call.children[1], types_, reg_, classes_).type;
        if (at.isConcrete()) {
            const ValueType dt = at.dtype;
            bool             v;
            if (name == "isnumeric")
                v = isIntegerType(dt) || isFloatType(dt) || dt == ValueType::COMPLEX;
            else if (name == "isfloat")
                v = isFloatType(dt) || dt == ValueType::COMPLEX;
            else if (name == "isinteger")
                v = isIntegerType(dt);
            else if (name == "ischar")
                v = dt == ValueType::CHAR;
            else  // islogical
                v = dt == ValueType::LOGICAL;
            return v ? "true" : "false";
        }
    }

    // size(A, dim) with a compile-time literal dim: the dim's size (2-D
    // rows/cols, N-D the dim, out-of-range -> 1).
    if (name == "size" && nargs == 2 && call.children[2]->type == NodeType::NUMBER_LITERAL) {
        const std::string arrVar = resolveArrayVar(*call.children[1]);
        if (!arrVar.empty()) {
            const ArrayInfo &ai  = arrays_.at(arrVar);
            const double     kd  = call.children[2]->numValue;
            const auto       k   = static_cast<std::size_t>(kd);
            const bool       kok = kd >= 1.0 && static_cast<double>(k) == kd;
            if (kok && (ai.isND || ai.is2D || ai.orient != VecOrient::Unknown))
                return "static_cast<double>(" + dimExpr(ai, k - 1) + ")";
        }
    }

    // Complex accessors: real/imag/angle/conj (scalar). std::real / std::imag /
    // std::arg accept both double and complex; conj of a REAL value is the
    // identity (std::conj(double) would return a std::complex, mismatching the
    // real-preserving transfer), so emit the bare value there.
    if (nargs == 1
        && (name == "real" || name == "imag" || name == "angle" || name == "conj")) {
        const std::string arg = emitExpr(*call.children[1]);
        if (name == "real")  return "std::real(" + arg + ")";
        if (name == "imag")  return "std::imag(" + arg + ")";
        if (name == "angle") return "std::arg(" + arg + ")";
        const ValueType adt = inferExpr(*call.children[1], types_, reg_, classes_).type.dtype;
        return adt == ValueType::COMPLEX ? ("std::conj(" + arg + ")") : ("(" + arg + ")");  // conj
    }

    // deg2rad / rad2deg: a real scalar scaling. No std:: fn -> emit nk's exact
    // constant (misc.cpp: x * k, k computed from these literals as pi/180 or 180/pi) so
    // it is bit-identical to the interpreter. Typed scalar-only (realScalarMathUnary-
    // Transfer); an array / complex arg is Dynamic -> bridged.
    if (nargs == 1 && (name == "deg2rad" || name == "rad2deg")) {
        const std::string x = emitExpr(*call.children[1]);
        return name == "deg2rad" ? "((" + x + ") * (3.14159265358979323846 / 180.0))"
                                 : "((" + x + ") * (180.0 / 3.14159265358979323846))";
    }

    if (!elementCtx_.empty() && !elementRows_.empty() && !elementCols_.empty()
        && (nargs == 1 || nargs == 2) && (name == "zeros" || name == "ones")) {
        auto dimArg = [&](const ASTNode &n) -> std::string {
            if (n.type == NodeType::NUMBER_LITERAL && n.numValue >= 0.0
                && std::floor(n.numValue) == n.numValue)
                return std::to_string(static_cast<std::size_t>(n.numValue));
            return "numkit_rt::dim(" + emitExpr(n) + ")";
        };
        const std::string rExpr = dimArg(*call.children[1]);
        const std::string cExpr = nargs == 2 ? dimArg(*call.children[2]) : rExpr;
        const std::string val   = name == "zeros" ? "0.0" : "1.0";
        if (rExpr == elementRows_ && cExpr == elementCols_) return val;
        return "([&]() -> double { if ((" + rExpr + ") != (" + elementRows_ + ") || (" + cExpr
               + ") != (" + elementCols_
               + ")) throw std::out_of_range(\"numkit: array dimensions must match\"); return " + val
               + "; }())";
    }

    // sign(x): MATLAB's signum for a REAL operand -> 1 / -1 / 0, with sign(0)=0 and
    // sign(NaN)=NaN. Not a std:: fn; inline via a single-eval IIFE. realOnlyMathUnary-
    // Transfer types sign for real DOUBLE/SINGLE only (complex -> Dynamic -> bridged), so
    // the operand here is always real. Lights up the scalar form AND the elementwise fill
    // (collectElementwise recognises sign), e.g. C = sign(A).
    if (nargs == 1 && name == "sign") {
        const std::string x = emitExpr(*call.children[1]);
        return "([](double _numkit_x){ return _numkit_x > 0.0 ? 1.0 : (_numkit_x < 0.0 ? -1.0 "
               ": (_numkit_x != _numkit_x ? _numkit_x : 0.0)); })(" + x + ")";
    }

    // any(<expr>) / all(<expr>) over a single 1-D array -> a LOGICAL scalar, as an IIFE so it
    // works in ANY expression position (an if-condition, a sub-expression, a whole rhs):
    // any short-circuits to true on the first nonzero, all to false on the first zero. EXACT +
    // order-independent. The arg is FUSED (elementCtx_ makes the whole array emit arr[_numkit_aa_i]
    // -- a fresh loop var, save/restore so a nesting fusion is unaffected). SINGLE-array 1-D
    // elementwise only; a 2-D any(A) is column-wise (a ROW VECTOR, not a scalar -> the scalar
    // gate fails -> bridged). A bare-VAR arg works too (collectElementwise of a whole array).
    if (nargs == 1 && (name == "any" || name == "all")) {
        std::set<std::string> srcArrays;
        const bool            pureEw = collectElementwise(*call.children[1], srcArrays);
        if (pureEw && srcArrays.size() == 1 && !arrays_.at(*srcArrays.begin()).is2D
            && !arrays_.at(*srcArrays.begin()).isND
            && inferExpr(call, types_, reg_, classes_).type.shape.isScalar()) {
            const ArrayInfo  &ba       = arrays_.at(*srcArrays.begin());
            const bool        isAny    = name == "any";
            const std::string saved     = elementCtx_;
            const std::string savedRows = elementRows_;
            const std::string savedCols = elementCols_;
            const auto        savedMap  = elementIdxMap_;
            elementIdxMap_.clear();
            elementRows_               = ba.orient == VecOrient::Col ? ba.lenVar : std::string("1");
            elementCols_               = ba.orient == VecOrient::Row ? ba.lenVar : std::string("1");
            elementCtx_                = "_numkit_aa_i";  // whole array in the arg -> arr[_numkit_aa_i]
            const std::string maskExpr = emitExpr(*call.children[1]);
            elementCtx_                = saved;
            elementRows_               = savedRows;
            elementCols_               = savedCols;
            elementIdxMap_             = savedMap;
            return "([&]() -> bool { for (std::size_t _numkit_aa_i = 0; _numkit_aa_i < (" + ba.lenVar
                   + "); ++_numkit_aa_i) if ((" + maskExpr + ") "
                   + (isAny ? "!= 0) return true; return false; }())"
                            : "== 0) return false; return true; }())");
        }
    }

    // min(<expr>) / max(<expr>) over a single 1-D DOUBLE array -> a scalar, as an IIFE (works
    // in any expr position). min/max are EXACT + order-independent (already native for a VAR),
    // so an inline elementwise arg (min(abs(x)), max(x.^2), ...) folds natively. NaN-skipping
    // mirrors the VAR reduction: seed = expr-at-0, then for i>=1 update on (v cmp acc) OR when
    // acc is NaN (so the first non-NaN seeds it; all-NaN stays NaN). SINGLE-array 1-D
    // elementwise only -- a 2-D min(A) is column-wise (a ROW VECTOR -> scalar gate fails ->
    // bridged). A bare-VAR arg works too (the statement-level VAR reduction takes a whole-rhs
    // r=min(v); this catches min(v) in expr position).
    if (nargs == 1 && (name == "min" || name == "max")) {
        std::set<std::string> srcArrays;
        const bool            pureEw = collectElementwise(*call.children[1], srcArrays);
        if (pureEw && srcArrays.size() == 1 && !arrays_.at(*srcArrays.begin()).is2D
            && !arrays_.at(*srcArrays.begin()).isND
            && arrays_.at(*srcArrays.begin()).dtype == ValueType::DOUBLE
            && inferExpr(call, types_, reg_, classes_).type.shape.isScalar()) {
            const ArrayInfo  &ba       = arrays_.at(*srcArrays.begin());
            const std::string cmp      = name == "max" ? ">" : "<";
            const std::string saved     = elementCtx_;
            const std::string savedRows = elementRows_;
            const std::string savedCols = elementCols_;
            const auto        savedMap  = elementIdxMap_;
            elementIdxMap_.clear();
            elementRows_               = ba.orient == VecOrient::Col ? ba.lenVar : std::string("1");
            elementCols_               = ba.orient == VecOrient::Row ? ba.lenVar : std::string("1");
            elementCtx_                = "0";  // seed: expr at element 0
            const std::string seed     = emitExpr(*call.children[1]);
            elementCtx_                = "_numkit_aa_i";  // body: expr at the loop index
            const std::string bodyE    = emitExpr(*call.children[1]);
            elementCtx_                = saved;
            elementRows_               = savedRows;
            elementCols_               = savedCols;
            elementIdxMap_             = savedMap;
            return "([&]() -> double { if ((" + ba.lenVar
                   + ") == 0) throw std::out_of_range(\"numkit: min/max of empty array requires array output\"); double _numkit_acc = ("
                   + seed + "); for (std::size_t _numkit_aa_i = 1; _numkit_aa_i < (" + ba.lenVar
                   + "); ++_numkit_aa_i) { double _numkit_v = (" + bodyE + "); if (_numkit_v " + cmp
                   + " _numkit_acc || _numkit_acc != _numkit_acc) _numkit_acc = _numkit_v; } return _numkit_acc; }())";
        }
    }

    if (!bridge_ && nargs == 1 && (name == "sum" || name == "prod" || name == "mean")) {
        std::set<std::string> srcArrays;
        const bool            pureEw = collectElementwise(*call.children[1], srcArrays);
        if (pureEw && srcArrays.size() == 1 && !arrays_.at(*srcArrays.begin()).is2D
            && !arrays_.at(*srcArrays.begin()).isND
            && arrays_.at(*srcArrays.begin()).dtype == ValueType::DOUBLE
            && inferExpr(call, types_, reg_, classes_).type.shape.isScalar()) {
            const ArrayInfo  &ba       = arrays_.at(*srcArrays.begin());
            const bool        isProd   = name == "prod";
            const bool        isMean   = name == "mean";
            const std::string saved     = elementCtx_;
            const std::string savedRows = elementRows_;
            const std::string savedCols = elementCols_;
            const auto        savedMap  = elementIdxMap_;
            elementIdxMap_.clear();
            elementRows_               = ba.orient == VecOrient::Col ? ba.lenVar : std::string("1");
            elementCols_               = ba.orient == VecOrient::Row ? ba.lenVar : std::string("1");
            elementCtx_                = "_numkit_aa_i";
            const std::string bodyE    = emitExpr(*call.children[1]);
            elementCtx_                = saved;
            elementRows_               = savedRows;
            elementCols_               = savedCols;
            elementIdxMap_             = savedMap;
            const std::string init   = isProd ? "1.0" : "0.0";
            const std::string op     = isProd ? "*=" : "+=";
            const std::string ret    = isMean
                                           ? ("_numkit_acc / static_cast<double>(" + ba.lenVar + ")")
                                           : std::string("_numkit_acc");
            return "([&]() -> double { double _numkit_acc = " + init
                   + "; for (std::size_t _numkit_aa_i = 0; _numkit_aa_i < (" + ba.lenVar
                   + "); ++_numkit_aa_i) { _numkit_acc " + op + " (" + bodyE
                   + "); } return " + ret + "; }())";
        }
    }

    if (nargs == 1)
        if (const char *fn = unaryMathStd(name))
            return std::string("std::") + fn + "(" + emitExpr(*call.children[1]) + ")";

    if (nargs == 2)
        if (const char *fn = binaryMathStd(name))
            return std::string("std::") + fn + "(" + emitExpr(*call.children[1]) + ", "
                   + emitExpr(*call.children[2]) + ")";

    // Bridged fallback (opt-in, DESIGN.md §6a): a builtin the emitter cannot
    // lower goes through the C-ABI to the runtime — but ONLY when inference
    // proves the result is an unboxed real scalar (Contract 2: the fast form
    // is emitted under a proven precondition; otherwise we still throw).
    if (bridge_) {
        const AbstractValue res = inferExpr(call, types_, reg_, classes_);
        if (res.type.isUnboxableScalar() && res.type.dtype != ValueType::COMPLEX) {
            bool allScalarArgs = true;
            for (std::size_t i = 1; i < call.children.size(); ++i) {
                const InferredType at = inferExpr(*call.children[i], types_, reg_, classes_).type;
                if (!isUnboxableScalarType(at) || at.dtype == ValueType::COMPLEX) {
                    allScalarArgs = false;
                    break;
                }
            }
            if (allScalarArgs) {
                std::string out = "numkit_rt::bridge_scalar(\"" + name + "\", {";
                for (std::size_t i = 1; i < call.children.size(); ++i)
                    out += (i > 1 ? ", " : "") + emitExpr(*call.children[i]);
                out += "})";
                return out;
            }
            std::string dynArgs;
            for (std::size_t i = 1; i < call.children.size(); ++i)
                dynArgs += (i > 1 ? ", " : "") + emitDynamicExpr(*call.children[i]);
            const std::string unbox =
                res.type.dtype == ValueType::LOGICAL ? ".truth()" : ".to_scalar()";
            return "numkit_rt::call_dynv(\"" + name + "\", {" + dynArgs + "})" + unbox;
        }
    }

    unsupported("builtin call '" + name + "' (arity " + std::to_string(nargs) + ")");
}

void Emitter::appendStructCallArgs(const std::string &prefix, const StructLayout &layout,
                                   std::string &argList)
{
    for (const auto &f : layout.fields) {
        const std::string   fld = prefix + "_" + f.first;
        const InferredType &ft  = f.second;
        if (isUnboxableScalarType(ft) || ft.isObject()) {
            if (!argList.empty()) argList += ", ";
            argList += cppIdent(fld);
        } else if (is2DMatrixType(ft) && isArrayVar(fld)) {
            const ArrayInfo &ai      = arrays_.at(fld);
            const auto       aliasIt = callAliasInputs_.find(fld);
            const std::string ptr    = aliasIt != callAliasInputs_.end() ? aliasIt->second.first : ai.dataExpr;
            if (!argList.empty()) argList += ", ";
            argList += ptr + ", " + ai.rowsVar + ", " + ai.colsVar;
        } else if (ft.isConcrete() && ft.shape.isNDims() && isArrayVar(fld)) {
            const ArrayInfo &ai      = arrays_.at(fld);
            const auto       aliasIt = callAliasInputs_.find(fld);
            const std::string ptr    = aliasIt != callAliasInputs_.end() ? aliasIt->second.first : ai.dataExpr;
            if (!argList.empty()) argList += ", ";
            argList += ptr;
            for (const std::string &dv : ai.ndDims) argList += ", " + dv;
        } else if (isBufferArrayType(ft) && isArrayVar(fld)) {
            const ArrayInfo &ai      = arrays_.at(fld);
            const auto       aliasIt = callAliasInputs_.find(fld);
            const std::string ptr    = aliasIt != callAliasInputs_.end() ? aliasIt->second.first : ai.dataExpr;
            const std::string len    = aliasIt != callAliasInputs_.end() ? aliasIt->second.second : ai.lenVar;
            if (!argList.empty()) argList += ", ";
            argList += ptr + ", " + len;
        } else if (ft.isStruct() && ft.structLayout && !ft.structLayout->fields.empty()) {
            appendStructCallArgs(fld, *ft.structLayout, argList);
        } else {
            unsupported("struct call argument field '" + fld + "' has an unsupported type");
        }
    }
}

void Emitter::appendStructOutArgs(const std::string &prefix, const StructLayout &layout,
                                  std::string &argList)
{
    for (const auto &f : layout.fields) {
        const std::string   fld = prefix + "_" + f.first;
        const InferredType &ft  = f.second;
        if (isUnboxableScalarType(ft) || ft.isObject() || isBufferArrayType(ft)) {
            if (!argList.empty()) argList += ", ";
            argList += cppIdent(fld);
            types_.set(fld, {ft, ConstVal::unknown()});
        } else if (ft.isStruct() && ft.structLayout && !ft.structLayout->fields.empty()) {
            types_.set(fld, {ft, ConstVal::unknown()});
            appendStructOutArgs(fld, *ft.structLayout, argList);
        } else {
            unsupported("struct-return field '" + fld + "' has an unsupported type");
        }
    }
}

void Emitter::collectStructOutRefArrays(const std::string &prefix, const StructLayout &layout,
                                        std::set<std::string> &out) const
{
    for (const auto &f : layout.fields) {
        const std::string   fld = prefix + "_" + f.first;
        const InferredType &ft  = f.second;
        if (isBufferArrayType(ft) && isArrayVar(fld)) {
            out.insert(fld);
        } else if (ft.isStruct() && ft.structLayout) {
            collectStructOutRefArrays(fld, *ft.structLayout, out);
        }
    }
}

void Emitter::prepareCallInputAliases(const ASTNode &call, std::size_t firstArgIdx,
                                      const std::set<std::string> &outRefArrays,
                                      std::vector<std::string> &aliasDecls)
{
    callAliasInputs_.clear();
    if (outRefArrays.empty()) return;
    auto aliasOne = [&](const std::string &arrVar) {
        if (!outRefArrays.count(arrVar) || callAliasInputs_.count(arrVar)) return;
        const ArrayInfo &ai = arrays_.at(arrVar);
        if (!ai.isLocal) return;
        const std::string tmp = "_numkit_in_alias_" + std::to_string(aliasCounter_++);
        aliasDecls.push_back("const auto " + tmp + " = " + cppIdent(arrVar) + ";");
        callAliasInputs_[arrVar] = {tmp + ".data()", tmp + ".size()"};
    };
    auto walkStructIn = [&](auto &self, const std::string &prefix,
                            const StructLayout &layout) -> void {
        for (const auto &f : layout.fields) {
            const std::string   fld = prefix + "_" + f.first;
            const InferredType &ft  = f.second;
            if (isBufferArrayType(ft) && isArrayVar(fld)) {
                aliasOne(fld);
            } else if (ft.isStruct() && ft.structLayout) {
                self(self, fld, *ft.structLayout);
            }
        }
    };
    for (std::size_t i = firstArgIdx; i < call.children.size(); ++i) {
        const ASTNode      &arg    = *call.children[i];
        const std::string   arrVar = resolveArrayVar(arg);
        const AbstractValue av     = inferExpr(arg, types_, reg_, classes_);
        if (!arrVar.empty()) {
            aliasOne(arrVar);
        } else if (av.type.isStruct() && av.type.structLayout) {
            std::string prefix;
            if (arg.type == NodeType::IDENTIFIER)
                prefix = "_numkit_fld_" + arg.strValue;
            else if (arg.type == NodeType::FIELD_ACCESS
                     || arg.type == NodeType::DYNAMIC_FIELD_ACCESS)
                prefix = structFieldLocal(arg);
            if (!prefix.empty()) walkStructIn(walkStructIn, prefix, *av.type.structLayout);
        }
    }
}

void Emitter::appendIgnoreOutArg(const std::string &tmpName, const InferredType &ot,
                                 std::vector<std::string> &ignoreDecls, std::string &argList)
{
    if (isUnboxableScalarType(ot)) {
        ignoreDecls.push_back(cppScalarType(ot.dtype) + " " + tmpName + " = "
                              + zeroLiteral(ot.dtype) + ";");
        if (!argList.empty()) argList += ", ";
        argList += tmpName;
    } else if (ot.isObject()) {
        const ClassInfo *oci = classes_ ? classes_->byId(ot.classId) : nullptr;
        if (!oci) unsupported("ignored object output with no class registry");
        const std::string clsName = cppGlobalIdent(oci->name);
        ignoreDecls.push_back(oci->isHandle ? ("numkit_rt::handle<" + clsName + "> " + tmpName + ";")
                                            : (clsName + " " + tmpName + "{};"));
        if (!argList.empty()) argList += ", ";
        argList += tmpName;
    } else if (isBufferArrayType(ot)) {
        ignoreDecls.push_back("std::vector<" + cppArrayElemType(ot.dtype) + "> " + tmpName + ";");
        if (!argList.empty()) argList += ", ";
        argList += tmpName;
    } else if (ot.isStruct() && ot.structLayout && !ot.structLayout->fields.empty()) {
        for (const auto &f : ot.structLayout->fields)
            appendIgnoreOutArg(tmpName + "_" + f.first, f.second, ignoreDecls, argList);
    } else {
        unsupported("ignored / unrequested multi-output has an unsupported type");
    }
}

void Emitter::emitStructCopy(const std::string &dstPrefix, const std::string &srcPrefix,
                             const StructLayout &layout)
{
    for (const auto &f : layout.fields) {
        const std::string   dstFld = dstPrefix + "_" + f.first;
        const std::string   srcFld = srcPrefix + "_" + f.first;
        const InferredType &ft     = f.second;
        if (isUnboxableScalarType(ft) || ft.isObject()) {
            line(cppIdent(dstFld) + " = " + cppIdent(srcFld) + ";");
            types_.set(dstFld, {ft, ConstVal::unknown()});
        } else if (isBufferArrayType(ft) && isArrayVar(dstFld) && isArrayVar(srcFld)) {
            const ArrayInfo &A = arrays_.at(srcFld);
            const ArrayInfo &B = arrays_.at(dstFld);
            std::string      numel;
            if (A.isND) {
                numel = A.ndDims[0];
                for (std::size_t i = 1; i < A.ndDims.size(); ++i) numel += " * " + A.ndDims[i];
            } else if (A.is2D) {
                numel = A.rowsVar + " * " + A.colsVar;
            } else {
                numel = A.lenVar;
            }
            if (B.isND && B.ndRuntimeLocal)
                for (std::size_t k = 0; k < B.ndDims.size(); ++k)
                    line(B.ndDims[k] + " = " + dimExpr(A, k) + ";");
            line(cppIdent(dstFld) + ".assign(" + A.dataExpr + ", " + A.dataExpr + " + (" + numel + "));");
            types_.set(dstFld, {ft, ConstVal::unknown()});
        } else if (ft.isStruct() && ft.structLayout) {
            types_.set(dstFld, {ft, ConstVal::unknown()});
            emitStructCopy(dstFld, srcFld, *ft.structLayout);
        } else {
            unsupported("struct copy field '" + dstFld + "' has an unsupported type");
        }
    }
}

std::string Emitter::emitMultiCallExpr(const std::string &mangled, const std::string &argList,
                                       const std::vector<InferredType> &outs)
{
    if (outs.empty()) unsupported("multi-output call with no outputs");
    const InferredType &out0        = outs[0];
    const bool          out0ByValue = isByValueReturnArrayType(out0);
    std::string         retCpp;
    if (isUnboxableScalarType(out0))
        retCpp = cppScalarType(out0.dtype);
    else if (out0.isObject())
        retCpp = cppObjectType(out0.classId, classes_);
    else if (out0ByValue)
        retCpp = "std::vector<" + cppArrayElemType(out0.dtype) + ">";
    else
        unsupported("expression-context multi-output call '" + mangled
                    + "' requires a scalar, object, or by-value array first output");

    const int                id       = multiCallCounter_++;
    std::string              callArgs = argList;
    std::vector<std::string> decls;
    const std::string        out0Tmp = "_numkit_mc_" + std::to_string(id) + "_0";
    for (std::size_t i = (out0ByValue ? 1 : 0); i < outs.size(); ++i) {
        const std::string tmp = "_numkit_mc_" + std::to_string(id) + "_" + std::to_string(i);
        appendIgnoreOutArg(tmp, outs[i], decls, callArgs);
    }
    std::string joinedDecls;
    for (const std::string &d : decls) joinedDecls += d + " ";
    if (out0ByValue)
        return "([&]() -> " + retCpp + " { " + joinedDecls + "return " + mangled + "(" + callArgs
               + "); }())";
    return "([&]() -> " + retCpp + " { " + joinedDecls + mangled + "(" + callArgs + "); return "
           + out0Tmp + "; }())";
}

// Append one interprocedural call argument: a scalar or object is passed by
// value (one C++ arg); an array VARIABLE is passed as `ptr, len` / `ptr, rows, cols` /
// `ptr, d0, ...`; a struct is recursively exploded into its field arguments.
void Emitter::appendCallArg(const ASTNode &arg, std::vector<InferredType> &argTypes,
                            std::string &argList)
{
    const AbstractValue av     = inferExpr(arg, types_, reg_, classes_);
    const std::string   arrVar = resolveArrayVar(arg);
    if (isUnboxableScalarType(av.type) || av.type.isObject()) {
        if (!argList.empty()) argList += ", ";
        argList += emitExpr(arg);
    } else if (is2DMatrixType(av.type) && !arrVar.empty()) {
        const ArrayInfo &ai      = arrays_.at(arrVar);
        const auto       aliasIt = callAliasInputs_.find(arrVar);
        const std::string ptr    = aliasIt != callAliasInputs_.end() ? aliasIt->second.first : ai.dataExpr;
        if (!argList.empty()) argList += ", ";
        argList += ptr + ", " + ai.rowsVar + ", " + ai.colsVar;
    } else if (av.type.isConcrete() && av.type.shape.isNDims() && !arrVar.empty()) {
        const ArrayInfo &ai      = arrays_.at(arrVar);
        const auto       aliasIt = callAliasInputs_.find(arrVar);
        const std::string ptr    = aliasIt != callAliasInputs_.end() ? aliasIt->second.first : ai.dataExpr;
        if (!argList.empty()) argList += ", ";
        argList += ptr;
        for (const std::string &dv : ai.ndDims) argList += ", " + dv;
    } else if (isBufferArrayType(av.type) && !arrVar.empty()) {
        const ArrayInfo &ai      = arrays_.at(arrVar);
        const auto       aliasIt = callAliasInputs_.find(arrVar);
        const std::string ptr    = aliasIt != callAliasInputs_.end() ? aliasIt->second.first : ai.dataExpr;
        const std::string len    = aliasIt != callAliasInputs_.end() ? aliasIt->second.second : ai.lenVar;
        if (!argList.empty()) argList += ", ";
        argList += ptr + ", " + len;  // ptr + len (local: .data()/.size())
    } else if (av.type.isStruct()) {
        std::string prefix;
        if (arg.type == NodeType::IDENTIFIER)
            prefix = "_numkit_fld_" + arg.strValue;
        else if (arg.type == NodeType::FIELD_ACCESS || arg.type == NodeType::DYNAMIC_FIELD_ACCESS)
            prefix = structFieldLocal(arg);
        if (prefix.empty() || !av.type.structLayout || av.type.structLayout->fields.empty())
            unsupported("struct call argument has no fields");
        appendStructCallArgs(prefix, *av.type.structLayout, argList);
    } else {
        unsupported("call argument must be a scalar, object, struct, or array variable (v1)");
    }
    argTypes.push_back(av.type);
}

// A call to a user-defined function within a program (ctx_ set). Arguments
// may be scalars, objects, structs, or array variables. Emits a direct call
// to the monomorphised specialisation and queues it for emission.
std::string Emitter::emitUserCall(const std::string &name, const ASTNode &call)
{
    const ASTNode *def = ctx_->funcs->find(name);
    if (!def) unsupported("user call to unknown function '" + name + "'");

    std::vector<InferredType> argTypes;
    std::string               argList;
    for (std::size_t i = 1; i < call.children.size(); ++i)
        appendCallArg(*call.children[i], argTypes, argList);
    if (argTypes.size() != def->paramNames.size())
        unsupported("arity mismatch calling '" + name + "'");

    const std::size_t nout    = def->returnNames.size();
    const std::string mangled = mangle(name, argTypes);
    if (nout >= 2) {
        const std::vector<InferredType> outs = reg_.applyMulti(name, toArgInfos(argTypes));
        if (outs.size() != nout)
            unsupported("multi-output call '" + name + "' has unknown output types");
        if (ctx_->seen.insert(mangled).second)
            ctx_->pending.push_back({def, argTypes, mangled});
        return emitMultiCallExpr(mangled, argList, outs);
    }
    if (nout == 1) {  // 0 -> void function, result discarded
        const InferredType ret = reg_.apply(name, toArgInfos(argTypes));
        const bool arrayByValue = isByValueReturnArrayType(ret);
        if (!isUnboxableScalarType(ret) && !ret.isObject() && !arrayByValue)
            unsupported("interprocedural call result must be an unboxed scalar, object, or a 1-D / "
                        "2-D / fully-known-N-D double/complex array (v1): '" + name + "'");
    }

    if (ctx_->seen.insert(mangled).second)
        ctx_->pending.push_back({def, argTypes, mangled});
    return mangled + "(" + argList + ")";
}

// A method call `obj.m(args)` (ctx_ set). The object is the implicit first
// argument; dispatch is monomorphic (the exact class is known). Emits a
// direct call to the specialisation `Class__m(self, args)` and queues it.
std::string Emitter::emitMethodCall(const ASTNode &call)
{
    const ASTNode &callee = *call.children[0];  // FIELD_ACCESS: method on object
    if (callee.children.empty()) unsupported("method call arity");
    if (!ctx_) unsupported("method call requires program emission (emitProgram)");

    const AbstractValue base = inferExpr(*callee.children[0], types_, reg_, classes_);
    if (!base.type.isObject() || !classes_) unsupported("method call on a non-object");
    const ClassInfo *ci = classes_->byId(base.type.classId);
    const ASTNode   *md = ci ? ci->method(callee.strValue) : nullptr;
    if (!md) unsupported("unknown method '" + callee.strValue + "'");

    std::vector<InferredType> argTypes;
    argTypes.push_back(base.type);                       // self is arg 0
    std::string argList = emitExpr(*callee.children[0]);  // the object expr
    for (std::size_t i = 1; i < call.children.size(); ++i)
        appendCallArg(*call.children[i], argTypes, argList);
    if (argTypes.size() != md->paramNames.size())
        unsupported("arity mismatch calling method '" + callee.strValue + "'");

    const std::size_t nout    = md->returnNames.size();
    const std::string mkey    = ci->name + "::" + callee.strValue;
    const std::string mangled = mangle(ci->name + "__" + callee.strValue, argTypes);
    if (nout >= 2) {
        const std::vector<InferredType> outs = reg_.applyMulti(mkey, toArgInfos(argTypes));
        if (outs.size() != nout)
            unsupported("multi-output method '" + callee.strValue + "' has unknown output types");
        if (ctx_->seen.insert(mangled).second)
            ctx_->pending.push_back({md, argTypes, mangled});
        return emitMultiCallExpr(mangled, argList, outs);
    }
    if (nout == 1) {  // 0 -> void method (in-place mutator), result discarded
        const InferredType ret = reg_.apply(mkey, toArgInfos(argTypes));
        if (!isUnboxableScalarType(ret) && !ret.isObject() && !isByValueReturnArrayType(ret))
            unsupported("method result must be an unboxed scalar, object, or array (v1): '"
                        + callee.strValue + "'");
    }

    if (ctx_->seen.insert(mangled).second)
        ctx_->pending.push_back({md, argTypes, mangled});
    return mangled + "(" + argList + ")";
}

// Object construction `Name(args)`. 1a: default construction only — a
// value class -> Name{} (fields default-initialised), a handle class ->
// numkit_rt::handle<Name>::make(). An explicit constructor (1b) and
// argument-taking construction are refused for now.
std::string Emitter::emitConstruct(const std::string &name, const ASTNode &call)
{
    const ClassInfo *ci = classes_->byName(name);
    if (!ci) unsupported("construction of unknown class '" + name + "'");
    const std::size_t nargs = call.children.size() - 1;
    const ASTNode    *ctor  = ci->method(name);

    if (ctor) {
        // Explicit constructor: emit a call to its specialisation, whose
        // OUTPUT object is seeded as Object(this class) so its field writes
        // type (the object can't be inferred from field writes alone).
        if (!ctx_) unsupported("constructor call requires program emission (emitProgram)");
        if (ctor->returnNames.size() != 1)
            unsupported("constructor of '" + name + "' must have one output");
        if (ctor->paramNames.size() != nargs)
            unsupported("constructor of '" + name + "' arity mismatch");
        std::vector<InferredType> argTypes;
        std::string               argList;
        for (std::size_t i = 1; i < call.children.size(); ++i) {
            const AbstractValue av = inferExpr(*call.children[i], types_, reg_, classes_);
            if (!isUnboxableScalarType(av.type))
                unsupported("constructor arg must be an unboxed scalar (v1): '" + name + "'");
            argTypes.push_back(av.type);
            argList += (i > 1 ? ", " : "") + emitExpr(*call.children[i]);
        }
        const std::string mangled = mangle(ci->name + "__ctor", argTypes);
        if (ctx_->seen.insert(mangled).second) {
            CallSite cs;
            cs.def       = ctor;
            cs.argTypes  = argTypes;
            cs.mangled   = mangled;
            cs.extraSeed = {{ctor->returnNames[0], InferredType::object(ci->id)}};
            ctx_->pending.push_back(cs);
        }
        return mangled + "(" + argList + ")";
    }

    // No explicit constructor: default construction.
    if (nargs != 0)
        unsupported("class '" + name + "' has no constructor accepting arguments");
    const std::string clsName = cppGlobalIdent(ci->name);
    return ci->isHandle ? "numkit_rt::handle<" + clsName + ">::make()" : clsName + "{}";
}

bool Emitter::collectElementwise(const ASTNode &e, std::set<std::string> &arrays) const
{
    switch (e.type) {
    case NodeType::NUMBER_LITERAL:
    case NodeType::BOOL_LITERAL:
    case NodeType::IMAG_LITERAL:
        return true;
    case NodeType::IDENTIFIER:
        if (isArrayVar(e.strValue)) arrays.insert(e.strValue);
        return true;  // a scalar var, or a whole array (recorded)
    case NodeType::FIELD_ACCESS:
    case NodeType::DYNAMIC_FIELD_ACCESS: {
        if (e.children.empty() || !e.children[0]) return false;
        const AbstractValue base = inferExpr(*e.children[0], types_, reg_, classes_);
        if (base.type.isObject()) return false;
        const std::string fld = structFieldLocal(e);
        if (fld.empty()) return false;
        if (isArrayVar(fld)) arrays.insert(fld);
        return true;
    }
    case NodeType::BINARY_OP: {
        // Operator SYMBOLS (the AST stores the symbol; inference maps it to the
        // transfer name). Unconditionally elementwise: + - .* ./ .\ .^. The
        // matrix forms * / are elementwise ONLY in their scalar-scaling cases —
        // mtimes (*) when EITHER operand is scalar (s*X == s.*X), mrdivide (/)
        // when the DENOMINATOR is scalar (X/s == X./s); s/X is a matrix divide.
        if (e.children.size() != 2) return false;
        static const std::set<std::string> kElementwise = {
            "+", "-", ".*", "./", ".\\", ".^",
            // Relational + elementwise-logical ops: the same broadcast shape rules
            // as arithmetic, a LOGICAL result. (Short-circuit && / || are
            // scalar-only and stay excluded.)
            "<", ">", "<=", ">=", "==", "~=", "&", "|"};
        auto isScalarArg = [&](const ASTNode &n) {
            return inferExpr(n, types_, reg_, classes_).type.shape.isScalar();
        };
        if (e.strValue == "*") {
            if (!isScalarArg(*e.children[0]) && !isScalarArg(*e.children[1])) return false;
        } else if (e.strValue == "/") {
            if (!isScalarArg(*e.children[1])) return false;
        } else if (kElementwise.count(e.strValue) == 0) {
            return false;
        }
        return collectElementwise(*e.children[0], arrays)
               && collectElementwise(*e.children[1], arrays);
    }
    case NodeType::UNARY_OP:
        return (e.strValue == "-" || e.strValue == "+" || e.strValue == "~")
               && e.children.size() == 1 && collectElementwise(*e.children[0], arrays);
    case NodeType::CALL: {
        // An ELEMENTWISE-math call (sin/cos/erf/atan2/…) over elementwise args
        // is itself elementwise — sin(x)[i] == std::sin(x[i]). An index `x(k)`
        // (callee is an array var) or any other call is not.
        if (e.children.empty() || e.children[0]->type != NodeType::IDENTIFIER) return false;
        const std::string &callee = e.children[0]->strValue;
        if (isArrayVar(callee)) return false;  // x(k) indexing -> not pure elementwise
        const std::size_t nargs = e.children.size() - 1;
        const bool isMath = (nargs == 1 && (unaryMathStd(callee) != nullptr || callee == "sign"))
                            || (nargs == 2 && binaryMathStd(callee) != nullptr);
        if (!isMath) {
            if ((nargs == 1 || nargs == 2) && (callee == "zeros" || callee == "ones")) {
                for (std::size_t i = 1; i < e.children.size(); ++i)
                    if (!inferExpr(*e.children[i], types_, reg_, classes_).type.shape.isScalar())
                        return false;
                return true;
            }
            if (nargs == 1
                && (callee == "sum" || callee == "prod" || callee == "mean"
                    || callee == "min" || callee == "max" || callee == "any"
                    || callee == "all" || callee == "numel" || callee == "length")
                && inferExpr(e, types_, reg_, classes_).type.shape.isScalar()) {
                std::set<std::string> innerArrays;
                if (collectElementwise(*e.children[1], innerArrays) && innerArrays.size() == 1
                    && !arrays_.at(*innerArrays.begin()).is2D
                    && !arrays_.at(*innerArrays.begin()).isND) {
                    return true;
                }
            }
            return false;
        }
        for (std::size_t i = 1; i < e.children.size(); ++i)
            if (!collectElementwise(*e.children[i], arrays)) return false;
        return true;
    }
    default:
        return false;  // indexing / fields / non-math calls -> not pure elementwise
    }
}

std::string Emitter::dimExpr(const ArrayInfo &a, std::size_t k) const
{
    if (a.isND) return k < a.ndDims.size() ? a.ndDims[k] : std::string("1");
    if (a.is2D) return k == 0 ? a.rowsVar : k == 1 ? a.colsVar : std::string("1");
    if (a.orient == VecOrient::Row) return k == 1 ? a.lenVar : std::string("1");
    if (a.orient == VecOrient::Col) return k == 0 ? a.lenVar : std::string("1");
    return std::string("1");
}

} // namespace numkit::codegen::detail
