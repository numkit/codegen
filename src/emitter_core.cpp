// codegen/src/emitter_core.cpp — scalar/type helpers, OptFacts, mangling, prepasses.

#include "emitter_internal.hpp"

namespace numkit::codegen {

// ── scalar dtype / literal helpers (3a) ───────────────────────────────
std::string cppScalarType(ValueType dtype)
{
    switch (dtype) {
    case ValueType::DOUBLE:  return "double";
    case ValueType::SINGLE:  return "float";
    case ValueType::COMPLEX: return "std::complex<double>";
    case ValueType::LOGICAL: return "bool";
    case ValueType::INT8:    return "std::int8_t";
    case ValueType::INT16:   return "std::int16_t";
    case ValueType::INT32:   return "std::int32_t";
    case ValueType::INT64:   return "std::int64_t";
    case ValueType::UINT8:   return "std::uint8_t";
    case ValueType::UINT16:  return "std::uint16_t";
    case ValueType::UINT32:  return "std::uint32_t";
    case ValueType::UINT64:  return "std::uint64_t";
    case ValueType::CHAR:    return "std::uint16_t";  // MATLAB char is a UTF-16 code unit
    default:
        throw std::runtime_error("cppScalarType: no scalar C++ mapping for this dtype");
    }
}

// The C++ ELEMENT type for an owned array buffer. Identical to cppScalarType
// except LOGICAL uses std::uint8_t, not bool: std::vector<bool> is bit-packed and
// exposes no .data(), which the buffer ABI (name.data()) relies on. A bool value
// assigns to / reads from a uint8 element as 0/1 (MATLAB logical 1-byte storage).
std::string cppArrayElemType(ValueType dtype)
{
    return dtype == ValueType::LOGICAL ? "std::uint8_t" : cppScalarType(dtype);
}

std::string formatDoubleLiteral(double v)
{
    if (std::isnan(v)) return "std::numeric_limits<double>::quiet_NaN()";
    if (std::isinf(v))
        return v < 0 ? "-std::numeric_limits<double>::infinity()"
                     : "std::numeric_limits<double>::infinity()";

    char buf[32];
    auto [ptr, ec] = std::to_chars(buf, buf + sizeof(buf), v);
    std::string s(buf, ptr);
    // Ensure it reads as a double in C++ (e.g. "2" -> "2.0", not an int).
    if (s.find('.') == std::string::npos && s.find('e') == std::string::npos &&
        s.find('E') == std::string::npos)
        s += ".0";
    return s;
}

namespace detail {

bool isCppKeyword(const std::string &name)
{
    static const std::unordered_set<std::string> kKeywords = {
        "alignas",      "alignof",      "and",           "and_eq",       "asm",
        "auto",         "bitand",       "bitor",         "bool",         "break",
        "case",         "catch",        "char",          "char8_t",      "char16_t",
        "char32_t",     "class",        "compl",         "concept",      "const",
        "consteval",    "constexpr",    "constinit",     "const_cast",   "continue",
        "co_await",     "co_return",    "co_yield",      "decltype",     "default",
        "delete",       "do",           "double",        "dynamic_cast", "else",
        "enum",         "explicit",     "export",        "extern",       "false",
        "float",        "for",          "friend",        "goto",         "if",
        "inline",       "int",          "long",          "mutable",      "namespace",
        "new",          "noexcept",     "not",           "not_eq",       "nullptr",
        "operator",     "or",           "or_eq",         "private",      "protected",
        "public",       "register",     "reinterpret_cast", "requires",  "return",
        "short",        "signed",       "sizeof",        "static",       "static_assert",
        "static_cast",  "struct",       "switch",        "template",     "this",
        "thread_local", "throw",        "true",          "try",          "typedef",
        "typeid",       "typename",     "union",         "unsigned",     "using",
        "virtual",      "void",         "volatile",      "wchar_t",      "while",
        "xor",          "xor_eq"};
    return kKeywords.count(name) != 0;
}

std::string cppIdent(const std::string &name)
{
    return isCppKeyword(name) ? ("_numkit_kw_" + name) : name;
}

std::string cppGlobalIdent(const std::string &name)
{
    return isCppKeyword(name) ? ("numkit_kw_" + name) : name;
}

// The zero / default-init literal for a hoisted local of `dtype`.
std::string zeroLiteral(ValueType dtype)
{
    switch (dtype) {
    case ValueType::DOUBLE:
    case ValueType::SINGLE:  return "0.0";
    case ValueType::COMPLEX: return "std::complex<double>(0.0, 0.0)";
    case ValueType::LOGICAL: return "false";
    default:                 return "0";  // integer dtypes
    }
}

// MATLAB binary operator -> C++ operator for the SCALAR case. Returns
// nullptr for power (handled via std::pow) or an unhandled operator.
const char *cppScalarBinOp(const std::string &op)
{
    if (op == "+")  return "+";
    if (op == "-")  return "-";
    if (op == "*" || op == ".*") return "*";
    if (op == "/" || op == "./") return "/";
    if (op == "==") return "==";
    if (op == "~=" || op == "!=") return "!=";
    if (op == "<")  return "<";
    if (op == ">")  return ">";
    if (op == "<=") return "<=";
    if (op == ">=") return ">=";
    if (op == "&&" || op == "&") return "&&";  // scalar logical
    if (op == "||" || op == "|") return "||";
    return nullptr;  // ^, .^ -> std::pow; anything else -> unsupported
}

[[noreturn]] void unsupported(const std::string &what)
{
    throw std::runtime_error("codegen emitter: unsupported — " + what);
}

// Operator joiners shared by emitScalarExpr (free) and Emitter::emitExpr.
std::string emitBinOpJoin(const std::string &op, const std::string &l,
                          const std::string &r)
{
    if (op == "^" || op == ".^")
        return "std::pow(" + l + ", " + r + ")";
    const char *o = cppScalarBinOp(op);
    if (!o) unsupported("operator " + op);
    return "(" + l + " " + o + " " + r + ")";
}

std::string emitUnOpJoin(const std::string &op, const std::string &a)
{
    if (op == "-") return "(-" + a + ")";
    if (op == "+") return "(+" + a + ")";
    if (op == "~" || op == "!") return "(!" + a + ")";
    unsupported("unary operator " + op);
}


} // namespace detail

using detail::cppIdent;
using detail::emitBinOpJoin;
using detail::emitUnOpJoin;
using detail::unsupported;

std::string emitScalarExpr(const ASTNode &expr)
{
    switch (expr.type) {
    case NodeType::NUMBER_LITERAL:
        return formatDoubleLiteral(expr.numValue);
    case NodeType::IMAG_LITERAL:
        return "std::complex<double>(0.0, " + formatDoubleLiteral(expr.numValue) + ")";
    case NodeType::BOOL_LITERAL:
        return expr.boolValue ? "true" : "false";
    case NodeType::IDENTIFIER:
        return cppIdent(expr.strValue);
    case NodeType::BINARY_OP: {
        if (expr.children.size() != 2) unsupported("binary op arity");
        return emitBinOpJoin(expr.strValue, emitScalarExpr(*expr.children[0]),
                             emitScalarExpr(*expr.children[1]));
    }
    case NodeType::UNARY_OP: {
        if (expr.children.size() != 1) unsupported("unary op arity");
        return emitUnOpJoin(expr.strValue, emitScalarExpr(*expr.children[0]));
    }
    default:
        unsupported("node kind (calls / indexing / matrices) in emitScalarExpr");
    }
}

// ── declaration-type prepass (3b) ─────────────────────────────────────
DeclTypeMap computeDeclTypes(const ASTNode &body, const TypeEnv &entryEnv,
                             const TransferRegistry &reg, const ClassRegistry *classes)
{
    // One inference pass records, at every definition site (including
    // loop-body temporaries and loop variables), the type the variable is
    // assigned THERE. That program-point type — not the post-loop env,
    // where a loop temporary is conservatively Dynamic ("maybe undefined
    // after 0 iterations") — is the right basis for one C++ declaration.
    DeclTypeMap dt;
    for (const auto &[name, av] : entryEnv.entries())  // seed parameters
        dt[name] = av.type;

    TypeEnv env = entryEnv;
    inferStmt(body, env, reg, &dt, classes);
    return dt;
}

namespace detail {

// A 2-D matrix type (KnownDims with both dims > 1): indexed A(i,j),
// column-major. A KnownDims(1,n)/(n,1) is treated as a 1-D vector.
bool is2DMatrixType(const InferredType &t)
{
    return t.isConcrete() && t.shape.kind == ShapeKind::KnownDims
           && t.shape.rows > 1 && t.shape.cols > 1
           && isBufferArray(AbstractValue{t, ConstVal::unknown()});
}

bool isUnboxableScalarType(const InferredType &t)
{
    return t.isConcrete() && t.shape.isScalar() && t.isUnboxableScalar();
}

// Row/Col orientation of a 1-D buffer from its compile-time shape (Unknown when
// the shape doesn't pin it down — then size(vec,dim) is not folded).
VecOrient orientOf(const InferredType &t)
{
    switch (t.shape.kind) {
    case ShapeKind::RowVector: return VecOrient::Row;
    case ShapeKind::ColVector: return VecOrient::Col;
    case ShapeKind::KnownDims:
        if (t.shape.rows == 1) return VecOrient::Row;
        if (t.shape.cols == 1) return VecOrient::Col;
        return VecOrient::Unknown;
    default: return VecOrient::Unknown;
    }
}

// The flattened field-local name for a plain-struct field chain (field-flattening).
// `s.a.b` (a FIELD_ACCESS) -> "_numkit_fld_s_a_b"; "" if not rooted at a plain identifier
// -> not a flattenable struct field. A single-level `s.a` gives "_numkit_fld_s_a"
// (unchanged). MUST match inference.cpp's copy.
std::string structFieldLocal(const ASTNode &fa)
{
    std::string    suffix;
    const ASTNode *n = &fa;
    // Walk FIELD_ACCESS (s.a) AND DYNAMIC_FIELD_ACCESS with a CONSTANT string name (s.('a')
    // folds to the same field-local; a runtime name s.(var) -> "" = not flattenable). G2.6.
    for (;;) {
        std::string field;
        if (n->type == NodeType::FIELD_ACCESS) {
            field = n->strValue;
        } else if (n->type == NodeType::DYNAMIC_FIELD_ACCESS && n->children.size() == 2
                   && n->children[1] && n->children[1]->type == NodeType::STRING_LITERAL) {
            field = n->children[1]->strValue;  // s.('a') : constant field name
        } else {
            break;
        }
        if (n->children.empty() || !n->children[0]) return "";
        suffix = "_" + field + suffix;
        n      = n->children[0].get();
    }
    if (n->type != NodeType::IDENTIFIER) return "";
    return "_numkit_fld_" + n->strValue + suffix;
}

// Extract the root struct variable and ordered field chain for a flattenable field
// access `s.a.b` (rootOut = "s", chainOut = {"a", "b"}).
bool extractStructFieldChain(const ASTNode &fa, std::string &rootOut,
                             std::vector<std::string> &chainOut)
{
    chainOut.clear();
    const ASTNode *n = &fa;
    for (;;) {
        std::string field;
        if (n->type == NodeType::FIELD_ACCESS) {
            field = n->strValue;
        } else if (n->type == NodeType::DYNAMIC_FIELD_ACCESS && n->children.size() == 2
                   && n->children[1] && n->children[1]->type == NodeType::STRING_LITERAL) {
            field = n->children[1]->strValue;
        } else {
            break;
        }
        if (n->children.empty() || !n->children[0]) return false;
        chainOut.push_back(field);
        n = n->children[0].get();
    }
    if (n->type != NodeType::IDENTIFIER || chainOut.empty()) return false;
    std::reverse(chainOut.begin(), chainOut.end());
    rootOut = n->strValue;
    return true;
}

void setNestedStructField(StructLayout &layout, const std::vector<std::string> &chain,
                          std::size_t idx, const InferredType &leafType)
{
    const std::string &fname = chain[idx];
    if (idx + 1 == chain.size()) {
        for (auto &fp : layout.fields) {
            if (fp.first == fname) {
                fp.second = leafType;
                return;
            }
        }
        layout.fields.emplace_back(fname, leafType);
        return;
    }
    for (auto &fp : layout.fields) {
        if (fp.first == fname) {
            auto sub = std::make_shared<StructLayout>();
            if (fp.second.isStruct() && fp.second.structLayout)
                sub->fields = fp.second.structLayout->fields;
            setNestedStructField(*sub, chain, idx + 1, leafType);
            fp.second = InferredType::structOf(std::move(sub));
            return;
        }
    }
    auto sub = std::make_shared<StructLayout>();
    setNestedStructField(*sub, chain, idx + 1, leafType);
    layout.fields.emplace_back(fname, InferredType::structOf(std::move(sub)));
}

void syncIntermediateStructLocals(const std::string &prefix, const StructLayout &layout,
                                  TypeEnv &env)
{
    for (const auto &f : layout.fields) {
        if (f.second.isStruct() && f.second.structLayout) {
            const std::string fld = prefix + "_" + f.first;
            env.set(fld, {f.second, ConstVal::unknown()});
            syncIntermediateStructLocals(fld, *f.second.structLayout, env);
        }
    }
}

// A concrete numeric/logical buffer of non-scalar shape (raw-buffer array).
bool isBufferArrayType(const InferredType &t)
{
    if (!t.isConcrete() || t.shape.isScalar()) return false;
    return isBufferArray(AbstractValue{t, ConstVal::unknown()});
}

// A numeric (double/complex) buffer returnable BY VALUE across an interproc call
// as a flat std::vector (column-major): 1-D vectors, 2-D KnownDims matrices, and
// fully-known N-D arrays. The data is self-describing (.size()) and the dims are
// compile-time-known on BOTH sides (the caller monomorphises the callee to the
// same return type), so no runtime dims need to travel with the buffer. A
// runtime-dim N-D result (an `nd` entry of 0) is excluded — its dims are not
// compile-time-known, so the caller could not hoist a matching local.
bool isByValueReturnArrayType(const InferredType &t)
{
    if (!isBufferArrayType(t)) return false;
    if (t.dtype != ValueType::DOUBLE && t.dtype != ValueType::COMPLEX
        && t.dtype != ValueType::LOGICAL
        && t.dtype != ValueType::CHAR)  // CHAR -> uint16 buffer (string-building helpers)
        return false;
    if (t.shape.isNDims())
        for (std::size_t d : t.shape.nd)
            if (d == 0) return false;
    return true;
}


// The C++ variable type for an object: a value class is the plain struct
// (value semantics); a handle class is numkit_rt::handle<T> (shared reference).
std::string cppObjectType(int classId, const ClassRegistry *classes)
{
    const ClassInfo *ci = classes ? classes->byId(classId) : nullptr;
    if (!ci) unsupported("object type with no class registry / unknown classId");
    const std::string clsName = cppGlobalIdent(ci->name);
    return ci->isHandle ? "numkit_rt::handle<" + clsName + ">" : clsName;
}

// numel(A) / length(A) of a tracked array variable A?
bool isNumelOfArray(const ASTNode &e,
                    const std::unordered_map<std::string, ArrayInfo> &arrays,
                    std::string &arrayOut)
{
    if (e.type != NodeType::CALL || e.children.size() != 2) return false;
    if (e.children[0]->type != NodeType::IDENTIFIER) return false;
    const std::string &fn = e.children[0]->strValue;
    if (fn != "numel" && fn != "length") return false;
    if (e.children[1]->type != NodeType::IDENTIFIER) return false;
    if (!arrays.count(e.children[1]->strValue)) return false;
    arrayOut = e.children[1]->strValue;
    return true;
}

// numel-equality fact store: arrayLenVar[A] = v means numel(A) == value of
// scalar variable v. The store is maintained FLOW-SENSITIVELY by
// walkForFacts (forward over the top-level statement sequence) so a fact
// is only ever used where it provably holds.

// A change to variable `v` invalidates (a) any fact about an array named
// `v` (its contents/size changed) and (b) any fact whose length TOKEN is
// `v` (the token's value changed, so numel(A)==v no longer pinned).
void invalidateLen(std::unordered_map<std::string, std::string> &lenVar,
                   const std::string &v)
{
    lenVar.erase(v);
    for (auto it = lenVar.begin(); it != lenVar.end();)
        it = (it->second == v) ? lenVar.erase(it) : std::next(it);
}

// All variables assigned anywhere within `node` (identifier / indexed-base
// / for-var / multi-assign). Used to conservatively invalidate facts a
// compound statement (if/while/loop body) may have changed.
void collectAssignedVars(const ASTNode &node, std::set<std::string> &out)
{
    if (node.type == NodeType::ASSIGN && !node.children.empty()) {
        const ASTNode &lhs = *node.children[0];
        if (lhs.type == NodeType::IDENTIFIER)
            out.insert(lhs.strValue);
        else if (!lhs.children.empty() && lhs.children[0]->type == NodeType::IDENTIFIER)
            out.insert(lhs.children[0]->strValue);  // indexed / field base
    }
    if (node.type == NodeType::FOR_STMT) out.insert(node.strValue);
    if (node.type == NodeType::MULTI_ASSIGN)
        for (const auto &rn : node.returnNames)
            if (!rn.empty()) out.insert(rn);
    for (const auto &c : node.children) if (c) collectAssignedVars(*c, out);
    for (const auto &br : node.branches) {
        if (br.first)  collectAssignedVars(*br.first, out);
        if (br.second) collectAssignedVars(*br.second, out);
    }
    if (node.elseBranch) collectAssignedVars(*node.elseBranch, out);
}

// Establish a fact from a single assignment `lhs = rhs` (lhs already
// invalidated by the caller): `lhs = numel(A)` -> numel(A)==lhs;
// `lhs = zeros(1,v)` / `zeros(v,1)` -> numel(lhs)==v.
void establishLenFact(const std::string &lhs, const ASTNode &rhs,
                      const std::unordered_map<std::string, ArrayInfo> &arrays,
                      std::unordered_map<std::string, std::string> &lenVar)
{
    std::string arr;
    if (isNumelOfArray(rhs, arrays, arr)) { lenVar[arr] = lhs; return; }
    if (rhs.type == NodeType::CALL && rhs.children.size() == 3
        && rhs.children[0]->type == NodeType::IDENTIFIER
        && (rhs.children[0]->strValue == "zeros" || rhs.children[0]->strValue == "ones")
        && arrays.count(lhs)) {
        const ASTNode &d1 = *rhs.children[1], &d2 = *rhs.children[2];
        if (d1.type == NodeType::NUMBER_LITERAL && d1.numValue == 1.0
            && d2.type == NodeType::IDENTIFIER)
            lenVar[lhs] = d2.strValue;
        else if (d2.type == NodeType::NUMBER_LITERAL && d2.numValue == 1.0
                 && d1.type == NodeType::IDENTIFIER)
            lenVar[lhs] = d1.strValue;
    }
}

// Every use of `k` in `node` is the sole index of a buffer-array access
// `A(k)` (collected into idxArrays). Any other occurrence of `k`
// (arithmetic, A(k,j), non-array call) returns false (not clean).
bool collectCleanIndexUses(const ASTNode &node, const std::string &k,
                           const std::unordered_map<std::string, ArrayInfo> &arrays,
                           std::set<std::string> &idxArrays)
{
    if (node.type == NodeType::CALL && node.children.size() == 2
        && node.children[0]->type == NodeType::IDENTIFIER
        && arrays.count(node.children[0]->strValue)
        && node.children[1]->type == NodeType::IDENTIFIER
        && node.children[1]->strValue == k) {
        idxArrays.insert(node.children[0]->strValue);
        return true;  // clean index — do not descend into the index arg
    }
    if (node.type == NodeType::IDENTIFIER && node.strValue == k)
        return false;  // a bare / non-index use of k
    for (const auto &c : node.children)
        if (c && !collectCleanIndexUses(*c, k, arrays, idxArrays)) return false;
    for (const auto &br : node.branches) {
        if (br.first && !collectCleanIndexUses(*br.first, k, arrays, idxArrays)) return false;
        if (br.second && !collectCleanIndexUses(*br.second, k, arrays, idxArrays)) return false;
    }
    if (node.elseBranch && !collectCleanIndexUses(*node.elseBranch, k, arrays, idxArrays))
        return false;
    return true;
}

int countIdentUses(const ASTNode &node, const std::string &name)
{
    int n = (node.type == NodeType::IDENTIFIER && node.strValue == name) ? 1 : 0;
    for (const auto &c : node.children) if (c) n += countIdentUses(*c, name);
    for (const auto &br : node.branches) {
        if (br.first)  n += countIdentUses(*br.first, name);
        if (br.second) n += countIdentUses(*br.second, name);
    }
    if (node.elseBranch) n += countIdentUses(*node.elseBranch, name);
    return n;
}

// Promote one top-level `for k = 1:H` loop iff (gate, all must hold with
// facts valid AT THE LOOP): every use of k is the sole index of a buffer
// array whose numel is provably H; and k is used nowhere outside the loop.
bool tryPromoteLoop(const ASTNode &f, const ASTNode &funcBody,
                    const std::unordered_map<std::string, ArrayInfo> &arrays,
                    const std::unordered_map<std::string, std::string> &lenVar,
                    OptFacts &facts)
{
    if (f.children.size() != 2) return false;
    const ASTNode     &range = *f.children[0];
    const std::string &k     = f.strValue;
    if (!(range.type == NodeType::COLON_EXPR && range.children.size() == 2
          && range.children[0]->type == NodeType::NUMBER_LITERAL
          && range.children[0]->numValue == 1.0
          && range.children[1]->type == NodeType::IDENTIFIER))
        return false;
    const std::string     boundVar = range.children[1]->strValue;
    const ASTNode        &lbody    = *f.children[1];
    std::set<std::string> idxArrays;
    bool safe = collectCleanIndexUses(lbody, k, arrays, idxArrays) && !idxArrays.empty();
    for (const auto &A : idxArrays) {
        auto it = lenVar.find(A);
        if (it == lenVar.end() || it->second != boundVar) { safe = false; break; }
    }
    if (!safe || countIdentUses(funcBody, k) != countIdentUses(lbody, k)) return false;

    std::string bound;  // prefer a param array's companion (== numel exactly)
    for (const auto &A : idxArrays)
        if (!arrays.at(A).isOutput) { bound = arrays.at(A).lenVar; break; }
    if (bound.empty()) bound = arrays.at(*idxArrays.begin()).lenVar;
    facts.promotedLoops.insert(&f);
    facts.loopBound[&f] = bound;
    return true;
}

// Flow-sensitive forward walk over a statement sequence, maintaining the
// numel-fact store (`lenVar`) with invalidation. A top-level `for` is
// promoted against the facts valid at its position; any compound statement
// conservatively invalidates every variable it could assign. (Only
// top-level loops are promoted in this MVP; a loop nested in a branch /
// another loop stays the always-correct checked form.)
void walkForFacts(const ASTNode &block, const ASTNode &funcBody,
                  const std::unordered_map<std::string, ArrayInfo> &arrays,
                  std::unordered_map<std::string, std::string> &lenVar, OptFacts &facts)
{
    for (const auto &sp : block.children) {
        if (!sp) continue;
        const ASTNode &s = *sp;
        if (s.type == NodeType::ASSIGN && !s.children.empty()) {
            const ASTNode &lhs = *s.children[0];
            if (lhs.type == NodeType::IDENTIFIER) {
                invalidateLen(lenVar, lhs.strValue);
                if (s.children.size() == 2)
                    establishLenFact(lhs.strValue, *s.children[1], arrays, lenVar);
            } else if (!lhs.children.empty()
                       && lhs.children[0]->type == NodeType::IDENTIFIER) {
                invalidateLen(lenVar, lhs.children[0]->strValue);  // indexed base changed
            }
        } else if (s.type == NodeType::FOR_STMT) {
            tryPromoteLoop(s, funcBody, arrays, lenVar, facts);
            std::set<std::string> assigned;  // body may rebind facts -> invalidate
            collectAssignedVars(s, assigned);
            for (const auto &v : assigned) invalidateLen(lenVar, v);
        } else {
            // if / while / switch / try / multi-assign / expr — conservative:
            // every variable it might assign becomes unknown.
            std::set<std::string> assigned;
            collectAssignedVars(s, assigned);
            for (const auto &v : assigned) invalidateLen(lenVar, v);
        }
    }
}

OptFacts analyzeOptimizations(const ASTNode &body,
                              const std::unordered_map<std::string, ArrayInfo> &arrays)
{
    std::unordered_map<std::string, std::string> lenVar;
    OptFacts                                     facts;
    walkForFacts(body, body, arrays, lenVar, facts);
    return facts;
}

std::string typeCode(const InferredType &t)
{
    if (!t.isConcrete()) return "X";
    if (t.dtype == ValueType::OBJECT) return "o" + std::to_string(t.classId);
    if (t.dtype == ValueType::STRUCT) {
        // Encode the layout so distinct structs (field NAMES + types) get distinct
        // specializations -- the callee accesses fields by name, so two structs with the
        // same field TYPES but different names are different callees. Each field: "F" +
        // escaped-name length + escaped name + the field's own typeCode. escapeBase keeps
        // it __-free (every '_' becomes "_0").
        std::string sc = "s" + std::to_string(t.structLayout ? t.structLayout->fields.size() : 0);
        if (t.structLayout)
            for (const auto &f : t.structLayout->fields) {
                const std::string en = escapeBase(f.first);
                sc += "F" + std::to_string(en.size()) + en + typeCode(f.second);
            }
        return sc;
    }
    std::string c;
    switch (t.dtype) {
    case ValueType::DOUBLE:  c = "d";   break;
    case ValueType::SINGLE:  c = "f";   break;
    case ValueType::COMPLEX: c = "c";   break;
    case ValueType::LOGICAL: c = "b";   break;
    case ValueType::INT8:    c = "i8";  break;
    case ValueType::INT16:   c = "i16"; break;
    case ValueType::INT32:   c = "i32"; break;
    case ValueType::INT64:   c = "i64"; break;
    case ValueType::UINT8:   c = "u8";  break;
    case ValueType::UINT16:  c = "u16"; break;
    case ValueType::UINT32:  c = "u32"; break;
    case ValueType::UINT64:  c = "u64"; break;
    default:                 c = "o";   break;
    }
    switch (t.shape.kind) {
    case ShapeKind::Scalar:    break;
    case ShapeKind::RowVector: c += "r"; break;
    case ShapeKind::ColVector: c += "k"; break;
    case ShapeKind::Vector1D:  c += "v"; break;
    case ShapeKind::KnownDims:
        c += "m" + std::to_string(t.shape.rows) + "x" + std::to_string(t.shape.cols);
        break;
    case ShapeKind::NDims:
        c += "n";  // ranked: rank + each dim (0 = unknown); 'x'-joined, NO '_'
        for (std::size_t d : t.shape.nd) c += std::to_string(d) + "x";  // '_'-free: mangle stays __-free
        break;
    case ShapeKind::Unknown:   c += "u"; break;
    }
    return c;
}

// Escape a base name so the mangle is injective AND free of "__" (which the
// C++ standard reserves to the implementation, [lex.name]): each '_' in the
// base becomes "_0", and argument segments are introduced by the "_1"
// separator — which escaping guarantees the base can never itself produce.
// typeCode is '_'-free, so every '_' in a mangled symbol is immediately
// followed by a digit (0/1): no "__" can ever form. The symbol also starts
// with the base's (letter) first char, so it is a conforming global identifier
// (no leading underscore). Distinct (base, args) -> distinct symbol.
std::string escapeBase(const std::string &base)
{
    std::string e;
    for (char ch : base) {
        e += ch;
        if (ch == '_') e += '0';
    }
    return e;
}

// Escape a string's bytes for a C++ string literal. Returns false (via `ok`) if a byte is
// not printable ASCII (a control byte would need octal/hex escaping we don't emit) -- the
// caller refuses such a literal. Used to box a bridged CHAR ARG (e.g. a sprintf format).
std::string cEscape(const std::string &s, bool &ok)
{
    ok = true;
    std::string out;
    for (unsigned char c : s) {
        if (c < 32 || c > 126) { ok = false; return {}; }  // non-printable -> refuse
        if (c == '\\' || c == '"') out += '\\';
        out += static_cast<char>(c);
    }
    return out;
}

std::string mangle(const std::string &base, const std::vector<InferredType> &args)
{
    std::string m = escapeBase(base);
    if (args.empty()) return m + "_1v";
    for (const auto &a : args) m += "_1" + typeCode(a);
    return m;
}

// A reserved-companion variable name (`_numkit_<escaped-base><suffix>`) for an
// array's length / dim sizes (suffix = "_len" / "_rows" / "_cols" / "_d0"...).
// The `_numkit_` prefix lives in the underscore namespace a MATLAB identifier can
// never enter, so it cannot collide with a user variable — yet it is a
// CONFORMING block-scope name: a leading underscore is reserved only for the
// GLOBAL namespace ([lex.name]), and companions are always params/locals. The
// base is escaped so a name like `x_` cannot form a reserved "__".
std::string companion(const std::string &base, const std::string &suffix)
{
    const std::string stem = (base.rfind("_numkit_", 0) == 0) ? base.substr(8) : base;
    return "_numkit_" + escapeBase(stem) + suffix;
}

std::vector<ArgInfo> toArgInfos(const std::vector<InferredType> &types)
{
    std::vector<ArgInfo> out;
    out.reserve(types.size());
    for (const auto &t : types) out.push_back(ArgInfo(t, ConstVal::unknown()));
    return out;
}

} // namespace detail
} // namespace numkit::codegen
