// codegen/src/monomorphize.cpp — see monomorphize.hpp.

#include <numkit/codegen/monomorphize.hpp>

#include <numkit/codegen/classinfo.hpp>

#include <keel/format.h>
#include <keel/hash_map.h>
#include <keel/hash_set.h>
#include <keel/smart_ptr.h>

namespace numkit::codegen {

namespace {

// Does `n` (a function body) contain a direct call to `name`? Used to detect
// DIRECT self-recursion so the return-type fixpoint runs only where it is needed
// (a non-recursive function keeps the single-pass inference -- no 2x cost, and no
// exponential blow-up down a deep non-recursive call chain). Mutual recursion is
// NOT detected here (each fn sees no direct self-call) -> it takes the sound
// Dynamic break, as before.
bool nodeCallsName(const ASTNode &n, keel::StringRef name)
{
    if (n.type == NodeType::CALL && !n.children.empty() && n.children[0]
        && n.children[0]->type == NodeType::IDENTIFIER && n.children[0]->strValue == name)
        return true;
    for (const auto &c : n.children)
        if (c && nodeCallsName(*c, name)) return true;
    for (const auto &br : n.branches) {
        if (br.cond && nodeCallsName(*br.cond, name)) return true;
        if (br.body && nodeCallsName(*br.body, name)) return true;
    }
    if (n.elseBranch && nodeCallsName(*n.elseBranch, name)) return true;
    return false;
}

// The fixpoint key: function name + each argument's TYPE signature (str() is a
// precise encoding for the eligible types -- numeric scalars/arrays/complex). A
// same-signature self-call shares the estimate; a different-signature self-call
// (polymorphic recursion) does not, and falls to the Dynamic break.
keel::String fixpointKey(keel::StringRef name, keel::ArrayRef<ArgInfo> args)
{
    keel::String s = keel::String::from(name);
    for (const ArgInfo &a : args) {
        s.push_back('#');
        s.append(a.type.str().as_ref());
    }
    return s;
}

// An argument is fixpoint-eligible when its type has a PRECISE str() signature.
// STRUCT (str = field names only, not field types) and OBJECT are excluded -- a
// signature collision there could share an estimate unsoundly; such recursion is
// rare, so it takes the sound Dynamic break instead.
bool fixpointEligible(keel::ArrayRef<ArgInfo> args)
{
    for (const ArgInfo &a : args)
        if (a.type.isStruct() || a.type.isObject()) return false;
    return true;
}

constexpr int kFixpointCap = 8;  // lattice height is tiny; 8 is far above need

}  // namespace

void FunctionTable::add(const ASTNode &funcDef)
{
    if (funcDef.type == NodeType::FUNCTION_DEF && !funcDef.strValue.empty())
        defs_.put(funcDef.strValue.clone(), &funcDef);
}

const ASTNode *FunctionTable::find(keel::StringRef name) const
{
    const ASTNode *const *p = defs_.get(name);
    return p ? *p : nullptr;
}

void collectFunctions(const ASTNode &root, FunctionTable &table)
{
    // A classdef's FUNCTION_DEF children are METHODS, not free functions —
    // they are handled by registerClassMethods. Do not descend into a
    // classdef, or methods would leak into the free-function table.
    if (root.type == NodeType::CLASSDEF_DEF) return;
    if (root.type == NodeType::FUNCTION_DEF) table.add(root);
    for (const auto &c : root.children)
        if (c) collectFunctions(*c, table);
    for (const auto &br : root.branches) {
        if (br.cond) collectFunctions(*br.cond, table);
        if (br.body) collectFunctions(*br.body, table);
    }
    if (root.elseBranch) collectFunctions(*root.elseBranch, table);
}

namespace {

void seedStructParamFields(keel::StringRef prefix, const StructLayout &layout, TypeEnv &env)
{
    for (const auto &f : layout.fields) {
        const keel::String fld = keel::format("{}_{}", prefix, f.first);
        env.set(fld, {f.second, ConstVal::unknown()});
        if (f.second.isStruct() && f.second.structLayout)
            seedStructParamFields(fld, *f.second.structLayout, env);
    }
}

} // namespace

keel::Array<InferredType> inferFunctionReturns(const ASTNode &funcDef,
                                               keel::ArrayRef<ArgInfo> args,
                                               const TransferRegistry &reg,
                                               const ClassRegistry *classes)
{
    if (funcDef.type != NodeType::FUNCTION_DEF || funcDef.children.empty())
        return {};
    if (args.size() != funcDef.paramNames.size())   // exact arity (MVP)
        return {};

    // Seed the entry env: each parameter takes the matching argument's type
    // and constant facet (the constant flows into shape-from-value).
    TypeEnv env;
    for (keel::usize i = 0; i < args.size(); ++i) {
        env.set(funcDef.paramNames[i], {args[i].type, args[i].constant});
        // A STRUCT param also seeds its field-locals (_numkit_fld_<p>_<f>...) recursively
        // so the body's `s.f` and nested `s.a.b` reads resolve them -- mirrors
        // emitOneFunction's struct-param explosion (G2.3).
        if (args[i].type.isStruct() && args[i].type.structLayout) {
            const keel::String pref = keel::format("_numkit_fld_{}", funcDef.paramNames[i]);
            seedStructParamFields(pref, *args[i].type.structLayout, env);
        }
    }

    inferStmt(*funcDef.children[0], env, reg, nullptr, classes);

    keel::Array<InferredType> outs;
    outs.reserve(funcDef.returnNames.size());
    for (const auto &rn : funcDef.returnNames)
        outs.push_back(rn.empty() ? InferredType::dynamic() : env.get(rn).type);
    return outs;
}

InferredType inferFunctionReturn(const ASTNode &funcDef,
                                 keel::ArrayRef<ArgInfo> args,
                                 const TransferRegistry &reg, const ClassRegistry *classes)
{
    const keel::Array<InferredType> outs = inferFunctionReturns(funcDef, args, reg, classes);
    return outs.size() == 1 ? outs[0] : InferredType::dynamic();
}

void registerClassConstructors(TransferRegistry &reg, const ClassRegistry &classes)
{
    for (keel::usize i = 0; i < classes.size(); ++i) {
        const ClassInfo *ci = classes.byId(static_cast<int>(i));
        if (!ci) continue;
        const int id = ci->id;
        reg.add(ci->name,
                [id](keel::ArrayRef<ArgInfo>) { return InferredType::object(id); });
    }
}

void registerClassMethods(TransferRegistry &reg, const ClassRegistry &classes)
{
    auto inProgress = keel::make_arc<keel::HashSet<keel::String>>();
    for (keel::usize i = 0; i < classes.size(); ++i) {
        const ClassInfo *ci = classes.byId(static_cast<int>(i));
        if (!ci) continue;
        for (auto [mname, mdef] : ci->methods) {
            const keel::String key = keel::format("{}::{}", ci->name, mname);
            const ASTNode     *md  = mdef;
            reg.add(key, [md, key = key.clone(), &reg, &classes, inProgress](keel::ArrayRef<ArgInfo> args) {
                if (!inProgress->insert(key.clone()))
                    return InferredType::dynamic();          // recursion -> sound break
                const InferredType r = inferFunctionReturn(*md, args, reg, &classes);
                inProgress->remove(key);
                return r;
            });
            reg.addMulti(key, [md, key = key.clone(), &reg, &classes, inProgress](keel::ArrayRef<ArgInfo> args)
                                  -> keel::Array<InferredType> {
                if (!inProgress->insert(key.clone())) return {};
                keel::Array<InferredType> r = inferFunctionReturns(*md, args, reg, &classes);
                inProgress->remove(key);
                return r;
            });
        }
    }
}

void registerUserFunctions(TransferRegistry &reg, const FunctionTable &table)
{
    // Expose the table so the inference can tell a user fn from a builtin (e.g.
    // the bare `c = f(x)` multi-output first-output projection). Borrowed -- the
    // table must outlive the registry (it already backs the transfers below).
    reg.setUserFunctions(&table);

    // Shared across all user-function transfers: the set of functions
    // currently being inferred, so a recursive (re-entrant) call returns
    // Dynamic instead of looping forever.
    auto inProgress = keel::make_arc<keel::HashSet<keel::String>>();
    // The recursion return-type FIXPOINT state (Rec.2), keyed by (fn + arg-type
    // signature). A same-signature self-call reads the current estimate (seeded
    // Bottom); the outer entry iterates the body inference to a fixpoint.
    auto estimates = keel::make_arc<keel::HashMap<keel::String, InferredType>>();

    for (auto [name, def] : table.entries()) {
        const keel::String key = name.clone();   // own copy per transfer
        const ASTNode     *fn  = def;
        // DIRECT self-recursion, computed once: only such a function runs the
        // fixpoint (others keep the cheap single pass).
        const bool selfRec =
            !fn->children.empty() && fn->children[0] && nodeCallsName(*fn->children[0], key);
        reg.add(key,
                [fn, key = key.clone(), selfRec, &reg, inProgress, estimates](keel::ArrayRef<ArgInfo> args) {
            const bool         eligible = selfRec && fixpointEligible(args);
            const keel::String ek       = eligible ? fixpointKey(key, args) : keel::String();
            if (eligible) {
                if (const InferredType *est = estimates->get(ek))
                    return *est;  // same-signature self-call: the current fixpoint estimate
            }
            if (!inProgress->insert(key.clone()))        // mutual / polymorphic / ineligible re-entry
                return InferredType::dynamic();          // -> sound break (no infinite monomorphisation)
            InferredType result;
            if (eligible) {
                // Seed Bottom; iterate the body inference (the self-call reads the
                // estimate) until the return type stops changing. The sequence is
                // monotone-ascending (Bottom -> ... -> at most Dynamic), so it
                // converges within the lattice height; the cap -> Dynamic is a sound
                // over-approximation if it somehow does not.
                estimates->put(ek.clone(), InferredType::bottom());
                result = InferredType::dynamic();  // default if not converged
                for (int iter = 0; iter < kFixpointCap; ++iter) {
                    const InferredType next = inferFunctionReturn(*fn, args, reg);
                    const InferredType *curEst = estimates->get(ek);
                    if (curEst && next == *curEst) { result = next; break; }
                    estimates->put(ek.clone(), next);
                }
                estimates->remove(ek);
            } else {
                result = inferFunctionReturn(*fn, args, reg);
            }
            inProgress->remove(key);
            return result;
        });
        reg.addMulti(key, [fn, key = key.clone(), &reg, inProgress](keel::ArrayRef<ArgInfo> args)
                              -> keel::Array<InferredType> {
            if (!inProgress->insert(key.clone())) return {};  // recursion -> empty (targets Dynamic)
            keel::Array<InferredType> r = inferFunctionReturns(*fn, args, reg);
            inProgress->remove(key);
            return r;
        });
    }
}

} // namespace numkit::codegen
