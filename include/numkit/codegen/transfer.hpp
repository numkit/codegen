// codegen/include/numkit/codegen/transfer.hpp
//
// Type transfer functions — the per-builtin rules the inference dataflow
// applies at call nodes. A transfer function maps the abstract argument
// info at a call site to the inferred type of the builtin's result.
//
// These live in src/codegen (NOT next to the builtin implementations):
// codegen is an L2 pass above the toolboxes, so a toolbox cannot depend
// on the lattice types. The registry describes the builtins from the
// outside and is validated against them by the differential test
// (tests/transfer_diff_test.cpp) — drift is caught automatically rather
// than prevented by co-location. See src/codegen/DESIGN.md §4.
//
// Organised by family, one .cpp per family under src/transfer/ (mirrors
// the src/math, src/lang category split): constructors, elementwise,
// reductions, shape, cast, bespoke.

#pragma once

#include <numkit/codegen/type_lattice.hpp>

#include <keel/array.h>
#include <keel/core.h>
#include <keel/function.h>
#include <keel/hash_map.h>
#include <keel/ref.h>
#include <keel/string.h>

namespace numkit::codegen {

// Abstract description of one argument at a call site: its inferred type
// plus its known-constant facet. The constant drives shape-from-value
// (the `n` in linspace(_,_,n) / zeros(n)); see DESIGN.md §4.
struct ArgInfo {
    InferredType type;
    ConstVal     constant;

    ArgInfo() noexcept = default;
    ArgInfo(InferredType t, ConstVal c) noexcept : type(keel::move(t)), constant(c) {}
    ArgInfo(const ArgInfo &o) : type(o.type.clone()), constant(o.constant) {}
    ArgInfo &operator=(const ArgInfo &o)
    {
        if (this != &o) {
            type     = o.type.clone();
            constant = o.constant;
        }
        return *this;
    }
    ArgInfo(ArgInfo &&) noexcept = default;
    ArgInfo &operator=(ArgInfo &&) noexcept = default;
    ~ArgInfo() noexcept = default;

    ArgInfo clone() const { return *this; }

    static ArgInfo of(InferredType t, ConstVal c = ConstVal::unknown())
    {
        return ArgInfo(keel::move(t), c);
    }
    // Convenience: a scalar of dtype `dt` with a known constant value `v`
    // (the common literal-argument case, e.g. the `64` in linspace).
    static ArgInfo scalarConst(ValueType dt, keel::f64 v)
    {
        return ArgInfo(InferredType::scalar(dt), ConstVal::known(v));
    }
};

// A transfer function: abstract args -> inferred type of the (first)
// result.
using TransferFn = keel::Function<InferredType(keel::ArrayRef<ArgInfo>)>;

// A multi-output transfer: abstract args -> all result types (one per
// output), for `[a,b] = f(...)`. Registered alongside the single-output
// TransferFn for functions/methods that can produce several outputs.
using MultiTransferFn = keel::Function<keel::Array<InferredType>(keel::ArrayRef<ArgInfo>)>;

class FunctionTable;  // monomorphize.hpp — the co-compiled user functions

// builtin name -> transfer function. A name with no entry yields Dynamic
// (the sound fallback: unknown builtin -> boxed Value).
class TransferRegistry {
public:
    TransferRegistry() noexcept = default;
    TransferRegistry(TransferRegistry &&) noexcept = default;
    TransferRegistry &operator=(TransferRegistry &&) noexcept = default;
    ~TransferRegistry() noexcept = default;

    void add(keel::StringRef name, TransferFn fn);
    void addMulti(keel::StringRef name, MultiTransferFn fn);

    bool        has(keel::StringRef name) const;
    keel::usize size() const;

    // The co-compiled user-function table (set by registerUserFunctions), so the
    // inference can tell a USER function from a builtin -- e.g. to type a bare
    // `c = f(x)` on a MULTI-output user fn as f's FIRST output (MATLAB single-LHS
    // semantics), a projection a builtin like `size` does NOT share. Borrowed
    // (same lifetime as the registry's user transfers); null until set.
    void                 setUserFunctions(const FunctionTable *t) { userFuncs_ = t; }
    const FunctionTable *userFunctions() const { return userFuncs_; }

    // Result type for a call to `name` with `args`; Dynamic if `name`
    // has no registered transfer.
    InferredType apply(keel::StringRef name, keel::ArrayRef<ArgInfo> args) const;

    // All result types for a multi-output call to `name`; empty if `name`
    // has no registered multi-output transfer.
    keel::Array<InferredType> applyMulti(keel::StringRef name,
                                         keel::ArrayRef<ArgInfo> args) const;

private:
    keel::HashMap<keel::String, TransferFn>      table_;
    keel::HashMap<keel::String, MultiTransferFn> multiTable_;
    const FunctionTable                         *userFuncs_ = nullptr;  // borrowed
};

// Populate a registry with every standard transfer (dispatches to the
// per-family registrars below).
void registerStandardTransfers(TransferRegistry &reg);

// ── Per-family registrars (one .cpp each under src/transfer/) ─────────
void registerConstructorTransfers(TransferRegistry &reg);
void registerElementwiseTransfers(TransferRegistry &reg);
void registerShapeTransfers(TransferRegistry &reg);
// future: registerReductionTransfers / registerCastTransfers / ...

} // namespace numkit::codegen
