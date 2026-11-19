// codegen/include/numkit/codegen/emitter.hpp
//
// The C++ emitter — turns a typed AST into C++ source text.
//
// Sub-brick 3a: the unboxed-scalar expression core (literals, identifiers,
// scalar arithmetic / comparison / logical / power) -> a plain C++
// expression. Sub-bricks 3b-3f add statements, control flow, builtin
// calls, indexing and whole-function emission, wired together by the
// stateful Emitter (in emitter.cpp) and exposed here via emitFunction().
//
// ABI (DESIGN.md §6, RawBuffer variant): the whole-function emitter
// produces a SELF-CONTAINED translation unit depending only on the C++
// standard library — scalars are unboxed primitives, an array parameter
// `a` becomes `const T* a, std::size_t a_len`, and a single output array
// becomes a trailing caller-allocated out-param `T* ret, std::size_t
// ret_len` (void return). This makes the end-to-end differential gate
// (compile with an external compiler + run + diff) robust without linking
// the nk runtime. The Value-ABI variant (array args/returns as
// numkit::Value) is a thin wrapper around the identical numeric core.
//
// Contract 2 (DESIGN.md §10): the emitter emits a fast unboxed form ONLY
// for a construct it can prove correct; anything outside the supported
// subset throws std::runtime_error (the explicit boundary) rather than
// emitting code that might be wrong.

#pragma once

#include <numkit/codegen/classinfo.hpp>
#include <numkit/codegen/inference.hpp>
#include <numkit/codegen/monomorphize.hpp>
#include <numkit/codegen/transfer.hpp>
#include <numkit/codegen/type_lattice.hpp>

#include <numkit/core/ast.hpp>

#include <keel/hash_map.h>
#include <keel/ref.h>
#include <keel/string.h>

namespace numkit::codegen {

// The C++ type name for an unboxable scalar dtype (double / float /
// std::complex<double> / bool / intN_t). Throws std::runtime_error for a
// dtype with no scalar C++ mapping (CELL/STRUCT/STRING/…).
keel::String cppScalarType(ValueType dtype);

// A round-trip C++ double literal for `v` (shortest form; integers get a
// trailing .0; Inf/NaN map to std::numeric_limits forms).
keel::String formatDoubleLiteral(double v);

// Emit a C++ expression string for an unboxed-scalar AST expression
// (NUMBER/IMAG/BOOL literal, IDENTIFIER, scalar BINARY_OP / UNARY_OP).
// `^`/`.^` lower to std::pow. Throws std::runtime_error on a node kind
// outside this pure-scalar subset (calls, indexing — handled by Emitter).
keel::String emitScalarExpr(const ASTNode &expr);

// Declaration type per variable = the lattice JOIN of every type it is
// ever assigned in `body` (so one hoisted C++ declaration is valid at
// every program point). RHS types are evaluated under the inference
// fixpoint env, making the result a sound over-approximation. `entryEnv`
// seeds parameter types, which are also recorded.
using DeclTypeMap = keel::HashMap<keel::String, InferredType>;
DeclTypeMap computeDeclTypes(const ASTNode &body, const TypeEnv &entryEnv,
                             const TransferRegistry &reg,
                             const ClassRegistry *classes = nullptr);

// Entry-point parameter type annotation (the inference seed for one
// function). The order must match the FUNCTION_DEF's parameter list.
struct ParamSpec {
    keel::String name;
    InferredType type;

    ParamSpec() noexcept = default;
    ParamSpec(keel::StringRef n, InferredType t)
        : name(keel::String::from(n)), type(keel::move(t))
    {}
    ParamSpec(const ParamSpec &o) : name(o.name.clone()), type(o.type.clone()) {}
    ParamSpec &operator=(const ParamSpec &o)
    {
        if (this != &o) {
            name = o.name.clone();
            type = o.type.clone();
        }
        return *this;
    }
    ParamSpec(ParamSpec &&) noexcept = default;
    ParamSpec &operator=(ParamSpec &&) noexcept = default;
    ~ParamSpec() noexcept = default;

    ParamSpec clone() const { return *this; }
};

// A complete emitted function.
struct EmittedFunction {
    keel::String source;     // a self-contained C++ translation unit
    keel::String name;       // the emitted C++ symbol
    keel::String signature;  // the function's declaration (for a header)

    EmittedFunction() noexcept = default;
    EmittedFunction(keel::String src, keel::String n, keel::String sig) noexcept
        : source(keel::move(src)), name(keel::move(n)), signature(keel::move(sig))
    {}
    EmittedFunction(const EmittedFunction &o)
        : source(o.source.clone()), name(o.name.clone()), signature(o.signature.clone())
    {}
    EmittedFunction &operator=(const EmittedFunction &o)
    {
        if (this != &o) {
            source    = o.source.clone();
            name      = o.name.clone();
            signature = o.signature.clone();
        }
        return *this;
    }
    EmittedFunction(EmittedFunction &&) noexcept = default;
    EmittedFunction &operator=(EmittedFunction &&) noexcept = default;
    ~EmittedFunction() noexcept = default;

    EmittedFunction clone() const { return *this; }
};

// Bridged emission (DESIGN.md §6a). OPT-IN: when disabled (the default), a
// call the emitter cannot lower throws (the TU stays stdlib-only,
// self-contained — the no-kludge litmus). When enabled, such a call — IF
// inference proves its result is an unboxed scalar — is emitted as a C-ABI
// call into the nk runtime (numkit_rt::bridge_scalar), and the TU #includes
// `runtimeHeader` (numkit_codegen_rt.h) + links the runtime. Correctness is
// unchanged either way; bridging only widens what compiles.
struct BridgeOptions {
    bool         enabled = false;
    keel::String runtimeHeader;  // include path to numkit_codegen_rt.h (when enabled)

    BridgeOptions() noexcept = default;
    BridgeOptions(bool en, keel::StringRef hdr)
        : enabled(en), runtimeHeader(keel::String::from(hdr))
    {}
    BridgeOptions(const BridgeOptions &o)
        : enabled(o.enabled), runtimeHeader(o.runtimeHeader.clone())
    {}
    BridgeOptions &operator=(const BridgeOptions &o)
    {
        if (this != &o) {
            enabled       = o.enabled;
            runtimeHeader = o.runtimeHeader.clone();
        }
        return *this;
    }
    BridgeOptions(BridgeOptions &&) noexcept = default;
    BridgeOptions &operator=(BridgeOptions &&) noexcept = default;
    ~BridgeOptions() noexcept = default;

    BridgeOptions clone() const { return *this; }
};

// Ops-kernel lowering: ops owns the compute kernels; codegen takes a kernel
// from ops (numkit::ops::, the public facade in <numkit/ops/kernels.hpp>)
// instead of emitting a naive loop. OPT-IN: when disabled (the default), heavy
// array ops lower to a self-contained inline loop (the TU stays stdlib-only —
// existing callers unchanged). When enabled, an op with a matching ops kernel
// (matmul, …) is emitted as a numkit::ops:: call, the TU #includes the facade,
// and the artifact links numkit_ops. Correctness is unchanged: the inline loop
// is the deletable fallback (no-kludge litmus); the kernel only swaps in a
// faster SIMD implementation.
struct OpsKernelOptions {
    bool enabled = false;
};

// Emit one FUNCTION_DEF as a self-contained C++ TU (RawBuffer ABI, above).
// Throws std::runtime_error on any construct outside the supported subset.
EmittedFunction emitFunction(const ASTNode &funcDef,
                             keel::ArrayRef<ParamSpec> params,
                             const TransferRegistry &reg,
                             const ClassRegistry *classes = nullptr,
                             const BridgeOptions &bridge = {},
                             const OpsKernelOptions &opsKernels = {});

// Emit a whole program (build plan §12, brick 1b): the entry function plus
// every user-function specialisation it transitively calls, monomorphised
// by argument types, into one self-contained C++ TU. `reg` must already
// carry the user functions from `table` (registerUserFunctions) so calls
// type. `EmittedFunction::name` is the mangled entry symbol to call.
//
// v1b scope: interprocedural calls pass and return UNBOXED SCALARS (a call
// whose arg or result is an array/Dynamic is refused — array-valued
// interprocedural calls are a later brick). The entry itself may still take
// /return arrays (RawBuffer ABI) as in emitFunction.
EmittedFunction emitProgram(const ASTNode &entryDef,
                            keel::ArrayRef<ParamSpec> params,
                            const FunctionTable &table, const TransferRegistry &reg,
                            const ClassRegistry *classes = nullptr,
                            const BridgeOptions &bridge = {},
                            const OpsKernelOptions &opsKernels = {});

// Emit the C++ struct for a class (build plan §12, brick 5). The object's
// storage is the same for value and handle classes — handle-ness changes
// only the VARIABLE type (`numkit_rt::handle<Foo>` vs `Foo`), not this struct.
// v1: stored scalar properties only (a non-scalar field is refused).
keel::String emitClassStruct(const ClassInfo &ci);

// Emit a SCALAR function wrapped as a loadable nk plugin (tiered
// acceleration / codegen-as-plugin, DESIGN.md §6b): the compiled function
// (emitFunction, RawBuffer ABI) + the plugin ABI hooks
// (numkit_plugin_abi_version / numkit_plugin_register) + an numkit_fn that marshals
// numkit_val<->scalar through the host table and calls it. Compiled to a shared
// library and loaded with numkit_load_plugin, the result makes `exportName` run
// native inside a live session.
//
// v1 scope: the single output must be an unboxed SCALAR (verified by
// inference); each parameter is a scalar OR a double row vector (unboxed into
// a buffer at the boundary). A multi-output / array-output / Dynamic function
// is refused via the explicit boundary — an array OUTPUT needs an output-size
// protocol (a later brick). `abiHeaderPath` is the include path to numkit_plugin.h
// that the generated TU will #include.
keel::String emitScalarPlugin(const ASTNode &funcDef,
                              keel::ArrayRef<ParamSpec> params,
                              const TransferRegistry &reg,
                              keel::StringRef exportName,
                              keel::StringRef abiHeaderPath,
                              const ClassRegistry *classes = nullptr);

} // namespace numkit::codegen
