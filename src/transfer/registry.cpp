// codegen/src/transfer/registry.cpp
//
// The TransferRegistry and the top-level registerStandardTransfers()
// aggregator (calls each per-family registrar).

#include <numkit/codegen/transfer.hpp>

namespace numkit::codegen {

void TransferRegistry::add(keel::StringRef name, TransferFn fn)
{
    table_.put(keel::String::from(name), keel::move(fn));
}

void TransferRegistry::addMulti(keel::StringRef name, MultiTransferFn fn)
{
    multiTable_.put(keel::String::from(name), keel::move(fn));
}

keel::Array<InferredType> TransferRegistry::applyMulti(keel::StringRef name,
                                                       keel::ArrayRef<ArgInfo> args) const
{
    // Bottom is ABSORBING in a transfer (see apply): a ⊥ operand carries no value,
    // so the outputs are unknown. Return "no info" (empty) -> the caller defaults
    // each output to Dynamic (sound; multi-output recursion is not fixpoint-refined
    // in v1 -- single-output recursion converges via apply's ⊥ propagation).
    for (const ArgInfo &a : args)
        if (a.type.isBottom()) return {};
    const MultiTransferFn *fn = multiTable_.get(name);
    if (!fn) return {};  // no multi-output transfer
    return (*fn)(args);
}

bool TransferRegistry::has(keel::StringRef name) const
{
    return table_.contains(name);
}

keel::usize TransferRegistry::size() const
{
    return table_.size();
}

InferredType TransferRegistry::apply(keel::StringRef name,
                                     keel::ArrayRef<ArgInfo> args) const
{
    // Bottom is ABSORBING in a transfer: an operand with no value yet (⊥ -- e.g.
    // the seed of a recursive self-call before its return type is known) yields no
    // value, so the result is ⊥ too. Short-circuit before the per-fn transfer so
    // every op / builtin propagates ⊥ uniformly. This is what lets the recursion
    // return-type fixpoint converge (⊥ seed -> body -> join collapses ⊥ to the
    // base-case type). Dormant until something produces a ⊥ operand (the fixpoint
    // seed); `n * ⊥` -> ⊥, not Dynamic. (⊥ is join's identity but a transfer's
    // absorber -- the dual roles, both sound.)
    for (const ArgInfo &a : args)
        if (a.type.isBottom()) return InferredType::bottom();
    const TransferFn *fn = table_.get(name);
    if (!fn)
        return InferredType::dynamic();  // unknown builtin -> boxed (sound)
    return (*fn)(args);
}

void registerStandardTransfers(TransferRegistry &reg)
{
    registerConstructorTransfers(reg);
    registerElementwiseTransfers(reg);
    registerShapeTransfers(reg);
    // future: registerReductionTransfers(reg);
    //         registerCastTransfers(reg);
    //         registerBespokeTransfers(reg);
}

} // namespace numkit::codegen
