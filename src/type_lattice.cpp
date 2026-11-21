// codegen/src/type_lattice.cpp
//
// Implementation of the type lattice — see type_lattice.hpp.

#include <numkit/codegen/type_lattice.hpp>

#include <keel/format.h>

#include <cmath>

namespace numkit::codegen {

bool Shape::operator==(const Shape &o) const
{
    if (kind != o.kind) return false;
    if (kind == ShapeKind::KnownDims) return rows == o.rows && cols == o.cols;
    if (kind == ShapeKind::NDims) return nd.as_ref() == o.nd.as_ref();
    return true;  // Unknown == Unknown, Scalar == Scalar
}

Shape joinShape(const Shape &a, const Shape &b)
{
    // Equal shapes (incl. Unknown==Unknown, Scalar==Scalar, matching
    // KnownDims / NDims) are their own LUB.
    if (a == b) return a;
    // Two NDims of the SAME rank join per dimension: a dimension both agree
    // on is kept, any disagreement becomes unknown (0). Rank disagreement, or
    // NDims vs a different kind, we cannot prove away -> Unknown.
    if (a.kind == ShapeKind::NDims && b.kind == ShapeKind::NDims
        && a.nd.size() == b.nd.size()) {
        keel::Array<keel::usize> d(a.nd.size());
        for (keel::usize i = 0; i < d.size(); ++i) d[i] = (a.nd[i] == b.nd[i]) ? a.nd[i] : 0;
        return Shape::ndShape(d);
    }
    // Two provably 1-D vector shapes preserve 1-D rank: same orientation joins to
    // RowVector / ColVector; differing or erased orientation joins to Vector1D.
    if (a.isVector1D() && b.isVector1D()) {
        auto isRow = [](const Shape &s) {
            return s.kind == ShapeKind::RowVector
                   || (s.kind == ShapeKind::KnownDims && s.rows == 1 && s.cols > 1)
                   || (s.kind == ShapeKind::NDims && s.nd.size() == 2
                       && s.nd[0] == 1 && s.nd[1] == 0);
        };
        auto isCol = [](const Shape &s) {
            return s.kind == ShapeKind::ColVector
                   || (s.kind == ShapeKind::KnownDims && s.cols == 1 && s.rows > 1)
                   || (s.kind == ShapeKind::NDims && s.nd.size() == 2
                       && s.nd[1] == 1 && s.nd[0] == 0);
        };
        if (isRow(a) && isRow(b)) return Shape::rowVector();
        if (isCol(a) && isCol(b)) return Shape::colVector();
        return Shape::vector1D();
    }
    // Any other disagreement (Scalar vs KnownDims, two different 2-D KnownDims,
    // mixed rank/kind) collapses to Unknown — no longer statically fixed.
    return Shape::unknown();
}

bool InferredType::isUnboxableScalar() const
{
    if (kind != TypeKind::Concrete) return false;
    if (!shape.isScalar())          return false;
    switch (dtype) {
        case ValueType::DOUBLE:
        case ValueType::SINGLE:
        case ValueType::LOGICAL:
        case ValueType::COMPLEX:
        case ValueType::INT8:
        case ValueType::INT16:
        case ValueType::INT32:
        case ValueType::INT64:
        case ValueType::UINT8:
        case ValueType::UINT16:
        case ValueType::UINT32:
        case ValueType::UINT64:
        case ValueType::CHAR:  // a 1x1 char is a uint16 code-unit scalar
            return true;
        // The aggregate / handle types (CELL / STRUCT / FUNC_HANDLE / STRING /
        // OBJECT / EMPTY) are not unboxable C++ scalars in the MVP.
        default:
            return false;
    }
}

const InferredType *StructLayout::field(keel::StringRef name) const
{
    for (const auto &f : fields)
        if (f.first == name) return &f.second;
    return nullptr;
}

bool StructLayout::operator==(const StructLayout &o) const
{
    if (fields.size() != o.fields.size()) return false;
    for (keel::usize i = 0; i < fields.size(); ++i)
        if (fields[i].first != o.fields[i].first || fields[i].second != o.fields[i].second)
            return false;  // InferredType::!= recurses into nested struct layouts
    return true;
}

// Equality of two (possibly null) shared struct layouts: same pointer (incl. both null)
// is equal; one null is unequal; otherwise a deep field-by-field compare.
static bool structLayoutEq(const keel::Arc<const StructLayout> &a,
                           const keel::Arc<const StructLayout> &b)
{
    if (a.get() == b.get()) return true;
    if (!a || !b) return false;
    return *a == *b;
}

bool InferredType::operator==(const InferredType &o) const
{
    if (kind != o.kind) return false;
    if (kind != TypeKind::Concrete) return true;  // Bottom/Dynamic carry no payload
    // classId is -1 for non-object types and structLayout is null for non-struct types,
    // so both are no-ops on the numeric path.
    return dtype == o.dtype && shape == o.shape && classId == o.classId
           && structLayoutEq(structLayout, o.structLayout);
}

InferredType join(const InferredType &a, const InferredType &b)
{
    if (a.isBottom()) return b;             // Bottom is the identity
    if (b.isBottom()) return a;
    if (a.isDynamic() || b.isDynamic())     // Dynamic is absorbing
        return InferredType::dynamic();

    // Both Concrete. Differing dtype => the variable is type-unstable at
    // this merge (e.g. `x=1.0` on one path, `x='c'` on another) and must
    // be boxed. Same dtype => keep it; the shape generalises.
    if (a.dtype != b.dtype)
        return InferredType::dynamic();
    // Two object instances unify only if they are the SAME class; distinct
    // classes at a merge are type-unstable -> Dynamic (boxed). (Common
    // superclass / polymorphism is a later refinement.)
    if (a.dtype == ValueType::OBJECT) {
        if (a.classId != b.classId) return InferredType::dynamic();
        return InferredType::object(a.classId);
    }
    // Two struct values join by WIDTH-UNION: the merged layout has every field of either
    // side, and a field present in BOTH takes the join of its two types. This is exactly
    // what struct accumulation needs -- a struct built incrementally (s.a=..; s.b=..) joins
    // its partial layouts {a} and {a,b} to the full {a,b}; a field built on only one branch
    // is still present (the field-locals are function-scoped). Identical layouts short-circuit.
    if (a.dtype == ValueType::STRUCT) {
        if (structLayoutEq(a.structLayout, b.structLayout)) return a;
        auto merged = keel::make_arc<StructLayout>();
        if (a.structLayout) merged->fields = a.structLayout->fields.clone();
        if (b.structLayout)
            for (const auto &bf : b.structLayout->fields) {
                bool found = false;
                for (auto &mf : merged->fields)
                    if (mf.first == bf.first) {
                        mf.second = join(mf.second, bf.second);  // common field -> join types
                        found     = true;
                        break;
                    }
                if (!found) merged->fields.push_back(bf.clone());
            }
        return InferredType::structOf(keel::move(merged));
    }
    return InferredType::concrete(a.dtype, joinShape(a.shape, b.shape));
}

keel::String InferredType::str() const
{
    switch (kind) {
        case TypeKind::Bottom:  return keel::String::from("bottom");
        case TypeKind::Dynamic: return keel::String::from("dynamic");
        case TypeKind::Concrete: {
            keel::String os;
            if (dtype == ValueType::OBJECT) {
                keel::format_into(os, "object#{}", classId);
                return os;
            }
            if (dtype == ValueType::STRUCT) {
                os.append("struct{");
                if (structLayout)
                    for (keel::usize i = 0; i < structLayout->fields.size(); ++i) {
                        if (i) os.push_back(',');
                        os.append(structLayout->fields[i].first.as_ref());
                    }
                os.push_back('}');
                return os;
            }
            os.append(mtypeName(dtype));
            switch (shape.kind) {
                case ShapeKind::Scalar:    os.append(" scalar"); break;
                case ShapeKind::RowVector: os.append(" 1x?"); break;
                case ShapeKind::ColVector: os.append(" ?x1"); break;
                case ShapeKind::Vector1D:  os.append(" vec1d"); break;
                case ShapeKind::KnownDims:
                    keel::format_into(os, " {}x{}", shape.rows, shape.cols);
                    break;
                case ShapeKind::NDims:
                    os.push_back(' ');
                    for (keel::usize i = 0; i < shape.nd.size(); ++i) {
                        if (i) os.push_back('x');
                        if (shape.nd[i] == 0) os.push_back('?');
                        else keel::format_into(os, "{}", shape.nd[i]);
                    }
                    break;
                case ShapeKind::Unknown:   os.append(" [?]"); break;
            }
            return os;
        }
    }
    return keel::String::from("?");
}

// ── ConstVal ──────────────────────────────────────────────────────────

bool ConstVal::asDim(keel::usize &out) const
{
    if (kind != ConstKind::KnownReal) return false;
    // A valid dimension is a non-negative integer (finite) value.
    if (!std::isfinite(value) || value < 0.0) return false;
    const double r = std::floor(value);
    if (r != value) return false;  // non-integral
    out = keel::as<keel::usize>(r);
    return true;
}

bool ConstVal::operator==(const ConstVal &o) const
{
    if (kind != o.kind) return false;
    if (kind != ConstKind::KnownReal) return true;  // both Unknown
    // Bit-exact compare so NaN-vs-NaN and +0/-0 are handled consistently
    // with the rest of the lattice (Known(NaN) only equals Known(NaN)).
    if (std::isnan(value) && std::isnan(o.value)) return true;
    return value == o.value;
}

ConstVal join(const ConstVal &a, const ConstVal &b)
{
    if (a.isKnown() && b.isKnown() && a == b) return a;  // same constant
    return ConstVal::unknown();                          // disagree → top
}

keel::String ConstVal::str() const
{
    if (kind != ConstKind::KnownReal) return keel::String::from("?");
    return keel::format("%g", value);
}

} // namespace numkit::codegen
