// codegen/include/numkit/codegen/type_lattice.hpp
//
// Type lattice for the nk static type-inference pass — the
// foundation of the AST -> typed-IR -> C++ transpiler.
//
// An InferredType abstracts, at a single program point, what a nk
// value's dtype and shape are KNOWN to be. The inference pass runs a
// forward dataflow over the control-flow graph; this lattice defines
// the values it propagates and the `join` used at control-flow merges.
//
// Lattice (the relation is "more precise than"):
//
//          Dynamic            <- top: type unknown -> stays a boxed Value
//        /    |    \
//   Concrete(double,3x3) ...  <- a definite, unboxable dtype + shape
//        \    |    /
//           Bottom            <- contradiction / unreachable
//
// The whole point: a value that infers to a Concrete *scalar* numeric
// type can be emitted as an unboxed C++ primitive (`double`, ...)
// instead of a heap-boxed `numkit::Value`. Boxing is the bulk of the
// interpreter's per-operation cost, so proving "this is an unboxed
// double" is exactly what makes transpiled code fast.
//
// This pass is offline analysis (sibling of scriptgraph): it consumes a
// parsed AST and registers no engine builtin.

#pragma once

#include <numkit/value/value_type.hpp>

#include <keel/array.h>
#include <keel/core.h>
#include <keel/ref.h>
#include <keel/smart_ptr.h>
#include <keel/string.h>

namespace numkit::codegen {

// ── Shape lattice ─────────────────────────────────────────────────────
// MVP: scalar / concrete-2D-dims / unknown. Symbolic dims and ranks
// beyond 2-D are deliberately deferred; the join already collapses
// anything it cannot prove equal to Unknown, so adding richer shapes
// later only refines results, never breaks soundness.
enum class ShapeKind : keel::u8 {
    Unknown,    // top — any rank / shape (not statically provable)
    Scalar,     // 1 x 1
    RowVector,  // 1 x N, N not statically known (e.g. 1:N, runtime linspace)
    ColVector,  // N x 1, N not statically known
    Vector1D,   // 1-D vector of unknown orientation and runtime length
    KnownDims,  // concrete rows x cols (rank 2, both dims compile-time)
    NDims,      // ranked: rank = nd.size(); each nd[k] is the dim size, 0 = runtime-unknown.
                // Used for rank >= 3 (true N-D) AND a rank-2 matrix with a runtime dim.
                // A fully-known rank-2 canonicalises to KnownDims (see ndShape).
};

struct Shape {
    ShapeKind                kind = ShapeKind::Unknown;
    keel::usize              rows = 0;  // meaningful only when kind == KnownDims
    keel::usize              cols = 0;
    keel::Array<keel::usize> nd;        // meaningful only when kind == NDims (0 = unknown dim)

    Shape() noexcept = default;
    Shape(ShapeKind k, keel::usize r, keel::usize c, keel::Array<keel::usize> n = {}) noexcept
        : kind(k), rows(r), cols(c), nd(keel::move(n))
    {}
    Shape(const Shape &o) : kind(o.kind), rows(o.rows), cols(o.cols), nd(o.nd.clone()) {}
    Shape &operator=(const Shape &o)
    {
        if (this != &o) {
            kind = o.kind;
            rows = o.rows;
            cols = o.cols;
            nd   = o.nd.clone();
        }
        return *this;
    }
    Shape(Shape &&) noexcept = default;
    Shape &operator=(Shape &&) noexcept = default;
    ~Shape() noexcept = default;

    Shape clone() const { return *this; }

    static Shape unknown()   { return Shape(ShapeKind::Unknown, 0, 0); }
    static Shape scalar()    { return Shape(ShapeKind::Scalar, 1, 1); }
    static Shape rowVector() { return Shape(ShapeKind::RowVector, 0, 0); }
    static Shape colVector() { return Shape(ShapeKind::ColVector, 0, 0); }
    static Shape vector1D()  { return Shape(ShapeKind::Vector1D, 0, 0); }
    // A 1x1 KnownDims is canonicalised to Scalar so the two never alias.
    static Shape dims(keel::usize r, keel::usize c)
    {
        if (r == 1 && c == 1) return scalar();
        return Shape(ShapeKind::KnownDims, r, c);
    }
    // A ranked shape (column-major). A fully-known rank-2 collapses to
    // KnownDims so static matrices keep their existing single representation;
    // everything else (rank >= 3, or a rank-2 with a runtime dim) is NDims.
    static Shape ndShape(keel::ArrayRef<keel::usize> d)
    {
        if (d.size() == 2 && d[0] >= 1 && d[1] >= 1) return dims(d[0], d[1]);
        Shape s;
        s.kind = ShapeKind::NDims;
        s.nd   = keel::Array<keel::usize>::from(d);
        return s;
    }

    bool        isScalar() const { return kind == ShapeKind::Scalar; }
    bool        isVector1D() const
    {
        return kind == ShapeKind::RowVector || kind == ShapeKind::ColVector
               || kind == ShapeKind::Vector1D
               || (kind == ShapeKind::KnownDims
                   && ((rows == 1 && cols > 1) || (cols == 1 && rows > 1)))
               || (kind == ShapeKind::NDims && nd.size() == 2
                   && ((nd[0] == 1 && nd[1] == 0) || (nd[1] == 1 && nd[0] == 0)));
    }
    bool        isNDims() const { return kind == ShapeKind::NDims; }
    keel::usize ndRank() const { return kind == ShapeKind::NDims ? nd.size() : 0; }
    bool        operator==(const Shape &o) const;
    bool        operator!=(const Shape &o) const { return !(*this == o); }
};

// Least upper bound of two shapes (used at control-flow merges).
Shape joinShape(const Shape &a, const Shape &b);

struct StructLayout;  // an anonymous struct's field layout (defined below InferredType)

// ── Type lattice ──────────────────────────────────────────────────────
enum class TypeKind : keel::u8 {
    Bottom,    // no value reaches here (identity of join)
    Concrete,  // a definite dtype + shape
    Dynamic,   // top — statically unknown -> boxed numkit::Value
};

struct InferredType {
    TypeKind  kind  = TypeKind::Dynamic;
    ValueType dtype = ValueType::EMPTY;  // meaningful only when Concrete
    Shape     shape{};                   // meaningful only when Concrete
    // Class identity, meaningful ONLY when dtype == OBJECT (a class
    // instance). -1 for every non-object type, so it never perturbs the
    // numeric path's equality / join. Indexes a ClassRegistry (classinfo.hpp).
    int       classId = -1;
    // Field layout, meaningful ONLY when dtype == STRUCT (a plain struct value).
    // null for every non-struct type, so it never perturbs the numeric path's
    // equality / join. Shared + immutable (cheap to copy across the lattice). A
    // field type may itself be a STRUCT, so the recursion goes through InferredType.
    keel::Arc<const StructLayout> structLayout;

    InferredType() noexcept = default;
    InferredType(TypeKind k, ValueType dt, Shape s, int cid,
                 keel::Arc<const StructLayout> sl = {}) noexcept
        : kind(k), dtype(dt), shape(keel::move(s)), classId(cid), structLayout(keel::move(sl))
    {}
    InferredType(const InferredType &o)
        : kind(o.kind), dtype(o.dtype), shape(o.shape.clone()), classId(o.classId),
          structLayout(o.structLayout)
    {}
    InferredType &operator=(const InferredType &o)
    {
        if (this != &o) {
            kind         = o.kind;
            dtype        = o.dtype;
            shape        = o.shape.clone();
            classId      = o.classId;
            structLayout = o.structLayout;
        }
        return *this;
    }
    InferredType(InferredType &&) noexcept = default;
    InferredType &operator=(InferredType &&) noexcept = default;
    ~InferredType() noexcept = default;

    InferredType clone() const { return *this; }

    // ── constructors ──
    static InferredType dynamic() { return InferredType(TypeKind::Dynamic, ValueType::EMPTY, {}, -1); }
    static InferredType bottom()  { return InferredType(TypeKind::Bottom,  ValueType::EMPTY, {}, -1); }
    static InferredType concrete(ValueType dt, Shape s)
    {
        return InferredType(TypeKind::Concrete, dt, keel::move(s), -1);
    }
    static InferredType scalar(ValueType dt) { return concrete(dt, Shape::scalar()); }
    // A class instance (value or handle decided by the ClassRegistry, not
    // the lattice). A scalar object in v1 (object arrays are a later tier).
    static InferredType object(int classId)
    {
        return InferredType(TypeKind::Concrete, ValueType::OBJECT, Shape::scalar(), classId);
    }
    // A plain (anonymous) struct value with the given field layout (name -> type).
    // A scalar struct in v1 (struct arrays are a later tier). The layout is shared
    // (cheap copy; null for every non-struct type).
    static InferredType structOf(keel::Arc<const StructLayout> layout)
    {
        return InferredType(TypeKind::Concrete, ValueType::STRUCT, Shape::scalar(), -1,
                            keel::move(layout));
    }

    bool isDynamic()  const { return kind == TypeKind::Dynamic; }
    bool isConcrete() const { return kind == TypeKind::Concrete; }
    bool isBottom()   const { return kind == TypeKind::Bottom; }
    bool isObject()   const { return kind == TypeKind::Concrete && dtype == ValueType::OBJECT; }
    bool isStruct()   const { return kind == TypeKind::Concrete && dtype == ValueType::STRUCT; }

    // True when this can be emitted as an unboxed C++ primitive scalar
    // (double / float / bool / intN / complex<double>) rather than a
    // heap-boxed numkit::Value. The transpiler's green light.
    bool isUnboxableScalar() const;

    bool operator==(const InferredType &o) const;
    bool operator!=(const InferredType &o) const { return !(*this == o); }

    keel::String str() const;  // human-readable, for debug / diagnostics
};

struct StructField {
    keel::String first;
    InferredType second;

    StructField() noexcept = default;
    StructField(keel::StringRef name, InferredType type)
        : first(keel::String::from(name)), second(keel::move(type))
    {}
    StructField(const StructField &o) : first(o.first.clone()), second(o.second.clone()) {}
    StructField &operator=(const StructField &o)
    {
        if (this != &o) {
            first  = o.first.clone();
            second = o.second.clone();
        }
        return *this;
    }
    StructField(StructField &&) noexcept = default;
    StructField &operator=(StructField &&) noexcept = default;
    ~StructField() noexcept = default;

    StructField clone() const { return *this; }

    bool operator==(const StructField &o) const
    {
        return first == o.first && second == o.second;
    }
    bool operator!=(const StructField &o) const { return !(*this == o); }
};

// An anonymous struct's field layout: ordered (name -> type) pairs in MATLAB field order.
// Carried by a STRUCT InferredType via a shared_ptr (immutable once built). A field type may
// itself be a STRUCT (nested s.a.b), so the recursion goes through InferredType. Two layouts
// are equal iff they have the same fields (name AND type) in the same order.
struct StructLayout {
    keel::Array<StructField> fields;

    StructLayout() noexcept = default;
    StructLayout(const StructLayout &o) : fields(o.fields.clone()) {}
    StructLayout &operator=(const StructLayout &o)
    {
        if (this != &o) fields = o.fields.clone();
        return *this;
    }
    StructLayout(StructLayout &&) noexcept = default;
    StructLayout &operator=(StructLayout &&) noexcept = default;
    ~StructLayout() noexcept = default;

    StructLayout clone() const { return *this; }

    const InferredType *field(keel::StringRef name) const;  // nullptr if absent
    bool                operator==(const StructLayout &o) const;
    bool                operator!=(const StructLayout &o) const { return !(*this == o); }
};

// Lattice join (least upper bound) — combine the two inferred types that
// reach a control-flow merge point. Bottom is the identity; Dynamic is
// absorbing; two Concretes join to a Concrete iff their dtype agrees
// (shape generalises via joinShape), otherwise the variable is
// type-unstable here and the result is Dynamic (must be boxed).
InferredType join(const InferredType &a, const InferredType &b);

// ── Constant lattice ──────────────────────────────────────────────────
// Sparse-conditional-constant-propagation facet, tracked alongside the
// type for every value. It exists because MATLAB *shapes* depend on
// argument *values*, not just types: `linspace(0,1,n)` / `zeros(n)` /
// `reshape(x,a,b)` need the compile-time value of `n`/`a`/`b` to infer a
// concrete shape. A value whose constant is Known lets a size-constructor
// transfer function produce KnownDims instead of an Unknown shape.
//
// Lattice: Unknown (top — could be anything) over Known(real scalar).
// (Bottom/undefined is not modelled separately in the MVP — an
//  unreachable value's type is already Bottom.) Only real scalar
// constants are tracked for now; that covers the size arguments that
// drive shape inference.
enum class ConstKind : keel::u8 {
    Unknown,    // top — not a known compile-time constant
    KnownReal,  // a known real scalar value
};

struct ConstVal {
    ConstKind kind  = ConstKind::Unknown;
    keel::f64    value = 0.0;  // meaningful only when kind == KnownReal

    static ConstVal unknown()       { return {ConstKind::Unknown, 0.0}; }
    static ConstVal known(keel::f64 v) { return {ConstKind::KnownReal, v}; }

    ConstVal clone() const noexcept { return *this; }

    bool isKnown() const { return kind == ConstKind::KnownReal; }

    // Read the constant as an array dimension: a known non-negative
    // integer value, else nullopt. This is the form size-constructor
    // transfer functions consume (the `n` in zeros(n) / linspace(_,_,n)).
    // Returns nullopt for Unknown, non-integral, or negative values.
    // (Out param keeps the header free of <optional> in hot include
    //  paths; returns true on success.)
    bool asDim(keel::usize &out) const;

    bool operator==(const ConstVal &o) const;
    bool operator!=(const ConstVal &o) const { return !(*this == o); }

    keel::String str() const;
};

// Lattice join for constants: equal Known values stay Known; anything
// else (Unknown present, or two different Known values) is Unknown.
ConstVal join(const ConstVal &a, const ConstVal &b);

} // namespace numkit::codegen
