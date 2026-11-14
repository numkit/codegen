// codegen/src/transfer/shape.cpp
//
// Transfer functions for the shape-query family: numel / length. Both
// return a real non-negative integer count -> a scalar double, regardless
// of the operand (sound and exact: MATLAB numel/length always produce a
// 1x1 double). `size` is deferred: its no-dim form returns a 1xNdims row
// (whose length depends on the operand's rank) and the (x,dim) form a
// scalar -- registering it precisely needs rank tracking, so for now it
// is intentionally left unregistered (-> Dynamic, the sound fallback).

#include <numkit/codegen/transfer.hpp>

namespace numkit::codegen {

namespace {

// numel / length / ndims: always a scalar double count.
InferredType countTransfer(const std::vector<ArgInfo> & /*args*/)
{
    return InferredType::scalar(ValueType::DOUBLE);
}

// size(A, dim) -> a scalar double (one dimension). size(A) (no dim) -> a
// 1 x rank row vector of the dim sizes. rank is MATLAB's: 2 for a scalar /
// vector / matrix, the array's rank for N-D. Unknown shape -> unknown rank ->
// Dynamic (the sound fallback). The VALUES are filled by the emitter; this
// transfer only fixes the result SHAPE (1 x rank).
InferredType sizeTransfer(const std::vector<ArgInfo> &args)
{
    if (args.size() == 2) return InferredType::scalar(ValueType::DOUBLE);  // size(A, dim)
    if (args.size() != 1) return InferredType::dynamic();
    const Shape &sh = args[0].type.shape;
    std::size_t  rank = 0;
    switch (sh.kind) {
    case ShapeKind::Scalar:
    case ShapeKind::RowVector:
    case ShapeKind::ColVector:
    case ShapeKind::Vector1D:
    case ShapeKind::KnownDims: rank = 2; break;  // scalars / vectors / matrices are 2-D
    case ShapeKind::NDims:     rank = sh.nd.size(); break;
    case ShapeKind::Unknown:   return InferredType::dynamic();  // unknown rank -> sound fallback
    }
    return InferredType::concrete(ValueType::DOUBLE, Shape::dims(1, rank));  // 1 x rank row
}

// [r, c] = size(A): the two-output size idiom. Both outputs are real scalar
// doubles (r = rows, c = the trailing dims folded). size is variadic, but
// applyMulti is nargout-blind, so v1 models exactly the supported two-output
// arity; the emitter computes the VALUES (folding the trailing dims into c)
// and refuses other nargout. Modelling only the supported shape keeps the
// fallback sound (an unsupported arity -> Dynamic / explicit refusal).
std::vector<InferredType> sizeMultiTransfer(const std::vector<ArgInfo> & /*args*/)
{
    return {InferredType::scalar(ValueType::DOUBLE), InferredType::scalar(ValueType::DOUBLE)};
}

// [m, i] = max(x) / min(x): two outputs -- the extremum VALUE and its 1-based INDEX
// (both real scalar doubles). Typed concrete ONLY for a DOUBLE VECTOR, which the
// emitter lowers natively (a scan tracking the extremum + its position) in EVERY
// tier -- so the transfer and the always-native emit agree. A matrix/N-D or non-
// double operand -> Dynamic, which routes to the bridged multi-output path.
std::vector<InferredType> maxMinMultiTransfer(const std::vector<ArgInfo> &args)
{
    if (args.size() != 1 || !args[0].type.isConcrete()
        || args[0].type.dtype != ValueType::DOUBLE)
        return {InferredType::dynamic(), InferredType::dynamic()};
    switch (args[0].type.shape.kind) {
    case ShapeKind::Scalar:
    case ShapeKind::RowVector:
    case ShapeKind::ColVector:
    case ShapeKind::Vector1D:
        return {InferredType::scalar(ValueType::DOUBLE), InferredType::scalar(ValueType::DOUBLE)};
    case ShapeKind::KnownDims:
        if ((args[0].type.shape.rows == 1 && args[0].type.shape.cols >= 1)
            || (args[0].type.shape.cols == 1 && args[0].type.shape.rows >= 1))
            return {InferredType::scalar(ValueType::DOUBLE), InferredType::scalar(ValueType::DOUBLE)};
        // a true 2-D matrix [m,i]=max(A) is column-wise -> two 1 x n row vectors (value + index).
        if (args[0].type.shape.rows > 1 && args[0].type.shape.cols > 1)
            return {InferredType::concrete(ValueType::DOUBLE, Shape::rowVector()),
                    InferredType::concrete(ValueType::DOUBLE, Shape::rowVector())};
        return {InferredType::dynamic(), InferredType::dynamic()};
    case ShapeKind::NDims:  // runtime-dim rank-2 matrix -> column-wise -> two 1-D row vectors
        if (args[0].type.shape.nd.size() == 2)
            return {InferredType::concrete(ValueType::DOUBLE, Shape::rowVector()),
                    InferredType::concrete(ValueType::DOUBLE, Shape::rowVector())};
        return {InferredType::dynamic(), InferredType::dynamic()};
    default: return {InferredType::dynamic(), InferredType::dynamic()};
    }
}

// [s, i] = sort(x[, 'ascend'|'descend']): sorted values (same shape as x) + the 1-based
// permutation indices (a DOUBLE array of the same shape). v1: a 1-D DOUBLE vector (a matrix
// is the next brick); a numeric dim arg -> deferred.
std::vector<InferredType> sortMultiTransfer(const std::vector<ArgInfo> &args)
{
    if (args.empty() || args.size() > 2 || !args[0].type.isConcrete()
        || args[0].type.dtype != ValueType::DOUBLE)
        return {InferredType::dynamic(), InferredType::dynamic()};
    if (args.size() == 2 && args[1].type.dtype != ValueType::CHAR)
        return {InferredType::dynamic(), InferredType::dynamic()};  // a numeric dim arg -> deferred
    switch (args[0].type.shape.kind) {
    case ShapeKind::RowVector:
    case ShapeKind::ColVector:
    case ShapeKind::Vector1D:
        return {args[0].type, InferredType::concrete(ValueType::DOUBLE, args[0].type.shape)};
    case ShapeKind::KnownDims:
        if ((args[0].type.shape.rows == 1 && args[0].type.shape.cols > 1)
            || (args[0].type.shape.cols == 1 && args[0].type.shape.rows > 1))
            return {args[0].type, args[0].type};
        // [s,i]=sort(A) on a matrix sorts each column -> both outputs are the same m x n shape
        // (sorted values + per-column 1-based permutation).
        if (args[0].type.shape.rows > 1 && args[0].type.shape.cols > 1)
            return {args[0].type, args[0].type};
        return {InferredType::dynamic(), InferredType::dynamic()};
    case ShapeKind::NDims:  // runtime-dim rank-2 matrix -> per-column, both outputs m x n
        if (args[0].type.shape.nd.size() == 2) return {args[0].type, args[0].type};
        return {InferredType::dynamic(), InferredType::dynamic()};
    default: return {InferredType::dynamic(), InferredType::dynamic()};
    }
}

// Binary set op (union / intersect / setdiff) over two 1-D DOUBLE vectors -> a sorted distinct
// 1-D DOUBLE result (runtime length -> Vector1D). Anything else -> Dynamic. Shared because the
// shape rule is identical; the per-op element selection lives in the emitter.
InferredType setopBinaryTransfer(const std::vector<ArgInfo> &args)
{
    if (args.size() != 2 || !args[0].type.isConcrete() || !args[1].type.isConcrete()
        || args[0].type.dtype != ValueType::DOUBLE || args[1].type.dtype != ValueType::DOUBLE)
        return InferredType::dynamic();
    if (args[0].type.shape.isVector1D() && args[1].type.shape.isVector1D())
        return InferredType::concrete(ValueType::DOUBLE, Shape::vector1D());
    return InferredType::dynamic();
}

// ismember(a, b): a LOGICAL mask the shape of a -- tf(i) = a(i) is a member of b. v1: two 1-D
// DOUBLE vectors -> a 1-D LOGICAL result. (Only the SINGLE-output form; [tf,loc]=ismember has no
// addMulti entry, so it stays Dynamic/bridged -- and is the bridged-multi-output e2e exemplar.)
InferredType ismemberTransfer(const std::vector<ArgInfo> &args)
{
    if (args.size() != 2 || !args[0].type.isConcrete() || !args[1].type.isConcrete()
        || args[0].type.dtype != ValueType::DOUBLE || args[1].type.dtype != ValueType::DOUBLE)
        return InferredType::dynamic();
    if (args[0].type.shape.isVector1D() && args[1].type.shape.isVector1D())
        return InferredType::concrete(ValueType::LOGICAL, Shape::vector1D());
    return InferredType::dynamic();
}

// median(x) / mode(x): a 1-D vector reduces to a scalar; a 2-D matrix reduces COLUMN-wise to a
// 1 x n row vector. Both are sort-based (exact, order-independent), so 2-D is native (unlike sum/
// mean, which stay bridged on the shared vectorReductionTransfer). A scalar / N-D -> Dynamic.
InferredType statColumnReduceTransfer(const std::vector<ArgInfo> &args)
{
    if (args.size() != 1 || !args[0].type.isConcrete() || args[0].type.dtype != ValueType::DOUBLE)
        return InferredType::dynamic();
    const Shape &s = args[0].type.shape;
    switch (s.kind) {
    case ShapeKind::RowVector:
    case ShapeKind::ColVector:
    case ShapeKind::Vector1D:
        return InferredType::scalar(ValueType::DOUBLE);
    case ShapeKind::KnownDims:
        if ((s.rows == 1 && s.cols > 1) || (s.cols == 1 && s.rows > 1))
            return InferredType::scalar(ValueType::DOUBLE);
        if (s.rows > 1 && s.cols > 1)  // a true 2-D matrix -> column-wise -> 1 x n row vector
            return InferredType::concrete(ValueType::DOUBLE, Shape::rowVector());
        return InferredType::dynamic();
    case ShapeKind::NDims:
        if (s.nd.size() == 2) return InferredType::concrete(ValueType::DOUBLE, Shape::rowVector());
        return InferredType::dynamic();
    default: return InferredType::dynamic();
    }
}

// sortrows(A): sort the ROWS of a 2-D matrix in ascending lexicographic order -> same shape. v1: a
// true 2-D DOUBLE matrix (a vector/scalar/N-D -> Dynamic/bridged, so the emit never sees a non-2-D).
InferredType sortrowsTransfer(const std::vector<ArgInfo> &args)
{
    if (args.size() != 1 || !args[0].type.isConcrete() || args[0].type.dtype != ValueType::DOUBLE)
        return InferredType::dynamic();
    const Shape &s = args[0].type.shape;
    if ((s.kind == ShapeKind::KnownDims && s.rows > 1 && s.cols > 1)
        || (s.kind == ShapeKind::NDims && s.nd.size() == 2))
        return args[0].type;  // same shape (rows reordered)
    return InferredType::dynamic();
}

// [u, ia, ic] = unique(x): u = sorted distinct values; ia s.t. u = x(ia) (the FIRST occurrence,
// nk default -- probed); ic s.t. x = u(ic). All three are 1-D DOUBLE vectors (u/ia length =
// #distinct, ic length = numel(x); all runtime -> Vector1D).
std::vector<InferredType> uniqueMultiTransfer(const std::vector<ArgInfo> &args)
{
    if (args.size() != 1 || !args[0].type.isConcrete() || args[0].type.dtype != ValueType::DOUBLE)
        return {InferredType::dynamic(), InferredType::dynamic(), InferredType::dynamic()};
    const Shape &sh = args[0].type.shape;
    const bool   ok = sh.isVector1D()
                    || (sh.kind == ShapeKind::KnownDims && sh.rows > 1 && sh.cols > 1)
                    || (sh.kind == ShapeKind::NDims && sh.nd.size() == 2);
    if (ok)
        return {InferredType::concrete(ValueType::DOUBLE, Shape::vector1D()),
                InferredType::concrete(ValueType::DOUBLE, Shape::vector1D()),
                InferredType::concrete(ValueType::DOUBLE, Shape::vector1D())};
    return {InferredType::dynamic(), InferredType::dynamic(), InferredType::dynamic()};
}

// [c, ia, ib] = intersect(a, b): c = sorted distinct common values; ia s.t. c = a(ia) (FIRST
// occurrence in a -- probed); ib s.t. c = b(ib) (first occurrence in b). All three 1-D DOUBLE
// (length = #common, runtime -> Vector1D). v1: two 1-D DOUBLE vectors; else Dynamic.
std::vector<InferredType> intersectMultiTransfer(const std::vector<ArgInfo> &args)
{
    if (args.size() != 2 || !args[0].type.isConcrete() || !args[1].type.isConcrete()
        || args[0].type.dtype != ValueType::DOUBLE || args[1].type.dtype != ValueType::DOUBLE)
        return {InferredType::dynamic(), InferredType::dynamic(), InferredType::dynamic()};
    if (args[0].type.shape.isVector1D() && args[1].type.shape.isVector1D())
        return {InferredType::concrete(ValueType::DOUBLE, Shape::vector1D()),
                InferredType::concrete(ValueType::DOUBLE, Shape::vector1D()),
                InferredType::concrete(ValueType::DOUBLE, Shape::vector1D())};
    return {InferredType::dynamic(), InferredType::dynamic(), InferredType::dynamic()};
}

// [c, ia] = setdiff(a, b): c = sorted distinct values of a not in b; ia s.t. c = a(ia) (FIRST
// occurrence in a). Both 1-D DOUBLE (length = #result). v1: two 1-D DOUBLE vectors; else Dynamic.
std::vector<InferredType> setdiffMultiTransfer(const std::vector<ArgInfo> &args)
{
    if (args.size() != 2 || !args[0].type.isConcrete() || !args[1].type.isConcrete()
        || args[0].type.dtype != ValueType::DOUBLE || args[1].type.dtype != ValueType::DOUBLE)
        return {InferredType::dynamic(), InferredType::dynamic()};
    if (args[0].type.shape.isVector1D() && args[1].type.shape.isVector1D())
        return {InferredType::concrete(ValueType::DOUBLE, Shape::vector1D()),
                InferredType::concrete(ValueType::DOUBLE, Shape::vector1D())};
    return {InferredType::dynamic(), InferredType::dynamic()};
}

// [u, ia, ib] = union(a, b): u = sorted distinct of [a b]; ia = the a-sourced first-occurrence
// indices (values whose first occurrence in the concatenation [a b] falls in the a part), ib = the
// b-sourced ones. |ia|+|ib| = |u| (3-output); the 2-output [u,ia] form drops ib (ia stays the
// a-sourced indices -- probed). All 1-D DOUBLE. v1: two 1-D DOUBLE vectors; else Dynamic.
std::vector<InferredType> unionMultiTransfer(const std::vector<ArgInfo> &args)
{
    if (args.size() != 2 || !args[0].type.isConcrete() || !args[1].type.isConcrete()
        || args[0].type.dtype != ValueType::DOUBLE || args[1].type.dtype != ValueType::DOUBLE)
        return {InferredType::dynamic(), InferredType::dynamic(), InferredType::dynamic()};
    if (args[0].type.shape.isVector1D() && args[1].type.shape.isVector1D())
        return {InferredType::concrete(ValueType::DOUBLE, Shape::vector1D()),
                InferredType::concrete(ValueType::DOUBLE, Shape::vector1D()),
                InferredType::concrete(ValueType::DOUBLE, Shape::vector1D())};
    return {InferredType::dynamic(), InferredType::dynamic(), InferredType::dynamic()};
}

// transpose (A.') / ctranspose (A'): swap the two dimensions. A row becomes a
// column and vice-versa; a matrix's dims swap; a scalar is unchanged. The
// dtype is preserved (ctranspose conjugates the VALUES of a complex operand
// but the type stays complex; the conjugation is emitted, not inferred). N-D
// transpose is undefined in MATLAB -> Dynamic (the sound fallback). Shared by
// both operators since the shape rule is identical.
InferredType transposeTransfer(const std::vector<ArgInfo> &args)
{
    if (args.size() != 1) return InferredType::dynamic();
    const InferredType &t = args[0].type;
    if (!t.isConcrete()) return InferredType::dynamic();
    switch (t.shape.kind) {
    case ShapeKind::Scalar:    return InferredType::concrete(t.dtype, Shape::scalar());
    case ShapeKind::RowVector: return InferredType::concrete(t.dtype, Shape::colVector());
    case ShapeKind::ColVector: return InferredType::concrete(t.dtype, Shape::rowVector());
    case ShapeKind::Vector1D:  return InferredType::concrete(t.dtype, Shape::vector1D());
    case ShapeKind::KnownDims:
        return InferredType::concrete(t.dtype, Shape::dims(t.shape.cols, t.shape.rows));
    case ShapeKind::NDims:
        // A rank-2 matrix (incl. a runtime-dim 2-D) -> swap the two dims. ndShape keeps
        // it NDims when a dim is runtime, or canonicalises to KnownDims if both are now
        // known. A true N-D (rank>=3) transpose is undefined in MATLAB -> Dynamic.
        if (t.shape.nd.size() == 2)
            return InferredType::concrete(t.dtype,
                                          Shape::ndShape({t.shape.nd[1], t.shape.nd[0]}));
        return InferredType::dynamic();
    case ShapeKind::Unknown: return InferredType::dynamic();
    }
    return InferredType::dynamic();
}

// rot90(A): rotate a 2-D matrix 90 degrees -> the dims SWAP (m x n -> n x m), like
// transpose. Only the 1-arg matrix forms are typed (KnownDims rank-2 / NDims rank-2); a
// vector / scalar / the rot90(A,k) 2-arg form -> Dynamic (bridged) -- the emit handles the
// 1-arg 2-D case only.
InferredType rot90Transfer(const std::vector<ArgInfo> &args)
{
    if (args.empty() || args.size() > 2 || !args[0].type.isConcrete())
        return InferredType::dynamic();
    const Shape &s = args[0].type.shape;
    // 90/270-degree rotations swap the first two dims; 0/180 keep the shape.
    auto swapped = [&]() -> InferredType {
        if (s.kind == ShapeKind::KnownDims)
            return InferredType::concrete(args[0].type.dtype, Shape::dims(s.cols, s.rows));
        if (s.kind == ShapeKind::NDims && s.nd.size() == 2)
            return InferredType::concrete(args[0].type.dtype, Shape::ndShape({s.nd[1], s.nd[0]}));
        if (s.kind == ShapeKind::NDims && s.nd.size() == 3)  // rank-3: per-page, dim 3 fixed
            return InferredType::concrete(args[0].type.dtype,
                                          Shape::ndShape({s.nd[1], s.nd[0], s.nd[2]}));
        return InferredType::dynamic();
    };
    auto same = [&]() -> InferredType {
        if (s.kind == ShapeKind::KnownDims
            || (s.kind == ShapeKind::NDims && (s.nd.size() == 2 || s.nd.size() == 3)))
            return args[0].type;
        return InferredType::dynamic();
    };
    if (args.size() == 1) return swapped();  // rot90(A) -> 90 CCW
    // rot90(A, k) with a non-negative LITERAL k: k%4 == 0/2 keeps the shape, 1/3 swaps it. A
    // non-literal / negative k -> Dynamic (bridged).
    if (!args[1].constant.isKnown() || args[1].constant.value < 0) return InferredType::dynamic();
    const std::size_t r = static_cast<std::size_t>(args[1].constant.value) % 4;
    return (r == 0 || r == 2) ? same() : swapped();
}

// kron(A,B): Kronecker product. A m x n, B p x q -> (m*p) x (n*q). Both operands must be
// 2-D matrices (KnownDims rank-2 or runtime-dim NDims rank-2), DOUBLE; the result is a
// runtime-dim 2-D (ndShape({0,0})). Vector / scalar / non-DOUBLE operands -> Dynamic
// (bridged); kron ships in the linalg toolbox, so the bridge handles those.
InferredType kronTransfer(const std::vector<ArgInfo> &args)
{
    if (args.size() != 2 || !args[0].type.isConcrete() || !args[1].type.isConcrete())
        return InferredType::dynamic();
    auto isMat = [](const Shape &s) {
        return (s.kind == ShapeKind::KnownDims && s.rows > 1 && s.cols > 1)
               || (s.kind == ShapeKind::NDims && s.nd.size() == 2);
    };
    if (isMat(args[0].type.shape) && isMat(args[1].type.shape)
        && args[0].type.dtype == ValueType::DOUBLE && args[1].type.dtype == ValueType::DOUBLE)
        return InferredType::concrete(ValueType::DOUBLE, Shape::ndShape({0, 0}));
    return InferredType::dynamic();
}

// cat(dim, A, B): concatenate along dim -- dim==1 vertical (like [A;B]), dim==2 horizontal
// (like [A B]). v1: a LITERAL dim 1|2 and exactly two 2-D DOUBLE matrix operands -> a
// runtime-dim 2-D result (ndShape({0,0})). A runtime dim / >2 operands / vector / non-DOUBLE
// -> Dynamic (bridged).
InferredType catTransfer(const std::vector<ArgInfo> &args)
{
    if (args.size() < 3) return InferredType::dynamic();  // dim + >=2 arrays
    std::size_t dim = 0;
    if (!args[0].constant.asDim(dim) || dim < 1 || dim > 4) return InferredType::dynamic();
    auto isMat = [](const Shape &s) {
        return (s.kind == ShapeKind::KnownDims && s.rows > 1 && s.cols > 1)
               || (s.kind == ShapeKind::NDims && s.nd.size() == 2);
    };
    // dim 1/2 -> a rank-2 result (vert/horz concat) of two 2-D matrices. dim 3 -> a rank-3
    // result, dim 4 -> a rank-4 result: stack/append along the (new) trailing dim (a contiguous
    // buffer append). dim 3 accepts 2-D OR rank-3 operands; dim 4 accepts rank-3 OR rank-4.
    auto okDim3 = [](const Shape &s) {
        return (s.kind == ShapeKind::KnownDims && s.rows > 1 && s.cols > 1)
               || (s.kind == ShapeKind::NDims && (s.nd.size() == 2 || s.nd.size() == 3));
    };
    auto okDim4 = [](const Shape &s) {
        return s.kind == ShapeKind::NDims && (s.nd.size() == 3 || s.nd.size() == 4);
    };
    // N-operand (>2) cat is supported for dim 3 AND dim 4 -- the trailing-dim contiguous
    // append (M = op0 slabs ++ op1 slabs ++ ...), which generalizes cleanly to any operand
    // count. dim 3: every operand 2-D or rank-3 DOUBLE -> a rank-3 result; dim 4: every operand
    // rank-3 or rank-4 DOUBLE -> a rank-4 result. dim 1/2 with >2 operands stay Dynamic
    // (bridged): the [A B C]/[A;B;C] bracket forms already cover those.
    if (args.size() > 3) {
        if (dim != 3 && dim != 4) return InferredType::dynamic();
        for (std::size_t i = 1; i < args.size(); ++i) {
            const bool shapeOk = dim == 3 ? okDim3(args[i].type.shape) : okDim4(args[i].type.shape);
            if (!(args[i].type.isConcrete() && args[i].type.dtype == ValueType::DOUBLE && shapeOk))
                return InferredType::dynamic();
        }
        return InferredType::concrete(
            ValueType::DOUBLE, Shape::ndShape(std::vector<std::size_t>(dim == 4 ? 4 : 3, 0)));
    }
    const bool bothOk = dim == 4 ? (okDim4(args[1].type.shape) && okDim4(args[2].type.shape))
                        : dim == 3 ? (okDim3(args[1].type.shape) && okDim3(args[2].type.shape))
                                   : (isMat(args[1].type.shape) && isMat(args[2].type.shape));
    if (args[1].type.isConcrete() && args[2].type.isConcrete() && bothOk
        && args[1].type.dtype == ValueType::DOUBLE && args[2].type.dtype == ValueType::DOUBLE)
        return InferredType::concrete(
            ValueType::DOUBLE, Shape::ndShape(std::vector<std::size_t>(dim == 4 ? 4 : dim == 3 ? 3
                                                                                              : 2,
                                                                       0)));
    return InferredType::dynamic();
}

// permute(A, perm): reorder A's dimensions per the permutation vector. The result has the SAME
// RANK as A, with its dims permuted -- all runtime (the actual permuted dims need perm's literal
// VALUES, which the emit reads; the transfer only fixes the rank). So a rank-r A -> an NDims of
// rank r with all-runtime dims (ndShape of r zeros). v1: a DOUBLE A that is NDims (rank>=2) or a
// KnownDims 2-D (rank 2), and a vector perm. A scalar/1-D A, a non-vector perm, or a non-DOUBLE
// operand -> Dynamic (bridged). (A LITERAL perm is enforced at emit; a runtime perm -> refusal.)
InferredType permuteTransfer(const std::vector<ArgInfo> &args)
{
    if (args.size() != 2 || !args[0].type.isConcrete() || !args[1].type.isConcrete())
        return InferredType::dynamic();
    if (args[0].type.dtype != ValueType::DOUBLE) return InferredType::dynamic();
    std::size_t rank = 0;
    if (args[0].type.shape.kind == ShapeKind::NDims) rank = args[0].type.shape.nd.size();
    else if (args[0].type.shape.kind == ShapeKind::KnownDims) rank = 2;
    else return InferredType::dynamic();  // scalar / vector -> permute is ~no-op, bridge
    if (!args[1].type.shape.isVector1D() && args[1].type.shape.kind != ShapeKind::KnownDims)
        return InferredType::dynamic();
    return InferredType::concrete(ValueType::DOUBLE,
                                  Shape::ndShape(std::vector<std::size_t>(rank, 0)));
}

// sum / prod / mean / max / min over a VECTOR (single-arg form) -> a scalar of
// the operand's dtype. MATLAB reduces a matrix along a dim (-> a row vector) and
// max/min take a 2nd arg (elementwise) or a 2nd output (index) — only the
// vector->scalar single-output case is typed here (the common reduction); it is
// bridged to the runtime for the exact result (summation order, NaN handling).
// Matrix / N-D input, or a 2-arg call, -> Dynamic (the sound fallback).
InferredType vectorReductionTransfer(const std::vector<ArgInfo> &args)
{
    if (args.size() != 1 || !args[0].type.isConcrete()) return InferredType::dynamic();
    const Shape &s = args[0].type.shape;
    switch (s.kind) {
    case ShapeKind::Scalar:
    case ShapeKind::RowVector:
    case ShapeKind::ColVector:
    case ShapeKind::Vector1D:
        return InferredType::scalar(args[0].type.dtype);
    case ShapeKind::KnownDims:
        if ((s.rows == 1 && s.cols >= 1) || (s.cols == 1 && s.rows >= 1))
            return InferredType::scalar(args[0].type.dtype);
        return InferredType::dynamic();
    default: return InferredType::dynamic();
    }
}

// sum / prod / mean: 1-arg over a vector -> scalar; 1-arg over a 2-D matrix -> 1-D vector
// (column-wise); 2-arg sum(A, dim) with a 2-D matrix and literal dim 1|2 -> 1-D vector.
InferredType sumProdMeanTransfer(const std::vector<ArgInfo> &args)
{
    if (args.empty() || !args[0].type.isConcrete()) return InferredType::dynamic();
    const Shape &s    = args[0].type.shape;
    const bool   is2D = (s.kind == ShapeKind::KnownDims && s.rows > 1 && s.cols > 1)
                      || (s.kind == ShapeKind::NDims && s.nd.size() == 2);
    if (args.size() == 1) {
        switch (s.kind) {
        case ShapeKind::Scalar:
        case ShapeKind::RowVector:
        case ShapeKind::ColVector:
        case ShapeKind::Vector1D:
            return InferredType::scalar(args[0].type.dtype);
        case ShapeKind::KnownDims:
            if ((s.rows == 1 && s.cols >= 1) || (s.cols == 1 && s.rows >= 1))
                return InferredType::scalar(args[0].type.dtype);
            if (is2D) return InferredType::concrete(args[0].type.dtype, Shape::rowVector());
            return InferredType::dynamic();
        case ShapeKind::NDims:
            if (is2D) return InferredType::concrete(args[0].type.dtype, Shape::rowVector());
            return InferredType::dynamic();
        default: return InferredType::dynamic();
        }
    }
    std::size_t dim = 0;
    if (args.size() == 2 && is2D && args[1].constant.asDim(dim) && (dim == 1 || dim == 2))
        return InferredType::concrete(args[0].type.dtype,
                                      dim == 1 ? Shape::rowVector() : Shape::colVector());
    return InferredType::dynamic();
}

// max / min: BOTH the 1-arg vector reduction (-> scalar, as vectorReductionTransfer) AND the
// 2-arg ELEMENTWISE form max(a,b) / min(a,b). MATLAB's 2-arg max/min ignore NaN (the result
// is the non-NaN operand) -- bit-identical to std::fmax/std::fmin, so the emitter lowers them
// via binaryMathStd (scalar) and the elementwise fill (arrays). REAL DOUBLE operands only
// (complex max is by magnitude; integer/single/object differ) -> else Dynamic (bridged). The
// 2-arg result is typed concrete ONLY for equal shapes or a scalar broadcast; any other shape
// mismatch (row vs col, vector vs matrix -> MATLAB implicit expansion) -> Dynamic, so the
// runtime does the broadcast (never a wrong-shape miscompile). 3-arg max(x,[],dim) -> Dynamic.
InferredType maxMinTransfer(const std::vector<ArgInfo> &args)
{
    if (args.empty() || !args[0].type.isConcrete()) return InferredType::dynamic();
    if (args.size() == 1) {  // reduction
        switch (args[0].type.shape.kind) {
        case ShapeKind::Scalar:
        case ShapeKind::RowVector:
        case ShapeKind::ColVector:
        case ShapeKind::Vector1D:
            return InferredType::scalar(args[0].type.dtype);  // -> scalar
        case ShapeKind::KnownDims:
            if ((args[0].type.shape.rows == 1 && args[0].type.shape.cols >= 1)
                || (args[0].type.shape.cols == 1 && args[0].type.shape.rows >= 1))
                return InferredType::scalar(args[0].type.dtype);
            // a true 2-D matrix max(A) is column-wise -> a 1 x n ROW vector (runtime length).
            if (args[0].type.shape.rows > 1 && args[0].type.shape.cols > 1)
                return InferredType::concrete(args[0].type.dtype, Shape::rowVector());
            return InferredType::dynamic();
        case ShapeKind::NDims:  // runtime-dim rank-2 matrix -> column-wise -> 1-D row vector
            if (args[0].type.shape.nd.size() == 2)
                return InferredType::concrete(args[0].type.dtype, Shape::rowVector());
            return InferredType::dynamic();
        default: return InferredType::dynamic();
        }
    }
    if (args.size() == 2 && args[1].type.isConcrete()) {  // elementwise max(a,b) / min(a,b)
        if (args[0].type.dtype != ValueType::DOUBLE || args[1].type.dtype != ValueType::DOUBLE)
            return InferredType::dynamic();
        const Shape &sa = args[0].type.shape, &sb = args[1].type.shape;
        if (sa.kind == ShapeKind::Unknown || sb.kind == ShapeKind::Unknown)
            return InferredType::dynamic();
        if (sa == sb)      return InferredType::concrete(ValueType::DOUBLE, sa);
        if (sa.isScalar()) return InferredType::concrete(ValueType::DOUBLE, sb);
        if (sb.isScalar()) return InferredType::concrete(ValueType::DOUBLE, sa);
        return InferredType::dynamic();  // shape mismatch / implicit expansion -> bridged
    }
    // max(A, [], dim) with a 2-D matrix A and a LITERAL dim 1|2 -> a 1-D vector (dim 1 -> 1 x n
    // column-wise; dim 2 -> m x 1 row-wise). The middle [] arg is ignored here (max's 3-arg form is
    // always max(A,[],dim)); the emit verifies it is an empty matrix. max(x,[],'all') / a non-
    // literal dim -> Dynamic/bridged.
    if (args.size() == 3 && args[0].type.dtype == ValueType::DOUBLE) {
        const Shape &s     = args[0].type.shape;
        const bool   is2D  = (s.kind == ShapeKind::KnownDims && s.rows > 1 && s.cols > 1)
                          || (s.kind == ShapeKind::NDims && s.nd.size() == 2);
        const bool   isND3 = s.kind == ShapeKind::NDims && s.nd.size() == 3;
        std::size_t  dim = 0;
        if (is2D && args[2].constant.asDim(dim) && (dim == 1 || dim == 2))
            return InferredType::concrete(args[0].type.dtype,
                                          dim == 1 ? Shape::rowVector() : Shape::colVector());
        // rank-3 reduction: dim 1|2 keep the (non-trailing) singleton -> rank-3; dim 3 removes the
        // trailing dim entirely -> rank-2. Both output ranks are deterministic, so sound.
        if (isND3 && args[2].constant.asDim(dim)) {
            if (dim == 1 || dim == 2)
                return InferredType::concrete(args[0].type.dtype, Shape::ndShape({0, 0, 0}));
            if (dim == 3) return InferredType::concrete(args[0].type.dtype, Shape::ndShape({0, 0}));
        }
    }
    return InferredType::dynamic();  // other 3+ arg forms (max(x,[],"all"), …) -> bridged
}

// any / all: a VECTOR -> a LOGICAL scalar; a 2-D matrix -> a LOGICAL row vector (column-wise). The
// 2-arg any(A, dim) form with a 2-D matrix A and a LITERAL dim 1|2 -> a LOGICAL 1-D vector (dim 1
// -> 1 x n; dim 2 -> m x 1). The emitter lowers all to native inline loops. N-D / non-literal dim
// -> Dynamic.
InferredType anyAllTransfer(const std::vector<ArgInfo> &args)
{
    if (args.empty() || !args[0].type.isConcrete()) return InferredType::dynamic();
    const Shape &s    = args[0].type.shape;
    const bool   is2D = (s.kind == ShapeKind::KnownDims && s.rows > 1 && s.cols > 1)
                      || (s.kind == ShapeKind::NDims && s.nd.size() == 2);
    if (args.size() == 1) {
        switch (s.kind) {
        case ShapeKind::Scalar:
        case ShapeKind::RowVector:
        case ShapeKind::ColVector:
        case ShapeKind::Vector1D:
            return InferredType::scalar(ValueType::LOGICAL);  // -> scalar
        default:
            if ((s.kind == ShapeKind::KnownDims)
                && ((s.rows == 1 && s.cols >= 1) || (s.cols == 1 && s.rows >= 1)))
                return InferredType::scalar(ValueType::LOGICAL);
            if (is2D) return InferredType::concrete(ValueType::LOGICAL, Shape::rowVector());  // col-wise
            return InferredType::dynamic();
        }
    }
    // any(A, dim) / all(A, dim): a 2-D matrix + a literal dim 1|2 -> a LOGICAL 1-D vector.
    std::size_t dim = 0;
    if (args.size() == 2 && is2D && args[1].constant.asDim(dim) && (dim == 1 || dim == 2))
        return InferredType::concrete(ValueType::LOGICAL,
                                      dim == 1 ? Shape::rowVector() : Shape::colVector());
    // rank-3 reduction: dim 1|2 keep the (non-trailing) singleton -> a LOGICAL rank-3; dim 3 removes
    // the trailing dim entirely -> a LOGICAL rank-2. Both output ranks are deterministic, so sound.
    const bool isND3 = s.kind == ShapeKind::NDims && s.nd.size() == 3;
    if (args.size() == 2 && isND3 && args[1].constant.asDim(dim)) {
        if (dim == 1 || dim == 2)
            return InferredType::concrete(ValueType::LOGICAL, Shape::ndShape({0, 0, 0}));
        if (dim == 3) return InferredType::concrete(ValueType::LOGICAL, Shape::ndShape({0, 0}));
    }
    return InferredType::dynamic();
}

// find(x) -> a 1-D DOUBLE vector of the 1-based positions of the nonzero
// elements. The length is runtime (-> Vector1D shape, still a buffer type); the
// emitter lowers it to a native filter loop.
InferredType findTransfer(const std::vector<ArgInfo> &args)
{
    if (args.size() != 1 || !args[0].type.isConcrete()) return InferredType::dynamic();
    switch (args[0].type.shape.kind) {
    case ShapeKind::Scalar:
    case ShapeKind::RowVector:
    case ShapeKind::ColVector:
    case ShapeKind::Vector1D:
    case ShapeKind::KnownDims:
    case ShapeKind::NDims:
        return InferredType::concrete(ValueType::DOUBLE, Shape::vector1D());
    case ShapeKind::Unknown:
        return InferredType::dynamic();
    }
    return InferredType::dynamic();
}

// diff(x) -> consecutive differences, a 1-D vector of length n-1 (runtime ->
// RowVector / ColVector / Vector1D) preserving the operand dtype.
InferredType diffTransfer(const std::vector<ArgInfo> &args)
{
    if (args.empty() || !args[0].type.isConcrete()) return InferredType::dynamic();
    const Shape &s     = args[0].type.shape;
    const bool   is2D  = (s.kind == ShapeKind::KnownDims && s.rows > 1 && s.cols > 1)
                      || (s.kind == ShapeKind::NDims && s.nd.size() == 2);
    if (args.size() == 1) {
        switch (s.kind) {
        case ShapeKind::RowVector:
            return InferredType::concrete(args[0].type.dtype, Shape::rowVector());
        case ShapeKind::ColVector:
            return InferredType::concrete(args[0].type.dtype, Shape::colVector());
        case ShapeKind::Vector1D:
            return InferredType::concrete(args[0].type.dtype, Shape::vector1D());
        default:
            if (s.kind == ShapeKind::KnownDims && s.rows == 1 && s.cols > 1)
                return InferredType::concrete(args[0].type.dtype, Shape::rowVector());
            if (s.kind == ShapeKind::KnownDims && s.cols == 1 && s.rows > 1)
                return InferredType::concrete(args[0].type.dtype, Shape::colVector());
            if (is2D) return InferredType::concrete(args[0].type.dtype, Shape::ndShape({0, 0}));
            return InferredType::dynamic();
        }
    }
    const bool   isND3 = s.kind == ShapeKind::NDims && s.nd.size() == 3;
    std::size_t  order = 0, dim = 0;
    if (args.size() == 3 && args[1].constant.asDim(order) && order == 1
        && args[2].constant.asDim(dim)) {
        if (is2D && (dim == 1 || dim == 2))
            return InferredType::concrete(args[0].type.dtype, Shape::ndShape({0, 0}));
        if (isND3 && (dim == 1 || dim == 2))
            return InferredType::concrete(args[0].type.dtype, Shape::ndShape({0, 0, 0}));
    }
    return InferredType::dynamic();
}

// dot(a,b) -> scalar inner product (sum of a.*b for real). v1: two 1-D real
// DOUBLE vectors -> a DOUBLE scalar; the emitter lowers it to an accumulation
// loop. Anything else (complex needs conj, matrix, wrong arity) -> Dynamic.
InferredType dotTransfer(const std::vector<ArgInfo> &args)
{
    if (args.size() != 2 || !args[0].type.isConcrete() || !args[1].type.isConcrete())
        return InferredType::dynamic();
    if (args[0].type.dtype != ValueType::DOUBLE || args[1].type.dtype != ValueType::DOUBLE)
        return InferredType::dynamic();
    auto isVec = [](const Shape &s) {
        return s.isScalar() || s.isVector1D();
    };
    if (isVec(args[0].type.shape) && isVec(args[1].type.shape))
        return InferredType::scalar(ValueType::DOUBLE);
    return InferredType::dynamic();
}

// strcmp(a,b) -> a LOGICAL scalar (true iff the two arrays are identical: same
// length AND elementwise equal). v1: two concrete array args -> LOGICAL scalar;
// the emitter lowers it to a length + elementwise compare.
InferredType strcmpTransfer(const std::vector<ArgInfo> &args)
{
    if (args.size() != 2 || !args[0].type.isConcrete() || !args[1].type.isConcrete())
        return InferredType::dynamic();
    if (args[0].type.shape.kind == ShapeKind::Unknown
        || args[1].type.shape.kind == ShapeKind::Unknown)
        return InferredType::dynamic();
    return InferredType::scalar(ValueType::LOGICAL);
}

// Single-arg type/shape query predicates (isempty / isscalar / isreal): always a
// LOGICAL scalar for a concrete arg.
InferredType logicalQueryTransfer(const std::vector<ArgInfo> &args)
{
    if (args.size() != 1 || !args[0].type.isConcrete()) return InferredType::dynamic();
    if (args[0].type.shape.kind == ShapeKind::Unknown) return InferredType::dynamic();
    return InferredType::scalar(ValueType::LOGICAL);
}

// sort(x) / sort(x, 'ascend'|'descend') -> same shape AND dtype as x.
InferredType sortTransfer(const std::vector<ArgInfo> &args)
{
    if (args.empty() || args.size() > 3 || !args[0].type.isConcrete()
        || args[0].type.shape.kind == ShapeKind::Unknown)
        return InferredType::dynamic();
    if (args.size() == 1) return args[0].type;  // sort(A): shape-preserving
    if (args.size() == 2 && args[1].type.dtype == ValueType::CHAR) return args[0].type;
    const bool  isND3 = args[0].type.shape.kind == ShapeKind::NDims
                     && args[0].type.shape.nd.size() == 3;
    std::size_t dim   = 0;
    const bool  okDim = isND3 && args[1].constant.asDim(dim) && (dim == 1 || dim == 2 || dim == 3);
    if (args.size() == 2 && okDim) return args[0].type;  // sort(A, dim)
    if (args.size() == 3 && okDim && args[2].type.dtype == ValueType::CHAR)
        return args[0].type;  // sort(A, dim, dir)
    return InferredType::dynamic();
}

// polyval(p, x) -> the polynomial with coefficients p (highest degree first)
// evaluated at each point of x, a DOUBLE result with the SAME shape as x.
InferredType polyvalTransfer(const std::vector<ArgInfo> &args)
{
    if (args.size() != 2 || !args[0].type.isConcrete() || !args[1].type.isConcrete())
        return InferredType::dynamic();
    if (!args[0].type.shape.isVector1D() && !args[0].type.shape.isScalar())
        return InferredType::dynamic();
    if (args[1].type.shape.isVector1D())
        return InferredType::concrete(ValueType::DOUBLE, args[1].type.shape);
    return InferredType::dynamic();  // scalar / matrix x -> deferred
}

// cross(a, b): the 3-D vector cross product -> a 3-element vector (1-D DOUBLE).
InferredType crossTransfer(const std::vector<ArgInfo> &args)
{
    if (args.size() != 2 || !args[0].type.isConcrete() || !args[1].type.isConcrete())
        return InferredType::dynamic();
    if (args[0].type.dtype != ValueType::DOUBLE || args[1].type.dtype != ValueType::DOUBLE)
        return InferredType::dynamic();
    if (args[0].type.shape.isVector1D() && args[1].type.shape.isVector1D())
        return InferredType::concrete(ValueType::DOUBLE, Shape::vector1D());
    return InferredType::dynamic();
}

// trace(A): the sum of the diagonal of a 2-D matrix -> a DOUBLE scalar.
InferredType traceTransfer(const std::vector<ArgInfo> &args)
{
    if (args.size() != 1 || !args[0].type.isConcrete()) return InferredType::dynamic();
    if (args[0].type.shape.kind == ShapeKind::KnownDims) return InferredType::scalar(ValueType::DOUBLE);
    return InferredType::dynamic();
}

// diag(A): the diagonal of a 2-D matrix as a 1-D vector (ColVector), or diag(v[,k]) -> N x N matrix.
InferredType diagTransfer(const std::vector<ArgInfo> &args)
{
    if (args.empty() || args.size() > 2 || !args[0].type.isConcrete())
        return InferredType::dynamic();
    if (args[0].type.shape.kind == ShapeKind::KnownDims)
        return args.size() == 1 ? InferredType::concrete(args[0].type.dtype, Shape::colVector())
                                : InferredType::dynamic();
    switch (args[0].type.shape.kind) {
    case ShapeKind::RowVector:
    case ShapeKind::ColVector:
    case ShapeKind::Vector1D:
        if (args.size() == 2 && !args[1].constant.isKnown()) return InferredType::dynamic();
        return InferredType::concrete(args[0].type.dtype, Shape::ndShape({0, 0}));
    default:
        return InferredType::dynamic();
    }
}

// unique(x) -> the sorted distinct values, a 1-D vector of runtime length (-> Vector1D
// shape) preserving the operand dtype.
InferredType uniqueTransfer(const std::vector<ArgInfo> &args)
{
    if (args.size() != 1 || !args[0].type.isConcrete()) return InferredType::dynamic();
    switch (args[0].type.shape.kind) {
    case ShapeKind::Scalar:
    case ShapeKind::RowVector:
    case ShapeKind::ColVector:
    case ShapeKind::Vector1D:
        return InferredType::concrete(args[0].type.dtype, Shape::vector1D());
    case ShapeKind::KnownDims:
        if (args[0].type.shape.rows >= 1 && args[0].type.shape.cols >= 1)
            return InferredType::concrete(args[0].type.dtype, Shape::vector1D());
        return InferredType::dynamic();
    case ShapeKind::NDims:
        if (args[0].type.shape.nd.size() == 2)
            return InferredType::concrete(args[0].type.dtype, Shape::vector1D());
        return InferredType::dynamic();
    default: return InferredType::dynamic();
    }
}

// circshift(x, k) -> same shape AND dtype as x.
InferredType circshiftTransfer(const std::vector<ArgInfo> &args)
{
    if ((args.size() != 2 && args.size() != 3) || !args[0].type.isConcrete()
        || args[0].type.shape.kind == ShapeKind::Unknown)
        return InferredType::dynamic();
    if (args[1].type.isConcrete() && args[1].type.shape.kind != ShapeKind::Scalar)
        return InferredType::dynamic();
    if (args.size() == 3) {
        if (!args[2].constant.isKnown()) return InferredType::dynamic();
        const double d     = args[2].constant.value;
        const bool   isND3 = args[0].type.shape.kind == ShapeKind::NDims
                          && args[0].type.shape.nd.size() == 3;
        if (!(d == 1.0 || d == 2.0 || (d == 3.0 && isND3)))
            return InferredType::dynamic();
    }
    return args[0].type;
}

InferredType identityShapeTransfer(const std::vector<ArgInfo> &args)
{
    if (args.size() != 1 || !args[0].type.isConcrete()
        || args[0].type.shape.kind == ShapeKind::Unknown)
        return InferredType::dynamic();
    return args[0].type;
}

InferredType trilTriuTransfer(const std::vector<ArgInfo> &args)
{
    if (args.empty() || args.size() > 2 || !args[0].type.isConcrete()
        || args[0].type.shape.kind == ShapeKind::Unknown)
        return InferredType::dynamic();
    return args[0].type;
}

InferredType identityShape1or2Transfer(const std::vector<ArgInfo> &args)
{
    if (args.empty() || args.size() > 2 || !args[0].type.isConcrete()
        || args[0].type.shape.kind == ShapeKind::Unknown)
        return InferredType::dynamic();
    return args[0].type;
}

InferredType num2strTransfer(const std::vector<ArgInfo> &args)
{
    if (args.size() != 1 || !args[0].type.isConcrete()) return InferredType::dynamic();
    if (args[0].type.shape.isScalar() || args[0].type.shape.isVector1D())
        return InferredType::concrete(ValueType::CHAR, Shape::rowVector());
    return InferredType::dynamic();
}

InferredType sprintfTransfer(const std::vector<ArgInfo> &args)
{
    if (args.empty() || !args[0].type.isConcrete()
        || args[0].type.shape.kind == ShapeKind::Unknown)
        return InferredType::dynamic();
    if (args[0].type.dtype != ValueType::CHAR) return InferredType::dynamic();
    for (const auto &a : args)
        if (a.type.isConcrete() && a.type.shape.kind == ShapeKind::Unknown)
            return InferredType::dynamic();
    return InferredType::concrete(ValueType::CHAR, Shape::rowVector());
}

} // namespace

void registerShapeTransfers(TransferRegistry &reg)
{
    reg.add("numel", countTransfer);
    reg.add("length", countTransfer);
    reg.add("ndims", countTransfer);
    reg.add("size", sizeTransfer);
    reg.addMulti("size", sizeMultiTransfer);
    reg.addMulti("max", maxMinMultiTransfer);  // [m,i]=max(x) -> {value, 1-based index}
    reg.addMulti("min", maxMinMultiTransfer);  // [m,i]=min(x) -> {value, 1-based index}
    reg.addMulti("sort", sortMultiTransfer);   // [s,i]=sort(x) -> {sorted values, permutation}
    reg.addMulti("unique", uniqueMultiTransfer);  // [u,ia,ic]=unique(x) -> {distinct, first-idx, map}
    reg.addMulti("intersect", intersectMultiTransfer);  // [c,ia,ib]=intersect(a,b) -> {common, a-idx, b-idx}
    reg.addMulti("setdiff", setdiffMultiTransfer);  // [c,ia]=setdiff(a,b) -> {a-not-in-b, a-idx}
    reg.addMulti("union", unionMultiTransfer);  // [u,ia,ib]=union(a,b) -> {distinct, a-idx, b-idx}
    reg.add("transpose", transposeTransfer);   // A.'
    reg.add("ctranspose", transposeTransfer);  // A'
    reg.add("rot90", rot90Transfer);           // rot90(A) -> 90deg rotation (dims swap)
    reg.add("kron", kronTransfer);             // kron(A,B) -> Kronecker product (mp x nq)
    reg.add("cat", catTransfer);               // cat(dim,A,B) literal dim -> vert/horz concat
    reg.add("permute", permuteTransfer);        // permute(A,perm) -> A's dims reordered (same rank)
    // Vector reductions -> scalar (bridged; the emitter boxes the array arg).
    for (const char *n : {"sum", "prod", "mean"})
        reg.add(n, sumProdMeanTransfer);
    for (const char *n : {"norm", "std", "var", "trapz"})
        reg.add(n, vectorReductionTransfer);
    // vector -> scalar; 2-D matrix -> column-wise row vector (both native, sort-based).
    reg.add("median", statColumnReduceTransfer);
    reg.add("mode", statColumnReduceTransfer);
    // max/min: 1-arg reduction (-> scalar) OR 2-arg elementwise max(a,b)/min(a,b) (native
    // via fmax/fmin). The single-output transfer; [m,i]=max(x) uses maxMinMultiTransfer.
    reg.add("max", maxMinTransfer);
    reg.add("min", maxMinTransfer);
    for (const char *n : {"any", "all"})  // vector -> LOGICAL scalar (native inline loop)
        reg.add(n, anyAllTransfer);
    reg.add("find", findTransfer);  // vector -> 1-D DOUBLE positions (native filter loop)
    reg.add("diff", diffTransfer);  // vector -> 1-D differences, length n-1 (native loop)
    reg.add("dot", dotTransfer);    // (vec, vec) -> DOUBLE scalar inner product (native loop)
    reg.add("strcmp", strcmpTransfer);  // (arr, arr) -> LOGICAL scalar string equality (native)
    reg.add("circshift", circshiftTransfer);  // (x, k) -> same shape (native circular shift)
    reg.add("diag", diagTransfer);  // diag(A) -> the diagonal of a matrix as a 1-D vector
    reg.add("trace", traceTransfer);  // trace(A) -> sum of the diagonal (DOUBLE scalar)
    reg.add("cross", crossTransfer);  // cross(a,b) -> the 3-D cross product (1-D, len 3)
    reg.add("unique", uniqueTransfer);  // x -> sorted distinct 1-D (native sort + dedup)
    reg.add("union", setopBinaryTransfer);      // union(a,b) -> sorted distinct of [a b]
    reg.add("intersect", setopBinaryTransfer);  // intersect(a,b) -> sorted distinct common values
    reg.add("setdiff", setopBinaryTransfer);    // setdiff(a,b) -> sorted distinct of a not in b
    reg.add("ismember", ismemberTransfer);      // tf=ismember(a,b) -> LOGICAL mask (a in b)
    reg.add("sortrows", sortrowsTransfer);      // sortrows(A) -> rows sorted lexicographically
    reg.add("polyval", polyvalTransfer);  // (p, x) -> p evaluated at x, shape of x (Horner)
    // x -> LOGICAL scalar query predicates (native: compile-time / numel / orientation).
    for (const char *n : {"isempty", "isscalar", "isreal", "isrow", "iscolumn", "isvector"})
        reg.add(n, logicalQueryTransfer);
    // dtype-classification predicates (compile-time constants from the static dtype).
    for (const char *n : {"isnumeric", "isfloat", "isinteger", "ischar", "islogical"})
        reg.add(n, logicalQueryTransfer);
    // Shape-preserving array->array builtins (bridged via the array-result path).
    for (const char *n : {"fliplr", "flipud", "gradient"})
        reg.add(n, identityShapeTransfer);  // strict 1-arg
    // 1-or-2-arg identity shape (the optional 2nd arg is a dim): flip / cumsum / cumprod / cummax /
    // cummin.
    for (const char *n : {"flip", "cumsum", "cumprod", "cummax", "cummin"})
        reg.add(n, identityShape1or2Transfer);
    reg.add("tril", trilTriuTransfer);  // tril(A[, k]) -- same shape, optional diag offset
    reg.add("triu", trilTriuTransfer);  // triu(A[, k])
    for (const char *n : {"upper", "lower"})  // char case transform (native, same shape)
        reg.add(n, identityShapeTransfer);
    reg.add("sort", sortTransfer);  // sort(x[, 'ascend'|'descend']) -> same shape (native std::sort)
    reg.add("num2str", num2strTransfer);  // num2str(scalar/vec) -> CHAR row (bridged; G3)
    reg.add("sprintf", sprintfTransfer);  // sprintf(fmt, args) -> CHAR row (bridged; G3b)
}

} // namespace numkit::codegen
