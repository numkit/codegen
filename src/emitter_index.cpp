// codegen/src/emitter_index.cpp — array index read & write lowering.

#include "emitter_internal.hpp"

namespace numkit::codegen::detail {

std::string Emitter::emitIndexRead(const std::string &base, const ASTNode &call)
{
    const ArrayInfo  &ai  = arrays_.at(base);
    const std::string ptr = ai.dataExpr;  // `base` (buffer) or `base.data()` (local vec)

    // brick 6: inside a promoted clean-index loop, A(counter) is provably
    // in-bounds -> direct 0-based access, no check, no -1.
    if (promotedCounter_ && call.children.size() == 2
        && call.children[1]->type == NodeType::IDENTIFIER
        && call.children[1]->strValue == *promotedCounter_)
        return ptr + "[" + cppIdent(*promotedCounter_) + "]";

    // Rank-N (N>=3) read A(i,j,k,...) -> column-major numkit_rt::indexN.
    if (ai.isND) {
        if (call.children.size() - 1 != ai.ndDims.size())
            unsupported("N-D index arity for '" + base + "' (expected "
                        + std::to_string(ai.ndDims.size()) + " subscripts)");
        std::string dims, subs;
        for (std::size_t k = 0; k < ai.ndDims.size(); ++k) dims += (k ? ", " : "") + ai.ndDims[k];
        for (std::size_t i = 1; i < call.children.size(); ++i) {
            endStack_.push_back(ai.ndDims[i - 1]);
            subs += (i > 1 ? ", " : "") + emitExpr(*call.children[i]);
            endStack_.pop_back();
        }
        return "numkit_rt::indexN(" + ptr + ", {" + dims + "}, {" + subs + "})";
    }

    std::vector<AbstractValue> idx;
    for (std::size_t i = 1; i < call.children.size(); ++i)
        idx.push_back(inferExpr(*call.children[i], types_, reg_, classes_));

    const IndexPlan plan = planIndexRead(arrayValue(base), idx);
    if (plan.form == IndexForm::Subscript2D) {
        if (!ai.is2D) unsupported("A(i,j) on a non-matrix '" + base + "'");
        endStack_.push_back(ai.rowsVar);
        const std::string sub1 = emitExpr(*call.children[1]);
        endStack_.pop_back();
        endStack_.push_back(ai.colsVar);
        const std::string sub2 = emitExpr(*call.children[2]);
        endStack_.pop_back();
        return "numkit_rt::index2(" + ptr + ", " + ai.rowsVar + ", " + ai.colsVar + ", "
               + sub1 + ", " + sub2 + ")";
    }
    if (plan.form != IndexForm::LinearScalar || ai.is2D)
        unsupported("index read form for '" + base
                    + "' (a 1-D scalar index, or A(i,j) on a matrix)");

    // `end` inside this 1-D scalar index resolves to the array's length; push it so
    // x(end) / x(end-1) / x(end/2) lower correctly, then pop after the arg is emitted.
    endStack_.push_back(ai.lenVar);
    const std::string idxExpr = emitExpr(*call.children[1]);
    endStack_.pop_back();
    if (plan.boundsChecked)
        return "numkit_rt::index(" + ptr + ", " + ai.lenVar + ", " + idxExpr + ")";
    return ptr + "[static_cast<std::size_t>(" + idxExpr + ") - 1]";  // proven in-bounds
}

void Emitter::emitIndexWrite(const ASTNode &lhsCall, const ASTNode &rhs)
{
    const std::string base = resolveArrayVar(*lhsCall.children[0]);
    if (base.empty())
        unsupported("indexed write to non-array '"
                    + (lhsCall.children[0]->type == NodeType::IDENTIFIER
                           ? lhsCall.children[0]->strValue
                           : std::string("<expr>"))
                    + "'");
    const ArrayInfo  &ai  = arrays_.at(base);
    const std::string ptr = ai.dataExpr;  // `base` (buffer) or `base.data()` (local vec)

    // brick 6: inside a promoted clean-index loop, A(counter) is provably
    // in-bounds -> direct 0-based store, no check.
    if (promotedCounter_ && lhsCall.children.size() == 2
        && lhsCall.children[1]->type == NodeType::IDENTIFIER
        && lhsCall.children[1]->strValue == *promotedCounter_) {
        line(ptr + "[" + cppIdent(*promotedCounter_) + "] = " + emitExpr(rhs) + ";");
        return;
    }

    // Whole-array FILL: `A(:) = rhs` over the flat column-major buffer (the bare colon is an
    // empty COLON_EXPR). A must be writable (local/output). Two rhs forms: a SCALAR s
    // (broadcast to every element) or a MATCHING-NUMEL array b (copied column-major,
    // preserving A's shape -- MATLAB's A(:) = b). numel = 1-D length / 2-D rows*cols / N-D
    // product. Placed before the N-D write branch so a runtime-dim 2-D A(:) = ... is filled
    // rather than rejected on subscript arity. v1: scalar rhs, or a DOUBLE array VAR rhs (an
    // array EXPRESSION rhs is deferred -> falls through to a clean refusal).
    if (lhsCall.children.size() == 2 && lhsCall.children[1]->type == NodeType::COLON_EXPR
        && lhsCall.children[1]->children.empty()) {
        if (!ai.isLocal && !ai.isOutput)
            unsupported("whole-array fill of a read-only parameter '" + base + "'");
        const AbstractValue rhsAV = inferExpr(rhs, types_, reg_, classes_);
        // Element count of any ranked/runtime array over its flat column-major buffer.
        const auto arrayNumel = [](const ArrayInfo &x) -> std::string {
            if (x.isND) {
                std::string s = x.ndDims[0];
                for (std::size_t i = 1; i < x.ndDims.size(); ++i) s += " * " + x.ndDims[i];
                return s;
            }
            if (x.is2D) return x.rowsVar + " * " + x.colsVar;
            return x.lenVar;
        };
        if (rhsAV.type.isConcrete() && rhsAV.type.shape.isScalar()) {
            const std::string numel = arrayNumel(ai);
            line("{");
            ++indent_;
            line(cppScalarType(ai.dtype) + " _numkit_fv = " + emitExpr(rhs) + ";");
            open("for (std::size_t _numkit_i = 0; _numkit_i < (" + numel + "); ++_numkit_i)");
            line(ptr + "[_numkit_i] = _numkit_fv;");
            close();
            --indent_;
            line("}");
            return;
        }
        if (rhsAV.type.isConcrete() && !rhsAV.type.shape.isScalar()
            && ai.dtype == ValueType::DOUBLE
            && rhs.type == NodeType::IDENTIFIER && isArrayVar(rhs.strValue)
            && arrays_.at(rhs.strValue).dtype == ValueType::DOUBLE) {
            const ArrayInfo &b = arrays_.at(rhs.strValue);
            line("{");
            ++indent_;
            line("const std::size_t _numkit_nA = (" + arrayNumel(ai) + ");");
            line("const std::size_t _numkit_nB = (" + arrayNumel(b) + ");");
            line("if (_numkit_nA != _numkit_nB)");
            line("    throw std::runtime_error(\"numkit: A(:) assignment element-count mismatch\");");
            open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_nA; ++_numkit_i)");
            line(ptr + "[_numkit_i] = " + b.dataExpr + "[_numkit_i];");
            close();
            --indent_;
            line("}");
            return;
        }
        // A(:) = <elementwise expr>: a pure-elementwise DOUBLE expression over arrays
        // (A(:)=A*2, A(:)=b+c, A(:)=A+1, ...) -> FUSE into the flat fill, A[i] = <expr at i>.
        // Sound: A(:) fills in place; an elementwise expr is per-element, so even a self-ref
        // (A[i]=f(A[i])) is fine, and a non-elementwise rhs (sort(A)) is rejected by collect-
        // Elementwise -> bridged. Each array operand's numel is runtime-checked == A's (MATLAB
        // errors on a mismatch). v1: A + every operand DOUBLE; reached only for a compound expr
        // (a bare array var took the arm above, a scalar the first arm).
        {
            std::set<std::string> srcArrays;
            const bool            pureEw = collectElementwise(rhs, srcArrays);
            bool                  allDouble = ai.dtype == ValueType::DOUBLE && !srcArrays.empty();
            for (const auto &s : srcArrays)
                if (arrays_.at(s).dtype != ValueType::DOUBLE) allDouble = false;
            if (pureEw && allDouble && rhsAV.type.isConcrete() && !rhsAV.type.shape.isScalar()) {
                line("{");
                ++indent_;
                line("const std::size_t _numkit_nA = (" + arrayNumel(ai) + ");");
                for (const auto &s : srcArrays) {
                    line("if ((" + arrayNumel(arrays_.at(s)) + ") != _numkit_nA)");
                    line("    throw std::runtime_error("
                         "\"numkit: A(:) = expr element-count mismatch\");");
                }
                elementCtx_ = "_numkit_i";  // whole arrays in the expr -> arr[_numkit_i]
                const std::string e = emitExpr(rhs);
                elementCtx_.clear();
                open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_nA; ++_numkit_i)");
                line(ptr + "[_numkit_i] = " + e + ";");
                close();
                --indent_;
                line("}");
                return;
            }
        }
    }

    // Helper: extract a 1-D vector of scalar ASTNodes from a MATRIX_LITERAL
    // (`[a, b, c]` or `[a; b; c]`), or a 2-D grid of scalar ASTNodes in
    // column-major order (`[a b; c d]`).
    auto extractScalarVecLiteral = [&](const ASTNode &lit,
                                       std::vector<const ASTNode *> &out) -> bool {
        out.clear();
        if (lit.type != NodeType::MATRIX_LITERAL || lit.children.empty()) return false;
        if (lit.children.size() == 1 && lit.children[0] && !lit.children[0]->children.empty()) {
            for (const auto &el : lit.children[0]->children) {
                if (!el) return false;
                const AbstractValue ev = inferExpr(*el, types_, reg_, classes_);
                if (!ev.type.isConcrete() || !ev.type.shape.isScalar()) return false;
                out.push_back(el.get());
            }
            return !out.empty();
        }
        if (lit.children.size() > 1) {
            for (const auto &rowN : lit.children) {
                if (!rowN || rowN->children.size() != 1 || !rowN->children[0]) return false;
                const AbstractValue ev = inferExpr(*rowN->children[0], types_, reg_, classes_);
                if (!ev.type.isConcrete() || !ev.type.shape.isScalar()) return false;
                out.push_back(rowN->children[0].get());
            }
            return !out.empty();
        }
        return false;
    };
    auto extractScalarGridLiteral = [&](const ASTNode &lit, std::size_t &rowsOut,
                                        std::size_t &colsOut,
                                        std::vector<const ASTNode *> &colMajorOut) -> bool {
        colMajorOut.clear();
        rowsOut = 0;
        colsOut = 0;
        if (lit.type != NodeType::MATRIX_LITERAL || lit.children.empty() || !lit.children[0]
            || lit.children[0]->children.empty())
            return false;
        const std::size_t R = lit.children.size();
        const std::size_t C = lit.children[0]->children.size();
        for (const auto &rowN : lit.children) {
            if (!rowN || rowN->children.size() != C) return false;
            for (const auto &el : rowN->children) {
                if (!el) return false;
                const AbstractValue ev = inferExpr(*el, types_, reg_, classes_);
                if (!ev.type.isConcrete() || !ev.type.shape.isScalar()) return false;
            }
        }
        rowsOut = R;
        colsOut = C;
        colMajorOut.reserve(R * C);
        for (std::size_t c = 0; c < C; ++c)
            for (std::size_t r = 0; r < R; ++r)
                colMajorOut.push_back(lit.children[r]->children[c].get());
        return true;
    };
    std::vector<const ASTNode *> rhsVecLit;
    const bool                   hasRhsVecLit = extractScalarVecLiteral(rhs, rhsVecLit);
    std::size_t                  rhsGridRows = 0, rhsGridCols = 0;
    std::vector<const ASTNode *> rhsGridLit;
    const bool                   hasRhsGridLit =
        extractScalarGridLiteral(rhs, rhsGridRows, rhsGridCols, rhsGridLit);

    // 2-D COLUMN slice write: A(:, j) = col -> overwrite column j (1-based) of a 2-D
    // matrix with the 1-D vector col (length = rows). Works for a KnownDims 2-D OR a
    // runtime-dim 2-D (dims via the rank-agnostic dimExpr). Column-major: column j is
    // the CONTIGUOUS block A[(j-1)*rows .. +rows), so it is a straight copy. Bounds-
    // checked j, length-checked col; A must be writable (local/output). Placed before
    // the N-D write branch so it intercepts a runtime-dim 2-D target (whose other
    // colon writes that branch refuses). v1: A 2-D DOUBLE, col a distinct 1-D DOUBLE
    // array var or inline scalar vector literal, j scalar.
    if (lhsCall.children.size() == 3
        && (ai.is2D || (ai.isND && ai.ndDims.size() == 2)) && ai.dtype == ValueType::DOUBLE
        && lhsCall.children[1]->type == NodeType::COLON_EXPR
        && lhsCall.children[1]->children.empty()
        && lhsCall.children[2]->type != NodeType::COLON_EXPR
        && ((rhs.type == NodeType::IDENTIFIER && isArrayVar(rhs.strValue)
             && rhs.strValue != base && !arrays_.at(rhs.strValue).is2D
             && !arrays_.at(rhs.strValue).isND
             && arrays_.at(rhs.strValue).dtype == ValueType::DOUBLE)
            || hasRhsVecLit)
        && inferExpr(*lhsCall.children[2], types_, reg_, classes_).type.shape.isScalar()) {
        if (!ai.isLocal && !ai.isOutput)
            unsupported("column-slice write to a read-only matrix parameter '" + base + "'");
        const std::string Arows = dimExpr(ai, 0), Acols = dimExpr(ai, 1);
        endStack_.push_back(Acols);  // `end` in the column index = cols
        const std::string j = emitExpr(*lhsCall.children[2]);
        endStack_.pop_back();
        line("{");
        ++indent_;
        std::string colData, colLen;
        if (hasRhsVecLit) {
            std::string elems;
            for (std::size_t k = 0; k < rhsVecLit.size(); ++k)
                elems += (k ? ", " : "") + emitExpr(*rhsVecLit[k]);
            line("const double _numkit_lit[] = { " + elems + " };");
            colData = "_numkit_lit";
            colLen  = std::to_string(rhsVecLit.size());
        } else {
            const ArrayInfo &col = arrays_.at(rhs.strValue);
            colData = col.dataExpr;
            colLen  = col.lenVar;
        }
        line("const std::size_t _numkit_j0 = numkit_rt::check_idx1(" + j + ", " + Acols
             + ", \"numkit: column index out of bounds\");");
        line("if (" + colLen + " != " + Arows + ")");
        line("    throw std::out_of_range(\"numkit: column assignment length mismatch\");");
        line("const std::size_t _numkit_off = _numkit_j0 * " + Arows + ";");
        open("for (std::size_t _numkit_i = 0; _numkit_i < " + Arows + "; ++_numkit_i)");
        line(ptr + "[_numkit_off + _numkit_i] = " + colData + "[_numkit_i];");
        close();
        --indent_;
        line("}");
        return;
    }

    // 2-D ROW slice write: A(i, :) = row -> overwrite row i (1-based) of a 2-D matrix
    // with the 1-D vector row (length = cols). The mirror of the column write above; in
    // column-major storage row i is STRIDED: A[(i-1) + j*rows] = row[j] for j in
    // [0, cols). Works for a KnownDims 2-D OR a runtime-dim 2-D (dims via dimExpr).
    // Bounds-checked i, length-checked row; A must be writable. v1: A 2-D DOUBLE, row a
    // distinct 1-D DOUBLE array var or inline scalar vector literal, i scalar.
    if (lhsCall.children.size() == 3
        && (ai.is2D || (ai.isND && ai.ndDims.size() == 2)) && ai.dtype == ValueType::DOUBLE
        && lhsCall.children[1]->type != NodeType::COLON_EXPR
        && lhsCall.children[2]->type == NodeType::COLON_EXPR
        && lhsCall.children[2]->children.empty()
        && ((rhs.type == NodeType::IDENTIFIER && isArrayVar(rhs.strValue)
             && rhs.strValue != base && !arrays_.at(rhs.strValue).is2D
             && !arrays_.at(rhs.strValue).isND
             && arrays_.at(rhs.strValue).dtype == ValueType::DOUBLE)
            || hasRhsVecLit)
        && inferExpr(*lhsCall.children[1], types_, reg_, classes_).type.shape.isScalar()) {
        if (!ai.isLocal && !ai.isOutput)
            unsupported("row-slice write to a read-only matrix parameter '" + base + "'");
        const std::string Arows = dimExpr(ai, 0), Acols = dimExpr(ai, 1);
        endStack_.push_back(Arows);  // `end` in the row index = rows
        const std::string i = emitExpr(*lhsCall.children[1]);
        endStack_.pop_back();
        line("{");
        ++indent_;
        std::string rowData, rowLen;
        if (hasRhsVecLit) {
            std::string elems;
            for (std::size_t k = 0; k < rhsVecLit.size(); ++k)
                elems += (k ? ", " : "") + emitExpr(*rhsVecLit[k]);
            line("const double _numkit_lit[] = { " + elems + " };");
            rowData = "_numkit_lit";
            rowLen  = std::to_string(rhsVecLit.size());
        } else {
            const ArrayInfo &row = arrays_.at(rhs.strValue);
            rowData = row.dataExpr;
            rowLen  = row.lenVar;
        }
        line("const std::size_t _numkit_i0 = numkit_rt::check_idx1(" + i + ", " + Arows
             + ", \"numkit: row index out of bounds\");");
        line("if (" + rowLen + " != " + Acols + ")");
        line("    throw std::out_of_range(\"numkit: row assignment length mismatch\");");
        open("for (std::size_t _numkit_j = 0; _numkit_j < " + Acols + "; ++_numkit_j)");
        line(ptr + "[_numkit_i0 + _numkit_j * " + Arows + "] = "
             + rowData + "[_numkit_j];");
        close();
        --indent_;
        line("}");
        return;
    }

    // 2-D COLUMN scalar-fill: A(:, j) = s -> set every element of column j (1-based) to
    // the scalar s (broadcast). Sibling of the column VECTOR write above; reached only for
    // a scalar rhs (the vector branch already returned for an array-var rhs, so the two are
    // disjoint). Common idiom: A(:,j) = 0. Column-major, so column j is the contiguous
    // block A[(j-1)*rows .. +rows). Works for KnownDims or runtime-dim 2-D (dims via dimExpr).
    if (lhsCall.children.size() == 3
        && (ai.is2D || (ai.isND && ai.ndDims.size() == 2)) && ai.dtype == ValueType::DOUBLE
        && lhsCall.children[1]->type == NodeType::COLON_EXPR
        && lhsCall.children[1]->children.empty()
        && lhsCall.children[2]->type != NodeType::COLON_EXPR
        && inferExpr(rhs, types_, reg_, classes_).type.isConcrete()
        && inferExpr(rhs, types_, reg_, classes_).type.shape.isScalar()
        && inferExpr(*lhsCall.children[2], types_, reg_, classes_).type.shape.isScalar()) {
        if (!ai.isLocal && !ai.isOutput)
            unsupported("column-slice fill to a read-only matrix parameter '" + base + "'");
        const std::string Arows = dimExpr(ai, 0), Acols = dimExpr(ai, 1);
        endStack_.push_back(Acols);  // `end` in the column index = cols
        const std::string j = emitExpr(*lhsCall.children[2]);
        endStack_.pop_back();
        line("{");
        ++indent_;
        line(cppScalarType(ai.dtype) + " _numkit_fv = " + emitExpr(rhs) + ";");
        line("const std::size_t _numkit_j0 = numkit_rt::check_idx1(" + j + ", " + Acols
             + ", \"numkit: column index out of bounds\");");
        line("const std::size_t _numkit_off = _numkit_j0 * " + Arows + ";");
        open("for (std::size_t _numkit_i = 0; _numkit_i < " + Arows + "; ++_numkit_i)");
        line(ptr + "[_numkit_off + _numkit_i] = _numkit_fv;");
        close();
        --indent_;
        line("}");
        return;
    }

    // 2-D ROW scalar-fill: A(i, :) = s -> set every element of row i (1-based) to the
    // scalar s (broadcast). Mirror of the column scalar-fill; in column-major storage row i
    // is STRIDED: A[(i-1) + j*rows] for j in [0, cols). Reached only for a scalar rhs.
    if (lhsCall.children.size() == 3
        && (ai.is2D || (ai.isND && ai.ndDims.size() == 2)) && ai.dtype == ValueType::DOUBLE
        && lhsCall.children[1]->type != NodeType::COLON_EXPR
        && lhsCall.children[2]->type == NodeType::COLON_EXPR
        && lhsCall.children[2]->children.empty()
        && inferExpr(rhs, types_, reg_, classes_).type.isConcrete()
        && inferExpr(rhs, types_, reg_, classes_).type.shape.isScalar()
        && inferExpr(*lhsCall.children[1], types_, reg_, classes_).type.shape.isScalar()) {
        if (!ai.isLocal && !ai.isOutput)
            unsupported("row-slice fill to a read-only matrix parameter '" + base + "'");
        const std::string Arows = dimExpr(ai, 0), Acols = dimExpr(ai, 1);
        endStack_.push_back(Arows);  // `end` in the row index = rows
        const std::string i = emitExpr(*lhsCall.children[1]);
        endStack_.pop_back();
        line("{");
        ++indent_;
        line(cppScalarType(ai.dtype) + " _numkit_fv = " + emitExpr(rhs) + ";");
        line("const std::size_t _numkit_i0 = numkit_rt::check_idx1(" + i + ", " + Arows
             + ", \"numkit: row index out of bounds\");");
        open("for (std::size_t _numkit_j = 0; _numkit_j < " + Acols + "; ++_numkit_j)");
        line(ptr + "[_numkit_i0 + _numkit_j * " + Arows + "] = _numkit_fv;");
        close();
        --indent_;
        line("}");
        return;
    }

    // page-slice WRITE A(:,:,k) = M: write a 2-D m x n matrix M into page k of a rank-3 A
    // (phase N4, the WRITE sibling of the N3 page-slice read). Column-major: page k (1-based)
    // is the CONTIGUOUS block A[(k-1)*m*n ..] -> a straight copy from M. A writable; bounds-
    // checked k (end = size(A,3)); m,n size guard. v1: rank-3 DOUBLE A, two leading bare colons
    // + a scalar k, rhs a distinct 2-D DOUBLE matrix var. Placed BEFORE the N-D write branch
    // (which refuses colon subscripts).
    if (ai.isND && ai.ndDims.size() == 3 && ai.dtype == ValueType::DOUBLE
        && lhsCall.children.size() == 4
        && lhsCall.children[1]->type == NodeType::COLON_EXPR && lhsCall.children[1]->children.empty()
        && lhsCall.children[2]->type == NodeType::COLON_EXPR && lhsCall.children[2]->children.empty()
        && lhsCall.children[3]->type != NodeType::COLON_EXPR
        && rhs.type == NodeType::IDENTIFIER && isArrayVar(rhs.strValue) && rhs.strValue != base
        && (arrays_.at(rhs.strValue).is2D
            || (arrays_.at(rhs.strValue).isND && arrays_.at(rhs.strValue).ndDims.size() == 2))
        && arrays_.at(rhs.strValue).dtype == ValueType::DOUBLE
        && inferExpr(*lhsCall.children[3], types_, reg_, classes_).type.shape.isScalar()) {
        if (!ai.isLocal && !ai.isOutput)
            unsupported("page-slice write to a read-only array parameter '" + base + "'");
        const ArrayInfo  &M = arrays_.at(rhs.strValue);  // the 2-D page source
        const std::string m = dimExpr(ai, 0), n = dimExpr(ai, 1), p = dimExpr(ai, 2);
        endStack_.push_back(p);  // `end` in the page index = size(A,3)
        const std::string k = emitExpr(*lhsCall.children[3]);
        endStack_.pop_back();
        line("{");
        ++indent_;
        line("const std::size_t _numkit_m = " + m + ";");
        line("const std::size_t _numkit_n = " + n + ";");
        line("if (" + dimExpr(M, 0) + " != _numkit_m || " + dimExpr(M, 1) + " != _numkit_n)");
        line("    throw std::out_of_range(\"numkit: page-slice assignment size mismatch\");");
        line("const std::size_t _numkit_k0 = numkit_rt::check_idx1(" + k + ", " + p
             + ", \"numkit: page index out of bounds\");");
        line("const std::size_t _numkit_pg = _numkit_m * _numkit_n;");
        line("const std::size_t _numkit_off = _numkit_k0 * _numkit_pg;");
        open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_pg; ++_numkit_i)");
        line(ptr + "[_numkit_off + _numkit_i] = " + M.dataExpr + "[_numkit_i];");
        close();
        --indent_;
        line("}");
        return;
    }

    // 2-D COLUMN-RANGE WRITE A(:, j1:j2) = B: write a 2-D rows x (j2-j1+1) matrix B into columns
    // j1..j2 of a 2-D A (phase N22, the write sibling of the 2-D column-range read). Column-major:
    // columns j1..j2 are the CONTIGUOUS block A[(j1-1)*rows .. j2*rows] -> a straight copy from B.
    // A writable; bounds-checked range (end = cols); a numel guard on B (rows*(j2-j1+1)). v1: 2-D
    // DOUBLE A (KnownDims or runtime), a leading bare colon + a step-1 range j1:j2, rhs a distinct
    // 2-D DOUBLE array var or inline scalar matrix literal. Placed before the rank-3+ trailing-range write.
    if ((ai.is2D || (ai.isND && ai.ndDims.size() == 2)) && ai.dtype == ValueType::DOUBLE
        && lhsCall.children.size() == 3
        && lhsCall.children[1]->type == NodeType::COLON_EXPR && lhsCall.children[1]->children.empty()
        && lhsCall.children[2]->type == NodeType::COLON_EXPR && lhsCall.children[2]->children.size() == 2
        && ((rhs.type == NodeType::IDENTIFIER && isArrayVar(rhs.strValue) && rhs.strValue != base
             && (arrays_.at(rhs.strValue).is2D
                 || (arrays_.at(rhs.strValue).isND && arrays_.at(rhs.strValue).ndDims.size() == 2))
             && arrays_.at(rhs.strValue).dtype == ValueType::DOUBLE)
            || hasRhsGridLit)) {
        if (!ai.isLocal && !ai.isOutput)
            unsupported("column-range write to a read-only matrix parameter '" + base + "'");
        const std::string rows = dimExpr(ai, 0), cols = dimExpr(ai, 1);
        const ASTNode    &rng  = *lhsCall.children[2];  // j1:j2
        endStack_.push_back(cols);                      // `end` in the column index = cols
        const std::string j1 = emitExpr(*rng.children[0]);
        const std::string j2 = emitExpr(*rng.children[1]);
        endStack_.pop_back();
        line("{");
        ++indent_;
        std::string bData, bNumel;
        if (hasRhsGridLit) {
            std::string elems;
            for (std::size_t k = 0; k < rhsGridLit.size(); ++k)
                elems += (k ? ", " : "") + emitExpr(*rhsGridLit[k]);
            line("const double _numkit_lit[] = { " + elems + " };");
            bData  = "_numkit_lit";
            bNumel = std::to_string(rhsGridLit.size());
        } else {
            const ArrayInfo &B = arrays_.at(rhs.strValue);
            bData  = B.dataExpr;
            bNumel = "(" + dimExpr(B, 0) + " * " + dimExpr(B, 1) + ")";
        }
        line("const std::size_t _numkit_rows = " + rows + ";");
        line("const std::size_t _numkit_j1_0 = numkit_rt::check_idx1(" + j1 + ", " + cols
             + ", \"numkit: column range out of bounds\");");
        line("const std::size_t _numkit_j2_0 = numkit_rt::check_idx1(" + j2 + ", " + cols
             + ", \"numkit: column range out of bounds\");");
        line("if (_numkit_j2_0 < _numkit_j1_0)");
        line("    throw std::out_of_range(\"numkit: column range out of bounds\");");
        line("const std::size_t _numkit_nc = _numkit_j2_0 - _numkit_j1_0 + 1;");
        line("if (" + bNumel + " != _numkit_rows * _numkit_nc)");
        line("    throw std::out_of_range(\"numkit: column-range assignment size mismatch\");");
        line("const std::size_t _numkit_off = _numkit_j1_0 * _numkit_rows;");
        open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_rows * _numkit_nc; ++_numkit_i)");
        line(ptr + "[_numkit_off + _numkit_i] = " + bData + "[_numkit_i];");
        close();
        --indent_;
        line("}");
        return;
    }

    // 2-D ROW-RANGE WRITE A(i1:i2, :) = B: write a 2-D (i2-i1+1) x cols matrix B into rows i1..i2
    // of a 2-D A (phase N23, the write sibling of the 2-D row-range read). STRIDED (a row block is
    // not contiguous in column-major): for each column j write the run A[(i1-1)+j*rows .. i2+j*rows)
    // from B's column j. A writable; bounds-checked range (end = rows); a numel guard on B
    // ((i2-i1+1)*cols). v1: 2-D DOUBLE A, a leading step-1 range i1:i2 + a trailing bare colon, rhs
    // a distinct 2-D DOUBLE array var or inline scalar matrix literal. Placed before the rank-3+ trailing-range write.
    if ((ai.is2D || (ai.isND && ai.ndDims.size() == 2)) && ai.dtype == ValueType::DOUBLE
        && lhsCall.children.size() == 3
        && lhsCall.children[1]->type == NodeType::COLON_EXPR && lhsCall.children[1]->children.size() == 2
        && lhsCall.children[2]->type == NodeType::COLON_EXPR && lhsCall.children[2]->children.empty()
        && ((rhs.type == NodeType::IDENTIFIER && isArrayVar(rhs.strValue) && rhs.strValue != base
             && (arrays_.at(rhs.strValue).is2D
                 || (arrays_.at(rhs.strValue).isND && arrays_.at(rhs.strValue).ndDims.size() == 2))
             && arrays_.at(rhs.strValue).dtype == ValueType::DOUBLE)
            || hasRhsGridLit)) {
        if (!ai.isLocal && !ai.isOutput)
            unsupported("row-range write to a read-only matrix parameter '" + base + "'");
        const std::string rows = dimExpr(ai, 0), cols = dimExpr(ai, 1);
        const ASTNode    &rng  = *lhsCall.children[1];  // i1:i2
        endStack_.push_back(rows);                      // `end` in the row index = rows
        const std::string i1 = emitExpr(*rng.children[0]);
        const std::string i2 = emitExpr(*rng.children[1]);
        endStack_.pop_back();
        line("{");
        ++indent_;
        std::string bData, bNumel;
        if (hasRhsGridLit) {
            std::string elems;
            for (std::size_t k = 0; k < rhsGridLit.size(); ++k)
                elems += (k ? ", " : "") + emitExpr(*rhsGridLit[k]);
            line("const double _numkit_lit[] = { " + elems + " };");
            bData  = "_numkit_lit";
            bNumel = std::to_string(rhsGridLit.size());
        } else {
            const ArrayInfo &B = arrays_.at(rhs.strValue);
            bData  = B.dataExpr;
            bNumel = "(" + dimExpr(B, 0) + " * " + dimExpr(B, 1) + ")";
        }
        line("const std::size_t _numkit_rows = " + rows + ";");
        line("const std::size_t _numkit_cols = " + cols + ";");
        line("const std::size_t _numkit_r0 = numkit_rt::check_idx1(" + i1
             + ", _numkit_rows, \"numkit: row range out of bounds\");");
        line("const std::size_t _numkit_r1 = numkit_rt::check_idx1(" + i2
             + ", _numkit_rows, \"numkit: row range out of bounds\");");
        line("if (_numkit_r1 < _numkit_r0)");
        line("    throw std::out_of_range(\"numkit: row range out of bounds\");");
        line("const std::size_t _numkit_nr = _numkit_r1 - _numkit_r0 + 1;");
        line("if (" + bNumel + " != _numkit_nr * _numkit_cols)");
        line("    throw std::out_of_range(\"numkit: row-range assignment size mismatch\");");
        open("for (std::size_t _numkit_j = 0; _numkit_j < _numkit_cols; ++_numkit_j)");
        open("for (std::size_t _numkit_r = 0; _numkit_r < _numkit_nr; ++_numkit_r)");
        line(ptr + "[(_numkit_r0 + _numkit_r) + _numkit_j * _numkit_rows] = " + bData
             + "[_numkit_r + _numkit_j * _numkit_nr];");
        close();
        close();
        --indent_;
        line("}");
        return;
    }

    // 2-D BOTH-RANGE WRITE A(i1:i2, j1:j2) = B: write a 2-D (i2-i1+1) x (j2-j1+1) matrix B into the
    // sub-block rows i1..i2 x columns j1..j2 of a 2-D A (phase N24, the write sibling of the both-
    // range read). STRIDED: for each kept column cc = j1-1+c, write the row run A[(i1-1)+cc*rows ..
    // i2+cc*rows) from B's column c. A writable; both ranges bounds-checked; a numel guard on B
    // ((i2-i1+1)*(j2-j1+1)). v1: 2-D DOUBLE A, two step-1 ranges, rhs a distinct 2-D DOUBLE var or inline scalar matrix literal.
    if ((ai.is2D || (ai.isND && ai.ndDims.size() == 2)) && ai.dtype == ValueType::DOUBLE
        && lhsCall.children.size() == 3
        && lhsCall.children[1]->type == NodeType::COLON_EXPR && lhsCall.children[1]->children.size() == 2
        && lhsCall.children[2]->type == NodeType::COLON_EXPR && lhsCall.children[2]->children.size() == 2
        && ((rhs.type == NodeType::IDENTIFIER && isArrayVar(rhs.strValue) && rhs.strValue != base
             && (arrays_.at(rhs.strValue).is2D
                 || (arrays_.at(rhs.strValue).isND && arrays_.at(rhs.strValue).ndDims.size() == 2))
             && arrays_.at(rhs.strValue).dtype == ValueType::DOUBLE)
            || hasRhsGridLit)) {
        if (!ai.isLocal && !ai.isOutput)
            unsupported("block-range write to a read-only matrix parameter '" + base + "'");
        const std::string rows = dimExpr(ai, 0), cols = dimExpr(ai, 1);
        const ASTNode    &rr   = *lhsCall.children[1];  // i1:i2
        const ASTNode    &cr   = *lhsCall.children[2];  // j1:j2
        endStack_.push_back(rows);
        const std::string i1 = emitExpr(*rr.children[0]);
        const std::string i2 = emitExpr(*rr.children[1]);
        endStack_.pop_back();
        endStack_.push_back(cols);
        const std::string j1 = emitExpr(*cr.children[0]);
        const std::string j2 = emitExpr(*cr.children[1]);
        endStack_.pop_back();
        line("{");
        ++indent_;
        std::string bData, bNumel;
        if (hasRhsGridLit) {
            std::string elems;
            for (std::size_t k = 0; k < rhsGridLit.size(); ++k)
                elems += (k ? ", " : "") + emitExpr(*rhsGridLit[k]);
            line("const double _numkit_lit[] = { " + elems + " };");
            bData  = "_numkit_lit";
            bNumel = std::to_string(rhsGridLit.size());
        } else {
            const ArrayInfo &B = arrays_.at(rhs.strValue);
            bData  = B.dataExpr;
            bNumel = "(" + dimExpr(B, 0) + " * " + dimExpr(B, 1) + ")";
        }
        line("const std::size_t _numkit_rows = " + rows + ";");
        line("const std::size_t _numkit_cols = " + cols + ";");
        line("const std::size_t _numkit_r0 = numkit_rt::check_idx1(" + i1
             + ", _numkit_rows, \"numkit: row range out of bounds\");");
        line("const std::size_t _numkit_r1 = numkit_rt::check_idx1(" + i2
             + ", _numkit_rows, \"numkit: row range out of bounds\");");
        line("if (_numkit_r1 < _numkit_r0)");
        line("    throw std::out_of_range(\"numkit: row range out of bounds\");");
        line("const std::size_t _numkit_c0 = numkit_rt::check_idx1(" + j1
             + ", _numkit_cols, \"numkit: column range out of bounds\");");
        line("const std::size_t _numkit_c1 = numkit_rt::check_idx1(" + j2
             + ", _numkit_cols, \"numkit: column range out of bounds\");");
        line("if (_numkit_c1 < _numkit_c0)");
        line("    throw std::out_of_range(\"numkit: column range out of bounds\");");
        line("const std::size_t _numkit_nr = _numkit_r1 - _numkit_r0 + 1;");
        line("const std::size_t _numkit_nc = _numkit_c1 - _numkit_c0 + 1;");
        line("if (" + bNumel + " != _numkit_nr * _numkit_nc)");
        line("    throw std::out_of_range(\"numkit: block-range assignment size mismatch\");");
        open("for (std::size_t _numkit_c = 0; _numkit_c < _numkit_nc; ++_numkit_c)");
        open("for (std::size_t _numkit_r = 0; _numkit_r < _numkit_nr; ++_numkit_r)");
        line(ptr + "[(_numkit_r0 + _numkit_r) + (_numkit_c0 + _numkit_c) * _numkit_rows] = " + bData
             + "[_numkit_r + _numkit_c * _numkit_nr];");
        close();
        close();
        --indent_;
        line("}");
        return;
    }

    // trailing-RANGE WRITE A(:,...,:,k1:k2) = B: write a rank-r array B into the contiguous
    // trailing-dim block k1..k2 of a rank-r A (r >= 3; phase N20 r=3 page-range + N21 r=4 slab-
    // range write, the WRITE sibling of the trailing-range read). Column-major: the leading
    // (r-1) dims form a slab of size d0*..*d(r-2), and the trailing values k1..k2 are the
    // CONTIGUOUS block A[(k1-1)*slab .. k2*slab] -> a straight copy from B. A writable; bounds-
    // checked range (end = size(A,r)); a numel guard on B (slab*(k2-k1+1)). v1: rank-3/4 DOUBLE
    // A, (r-1) leading bare colons + a step-1 range k1:k2, rhs a distinct rank-r DOUBLE array
    // var. Placed before the N-D write branch.
    if (ai.isND && ai.ndDims.size() >= 3 && ai.dtype == ValueType::DOUBLE
        && lhsCall.children.size() == ai.ndDims.size() + 1
        && rhs.type == NodeType::IDENTIFIER && isArrayVar(rhs.strValue) && rhs.strValue != base
        && arrays_.at(rhs.strValue).isND
        && arrays_.at(rhs.strValue).ndDims.size() == ai.ndDims.size()
        && arrays_.at(rhs.strValue).dtype == ValueType::DOUBLE) {
        const std::size_t r = ai.ndDims.size();
        bool              ok = lhsCall.children[r]->type == NodeType::COLON_EXPR
                  && lhsCall.children[r]->children.size() == 2;
        for (std::size_t i = 1; i < r && ok; ++i)
            if (!(lhsCall.children[i]->type == NodeType::COLON_EXPR
                  && lhsCall.children[i]->children.empty()))
                ok = false;
        if (ok) {
            if (!ai.isLocal && !ai.isOutput)
                unsupported("range write to a read-only array parameter '" + base + "'");
            const ArrayInfo  &B       = arrays_.at(rhs.strValue);  // value source
            const std::string lastDim = dimExpr(ai, r - 1);
            const ASTNode    &rng     = *lhsCall.children[r];  // k1:k2
            endStack_.push_back(lastDim);                      // `end` in the range = size(A,r)
            const std::string k1 = emitExpr(*rng.children[0]);
            const std::string k2 = emitExpr(*rng.children[1]);
            endStack_.pop_back();
            line("{");
            ++indent_;
            std::string slabExpr, bNumel;
            for (std::size_t i = 0; i + 1 < r; ++i) {
                line("const std::size_t _numkit_d" + std::to_string(i) + " = " + dimExpr(ai, i) + ";");
                slabExpr += (i ? " * " : "") + ("_numkit_d" + std::to_string(i));
            }
            for (std::size_t i = 0; i < r; ++i) bNumel += (i ? " * " : "") + dimExpr(B, i);
            line("const std::size_t _numkit_slab = " + slabExpr + ";");
            line("const std::size_t _numkit_k1_0 = numkit_rt::check_idx1(" + k1 + ", " + lastDim
                 + ", \"numkit: range out of bounds\");");
            line("const std::size_t _numkit_k2_0 = numkit_rt::check_idx1(" + k2 + ", " + lastDim
                 + ", \"numkit: range out of bounds\");");
            line("if (_numkit_k2_0 < _numkit_k1_0)");
            line("    throw std::out_of_range(\"numkit: range out of bounds\");");
            line("const std::size_t _numkit_nb = _numkit_k2_0 - _numkit_k1_0 + 1;");
            line("if ((" + bNumel + ") != _numkit_slab * _numkit_nb)");
            line("    throw std::out_of_range(\"numkit: range assignment size mismatch\");");
            line("const std::size_t _numkit_off = _numkit_k1_0 * _numkit_slab;");
            open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_slab * _numkit_nb; ++_numkit_i)");
            line(ptr + "[_numkit_off + _numkit_i] = " + B.dataExpr + "[_numkit_i];");
            close();
            --indent_;
            line("}");
            return;
        }
    }

    // GENERAL scalar/colon slice WRITE A(s_0..s_{r-1}) = rhs for a rank r >= 4 A (the WRITE
    // sibling of the rank-4+ general slice read). Each subscript is a bare colon or a scalar
    // (>=1 of each). The kept slice dims (colon -> size(A,k), scalar -> 1, trailing scalars
    // drop) describe a column-major region; rhs (a distinct array var) supplies the values in
    // column-major order. STRIDED scatter: A flat = base(from the scalar subscripts, kept AND
    // dropped) + sum over kept COLON dims of j_k * As_k; A[flat] = rhs[o]. Per-axis bounds
    // checks on the scalars + a numel guard on rhs. v1: rank-4+ writable DOUBLE A, all bare-
    // colon/scalar subscripts, rhs a distinct DOUBLE array var. Placed before the N-D write
    // branch (which refuses colon subscripts).
    if (ai.isND && ai.ndDims.size() >= 4 && ai.dtype == ValueType::DOUBLE
        && lhsCall.children.size() - 1 == ai.ndDims.size()
        && rhs.type == NodeType::IDENTIFIER && isArrayVar(rhs.strValue) && rhs.strValue != base
        && arrays_.at(rhs.strValue).dtype == ValueType::DOUBLE) {
        const std::size_t r = ai.ndDims.size();
        std::vector<bool> colon(r, false);
        int               lastCol = -1, scalarN = 0;
        bool              allCS = true;
        for (std::size_t k = 0; k < r; ++k) {
            const ASTNode &s     = *lhsCall.children[k + 1];
            const bool     isCol = s.type == NodeType::COLON_EXPR && s.children.empty();
            colon[k]             = isCol;
            if (isCol)
                lastCol = static_cast<int>(k);
            else if (s.type != NodeType::COLON_EXPR
                     && inferExpr(s, types_, reg_, classes_).type.shape.isScalar())
                ++scalarN;
            else {
                allCS = false;
                break;
            }
        }
        if (allCS && lastCol >= 0 && scalarN > 0) {
            if (!ai.isLocal && !ai.isOutput)
                unsupported("slice write to a read-only array parameter '" + base + "'");
            const std::size_t keptRank = static_cast<std::size_t>(lastCol) + 1 < 2
                                             ? 2u
                                             : static_cast<std::size_t>(lastCol) + 1;
            const ArrayInfo  &R        = arrays_.at(rhs.strValue);  // value source
            line("{");
            ++indent_;
            line("const std::size_t _numkit_As0 = 1;");  // A column-major strides
            for (std::size_t k = 1; k < r; ++k)
                line("const std::size_t _numkit_As" + std::to_string(k) + " = _numkit_As"
                     + std::to_string(k - 1) + " * (" + dimExpr(ai, k - 1) + ");");
            line("std::size_t _numkit_base = 0;");  // fixed offset from the scalar subscripts
            for (std::size_t k = 0; k < r; ++k) {
                if (colon[k]) continue;
                endStack_.push_back(dimExpr(ai, k));
                const std::string sk = emitExpr(*lhsCall.children[k + 1]);
                endStack_.pop_back();
                line("{ const std::size_t _numkit_s = numkit_rt::check_idx1(" + sk + ", " + dimExpr(ai, k)
                     + ", \"numkit: slice index out of bounds\");");
                line("  _numkit_base += _numkit_s * _numkit_As" + std::to_string(k) + "; }");
            }
            for (std::size_t kk = 0; kk < keptRank; ++kk)  // kept (output) dims
                line("const std::size_t _numkit_Bd" + std::to_string(kk) + " = "
                     + (colon[kk] ? dimExpr(ai, kk) : std::string("1")) + ";");
            line("const std::size_t _numkit_Bs0 = 1;");  // kept-region strides
            for (std::size_t kk = 1; kk < keptRank; ++kk)
                line("const std::size_t _numkit_Bs" + std::to_string(kk) + " = _numkit_Bs"
                     + std::to_string(kk - 1) + " * _numkit_Bd" + std::to_string(kk - 1) + ";");
            std::string numel = "_numkit_Bd0";
            for (std::size_t kk = 1; kk < keptRank; ++kk) numel += " * _numkit_Bd" + std::to_string(kk);
            std::string rNumel;  // rhs element count
            if (R.isND) {
                rNumel = R.ndDims[0];
                for (std::size_t i = 1; i < R.ndDims.size(); ++i) rNumel += " * " + R.ndDims[i];
            } else if (R.is2D) {
                rNumel = dimExpr(R, 0) + " * " + dimExpr(R, 1);
            } else {
                rNumel = R.lenVar;
            }
            line("if ((" + rNumel + ") != (" + numel + "))");
            line("    throw std::out_of_range(\"numkit: slice assignment size mismatch\");");
            open("for (std::size_t _numkit_o = 0; _numkit_o < (" + numel + "); ++_numkit_o)");
            line("std::size_t _numkit_af = _numkit_base;");
            for (std::size_t kk = 0; kk < keptRank; ++kk)
                if (colon[kk])
                    line("_numkit_af += ((_numkit_o / _numkit_Bs" + std::to_string(kk) + ") % _numkit_Bd"
                         + std::to_string(kk) + ") * _numkit_As" + std::to_string(kk) + ";");
            line(ptr + "[_numkit_af] = " + R.dataExpr + "[_numkit_o];");
            close();
            --indent_;
            line("}");
            return;
        }
    }

    // Rank-N (N>=3) AND runtime-dim 2-D write A(i,j,k,...) = v -> column-major
    // numkit_rt::indexN_set. The companions in ai.ndDims give the per-axis sizes, so a
    // SCALAR-subscript element store works for any rank, including a runtime-dim 2-D
    // matrix (the canonical `A = zeros(m,n); A(i,j) = v` fill). A colon subscript (a
    // slice write A(i,:) = row) is refused here -- scalar subscripts only (v1) -- so it
    // is a clean boundary rather than a mis-emit of emitExpr on a bare colon. EXCEPTION:
    // a single LOGICAL-array subscript A(A>0)=... is a masked scatter (linear over the
    // flat buffer), not an N-D element write -> fall through to the masked-write branch.
    bool singleLogicalMask = false;
    if (lhsCall.children.size() == 2) {
        const InferredType st = inferExpr(*lhsCall.children[1], types_, reg_, classes_).type;
        singleLogicalMask =
            st.isConcrete() && st.dtype == ValueType::LOGICAL && !st.shape.isScalar();
    }
    if (ai.isND && !singleLogicalMask) {
        if (!ai.isLocal && !ai.isOutput)
            unsupported("N-D write to a read-only matrix parameter '" + base + "'");
        if (lhsCall.children.size() - 1 != ai.ndDims.size())
            unsupported("N-D index arity for '" + base + "' (expected "
                        + std::to_string(ai.ndDims.size()) + " subscripts)");
        for (std::size_t i = 1; i < lhsCall.children.size(); ++i)
            if (lhsCall.children[i]->type == NodeType::COLON_EXPR)
                unsupported("N-D slice write (a colon subscript) for '" + base
                            + "' -- scalar subscripts only (v1)");
        std::string dims, subs;
        for (std::size_t k = 0; k < ai.ndDims.size(); ++k) dims += (k ? ", " : "") + ai.ndDims[k];
        for (std::size_t i = 1; i < lhsCall.children.size(); ++i) {
            endStack_.push_back(ai.ndDims[i - 1]);
            subs += (i > 1 ? ", " : "") + emitExpr(*lhsCall.children[i]);
            endStack_.pop_back();
        }
        line("numkit_rt::indexN_set(" + ptr + ", {" + dims + "}, {" + subs + "}, " + emitExpr(rhs)
             + ");");
        return;
    }

    // LOGICAL-INDEXING WRITE: `x(m) = c` where m is a LOGICAL-array var and c a
    // SCALAR -> scatter c into x at the positions where m is true (MATLAB
    // x(logical)=scalar, e.g. clamp x(x<0)=0). x must be writable (local/output).
    // v1: a single LOGICAL-array subscript that is a VARIABLE; x + m 1-D; scalar
    // rhs. Bound on min(len) so a too-long mask cannot write x out of bounds.
    if (lhsCall.children.size() == 2 && lhsCall.children[1]->type == NodeType::IDENTIFIER
        && isArrayVar(lhsCall.children[1]->strValue)
        && arrays_.at(lhsCall.children[1]->strValue).dtype == ValueType::LOGICAL && !ai.is2D) {
        if (!ai.isLocal && !ai.isOutput)
            unsupported("logical-indexing write to a read-only array parameter '" + base + "'");
        const ArrayInfo &bm = arrays_.at(lhsCall.children[1]->strValue);  // m (mask)
        if (bm.is2D || bm.isND) unsupported("logical-indexing write: a 1-D mask only (v1)");
        const AbstractValue rhsScalar = inferExpr(rhs, types_, reg_, classes_);
        if (!rhsScalar.type.shape.isScalar())
            unsupported("logical-indexing write: a scalar rhs only (v1)");
        line("{");
        ++indent_;
        line("const " + cppScalarType(ai.dtype) + " _numkit_c = " + emitExpr(rhs) + ";");
        line("const std::size_t _numkit_n = " + bm.lenVar + " < " + ai.lenVar + " ? "
             + bm.lenVar + " : " + ai.lenVar + ";");
        open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_n; ++_numkit_i)");
        line("if (" + bm.dataExpr + "[_numkit_i]) " + ptr + "[_numkit_i] = _numkit_c;");
        close();
        --indent_;
        line("}");
        return;
    }

    // LOGICAL-INDEXING WRITE with an INLINE elementwise mask: `x(<expr>) = c` where <expr>
    // is a pure-elementwise LOGICAL array over x ITSELF (x(x<0)=0, A(A>lo & A<hi)=v, ...) and
    // c is a SCALAR. The mask is FUSED into the scatter loop -- no temp vector: for each flat
    // element i, if the per-element mask holds, x[i] = c. Works for a 1-D vector OR a 2-D / N-D
    // matrix -- the buffer is column-major flat, the mask is flat-elementwise, and the write is
    // flat, so it is rank-agnostic (bound on NUMEL). Restricted to a SELF-mask (the only array
    // operand is x): every reference is to x so indexing is in bounds, AND the fusion is sound
    // -- a per-element mask over x alone matches MATLAB's compute-all-then-scatter (setting x[i]
    // cannot change x[j!=i]'s mask). A non-elementwise mask (A(A>mean(A))=0) is rejected by
    // collectElementwise -> bridged, so fusion is never wrong. x writable; a mask over OTHER
    // arrays / an array rhs -> refused. After the logical-VAR write (a pre-bound mask var takes
    // that simpler branch).
    if (lhsCall.children.size() == 2 && lhsCall.children[1]->type != NodeType::COLON_EXPR
        && lhsCall.children[1]->type != NodeType::IDENTIFIER) {
        const AbstractValue   maskAV = inferExpr(*lhsCall.children[1], types_, reg_, classes_);
        std::set<std::string> maskArrays;
        const bool            pureEw = collectElementwise(*lhsCall.children[1], maskArrays);
        if (maskAV.type.isConcrete() && maskAV.type.dtype == ValueType::LOGICAL
            && !maskAV.type.shape.isScalar() && pureEw && maskArrays.size() == 1
            && *maskArrays.begin() == base) {
            if (!ai.isLocal && !ai.isOutput)
                unsupported("logical-indexing write to a read-only array parameter '" + base + "'");
            const AbstractValue rhsScalar = inferExpr(rhs, types_, reg_, classes_);
            if (!rhsScalar.type.isConcrete() || !rhsScalar.type.shape.isScalar())
                unsupported("logical-indexing write (inline mask): a scalar rhs only (v1)");
            std::string numel;  // flat element count (1-D length / 2-D rows*cols / N-D product)
            if (ai.isND) {
                numel = ai.ndDims[0];
                for (std::size_t i = 1; i < ai.ndDims.size(); ++i) numel += " * " + ai.ndDims[i];
            } else if (ai.is2D) {
                numel = ai.rowsVar + " * " + ai.colsVar;
            } else {
                numel = ai.lenVar;
            }
            line("{");
            ++indent_;
            line("const " + cppScalarType(ai.dtype) + " _numkit_c = " + emitExpr(rhs) + ";");
            elementCtx_ = "_numkit_i";  // whole x in the mask -> x[_numkit_i] (flat)
            const std::string maskExpr = emitExpr(*lhsCall.children[1]);
            elementCtx_.clear();
            open("for (std::size_t _numkit_i = 0; _numkit_i < (" + numel + "); ++_numkit_i)");
            line("if (" + maskExpr + ") " + ptr + "[_numkit_i] = _numkit_c;");
            close();
            --indent_;
            line("}");
            return;
        }
    }

    // NUMERIC SCATTER WRITE: `x(idx) = v` where idx is a 1-D NUMERIC (DOUBLE) index vector ->
    // assign into x at the 1-based positions idx. rhs is EITHER a SCALAR (broadcast: x(idx[i])
    // = c for all i) OR a distinct 1-D DOUBLE array of length numel(idx) (x(idx[i]) = rhs[i]).
    // The WRITE sibling of the numeric gather read. Each index is range+integer checked BEFORE
    // the cast (a NaN fails `>= 1.0`, so no UB; the throw is faithful to MATLAB's subscript
    // error). Repeated indices keep the LAST write -- a forward loop matches MATLAB. x writable
    // 1-D DOUBLE; idx a DOUBLE array VAR distinct from x. An inline-literal idx / a 2-D x / a
    // length-mismatched or aliasing array rhs -> refused (sound). Placed after the logical
    // writes (a LOGICAL idx takes those; this is the numeric sibling).
    if (lhsCall.children.size() == 2 && !ai.is2D && !ai.isND && ai.dtype == ValueType::DOUBLE
        && lhsCall.children[1]->type == NodeType::IDENTIFIER
        && isArrayVar(lhsCall.children[1]->strValue)
        && arrays_.at(lhsCall.children[1]->strValue).dtype == ValueType::DOUBLE
        && !arrays_.at(lhsCall.children[1]->strValue).is2D
        && !arrays_.at(lhsCall.children[1]->strValue).isND
        && lhsCall.children[1]->strValue != base) {
        if (!ai.isLocal && !ai.isOutput)
            unsupported("numeric scatter write to a read-only array parameter '" + base + "'");
        const ArrayInfo    &bi  = arrays_.at(lhsCall.children[1]->strValue);  // index vector
        const AbstractValue rav = inferExpr(rhs, types_, reg_, classes_);
        const bool rhsScalar = rav.type.isConcrete() && rav.type.shape.isScalar();
        const bool rhsArr    = rhs.type == NodeType::IDENTIFIER && isArrayVar(rhs.strValue)
                            && rhs.strValue != base && rhs.strValue != lhsCall.children[1]->strValue
                            && !arrays_.at(rhs.strValue).is2D && !arrays_.at(rhs.strValue).isND
                            && arrays_.at(rhs.strValue).dtype == ValueType::DOUBLE;
        if (!rhsScalar && !rhsArr)
            unsupported("numeric scatter write rhs: a scalar or a distinct 1-D DOUBLE array (v1)");
        line("{");
        ++indent_;
        if (rhsScalar)
            line("const double _numkit_c = " + emitExpr(rhs) + ";");
        else {
            line("if (" + arrays_.at(rhs.strValue).lenVar + " != " + bi.lenVar + ")");
            line("    throw std::out_of_range(\"numkit: scatter assignment length mismatch\");");
        }
        const std::string rhsElem =
            rhsScalar ? std::string("_numkit_c") : (arrays_.at(rhs.strValue).dataExpr + "[_numkit_i]");
        open("for (std::size_t _numkit_i = 0; _numkit_i < " + bi.lenVar + "; ++_numkit_i)");
        line("const std::size_t _numkit_idx0 = numkit_rt::check_idx1(" + bi.dataExpr + "[_numkit_i], "
             + ai.lenVar + ", \"numkit: array index out of bounds\");");
        line(ptr + "[_numkit_idx0] = " + rhsElem + ";");
        close();
        --indent_;
        line("}");
        return;
    }

    // 1-D SLICE WRITE: x(a:b) = rhs / x(a:s:b) = rhs. The lhs colon ranges over x's
    // 1-based positions (`end` -> x's length, pushed here). count = the colon count;
    // then either broadcast a SCALAR rhs into each position, or copy a matched-length
    // 1-D array rhs element-wise. x must be writable (local/output) + DOUBLE; bounds-
    // checked, and (array rhs) length-checked. Aliasing (rhs is x itself) is refused
    // (overlapping copy) -- sound. v1: x 1-D DOUBLE.
    if (lhsCall.children.size() == 2 && lhsCall.children[1]->type == NodeType::COLON_EXPR
        && !ai.is2D && !ai.isND && ai.dtype == ValueType::DOUBLE
        && (lhsCall.children[1]->children.size() == 2 || lhsCall.children[1]->children.size() == 3)) {
        if (!ai.isLocal && !ai.isOutput)
            unsupported("slice write to a read-only array parameter '" + base + "'");
        const ASTNode      &colon     = *lhsCall.children[1];
        const bool          three     = colon.children.size() == 3;
        const AbstractValue rhsAV     = inferExpr(rhs, types_, reg_, classes_);
        const bool          rhsScalar = rhsAV.type.isConcrete() && rhsAV.type.shape.isScalar();
        const bool          rhsArr    = rhs.type == NodeType::IDENTIFIER && isArrayVar(rhs.strValue)
                                     && rhs.strValue != base && !arrays_.at(rhs.strValue).is2D
                                     && !arrays_.at(rhs.strValue).isND
                                     && arrays_.at(rhs.strValue).dtype == ValueType::DOUBLE;
        if (!rhsScalar && !rhsArr)
            unsupported("slice write rhs: a scalar or a distinct 1-D DOUBLE array var (v1)");
        endStack_.push_back(ai.lenVar);
        const std::string start = emitExpr(*colon.children[0]);
        const std::string step  = three ? emitExpr(*colon.children[1]) : std::string("1.0");
        const std::string stop  = emitExpr(*colon.children[three ? 2 : 1]);
        endStack_.pop_back();
        line("{");
        ++indent_;
        line("std::ptrdiff_t _numkit_s0 = 0, _numkit_st = 0;");
        line("const std::ptrdiff_t _numkit_cnt = numkit_rt::slice_bounds(" + start + ", " + step
             + ", " + stop + ", " + ai.lenVar + ", _numkit_s0, _numkit_st);");
        if (rhsScalar) {
            line("const double _numkit_v = " + emitExpr(rhs) + ";");
            open("for (std::ptrdiff_t _numkit_k = 0; _numkit_k < _numkit_cnt; ++_numkit_k)");
            line(ptr + "[static_cast<std::size_t>(_numkit_s0 + _numkit_k * _numkit_st)] = _numkit_v;");
            close();
        } else {
            const ArrayInfo &ra = arrays_.at(rhs.strValue);
            line("if (static_cast<std::ptrdiff_t>(" + ra.lenVar + ") != _numkit_cnt)");
            line("    throw std::out_of_range(\"numkit: slice assignment length mismatch\");");
            open("for (std::ptrdiff_t _numkit_k = 0; _numkit_k < _numkit_cnt; ++_numkit_k)");
            line(ptr + "[static_cast<std::size_t>(_numkit_s0 + _numkit_k * _numkit_st)] = " + ra.dataExpr
                 + "[static_cast<std::size_t>(_numkit_k)];");
            close();
        }
        --indent_;
        line("}");
        return;
    }

    std::vector<AbstractValue> idx;
    for (std::size_t i = 1; i < lhsCall.children.size(); ++i)
        idx.push_back(inferExpr(*lhsCall.children[i], types_, reg_, classes_));
    const AbstractValue rhsAV = inferExpr(rhs, types_, reg_, classes_);

    const IndexPlan plan = planIndexWrite(arrayValue(base), idx, rhsAV);
    if (plan.form == IndexForm::Subscript2D) {
        // A 2-D PARAM is read-only (const T*); a mutable 2-D (a local or the
        // output) is writable via column-major index2_set.
        if (!ai.is2D) unsupported("A(i,j) on a non-matrix '" + base + "'");
        if (!ai.isLocal && !ai.isOutput)
            unsupported("2-D write to a read-only matrix parameter '" + base + "'");
        endStack_.push_back(ai.rowsVar);
        const std::string sub1 = emitExpr(*lhsCall.children[1]);
        endStack_.pop_back();
        endStack_.push_back(ai.colsVar);
        const std::string sub2 = emitExpr(*lhsCall.children[2]);
        endStack_.pop_back();
        line("numkit_rt::index2_set(" + ptr + ", " + ai.rowsVar + ", " + ai.colsVar + ", "
             + sub1 + ", " + sub2 + ", " + emitExpr(rhs) + ");");
        return;
    }
    if (plan.form != IndexForm::LinearScalar || ai.is2D)
        unsupported("index write form for '" + base
                    + "' (a 1-D scalar store, or A(i,j) on a matrix)");

    // `end` in the LHS index resolves to the array's current length, so `x(end+1) = v`
    // appends (index numel+1 -> grow by one). Pushed only around the index expression; the
    // rhs has its own end-context (e.g. x(end+1) = y(end) indexes y).
    endStack_.push_back(ai.lenVar);
    const std::string idxExpr = emitExpr(*lhsCall.children[1]);
    endStack_.pop_back();
    const std::string rhsExpr = emitExpr(rhs);
    if (plan.boundsChecked) {
        if (ai.isLocal)
            // A LOCAL owned vector GROWS on an out-of-range store (x(i)=v past the end
            // resizes, zero-filling the gap -- MATLAB semantics). A param / output is a raw
            // caller buffer that cannot grow -> the throwing index_set.
            line("numkit_rt::index_set_grow(" + cppIdent(base) + ", " + idxExpr + ", " + rhsExpr + ");");
        else
            line("numkit_rt::index_set(" + ptr + ", " + ai.lenVar + ", " + idxExpr + ", " + rhsExpr
                 + ");");
    } else {
        line(ptr + "[static_cast<std::size_t>(" + idxExpr + ") - 1] = " + rhsExpr + ";");
    }
}

} // namespace numkit::codegen::detail
