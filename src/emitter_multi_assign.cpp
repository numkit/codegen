// codegen/src/emitter_multi_assign.cpp — multi-output assignment lowering.

#include "emitter_internal.hpp"

namespace numkit::codegen::detail {

// `[a, b, ...] = f(args)` -> a void call to f's specialisation with the
// targets appended as reference out-args. v1: simple-identifier targets, a
// bare-name user-function RHS, and all of f's outputs requested.
void Emitter::emitMultiAssign(const ASTNode &s)
{
    if (!s.lhsTargets.empty()) unsupported("complex multi-assign targets (v1)");
    if (s.children.empty()) unsupported("multi-assign arity");
    const ASTNode &rhs = *s.children[0];

    // [r, c] = size(A): native two-output size. r = axis 0; c = the remaining
    // axes folded (cols for a matrix, the trailing-dim product for N-D) -- so
    // [r,c]=size(rand(2,3,4)) gives r=2, c=12, matching MATLAB. Both targets
    // are real scalars. BEFORE the user-function path so it lowers natively
    // instead of bridging into numkit::size. v1 is exactly this two-output
    // idiom; any other nargout (or a user-shadowed `size`) falls through.
    if (rhs.type == NodeType::CALL && rhs.children.size() == 2
        && rhs.children[0]->type == NodeType::IDENTIFIER
        && rhs.children[0]->strValue == "size"
        && rhs.children[1]->type == NodeType::IDENTIFIER
        && isArrayVar(rhs.children[1]->strValue) && s.returnNames.size() == 2
        && !(ctx_ && ctx_->funcs && ctx_->funcs->has("size"))) {
        const std::string &rn0 = s.returnNames[0];
        const std::string &rn1 = s.returnNames[1];
        if (rn0.empty() || rn0 == "~" || rn1.empty() || rn1 == "~")
            unsupported("[r,c]=size with an ignored (~) output (v1)");
        if (isArrayVar(rn0) || isArrayVar(rn1))
            unsupported("[r,c]=size into a non-scalar target");
        const ArrayInfo &aai = arrays_.at(rhs.children[1]->strValue);
        if (!aai.is2D && !aai.isND && aai.orient == VecOrient::Unknown)
            unsupported("[r,c]=size of an orientation-unknown vector");
        const std::size_t rank = aai.isND ? aai.ndDims.size() : 2;
        std::string       cols = dimExpr(aai, 1);  // fold dims 1..rank-1 into c
        for (std::size_t k = 2; k < rank; ++k) cols += " * " + dimExpr(aai, k);
        line(cppIdent(rn0) + " = static_cast<double>(" + dimExpr(aai, 0) + ");");
        line(cppIdent(rn1) + " = static_cast<double>(" + cols + ");");
        types_.set(rn0, {InferredType::scalar(ValueType::DOUBLE), ConstVal::unknown()});
        types_.set(rn1, {InferredType::scalar(ValueType::DOUBLE), ConstVal::unknown()});
        return;
    }
    // [m, i] = max(x) / [m, i] = min(x): native argmax/argmin. m = the extremum
    // value, i = its 1-based index (the FIRST occurrence on ties, MATLAB). NaN is
    // ignored (the extremum of the non-NaNs); an all-NaN vector -> the value NaN at
    // index 1; the empty vector -> (NaN, 0) [MATLAB returns [], not representable as
    // a codegen scalar]. The update fires on the first element, on a strict cmp, or
    // when acc is NaN and the candidate is not (so the first non-NaN seeds it).
    // ALWAYS native (like [r,c]=size): maxMinMultiTransfer types m,i concrete for a
    // DOUBLE vector, so this emit must match in every tier (a matrix/non-double
    // operand stays Dynamic -> the bridged multi path below). BEFORE the user-fn
    // path. v1: a single 1-D DOUBLE array var, the two-output form.
    if (rhs.type == NodeType::CALL && rhs.children.size() == 2
        && rhs.children[0]->type == NodeType::IDENTIFIER
        && (rhs.children[0]->strValue == "max" || rhs.children[0]->strValue == "min")
        && rhs.children[1]->type == NodeType::IDENTIFIER
        && isArrayVar(rhs.children[1]->strValue) && s.returnNames.size() == 2
        && !arrays_.at(rhs.children[1]->strValue).is2D
        && !arrays_.at(rhs.children[1]->strValue).isND
        && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE
        && !(ctx_ && ctx_->funcs && ctx_->funcs->has(rhs.children[0]->strValue))) {
        const std::string &rn0 = s.returnNames[0];  // value
        const std::string &rn1 = s.returnNames[1];  // 1-based index
        if (rn0.empty() || rn0 == "~" || rn1.empty() || rn1 == "~")
            unsupported("[m,i]=max/min with an ignored (~) output (v1)");
        if (isArrayVar(rn0) || isArrayVar(rn1))
            unsupported("[m,i]=max/min into a non-scalar target");
        const ArrayInfo  &xa  = arrays_.at(rhs.children[1]->strValue);
        const std::string cmp = rhs.children[0]->strValue == "max" ? ">" : "<";
        line("{");
        ++indent_;
        line("const std::size_t _numkit_n = " + xa.lenVar + ";");
        open("if (_numkit_n == 0)");
        line(cppIdent(rn0) + " = std::numeric_limits<double>::quiet_NaN(); " + cppIdent(rn1)
             + " = 0.0;");
        close();
        open("else");
        line("double _numkit_acc = " + xa.dataExpr + "[0]; std::size_t _numkit_idx = 1;");
        open("for (std::size_t _numkit_i = 1; _numkit_i < _numkit_n; ++_numkit_i)");
        line("const double _numkit_v = " + xa.dataExpr + "[_numkit_i];");
        line("if (_numkit_v " + cmp + " _numkit_acc || (_numkit_acc != _numkit_acc && _numkit_v == _numkit_v))");
        line("    { _numkit_acc = _numkit_v; _numkit_idx = _numkit_i + 1; }");
        close();
        line(cppIdent(rn0) + " = _numkit_acc; " + cppIdent(rn1)
             + " = static_cast<double>(_numkit_idx);");
        close();
        --indent_;
        line("}");
        types_.set(rn0, {InferredType::scalar(ValueType::DOUBLE), ConstVal::unknown()});
        types_.set(rn1, {InferredType::scalar(ValueType::DOUBLE), ConstVal::unknown()});
        return;
    }
    // [m, i] = max(A) / min(A) on a 2-D MATRIX -> column-wise argmax/argmin: m and i are both
    // 1 x n ROW vectors (1-D LOCALs), m(j) = the column-j extremum, i(j) = its 1-based row index
    // (FIRST occurrence on ties, MATLAB). NaN-skipping (seed on the column's first row, update on a
    // strict cmp or when acc is NaN and the candidate is not). EXACT -> every tier. Column-major.
    // v1: a single DOUBLE matrix var (KnownDims or runtime rank-2); the two outputs ARRAY targets.
    if (rhs.type == NodeType::CALL && rhs.children.size() == 2
        && rhs.children[0]->type == NodeType::IDENTIFIER
        && (rhs.children[0]->strValue == "max" || rhs.children[0]->strValue == "min")
        && rhs.children[1]->type == NodeType::IDENTIFIER
        && isArrayVar(rhs.children[1]->strValue) && s.returnNames.size() == 2
        && (arrays_.at(rhs.children[1]->strValue).is2D
            || (arrays_.at(rhs.children[1]->strValue).isND
                && arrays_.at(rhs.children[1]->strValue).ndDims.size() == 2))
        && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE
        && !(ctx_ && ctx_->funcs && ctx_->funcs->has(rhs.children[0]->strValue))) {
        const std::string &rn0 = s.returnNames[0];  // value row vector
        const std::string &rn1 = s.returnNames[1];  // 1-based index row vector
        if (rn0.empty() || rn0 == "~" || rn1.empty() || rn1 == "~")
            unsupported("[m,i]=max/min(matrix) with an ignored (~) output (v1)");
        const ArrayInfo  &A   = arrays_.at(rhs.children[1]->strValue);
        const std::string cmp = rhs.children[0]->strValue == "max" ? ">" : "<";
        line("{");
        ++indent_;
        line("const std::size_t _numkit_m = " + dimExpr(A, 0) + ";");
        line("const std::size_t _numkit_n = " + dimExpr(A, 1) + ";");
        line(cppIdent(rn0) + ".assign(_numkit_n, 0.0);");
        line(cppIdent(rn1) + ".assign(_numkit_n, 0.0);");
        open("for (std::size_t _numkit_j = 0; _numkit_j < _numkit_n; ++_numkit_j)");
        line("double _numkit_acc = 0.0; std::size_t _numkit_idx = 0;");  // seeded on _numkit_i == 0
        open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_m; ++_numkit_i)");
        line("const double _numkit_v = " + A.dataExpr + "[_numkit_i + _numkit_j * _numkit_m];");
        line("if (_numkit_i == 0 || _numkit_v " + cmp + " _numkit_acc || (_numkit_acc != _numkit_acc && _numkit_v == _numkit_v))");
        line("    { _numkit_acc = _numkit_v; _numkit_idx = _numkit_i + 1; }");
        close();
        line(cppIdent(rn0) + "[_numkit_j] = _numkit_acc; " + cppIdent(rn1)
             + "[_numkit_j] = static_cast<double>(_numkit_idx);");
        close();
        --indent_;
        line("}");
        types_.set(rn0,
                   {InferredType::concrete(ValueType::DOUBLE, Shape::rowVector()), ConstVal::unknown()});
        types_.set(rn1,
                   {InferredType::concrete(ValueType::DOUBLE, Shape::rowVector()), ConstVal::unknown()});
        return;
    }
    // [s, i] = sort(x[, 'descend']) on a 1-D vector -> sorted values s + the 1-based permutation i
    // (which original position each sorted element came from). STABLE (std::stable_sort) so ties
    // keep ascending original order, matching MATLAB. NaN-extremum comparator identical to the
    // value-only 1-D sort (NaN last ascending / first descending). Sorts an index vector by the
    // data, then gathers values + indices. v1: a single 1-D DOUBLE array var; the two outputs ARRAY
    // targets; ascending or 'descend'. (A matrix [s,i]=sort is the next brick.)
    if (rhs.type == NodeType::CALL
        && (rhs.children.size() == 2
            || (rhs.children.size() == 3 && rhs.children[2]->type == NodeType::STRING_LITERAL))
        && rhs.children[0]->type == NodeType::IDENTIFIER && rhs.children[0]->strValue == "sort"
        && rhs.children[1]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[1]->strValue)
        && s.returnNames.size() == 2 && !arrays_.at(rhs.children[1]->strValue).is2D
        && !arrays_.at(rhs.children[1]->strValue).isND
        && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE
        && !(ctx_ && ctx_->funcs && ctx_->funcs->has(rhs.children[0]->strValue))) {
        const std::string &rn0 = s.returnNames[0];  // sorted values
        const std::string &rn1 = s.returnNames[1];  // 1-based permutation indices
        if (rn0.empty() || rn0 == "~" || rn1.empty() || rn1 == "~")
            unsupported("[s,i]=sort with an ignored (~) output (v1)");
        const ArrayInfo &xa      = arrays_.at(rhs.children[1]->strValue);
        const bool       descend = rhs.children.size() == 3 && rhs.children[2]->strValue == "descend";
        line("{");
        ++indent_;
        line("const std::size_t _numkit_n = " + xa.lenVar + ";");
        line("const double* _numkit_d = " + xa.dataExpr + ";");
        line("std::vector<std::size_t> _numkit_p(_numkit_n);");
        open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_n; ++_numkit_i)");
        line("_numkit_p[_numkit_i] = _numkit_i;");
        close();
        line("std::stable_sort(_numkit_p.begin(), _numkit_p.end(),");
        line(descend ? "    [_numkit_d](std::size_t _a, std::size_t _b){ return _numkit_d[_a] > _numkit_d[_b] || "
                       "(_numkit_d[_a] != _numkit_d[_a] && _numkit_d[_b] == _numkit_d[_b]); });"
                     : "    [_numkit_d](std::size_t _a, std::size_t _b){ return _numkit_d[_a] < _numkit_d[_b] || "
                       "(_numkit_d[_b] != _numkit_d[_b] && _numkit_d[_a] == _numkit_d[_a]); });");
        line(cppIdent(rn0) + ".assign(_numkit_n, 0.0);");
        line(cppIdent(rn1) + ".assign(_numkit_n, 0.0);");
        open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_n; ++_numkit_i)");
        line(cppIdent(rn0) + "[_numkit_i] = _numkit_d[_numkit_p[_numkit_i]];");
        line(cppIdent(rn1) + "[_numkit_i] = static_cast<double>(_numkit_p[_numkit_i] + 1);");
        close();
        --indent_;
        line("}");
        types_.set(rn0,
                   {InferredType::concrete(ValueType::DOUBLE, Shape::vector1D()), ConstVal::unknown()});
        types_.set(rn1,
                   {InferredType::concrete(ValueType::DOUBLE, Shape::vector1D()), ConstVal::unknown()});
        return;
    }
    // [s, i] = sort(A) on a 2-D MATRIX -> sort each COLUMN independently: s = sorted columns,
    // i = the per-column 1-based permutation (both same m x n shape). STABLE per column (ties keep
    // ascending original row order, MATLAB); NaN-extremum comparator (NaN last asc / first desc).
    // Column-major: each column j is the block [j*m, (j+1)*m). v1: a DOUBLE matrix var; both outputs
    // ARRAY targets; ascending or 'descend'.
    if (rhs.type == NodeType::CALL
        && (rhs.children.size() == 2
            || (rhs.children.size() == 3 && rhs.children[2]->type == NodeType::STRING_LITERAL))
        && rhs.children[0]->type == NodeType::IDENTIFIER && rhs.children[0]->strValue == "sort"
        && rhs.children[1]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[1]->strValue)
        && s.returnNames.size() == 2
        && (arrays_.at(rhs.children[1]->strValue).is2D
            || (arrays_.at(rhs.children[1]->strValue).isND
                && arrays_.at(rhs.children[1]->strValue).ndDims.size() == 2))
        && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE
        && !(ctx_ && ctx_->funcs && ctx_->funcs->has(rhs.children[0]->strValue))) {
        const std::string &rn0 = s.returnNames[0];  // sorted values
        const std::string &rn1 = s.returnNames[1];  // per-column 1-based permutation
        if (rn0.empty() || rn0 == "~" || rn1.empty() || rn1 == "~")
            unsupported("[s,i]=sort(matrix) with an ignored (~) output (v1)");
        const ArrayInfo &A       = arrays_.at(rhs.children[1]->strValue);
        const ArrayInfo &Ms      = arrays_.at(rn0);
        const ArrayInfo &Mi      = arrays_.at(rn1);
        const bool       descend = rhs.children.size() == 3 && rhs.children[2]->strValue == "descend";
        line("{");
        ++indent_;
        line("const std::size_t _numkit_m = " + dimExpr(A, 0) + ";");
        line("const std::size_t _numkit_n = " + dimExpr(A, 1) + ";");
        line("const double* _numkit_d = " + A.dataExpr + ";");
        if (Ms.isND && Ms.ndDims.size() == 2) {
            line(Ms.ndDims[0] + " = _numkit_m;");
            line(Ms.ndDims[1] + " = _numkit_n;");
        }
        if (Mi.isND && Mi.ndDims.size() == 2) {
            line(Mi.ndDims[0] + " = _numkit_m;");
            line(Mi.ndDims[1] + " = _numkit_n;");
        }
        line(cppIdent(rn0) + ".assign(_numkit_m * _numkit_n, 0.0);");
        line(cppIdent(rn1) + ".assign(_numkit_m * _numkit_n, 0.0);");
        line("std::vector<std::size_t> _numkit_p(_numkit_m);");
        open("for (std::size_t _numkit_j = 0; _numkit_j < _numkit_n; ++_numkit_j)");
        line("const std::size_t _numkit_b = _numkit_j * _numkit_m;");
        open("for (std::size_t _numkit_r = 0; _numkit_r < _numkit_m; ++_numkit_r)");
        line("_numkit_p[_numkit_r] = _numkit_r;");
        close();
        line("std::stable_sort(_numkit_p.begin(), _numkit_p.end(),");
        line(descend
                 ? "    [_numkit_d, _numkit_b](std::size_t _a, std::size_t _b){ return _numkit_d[_numkit_b + _a] > "
                   "_numkit_d[_numkit_b + _b] || (_numkit_d[_numkit_b + _a] != _numkit_d[_numkit_b + _a] && _numkit_d[_numkit_b + _b] "
                   "== _numkit_d[_numkit_b + _b]); });"
                 : "    [_numkit_d, _numkit_b](std::size_t _a, std::size_t _b){ return _numkit_d[_numkit_b + _a] < "
                   "_numkit_d[_numkit_b + _b] || (_numkit_d[_numkit_b + _b] != _numkit_d[_numkit_b + _b] && _numkit_d[_numkit_b + _a] "
                   "== _numkit_d[_numkit_b + _a]); });");
        open("for (std::size_t _numkit_r = 0; _numkit_r < _numkit_m; ++_numkit_r)");
        line(cppIdent(rn0) + "[_numkit_b + _numkit_r] = _numkit_d[_numkit_b + _numkit_p[_numkit_r]];");
        line(cppIdent(rn1) + "[_numkit_b + _numkit_r] = static_cast<double>(_numkit_p[_numkit_r] + 1);");
        close();
        close();
        --indent_;
        line("}");
        // s and i share A's shape (DOUBLE m x n); re-affirm from the source's inferred type.
        const AbstractValue _numkit_at = inferExpr(*rhs.children[1], types_, reg_, classes_);
        types_.set(rn0, {_numkit_at.type, ConstVal::unknown()});
        types_.set(rn1, {_numkit_at.type, ConstVal::unknown()});
        return;
    }
    // [u, ia, ic] = unique(x) on a 1-D vector -> sorted distinct values u, the FIRST-occurrence
    // index ia (u = x(ia)), and the map ic (x = u(ic)); all 1-D DOUBLE LOCALs, 1-based. A stable
    // index sort (NaN-extremum comparator) puts the first of each equal run at the smallest
    // original index = first occurrence; NaN stays distinct (NaN != NaN). 2 outputs ([u,ia]) or 3
    // ([u,ia,ic]). v1: a single 1-D DOUBLE array var. (A matrix unique stays bridged.)
    if (rhs.type == NodeType::CALL && rhs.children.size() == 2
        && rhs.children[0]->type == NodeType::IDENTIFIER && rhs.children[0]->strValue == "unique"
        && rhs.children[1]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[1]->strValue)
        && (s.returnNames.size() == 2 || s.returnNames.size() == 3)
        // 1-D OR 2-D source (unique flattens column-major); reject only rank >= 3.
        && !(arrays_.at(rhs.children[1]->strValue).isND
             && arrays_.at(rhs.children[1]->strValue).ndDims.size() != 2)
        && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE
        && !(ctx_ && ctx_->funcs && ctx_->funcs->has(rhs.children[0]->strValue))) {
        const bool         wantIc = s.returnNames.size() == 3;
        const std::string &rnU    = s.returnNames[0];  // sorted distinct values
        const std::string &rnIa   = s.returnNames[1];  // first-occurrence indices (u = x(ia))
        const std::string  rnIc   = wantIc ? s.returnNames[2] : std::string();  // map (x = u(ic))
        if (rnU.empty() || rnU == "~" || rnIa.empty() || rnIa == "~"
            || (wantIc && (rnIc.empty() || rnIc == "~")))
            unsupported("[u,ia,ic]=unique with an ignored (~) output (v1)");
        const ArrayInfo  &xa  = arrays_.at(rhs.children[1]->strValue);
        // numel: a 2-D source flattens column-major (rows*cols); a 1-D source uses its length.
        const std::string cnt =
            (xa.is2D || xa.isND) ? (dimExpr(xa, 0) + " * " + dimExpr(xa, 1)) : xa.lenVar;
        line("{");
        ++indent_;
        line("const std::size_t _numkit_n = " + cnt + ";");
        line("const double* _numkit_d = " + xa.dataExpr + ";");
        line("std::vector<std::size_t> _numkit_p(_numkit_n);");
        open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_n; ++_numkit_i)");
        line("_numkit_p[_numkit_i] = _numkit_i;");
        close();
        line("std::stable_sort(_numkit_p.begin(), _numkit_p.end(),");
        line("    [_numkit_d](std::size_t _a, std::size_t _b){ return _numkit_d[_a] < _numkit_d[_b] || (_numkit_d[_b] "
             "!= _numkit_d[_b] && _numkit_d[_a] == _numkit_d[_a]); });");
        line(cppIdent(rnU) + ".clear();");
        line(cppIdent(rnIa) + ".clear();");
        if (wantIc) {
            line(cppIdent(rnIc) + ".assign(_numkit_n, 0.0);");
            line("std::size_t _numkit_uk = 0;");
        }
        open("for (std::size_t _numkit_s = 0; _numkit_s < _numkit_n; ++_numkit_s)");
        open("if (_numkit_s == 0 || _numkit_d[_numkit_p[_numkit_s]] != _numkit_d[_numkit_p[_numkit_s - 1]])");
        if (wantIc) line("if (_numkit_s != 0) ++_numkit_uk;");
        line(cppIdent(rnU) + ".push_back(_numkit_d[_numkit_p[_numkit_s]]);");
        line(cppIdent(rnIa) + ".push_back(static_cast<double>(_numkit_p[_numkit_s] + 1));");
        close();
        if (wantIc) line(cppIdent(rnIc) + "[_numkit_p[_numkit_s]] = static_cast<double>(_numkit_uk + 1);");
        close();
        --indent_;
        line("}");
        types_.set(rnU,
                   {InferredType::concrete(ValueType::DOUBLE, Shape::vector1D()), ConstVal::unknown()});
        types_.set(rnIa,
                   {InferredType::concrete(ValueType::DOUBLE, Shape::vector1D()), ConstVal::unknown()});
        if (wantIc)
            types_.set(rnIc, {InferredType::concrete(ValueType::DOUBLE, Shape::vector1D()),
                              ConstVal::unknown()});
        return;
    }
    // [c, ia, ib] = intersect(a, b) on two 1-D vectors -> c = sorted distinct common values, ia the
    // FIRST-occurrence index in a (c = a(ia)), ib the first-occurrence index in b. Build sorted-
    // unique ua + sorted ub; for each v in ua present in ub (binary_search, v==v so NaN is never
    // common), push v and scan a/b for its first index. 2 outputs ([c,ia]) or 3 ([c,ia,ib]).
    // !bridge_ guard. v1: two 1-D DOUBLE array vars.
    if (rhs.type == NodeType::CALL && rhs.children.size() == 3
        && rhs.children[0]->type == NodeType::IDENTIFIER && rhs.children[0]->strValue == "intersect"
        && rhs.children[1]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[1]->strValue)
        && rhs.children[2]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[2]->strValue)
        && (s.returnNames.size() == 2 || s.returnNames.size() == 3) && !bridge_
        && !arrays_.at(rhs.children[1]->strValue).is2D && !arrays_.at(rhs.children[1]->strValue).isND
        && !arrays_.at(rhs.children[2]->strValue).is2D && !arrays_.at(rhs.children[2]->strValue).isND
        && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE
        && arrays_.at(rhs.children[2]->strValue).dtype == ValueType::DOUBLE
        && !(ctx_ && ctx_->funcs && ctx_->funcs->has(rhs.children[0]->strValue))) {
        const bool         wantIb = s.returnNames.size() == 3;
        const std::string &rnC    = s.returnNames[0];  // common values
        const std::string &rnIa   = s.returnNames[1];  // first index in a
        const std::string  rnIb   = wantIb ? s.returnNames[2] : std::string();  // first index in b
        if (rnC.empty() || rnC == "~" || rnIa.empty() || rnIa == "~"
            || (wantIb && (rnIb.empty() || rnIb == "~")))
            unsupported("[c,ia,ib]=intersect with an ignored (~) output (v1)");
        const ArrayInfo &aa = arrays_.at(rhs.children[1]->strValue);
        const ArrayInfo &bb = arrays_.at(rhs.children[2]->strValue);
        line("{");
        ++indent_;
        line("auto _numkit_cmp = [](double _a, double _b){ return _a < _b || (_b != _b && _a == _a); };");
        line("std::vector<double> _numkit_ua(" + aa.dataExpr + ", " + aa.dataExpr + " + " + aa.lenVar
             + ");");
        line("std::vector<double> _numkit_ub(" + bb.dataExpr + ", " + bb.dataExpr + " + " + bb.lenVar
             + ");");
        line("std::sort(_numkit_ua.begin(), _numkit_ua.end(), _numkit_cmp);");
        line("std::sort(_numkit_ub.begin(), _numkit_ub.end(), _numkit_cmp);");
        line("_numkit_ua.erase(std::unique(_numkit_ua.begin(), _numkit_ua.end()), _numkit_ua.end());");
        line(cppIdent(rnC) + ".clear();");
        line(cppIdent(rnIa) + ".clear();");
        if (wantIb) line(cppIdent(rnIb) + ".clear();");
        open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_ua.size(); ++_numkit_i)");
        line("const double _numkit_v = _numkit_ua[_numkit_i];");
        open("if (_numkit_v == _numkit_v && std::binary_search(_numkit_ub.begin(), _numkit_ub.end(), _numkit_v, _numkit_cmp))");
        line("std::size_t _numkit_ja = 0; while (_numkit_ja < " + aa.lenVar + " && !(" + aa.dataExpr
             + "[_numkit_ja] == _numkit_v)) ++_numkit_ja;");
        line(cppIdent(rnC) + ".push_back(_numkit_v);");
        line(cppIdent(rnIa) + ".push_back(static_cast<double>(_numkit_ja + 1));");
        if (wantIb) {
            line("std::size_t _numkit_jb = 0; while (_numkit_jb < " + bb.lenVar + " && !(" + bb.dataExpr
                 + "[_numkit_jb] == _numkit_v)) ++_numkit_jb;");
            line(cppIdent(rnIb) + ".push_back(static_cast<double>(_numkit_jb + 1));");
        }
        close();
        close();
        --indent_;
        line("}");
        types_.set(rnC,
                   {InferredType::concrete(ValueType::DOUBLE, Shape::rowVector()), ConstVal::unknown()});
        types_.set(rnIa,
                   {InferredType::concrete(ValueType::DOUBLE, Shape::colVector()), ConstVal::unknown()});
        if (wantIb)
            types_.set(rnIb, {InferredType::concrete(ValueType::DOUBLE, Shape::colVector()),
                              ConstVal::unknown()});
        return;
    }
    // [c, ia] = setdiff(a, b) on two 1-D vectors -> c = sorted distinct values of a not in b, ia the
    // FIRST-occurrence index in a (c = a(ia)). Sort an index vector of a (stable, NaN-last) and walk
    // it: at each new distinct value (a[p[s]] != a[p[s-1]], so each NaN is its own distinct with its
    // own index -- multi-NaN correct), keep it when it is NaN (v!=v, never in b) OR absent from the
    // sorted ub. This mirrors the unique-multi index logic, so ia is correct including duplicate
    // NaNs. 2-output form. !bridge_ guard. v1: two 1-D DOUBLE array vars.
    if (rhs.type == NodeType::CALL && rhs.children.size() == 3
        && rhs.children[0]->type == NodeType::IDENTIFIER && rhs.children[0]->strValue == "setdiff"
        && rhs.children[1]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[1]->strValue)
        && rhs.children[2]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[2]->strValue)
        && s.returnNames.size() == 2 && !bridge_
        && !arrays_.at(rhs.children[1]->strValue).is2D && !arrays_.at(rhs.children[1]->strValue).isND
        && !arrays_.at(rhs.children[2]->strValue).is2D && !arrays_.at(rhs.children[2]->strValue).isND
        && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE
        && arrays_.at(rhs.children[2]->strValue).dtype == ValueType::DOUBLE
        && !(ctx_ && ctx_->funcs && ctx_->funcs->has(rhs.children[0]->strValue))) {
        const std::string &rnC  = s.returnNames[0];  // a-not-in-b values
        const std::string &rnIa = s.returnNames[1];  // first index in a
        if (rnC.empty() || rnC == "~" || rnIa.empty() || rnIa == "~")
            unsupported("[c,ia]=setdiff with an ignored (~) output (v1)");
        const ArrayInfo &aa = arrays_.at(rhs.children[1]->strValue);
        const ArrayInfo &bb = arrays_.at(rhs.children[2]->strValue);
        line("{");
        ++indent_;
        line("auto _numkit_cmp = [](double _a, double _b){ return _a < _b || (_b != _b && _a == _a); };");
        line("const double* _numkit_da = " + aa.dataExpr + ";");
        line("std::vector<double> _numkit_ub(" + bb.dataExpr + ", " + bb.dataExpr + " + " + bb.lenVar
             + ");");
        line("std::sort(_numkit_ub.begin(), _numkit_ub.end(), _numkit_cmp);");
        line("std::vector<std::size_t> _numkit_p(" + aa.lenVar + ");");
        open("for (std::size_t _numkit_i = 0; _numkit_i < " + aa.lenVar + "; ++_numkit_i)");
        line("_numkit_p[_numkit_i] = _numkit_i;");
        close();
        line("std::stable_sort(_numkit_p.begin(), _numkit_p.end(), [_numkit_da](std::size_t _x, std::size_t _y){ "
             "return _numkit_da[_x] < _numkit_da[_y] || (_numkit_da[_y] != _numkit_da[_y] && _numkit_da[_x] == "
             "_numkit_da[_x]); });");
        line(cppIdent(rnC) + ".clear();");
        line(cppIdent(rnIa) + ".clear();");
        open("for (std::size_t _numkit_s = 0; _numkit_s < " + aa.lenVar + "; ++_numkit_s)");
        open("if (_numkit_s == 0 || _numkit_da[_numkit_p[_numkit_s]] != _numkit_da[_numkit_p[_numkit_s - 1]])");
        line("const double _numkit_v = _numkit_da[_numkit_p[_numkit_s]];");
        open("if (_numkit_v != _numkit_v || !std::binary_search(_numkit_ub.begin(), _numkit_ub.end(), _numkit_v, "
             "_numkit_cmp))");
        line(cppIdent(rnC) + ".push_back(_numkit_v);");
        line(cppIdent(rnIa) + ".push_back(static_cast<double>(_numkit_p[_numkit_s] + 1));");
        close();
        close();
        close();
        --indent_;
        line("}");
        types_.set(rnC,
                   {InferredType::concrete(ValueType::DOUBLE, Shape::rowVector()), ConstVal::unknown()});
        types_.set(rnIa,
                   {InferredType::concrete(ValueType::DOUBLE, Shape::colVector()), ConstVal::unknown()});
        return;
    }
    // [u, ia, ib] = union(a, b) on two 1-D vectors -> u = sorted distinct of [a b]; ia = the a-
    // sourced first-occurrence indices, ib = the b-sourced ones. Construction = unique-multi over
    // the concatenation [a b]: stable index sort, and at each new distinct value its first concat
    // index k decides the source (k < |a| -> a-sourced, ia.push(k+1); else b-sourced, ib.push(k-
    // |a|+1)). Sound incl. multi-NaN (the stable index sort gives each NaN its own concat index, so
    // NaNs split by source exactly as MATLAB). 2-output [u,ia] drops ib (ia stays a-sourced).
    // !bridge_ guard. v1: two 1-D DOUBLE array vars.
    if (rhs.type == NodeType::CALL && rhs.children.size() == 3
        && rhs.children[0]->type == NodeType::IDENTIFIER && rhs.children[0]->strValue == "union"
        && rhs.children[1]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[1]->strValue)
        && rhs.children[2]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[2]->strValue)
        && (s.returnNames.size() == 2 || s.returnNames.size() == 3) && !bridge_
        && !arrays_.at(rhs.children[1]->strValue).is2D && !arrays_.at(rhs.children[1]->strValue).isND
        && !arrays_.at(rhs.children[2]->strValue).is2D && !arrays_.at(rhs.children[2]->strValue).isND
        && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE
        && arrays_.at(rhs.children[2]->strValue).dtype == ValueType::DOUBLE
        && !(ctx_ && ctx_->funcs && ctx_->funcs->has(rhs.children[0]->strValue))) {
        const bool         wantIb = s.returnNames.size() == 3;
        const std::string &rnU    = s.returnNames[0];  // sorted distinct union
        const std::string &rnIa   = s.returnNames[1];  // a-sourced first-occurrence indices
        const std::string  rnIb   = wantIb ? s.returnNames[2] : std::string();  // b-sourced indices
        if (rnU.empty() || rnU == "~" || rnIa.empty() || rnIa == "~"
            || (wantIb && (rnIb.empty() || rnIb == "~")))
            unsupported("[u,ia,ib]=union with an ignored (~) output (v1)");
        const ArrayInfo &aa = arrays_.at(rhs.children[1]->strValue);
        const ArrayInfo &bb = arrays_.at(rhs.children[2]->strValue);
        line("{");
        ++indent_;
        line("std::vector<double> _numkit_cv;");
        line("_numkit_cv.reserve(" + aa.lenVar + " + " + bb.lenVar + ");");
        line("_numkit_cv.insert(_numkit_cv.end(), " + aa.dataExpr + ", " + aa.dataExpr + " + " + aa.lenVar
             + ");");
        line("_numkit_cv.insert(_numkit_cv.end(), " + bb.dataExpr + ", " + bb.dataExpr + " + " + bb.lenVar
             + ");");
        line("const std::size_t _numkit_al = " + aa.lenVar + ";");
        line("std::vector<std::size_t> _numkit_p(_numkit_cv.size());");
        open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_cv.size(); ++_numkit_i)");
        line("_numkit_p[_numkit_i] = _numkit_i;");
        close();
        line("std::stable_sort(_numkit_p.begin(), _numkit_p.end(), [&_numkit_cv](std::size_t _x, std::size_t _y){ "
             "return _numkit_cv[_x] < _numkit_cv[_y] || (_numkit_cv[_y] != _numkit_cv[_y] && _numkit_cv[_x] == "
             "_numkit_cv[_x]); });");
        line(cppIdent(rnU) + ".clear();");
        line(cppIdent(rnIa) + ".clear();");
        if (wantIb) line(cppIdent(rnIb) + ".clear();");
        open("for (std::size_t _numkit_s = 0; _numkit_s < _numkit_cv.size(); ++_numkit_s)");
        open("if (_numkit_s == 0 || _numkit_cv[_numkit_p[_numkit_s]] != _numkit_cv[_numkit_p[_numkit_s - 1]])");
        line("const double _numkit_v = _numkit_cv[_numkit_p[_numkit_s]]; const std::size_t _numkit_k = _numkit_p[_numkit_s];");
        line(cppIdent(rnU) + ".push_back(_numkit_v);");
        open("if (_numkit_k < _numkit_al)");
        line(cppIdent(rnIa) + ".push_back(static_cast<double>(_numkit_k + 1));");
        close();
        if (wantIb) {
            open("if (_numkit_k >= _numkit_al)");
            line(cppIdent(rnIb) + ".push_back(static_cast<double>(_numkit_k - _numkit_al + 1));");
            close();
        }
        close();
        close();
        --indent_;
        line("}");
        types_.set(rnU,
                   {InferredType::concrete(ValueType::DOUBLE, Shape::rowVector()), ConstVal::unknown()});
        types_.set(rnIa,
                   {InferredType::concrete(ValueType::DOUBLE, Shape::colVector()), ConstVal::unknown()});
        if (wantIb)
            types_.set(rnIb, {InferredType::concrete(ValueType::DOUBLE, Shape::colVector()),
                              ConstVal::unknown()});
        return;
    }
    // Bridged builtin multi-output (opt-in, DESIGN.md §10 C1): `[a, b, ...] =
    // builtin(args)` where builtin is NOT a compiled user function (nor a
    // variable). The runtime owns nargout and computes all outputs; each result
    // is kept BOXED as a Dynamic numkit_rt::val (sound — no per-builtin type
    // assumption; reuses the Dynamic tier). Targets are Dynamic locals (the
    // inference types them so); a ~ slot discards its output. v1: 2+ outputs;
    // args boxed by emitDynamicExpr (scalar / 1-D/2-D/N-D array / Dynamic).
    if (bridge_ && rhs.type == NodeType::CALL && !rhs.children.empty()
        && rhs.children[0]->type == NodeType::IDENTIFIER && s.returnNames.size() >= 2
        && !types_.has(rhs.children[0]->strValue)  // not a variable (would be indexing)
        && !(ctx_ && ctx_->funcs && ctx_->funcs->has(rhs.children[0]->strValue))) {  // not a user fn
        const std::string callee = rhs.children[0]->strValue;
        const std::size_t nout   = s.returnNames.size();
        std::string       boxedArgs;
        for (std::size_t i = 1; i < rhs.children.size(); ++i)
            boxedArgs += (i > 1 ? ", " : "") + emitDynamicExpr(*rhs.children[i]);
        const std::string call = "numkit_rt::call_dyn_multi(\"" + callee + "\", {" + boxedArgs + "}, "
                                 + std::to_string(nout) + ", _numkit_mo)";
        open("");  // scope for the extra-outputs array (outputs 1..nout-1)
        line("numkit_rt::val _numkit_mo[" + std::to_string(nout - 1) + "];");
        const std::string &t0 = s.returnNames[0];  // output 0 = call_dyn_multi's return
        if (t0.empty() || t0 == "~") {
            line(call + ";");  // discarded
        } else {
            line(cppIdent(t0) + " = " + call + ";");
            types_.set(t0, {InferredType::dynamic(), ConstVal::unknown()});
        }
        for (std::size_t k = 1; k < nout; ++k) {  // outputs 1.. = the extras
            const std::string &tk = s.returnNames[k];
            if (tk.empty() || tk == "~") continue;  // discarded (released at block close)
            line(cppIdent(tk) + " = std::move(_numkit_mo[" + std::to_string(k - 1) + "]);");
            types_.set(tk, {InferredType::dynamic(), ConstVal::unknown()});
        }
        close();
        return;
    }
    // Multi-assign RHS can be either a user-function call `f(args)` or a value/handle
    // class method call `obj.m(args)`.
    if (rhs.type != NodeType::CALL || rhs.children.empty())
        unsupported("multi-assign RHS must be a function or method call");

    const ASTNode    &callee     = *rhs.children[0];
    const ASTNode    *def        = nullptr;
    std::string       calleeName;
    std::string       mangledBase;
    bool              isMethodCall = false;
    InferredType      selfType;
    std::vector<InferredType> probeArgTypes;

    if (callee.type == NodeType::IDENTIFIER) {
        calleeName  = callee.strValue;
        mangledBase = calleeName;
        if (!ctx_ || !ctx_->funcs || types_.has(calleeName) || !ctx_->funcs->has(calleeName))
            unsupported("multi-assign of a non-user-function '" + calleeName + "'");
        def = ctx_->funcs->find(calleeName);
    } else if (callee.type == NodeType::FIELD_ACCESS && !callee.children.empty()) {
        const std::string &method  = callee.strValue;
        const ASTNode     &objNode = *callee.children[0];
        const InferredType objT    = inferExpr(objNode, types_, reg_, classes_).type;
        if (!objT.isObject() || !classes_ || !ctx_)
            unsupported("multi-assign method call on non-object '" + method + "'");
        const ClassInfo *ci = classes_->byId(objT.classId);
        if (!ci) unsupported("unknown class id in multi-assign method call");
        const ASTNode *md = ci->method(method);
        if (!md)
            unsupported("class '" + ci->name + "' has no method '" + method + "'");
        def          = md;
        calleeName   = ci->name + "::" + method;
        mangledBase  = ci->name + "__" + method;
        isMethodCall = true;
        selfType     = objT;
        probeArgTypes.push_back(selfType);
    } else {
        unsupported("multi-assign RHS must be a user-function or method call");
    }

    for (std::size_t i = 1; i < rhs.children.size(); ++i)
        probeArgTypes.push_back(inferExpr(*rhs.children[i], types_, reg_, classes_).type);
    if (probeArgTypes.size() != def->paramNames.size())
        unsupported("arity mismatch calling '" + calleeName + "'");
    const std::vector<InferredType> outs = reg_.applyMulti(calleeName, toArgInfos(probeArgTypes));

    const std::size_t ncallee = def->returnNames.size();
    if (s.returnNames.size() > ncallee)
        unsupported("multi-assign requests more outputs than '" + calleeName + "' returns");

    const InferredType out0        = !outs.empty() ? outs[0] : InferredType::dynamic();
    const bool         out0ByValue = isByValueReturnArrayType(out0);

    // Protect input array arguments that alias any `std::vector<T>&` out-param so resizing an
    // output mid-call never invalidates a raw `const T*` input pointer (`[s, a] = f(a)`).
    std::set<std::string>    outRefArrays;
    std::vector<std::string> aliasDecls;
    for (std::size_t i = (out0ByValue ? 1 : 0); i < s.returnNames.size(); ++i) {
        const std::string &rn = s.returnNames[i];
        if (rn.empty() || rn == "~") continue;
        const InferredType ot = i < outs.size() ? outs[i] : InferredType::dynamic();
        if (ot.isStruct() && ot.structLayout) {
            collectStructOutRefArrays("_numkit_fld_" + rn, *ot.structLayout, outRefArrays);
        } else if ((is2DMatrixType(ot) || (ot.isConcrete() && ot.shape.isNDims())
                    || isBufferArrayType(ot))
                   && isArrayVar(rn)) {
            outRefArrays.insert(rn);
        }
    }
    prepareCallInputAliases(rhs, 1, outRefArrays, aliasDecls);

    std::vector<InferredType> argTypes;
    std::string               argList;
    if (isMethodCall) {
        argTypes.push_back(selfType);
        argList = emitExpr(*callee.children[0]);
    }
    for (std::size_t i = 1; i < rhs.children.size(); ++i)
        appendCallArg(*rhs.children[i], argTypes, argList);
    callAliasInputs_.clear();

    std::vector<std::string> ignoreDecls;
    for (std::size_t i = (out0ByValue ? 1 : 0); i < ncallee; ++i) {
        const bool          requested = i < s.returnNames.size();
        const std::string   rn = requested ? s.returnNames[i] : std::string();  // unrequested -> ""
        const InferredType  ot = i < outs.size() ? outs[i] : InferredType::dynamic();
        if (!requested || rn.empty() || rn == "~") {
            appendIgnoreOutArg("_numkit_ignore_" + std::to_string(i), ot, ignoreDecls, argList);
        } else if (ot.isStruct()) {
            if (!ot.structLayout || ot.structLayout->fields.empty())
                unsupported("multi-output struct '" + rn + "' has no fields");
            types_.set(rn, {ot, ConstVal::unknown()});
            appendStructOutArgs("_numkit_fld_" + rn, *ot.structLayout, argList);
        } else if (is2DMatrixType(ot) || (ot.isConcrete() && ot.shape.isNDims())
                   || isBufferArrayType(ot)) {
            if (!isArrayVar(rn) || !arrays_.at(rn).isLocal)
                unsupported("multi-output array target '" + rn
                            + "' must be a local vector (RawBuffer ABI parameters cannot be rebound)");
            if (!argList.empty()) argList += ", ";
            argList += cppIdent(rn);  // out-arg, bound to the callee's reference out-param
            types_.set(rn, {ot, ConstVal::unknown()});
        } else if (isUnboxableScalarType(ot) || ot.isObject()) {
            if (!argList.empty()) argList += ", ";
            argList += cppIdent(rn);  // out-arg, bound to the callee's reference out-param
            types_.set(rn, {ot, ConstVal::unknown()});
        } else {
            unsupported("multi-output '" + rn + "' of '" + calleeName + "' has unsupported type "
                        + ot.str());
        }
    }

    const std::string mangled = mangle(mangledBase, argTypes);
    if (ctx_->seen.insert(mangled).second)
        ctx_->pending.push_back({def, argTypes, mangled});
    // With a by-value output 0, target 0 takes the RETURN (an array LOCAL —
    // std::vector); a ~ slot discards the returned vector.
    std::string callStmt;
    if (out0ByValue && !s.returnNames.empty()
        && !(s.returnNames[0].empty() || s.returnNames[0] == "~")) {
        if (!isArrayVar(s.returnNames[0]) || !arrays_.at(s.returnNames[0]).isLocal)
            unsupported("multi-output array target '" + s.returnNames[0]
                        + "' must be a local vector (RawBuffer ABI parameters cannot be rebound)");
        callStmt = cppIdent(s.returnNames[0]) + " = " + mangled + "(" + argList + ");";
        types_.set(s.returnNames[0], {out0, ConstVal::unknown()});
    } else {
        callStmt = mangled + "(" + argList + ");";  // all-ref outputs, or out0 discarded
    }
    if (aliasDecls.empty() && ignoreDecls.empty()) {
        line(callStmt);
    } else {
        open("");  // fresh scope for input-alias snapshots and ~ throwaway locals
        for (const std::string &d : aliasDecls) line(d);
        for (const std::string &d : ignoreDecls) line(d);
        line(callStmt);
        close();
    }
}

} // namespace numkit::codegen::detail
