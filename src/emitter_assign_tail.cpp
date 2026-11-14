// codegen/src/emitter_assign_tail.cpp — identifier/struct-field assignment lowering (part 2).

#include "emitter_internal.hpp"

namespace numkit::codegen::detail {

void Emitter::emitAssignIdentifierTail(const ASTNode &lhs, const ASTNode &rhs,
                                       const std::string &name,
                                       const std::string &structWriteFld)
{
    (void)lhs;
        // Native kron(A, B) -> the Kronecker product, A m x n and B p x q -> C (m*p)x(n*q),
        // a rank-2 ndRuntimeLocal. Each result element C(R,C) = A(R/p, C/q) * B(R%p, C%q)
        // (0-based; column-major). Both tiers via dimExpr; the runtime dst gets its
        // companions set. v1: two DOUBLE matrix vars, distinct from the dest.
        if (isArrayVar(name) && arrays_.at(name).isLocal && arrays_.at(name).ndRuntimeLocal
            && arrays_.at(name).ndDims.size() == 2 && rhs.type == NodeType::CALL
            && rhs.children.size() == 3 && rhs.children[0]->type == NodeType::IDENTIFIER
            && rhs.children[0]->strValue == "kron"
            && rhs.children[1]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[1]->strValue)
            && rhs.children[2]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[2]->strValue)
            && rhs.children[1]->strValue != name && rhs.children[2]->strValue != name) {
            const ArrayInfo &A = arrays_.at(rhs.children[1]->strValue);
            const ArrayInfo &B = arrays_.at(rhs.children[2]->strValue);
            const bool aMat = A.is2D || (A.isND && A.ndDims.size() == 2);
            const bool bMat = B.is2D || (B.isND && B.ndDims.size() == 2);
            if (aMat && bMat && A.dtype == ValueType::DOUBLE && B.dtype == ValueType::DOUBLE) {
                const ArrayInfo    &M   = arrays_.at(name);
                const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
                line("{");
                ++indent_;
                line("const std::size_t _numkit_m = " + dimExpr(A, 0) + ";");  // A rows
                line("const std::size_t _numkit_n = " + dimExpr(A, 1) + ";");  // A cols
                line("const std::size_t _numkit_p = " + dimExpr(B, 0) + ";");  // B rows
                line("const std::size_t _numkit_q = " + dimExpr(B, 1) + ";");  // B cols
                line("const std::size_t _numkit_mp = _numkit_m * _numkit_p;");         // result rows
                line("const std::size_t _numkit_nq = _numkit_n * _numkit_q;");         // result cols
                line(M.ndDims[0] + " = _numkit_mp;");
                line(M.ndDims[1] + " = _numkit_nq;");
                line(cppIdent(name) + ".assign(_numkit_mp * _numkit_nq, 0.0);");
                open("for (std::size_t _numkit_C = 0; _numkit_C < _numkit_nq; ++_numkit_C)");
                open("for (std::size_t _numkit_R = 0; _numkit_R < _numkit_mp; ++_numkit_R)");
                line(cppIdent(name) + "[_numkit_R + _numkit_C * _numkit_mp] = "
                     + A.dataExpr + "[(_numkit_R / _numkit_p) + (_numkit_C / _numkit_q) * _numkit_m] * "
                     + B.dataExpr + "[(_numkit_R % _numkit_p) + (_numkit_C % _numkit_q) * _numkit_p];");
                close();
                close();
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native cat(dim, A, B) with a LITERAL dim -> concatenate two 2-D matrices: dim==2 is
        // horizontal (M cols = A cols + B cols, a column-major buffer concat -- A's columns
        // then B's), dim==1 is vertical (M rows = A rows + B rows, a per-column interleave).
        // A runtime-dim 2-D result; both tiers via dimExpr; runtime dst companions set;
        // matching-dimension guard. v1: two DOUBLE matrix vars distinct from the dest.
        if (isArrayVar(name) && arrays_.at(name).isLocal && arrays_.at(name).ndRuntimeLocal
            && arrays_.at(name).ndDims.size() == 2 && rhs.type == NodeType::CALL
            && rhs.children.size() == 4 && rhs.children[0]->type == NodeType::IDENTIFIER
            && rhs.children[0]->strValue == "cat"
            && rhs.children[1]->type == NodeType::NUMBER_LITERAL
            && (rhs.children[1]->numValue == 1.0 || rhs.children[1]->numValue == 2.0)
            && rhs.children[2]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[2]->strValue)
            && rhs.children[3]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[3]->strValue)
            && rhs.children[2]->strValue != name && rhs.children[3]->strValue != name) {
            const ArrayInfo &A = arrays_.at(rhs.children[2]->strValue);
            const ArrayInfo &B = arrays_.at(rhs.children[3]->strValue);
            const bool aMat = A.is2D || (A.isND && A.ndDims.size() == 2);
            const bool bMat = B.is2D || (B.isND && B.ndDims.size() == 2);
            if (aMat && bMat && A.dtype == ValueType::DOUBLE && B.dtype == ValueType::DOUBLE) {
                const ArrayInfo    &M   = arrays_.at(name);
                const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
                const bool          horz = rhs.children[1]->numValue == 2.0;
                line("{");
                ++indent_;
                line("const std::size_t _numkit_ar = " + dimExpr(A, 0) + ";");  // A rows
                line("const std::size_t _numkit_ac = " + dimExpr(A, 1) + ";");  // A cols
                line("const std::size_t _numkit_br = " + dimExpr(B, 0) + ";");  // B rows
                line("const std::size_t _numkit_bc = " + dimExpr(B, 1) + ";");  // B cols
                if (horz) {  // cat(2,..): A cols then B cols; rows must agree
                    line("if (_numkit_ar != _numkit_br) throw std::out_of_range(\"numkit: cat dim-1 "
                         "sizes must agree\");");
                    line(M.ndDims[0] + " = _numkit_ar;");
                    line(M.ndDims[1] + " = _numkit_ac + _numkit_bc;");
                    line(cppIdent(name) + ".assign(_numkit_ar * (_numkit_ac + _numkit_bc), 0.0);");
                    open("for (std::size_t _numkit_k = 0; _numkit_k < _numkit_ar * _numkit_ac; ++_numkit_k)");
                    line(cppIdent(name) + "[_numkit_k] = " + A.dataExpr + "[_numkit_k];");
                    close();
                    open("for (std::size_t _numkit_k = 0; _numkit_k < _numkit_ar * _numkit_bc; ++_numkit_k)");
                    line(cppIdent(name) + "[_numkit_ar * _numkit_ac + _numkit_k] = " + B.dataExpr + "[_numkit_k];");
                    close();
                } else {  // cat(1,..): A rows then B rows interleaved per column; cols agree
                    line("if (_numkit_ac != _numkit_bc) throw std::out_of_range(\"numkit: cat dim-2 "
                         "sizes must agree\");");
                    line("const std::size_t _numkit_tr = _numkit_ar + _numkit_br;");
                    line(M.ndDims[0] + " = _numkit_tr;");
                    line(M.ndDims[1] + " = _numkit_ac;");
                    line(cppIdent(name) + ".assign(_numkit_tr * _numkit_ac, 0.0);");
                    open("for (std::size_t _numkit_j = 0; _numkit_j < _numkit_ac; ++_numkit_j)");
                    open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_ar; ++_numkit_i)");
                    line(cppIdent(name) + "[_numkit_i + _numkit_j * _numkit_tr] = " + A.dataExpr + "[_numkit_i + _numkit_j * _numkit_ar];");
                    close();
                    open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_br; ++_numkit_i)");
                    line(cppIdent(name) + "[(_numkit_ar + _numkit_i) + _numkit_j * _numkit_tr] = " + B.dataExpr
                         + "[_numkit_i + _numkit_j * _numkit_br];");
                    close();
                    close();
                }
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native cat(3, A, B, C, ...): stack N same-leading-size matrices/pages into a rank-3
        // array (phase N2 + N18 N-operand). Concat along the new TRAILING dim is a CONTIGUOUS
        // buffer append in column-major -- M = op0 pages ++ op1 pages ++ ... A 2-D operand is
        // one page; a rank-3 operand has size(.,3) pages. A rank-3 ndRuntimeLocal dest; dims
        // [m, n, sum(pages)]; per-operand page-size guard; running offset across operands. v1:
        // >=2 DOUBLE 2-D/rank-3 array vars distinct from the dest.
        if (isArrayVar(name) && arrays_.at(name).isLocal && arrays_.at(name).ndRuntimeLocal
            && arrays_.at(name).ndDims.size() == 3 && rhs.type == NodeType::CALL
            && rhs.children.size() >= 4 && rhs.children[0]->type == NodeType::IDENTIFIER
            && rhs.children[0]->strValue == "cat"
            && rhs.children[1]->type == NodeType::NUMBER_LITERAL
            && rhs.children[1]->numValue == 3.0) {
            const std::size_t nOps = rhs.children.size() - 2;
            bool              allOk = true;
            for (std::size_t i = 0; i < nOps && allOk; ++i) {
                const ASTNode &c = *rhs.children[2 + i];
                if (!(c.type == NodeType::IDENTIFIER && isArrayVar(c.strValue) && c.strValue != name)) {
                    allOk = false;
                    break;
                }
                const ArrayInfo &op = arrays_.at(c.strValue);
                if (!((op.is2D || (op.isND && (op.ndDims.size() == 2 || op.ndDims.size() == 3)))
                      && op.dtype == ValueType::DOUBLE))
                    allOk = false;
            }
            if (allOk) {
                const ArrayInfo    &M   = arrays_.at(name);
                const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
                const ArrayInfo    &A0  = arrays_.at(rhs.children[2]->strValue);
                line("{");
                ++indent_;
                line("const std::size_t _numkit_ar = " + dimExpr(A0, 0) + ";");
                line("const std::size_t _numkit_ac = " + dimExpr(A0, 1) + ";");
                line("const std::size_t _numkit_pg = _numkit_ar * _numkit_ac;");  // one-page element count
                std::string totalPages;
                for (std::size_t i = 0; i < nOps; ++i) {
                    const ArrayInfo  &op = arrays_.at(rhs.children[2 + i]->strValue);
                    if (i > 0)
                        line("if (" + dimExpr(op, 0) + " != _numkit_ar || " + dimExpr(op, 1)
                             + " != _numkit_ac) throw std::out_of_range(\"numkit: cat(3) page sizes "
                               "must agree\");");
                    const std::string pi =
                        (op.isND && op.ndDims.size() == 3) ? dimExpr(op, 2) : std::string("1");
                    line("const std::size_t _numkit_p" + std::to_string(i) + " = " + pi + ";");
                    totalPages += (i ? " + " : "") + ("_numkit_p" + std::to_string(i));
                }
                line(M.ndDims[0] + " = _numkit_ar;");
                line(M.ndDims[1] + " = _numkit_ac;");
                line(M.ndDims[2] + " = " + totalPages + ";");
                line(cppIdent(name) + ".assign(_numkit_pg * (" + totalPages + "), 0.0);");
                line("std::size_t _numkit_off = 0;");
                for (std::size_t i = 0; i < nOps; ++i) {
                    const ArrayInfo &op = arrays_.at(rhs.children[2 + i]->strValue);
                    line("{");
                    ++indent_;
                    line("const std::size_t _numkit_n = _numkit_pg * _numkit_p" + std::to_string(i) + ";");
                    open("for (std::size_t _numkit_k = 0; _numkit_k < _numkit_n; ++_numkit_k)");
                    line(cppIdent(name) + "[_numkit_off + _numkit_k] = " + op.dataExpr + "[_numkit_k];");
                    close();
                    line("_numkit_off += _numkit_n;");
                    --indent_;
                    line("}");
                }
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native cat(4, A, B, C, ...): append N operands along dim 4 -> rank-4 (m x n x p x
        // sum(slabs); phase N15 + N19 N-operand). The mirror of N-operand cat(3) one rank up --
        // a contiguous trailing-dim buffer append (M = op0 slabs ++ op1 slabs ++ ...). A rank-3
        // operand is one slab (m*n*p elems); a rank-4 operand has size(.,4) slabs. Operands share
        // the leading dims m,n,p. Running offset across operands; per-operand leading-dim guard.
        // v1: >=2 DOUBLE rank-3/rank-4 array vars distinct from the dest.
        if (isArrayVar(name) && arrays_.at(name).isLocal && arrays_.at(name).ndRuntimeLocal
            && arrays_.at(name).ndDims.size() == 4 && rhs.type == NodeType::CALL
            && rhs.children.size() >= 4 && rhs.children[0]->type == NodeType::IDENTIFIER
            && rhs.children[0]->strValue == "cat"
            && rhs.children[1]->type == NodeType::NUMBER_LITERAL
            && rhs.children[1]->numValue == 4.0) {
            const std::size_t nOps = rhs.children.size() - 2;
            bool              allOk = true;
            for (std::size_t i = 0; i < nOps && allOk; ++i) {
                const ASTNode &c = *rhs.children[2 + i];
                if (!(c.type == NodeType::IDENTIFIER && isArrayVar(c.strValue) && c.strValue != name)) {
                    allOk = false;
                    break;
                }
                const ArrayInfo &op = arrays_.at(c.strValue);
                if (!(op.isND && (op.ndDims.size() == 3 || op.ndDims.size() == 4)
                      && op.dtype == ValueType::DOUBLE))
                    allOk = false;
            }
            if (allOk) {
                const ArrayInfo    &M   = arrays_.at(name);
                const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
                const ArrayInfo    &A0  = arrays_.at(rhs.children[2]->strValue);
                line("{");
                ++indent_;
                line("const std::size_t _numkit_d0 = " + dimExpr(A0, 0) + ";");
                line("const std::size_t _numkit_d1 = " + dimExpr(A0, 1) + ";");
                line("const std::size_t _numkit_d2 = " + dimExpr(A0, 2) + ";");
                line("const std::size_t _numkit_sl = _numkit_d0 * _numkit_d1 * _numkit_d2;");  // one-slab elems
                std::string totalSlabs;
                for (std::size_t i = 0; i < nOps; ++i) {
                    const ArrayInfo &op = arrays_.at(rhs.children[2 + i]->strValue);
                    if (i > 0)
                        line("if (" + dimExpr(op, 0) + " != _numkit_d0 || " + dimExpr(op, 1)
                             + " != _numkit_d1 || " + dimExpr(op, 2)
                             + " != _numkit_d2) throw std::out_of_range(\"numkit: cat(4) leading dims "
                               "must agree\");");
                    const std::string si =
                        op.ndDims.size() == 4 ? dimExpr(op, 3) : std::string("1");
                    line("const std::size_t _numkit_s" + std::to_string(i) + " = " + si + ";");
                    totalSlabs += (i ? " + " : "") + ("_numkit_s" + std::to_string(i));
                }
                line(M.ndDims[0] + " = _numkit_d0;");
                line(M.ndDims[1] + " = _numkit_d1;");
                line(M.ndDims[2] + " = _numkit_d2;");
                line(M.ndDims[3] + " = " + totalSlabs + ";");
                line(cppIdent(name) + ".assign(_numkit_sl * (" + totalSlabs + "), 0.0);");
                line("std::size_t _numkit_off = 0;");
                for (std::size_t i = 0; i < nOps; ++i) {
                    const ArrayInfo &op = arrays_.at(rhs.children[2 + i]->strValue);
                    line("{");
                    ++indent_;
                    line("const std::size_t _numkit_n = _numkit_sl * _numkit_s" + std::to_string(i) + ";");
                    open("for (std::size_t _numkit_k = 0; _numkit_k < _numkit_n; ++_numkit_k)");
                    line(cppIdent(name) + "[_numkit_off + _numkit_k] = " + op.dataExpr + "[_numkit_k];");
                    close();
                    line("_numkit_off += _numkit_n;");
                    --indent_;
                    line("}");
                }
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native page-slice read B = A(:,:,k): extract page k of a rank-3 A as a 2-D m x n
        // matrix (phase N3). Column-major: page k (1-based) is the CONTIGUOUS block
        // A[(k-1)*m*n .. +m*n] -> a straight copy. B a runtime-dim 2-D ndRuntimeLocal; bounds-
        // checked k; `end` in the page index = size(A,3). v1: rank-3 DOUBLE A, two leading bare
        // colons + a scalar k; B distinct from A.
        if (isArrayVar(name) && arrays_.at(name).isLocal && arrays_.at(name).ndRuntimeLocal
            && arrays_.at(name).ndDims.size() == 2 && rhs.type == NodeType::CALL
            && rhs.children.size() == 4 && rhs.children[0]->type == NodeType::IDENTIFIER
            && isArrayVar(rhs.children[0]->strValue) && rhs.children[0]->strValue != name
            && arrays_.at(rhs.children[0]->strValue).isND
            && arrays_.at(rhs.children[0]->strValue).ndDims.size() == 3
            && arrays_.at(rhs.children[0]->strValue).dtype == ValueType::DOUBLE
            && rhs.children[1]->type == NodeType::COLON_EXPR && rhs.children[1]->children.empty()
            && rhs.children[2]->type == NodeType::COLON_EXPR && rhs.children[2]->children.empty()
            && rhs.children[3]->type != NodeType::COLON_EXPR) {
            const ArrayInfo    &A   = arrays_.at(rhs.children[0]->strValue);
            const ArrayInfo    &B   = arrays_.at(name);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete() && !res.type.shape.isScalar()
                && inferExpr(*rhs.children[3], types_, reg_, classes_).type.shape.isScalar()) {
                const std::string m = dimExpr(A, 0), n = dimExpr(A, 1), p = dimExpr(A, 2);
                endStack_.push_back(p);  // `end` in the page index = size(A,3)
                const std::string k = emitExpr(*rhs.children[3]);
                endStack_.pop_back();
                line("{");
                ++indent_;
                line("const std::size_t _numkit_m = " + m + ";");
                line("const std::size_t _numkit_n = " + n + ";");
                line("const std::size_t _numkit_k0 = numkit_rt::check_idx1(" + k + ", " + p
                     + ", \"numkit: page index out of bounds\");");
                line("const std::size_t _numkit_pg = _numkit_m * _numkit_n;");
                line("const std::size_t _numkit_off = _numkit_k0 * _numkit_pg;");
                line(B.ndDims[0] + " = _numkit_m;");
                line(B.ndDims[1] + " = _numkit_n;");
                line(cppIdent(name) + ".assign(" + A.dataExpr + " + _numkit_off, " + A.dataExpr
                     + " + _numkit_off + _numkit_pg);");
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native 2-D COLUMN-RANGE read B = A(:, j1:j2): extract columns j1..j2 of a 2-D A as a
        // runtime-dim 2-D rows x (j2-j1+1) sub-matrix (phase N22). Column-major: columns j1..j2
        // are the CONTIGUOUS block A[(j1-1)*rows .. j2*rows] -> a straight copy. B a runtime-dim
        // 2-D ndRuntimeLocal; bounds-checked range; `end` in the column index = cols. v1: 2-D
        // DOUBLE A (KnownDims or runtime), a leading bare colon + a step-1 range j1:j2, B distinct.
        if (isArrayVar(name) && arrays_.at(name).isLocal && arrays_.at(name).ndRuntimeLocal
            && arrays_.at(name).ndDims.size() == 2 && rhs.type == NodeType::CALL
            && rhs.children.size() == 3 && rhs.children[0]->type == NodeType::IDENTIFIER
            && isArrayVar(rhs.children[0]->strValue) && rhs.children[0]->strValue != name
            && (arrays_.at(rhs.children[0]->strValue).is2D
                || (arrays_.at(rhs.children[0]->strValue).isND
                    && arrays_.at(rhs.children[0]->strValue).ndDims.size() == 2))
            && arrays_.at(rhs.children[0]->strValue).dtype == ValueType::DOUBLE
            && rhs.children[1]->type == NodeType::COLON_EXPR && rhs.children[1]->children.empty()
            && rhs.children[2]->type == NodeType::COLON_EXPR && rhs.children[2]->children.size() == 2) {
            const ArrayInfo    &A   = arrays_.at(rhs.children[0]->strValue);
            const ArrayInfo    &B   = arrays_.at(name);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete() && !res.type.shape.isScalar()) {
                const std::string rows = dimExpr(A, 0), cols = dimExpr(A, 1);
                const ASTNode    &rng  = *rhs.children[2];  // j1:j2
                endStack_.push_back(cols);                  // `end` in the column index = cols
                const std::string j1 = emitExpr(*rng.children[0]);
                const std::string j2 = emitExpr(*rng.children[1]);
                endStack_.pop_back();
                line("{");
                ++indent_;
                line("const std::size_t _numkit_rows = " + rows + ";");
                line("const std::size_t _numkit_cols = " + cols + ";");
                line("const std::size_t _numkit_j1_0 = numkit_rt::check_idx1(" + j1
                     + ", _numkit_cols, \"numkit: column range out of bounds\");");
                line("const std::size_t _numkit_j2_0 = numkit_rt::check_idx1(" + j2
                     + ", _numkit_cols, \"numkit: column range out of bounds\");");
                line("if (_numkit_j2_0 < _numkit_j1_0)");
                line("    throw std::out_of_range(\"numkit: column range out of bounds\");");
                line("const std::size_t _numkit_nc = _numkit_j2_0 - _numkit_j1_0 + 1;");
                line("const std::size_t _numkit_off = _numkit_j1_0 * _numkit_rows;");
                line(B.ndDims[0] + " = _numkit_rows;");
                line(B.ndDims[1] + " = _numkit_nc;");
                line(cppIdent(name) + ".assign(" + A.dataExpr + " + _numkit_off, " + A.dataExpr
                     + " + _numkit_off + _numkit_nc * _numkit_rows);");
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native 2-D ROW-RANGE read B = A(i1:i2, :): extract rows i1..i2 of a 2-D A as a runtime-
        // dim 2-D (i2-i1+1) x cols sub-matrix (phase N23). STRIDED (a row block is not contiguous
        // in column-major): for each column j the kept rows are the run A[(i1-1)+j*rows ..
        // i2+j*rows). B a runtime-dim 2-D ndRuntimeLocal; bounds-checked range; `end` in the row
        // index = rows. v1: 2-D DOUBLE A, a leading step-1 range i1:i2 + a trailing bare colon, B
        // distinct.
        if (isArrayVar(name) && arrays_.at(name).isLocal && arrays_.at(name).ndRuntimeLocal
            && arrays_.at(name).ndDims.size() == 2 && rhs.type == NodeType::CALL
            && rhs.children.size() == 3 && rhs.children[0]->type == NodeType::IDENTIFIER
            && isArrayVar(rhs.children[0]->strValue) && rhs.children[0]->strValue != name
            && (arrays_.at(rhs.children[0]->strValue).is2D
                || (arrays_.at(rhs.children[0]->strValue).isND
                    && arrays_.at(rhs.children[0]->strValue).ndDims.size() == 2))
            && arrays_.at(rhs.children[0]->strValue).dtype == ValueType::DOUBLE
            && rhs.children[1]->type == NodeType::COLON_EXPR && rhs.children[1]->children.size() == 2
            && rhs.children[2]->type == NodeType::COLON_EXPR && rhs.children[2]->children.empty()) {
            const ArrayInfo    &A   = arrays_.at(rhs.children[0]->strValue);
            const ArrayInfo    &B   = arrays_.at(name);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete() && !res.type.shape.isScalar()) {
                const std::string rows = dimExpr(A, 0), cols = dimExpr(A, 1);
                const ASTNode    &rng  = *rhs.children[1];  // i1:i2
                endStack_.push_back(rows);                  // `end` in the row index = rows
                const std::string i1 = emitExpr(*rng.children[0]);
                const std::string i2 = emitExpr(*rng.children[1]);
                endStack_.pop_back();
                line("{");
                ++indent_;
                line("const std::size_t _numkit_rows = " + rows + ";");
                line("const std::size_t _numkit_cols = " + cols + ";");
                line("const std::size_t _numkit_r0 = numkit_rt::check_idx1(" + i1
                     + ", _numkit_rows, \"numkit: row range out of bounds\");");
                line("const std::size_t _numkit_i2_0 = numkit_rt::check_idx1(" + i2
                     + ", _numkit_rows, \"numkit: row range out of bounds\");");
                line("if (_numkit_i2_0 < _numkit_r0)");
                line("    throw std::out_of_range(\"numkit: row range out of bounds\");");
                line("const std::size_t _numkit_nr = _numkit_i2_0 - _numkit_r0 + 1;");
                line(B.ndDims[0] + " = _numkit_nr;");
                line(B.ndDims[1] + " = _numkit_cols;");
                line(cppIdent(name) + ".assign(_numkit_nr * _numkit_cols, 0.0);");
                open("for (std::size_t _numkit_j = 0; _numkit_j < _numkit_cols; ++_numkit_j)");
                open("for (std::size_t _numkit_r = 0; _numkit_r < _numkit_nr; ++_numkit_r)");
                line(cppIdent(name) + "[_numkit_r + _numkit_j * _numkit_nr] = " + A.dataExpr
                     + "[(_numkit_r0 + _numkit_r) + _numkit_j * _numkit_rows];");
                close();
                close();
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native 2-D BOTH-RANGE read B = A(i1:i2, j1:j2): extract the sub-block rows i1..i2 x
        // columns j1..j2 of a 2-D A as a runtime-dim 2-D (i2-i1+1) x (j2-j1+1) matrix (phase N24).
        // STRIDED: for each kept column cc = j1-1+c, copy the row run A[(i1-1)+cc*rows ..
        // i2+cc*rows) into B's column c. B a runtime-dim 2-D ndRuntimeLocal; both ranges bounds-
        // checked. v1: 2-D DOUBLE A, two step-1 ranges, B distinct.
        if (isArrayVar(name) && arrays_.at(name).isLocal && arrays_.at(name).ndRuntimeLocal
            && arrays_.at(name).ndDims.size() == 2 && rhs.type == NodeType::CALL
            && rhs.children.size() == 3 && rhs.children[0]->type == NodeType::IDENTIFIER
            && isArrayVar(rhs.children[0]->strValue) && rhs.children[0]->strValue != name
            && (arrays_.at(rhs.children[0]->strValue).is2D
                || (arrays_.at(rhs.children[0]->strValue).isND
                    && arrays_.at(rhs.children[0]->strValue).ndDims.size() == 2))
            && arrays_.at(rhs.children[0]->strValue).dtype == ValueType::DOUBLE
            && rhs.children[1]->type == NodeType::COLON_EXPR && rhs.children[1]->children.size() == 2
            && rhs.children[2]->type == NodeType::COLON_EXPR && rhs.children[2]->children.size() == 2) {
            const ArrayInfo    &A   = arrays_.at(rhs.children[0]->strValue);
            const ArrayInfo    &B   = arrays_.at(name);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete() && !res.type.shape.isScalar()) {
                const std::string rows = dimExpr(A, 0), cols = dimExpr(A, 1);
                const ASTNode    &rr = *rhs.children[1];  // i1:i2
                const ASTNode    &cr = *rhs.children[2];  // j1:j2
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
                line("const std::size_t _numkit_rows = " + rows + ";");
                line("const std::size_t _numkit_cols = " + cols + ";");
                line("const std::size_t _numkit_r0 = numkit_rt::check_idx1(" + i1
                     + ", _numkit_rows, \"numkit: row range out of bounds\");");
                line("const std::size_t _numkit_i2_0 = numkit_rt::check_idx1(" + i2
                     + ", _numkit_rows, \"numkit: row range out of bounds\");");
                line("if (_numkit_i2_0 < _numkit_r0)");
                line("    throw std::out_of_range(\"numkit: row range out of bounds\");");
                line("const std::size_t _numkit_c0 = numkit_rt::check_idx1(" + j1
                     + ", _numkit_cols, \"numkit: column range out of bounds\");");
                line("const std::size_t _numkit_j2_0 = numkit_rt::check_idx1(" + j2
                     + ", _numkit_cols, \"numkit: column range out of bounds\");");
                line("if (_numkit_j2_0 < _numkit_c0)");
                line("    throw std::out_of_range(\"numkit: column range out of bounds\");");
                line("const std::size_t _numkit_nr = _numkit_i2_0 - _numkit_r0 + 1;");
                line("const std::size_t _numkit_nc = _numkit_j2_0 - _numkit_c0 + 1;");
                line(B.ndDims[0] + " = _numkit_nr;");
                line(B.ndDims[1] + " = _numkit_nc;");
                line(cppIdent(name) + ".assign(_numkit_nr * _numkit_nc, 0.0);");
                open("for (std::size_t _numkit_c = 0; _numkit_c < _numkit_nc; ++_numkit_c)");
                open("for (std::size_t _numkit_r = 0; _numkit_r < _numkit_nr; ++_numkit_r)");
                line(cppIdent(name) + "[_numkit_r + _numkit_c * _numkit_nr] = " + A.dataExpr
                     + "[(_numkit_r0 + _numkit_r) + (_numkit_c0 + _numkit_c) * _numkit_rows];");
                close();
                close();
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native trailing-RANGE read B = A(:,...,:,k1:k2): extract the contiguous trailing-dim
        // block k1..k2 of a rank-r A (r >= 3) as a rank-r sub-array (phase N7 r=3 page-range +
        // N21 r=4 slab-range). Column-major: the leading (r-1) dims form a "slab" of size
        // d0*..*d(r-2), and the trailing-dim values k1..k2 are the CONTIGUOUS block
        // A[(k1-1)*slab .. k2*slab] -> a straight copy. B a runtime-dim rank-r ndRuntimeLocal;
        // bounds-checked range; `end` in the trailing index = size(A,r). v1: rank-3/4 DOUBLE A,
        // (r-1) leading bare colons + a step-1 range k1:k2, B distinct from A.
        if (isArrayVar(name) && arrays_.at(name).isLocal && arrays_.at(name).ndRuntimeLocal
            && rhs.type == NodeType::CALL && rhs.children[0]->type == NodeType::IDENTIFIER
            && isArrayVar(rhs.children[0]->strValue) && rhs.children[0]->strValue != name
            && arrays_.at(rhs.children[0]->strValue).isND
            && arrays_.at(rhs.children[0]->strValue).ndDims.size() >= 3
            && arrays_.at(rhs.children[0]->strValue).ndDims.size() == arrays_.at(name).ndDims.size()
            && arrays_.at(rhs.children[0]->strValue).dtype == ValueType::DOUBLE
            && rhs.children.size() == arrays_.at(rhs.children[0]->strValue).ndDims.size() + 1) {
            const ArrayInfo  &A = arrays_.at(rhs.children[0]->strValue);
            const std::size_t r = A.ndDims.size();
            bool              ok = rhs.children[r]->type == NodeType::COLON_EXPR
                      && rhs.children[r]->children.size() == 2;
            for (std::size_t i = 1; i < r && ok; ++i)
                if (!(rhs.children[i]->type == NodeType::COLON_EXPR
                      && rhs.children[i]->children.empty()))
                    ok = false;
            if (ok) {
                const ArrayInfo    &B   = arrays_.at(name);
                const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
                if (res.type.isConcrete() && !res.type.shape.isScalar()) {
                    const std::string lastDim = dimExpr(A, r - 1);  // size along the range dim
                    const ASTNode    &rng     = *rhs.children[r];   // k1:k2
                    endStack_.push_back(lastDim);                   // `end` in the range = size(A,r)
                    const std::string k1 = emitExpr(*rng.children[0]);
                    const std::string k2 = emitExpr(*rng.children[1]);
                    endStack_.pop_back();
                    line("{");
                    ++indent_;
                    std::string slabExpr;
                    for (std::size_t i = 0; i + 1 < r; ++i) {  // leading r-1 dims form one slab
                        line("const std::size_t _numkit_d" + std::to_string(i) + " = " + dimExpr(A, i)
                             + ";");
                        slabExpr += (i ? " * " : "") + ("_numkit_d" + std::to_string(i));
                    }
                    line("const std::size_t _numkit_slab = " + slabExpr + ";");
                    line("const std::size_t _numkit_last_d = " + lastDim + ";");
                    line("const std::size_t _numkit_k1_0 = numkit_rt::check_idx1(" + k1
                         + ", _numkit_last_d, \"numkit: range out of bounds\");");
                    line("const std::size_t _numkit_k2_0 = numkit_rt::check_idx1(" + k2
                         + ", _numkit_last_d, \"numkit: range out of bounds\");");
                    line("if (_numkit_k2_0 < _numkit_k1_0)");
                    line("    throw std::out_of_range(\"numkit: range out of bounds\");");
                    line("const std::size_t _numkit_nb = _numkit_k2_0 - _numkit_k1_0 + 1;");
                    line("const std::size_t _numkit_off = _numkit_k1_0 * _numkit_slab;");
                    for (std::size_t i = 0; i + 1 < r; ++i)
                        line(B.ndDims[i] + " = _numkit_d" + std::to_string(i) + ";");
                    line(B.ndDims[r - 1] + " = _numkit_nb;");
                    line(cppIdent(name) + ".assign(" + A.dataExpr + " + _numkit_off, " + A.dataExpr
                         + " + _numkit_off + _numkit_nb * _numkit_slab);");
                    --indent_;
                    line("}");
                    types_.set(name, res);
                    return;
                }
            }
        }
        // Native A(i,:,:) read: a LEADING-scalar strided slice of a rank-3 A -> a rank-3
        // [1, n, p] sub-array (phase N8). The fixed row i is kept as a singleton first dim
        // (only TRAILING scalar dims drop). STRIDED (not contiguous): B(1,j,k) = A(i,j,k);
        // A flat = (i-1) + j*m + k*m*n, B flat (dims [1,n,p]) = j + k*n. B a runtime-dim rank-3
        // ndRuntimeLocal; bounds-checked i (end = size(A,1)). v1: rank-3 DOUBLE A, a scalar i +
        // two trailing bare colons, B distinct from A.
        if (isArrayVar(name) && arrays_.at(name).isLocal && arrays_.at(name).ndRuntimeLocal
            && arrays_.at(name).ndDims.size() == 3 && rhs.type == NodeType::CALL
            && rhs.children.size() == 4 && rhs.children[0]->type == NodeType::IDENTIFIER
            && isArrayVar(rhs.children[0]->strValue) && rhs.children[0]->strValue != name
            && arrays_.at(rhs.children[0]->strValue).isND
            && arrays_.at(rhs.children[0]->strValue).ndDims.size() == 3
            && arrays_.at(rhs.children[0]->strValue).dtype == ValueType::DOUBLE
            && rhs.children[1]->type != NodeType::COLON_EXPR
            && rhs.children[2]->type == NodeType::COLON_EXPR && rhs.children[2]->children.empty()
            && rhs.children[3]->type == NodeType::COLON_EXPR && rhs.children[3]->children.empty()) {
            const ArrayInfo    &A   = arrays_.at(rhs.children[0]->strValue);
            const ArrayInfo    &B   = arrays_.at(name);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete() && !res.type.shape.isScalar()
                && inferExpr(*rhs.children[1], types_, reg_, classes_).type.shape.isScalar()) {
                const std::string m = dimExpr(A, 0), n = dimExpr(A, 1), p = dimExpr(A, 2);
                endStack_.push_back(m);  // `end` in the row index = size(A,1)
                const std::string i = emitExpr(*rhs.children[1]);
                endStack_.pop_back();
                line("{");
                ++indent_;
                line("const std::size_t _numkit_m = " + m + ";");
                line("const std::size_t _numkit_n = " + n + ";");
                line("const std::size_t _numkit_p = " + p + ";");
                line("const std::size_t _numkit_i0 = numkit_rt::check_idx1(" + i
                     + ", _numkit_m, \"numkit: row index out of bounds\");");
                line(B.ndDims[0] + " = 1;");
                line(B.ndDims[1] + " = _numkit_n;");
                line(B.ndDims[2] + " = _numkit_p;");
                line(cppIdent(name) + ".assign(_numkit_n * _numkit_p, 0.0);");
                open("for (std::size_t _numkit_k = 0; _numkit_k < _numkit_p; ++_numkit_k)");
                open("for (std::size_t _numkit_j = 0; _numkit_j < _numkit_n; ++_numkit_j)");
                line(cppIdent(name) + "[_numkit_j + _numkit_k * _numkit_n] = " + A.dataExpr
                     + "[_numkit_i0 + _numkit_j * _numkit_m + _numkit_k * _numkit_m * _numkit_n];");
                close();
                close();
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native A(:,j,:) read: a MIDDLE-scalar strided slice of a rank-3 A -> a rank-3
        // [m, 1, p] sub-array (phase N9, the sibling of N8). The fixed col j is kept as a
        // singleton middle dim. STRIDED: B(i,1,k) = A(i,j,k); A flat = i + (j-1)*m + k*m*n,
        // B flat (dims [m,1,p]) = i + k*m. v1: rank-3 DOUBLE A, colon + scalar j + colon, B
        // distinct from A; bounds-checked j (end = size(A,2)).
        if (isArrayVar(name) && arrays_.at(name).isLocal && arrays_.at(name).ndRuntimeLocal
            && arrays_.at(name).ndDims.size() == 3 && rhs.type == NodeType::CALL
            && rhs.children.size() == 4 && rhs.children[0]->type == NodeType::IDENTIFIER
            && isArrayVar(rhs.children[0]->strValue) && rhs.children[0]->strValue != name
            && arrays_.at(rhs.children[0]->strValue).isND
            && arrays_.at(rhs.children[0]->strValue).ndDims.size() == 3
            && arrays_.at(rhs.children[0]->strValue).dtype == ValueType::DOUBLE
            && rhs.children[1]->type == NodeType::COLON_EXPR && rhs.children[1]->children.empty()
            && rhs.children[2]->type != NodeType::COLON_EXPR
            && rhs.children[3]->type == NodeType::COLON_EXPR && rhs.children[3]->children.empty()) {
            const ArrayInfo    &A   = arrays_.at(rhs.children[0]->strValue);
            const ArrayInfo    &B   = arrays_.at(name);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete() && !res.type.shape.isScalar()
                && inferExpr(*rhs.children[2], types_, reg_, classes_).type.shape.isScalar()) {
                const std::string m = dimExpr(A, 0), n = dimExpr(A, 1), p = dimExpr(A, 2);
                endStack_.push_back(n);  // `end` in the col index = size(A,2)
                const std::string j = emitExpr(*rhs.children[2]);
                endStack_.pop_back();
                line("{");
                ++indent_;
                line("const std::size_t _numkit_m = " + m + ";");
                line("const std::size_t _numkit_n = " + n + ";");
                line("const std::size_t _numkit_p = " + p + ";");
                line("const std::size_t _numkit_j0 = numkit_rt::check_idx1(" + j
                     + ", _numkit_n, \"numkit: column index out of bounds\");");
                line(B.ndDims[0] + " = _numkit_m;");
                line(B.ndDims[1] + " = 1;");
                line(B.ndDims[2] + " = _numkit_p;");
                line(cppIdent(name) + ".assign(_numkit_m * _numkit_p, 0.0);");
                open("for (std::size_t _numkit_k = 0; _numkit_k < _numkit_p; ++_numkit_k)");
                open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_m; ++_numkit_i)");
                line(cppIdent(name) + "[_numkit_i + _numkit_k * _numkit_m] = " + A.dataExpr
                     + "[_numkit_i + _numkit_j0 * _numkit_m + _numkit_k * _numkit_m * _numkit_n];");
                close();
                close();
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native A(i,j,:) read: a FIBER of a rank-3 A -> a rank-3 [1, 1, p] sub-array (phase
        // N10). Two fixed dims kept as singletons (the trailing colon is not droppable).
        // STRIDED: B(1,1,k) = A(i,j,k); A flat = (i-1) + (j-1)*m + k*m*n, B flat (dims [1,1,p])
        // = k. B a runtime-dim rank-3 ndRuntimeLocal; bounds-checked i,j. v1: rank-3 DOUBLE A,
        // scalar i + scalar j + trailing bare colon, B distinct from A.
        if (isArrayVar(name) && arrays_.at(name).isLocal && arrays_.at(name).ndRuntimeLocal
            && arrays_.at(name).ndDims.size() == 3 && rhs.type == NodeType::CALL
            && rhs.children.size() == 4 && rhs.children[0]->type == NodeType::IDENTIFIER
            && isArrayVar(rhs.children[0]->strValue) && rhs.children[0]->strValue != name
            && arrays_.at(rhs.children[0]->strValue).isND
            && arrays_.at(rhs.children[0]->strValue).ndDims.size() == 3
            && arrays_.at(rhs.children[0]->strValue).dtype == ValueType::DOUBLE
            && rhs.children[1]->type != NodeType::COLON_EXPR
            && rhs.children[2]->type != NodeType::COLON_EXPR
            && rhs.children[3]->type == NodeType::COLON_EXPR && rhs.children[3]->children.empty()) {
            const ArrayInfo    &A   = arrays_.at(rhs.children[0]->strValue);
            const ArrayInfo    &B   = arrays_.at(name);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete() && !res.type.shape.isScalar()
                && inferExpr(*rhs.children[1], types_, reg_, classes_).type.shape.isScalar()
                && inferExpr(*rhs.children[2], types_, reg_, classes_).type.shape.isScalar()) {
                const std::string m = dimExpr(A, 0), n = dimExpr(A, 1), p = dimExpr(A, 2);
                endStack_.push_back(m);  // `end` in the row index = size(A,1)
                const std::string i = emitExpr(*rhs.children[1]);
                endStack_.pop_back();
                endStack_.push_back(n);  // `end` in the col index = size(A,2)
                const std::string j = emitExpr(*rhs.children[2]);
                endStack_.pop_back();
                line("{");
                ++indent_;
                line("const std::size_t _numkit_m = " + m + ";");
                line("const std::size_t _numkit_n = " + n + ";");
                line("const std::size_t _numkit_p = " + p + ";");
                line("const std::size_t _numkit_i0 = numkit_rt::check_idx1(" + i
                     + ", _numkit_m, \"numkit: fiber index out of bounds\");");
                line("const std::size_t _numkit_j0 = numkit_rt::check_idx1(" + j
                     + ", _numkit_n, \"numkit: fiber index out of bounds\");");
                line(B.ndDims[0] + " = 1;");
                line(B.ndDims[1] + " = 1;");
                line(B.ndDims[2] + " = _numkit_p;");
                line("const std::size_t _numkit_b0 = _numkit_i0 + _numkit_j0 * _numkit_m;");  // (i-1)+(j-1)*m
                line(cppIdent(name) + ".assign(_numkit_p, 0.0);");
                open("for (std::size_t _numkit_k = 0; _numkit_k < _numkit_p; ++_numkit_k)");
                line(cppIdent(name) + "[_numkit_k] = " + A.dataExpr + "[_numkit_b0 + _numkit_k * _numkit_m * _numkit_n];");
                close();
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native A(i,:,k) read: a (scalar, colon, scalar) slice of a rank-3 A -> a 2-D [1, n]
        // ROW (phase N11). Row i of page k; the trailing scalar k drops the page dim, the
        // leading scalar i keeps a singleton first dim. STRIDED: B(1,j) = A(i,j,k); A flat =
        // (i-1) + j*m + (k-1)*m*n, B flat (dims [1,n]) = j. B a runtime-dim 2-D ndRuntimeLocal;
        // bounds-checked i,k. v1: rank-3 DOUBLE A, scalar + colon + scalar, B distinct from A.
        if (isArrayVar(name) && arrays_.at(name).isLocal && arrays_.at(name).ndRuntimeLocal
            && arrays_.at(name).ndDims.size() == 2 && rhs.type == NodeType::CALL
            && rhs.children.size() == 4 && rhs.children[0]->type == NodeType::IDENTIFIER
            && isArrayVar(rhs.children[0]->strValue) && rhs.children[0]->strValue != name
            && arrays_.at(rhs.children[0]->strValue).isND
            && arrays_.at(rhs.children[0]->strValue).ndDims.size() == 3
            && arrays_.at(rhs.children[0]->strValue).dtype == ValueType::DOUBLE
            && rhs.children[1]->type != NodeType::COLON_EXPR
            && rhs.children[2]->type == NodeType::COLON_EXPR && rhs.children[2]->children.empty()
            && rhs.children[3]->type != NodeType::COLON_EXPR) {
            const ArrayInfo    &A   = arrays_.at(rhs.children[0]->strValue);
            const ArrayInfo    &B   = arrays_.at(name);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete() && !res.type.shape.isScalar()
                && inferExpr(*rhs.children[1], types_, reg_, classes_).type.shape.isScalar()
                && inferExpr(*rhs.children[3], types_, reg_, classes_).type.shape.isScalar()) {
                const std::string m = dimExpr(A, 0), n = dimExpr(A, 1), p = dimExpr(A, 2);
                endStack_.push_back(m);  // `end` in the row index = size(A,1)
                const std::string i = emitExpr(*rhs.children[1]);
                endStack_.pop_back();
                endStack_.push_back(p);  // `end` in the page index = size(A,3)
                const std::string k = emitExpr(*rhs.children[3]);
                endStack_.pop_back();
                line("{");
                ++indent_;
                line("const std::size_t _numkit_m = " + m + ";");
                line("const std::size_t _numkit_n = " + n + ";");
                line("const std::size_t _numkit_i0 = numkit_rt::check_idx1(" + i
                     + ", _numkit_m, \"numkit: slice index out of bounds\");");
                line("const std::size_t _numkit_k0 = numkit_rt::check_idx1(" + k + ", " + p
                     + ", \"numkit: slice index out of bounds\");");
                line(B.ndDims[0] + " = 1;");
                line(B.ndDims[1] + " = _numkit_n;");
                line("const std::size_t _numkit_b0 = _numkit_i0 + _numkit_k0 * _numkit_m * _numkit_n;");  // (i-1)+(k-1)*m*n
                line(cppIdent(name) + ".assign(_numkit_n, 0.0);");
                open("for (std::size_t _numkit_j = 0; _numkit_j < _numkit_n; ++_numkit_j)");
                line(cppIdent(name) + "[_numkit_j] = " + A.dataExpr + "[_numkit_b0 + _numkit_j * _numkit_m];");
                close();
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native A(:,j,k) read: a (colon, scalar, scalar) slice of a rank-3 A -> a 2-D [m, 1]
        // COLUMN (phase N12). Col j of page k; the trailing scalar k drops the page dim. Column-
        // major: this is the CONTIGUOUS block A[(j-1)*m + (k-1)*m*n .. +m] -> a straight copy
        // (unlike N11 which is strided). B a runtime-dim 2-D ndRuntimeLocal dims [m,1]; bounds-
        // checked j,k. v1: rank-3 DOUBLE A, colon + scalar + scalar, B distinct from A.
        if (isArrayVar(name) && arrays_.at(name).isLocal && arrays_.at(name).ndRuntimeLocal
            && arrays_.at(name).ndDims.size() == 2 && rhs.type == NodeType::CALL
            && rhs.children.size() == 4 && rhs.children[0]->type == NodeType::IDENTIFIER
            && isArrayVar(rhs.children[0]->strValue) && rhs.children[0]->strValue != name
            && arrays_.at(rhs.children[0]->strValue).isND
            && arrays_.at(rhs.children[0]->strValue).ndDims.size() == 3
            && arrays_.at(rhs.children[0]->strValue).dtype == ValueType::DOUBLE
            && rhs.children[1]->type == NodeType::COLON_EXPR && rhs.children[1]->children.empty()
            && rhs.children[2]->type != NodeType::COLON_EXPR
            && rhs.children[3]->type != NodeType::COLON_EXPR) {
            const ArrayInfo    &A   = arrays_.at(rhs.children[0]->strValue);
            const ArrayInfo    &B   = arrays_.at(name);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete() && !res.type.shape.isScalar()
                && inferExpr(*rhs.children[2], types_, reg_, classes_).type.shape.isScalar()
                && inferExpr(*rhs.children[3], types_, reg_, classes_).type.shape.isScalar()) {
                const std::string m = dimExpr(A, 0), n = dimExpr(A, 1), p = dimExpr(A, 2);
                endStack_.push_back(n);  // `end` in the col index = size(A,2)
                const std::string j = emitExpr(*rhs.children[2]);
                endStack_.pop_back();
                endStack_.push_back(p);  // `end` in the page index = size(A,3)
                const std::string k = emitExpr(*rhs.children[3]);
                endStack_.pop_back();
                line("{");
                ++indent_;
                line("const std::size_t _numkit_m = " + m + ";");
                line("const std::size_t _numkit_n = " + n + ";");
                line("const std::size_t _numkit_j0 = numkit_rt::check_idx1(" + j
                     + ", _numkit_n, \"numkit: slice index out of bounds\");");
                line("const std::size_t _numkit_k0 = numkit_rt::check_idx1(" + k + ", " + p
                     + ", \"numkit: slice index out of bounds\");");
                line(B.ndDims[0] + " = _numkit_m;");
                line(B.ndDims[1] + " = 1;");
                line("const std::size_t _numkit_off = _numkit_j0 * _numkit_m"
                     " + _numkit_k0 * _numkit_m * _numkit_n;");
                line(cppIdent(name) + ".assign(" + A.dataExpr + " + _numkit_off, " + A.dataExpr
                     + " + _numkit_off + _numkit_m);");
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native GENERAL scalar/colon slice B = A(s_0..s_{r-1}) for a rank r >= 4 A (phases
        // N8-N12 cover rank 3 per-pattern; this is the rank-4+ generalization). Each subscript
        // is a bare colon or a scalar (>=1 of each). Output dim per kept subscript: colon ->
        // size(A,k), scalar -> 1; trailing scalar dims drop (kept rank = lastColon+1, >=2).
        // STRIDED gather: A flat = base(from scalar dims) + sum over kept COLON dims of j_k *
        // As_k. v1: rank-4+ DOUBLE A, all bare-colon/scalar subscripts, B distinct from A.
        if (isArrayVar(name) && arrays_.at(name).isLocal && arrays_.at(name).ndRuntimeLocal
            && rhs.type == NodeType::CALL && rhs.children.size() >= 5
            && rhs.children[0]->type == NodeType::IDENTIFIER
            && isArrayVar(rhs.children[0]->strValue) && rhs.children[0]->strValue != name
            && arrays_.at(rhs.children[0]->strValue).isND
            && arrays_.at(rhs.children[0]->strValue).ndDims.size() >= 4
            && arrays_.at(rhs.children[0]->strValue).ndDims.size() == rhs.children.size() - 1
            && arrays_.at(rhs.children[0]->strValue).dtype == ValueType::DOUBLE) {
            const ArrayInfo  &A = arrays_.at(rhs.children[0]->strValue);
            const std::size_t r = A.ndDims.size();
            std::vector<bool> colon(r, false);
            int               lastCol = -1, scalarN = 0;
            bool              allCS = true;
            for (std::size_t k = 0; k < r; ++k) {
                const ASTNode &s     = *rhs.children[k + 1];
                const bool     isCol = s.type == NodeType::COLON_EXPR && s.children.empty();
                colon[k]             = isCol;
                if (isCol)
                    lastCol = static_cast<int>(k);
                else if (s.type != NodeType::COLON_EXPR)
                    ++scalarN;
                else {
                    allCS = false;  // a non-empty COLON (range) -> not handled here
                    break;
                }
            }
            const ArrayInfo    &B   = arrays_.at(name);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            bool scalarsOk = allCS && lastCol >= 0 && scalarN > 0 && res.type.isConcrete()
                             && !res.type.shape.isScalar();
            for (std::size_t k = 0; k < r && scalarsOk; ++k)
                if (!colon[k]
                    && !inferExpr(*rhs.children[k + 1], types_, reg_, classes_).type.shape.isScalar())
                    scalarsOk = false;
            const std::size_t keptRank =
                lastCol >= 0 ? (static_cast<std::size_t>(lastCol) + 1 < 2
                                    ? 2u
                                    : static_cast<std::size_t>(lastCol) + 1)
                             : 0;
            if (scalarsOk && B.ndDims.size() == keptRank) {
                line("{");
                ++indent_;
                line("const std::size_t _numkit_As0 = 1;");  // A column-major strides
                for (std::size_t k = 1; k < r; ++k)
                    line("const std::size_t _numkit_As" + std::to_string(k) + " = _numkit_As"
                         + std::to_string(k - 1) + " * (" + dimExpr(A, k - 1) + ");");
                line("std::size_t _numkit_base = 0;");  // fixed offset from the scalar subscripts
                for (std::size_t k = 0; k < r; ++k) {
                    if (colon[k]) continue;
                    endStack_.push_back(dimExpr(A, k));
                    const std::string sk = emitExpr(*rhs.children[k + 1]);
                    endStack_.pop_back();
                    line("{ const std::size_t _numkit_s = numkit_rt::check_idx1(" + sk + ", "
                         + dimExpr(A, k) + ", \"numkit: slice index out of bounds\");");
                    line("  _numkit_base += _numkit_s * _numkit_As" + std::to_string(k) + "; }");
                }
                for (std::size_t kk = 0; kk < keptRank; ++kk)  // output (kept) dims
                    line("const std::size_t _numkit_Bd" + std::to_string(kk) + " = "
                         + (colon[kk] ? dimExpr(A, kk) : std::string("1")) + ";");
                line("const std::size_t _numkit_Bs0 = 1;");  // output strides
                for (std::size_t kk = 1; kk < keptRank; ++kk)
                    line("const std::size_t _numkit_Bs" + std::to_string(kk) + " = _numkit_Bs"
                         + std::to_string(kk - 1) + " * _numkit_Bd" + std::to_string(kk - 1) + ";");
                std::string numel = "_numkit_Bd0";
                for (std::size_t kk = 1; kk < keptRank; ++kk) numel += " * _numkit_Bd" + std::to_string(kk);
                for (std::size_t kk = 0; kk < keptRank; ++kk)
                    line(B.ndDims[kk] + " = _numkit_Bd" + std::to_string(kk) + ";");
                line(cppIdent(name) + ".assign((" + numel + "), 0.0);");
                open("for (std::size_t _numkit_o = 0; _numkit_o < (" + numel + "); ++_numkit_o)");
                line("std::size_t _numkit_af = _numkit_base;");
                for (std::size_t kk = 0; kk < keptRank; ++kk)
                    if (colon[kk])
                        line("_numkit_af += ((_numkit_o / _numkit_Bs" + std::to_string(kk) + ") % _numkit_Bd"
                             + std::to_string(kk) + ") * _numkit_As" + std::to_string(kk) + ";");
                line(cppIdent(name) + "[_numkit_o] = " + A.dataExpr + "[_numkit_af];");
                close();
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native sort(A) on a 2-D MATRIX -> sort each COLUMN independently into a fresh same-shape
        // matrix (ascending default / 'descend'). Column-major: column j is the contiguous block
        // [j*m, (j+1)*m), so copy A then std::sort each column's block with the NaN-extremum
        // comparator (NaN last ascending / first descending, MATLAB order). EXACT but !bridge_
        // (preserve the bridged array-result tier when the bridge is on, like the 1-D form). v1: a
        // DOUBLE matrix var distinct from the dest; ascending or 'descend', single-output.
        if (isArrayVar(name) && arrays_.at(name).isLocal && !bridge_
            && (arrays_.at(name).is2D
                || (arrays_.at(name).isND && arrays_.at(name).ndDims.size() == 2))
            && rhs.type == NodeType::CALL
            && (rhs.children.size() == 2
                || (rhs.children.size() == 3 && rhs.children[2]->type == NodeType::STRING_LITERAL))
            && rhs.children[0]->type == NodeType::IDENTIFIER && rhs.children[0]->strValue == "sort"
            && rhs.children[1]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[1]->strValue)
            && rhs.children[1]->strValue != name
            && (arrays_.at(rhs.children[1]->strValue).is2D
                || (arrays_.at(rhs.children[1]->strValue).isND
                    && arrays_.at(rhs.children[1]->strValue).ndDims.size() == 2))
            && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE) {
            const ArrayInfo    &A       = arrays_.at(rhs.children[1]->strValue);
            const ArrayInfo    &M       = arrays_.at(name);
            const AbstractValue res     = inferExpr(rhs, types_, reg_, classes_);
            const bool          descend = rhs.children.size() == 3
                                          && rhs.children[2]->strValue == "descend";
            if (res.type.isConcrete() && !res.type.shape.isScalar()) {
                line("{");
                ++indent_;
                line("const std::size_t _numkit_m = " + dimExpr(A, 0) + ";");
                line("const std::size_t _numkit_n = " + dimExpr(A, 1) + ";");
                if (M.isND && M.ndDims.size() == 2) {  // runtime-dim dst: set its companions
                    line(M.ndDims[0] + " = _numkit_m;");
                    line(M.ndDims[1] + " = _numkit_n;");
                }
                line(cppIdent(name) + ".assign(" + A.dataExpr + ", " + A.dataExpr + " + _numkit_m * _numkit_n);");
                open("for (std::size_t _numkit_j = 0; _numkit_j < _numkit_n; ++_numkit_j)");
                line("std::sort(" + cppIdent(name) + ".begin() + _numkit_j * _numkit_m, " + cppIdent(name)
                     + ".begin() + (_numkit_j + 1) * _numkit_m,");
                line(descend
                         ? "    [](double _a, double _b){ return _a > _b || (_a != _a && _b == _b); });"
                         : "    [](double _a, double _b){ return _a < _b || (_b != _b && _a == _a); });");
                close();
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native sort(A, dim[, dir]) along a LITERAL dim 1|2|3 of a rank-3 array -> a fresh same-
        // shape rank-3 array, each fiber along the scan dim sorted (ascending default / 'descend'),
        // NaN at the extremum per MATLAB (last ascending / first descending). Reuses the cumsum
        // stride scheme; each fiber is gathered into a temp, std::sort-ed (NaN comparator), scattered
        // back. EXACT but !bridge_ (preserve the bridged tier when the bridge is on, like the 1-D/
        // 2-D forms). The explicit-dim form only (the default-dim sort is the first non-singleton --
        // ambiguous on a runtime-dim rank-3 -> bridged). children: [sort, A, dim] or [sort, A, dim,
        // dir]. v1: a DOUBLE rank-3 var distinct from the dest.
        {
            const bool sortDim =
                rhs.type == NodeType::CALL
                && (rhs.children.size() == 3
                    || (rhs.children.size() == 4
                        && rhs.children[3]->type == NodeType::STRING_LITERAL))
                && rhs.children[0]->type == NodeType::IDENTIFIER && rhs.children[0]->strValue == "sort"
                && rhs.children[2]->type == NodeType::NUMBER_LITERAL
                && (rhs.children[2]->numValue == 1.0 || rhs.children[2]->numValue == 2.0
                    || rhs.children[2]->numValue == 3.0);
            if (isArrayVar(name) && arrays_.at(name).isLocal && arrays_.at(name).ndRuntimeLocal
                && !bridge_ && arrays_.at(name).ndDims.size() == 3 && sortDim
                && rhs.children[1]->type == NodeType::IDENTIFIER
                && isArrayVar(rhs.children[1]->strValue) && rhs.children[1]->strValue != name
                && arrays_.at(rhs.children[1]->strValue).isND
                && arrays_.at(rhs.children[1]->strValue).ndDims.size() == 3
                && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE) {
                const ArrayInfo    &A   = arrays_.at(rhs.children[1]->strValue);
                const ArrayInfo    &M   = arrays_.at(name);
                const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
                if (res.type.isConcrete() && !res.type.shape.isScalar()) {
                    const int  dimv = static_cast<int>(rhs.children[2]->numValue);
                    const bool descend =
                        rhs.children.size() == 4 && rhs.children[3]->strValue == "descend";
                    const std::string sd = dimv == 1 ? "1" : (dimv == 2 ? "_numkit_m" : "_numkit_m * _numkit_n");
                    const std::string ld = dimv == 1 ? "_numkit_m" : (dimv == 2 ? "_numkit_n" : "_numkit_p");
                    line("{");
                    ++indent_;
                    line("const std::size_t _numkit_m = " + dimExpr(A, 0) + ";");
                    line("const std::size_t _numkit_n = " + dimExpr(A, 1) + ";");
                    line("const std::size_t _numkit_p = " + dimExpr(A, 2) + ";");
                    line(M.ndDims[0] + " = _numkit_m;");
                    line(M.ndDims[1] + " = _numkit_n;");
                    line(M.ndDims[2] + " = _numkit_p;");
                    line("const std::size_t _numkit_tot = _numkit_m * _numkit_n * _numkit_p;");
                    line("const std::size_t _numkit_sd = " + sd + ";");
                    line("const std::size_t _numkit_ld = " + ld + ";");
                    line("const std::size_t _numkit_blk = _numkit_sd * _numkit_ld;");
                    line(cppIdent(name) + ".assign(_numkit_tot, 0.0);");
                    line("std::vector<double> _numkit_f(_numkit_ld);");
                    open("for (std::size_t _numkit_b = 0; _numkit_b < _numkit_tot; _numkit_b += _numkit_blk)");
                    open("for (std::size_t _numkit_r = 0; _numkit_r < _numkit_sd; ++_numkit_r)");
                    open("for (std::size_t _numkit_t = 0; _numkit_t < _numkit_ld; ++_numkit_t)");
                    line("_numkit_f[_numkit_t] = " + A.dataExpr + "[_numkit_b + _numkit_r + _numkit_t * _numkit_sd];");
                    close();
                    line("std::sort(_numkit_f.begin(), _numkit_f.end(),");
                    line(descend
                             ? "    [](double _a, double _b){ return _a > _b || (_a != _a && _b == _b); });"
                             : "    [](double _a, double _b){ return _a < _b || (_b != _b && _a == _a); });");
                    open("for (std::size_t _numkit_t = 0; _numkit_t < _numkit_ld; ++_numkit_t)");
                    line(cppIdent(name) + "[_numkit_b + _numkit_r + _numkit_t * _numkit_sd] = _numkit_f[_numkit_t];");
                    close();
                    close();
                    close();
                    --indent_;
                    line("}");
                    types_.set(name, res);
                    return;
                }
            }
        }
        // Native sortrows(A) on a 2-D MATRIX -> rows sorted ascending lexicographically (by col 1,
        // ties by col 2, ...). Sort an m-row index vector with a lexicographic comparator (per
        // column, NaN-last: NaN sorts after numbers; equal/both-NaN -> next column), then gather the
        // rows in the new order into a fresh same-shape matrix. STABLE so fully-equal rows keep
        // their order. Column-major: A(i,k) = A[i + k*m]. !bridge_ guard. v1: a DOUBLE matrix var
        // distinct from the dest.
        if (isArrayVar(name) && arrays_.at(name).isLocal && !bridge_
            && (arrays_.at(name).is2D
                || (arrays_.at(name).isND && arrays_.at(name).ndDims.size() == 2))
            && rhs.type == NodeType::CALL && rhs.children.size() == 2
            && rhs.children[0]->type == NodeType::IDENTIFIER && rhs.children[0]->strValue == "sortrows"
            && rhs.children[1]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[1]->strValue)
            && rhs.children[1]->strValue != name
            && (arrays_.at(rhs.children[1]->strValue).is2D
                || (arrays_.at(rhs.children[1]->strValue).isND
                    && arrays_.at(rhs.children[1]->strValue).ndDims.size() == 2))
            && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE) {
            const ArrayInfo    &A   = arrays_.at(rhs.children[1]->strValue);
            const ArrayInfo    &M   = arrays_.at(name);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete() && !res.type.shape.isScalar()) {
                line("{");
                ++indent_;
                line("const std::size_t _numkit_m = " + dimExpr(A, 0) + ";");
                line("const std::size_t _numkit_n = " + dimExpr(A, 1) + ";");
                line("const double* _numkit_da = " + A.dataExpr + ";");
                if (M.isND && M.ndDims.size() == 2) {
                    line(M.ndDims[0] + " = _numkit_m;");
                    line(M.ndDims[1] + " = _numkit_n;");
                }
                line("std::vector<std::size_t> _numkit_p(_numkit_m);");
                open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_m; ++_numkit_i)");
                line("_numkit_p[_numkit_i] = _numkit_i;");
                close();
                line("std::stable_sort(_numkit_p.begin(), _numkit_p.end(), [_numkit_da, _numkit_m, _numkit_n]("
                     "std::size_t _ri, std::size_t _rj){ for (std::size_t _k = 0; _k < _numkit_n; ++_k) "
                     "{ double _a = _numkit_da[_ri + _k * _numkit_m], _b = _numkit_da[_rj + _k * _numkit_m]; if (_a "
                     "< _b || (_b != _b && _a == _a)) return true; if (_b < _a || (_a != _a && _b == "
                     "_b)) return false; } return false; });");
                line(cppIdent(name) + ".assign(_numkit_m * _numkit_n, 0.0);");
                open("for (std::size_t _numkit_k = 0; _numkit_k < _numkit_n; ++_numkit_k)");
                open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_m; ++_numkit_i)");
                line(cppIdent(name) + "[_numkit_i + _numkit_k * _numkit_m] = _numkit_da[_numkit_p[_numkit_i] + _numkit_k * _numkit_m];");
                close();
                close();
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native sort(x) ascending -> a sorted copy of x in a fresh 1-D LOCAL. The
        // comparator puts NaN last (MATLAB's order) and is a valid strict-weak-
        // ordering (NaN treated as the maximum), so std::sort stays well-defined even
        // with NaNs. Bridged otherwise; native closes it for the standalone tier.
        // v1: a single 1-D DOUBLE array VAR; ascending single-output (sort(x,'descend')
        // and [s,i]=sort are deferred). result a 1-D array LOCAL. `!bridge_` keeps the
        // bridged array-result path as the tier when the bridge is on (sort is exact,
        // but this preserves the existing bridged-sort behaviour); native is the
        // standalone (no-bridge) tier.
        if (isArrayVar(name) && arrays_.at(name).isLocal && !bridge_ && !arrays_.at(name).is2D
            && !arrays_.at(name).isND && rhs.type == NodeType::CALL
            && (rhs.children.size() == 2
                || (rhs.children.size() == 3 && rhs.children[2]->type == NodeType::STRING_LITERAL))
            && rhs.children[0]->type == NodeType::IDENTIFIER && rhs.children[0]->strValue == "sort"
            && rhs.children[1]->type == NodeType::IDENTIFIER
            && isArrayVar(rhs.children[1]->strValue)
            && !arrays_.at(rhs.children[1]->strValue).is2D
            && !arrays_.at(rhs.children[1]->strValue).isND
            && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE) {
            const ArrayInfo    &xa  = arrays_.at(rhs.children[1]->strValue);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            // 'descend' reverses the order; MATLAB puts NaN first descending (last
            // ascending). Both comparators are valid strict-weak-orderings (NaN as
            // the extremum), so std::sort stays well-defined with NaNs.
            const bool descend = rhs.children.size() == 3
                                 && rhs.children[2]->strValue == "descend";
            if (res.type.isConcrete() && !res.type.shape.isScalar()) {
                line("{");
                ++indent_;
                line(cppIdent(name) + ".assign(" + xa.dataExpr + ", " + xa.dataExpr + " + " + xa.lenVar
                     + ");");
                line("std::sort(" + cppIdent(name) + ".begin(), " + cppIdent(name) + ".end(),");
                line(descend
                         ? "    [](double _a, double _b){ return _a > _b || (_a != _a && _b == _b); });"
                         : "    [](double _a, double _b){ return _a < _b || (_b != _b && _a == _a); });");
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native unique(A) on a 2-D MATRIX -> the sorted distinct values as a 1-D COLUMN vector
        // (MATLAB flattens the matrix column-major). Same algorithm as the 1-D form on the flat
        // m*n buffer: sort a temp copy (NaN-last comparator) then keep each element differing from
        // the previous (NaN kept, each distinct). !bridge_ keeps the bridged array-result path as
        // the tier when the bridge is on. v1: a DOUBLE matrix var; result a 1-D array LOCAL.
        if (isArrayVar(name) && arrays_.at(name).isLocal && !bridge_ && !arrays_.at(name).is2D
            && !arrays_.at(name).isND && rhs.type == NodeType::CALL && rhs.children.size() == 2
            && rhs.children[0]->type == NodeType::IDENTIFIER && rhs.children[0]->strValue == "unique"
            && rhs.children[1]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[1]->strValue)
            && (arrays_.at(rhs.children[1]->strValue).is2D
                || (arrays_.at(rhs.children[1]->strValue).isND
                    && arrays_.at(rhs.children[1]->strValue).ndDims.size() == 2))
            && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE) {
            const ArrayInfo    &xa  = arrays_.at(rhs.children[1]->strValue);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete() && !res.type.shape.isScalar()) {
                line("{");
                ++indent_;
                line("const std::size_t _numkit_cnt = " + dimExpr(xa, 0) + " * " + dimExpr(xa, 1) + ";");
                line("std::vector<double> _numkit_t(" + xa.dataExpr + ", " + xa.dataExpr + " + _numkit_cnt);");
                line("std::sort(_numkit_t.begin(), _numkit_t.end(),");
                line("    [](double _a, double _b){ return _a < _b || (_b != _b && _a == _a); });");
                line(cppIdent(name) + ".clear();");
                open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_t.size(); ++_numkit_i)");
                line("if (_numkit_i == 0 || _numkit_t[_numkit_i] != _numkit_t[_numkit_i - 1]) " + cppIdent(name)
                     + ".push_back(_numkit_t[_numkit_i]);");
                close();
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native unique(x) -> the sorted distinct values in a fresh 1-D LOCAL. Sort a
        // temp copy (NaN-last comparator) then push each element that differs from the
        // previous (consecutive-equal skip). NaN are KEPT (MATLAB: each NaN distinct)
        // because NaN != NaN is true. !bridge_ keeps the bridged array-result path as
        // the tier when the bridge is on. v1: a single 1-D DOUBLE array var.
        if (isArrayVar(name) && arrays_.at(name).isLocal && !bridge_ && !arrays_.at(name).is2D
            && !arrays_.at(name).isND && rhs.type == NodeType::CALL && rhs.children.size() == 2
            && rhs.children[0]->type == NodeType::IDENTIFIER && rhs.children[0]->strValue == "unique"
            && rhs.children[1]->type == NodeType::IDENTIFIER
            && isArrayVar(rhs.children[1]->strValue)
            && !arrays_.at(rhs.children[1]->strValue).is2D
            && !arrays_.at(rhs.children[1]->strValue).isND
            && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE) {
            const ArrayInfo    &xa  = arrays_.at(rhs.children[1]->strValue);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete() && !res.type.shape.isScalar()) {
                line("{");
                ++indent_;
                line("std::vector<double> _numkit_t(" + xa.dataExpr + ", " + xa.dataExpr + " + "
                     + xa.lenVar + ");");
                line("std::sort(_numkit_t.begin(), _numkit_t.end(),");
                line("    [](double _a, double _b){ return _a < _b || (_b != _b && _a == _a); });");
                line(cppIdent(name) + ".clear();");
                open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_t.size(); ++_numkit_i)");
                line("if (_numkit_i == 0 || _numkit_t[_numkit_i] != _numkit_t[_numkit_i - 1]) " + cppIdent(name)
                     + ".push_back(_numkit_t[_numkit_i]);");
                close();
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native union(a, b) on two 1-D vectors -> the sorted distinct values of [a, b] (= unique
        // of the concatenation). Concatenate into a temp, sort with the NaN-last comparator, then
        // keep each element differing from the previous (NaN kept distinct, matching MATLAB union).
        // !bridge_ keeps the bridged array-result tier when the bridge is on. v1: two 1-D DOUBLE
        // array vars; result a 1-D array LOCAL.
        if (isArrayVar(name) && arrays_.at(name).isLocal && !bridge_ && !arrays_.at(name).is2D
            && !arrays_.at(name).isND && rhs.type == NodeType::CALL && rhs.children.size() == 3
            && rhs.children[0]->type == NodeType::IDENTIFIER && rhs.children[0]->strValue == "union"
            && rhs.children[1]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[1]->strValue)
            && rhs.children[2]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[2]->strValue)
            && !arrays_.at(rhs.children[1]->strValue).is2D
            && !arrays_.at(rhs.children[1]->strValue).isND
            && !arrays_.at(rhs.children[2]->strValue).is2D
            && !arrays_.at(rhs.children[2]->strValue).isND
            && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE
            && arrays_.at(rhs.children[2]->strValue).dtype == ValueType::DOUBLE) {
            const ArrayInfo    &aa  = arrays_.at(rhs.children[1]->strValue);
            const ArrayInfo    &bb  = arrays_.at(rhs.children[2]->strValue);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete() && !res.type.shape.isScalar()) {
                line("{");
                ++indent_;
                line("std::vector<double> _numkit_t;");
                line("_numkit_t.reserve(" + aa.lenVar + " + " + bb.lenVar + ");");
                line("_numkit_t.insert(_numkit_t.end(), " + aa.dataExpr + ", " + aa.dataExpr + " + "
                     + aa.lenVar + ");");
                line("_numkit_t.insert(_numkit_t.end(), " + bb.dataExpr + ", " + bb.dataExpr + " + "
                     + bb.lenVar + ");");
                line("std::sort(_numkit_t.begin(), _numkit_t.end(),");
                line("    [](double _a, double _b){ return _a < _b || (_b != _b && _a == _a); });");
                line(cppIdent(name) + ".clear();");
                open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_t.size(); ++_numkit_i)");
                line("if (_numkit_i == 0 || _numkit_t[_numkit_i] != _numkit_t[_numkit_i - 1]) " + cppIdent(name)
                     + ".push_back(_numkit_t[_numkit_i]);");
                close();
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native intersect(a, b) / setdiff(a, b) on two 1-D vectors -> sorted distinct values. Build
        // the sorted-unique ua (sort NaN-last + dedup, NaN kept distinct) and a sorted ub; then walk
        // ua -- intersect keeps v when v is non-NaN (v==v) AND present in ub (binary_search, ub sorted
        // by the same cmp); setdiff keeps v when v is NaN (v!=v, never in b) OR v is absent from ub.
        // The v==v / v!=v guards give MATLAB NaN semantics (NaN never matches) and short-circuit
        // before any NaN binary_search. !bridge_ guard. v1: two 1-D DOUBLE array vars; result a 1-D
        // LOCAL.
        if (isArrayVar(name) && arrays_.at(name).isLocal && !bridge_ && !arrays_.at(name).is2D
            && !arrays_.at(name).isND && rhs.type == NodeType::CALL && rhs.children.size() == 3
            && rhs.children[0]->type == NodeType::IDENTIFIER
            && (rhs.children[0]->strValue == "intersect" || rhs.children[0]->strValue == "setdiff")
            && rhs.children[1]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[1]->strValue)
            && rhs.children[2]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[2]->strValue)
            && !arrays_.at(rhs.children[1]->strValue).is2D
            && !arrays_.at(rhs.children[1]->strValue).isND
            && !arrays_.at(rhs.children[2]->strValue).is2D
            && !arrays_.at(rhs.children[2]->strValue).isND
            && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE
            && arrays_.at(rhs.children[2]->strValue).dtype == ValueType::DOUBLE) {
            const ArrayInfo    &aa          = arrays_.at(rhs.children[1]->strValue);
            const ArrayInfo    &bb          = arrays_.at(rhs.children[2]->strValue);
            const bool          isIntersect = rhs.children[0]->strValue == "intersect";
            const AbstractValue res         = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete() && !res.type.shape.isScalar()) {
                line("{");
                ++indent_;
                line("auto _numkit_cmp = [](double _a, double _b){ return _a < _b || (_b != _b && _a == "
                     "_a); };");
                line("std::vector<double> _numkit_ua(" + aa.dataExpr + ", " + aa.dataExpr + " + "
                     + aa.lenVar + ");");
                line("std::vector<double> _numkit_ub(" + bb.dataExpr + ", " + bb.dataExpr + " + "
                     + bb.lenVar + ");");
                line("std::sort(_numkit_ua.begin(), _numkit_ua.end(), _numkit_cmp);");
                line("std::sort(_numkit_ub.begin(), _numkit_ub.end(), _numkit_cmp);");
                line("_numkit_ua.erase(std::unique(_numkit_ua.begin(), _numkit_ua.end()), _numkit_ua.end());");
                line(cppIdent(name) + ".clear();");
                open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_ua.size(); ++_numkit_i)");
                line("const double _numkit_v = _numkit_ua[_numkit_i];");
                if (isIntersect)
                    line("if (_numkit_v == _numkit_v && std::binary_search(_numkit_ub.begin(), _numkit_ub.end(), "
                         "_numkit_v, _numkit_cmp)) "
                         + cppIdent(name) + ".push_back(_numkit_v);");
                else
                    line("if (_numkit_v != _numkit_v || !std::binary_search(_numkit_ub.begin(), _numkit_ub.end(), "
                         "_numkit_v, _numkit_cmp)) "
                         + cppIdent(name) + ".push_back(_numkit_v);");
                close();
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native single-output ismember(a, b) on two 1-D vectors -> a LOGICAL mask the length of a:
        // tf(i) = a(i) is a member of b. Sort b once, then per element binary_search (NaN-last
        // comparator); a(i) NaN -> false (v==v guard, MATLAB: NaN is never a member). Result a
        // LOGICAL (uint8) LOCAL. !bridge_ guard. v1: two 1-D DOUBLE array vars. (Only the single-
        // output form; [tf,loc]=ismember has no multi-transfer so it stays bridged.)
        if (isArrayVar(name) && arrays_.at(name).isLocal && !bridge_ && !arrays_.at(name).is2D
            && !arrays_.at(name).isND && rhs.type == NodeType::CALL && rhs.children.size() == 3
            && rhs.children[0]->type == NodeType::IDENTIFIER && rhs.children[0]->strValue == "ismember"
            && rhs.children[1]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[1]->strValue)
            && rhs.children[2]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[2]->strValue)
            && !arrays_.at(rhs.children[1]->strValue).is2D
            && !arrays_.at(rhs.children[1]->strValue).isND
            && !arrays_.at(rhs.children[2]->strValue).is2D
            && !arrays_.at(rhs.children[2]->strValue).isND
            && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE
            && arrays_.at(rhs.children[2]->strValue).dtype == ValueType::DOUBLE) {
            const ArrayInfo    &aa  = arrays_.at(rhs.children[1]->strValue);
            const ArrayInfo    &bb  = arrays_.at(rhs.children[2]->strValue);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete() && !res.type.shape.isScalar()) {
                line("{");
                ++indent_;
                line("auto _numkit_cmp = [](double _a, double _b){ return _a < _b || (_b != _b && _a == "
                     "_a); };");
                line("std::vector<double> _numkit_ub(" + bb.dataExpr + ", " + bb.dataExpr + " + "
                     + bb.lenVar + ");");
                line("std::sort(_numkit_ub.begin(), _numkit_ub.end(), _numkit_cmp);");
                line(cppIdent(name) + ".assign(" + aa.lenVar + ", 0);");  // LOGICAL uint8 buffer
                open("for (std::size_t _numkit_i = 0; _numkit_i < " + aa.lenVar + "; ++_numkit_i)");
                line("const double _numkit_v = " + aa.dataExpr + "[_numkit_i];");
                line(cppIdent(name) + "[_numkit_i] = (_numkit_v == _numkit_v && std::binary_search(_numkit_ub.begin(), "
                     "_numkit_ub.end(), _numkit_v, _numkit_cmp)) ? 1 : 0;");
                close();
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native polyval(p, x) with x a vector -> the polynomial p (coeffs highest
        // degree first) evaluated at each x[i] by Horner, a same-length-as-x 1-D
        // LOCAL: out[i] = ((p[0]*x[i] + p[1])*x[i] + ...) + p[np-1]. Exact (a fused
        // multiply-add chain) -> every tier. v1: p and x both 1-D DOUBLE array vars.
        if (isArrayVar(name) && arrays_.at(name).isLocal && !arrays_.at(name).is2D
            && !arrays_.at(name).isND && rhs.type == NodeType::CALL && rhs.children.size() == 3
            && rhs.children[0]->type == NodeType::IDENTIFIER
            && rhs.children[0]->strValue == "polyval"
            && rhs.children[1]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[1]->strValue)
            && rhs.children[2]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[2]->strValue)
            && !arrays_.at(rhs.children[1]->strValue).is2D
            && !arrays_.at(rhs.children[1]->strValue).isND
            && !arrays_.at(rhs.children[2]->strValue).is2D
            && !arrays_.at(rhs.children[2]->strValue).isND
            && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE
            && arrays_.at(rhs.children[2]->strValue).dtype == ValueType::DOUBLE) {
            const ArrayInfo    &p   = arrays_.at(rhs.children[1]->strValue);
            const ArrayInfo    &x   = arrays_.at(rhs.children[2]->strValue);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete() && !res.type.shape.isScalar()) {
                line("{");
                ++indent_;
                line(cppIdent(name) + ".assign(" + x.lenVar + ", 0.0);");
                open("for (std::size_t _numkit_i = 0; _numkit_i < " + x.lenVar + "; ++_numkit_i)");
                line("double _numkit_acc = 0.0;");
                open("for (std::size_t _numkit_k = 0; _numkit_k < " + p.lenVar + "; ++_numkit_k)");
                line("_numkit_acc = _numkit_acc * " + x.dataExpr + "[_numkit_i] + " + p.dataExpr + "[_numkit_k];");
                close();
                line(cppIdent(name) + "[_numkit_i] = _numkit_acc;");
                close();
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native upper(s)/lower(s) -> a char-array ASCII case transform (A-Z <-> a-z),
        // a runtime-sized 1-D char LOCAL. Self-contained. v1: a single 1-D CHAR
        // array var; a non-char arg is not handled here (-> bridged/refused).
        if (isArrayVar(name) && arrays_.at(name).isLocal && !arrays_.at(name).is2D
            && !arrays_.at(name).isND && rhs.type == NodeType::CALL
            && rhs.children.size() == 2 && rhs.children[0]->type == NodeType::IDENTIFIER
            && (rhs.children[0]->strValue == "upper" || rhs.children[0]->strValue == "lower")
            && rhs.children[1]->type == NodeType::IDENTIFIER
            && isArrayVar(rhs.children[1]->strValue)
            && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::CHAR) {
            const ArrayInfo &s = arrays_.at(rhs.children[1]->strValue);
            if (!s.is2D && !s.isND) {
                const bool          isUpper = rhs.children[0]->strValue == "upper";
                const AbstractValue rv      = inferExpr(rhs, types_, reg_, classes_);
                line("{");
                ++indent_;
                line(cppIdent(name) + ".clear();");
                open("for (std::size_t _numkit_i = 0; _numkit_i < " + s.lenVar + "; ++_numkit_i)");
                line("std::uint16_t _numkit_c = " + s.dataExpr + "[_numkit_i];");
                line(isUpper ? "if (_numkit_c >= 97 && _numkit_c <= 122) _numkit_c = std::uint16_t(_numkit_c - 32);"
                             : "if (_numkit_c >= 65 && _numkit_c <= 90) _numkit_c = std::uint16_t(_numkit_c + 32);");
                line(cppIdent(name) + ".push_back(_numkit_c);");
                close();
                --indent_;
                line("}");
                types_.set(name, rv);
                return;
            }
        }
        // Output array from a BRIDGED call (opt-in): y = sin(x). Sound ONLY
        // when inference proves the RHS is a concrete array (Contract 2); box
        // the (array-var / scalar) args, call the runtime (1 output), and
        // unbox into the caller-allocated out-param. v1: a DOUBLE output.
        if (bridge_ && isArrayVar(name) && !arrays_.at(name).is2D
            && !arrays_.at(name).isND  // 1-D dest only; an N-D bridge needs dim handling
            && (arrays_.at(name).isOutput || arrays_.at(name).isLocal)
            && (arrays_.at(name).dtype == ValueType::DOUBLE
                || arrays_.at(name).dtype == ValueType::COMPLEX
                // a CHAR (uint16) result is bridged into a LOCAL only (num2str/sprintf/...);
                // there is no bridge_into_char for a fixed output out-param.
                || (arrays_.at(name).dtype == ValueType::CHAR && arrays_.at(name).isLocal))
            && rhs.type == NodeType::CALL && !rhs.children.empty()
            && rhs.children[0]->type == NodeType::IDENTIFIER
            // An elementwise-math call (sin/erf/…) lowers NATIVELY below — only
            // bridge a call the emitter cannot lower (sort, fft, …).
            && unaryMathStd(rhs.children[0]->strValue) == nullptr
            && binaryMathStd(rhs.children[0]->strValue) == nullptr
            // A USER function is compiled, not a runtime builtin — never bridge it
            // (numkit_call by name would fail). It returns its 1-D array BY VALUE and
            // is assigned via the scalar tail (`name = <mangled>(args);`).
            && !(ctx_ && ctx_->funcs && !types_.has(rhs.children[0]->strValue)
                 && ctx_->funcs->has(rhs.children[0]->strValue))) {
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete() && !res.type.shape.isScalar()
                && (res.type.dtype == ValueType::DOUBLE
                    || res.type.dtype == ValueType::COMPLEX
                    || res.type.dtype == ValueType::CHAR)) {
                const ArrayInfo  &ai     = arrays_.at(name);
                const std::string callee = rhs.children[0]->strValue;
                const std::size_t nargs  = rhs.children.size() - 1;
                std::string       boxed;
                for (std::size_t i = 1; i < rhs.children.size(); ++i) {
                    const ASTNode &arg = *rhs.children[i];
                    if (i > 1) boxed += ", ";
                    if (arg.type == NodeType::IDENTIFIER && isArrayVar(arg.strValue)) {
                        const ArrayInfo &aai = arrays_.at(arg.strValue);
                        if (aai.is2D || (aai.isND && aai.ndDims.size() == 2)) {
                            const std::string ar = dimExpr(aai, 0), ac = dimExpr(aai, 1);
                            boxed += aai.dtype == ValueType::COMPLEX
                                         ? ("numkit_box_complex_matrix(reinterpret_cast<const double*>("
                                            + aai.dataExpr + "), " + ar + ", " + ac + ")")
                                         : ("numkit_box_matrix(" + aai.dataExpr + ", " + ar + ", "
                                            + ac + ")");
                        } else if (aai.isND) {
                            unsupported("bridged call: N-D array argument (v1)");
                        } else {
                            // a complex array boxes via the interleaved-re,im C ABI
                            boxed += aai.dtype == ValueType::COMPLEX
                                         ? ("numkit_box_complex_array(reinterpret_cast<const double*>("
                                            + aai.dataExpr + "), " + aai.lenVar + ")")
                                         : ("numkit_box_array(" + aai.dataExpr + ", " + aai.lenVar + ")");
                        }
                    } else if (arg.type == NodeType::STRING_LITERAL) {
                        // a CHAR LITERAL arg (e.g. a sprintf format) boxes as a CHAR row.
                        bool              ok  = false;
                        const std::string esc = cEscape(arg.strValue, ok);
                        if (!ok)
                            unsupported("bridged call: char-literal arg with a non-printable byte "
                                        "(v1)");
                        boxed += "numkit_box_string(\"" + esc + "\")";
                    } else {
                        if (inferExpr(arg, types_, reg_, classes_).type.dtype == ValueType::COMPLEX)
                            unsupported("bridged call: complex scalar argument (v1)");
                        boxed += "numkit_box_scalar(" + emitExpr(arg) + ")";
                    }
                }
                // A LOCAL resizes its owned vector (bridge_to_vec); the OUTPUT
                // fills its fixed, caller-sized out-param (bridge_into). A COMPLEX
                // result routes to the _cx variants (complex unbox).
                const bool        cx   = res.type.dtype == ValueType::COMPLEX;
                const bool        ch   = res.type.dtype == ValueType::CHAR;  // -> bridge_to_vec_char
                const std::string fn   =
                    ai.isLocal
                        ? (ch ? "numkit_rt::bridge_to_vec_char(\""
                              : (cx ? "numkit_rt::bridge_to_vec_cx(\"" : "numkit_rt::bridge_to_vec(\""))
                        : (cx ? "numkit_rt::bridge_into_cx(\"" : "numkit_rt::bridge_into(\"");
                const std::string dest = ai.isLocal ? (", " + cppIdent(name) + ");")
                                                    : (", " + cppIdent(name) + ", " + ai.lenVar + ");");
                if (nargs == 0) {
                    line(fn + callee + "\", nullptr, 0" + dest);
                } else {
                    open("");  // a fresh scope for the temporary arg array
                    line("numkit_val _numkit_args[] = { " + boxed + " };");
                    line(fn + callee + "\", _numkit_args, " + std::to_string(nargs) + dest);
                    close();
                }
                types_.set(name, {InferredType::concrete(ai.dtype, Shape::rowVector()),
                                  ConstVal::unknown()});
                return;
            }
        }
        // Elementwise array ARITHMETIC: dest = <expr over ONE whole array +
        // scalars> (self-contained). Sound only with a single distinct array
        // operand (no length-mismatch, no matrix semantics) and an
        // inference-proven array result. Emit a fill loop; a LOCAL is resized
        // to the operand's length, the OUTPUT uses its caller-sized length.
        if (isArrayVar(name)
            && (arrays_.at(name).dtype == ValueType::DOUBLE
                || arrays_.at(name).dtype == ValueType::COMPLEX
                || arrays_.at(name).dtype == ValueType::LOGICAL)) {
            const std::string bareRhsArr = resolveArrayVar(rhs);
            const bool isLocalWholeCopy =
                arrays_.at(name).isLocal && !bareRhsArr.empty() && bareRhsArr != name
                && arrays_.at(name).dtype == arrays_.at(bareRhsArr).dtype;
            std::set<std::string> srcArrays;
            if (!isLocalWholeCopy && collectElementwise(rhs, srcArrays) && !srcArrays.empty()) {
                const bool dst2D = arrays_.at(name).is2D;
                const bool dstND = arrays_.at(name).isND;
                // The flat per-element loop bounds on numel (column-major
                // storage, so elementwise is rank-agnostic over the flat
                // buffer). Rank discipline: every array operand must match the
                // dest's rank (and, for N-D, its rank count) — no implicit
                // broadcast. The soundness guard below enforces matching shapes
                // PER AXIS (rows/cols for 2-D, every dim for N-D — equal numel is
                // not equal shape); 1-D compares numel. Any number of array
                // operands at each rank. Explicit boundary, never wrong code.
                bool rankMismatch = false;
                for (const std::string &an : srcArrays) {
                    const ArrayInfo &sa = arrays_.at(an);
                    if (sa.is2D != dst2D || sa.isND != dstND
                        || (dstND && sa.ndDims.size() != arrays_.at(name).ndDims.size()))
                        rankMismatch = true;
                }
                const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
                if (!rankMismatch && res.type.isConcrete()
                    && !res.type.shape.isScalar()
                    && (res.type.dtype == ValueType::DOUBLE
                        || res.type.dtype == ValueType::COMPLEX
                        || res.type.dtype == ValueType::LOGICAL)) {
                    const ArrayInfo &ai = arrays_.at(name);
                    // Loop count = numel; the per-element loop is flat
                    // (column-major, so elementwise is rank-agnostic). SOUNDNESS
                    // guard = every array operand agrees with a reference shape
                    // (the OUTPUT's own dims, else the first operand's). 1-D
                    // compares numel; 2-D compares BOTH dims (equal numel is not
                    // equal shape — 2x3 vs 3x2). MATLAB errors on a mismatch too.
                    std::string bound, guard;
                    if (dst2D) {
                        const ArrayInfo &ref = ai.isLocal ? arrays_.at(*srcArrays.begin()) : ai;
                        bound = "(" + ref.rowsVar + " * " + ref.colsVar + ")";
                        for (const std::string &an : srcArrays) {
                            const ArrayInfo &sa = arrays_.at(an);
                            if (sa.rowsVar == ref.rowsVar && sa.colsVar == ref.colsVar) continue;
                            guard += (guard.empty() ? "" : " || ")
                                     + ("(" + sa.rowsVar + " != " + ref.rowsVar + " || "
                                        + sa.colsVar + " != " + ref.colsVar + ")");
                        }
                    } else if (dstND) {
                        // N-D: numel = product of the per-axis dims; the guard
                        // compares EVERY axis (equal numel is not equal shape —
                        // 2x3x4 vs 4x3x2). ref = the OUTPUT's dims, else the first
                        // operand's. Same-rank is already enforced above.
                        const ArrayInfo &ref = ai.isLocal ? arrays_.at(*srcArrays.begin()) : ai;
                        std::string      prod;
                        for (const std::string &d : ref.ndDims)
                            prod += (prod.empty() ? "" : " * ") + d;
                        bound = "(" + prod + ")";
                        for (const std::string &an : srcArrays) {
                            const ArrayInfo &sa = arrays_.at(an);
                            for (std::size_t k = 0; k < ref.ndDims.size(); ++k) {
                                if (sa.ndDims[k] == ref.ndDims[k]) continue;
                                guard += (guard.empty() ? "" : " || ")
                                         + (sa.ndDims[k] + " != " + ref.ndDims[k]);
                            }
                        }
                    } else {
                        bound = ai.isLocal ? arrays_.at(*srcArrays.begin()).lenVar : ai.lenVar;
                        for (const std::string &an : srcArrays) {
                            const std::string &alen = arrays_.at(an).lenVar;
                            if (alen == bound) continue;  // trivially equal
                            guard += (guard.empty() ? "" : " || ") + (alen + " != " + bound);
                        }
                    }
                    if (!guard.empty())
                        line("if (" + guard
                             + ") throw std::out_of_range(\"numkit: array dimensions must match\");");
                    if (ai.isLocal) line(cppIdent(name) + ".resize(" + bound + ");");
                    // A runtime-dim N-D LOCAL result (an ndRuntimeLocal, e.g. a
                    // runtime-dim 2-D matrix produced by C = A + B): copy the reference
                    // shape into the dst's OWN dim companions. The resize above only
                    // sizes the flat buffer; without this the companions stay 0 and the
                    // dst's later numel / N-D indexing read stale dims. (A KnownDims 2-D
                    // or 1-D dst has no companions -> this is a no-op for them.)
                    if (ai.isLocal && ai.ndRuntimeLocal) {
                        const ArrayInfo &ref = arrays_.at(*srcArrays.begin());
                        for (std::size_t k = 0; k < ai.ndDims.size(); ++k)
                            line(ai.ndDims[k] + " = " + ref.ndDims[k] + ";");
                    }
                    // Opt-in ops-kernel tier: a SINGLE binary op over exactly
                    // two whole DOUBLE arrays (out = a OP b) maps 1:1 to an ops
                    // SIMD kernel (flat over numel; internal small-N gate). Any
                    // scalar operand, compound/math expression, or complex dtype
                    // has no matching kernel -> the inline fill loop (which is
                    // also the self-contained default when ops kernels are off,
                    // and already auto-vectorises cheap arithmetic — A3).
                    const char *opsFn     = nullptr;  // binary: a OP b
                    const char *opsMathFn = nullptr;  // unary transcendental: fn(x)
                    if (opsKernels_ && ai.dtype == ValueType::DOUBLE) {
                        if (rhs.type == NodeType::BINARY_OP && rhs.children.size() == 2
                            && rhs.children[0]->type == NodeType::IDENTIFIER
                            && isArrayVar(rhs.children[0]->strValue)
                            && rhs.children[1]->type == NodeType::IDENTIFIER
                            && isArrayVar(rhs.children[1]->strValue)) {
                            const std::string &op = rhs.strValue;
                            opsFn = op == "+"    ? "plusDouble"
                                    : op == "-"  ? "minusDouble"
                                    : op == ".*" ? "timesDouble"
                                    : op == "./" ? "rdivideDouble"
                                                 : nullptr;
                        } else if (rhs.type == NodeType::CALL && rhs.children.size() == 2
                                   && rhs.children[0]->type == NodeType::IDENTIFIER
                                   && rhs.children[1]->type == NodeType::IDENTIFIER
                                   && isArrayVar(rhs.children[1]->strValue)) {
                            // fn(x): a single transcendental over a whole array.
                            opsMathFn = opsTranscendentalFn(rhs.children[0]->strValue);
                        }
                    }
                    if (opsFn) {
                        line("numkit::ops::" + std::string(opsFn) + "("
                             + arrays_.at(rhs.children[0]->strValue).dataExpr + ", "
                             + arrays_.at(rhs.children[1]->strValue).dataExpr + ", " + ai.dataExpr
                             + ", " + bound + ");");
                    } else if (opsMathFn) {
                        line("numkit::ops::" + std::string(opsMathFn) + "("
                             + arrays_.at(rhs.children[1]->strValue).dataExpr + ", " + ai.dataExpr
                             + ", " + bound + ");");
                    } else {
                        const ArrayInfo &ref = ai.isLocal ? arrays_.at(*srcArrays.begin()) : ai;
                        if (dst2D || (dstND && ref.ndDims.size() == 2)) {
                            elementRows_ = dimExpr(ref, 0);
                            elementCols_ = dimExpr(ref, 1);
                        } else if (!dst2D && !dstND) {
                            elementRows_ = ref.orient == VecOrient::Col ? bound : std::string("1");
                            elementCols_ = ref.orient == VecOrient::Row ? bound : std::string("1");
                        }
                        elementCtx_               = "_numkit_i";
                        const std::string rhsExpr = emitExpr(rhs);  // whole arrays -> [_numkit_i]
                        elementCtx_.clear();
                        elementRows_.clear();
                        elementCols_.clear();
                        open("for (std::size_t _numkit_i = 0; _numkit_i < " + bound + "; ++_numkit_i)");
                        line(ai.dataExpr + "[_numkit_i] = " + rhsExpr + ";");
                        close();
                    }
                    // A 2-D / N-D dest keeps its true dims (res); a 1-D dest a
                    // row stand-in (arrays_ drives indexing/queries either way).
                    if (dst2D || dstND)
                        types_.set(name, res);
                    else
                        types_.set(name, {InferredType::concrete(ai.dtype, Shape::rowVector()),
                                          ConstVal::unknown()});
                    return;
                }
                // 2-D + 1-D Implicit Expansion (Broadcasting): e.g. `B = A - mean(A)`,
                // `C = A + col_vec`, `C = (A - mu) ./ sigma`.
                // When dst is a 2-D matrix (`dst2D` or rank-2 `dstND`) and `srcArrays`
                // mixes at least one 2-D matrix with 1-D vectors (and no rank >= 3 N-D):
                const bool dstMat = dst2D || (dstND && arrays_.at(name).ndDims.size() == 2);
                if (rankMismatch && dstMat && res.type.isConcrete()
                    && !res.type.shape.isScalar()
                    && (res.type.dtype == ValueType::DOUBLE
                        || res.type.dtype == ValueType::COMPLEX
                        || res.type.dtype == ValueType::LOGICAL)) {
                    const ArrayInfo *refMat = nullptr;
                    bool             validMix = true;
                    for (const std::string &an : srcArrays) {
                        const ArrayInfo &sa = arrays_.at(an);
                        const bool       m2 = sa.is2D || (sa.isND && sa.ndDims.size() == 2);
                        const bool       v1 = !sa.is2D && !sa.isND
                                        && (sa.orient == VecOrient::Row || sa.orient == VecOrient::Col);
                        if (m2) {
                            if (!refMat) refMat = &sa;
                        } else if (!v1) {
                            validMix = false;
                            break;
                        }
                    }
                    if (validMix && refMat) {
                        const ArrayInfo  &ai      = arrays_.at(name);
                        const std::string refRows = dimExpr(*refMat, 0);
                        const std::string refCols = dimExpr(*refMat, 1);
                        std::string       guard;
                        for (const std::string &an : srcArrays) {
                            const ArrayInfo &sa = arrays_.at(an);
                            const bool       m2 = sa.is2D || (sa.isND && sa.ndDims.size() == 2);
                            if (m2) {
                                const std::string sr = dimExpr(sa, 0), sc = dimExpr(sa, 1);
                                if (sr != refRows || sc != refCols)
                                    guard += (guard.empty() ? "" : " || ")
                                             + ("(" + sr + " != " + refRows + " || " + sc + " != " + refCols + ")");
                            } else if (sa.orient == VecOrient::Row) {
                                if (sa.lenVar != refCols)
                                    guard += (guard.empty() ? "" : " || ") + ("(" + sa.lenVar + " != " + refCols + ")");
                            } else {
                                if (sa.lenVar != refRows)
                                    guard += (guard.empty() ? "" : " || ") + ("(" + sa.lenVar + " != " + refRows + ")");
                            }
                        }
                        if (!ai.isLocal) {
                            const std::string dr = dimExpr(ai, 0), dc = dimExpr(ai, 1);
                            if (dr != refRows || dc != refCols)
                                guard += (guard.empty() ? "" : " || ")
                                         + ("(" + dr + " != " + refRows + " || " + dc + " != " + refCols + ")");
                        }
                        if (!guard.empty())
                            line("if (" + guard
                                 + ") throw std::out_of_range(\"numkit: broadcast dimensions must agree\");");
                        if (ai.isLocal) {
                            line(cppIdent(name) + ".resize(" + refRows + " * " + refCols + ");");
                            if (ai.ndRuntimeLocal) {
                                line(ai.ndDims[0] + " = " + refRows + ";");
                                line(ai.ndDims[1] + " = " + refCols + ";");
                            }
                        }
                        elementIdxMap_.clear();
                        for (const std::string &an : srcArrays) {
                            const ArrayInfo &sa = arrays_.at(an);
                            const bool       m2 = sa.is2D || (sa.isND && sa.ndDims.size() == 2);
                            if (m2) {
                                elementIdxMap_[an] = "_numkit_i + _numkit_j * " + refRows;
                            } else if (sa.orient == VecOrient::Row) {
                                elementIdxMap_[an] = "_numkit_j";
                            } else {
                                elementIdxMap_[an] = "_numkit_i";
                            }
                        }
                        elementCtx_               = "_numkit_i + _numkit_j * " + refRows;
                        elementRows_              = refRows;
                        elementCols_              = refCols;
                        const std::string rhsExpr = emitExpr(rhs);
                        elementCtx_.clear();
                        elementRows_.clear();
                        elementCols_.clear();
                        elementIdxMap_.clear();
                        open("for (std::size_t _numkit_j = 0; _numkit_j < " + refCols + "; ++_numkit_j)");
                        open("for (std::size_t _numkit_i = 0; _numkit_i < " + refRows + "; ++_numkit_i)");
                        line(ai.dataExpr + "[_numkit_i + _numkit_j * " + refRows + "] = " + rhsExpr + ";");
                        close();
                        close();
                        types_.set(name, res);
                        return;
                    }
                }
            }
        }

        // Interproc 1-D array RETURN (typed): `arrLocal = g(args)` where g is a
        // compiled user function returning a 1-D array. The callee returns an
        // owned std::vector<double> (interprocByValueReturn); the array LOCAL is a
        // std::vector too, so this is a move/copy-assign. emitExpr routes the CALL
        // to emitUserCall (queues g's specialisation; allows the 1-D array result).
        // Only a LOCAL dest (the output out-param can't take a vector — that
        // vector->buffer copy is a later step; sound refusal below otherwise).
        if (isArrayVar(name) && arrays_.at(name).isLocal && rhs.type == NodeType::CALL
            && !rhs.children.empty() && rhs.children[0]->type == NodeType::IDENTIFIER
            && ctx_ && ctx_->funcs && !types_.has(rhs.children[0]->strValue)
            && ctx_->funcs->has(rhs.children[0]->strValue)) {
            const AbstractValue rv = inferExpr(rhs, types_, reg_, classes_);
            line(cppIdent(name) + " = " + emitExpr(rhs) + ";");  // std::vector<double> = g_spec(args)
            types_.set(name, rv);
            return;
        }

        // Interproc 1-D array RETURN into the OUTPUT out-param (P1.5): like the
        // LOCAL case but the dest is the caller-allocated buffer (T* + _len), so
        // copy the callee's self-describing vector into it, bounded by the
        // caller's _len — the same size contract the zeros-fill output already
        // trusts (the caller allocates the true output length). 1-D only; 2-D/N-D
        // results are refused upstream in emitUserCall (need dims with the buffer).
        if (isArrayVar(name) && arrays_.at(name).isOutput && !arrays_.at(name).is2D
            && !arrays_.at(name).isND && rhs.type == NodeType::CALL
            && !rhs.children.empty() && rhs.children[0]->type == NodeType::IDENTIFIER
            && ctx_ && ctx_->funcs && !types_.has(rhs.children[0]->strValue)
            && ctx_->funcs->has(rhs.children[0]->strValue)) {
            const AbstractValue rv = inferExpr(rhs, types_, reg_, classes_);
            const ArrayInfo    &ai = arrays_.at(name);
            line("{");
            ++indent_;
            line("const std::vector<" + cppArrayElemType(ai.dtype) + "> _numkit_ret = "
                 + emitExpr(rhs) + ";");
            line("const std::size_t _numkit_n = _numkit_ret.size() < (std::size_t)(" + ai.lenVar
                 + ") ? _numkit_ret.size() : (std::size_t)(" + ai.lenVar + ");");
            open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_n; ++_numkit_i)");
            line(ai.dataExpr + "[_numkit_i] = _numkit_ret[_numkit_i];");
            close();
            --indent_;
            line("}");
            types_.set(name, rv);
            return;
        }

        // LOGICAL-INDEXING READ: `y = x(m)` where x is an array var and m a LOGICAL
        // array (mask). Build a runtime-sized result by FILTERING x — the elements
        // of x where m is true (MATLAB x(logical)). v1: a single LOGICAL-array
        // subscript that is itself a VARIABLE (an inline mask would need
        // materialising first); x and m both 1-D; y a 1-D array LOCAL (push_back).
        // Bound on min(len) so a too-long mask cannot read x out of bounds.
        if (isArrayVar(name) && arrays_.at(name).isLocal && !arrays_.at(name).is2D
            && !arrays_.at(name).isND && rhs.type == NodeType::CALL
            && rhs.children.size() == 2 && rhs.children[0]->type == NodeType::IDENTIFIER
            && isArrayVar(rhs.children[0]->strValue)
            && rhs.children[1]->type == NodeType::IDENTIFIER
            && isArrayVar(rhs.children[1]->strValue)
            && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::LOGICAL) {
            const ArrayInfo &bx = arrays_.at(rhs.children[0]->strValue);  // x (source)
            const ArrayInfo &bm = arrays_.at(rhs.children[1]->strValue);  // m (mask)
            if (bx.is2D || bx.isND || bm.is2D || bm.isND)
                unsupported("logical indexing: a 1-D array + a 1-D mask only (v1)");
            line("{");
            ++indent_;
            line(cppIdent(name) + ".clear();");
            line("const std::size_t _numkit_n = " + bm.lenVar + " < " + bx.lenVar + " ? "
                 + bm.lenVar + " : " + bx.lenVar + ";");
            open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_n; ++_numkit_i)");
            line("if (" + bm.dataExpr + "[_numkit_i]) " + cppIdent(name) + ".push_back("
                 + bx.dataExpr + "[_numkit_i]);");
            close();
            --indent_;
            line("}");
            types_.set(name, {InferredType::concrete(bx.dtype, Shape::rowVector()),
                              ConstVal::unknown()});
            return;
        }

        // LOGICAL-INDEXING READ with an INLINE elementwise mask: `y = x(<expr>)` where <expr>
        // is a pure-elementwise LOGICAL self-mask over x (y = x(x>0), y = A(A>lo & A<hi), ...).
        // The sibling of the inline masked WRITE: the mask is FUSED into the filter loop (no
        // temp vector) -- elementCtx_ makes whole-x emit x[_numkit_i] (flat), then for each flat
        // element i, if the mask holds, push x[i]. Works for a 1-D vector OR a 2-D / N-D matrix
        // source (filter is flat over the column-major buffer; the result is the selected
        // elements in MATLAB linear order -> bound on NUMEL). A self-mask (collectElementwise
        // srcArrays=={x}) keeps it in bounds and per-element-emittable; a READ never mutates x,
        // so no aliasing subtlety. y a 1-D array LOCAL (push_back). Placed after the VAR-mask
        // read so a pre-bound mask var takes that branch.
        if (isArrayVar(name) && arrays_.at(name).isLocal && !arrays_.at(name).is2D
            && !arrays_.at(name).isND && rhs.type == NodeType::CALL
            && rhs.children.size() == 2 && rhs.children[0]->type == NodeType::IDENTIFIER
            && isArrayVar(rhs.children[0]->strValue)
            && rhs.children[1]->type != NodeType::IDENTIFIER) {
            const AbstractValue   maskAV = inferExpr(*rhs.children[1], types_, reg_, classes_);
            std::set<std::string> maskArrays;
            const bool            pureEw = collectElementwise(*rhs.children[1], maskArrays);
            if (maskAV.type.isConcrete() && maskAV.type.dtype == ValueType::LOGICAL
                && !maskAV.type.shape.isScalar() && pureEw && maskArrays.size() == 1
                && *maskArrays.begin() == rhs.children[0]->strValue) {
                const ArrayInfo &bx = arrays_.at(rhs.children[0]->strValue);  // x (source)
                std::string      numel;  // flat element count (1-D len / 2-D rows*cols / N-D prod)
                if (bx.isND) {
                    numel = bx.ndDims[0];
                    for (std::size_t i = 1; i < bx.ndDims.size(); ++i) numel += " * " + bx.ndDims[i];
                } else if (bx.is2D) {
                    numel = bx.rowsVar + " * " + bx.colsVar;
                } else {
                    numel = bx.lenVar;
                }
                line("{");
                ++indent_;
                line(cppIdent(name) + ".clear();");
                elementCtx_ = "_numkit_i";  // whole x in the mask -> x[_numkit_i] (flat)
                const std::string maskExpr = emitExpr(*rhs.children[1]);
                elementCtx_.clear();
                open("for (std::size_t _numkit_i = 0; _numkit_i < (" + numel + "); ++_numkit_i)");
                line("if (" + maskExpr + ") " + cppIdent(name) + ".push_back(" + bx.dataExpr + "[_numkit_i]);");
                close();
                --indent_;
                line("}");
                types_.set(name, {InferredType::concrete(bx.dtype, Shape::rowVector()),
                                  ConstVal::unknown()});
                return;
            }
        }

        // NUMERIC GATHER READ: `y = x(idx)` where idx is a 1-D NUMERIC (DOUBLE) index vector
        // -> y[i] = x(idx[i]) 1-based, i.e. reindexing / selection. Result length = numel(idx).
        // Each index is range+integer checked (MATLAB errors on a non-positive-integer or out-
        // of-range subscript) BEFORE the 1-based->0-based read, so there is no UB (a NaN fails
        // `>= 1.0`) and the throw is faithful. y a fresh 1-D LOCAL distinct from x and idx (an
        // in-place x = x(idx) would need a temp -> refused, sound). Placed after the logical
        // reads (a LOGICAL idx takes those; this is the numeric sibling). v1: x + idx 1-D, idx
        // DOUBLE; a 2-D x / inline-literal index / an int-typed idx -> refused.
        if (isArrayVar(name) && arrays_.at(name).isLocal && !arrays_.at(name).is2D
            && !arrays_.at(name).isND && rhs.type == NodeType::CALL && rhs.children.size() == 2
            && rhs.children[0]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[0]->strValue)
            && !arrays_.at(rhs.children[0]->strValue).is2D && !arrays_.at(rhs.children[0]->strValue).isND
            && rhs.children[1]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[1]->strValue)
            && !arrays_.at(rhs.children[1]->strValue).is2D && !arrays_.at(rhs.children[1]->strValue).isND
            && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE
            && rhs.children[0]->strValue != name && rhs.children[1]->strValue != name) {
            const ArrayInfo &bx = arrays_.at(rhs.children[0]->strValue);  // source
            const ArrayInfo &bi = arrays_.at(rhs.children[1]->strValue);  // numeric index vector
            line(cppIdent(name) + ".resize(static_cast<std::size_t>(" + bi.lenVar + "));");
            open("for (std::size_t _numkit_i = 0; _numkit_i < " + bi.lenVar + "; ++_numkit_i)");
            line("const double _numkit_d = " + bi.dataExpr + "[_numkit_i];");
            line("if (!(_numkit_d >= 1.0 && _numkit_d <= static_cast<double>(" + bx.lenVar
                 + ") && _numkit_d == std::floor(_numkit_d)))");
            line("    throw std::out_of_range(\"numkit: array index out of bounds\");");
            line(cppIdent(name) + "[_numkit_i] = " + bx.dataExpr + "[static_cast<std::size_t>(_numkit_d) - 1];");
            close();
            types_.set(name, {InferredType::concrete(bx.dtype, Shape::rowVector()),
                              ConstVal::unknown()});
            return;
        }

        // CHAR row-vector literal: `c = 'abc'` -> a char-array LOCAL initialised
        // from the literal's code units (1 per element; v1: ASCII/BMP, so a UTF-8
        // byte == the UTF-16 unit, matching the inference's byte-count length). char
        // is a uint16 buffer (cppArrayElemType). A 1x1 char ('a') is a Scalar ->
        // not handled here (char scalars deferred).
        if (isArrayVar(name) && arrays_.at(name).isLocal
            && arrays_.at(name).dtype == ValueType::CHAR
            && rhs.type == NodeType::STRING_LITERAL) {
            std::string codes;
            for (unsigned char ch : rhs.strValue)
                codes += (codes.empty() ? "" : ", ")
                         + ("std::uint16_t(" + std::to_string(static_cast<unsigned>(ch)) + ")");
            line(cppIdent(name) + ".assign({" + codes + "});");
            types_.set(name, inferExpr(rhs, types_, reg_, classes_));
            return;
        }

        // 1-D horzcat: `r = [a b ...]` (a single row of 1-D array operands of one
        // dtype) -> a runtime-sized 1-D array LOCAL built by appending each operand
        // in order. Works for char (uint16) + numeric. v1: every element is an
        // array VARIABLE; r a 1-D LOCAL. (Mixed-dtype / scalar element / multi-row
        // give a Dynamic or non-buffer inference -> skipped here.)
        if (isArrayVar(name) && arrays_.at(name).isLocal && !arrays_.at(name).is2D
            && !arrays_.at(name).isND && rhs.type == NodeType::MATRIX_LITERAL
            && rhs.children.size() == 1 && rhs.children[0]
            && !rhs.children[0]->children.empty()) {
            const AbstractValue rv = inferExpr(rhs, types_, reg_, classes_);
            if (rv.type.isConcrete() && !rv.type.shape.isScalar()) {
                const ASTNode &row = *rhs.children[0];
                line("{");
                ++indent_;
                line(cppIdent(name) + ".clear();");
                for (const auto &el : row.children) {
                    if (el->type == NodeType::IDENTIFIER && isArrayVar(el->strValue)) {
                        const ArrayInfo &op = arrays_.at(el->strValue);  // append a 1-D array
                        open("for (std::size_t _numkit_i = 0; _numkit_i < " + op.lenVar + "; ++_numkit_i)");
                        line(cppIdent(name) + ".push_back(" + op.dataExpr + "[_numkit_i]);");
                        close();
                    } else if (el->type == NodeType::STRING_LITERAL) {
                        for (unsigned char ch : el->strValue)  // append a char literal's units
                            line(cppIdent(name) + ".push_back(std::uint16_t("
                                 + std::to_string(static_cast<unsigned>(ch)) + "));");
                    } else if (inferExpr(*el, types_, reg_, classes_).type.shape.isScalar()) {
                        line(cppIdent(name) + ".push_back(" + emitExpr(*el) + ");");  // append one scalar
                    } else {
                        unsupported("horzcat element (v1: an array var, char/string literal, "
                                    "or scalar)");
                    }
                }
                --indent_;
                line("}");
                arrays_.at(name).orient = VecOrient::Row;
                types_.set(name, rv);
                return;
            }
        }
        // 2-D horzcat of columns: `M = [c1 c2 ... ck]` (single-row MATRIX_LITERAL, each
        // element a COLUMN-VECTOR var of a common length n) -> an n x k 2-D matrix, a
        // rank-2 ndRuntimeLocal. Column-major: column j is contiguous, M[i + j*n] =
        // c_j[i]. The k columns are distinct locals so the j-loop is unrolled; every
        // inner loop is bounded by the first column's length (the matrix's n). All
        // columns must share length n (precondition). v1: DOUBLE 1-D col vars.
        if (isArrayVar(name) && arrays_.at(name).isLocal && arrays_.at(name).ndRuntimeLocal
            && arrays_.at(name).ndDims.size() == 2 && rhs.type == NodeType::MATRIX_LITERAL
            && rhs.children.size() == 1 && rhs.children[0]
            && rhs.children[0]->children.size() >= 2) {
            bool allColVars = true;
            for (const auto &el : rhs.children[0]->children)
                if (!el || el->type != NodeType::IDENTIFIER || !isArrayVar(el->strValue)
                    || arrays_.at(el->strValue).is2D || arrays_.at(el->strValue).isND
                    || arrays_.at(el->strValue).dtype != ValueType::DOUBLE) {
                    allColVars = false;
                    break;
                }
            const AbstractValue rv = inferExpr(rhs, types_, reg_, classes_);
            if (allColVars && rv.type.isConcrete()) {
                const ArrayInfo  &M  = arrays_.at(name);
                const ArrayInfo  &c0 = arrays_.at(rhs.children[0]->children[0]->strValue);
                const std::string k  = std::to_string(rhs.children[0]->children.size());
                line("{");
                ++indent_;
                line("const std::size_t _numkit_n = " + c0.lenVar + ";");  // matrix rows
                line(M.ndDims[0] + " = _numkit_n;");
                line(M.ndDims[1] + " = " + k + ";");
                line(cppIdent(name) + ".assign(_numkit_n * " + k + ", 0.0);");
                for (std::size_t j = 0; j < rhs.children[0]->children.size(); ++j) {
                    const ArrayInfo &cj = arrays_.at(rhs.children[0]->children[j]->strValue);
                    open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_n; ++_numkit_i)");
                    line(cppIdent(name) + "[_numkit_i + " + std::to_string(j) + " * _numkit_n] = "
                         + cj.dataExpr + "[_numkit_i];");
                    close();
                }
                --indent_;
                line("}");
                types_.set(name, rv);
                return;
            }
        }
        // 2-D horzcat of MATRIX / COLUMN-VECTOR blocks: `M = [A B ...]` / `M = [A b]`
        // (augmented matrix) -- a single-row MATRIX_LITERAL whose every element is a 2-D
        // matrix var OR an n x 1 column vector, all of the same row count -> horizontal
        // concatenation, a rank-2 ndRuntimeLocal. In column-major each block's buffer IS
        // its columns in order, so the result is the buffers concatenated end to end: M
        // rows = block rows, M cols = sum of the blocks' cols (a column vector = 1 col).
        // dimExpr gives a matrix its rows/cols and a Col vector its (len, 1). A runtime
        // guard checks all blocks share the row count. v1: DOUBLE vars, >=2, none aliasing
        // the dest. (All-column-vector is the horzcat-of-columns case above.)
        if (isArrayVar(name) && arrays_.at(name).isLocal && arrays_.at(name).ndRuntimeLocal
            && arrays_.at(name).ndDims.size() == 2 && rhs.type == NodeType::MATRIX_LITERAL
            && rhs.children.size() == 1 && rhs.children[0]
            && rhs.children[0]->children.size() >= 2) {
            bool allBlockVars = true;
            for (const auto &el : rhs.children[0]->children)
                if (!el || el->type != NodeType::IDENTIFIER || !isArrayVar(el->strValue)
                    || el->strValue == name  // in-place horzcat -> fall through to refusal
                    || !(arrays_.at(el->strValue).is2D
                         || (arrays_.at(el->strValue).isND
                             && arrays_.at(el->strValue).ndDims.size() == 2)
                         || (!arrays_.at(el->strValue).is2D && !arrays_.at(el->strValue).isND
                             && arrays_.at(el->strValue).orient == VecOrient::Col))
                    || arrays_.at(el->strValue).dtype != ValueType::DOUBLE) {
                    allBlockVars = false;
                    break;
                }
            const AbstractValue rv = inferExpr(rhs, types_, reg_, classes_);
            if (allBlockVars && rv.type.isConcrete()) {
                const ArrayInfo  &M    = arrays_.at(name);
                const ArrayInfo  &op0  = arrays_.at(rhs.children[0]->children[0]->strValue);
                const std::string rows = dimExpr(op0, 0);
                std::string       totalCols;
                for (const auto &el : rhs.children[0]->children)
                    totalCols += (totalCols.empty() ? "" : " + ")
                                 + dimExpr(arrays_.at(el->strValue), 1);
                line("{");
                ++indent_;
                line(M.ndDims[0] + " = " + rows + ";");
                line(M.ndDims[1] + " = (" + totalCols + ");");
                for (std::size_t e = 1; e < rhs.children[0]->children.size(); ++e) {
                    const ArrayInfo &op = arrays_.at(rhs.children[0]->children[e]->strValue);
                    line("if (" + dimExpr(op, 0) + " != " + rows
                         + ") throw std::out_of_range(\"numkit: horzcat row dimensions must "
                           "agree\");");
                }
                line(cppIdent(name) + ".resize(" + rows + " * (" + totalCols + "));");
                line("std::size_t _numkit_off = 0;");
                for (const auto &el : rhs.children[0]->children) {
                    const ArrayInfo  &op = arrays_.at(el->strValue);
                    const std::string sz = "(" + dimExpr(op, 0) + " * " + dimExpr(op, 1) + ")";
                    open("for (std::size_t _numkit_k = 0; _numkit_k < " + sz + "; ++_numkit_k)");
                    line(cppIdent(name) + "[_numkit_off + _numkit_k] = " + op.dataExpr + "[_numkit_k];");
                    close();
                    line("_numkit_off += " + sz + ";");
                }
                --indent_;
                line("}");
                types_.set(name, rv);
                return;
            }
        }
        // 1-D vertcat of scalars: `v = [a; b; c]` (multi-row MATRIX_LITERAL, each row a
        // single scalar) -> a 1-D column LOCAL, pushing each row's scalar in order. The
        // column counterpart of the 1-D horzcat above. v1: scalar elements (a multi-
        // row stack of arrays/rows = a true 2-D vertcat, deferred).
        if (isArrayVar(name) && arrays_.at(name).isLocal && !arrays_.at(name).is2D
            && !arrays_.at(name).isND && rhs.type == NodeType::MATRIX_LITERAL
            && rhs.children.size() > 1) {
            bool allScalarRows = true;
            for (const auto &rowN : rhs.children)
                if (!rowN || rowN->children.size() != 1 || !rowN->children[0]
                    || !inferExpr(*rowN->children[0], types_, reg_, classes_).type.shape.isScalar()) {
                    allScalarRows = false;
                    break;
                }
            const AbstractValue rv = inferExpr(rhs, types_, reg_, classes_);
            if (allScalarRows && rv.type.isConcrete() && !rv.type.shape.isScalar()) {
                line("{");
                ++indent_;
                line(cppIdent(name) + ".clear();");
                for (const auto &rowN : rhs.children)
                    line(cppIdent(name) + ".push_back(" + emitExpr(*rowN->children[0]) + ");");
                --indent_;
                line("}");
                arrays_.at(name).orient = VecOrient::Col;
                types_.set(name, rv);
                return;
            }
        }
        // 2-D vertcat of rows: `M = [r1; r2; ...; rk]` (multi-row MATRIX_LITERAL, each
        // row a single ROW-VECTOR var of a common length n) -> a k x n 2-D matrix, a
        // rank-2 ndRuntimeLocal. Column-major: M[i + j*k] = r_i[j]. The k rows are
        // distinct locals so the i-loop is unrolled; every inner loop is bounded by the
        // FIRST row's length (the matrix's n) so a write can never escape the buffer.
        // All rows must share length n (precondition; MATLAB errors otherwise). v1:
        // DOUBLE 1-D row vars.
        if (isArrayVar(name) && arrays_.at(name).isLocal && arrays_.at(name).ndRuntimeLocal
            && arrays_.at(name).ndDims.size() == 2 && rhs.type == NodeType::MATRIX_LITERAL
            && rhs.children.size() > 1) {
            bool allRowVars = true;
            for (const auto &rowN : rhs.children)
                if (!rowN || rowN->children.size() != 1 || !rowN->children[0]
                    || rowN->children[0]->type != NodeType::IDENTIFIER
                    || !isArrayVar(rowN->children[0]->strValue)
                    || arrays_.at(rowN->children[0]->strValue).is2D
                    || arrays_.at(rowN->children[0]->strValue).isND
                    || arrays_.at(rowN->children[0]->strValue).dtype != ValueType::DOUBLE) {
                    allRowVars = false;
                    break;
                }
            const AbstractValue rv = inferExpr(rhs, types_, reg_, classes_);
            if (allRowVars && rv.type.isConcrete()) {
                const ArrayInfo  &M  = arrays_.at(name);
                const ArrayInfo  &r0 = arrays_.at(rhs.children[0]->children[0]->strValue);
                const std::string k  = std::to_string(rhs.children.size());
                line("{");
                ++indent_;
                line("const std::size_t _numkit_n = " + r0.lenVar + ";");  // matrix cols
                line(M.ndDims[0] + " = " + k + ";");                   // rows = k
                line(M.ndDims[1] + " = _numkit_n;");
                line(cppIdent(name) + ".assign(" + k + " * _numkit_n, 0.0);");
                for (std::size_t i = 0; i < rhs.children.size(); ++i) {
                    const ArrayInfo &ri = arrays_.at(rhs.children[i]->children[0]->strValue);
                    open("for (std::size_t _numkit_j = 0; _numkit_j < _numkit_n; ++_numkit_j)");
                    line(cppIdent(name) + "[" + std::to_string(i) + " + _numkit_j * " + k + "] = "
                         + ri.dataExpr + "[_numkit_j];");
                    close();
                }
                --indent_;
                line("}");
                types_.set(name, rv);
                return;
            }
        }
        // 2-D vertcat of MATRIX / ROW-VECTOR blocks: `M = [A; B; ...]` / `M = [A; r]`
        // (appending a row) -- a multi-row MATRIX_LITERAL whose every row is a single 2-D
        // matrix var OR a 1 x n row vector, all of the same column count -> vertical
        // concatenation, a rank-2 ndRuntimeLocal. M rows = sum of block rows (a row vector
        // = 1), M cols = shared. In column-major the blocks INTERLEAVE per column: for a
        // block at row-offset ro with pr rows, M[(ro+ii) + j*totalRows] = op[ii + j*pr]
        // (pr=1 for a row vector). dimExpr gives a Row vector its ("1", len). A runtime
        // guard checks the column counts agree. v1: DOUBLE vars, >=2 rows, none aliasing the
        // dest. (All-row-vector is the vertcat-of-rows case above.)
        if (isArrayVar(name) && arrays_.at(name).isLocal && arrays_.at(name).ndRuntimeLocal
            && arrays_.at(name).ndDims.size() == 2 && rhs.type == NodeType::MATRIX_LITERAL
            && rhs.children.size() > 1) {
            bool allMatRows = true;
            for (const auto &rowN : rhs.children)
                if (!rowN || rowN->children.size() != 1 || !rowN->children[0]
                    || rowN->children[0]->type != NodeType::IDENTIFIER
                    || !isArrayVar(rowN->children[0]->strValue)
                    || rowN->children[0]->strValue == name  // in-place -> fall through
                    || !(arrays_.at(rowN->children[0]->strValue).is2D
                         || (arrays_.at(rowN->children[0]->strValue).isND
                             && arrays_.at(rowN->children[0]->strValue).ndDims.size() == 2)
                         || (!arrays_.at(rowN->children[0]->strValue).is2D
                             && !arrays_.at(rowN->children[0]->strValue).isND
                             && arrays_.at(rowN->children[0]->strValue).orient == VecOrient::Row))
                    || arrays_.at(rowN->children[0]->strValue).dtype != ValueType::DOUBLE) {
                    allMatRows = false;
                    break;
                }
            const AbstractValue rv = inferExpr(rhs, types_, reg_, classes_);
            if (allMatRows && rv.type.isConcrete()) {
                const ArrayInfo  &M    = arrays_.at(name);
                const ArrayInfo  &op0  = arrays_.at(rhs.children[0]->children[0]->strValue);
                const std::string cols = dimExpr(op0, 1);
                std::string       totalRows;
                for (const auto &rowN : rhs.children)
                    totalRows += (totalRows.empty() ? "" : " + ")
                                 + dimExpr(arrays_.at(rowN->children[0]->strValue), 0);
                line("{");
                ++indent_;
                line("const std::size_t _numkit_tr = (" + totalRows + ");");  // total rows
                line("const std::size_t _numkit_nc = " + cols + ";");          // shared cols
                line(M.ndDims[0] + " = _numkit_tr;");
                line(M.ndDims[1] + " = _numkit_nc;");
                for (std::size_t e = 1; e < rhs.children.size(); ++e) {
                    const ArrayInfo &op = arrays_.at(rhs.children[e]->children[0]->strValue);
                    line("if (" + dimExpr(op, 1) + " != _numkit_nc) throw std::out_of_range(\""
                         "numkit: vertcat column dimensions must agree\");");
                }
                line(cppIdent(name) + ".resize(_numkit_tr * _numkit_nc);");
                line("std::size_t _numkit_ro = 0;");
                for (const auto &rowN : rhs.children) {
                    const ArrayInfo &op = arrays_.at(rowN->children[0]->strValue);
                    line("{");
                    ++indent_;
                    line("const std::size_t _numkit_pr = " + dimExpr(op, 0) + ";");
                    open("for (std::size_t _numkit_j = 0; _numkit_j < _numkit_nc; ++_numkit_j)");
                    open("for (std::size_t _numkit_ii = 0; _numkit_ii < _numkit_pr; ++_numkit_ii)");
                    line(cppIdent(name) + "[(_numkit_ro + _numkit_ii) + _numkit_j * _numkit_tr] = "
                         + op.dataExpr + "[_numkit_ii + _numkit_j * _numkit_pr];");
                    close();
                    close();
                    line("_numkit_ro += _numkit_pr;");
                    --indent_;
                    line("}");
                }
                --indent_;
                line("}");
                types_.set(name, rv);
                return;
            }
        }
        // BLOCK-matrix literal `M = [A B; C D]`: a multi-row MATRIX_LITERAL whose every row
        // is a HORZCAT of >=1 matrix vars (the >1-block rows the single-matrix vertcat above
        // does not match). Result (sum row-heights) x (common total cols), a rank-2
        // ndRuntimeLocal. Each block is copied column-major into its (rowOff, colOff) region;
        // runtime guards check the within-row block heights agree and each row's total width
        // matches the common width. v1: DOUBLE matrix vars, none aliasing the dest.
        if (isArrayVar(name) && arrays_.at(name).isLocal && arrays_.at(name).ndRuntimeLocal
            && arrays_.at(name).ndDims.size() == 2 && rhs.type == NodeType::MATRIX_LITERAL
            && rhs.children.size() > 1) {
            bool allBlocks = true;
            for (const auto &rowN : rhs.children) {
                if (!rowN || rowN->children.empty()) { allBlocks = false; break; }
                for (const auto &el : rowN->children)
                    if (!el || el->type != NodeType::IDENTIFIER || !isArrayVar(el->strValue)
                        || el->strValue == name
                        || !(arrays_.at(el->strValue).is2D
                             || (arrays_.at(el->strValue).isND
                                 && arrays_.at(el->strValue).ndDims.size() == 2))
                        || arrays_.at(el->strValue).dtype != ValueType::DOUBLE) {
                        allBlocks = false;
                        break;
                    }
                if (!allBlocks) break;
            }
            const AbstractValue rv = inferExpr(rhs, types_, reg_, classes_);
            if (allBlocks && rv.type.isConcrete()) {
                const ArrayInfo &M = arrays_.at(name);
                std::string      P, Q;  // P = sum of row heights; Q = row 0's total width
                for (const auto &rowN : rhs.children)
                    P += (P.empty() ? "" : " + ")
                         + dimExpr(arrays_.at(rowN->children[0]->strValue), 0);
                for (const auto &el : rhs.children[0]->children)
                    Q += (Q.empty() ? "" : " + ") + dimExpr(arrays_.at(el->strValue), 1);
                line("{");
                ++indent_;
                line("const std::size_t _numkit_P = (" + P + ");");
                line("const std::size_t _numkit_Q = (" + Q + ");");
                line(M.ndDims[0] + " = _numkit_P;");
                line(M.ndDims[1] + " = _numkit_Q;");
                line(cppIdent(name) + ".assign(_numkit_P * _numkit_Q, 0.0);");
                line("std::size_t _numkit_ro = 0;");
                for (const auto &rowN : rhs.children) {
                    line("{");
                    ++indent_;
                    line("const std::size_t _numkit_rh = "
                         + dimExpr(arrays_.at(rowN->children[0]->strValue), 0) + ";");
                    line("std::size_t _numkit_co = 0;");
                    for (const auto &el : rowN->children) {
                        const ArrayInfo &b = arrays_.at(el->strValue);
                        line("{");
                        ++indent_;
                        line("const std::size_t _numkit_br = " + dimExpr(b, 0) + ";");
                        line("const std::size_t _numkit_bc = " + dimExpr(b, 1) + ";");
                        line("if (_numkit_br != _numkit_rh) throw std::out_of_range(\"numkit: block row "
                             "heights must agree\");");
                        open("for (std::size_t _numkit_j = 0; _numkit_j < _numkit_bc; ++_numkit_j)");
                        open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_br; ++_numkit_i)");
                        line(cppIdent(name) + "[(_numkit_ro + _numkit_i) + (_numkit_co + _numkit_j) * _numkit_P] = "
                             + b.dataExpr + "[_numkit_i + _numkit_j * _numkit_br];");
                        close();
                        close();
                        line("_numkit_co += _numkit_bc;");
                        --indent_;
                        line("}");
                    }
                    line("if (_numkit_co != _numkit_Q) throw std::out_of_range(\"numkit: block row widths "
                         "must agree\");");
                    line("_numkit_ro += _numkit_rh;");
                    --indent_;
                    line("}");
                }
                --indent_;
                line("}");
                types_.set(name, rv);
                return;
            }
        }
        // Rectangular grid of SCALARS `A = [1 2; 3 4]` -> a KnownDims R x C 2-D matrix local
        // (or a runtime-dim 2-D local when A is reassigned to different sizes). Stored
        // column-major: element (r, c) is at flat index c * R + r.
        if (isArrayVar(name) && arrays_.at(name).isLocal
            && (arrays_.at(name).is2D
                || (arrays_.at(name).isND && arrays_.at(name).ndDims.size() == 2))
            && rhs.type == NodeType::MATRIX_LITERAL && rhs.children.size() > 1
            && rhs.children[0] && rhs.children[0]->children.size() > 1) {
            const std::size_t R      = rhs.children.size();
            const std::size_t C      = rhs.children[0]->children.size();
            bool              gridOk = true;
            for (const auto &rowN : rhs.children) {
                if (!rowN || rowN->children.size() != C) { gridOk = false; break; }
                for (const auto &el : rowN->children) {
                    if (!el) { gridOk = false; break; }
                    const AbstractValue ev = inferExpr(*el, types_, reg_, classes_);
                    if (!ev.type.isConcrete() || !ev.type.shape.isScalar()) {
                        gridOk = false;
                        break;
                    }
                }
                if (!gridOk) break;
            }
            const AbstractValue rv = inferExpr(rhs, types_, reg_, classes_);
            if (gridOk && rv.type.isConcrete()) {
                const ArrayInfo &M = arrays_.at(name);
                if (M.ndRuntimeLocal) {
                    line(M.ndDims[0] + " = " + std::to_string(R) + ";");
                    line(M.ndDims[1] + " = " + std::to_string(C) + ";");
                }
                line(cppIdent(name) + ".assign(" + std::to_string(R * C) + ", " + zeroLiteral(M.dtype) + ");");
                for (std::size_t c = 0; c < C; ++c) {
                    for (std::size_t r = 0; r < R; ++r) {
                        line(cppIdent(name) + "[" + std::to_string(c * R + r) + "] = "
                             + emitExpr(*rhs.children[r]->children[c]) + ";");
                    }
                }
                types_.set(name, rv);
                return;
            }
        }

        // Whole-array struct-field READ: `y = s.v` / `y = s.a.v` where the field is a local
        // std::vector (non-ND, or known-dim ND).
        if (isArrayVar(name) && arrays_.at(name).isLocal
            && (rhs.type == NodeType::FIELD_ACCESS || rhs.type == NodeType::DYNAMIC_FIELD_ACCESS)) {
            const std::string rhsFld = resolveArrayVar(rhs);
            const AbstractValue rv   = inferExpr(rhs, types_, reg_, classes_);
            if (!rhsFld.empty() && rhsFld != name && arrays_.at(rhsFld).isLocal
                && !arrays_.at(name).ndRuntimeLocal && !arrays_.at(rhsFld).ndRuntimeLocal
                && rv.type.isConcrete() && !rv.type.shape.isScalar()) {
                line(cppIdent(name) + " = " + cppIdent(rhsFld) + ";");  // vector copy from the field-local
                types_.set(name, rv);
                return;
            }
        }

        // Whole-array COPY: `B = A` (or `y = s.v` for a struct param buffer / runtime-dim N-D,
        // or `s.v = x` / `s.A = M`).
        {
            const std::string rhsArr = resolveArrayVar(rhs);
            if (isArrayVar(name) && arrays_.at(name).isLocal && !rhsArr.empty() && rhsArr != name) {
                const ArrayInfo    &A  = arrays_.at(rhsArr);
                const ArrayInfo    &B  = arrays_.at(name);
                const AbstractValue rv = inferExpr(rhs, types_, reg_, classes_);
                if (rv.type.isConcrete() && !rv.type.shape.isScalar()) {
                    if (!structWriteFld.empty() && !A.is2D && !A.isND && !B.is2D && !B.isND) {
                        line(cppIdent(name) + ".assign(" + A.dataExpr + ", " + A.dataExpr + " + "
                             + A.lenVar + ");");
                        types_.set(name, rv);
                        return;
                    }
                    std::string numel;
                    if (A.isND) {
                        numel = A.ndDims[0];
                        for (std::size_t i = 1; i < A.ndDims.size(); ++i)
                            numel += " * " + A.ndDims[i];
                    } else if (A.is2D) {
                        numel = A.rowsVar + " * " + A.colsVar;
                    } else {
                        numel = A.lenVar;
                    }
                    line("{");
                    ++indent_;
                    if (B.isND && B.ndRuntimeLocal)
                        for (std::size_t k = 0; k < B.ndDims.size(); ++k)
                            line(B.ndDims[k] + " = " + dimExpr(A, k) + ";");
                    line(cppIdent(name) + ".assign(" + A.dataExpr + ", " + A.dataExpr + " + (" + numel
                         + "));");
                    --indent_;
                    line("}");
                    types_.set(name, rv);
                    return;
                }
            }
        }

        if (isArrayVar(name))
            unsupported("array-valued assignment to '" + name
                        + "' (only size-constructor init of the output in RawBuffer ABI)");

        // Scalar / object / struct assignment (the local was hoisted at function entry).
        const AbstractValue rv = inferExpr(rhs, types_, reg_, classes_);
        if (rv.type.isStruct() && rv.type.structLayout) {
            std::string srcPrefix;
            if (rhs.type == NodeType::IDENTIFIER)
                srcPrefix = "_numkit_fld_" + rhs.strValue;
            else if (rhs.type == NodeType::FIELD_ACCESS
                     || rhs.type == NodeType::DYNAMIC_FIELD_ACCESS)
                srcPrefix = structFieldLocal(rhs);
            const std::string dstPrefix =
                structWriteFld.empty() ? ("_numkit_fld_" + name) : name;
            if (!srcPrefix.empty()) {
                if (srcPrefix != dstPrefix)
                    emitStructCopy(dstPrefix, srcPrefix, *rv.type.structLayout);
                types_.set(name, rv);
                return;
            }
        }
        const bool lhsDynamic = dynamicLocals_.count(name) != 0;
        if (bridge_ && (rv.type.isDynamic() || lhsDynamic)) {
            line(cppIdent(name) + " = " + emitDynamicExpr(rhs) + ";");
            types_.set(name, lhsDynamic ? AbstractValue::dynamic() : rv);
            return;
        }
        line(cppIdent(name) + " = " + emitExpr(rhs) + ";");
        types_.set(name, rv);
        return;
}

} // namespace numkit::codegen::detail
