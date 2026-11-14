// codegen/src/emitter_assign.cpp — statement assignment lowering (part 1 & non-ident LHS).

#include "emitter_internal.hpp"

namespace numkit::codegen::detail {

void Emitter::emitAssign(const ASTNode &s)
{
    if (s.children.size() != 2) unsupported("assign arity");
    const ASTNode &lhs = *s.children[0];
    const ASTNode &rhs = *s.children[1];

    std::string structWriteFld;
    if ((lhs.type == NodeType::FIELD_ACCESS || lhs.type == NodeType::DYNAMIC_FIELD_ACCESS)
        && !lhs.children.empty() && lhs.children[0]
        && !(lhs.children[0]->type == NodeType::CALL && lhs.children[0]->children.size() == 2
             && lhs.children[0]->children[0]->type == NodeType::IDENTIFIER
             && !types_.has(lhs.children[0]->children[0]->strValue))) {
        const AbstractValue base = inferExpr(*lhs.children[0], types_, reg_, classes_);
        if (!base.type.isObject())
            structWriteFld = structFieldLocal(lhs);
    }

    if (lhs.type == NodeType::IDENTIFIER || !structWriteFld.empty()) {
        const std::string &name = structWriteFld.empty() ? lhs.strValue : structWriteFld;
        struct StructWriteSync {
            Emitter           *self;
            const ASTNode     *lhsNode;
            const std::string *fldName;
            ~StructWriteSync()
            {
                if (std::uncaught_exceptions() == 0 && self && lhsNode && fldName
                    && !fldName->empty() && self->types_.has(*fldName)) {
                    self->recordStructFieldWrite(*lhsNode, *fldName,
                                                 self->types_.get(*fldName).type);
                }
            }
        } syncGuard{structWriteFld.empty() ? nullptr : this, &lhs, &structWriteFld};

        if (bridge_ && dynamicLocals_.count(name)) {
            line(cppIdent(name) + " = " + emitDynamicExpr(rhs) + ";");
            types_.set(name, AbstractValue::dynamic());
            return;
        }

        // Struct-return call (`t = f(args)` or `t = obj.m(args)`) and single-LHS multi-output
        // call (`c = f(args)` or `c = obj.m(args)`).
        if (rhs.type == NodeType::CALL && !rhs.children.empty() && ctx_) {
            const ASTNode            *def = nullptr;
            std::string               calleeKey;
            std::string               mangleBase;
            bool                      isMethodCall = false;
            InferredType              selfType     = InferredType::dynamic();
            std::vector<InferredType> probeArgTypes;
            if (rhs.children[0]->type == NodeType::IDENTIFIER && ctx_->funcs
                && !types_.has(rhs.children[0]->strValue)
                && ctx_->funcs->has(rhs.children[0]->strValue)) {
                const std::string &callee = rhs.children[0]->strValue;
                def                       = ctx_->funcs->find(callee);
                calleeKey                 = callee;
                mangleBase                = callee;
            } else if (rhs.children[0]->type == NodeType::FIELD_ACCESS
                       && !rhs.children[0]->children.empty() && classes_) {
                const ASTNode      &callee = *rhs.children[0];
                const AbstractValue base =
                    inferExpr(*callee.children[0], types_, reg_, classes_);
                if (base.type.isObject()) {
                    const ClassInfo *ci = classes_->byId(base.type.classId);
                    def                 = ci ? ci->method(callee.strValue) : nullptr;
                    if (def) {
                        calleeKey    = ci->name + "::" + callee.strValue;
                        mangleBase   = ci->name + "__" + callee.strValue;
                        isMethodCall = true;
                        selfType     = base.type;
                    }
                }
            }
            if (def) {
                if (isMethodCall) probeArgTypes.push_back(selfType);
                for (std::size_t i = 1; i < rhs.children.size(); ++i)
                    probeArgTypes.push_back(inferExpr(*rhs.children[i], types_, reg_, classes_).type);
                if (probeArgTypes.size() != def->paramNames.size())
                    unsupported("arity mismatch calling '" + calleeKey + "'");
                const std::vector<InferredType> outs =
                    reg_.applyMulti(calleeKey, toArgInfos(probeArgTypes));
                const std::string dstPrefix =
                    structWriteFld.empty() ? ("_numkit_fld_" + name) : name;
                if (def->returnNames.size() == 1) {
                    const AbstractValue retAv = inferExpr(rhs, types_, reg_, classes_);
                    if (retAv.type.isStruct() && retAv.type.structLayout) {
                        std::set<std::string>    outRefArrays;
                        std::vector<std::string> aliasDecls;
                        collectStructOutRefArrays(dstPrefix, *retAv.type.structLayout, outRefArrays);
                        prepareCallInputAliases(rhs, 1, outRefArrays, aliasDecls);
                        std::vector<InferredType> argTypes;
                        std::string               argList;
                        if (isMethodCall) {
                            argTypes.push_back(selfType);
                            argList = emitExpr(*rhs.children[0]->children[0]);
                        }
                        for (std::size_t i = 1; i < rhs.children.size(); ++i)
                            appendCallArg(*rhs.children[i], argTypes, argList);
                        callAliasInputs_.clear();
                        appendStructOutArgs(dstPrefix, *retAv.type.structLayout, argList);
                        const std::string mangled = mangle(mangleBase, argTypes);
                        if (ctx_->seen.insert(mangled).second)
                            ctx_->pending.push_back({def, argTypes, mangled});
                        if (aliasDecls.empty()) {
                            line(mangled + "(" + argList + ");");
                        } else {
                            open("");
                            for (const std::string &d : aliasDecls) line(d);
                            line(mangled + "(" + argList + ");");
                            close();
                        }
                        types_.set(name, {retAv.type, ConstVal::unknown()});
                        return;
                    }
                } else if (outs.size() > 1) {
                    const InferredType       out0        = outs[0];
                    const bool               out0ByValue = isByValueReturnArrayType(out0);
                    std::set<std::string>    outRefArrays;
                    std::vector<std::string> aliasDecls;
                    if (!out0ByValue && out0.isStruct() && out0.structLayout)
                        collectStructOutRefArrays(dstPrefix, *out0.structLayout, outRefArrays);
                    prepareCallInputAliases(rhs, 1, outRefArrays, aliasDecls);
                    std::vector<InferredType> argTypes;
                    std::string               argList;
                    if (isMethodCall) {
                        argTypes.push_back(selfType);
                        argList = emitExpr(*rhs.children[0]->children[0]);
                    }
                    for (std::size_t i = 1; i < rhs.children.size(); ++i)
                        appendCallArg(*rhs.children[i], argTypes, argList);
                    callAliasInputs_.clear();
                    if (out0ByValue) {
                        // Returned by value -> assigned to `name` below.
                    } else if (isUnboxableScalarType(out0) || out0.isObject()) {
                        if (!argList.empty()) argList += ", ";
                        argList += cppIdent(name);
                    } else if (out0.isStruct() && out0.structLayout) {
                        appendStructOutArgs(dstPrefix, *out0.structLayout, argList);
                    } else {
                        unsupported("c = f(x): first output of multi-output '" + calleeKey
                                    + "' has an unsupported type");
                    }
                    std::vector<std::string> ignoreDecls;
                    for (std::size_t i = 1; i < outs.size(); ++i) {
                        const std::string tmp = "_numkit_ignore_" + std::to_string(i);
                        appendIgnoreOutArg(tmp, outs[i], ignoreDecls, argList);
                    }
                    const std::string mangled = mangle(mangleBase, argTypes);
                    if (ctx_->seen.insert(mangled).second)
                        ctx_->pending.push_back({def, argTypes, mangled});
                    open("");  // fresh scope for the throwaway locals
                    for (const std::string &d : aliasDecls) line(d);
                    for (const std::string &d : ignoreDecls) line(d);
                    if (out0ByValue)
                        line(cppIdent(name) + " = " + mangled + "(" + argList + ");");
                    else
                        line(mangled + "(" + argList + ");");
                    close();
                    types_.set(name, {out0, ConstVal::unknown()});
                    return;
                }
            }
        }

        // x = [] : empty a 1-D LOCAL vector.
        if (isArrayVar(name) && arrays_.at(name).isLocal && !arrays_.at(name).is2D
            && !arrays_.at(name).isND && rhs.type == NodeType::MATRIX_LITERAL) {
            bool emptyMat = true;
            for (const auto &row : rhs.children)
                if (row && !row->children.empty()) { emptyMat = false; break; }
            if (emptyMat) {
                line(cppIdent(name) + ".clear();");
                types_.set(name, {InferredType::concrete(arrays_.at(name).dtype, Shape::rowVector()),
                                  ConstVal::unknown()});
                return;
            }
        }

        // Array initialised by a size constructor: `a = zeros(1,n)` / `ones`.
        // The OUTPUT out-param (caller-sized) becomes a fill loop over its
        // length; an owned-vector LOCAL is `a.assign(numel, fill)` (numel =
        // product of the size args).
        if (isArrayVar(name)
            && (arrays_.at(name).isOutput || arrays_.at(name).isLocal)  // local/output, any rank
            && rhs.type == NodeType::CALL && !rhs.children.empty()
            && rhs.children[0]->type == NodeType::IDENTIFIER
            && (rhs.children[0]->strValue == "zeros"
                || rhs.children[0]->strValue == "ones")) {
            const ArrayInfo  &ai   = arrays_.at(name);
            const std::string fill = rhs.children[0]->strValue == "zeros"
                                         ? zeroLiteral(ai.dtype)
                                         : "1";
            if (ai.ndRuntimeLocal) {
                // Capture each runtime dim into its companion var, then size the
                // owned vector to their product (column-major flat storage).
                std::string prod;
                for (std::size_t d = 0; d < ai.ndDims.size(); ++d) {
                    line(ai.ndDims[d] + " = numkit_rt::dim(" + emitExpr(*rhs.children[d + 1]) + ");");
                    prod += (d ? " * " : "") + ai.ndDims[d];
                }
                line(cppIdent(name) + ".assign(" + prod + ", " + fill + ");");
            } else if (ai.isLocal) {
                std::string numel;  // product of the size args (zeros(1,n) -> 1*n)
                for (std::size_t i = 1; i < rhs.children.size(); ++i)
                    numel += (i > 1 ? " * " : "")
                             + ("numkit_rt::dim(" + emitExpr(*rhs.children[i]) + ")");
                if (numel.empty()) numel = "0";
                line(cppIdent(name) + ".assign(" + numel + ", " + fill + ");");
            } else {
                open("for (std::size_t _numkit_i = 0; _numkit_i < " + ai.lenVar + "; ++_numkit_i)");
                line(ai.dataExpr + "[_numkit_i] = " + fill + ";");
                close();
            }
            // Record the array type so a later `name(k)` infers element-access
            // (the env, not arrays_, drives inferExpr). A 2-D / N-D array records
            // its TRUE shape (inferExpr of the zeros/ones RHS) instead of a 1-D
            // row-vector stand-in — accurate, no type-lie; arrays_ still drives
            // indexing/queries either way.
            if (ai.isND || ai.is2D)
                types_.set(name, inferExpr(rhs, types_, reg_, classes_));
            else
                types_.set(name, {InferredType::concrete(ai.dtype, Shape::rowVector()),
                                  ConstVal::unknown()});
            return;
        }
        // linspace(a, b [, n]) -> n linearly-spaced points from a to b, a 1 x n
        // owned-vector LOCAL (the size-constructor transfer gives it a 1 x n /
        // 1 x 100 / unknown shape). Like zeros/ones it sizes the local, then fills a
        // ramp -- but MATLAB forces the LAST point to exactly b (no rounding drift),
        // so we seed every slot to b and overwrite [0, n-1) with a + i*step. n==1 ->
        // {b}; n<=0 -> empty. v1: an owned LOCAL, 2- or 3-arg; an OUTPUT-param target
        // (caller-sized) falls through to the bridge.
        if (isArrayVar(name) && arrays_.at(name).isLocal
            && rhs.type == NodeType::CALL && !rhs.children.empty()
            && rhs.children[0]->type == NodeType::IDENTIFIER
            && rhs.children[0]->strValue == "linspace"
            && (rhs.children.size() == 3 || rhs.children.size() == 4)) {
            const std::string a = emitExpr(*rhs.children[1]);
            const std::string b = emitExpr(*rhs.children[2]);
            const std::string n = rhs.children.size() == 4
                                      ? ("numkit_rt::dim(" + emitExpr(*rhs.children[3]) + ")")
                                      : std::string("100");  // 2-arg default (MATLAB)
            line("{");
            ++indent_;
            line("const double _numkit_a = " + a + ";");
            line("const double _numkit_b = " + b + ";");
            line("const std::size_t _numkit_n = " + n + ";");
            line(cppIdent(name) + ".assign(_numkit_n, _numkit_b);");  // size n; the last point is already b
            open("if (_numkit_n >= 2)");
            line("const double _numkit_step = (_numkit_b - _numkit_a) / static_cast<double>(_numkit_n - 1);");
            open("for (std::size_t _numkit_i = 0; _numkit_i + 1 < _numkit_n; ++_numkit_i)");
            line(cppIdent(name) + "[_numkit_i] = _numkit_a + static_cast<double>(_numkit_i) * _numkit_step;");
            close();
            close();
            --indent_;
            line("}");
            const ArrayInfo &ai = arrays_.at(name);
            if (!ai.isND && !ai.is2D) arrays_.at(name).orient = VecOrient::Row;
            types_.set(name, inferExpr(rhs, types_, reg_, classes_));
            return;
        }
        // logspace(a, b [, n]) -> n decade-spaced points 10^a .. 10^b (= 10^linspace).
        // Mirrors the linspace fill but each slot is 10^(a + i*step); the last point
        // is seeded to exactly 10^b. n==1 -> {10^b}; n<=0 -> empty; the 2-arg form
        // defaults n=50 (MATLAB logspace, NOT 100). v1: an owned LOCAL, 2- or 3-arg.
        // (The logspace(a,pi,n) "upper limit = pi" quirk is a deferred gap.)
        if (isArrayVar(name) && arrays_.at(name).isLocal
            && rhs.type == NodeType::CALL && !rhs.children.empty()
            && rhs.children[0]->type == NodeType::IDENTIFIER
            && rhs.children[0]->strValue == "logspace"
            && (rhs.children.size() == 3 || rhs.children.size() == 4)) {
            const std::string a = emitExpr(*rhs.children[1]);
            const std::string b = emitExpr(*rhs.children[2]);
            const std::string n = rhs.children.size() == 4
                                      ? ("numkit_rt::dim(" + emitExpr(*rhs.children[3]) + ")")
                                      : std::string("50");  // 2-arg default (MATLAB logspace)
            line("{");
            ++indent_;
            line("const double _numkit_a = " + a + ";");
            line("const double _numkit_b = " + b + ";");
            line("const std::size_t _numkit_n = " + n + ";");
            line(cppIdent(name) + ".assign(_numkit_n, std::pow(10.0, _numkit_b));");  // last point already 10^b
            open("if (_numkit_n >= 2)");
            line("const double _numkit_step = (_numkit_b - _numkit_a) / static_cast<double>(_numkit_n - 1);");
            open("for (std::size_t _numkit_i = 0; _numkit_i + 1 < _numkit_n; ++_numkit_i)");
            line(cppIdent(name) + "[_numkit_i] = std::pow(10.0, _numkit_a + static_cast<double>(_numkit_i) * _numkit_step);");
            close();
            close();
            --indent_;
            line("}");
            const ArrayInfo &ai = arrays_.at(name);
            if (!ai.isND && !ai.is2D) arrays_.at(name).orient = VecOrient::Row;
            types_.set(name, inferExpr(rhs, types_, reg_, classes_));
            return;
        }
        // Colon range MATERIALISED to an array: v = a:b (step 1) / v = a:s:b. A
        // double row built into the owned 1-D local. Parser child order (verified):
        // 2 children = [start, stop]; 3 children = [start, step, stop]. MATLAB count
        // n = floor((stop-start)/step + tol) + 1, clamped to >=0 (empty when step==0
        // or the direction disagrees); a small tol matches MATLAB's float-range
        // length (e.g. 0:0.1:1 -> 11). The `for i=a:b` HEADER is special-cased
        // elsewhere (no array); this is the value/materialise form. v1: owned 1-D
        // LOCAL. (Exact MATLAB tolerance for pathological float ranges is deferred.)
        if (isArrayVar(name) && arrays_.at(name).isLocal && !arrays_.at(name).is2D
            && !arrays_.at(name).isND && rhs.type == NodeType::COLON_EXPR
            && (rhs.children.size() == 2 || rhs.children.size() == 3)) {
            const bool        three = rhs.children.size() == 3;
            const std::string start = emitExpr(*rhs.children[0]);
            const std::string step  = three ? emitExpr(*rhs.children[1]) : std::string("1.0");
            const std::string stop  = emitExpr(*rhs.children[three ? 2 : 1]);
            line("{");
            ++indent_;
            line("const double _numkit_start = " + start + ";");
            line("const double _numkit_step = " + step + ";");
            line("const double _numkit_stop = " + stop + ";");
            line("const double _numkit_nr = (_numkit_stop - _numkit_start) / _numkit_step;");
            line("if (_numkit_step != 0.0 && !std::isnan(_numkit_nr) && !std::isfinite(_numkit_nr) && _numkit_nr > 0.0)");
            line("    throw std::runtime_error(\"Maximum variable size allowed by the program is exceeded.\");");
            line("const std::ptrdiff_t _numkit_cnt = (_numkit_step == 0.0 || !std::isfinite(_numkit_nr) || _numkit_nr < 0.0)");
            line("    ? 0 : static_cast<std::ptrdiff_t>(_numkit_nr + 1e-10) + 1;");
            line(cppIdent(name) + ".assign(static_cast<std::size_t>(_numkit_cnt), 0.0);");
            open("for (std::ptrdiff_t _numkit_i = 0; _numkit_i < _numkit_cnt; ++_numkit_i)");
            line(cppIdent(name) + "[static_cast<std::size_t>(_numkit_i)] = _numkit_start + static_cast<double>(_numkit_i) * _numkit_step;");
            close();
            --indent_;
            line("}");
            arrays_.at(name).orient = VecOrient::Row;
            types_.set(name, inferExpr(rhs, types_, reg_, classes_));
            return;
        }
        // y = x(:) -> all elements of x as a column vector (column-major order), a
        // fresh 1-D LOCAL. The bare colon parses as an EMPTY COLON_EXPR. Column-major
        // storage means the flat buffer IS the column order, so x(:) is a straight
        // copy of numel(x) elements (1-D length / 2-D rows*cols / N-D product of dims).
        // v1: x a 1-D/2-D/N-D DOUBLE array var; result a 1-D LOCAL.
        if (isArrayVar(name) && arrays_.at(name).isLocal && !arrays_.at(name).is2D
            && !arrays_.at(name).isND && rhs.type == NodeType::CALL && rhs.children.size() == 2
            && rhs.children[0]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[0]->strValue)
            && rhs.children[1]->type == NodeType::COLON_EXPR && rhs.children[1]->children.empty()
            && arrays_.at(rhs.children[0]->strValue).dtype == ValueType::DOUBLE) {
            const ArrayInfo    &x   = arrays_.at(rhs.children[0]->strValue);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete() && !res.type.shape.isScalar()) {
                std::string numel;
                if (x.isND) {
                    numel = x.ndDims[0];
                    for (std::size_t i = 1; i < x.ndDims.size(); ++i) numel += " * " + x.ndDims[i];
                } else if (x.is2D) {
                    numel = x.rowsVar + " * " + x.colsVar;
                } else {
                    numel = x.lenVar;
                }
                line(cppIdent(name) + ".assign(" + x.dataExpr + ", " + x.dataExpr + " + (" + numel + "));");
                arrays_.at(name).orient = VecOrient::Col;
                types_.set(name, res);
                return;
            }
        }
        // A(:, j) -> column j of a 2-D matrix as a column vector, a fresh 1-D LOCAL.
        // Column-major storage: column j (1-based) is the CONTIGUOUS slice
        // A[(j-1)*rows .. +rows), so it's a straight copy. `end` in the column index
        // = cols (pushed). Bounds-checked j in [1, cols]. v1: A a 2-D DOUBLE matrix
        // var, j a scalar; result a 1-D LOCAL of `rows`. (A(i,:) row slice = strided,
        // deferred; A(:,:) -> both colons, deferred.)
        if (isArrayVar(name) && arrays_.at(name).isLocal && !arrays_.at(name).is2D
            && !arrays_.at(name).isND && rhs.type == NodeType::CALL && rhs.children.size() == 3
            && rhs.children[0]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[0]->strValue)
            && (arrays_.at(rhs.children[0]->strValue).is2D
                || (arrays_.at(rhs.children[0]->strValue).isND
                    && arrays_.at(rhs.children[0]->strValue).ndDims.size() == 2))
            && arrays_.at(rhs.children[0]->strValue).dtype == ValueType::DOUBLE
            && rhs.children[1]->type == NodeType::COLON_EXPR && rhs.children[1]->children.empty()
            && rhs.children[2]->type != NodeType::COLON_EXPR) {
            const ArrayInfo    &A     = arrays_.at(rhs.children[0]->strValue);
            const std::string   Arows = dimExpr(A, 0), Acols = dimExpr(A, 1);  // rank-agnostic
            const AbstractValue res   = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete() && !res.type.shape.isScalar()
                && inferExpr(*rhs.children[2], types_, reg_, classes_).type.shape.isScalar()) {
                endStack_.push_back(Acols);  // `end` in the column index = cols
                const std::string j = emitExpr(*rhs.children[2]);
                endStack_.pop_back();
                line("{");
                ++indent_;
                line("const std::size_t _numkit_j0 = numkit_rt::check_idx1(" + j + ", " + Acols
                     + ", \"numkit: column index out of bounds\");");
                line("const std::size_t _numkit_off = _numkit_j0 * " + Arows + ";");
                line(cppIdent(name) + ".assign(" + A.dataExpr + " + _numkit_off, " + A.dataExpr + " + _numkit_off + "
                     + Arows + ");");
                --indent_;
                line("}");
                arrays_.at(name).orient = VecOrient::Col;
                types_.set(name, res);
                return;
            }
        }
        // A(i, :) -> row i of a 2-D matrix as a row vector, a fresh 1-D LOCAL. Row i
        // (1-based) is STRIDED in column-major storage: out[j] = A[(i-1) + j*rows] for
        // j in [0, cols). `end` in the ROW index = rows (pushed). Bounds-checked i in
        // [1, rows]. v1: A a 2-D DOUBLE matrix var, i a scalar; result a 1-D LOCAL of
        // `cols`. (The mirror of A(:,j); A(:,:) -- both colons -- is deferred.)
        if (isArrayVar(name) && arrays_.at(name).isLocal && !arrays_.at(name).is2D
            && !arrays_.at(name).isND && rhs.type == NodeType::CALL && rhs.children.size() == 3
            && rhs.children[0]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[0]->strValue)
            && (arrays_.at(rhs.children[0]->strValue).is2D
                || (arrays_.at(rhs.children[0]->strValue).isND
                    && arrays_.at(rhs.children[0]->strValue).ndDims.size() == 2))
            && arrays_.at(rhs.children[0]->strValue).dtype == ValueType::DOUBLE
            && rhs.children[1]->type != NodeType::COLON_EXPR
            && rhs.children[2]->type == NodeType::COLON_EXPR && rhs.children[2]->children.empty()) {
            const ArrayInfo    &A     = arrays_.at(rhs.children[0]->strValue);
            const std::string   Arows = dimExpr(A, 0), Acols = dimExpr(A, 1);  // rank-agnostic
            const AbstractValue res   = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete() && !res.type.shape.isScalar()
                && inferExpr(*rhs.children[1], types_, reg_, classes_).type.shape.isScalar()) {
                endStack_.push_back(Arows);  // `end` in the row index = rows
                const std::string i = emitExpr(*rhs.children[1]);
                endStack_.pop_back();
                line("{");
                ++indent_;
                line("const std::size_t _numkit_i0 = numkit_rt::check_idx1(" + i + ", " + Arows
                     + ", \"numkit: row index out of bounds\");");
                line(cppIdent(name) + ".assign(" + Acols + ", 0.0);");
                open("for (std::size_t _numkit_j = 0; _numkit_j < " + Acols + "; ++_numkit_j)");
                line(cppIdent(name) + "[_numkit_j] = " + A.dataExpr + "[_numkit_i0 + _numkit_j * "
                     + Arows + "];");
                close();
                --indent_;
                line("}");
                arrays_.at(name).orient = VecOrient::Row;
                types_.set(name, res);
                return;
            }
        }
        // Native diag(A) -> the diagonal of a 2-D matrix as a fresh 1-D LOCAL. Column-
        // major: the (i,i) element is A[i + i*rows]; the length is min(rows, cols).
        // v1: A a 2-D DOUBLE matrix var; result a 1-D LOCAL (push_back).
        if (isArrayVar(name) && arrays_.at(name).isLocal && !arrays_.at(name).is2D
            && !arrays_.at(name).isND && rhs.type == NodeType::CALL && rhs.children.size() == 2
            && rhs.children[0]->type == NodeType::IDENTIFIER && rhs.children[0]->strValue == "diag"
            && rhs.children[1]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[1]->strValue)
            && arrays_.at(rhs.children[1]->strValue).is2D
            && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE) {
            const ArrayInfo    &A   = arrays_.at(rhs.children[1]->strValue);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete() && !res.type.shape.isScalar()) {
                line("{");
                ++indent_;
                line("const std::size_t _numkit_d = " + A.rowsVar + " < " + A.colsVar + " ? " + A.rowsVar
                     + " : " + A.colsVar + ";");
                line(cppIdent(name) + ".clear();");
                open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_d; ++_numkit_i)");
                line(cppIdent(name) + ".push_back(" + A.dataExpr + "[_numkit_i + _numkit_i * " + A.rowsVar + "]);");
                close();
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // diag(v[,k]) with v a VECTOR -> an N x N diagonal MATRIX (N = numel(v) + k), a
        // rank-2 ndRuntimeLocal. k=0: M[i + i*N] = v[i] (main diagonal). A non-negative
        // LITERAL offset k places v on the k-th super-diagonal: M[i + (i+k)*N] = v[i]. v1:
        // v a 1-D DOUBLE vector var; k (if present) a non-negative literal integer. A
        // negative k (`diag(v,-1)`, a sub-diagonal) parses as a unary-minus -> not folded
        // -> diagTransfer keeps it Dynamic -> bridged; a runtime k is likewise bridged.
        if (isArrayVar(name) && arrays_.at(name).isLocal && arrays_.at(name).ndRuntimeLocal
            && arrays_.at(name).ndDims.size() == 2 && rhs.type == NodeType::CALL
            && (rhs.children.size() == 2
                || (rhs.children.size() == 3 && rhs.children[2]->type == NodeType::NUMBER_LITERAL
                    && rhs.children[2]->numValue == std::floor(rhs.children[2]->numValue)
                    && rhs.children[2]->numValue >= 0.0))
            && rhs.children[0]->type == NodeType::IDENTIFIER && rhs.children[0]->strValue == "diag"
            && rhs.children[1]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[1]->strValue)
            && !arrays_.at(rhs.children[1]->strValue).is2D
            && !arrays_.at(rhs.children[1]->strValue).isND
            && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE) {
            const ArrayInfo    &v   = arrays_.at(rhs.children[1]->strValue);
            const ArrayInfo    &M   = arrays_.at(name);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            const std::string   k =  // non-negative literal offset (0 = main diagonal)
                rhs.children.size() == 3
                    ? std::to_string(static_cast<long>(rhs.children[2]->numValue))
                    : std::string("0");
            const std::string N = "(" + v.lenVar + " + " + k + ")";  // N = n + k
            line("{");
            ++indent_;
            line("const std::size_t _numkit_N = " + N + ";");
            line(M.ndDims[0] + " = _numkit_N;");  // rows = N
            line(M.ndDims[1] + " = _numkit_N;");  // cols = N
            line(cppIdent(name) + ".assign(_numkit_N * _numkit_N, 0.0);");
            open("for (std::size_t _numkit_i = 0; _numkit_i < " + v.lenVar + "; ++_numkit_i)");
            // v[i] on the k-th super-diagonal: (row i, col i+k), column-major.
            line(cppIdent(name) + "[_numkit_i + (_numkit_i + " + k + ") * _numkit_N] = " + v.dataExpr + "[_numkit_i];");
            close();
            --indent_;
            line("}");
            types_.set(name, res);
            return;
        }
        // Native cross(a, b) -> the 3-D cross product, a fresh 3-element 1-D LOCAL:
        // c = [a2*b3-a3*b2, a3*b1-a1*b3, a1*b2-a2*b1] (1-based; 0-based in the emit).
        // v1: a, b 1-D DOUBLE array vars (length 3).
        if (isArrayVar(name) && arrays_.at(name).isLocal && !arrays_.at(name).is2D
            && !arrays_.at(name).isND && rhs.type == NodeType::CALL && rhs.children.size() == 3
            && rhs.children[0]->type == NodeType::IDENTIFIER && rhs.children[0]->strValue == "cross"
            && rhs.children[1]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[1]->strValue)
            && rhs.children[2]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[2]->strValue)
            && !arrays_.at(rhs.children[1]->strValue).is2D
            && !arrays_.at(rhs.children[1]->strValue).isND
            && !arrays_.at(rhs.children[2]->strValue).is2D
            && !arrays_.at(rhs.children[2]->strValue).isND
            && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE
            && arrays_.at(rhs.children[2]->strValue).dtype == ValueType::DOUBLE) {
            const ArrayInfo    &a   = arrays_.at(rhs.children[1]->strValue);
            const ArrayInfo    &b   = arrays_.at(rhs.children[2]->strValue);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete() && !res.type.shape.isScalar()) {
                const std::string ad = a.dataExpr, bd = b.dataExpr;
                line("{");
                ++indent_;
                line(cppIdent(name) + ".clear();");
                line(cppIdent(name) + ".push_back(" + ad + "[1] * " + bd + "[2] - " + ad + "[2] * " + bd
                     + "[1]);");
                line(cppIdent(name) + ".push_back(" + ad + "[2] * " + bd + "[0] - " + ad + "[0] * " + bd
                     + "[2]);");
                line(cppIdent(name) + ".push_back(" + ad + "[0] * " + bd + "[1] - " + ad + "[1] * " + bd
                     + "[0]);");
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // reshape(x, m, n) -> reinterpret x's elements as an m x n matrix. Column-major
        // storage means a reshape is just the SAME flat buffer reinterpreted, so copy
        // x's elements into the 2-D KnownDims LOCAL (numel must match: m*n==numel(x),
        // runtime-checked). v(i,j) 2-D indexing then works on the result. v1: x a
        // 1-D/2-D DOUBLE array var, m,n compile-time literals -> a 2-D LOCAL.
        if (isArrayVar(name) && arrays_.at(name).isLocal && arrays_.at(name).is2D
            && rhs.type == NodeType::CALL && rhs.children.size() == 4
            && rhs.children[0]->type == NodeType::IDENTIFIER
            && rhs.children[0]->strValue == "reshape"
            && rhs.children[1]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[1]->strValue)
            && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE) {
            const ArrayInfo    &v   = arrays_.at(name);  // the 2-D result
            const ArrayInfo    &x   = arrays_.at(rhs.children[1]->strValue);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete() && res.type.shape.kind == ShapeKind::KnownDims) {
                std::string xnumel;
                if (x.isND) {
                    xnumel = x.ndDims[0];
                    for (std::size_t i = 1; i < x.ndDims.size(); ++i) xnumel += " * " + x.ndDims[i];
                } else if (x.is2D) {
                    xnumel = x.rowsVar + " * " + x.colsVar;
                } else {
                    xnumel = x.lenVar;
                }
                line("{");
                ++indent_;
                line("if ((" + v.rowsVar + " * " + v.colsVar + ") != (" + xnumel + "))");
                line("    throw std::out_of_range(\"numkit: reshape element count mismatch\");");
                line(cppIdent(name) + ".assign(" + x.dataExpr + ", " + x.dataExpr + " + (" + v.rowsVar + " * "
                     + v.colsVar + "));");
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // reshape(x, m, n) with a RUNTIME dim -> reinterpret x's flat data as a runtime-dim
        // 2-D matrix (same column-major buffer). Copy x's elements, set the dim companions
        // from the args via numkit_rt::dim, numel-guard (m*n == numel(x)). The runtime mirror of
        // the KnownDims reshape above. v1: x a DOUBLE array var, distinct from the dest (an
        // in-place A=reshape(A,..) would self-alias the .assign -> refused via the guard).
        if (isArrayVar(name) && arrays_.at(name).isLocal && arrays_.at(name).ndRuntimeLocal
            && arrays_.at(name).ndDims.size() == 2 && rhs.type == NodeType::CALL
            && rhs.children.size() == 4 && rhs.children[0]->type == NodeType::IDENTIFIER
            && rhs.children[0]->strValue == "reshape"
            && rhs.children[1]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[1]->strValue)
            && rhs.children[1]->strValue != name
            && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE) {
            const ArrayInfo    &M   = arrays_.at(name);
            const ArrayInfo    &x   = arrays_.at(rhs.children[1]->strValue);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete()) {
                std::string xnumel;
                if (x.isND) {
                    xnumel = x.ndDims[0];
                    for (std::size_t i = 1; i < x.ndDims.size(); ++i) xnumel += " * " + x.ndDims[i];
                } else if (x.is2D) {
                    xnumel = x.rowsVar + " * " + x.colsVar;
                } else {
                    xnumel = x.lenVar;
                }
                line("{");
                ++indent_;
                line("const std::size_t _numkit_m = numkit_rt::dim(" + emitExpr(*rhs.children[2]) + ");");
                line("const std::size_t _numkit_n = numkit_rt::dim(" + emitExpr(*rhs.children[3]) + ");");
                line("const std::size_t _numkit_xn = (" + xnumel + ");");
                line("if (_numkit_m * _numkit_n != _numkit_xn)");
                line("    throw std::out_of_range(\"numkit: reshape element count mismatch\");");
                line(M.ndDims[0] + " = _numkit_m;");
                line(M.ndDims[1] + " = _numkit_n;");
                line(cppIdent(name) + ".assign(" + x.dataExpr + ", " + x.dataExpr + " + _numkit_xn);");
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // reshape(x, d1, ..., dN) -> reinterpret x's flat data as a RANK-N array (N = number of
        // dim args >= 3; same column-major buffer; phases N5/N14). Copy x's elements, set the N
        // dim companions from numkit_rt::dim(arg), numel-guard (product == numel(x)). Rank-agnostic
        // (covers rank-3, rank-4, ...). v1: x a DOUBLE array var distinct from the dest (an
        // in-place A=reshape(A,..) would self-alias the .assign -> refused).
        if (isArrayVar(name) && arrays_.at(name).isLocal && arrays_.at(name).ndRuntimeLocal
            && arrays_.at(name).ndDims.size() >= 3 && rhs.type == NodeType::CALL
            && rhs.children.size() == arrays_.at(name).ndDims.size() + 2
            && rhs.children[0]->type == NodeType::IDENTIFIER
            && rhs.children[0]->strValue == "reshape"
            && rhs.children[1]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[1]->strValue)
            && rhs.children[1]->strValue != name
            && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE) {
            const ArrayInfo    &M   = arrays_.at(name);
            const ArrayInfo    &x   = arrays_.at(rhs.children[1]->strValue);
            const std::size_t   N   = M.ndDims.size();  // result rank = number of dim args
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete()) {
                std::string xnumel;
                if (x.isND) {
                    xnumel = x.ndDims[0];
                    for (std::size_t i = 1; i < x.ndDims.size(); ++i) xnumel += " * " + x.ndDims[i];
                } else if (x.is2D) {
                    xnumel = x.rowsVar + " * " + x.colsVar;
                } else {
                    xnumel = x.lenVar;
                }
                line("{");
                ++indent_;
                for (std::size_t k = 0; k < N; ++k)
                    line("const std::size_t _numkit_d" + std::to_string(k) + " = numkit_rt::dim("
                         + emitExpr(*rhs.children[2 + k]) + ");");
                std::string prod = "_numkit_d0";
                for (std::size_t k = 1; k < N; ++k) prod += " * _numkit_d" + std::to_string(k);
                line("const std::size_t _numkit_xn = (" + xnumel + ");");
                line("if ((" + prod + ") != _numkit_xn)");
                line("    throw std::out_of_range(\"numkit: reshape element count mismatch\");");
                for (std::size_t k = 0; k < N; ++k)
                    line(M.ndDims[k] + " = _numkit_d" + std::to_string(k) + ";");
                line(cppIdent(name) + ".assign(" + x.dataExpr + ", " + x.dataExpr + " + _numkit_xn);");
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // repmat(s, m, n) / repmat(s, n) with s a SCALAR -> an m x n (or n x n) matrix
        // all = s, a 2-D KnownDims LOCAL (the tiling of a 1x1). Like zeros/ones but the
        // fill is the scalar value. v1: s a DOUBLE scalar, m,n literals. (repmat of a
        // vector/matrix = true tiling, deferred.)
        if (isArrayVar(name) && arrays_.at(name).isLocal && arrays_.at(name).is2D
            && rhs.type == NodeType::CALL
            && (rhs.children.size() == 3 || rhs.children.size() == 4)
            && rhs.children[0]->type == NodeType::IDENTIFIER
            && rhs.children[0]->strValue == "repmat"
            && inferExpr(*rhs.children[1], types_, reg_, classes_).type.shape.isScalar()) {
            const ArrayInfo    &v   = arrays_.at(name);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete() && res.type.shape.kind == ShapeKind::KnownDims) {
                const std::string s = emitExpr(*rhs.children[1]);
                line(cppIdent(name) + ".assign(static_cast<std::size_t>(" + v.rowsVar + " * " + v.colsVar
                     + "), " + s + ");");
                types_.set(name, res);
                return;
            }
        }
        // repmat(x, 1, q) with x a ROW vector -> tile q copies into a 1 x (q*n) row,
        // a fresh 1-D LOCAL: out[k*n + i] = x[i] for k in [0,q), i in [0,n). The down-
        // rep (children[2]) must be the literal 1. v1: x a 1-D DOUBLE row var.
        if (isArrayVar(name) && arrays_.at(name).isLocal && !arrays_.at(name).is2D
            && !arrays_.at(name).isND && rhs.type == NodeType::CALL && rhs.children.size() == 4
            && rhs.children[0]->type == NodeType::IDENTIFIER
            && rhs.children[0]->strValue == "repmat"
            && rhs.children[1]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[1]->strValue)
            && !arrays_.at(rhs.children[1]->strValue).is2D
            && !arrays_.at(rhs.children[1]->strValue).isND
            && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE
            && rhs.children[2]->type == NodeType::NUMBER_LITERAL
            && rhs.children[2]->numValue == 1.0) {
            const ArrayInfo    &x   = arrays_.at(rhs.children[1]->strValue);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete() && !res.type.shape.isScalar()) {
                const std::string q = emitExpr(*rhs.children[3]);
                line("{");
                ++indent_;
                line("const std::size_t _numkit_q = numkit_rt::dim(" + q + ");");
                line(cppIdent(name) + ".clear();");
                open("for (std::size_t _numkit_k = 0; _numkit_k < _numkit_q; ++_numkit_k)");
                open("for (std::size_t _numkit_i = 0; _numkit_i < " + x.lenVar + "; ++_numkit_i)");
                line(cppIdent(name) + ".push_back(" + x.dataExpr + "[_numkit_i]);");
                close();
                close();
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // repmat(x, p, 1) with x a COL vector -> p copies stacked into a (p*n) x 1
        // column, a fresh 1-D LOCAL: out[k*n + i] = x[i]. The mirror of the row tile;
        // the across-rep (children[3]) must be the literal 1, reps = children[2]. A
        // row operand here yields a Dynamic result (2-D) -> the concrete-result guard
        // gates it out. v1: x a 1-D DOUBLE col var.
        if (isArrayVar(name) && arrays_.at(name).isLocal && !arrays_.at(name).is2D
            && !arrays_.at(name).isND && rhs.type == NodeType::CALL && rhs.children.size() == 4
            && rhs.children[0]->type == NodeType::IDENTIFIER
            && rhs.children[0]->strValue == "repmat"
            && rhs.children[1]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[1]->strValue)
            && !arrays_.at(rhs.children[1]->strValue).is2D
            && !arrays_.at(rhs.children[1]->strValue).isND
            && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE
            && rhs.children[3]->type == NodeType::NUMBER_LITERAL
            && rhs.children[3]->numValue == 1.0) {
            const ArrayInfo    &x   = arrays_.at(rhs.children[1]->strValue);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete() && !res.type.shape.isScalar()) {
                const std::string p = emitExpr(*rhs.children[2]);
                line("{");
                ++indent_;
                line("const std::size_t _numkit_p = numkit_rt::dim(" + p + ");");
                line(cppIdent(name) + ".clear();");
                open("for (std::size_t _numkit_k = 0; _numkit_k < _numkit_p; ++_numkit_k)");
                open("for (std::size_t _numkit_i = 0; _numkit_i < " + x.lenVar + "; ++_numkit_i)");
                line(cppIdent(name) + ".push_back(" + x.dataExpr + "[_numkit_i]);");
                close();
                close();
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // repmat(x, p, q) with x a ROW vector and p>1 -> a true 2-D p x (q*n) tiling, a
        // rank-2 ndRuntimeLocal (rows = p known, cols = q*n runtime; n = x's length).
        // Column-major: M[r + c*p] = x[c % n] -- every output row is a copy of the 1xn
        // source, and the columns repeat the source every n. The (1,q) row tile and the
        // (p,1) col tile above stay 1-D; this is the genuinely-2-D case. v1: x a 1-D
        // DOUBLE row var, p,q literals, p>1.
        if (isArrayVar(name) && arrays_.at(name).isLocal && arrays_.at(name).ndRuntimeLocal
            && arrays_.at(name).ndDims.size() == 2 && rhs.type == NodeType::CALL
            && rhs.children.size() == 4 && rhs.children[0]->type == NodeType::IDENTIFIER
            && rhs.children[0]->strValue == "repmat"
            && rhs.children[1]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[1]->strValue)
            && !arrays_.at(rhs.children[1]->strValue).is2D
            && !arrays_.at(rhs.children[1]->strValue).isND
            && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE
            && inferExpr(*rhs.children[1], types_, reg_, classes_).type.shape.kind
                   == ShapeKind::RowVector) {
            const ArrayInfo    &x   = arrays_.at(rhs.children[1]->strValue);
            const ArrayInfo    &M   = arrays_.at(name);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            const std::string   p   = emitExpr(*rhs.children[2]);
            const std::string   q   = emitExpr(*rhs.children[3]);
            line("{");
            ++indent_;
            line("const std::size_t _numkit_p = numkit_rt::dim(" + p + ");");
            line("const std::size_t _numkit_n = " + x.lenVar + ";");
            line("const std::size_t _numkit_cols = numkit_rt::dim(" + q + ") * _numkit_n;");
            line(M.ndDims[0] + " = _numkit_p;");
            line(M.ndDims[1] + " = _numkit_cols;");
            line(cppIdent(name) + ".assign(_numkit_p * _numkit_cols, 0.0);");
            open("for (std::size_t _numkit_c = 0; _numkit_c < _numkit_cols; ++_numkit_c)");
            open("for (std::size_t _numkit_r = 0; _numkit_r < _numkit_p; ++_numkit_r)");
            line(cppIdent(name) + "[_numkit_r + _numkit_c * _numkit_p] = " + x.dataExpr + "[_numkit_c % _numkit_n];");
            close();
            close();
            --indent_;
            line("}");
            types_.set(name, res);
            return;
        }
        // repmat(x, p, q) with x a COLUMN vector and q>1 -> a true 2-D (p*n) x q tiling, a
        // rank-2 ndRuntimeLocal (rows = p*n runtime, cols = q known; n = x's length).
        // Column-major: M[r + c*(p*n)] = x[r % n] -- every output column is p stacked copies
        // of the n x 1 source. The mirror of the row tile above; gated on a ColVector
        // operand. v1: x a 1-D DOUBLE col var, q>1.
        if (isArrayVar(name) && arrays_.at(name).isLocal && arrays_.at(name).ndRuntimeLocal
            && arrays_.at(name).ndDims.size() == 2 && rhs.type == NodeType::CALL
            && rhs.children.size() == 4 && rhs.children[0]->type == NodeType::IDENTIFIER
            && rhs.children[0]->strValue == "repmat"
            && rhs.children[1]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[1]->strValue)
            && !arrays_.at(rhs.children[1]->strValue).is2D
            && !arrays_.at(rhs.children[1]->strValue).isND
            && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE
            && inferExpr(*rhs.children[1], types_, reg_, classes_).type.shape.kind
                   == ShapeKind::ColVector) {
            const ArrayInfo    &x   = arrays_.at(rhs.children[1]->strValue);
            const ArrayInfo    &M   = arrays_.at(name);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            const std::string   p   = emitExpr(*rhs.children[2]);
            const std::string   q   = emitExpr(*rhs.children[3]);
            line("{");
            ++indent_;
            line("const std::size_t _numkit_n = " + x.lenVar + ";");
            line("const std::size_t _numkit_rows = numkit_rt::dim(" + p + ") * _numkit_n;");
            line("const std::size_t _numkit_q = numkit_rt::dim(" + q + ");");
            line(M.ndDims[0] + " = _numkit_rows;");
            line(M.ndDims[1] + " = _numkit_q;");
            line(cppIdent(name) + ".assign(_numkit_rows * _numkit_q, 0.0);");
            open("for (std::size_t _numkit_c = 0; _numkit_c < _numkit_q; ++_numkit_c)");
            open("for (std::size_t _numkit_r = 0; _numkit_r < _numkit_rows; ++_numkit_r)");
            line(cppIdent(name) + "[_numkit_r + _numkit_c * _numkit_rows] = " + x.dataExpr + "[_numkit_r % _numkit_n];");
            close();
            close();
            --indent_;
            line("}");
            types_.set(name, res);
            return;
        }
        // repmat(A, p, q) with A a 2-D MATRIX -> a (p*Arows) x (q*Acols) block tiling, a
        // rank-2 ndRuntimeLocal. Column-major: M[R + C*(p*Arows)] = A[(R % Arows) +
        // (C % Acols)*Arows] -- A repeats every Arows rows and every Acols cols. A's dims
        // via dimExpr (KnownDims or runtime-dim 2-D). v1: A a DOUBLE matrix var, distinct
        // from the dest (an in-place M=repmat(M,..) would read the zeroed/resized buffer).
        if (isArrayVar(name) && arrays_.at(name).isLocal && arrays_.at(name).ndRuntimeLocal
            && arrays_.at(name).ndDims.size() == 2 && rhs.type == NodeType::CALL
            && rhs.children.size() == 4 && rhs.children[0]->type == NodeType::IDENTIFIER
            && rhs.children[0]->strValue == "repmat"
            && rhs.children[1]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[1]->strValue)
            && rhs.children[1]->strValue != name
            && (arrays_.at(rhs.children[1]->strValue).is2D
                || (arrays_.at(rhs.children[1]->strValue).isND
                    && arrays_.at(rhs.children[1]->strValue).ndDims.size() == 2))
            && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE) {
            const ArrayInfo    &A   = arrays_.at(rhs.children[1]->strValue);
            const ArrayInfo    &M   = arrays_.at(name);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            const std::string   p   = emitExpr(*rhs.children[2]);
            const std::string   q   = emitExpr(*rhs.children[3]);
            line("{");
            ++indent_;
            line("const std::size_t _numkit_ar = " + dimExpr(A, 0) + ";");
            line("const std::size_t _numkit_ac = " + dimExpr(A, 1) + ";");
            line("const std::size_t _numkit_rows = numkit_rt::dim(" + p + ") * _numkit_ar;");
            line("const std::size_t _numkit_cols = numkit_rt::dim(" + q + ") * _numkit_ac;");
            line(M.ndDims[0] + " = _numkit_rows;");
            line(M.ndDims[1] + " = _numkit_cols;");
            line(cppIdent(name) + ".assign(_numkit_rows * _numkit_cols, 0.0);");
            open("for (std::size_t _numkit_C = 0; _numkit_C < _numkit_cols; ++_numkit_C)");
            open("for (std::size_t _numkit_R = 0; _numkit_R < _numkit_rows; ++_numkit_R)");
            line(cppIdent(name) + "[_numkit_R + _numkit_C * _numkit_rows] = " + A.dataExpr
                 + "[(_numkit_R % _numkit_ar) + (_numkit_C % _numkit_ac) * _numkit_ar];");
            close();
            close();
            --indent_;
            line("}");
            types_.set(name, res);
            return;
        }
        // eye(n) / eye(m, n) -> the identity matrix (1 on the main diagonal, 0 else), a
        // 2-D KnownDims LOCAL. Like zeros but the diagonal is set to 1. v1: literal dims.
        if (isArrayVar(name) && arrays_.at(name).isLocal && arrays_.at(name).is2D
            && rhs.type == NodeType::CALL && !rhs.children.empty()
            && rhs.children[0]->type == NodeType::IDENTIFIER && rhs.children[0]->strValue == "eye") {
            const ArrayInfo    &v   = arrays_.at(name);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete() && res.type.shape.kind == ShapeKind::KnownDims) {
                line("{");
                ++indent_;
                line(cppIdent(name) + ".assign(static_cast<std::size_t>(" + v.rowsVar + " * " + v.colsVar
                     + "), 0.0);");
                line("const std::size_t _numkit_d = " + v.rowsVar + " < " + v.colsVar + " ? " + v.rowsVar
                     + " : " + v.colsVar + ";");
                open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_d; ++_numkit_i)");
                line(cppIdent(name) + "[_numkit_i + _numkit_i * " + v.rowsVar + "] = 1.0;");
                close();
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // eye(n) / eye(m,n) with a RUNTIME dim -> the identity as a rank-2 ndRuntimeLocal
        // (the runtime mirror of the KnownDims eye above + zeros(m,n) runtime). Dims from
        // the args via numkit_rt::dim (eye(n): cols = rows); zero the buffer, set the main
        // diagonal M[i + i*rows] = 1 for i < min(rows, cols). v1: 1 or 2 scalar dim args.
        if (isArrayVar(name) && arrays_.at(name).isLocal && arrays_.at(name).ndRuntimeLocal
            && arrays_.at(name).ndDims.size() == 2 && rhs.type == NodeType::CALL
            && (rhs.children.size() == 2 || rhs.children.size() == 3)
            && rhs.children[0]->type == NodeType::IDENTIFIER
            && rhs.children[0]->strValue == "eye") {
            const ArrayInfo    &M   = arrays_.at(name);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete()) {
                line("{");
                ++indent_;
                line("const std::size_t _numkit_r = numkit_rt::dim(" + emitExpr(*rhs.children[1]) + ");");
                if (rhs.children.size() == 3)
                    line("const std::size_t _numkit_c = numkit_rt::dim(" + emitExpr(*rhs.children[2]) + ");");
                else
                    line("const std::size_t _numkit_c = _numkit_r;");  // eye(n): n x n
                line(M.ndDims[0] + " = _numkit_r;");
                line(M.ndDims[1] + " = _numkit_c;");
                line(cppIdent(name) + ".assign(_numkit_r * _numkit_c, 0.0);");
                line("const std::size_t _numkit_d = _numkit_r < _numkit_c ? _numkit_r : _numkit_c;");
                open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_d; ++_numkit_i)");
                line(cppIdent(name) + "[_numkit_i + _numkit_i * _numkit_r] = 1.0;");
                close();
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // tril(A[,k]) / triu(A[,k]) -> the lower/upper triangular part of a 2-D matrix (the
        // other triangle zeroed), SAME shape as A. The optional diagonal offset k shifts the
        // boundary: tril keeps A(i,j) where i >= j - k (k>0 keeps super-diagonals, k<0 drops
        // sub-diagonals); triu keeps i <= j - k (k=0 is the main diagonal). Both tiers via
        // dimExpr (KnownDims 2-D or runtime-dim 2-D). Column-major out[i+j*rows] = keep ?
        // A[...] : 0. v1: A a DOUBLE matrix var, a DISTINCT local (in-place refused -- the
        // zero-then-copy would alias).
        if (isArrayVar(name) && arrays_.at(name).isLocal
            && (arrays_.at(name).is2D
                || (arrays_.at(name).isND && arrays_.at(name).ndDims.size() == 2))
            && rhs.type == NodeType::CALL
            && (rhs.children.size() == 2 || rhs.children.size() == 3)
            && rhs.children[0]->type == NodeType::IDENTIFIER
            && (rhs.children[0]->strValue == "tril" || rhs.children[0]->strValue == "triu")
            && rhs.children[1]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[1]->strValue)
            && rhs.children[1]->strValue != name
            && (arrays_.at(rhs.children[1]->strValue).is2D
                || (arrays_.at(rhs.children[1]->strValue).isND
                    && arrays_.at(rhs.children[1]->strValue).ndDims.size() == 2))
            && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE) {
            const ArrayInfo    &A   = arrays_.at(rhs.children[1]->strValue);
            const ArrayInfo    &M   = arrays_.at(name);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            const std::string   cmp = rhs.children[0]->strValue == "tril" ? ">=" : "<=";
            if (res.type.isConcrete()) {
                line("{");
                ++indent_;
                line("const std::size_t _numkit_r = " + dimExpr(A, 0) + ";");
                line("const std::size_t _numkit_c = " + dimExpr(A, 1) + ";");
                line("const std::ptrdiff_t _numkit_k = "
                     + (rhs.children.size() == 3
                            ? ("static_cast<std::ptrdiff_t>(" + emitExpr(*rhs.children[2]) + ")")
                            : std::string("0"))
                     + ";");
                if (M.isND && M.ndDims.size() == 2) {  // runtime-dim dst: set companions
                    line(M.ndDims[0] + " = _numkit_r;");
                    line(M.ndDims[1] + " = _numkit_c;");
                }
                line(cppIdent(name) + ".assign(_numkit_r * _numkit_c, 0.0);");
                open("for (std::size_t _numkit_j = 0; _numkit_j < _numkit_c; ++_numkit_j)");
                open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_r; ++_numkit_i)");
                line("if (static_cast<std::ptrdiff_t>(_numkit_i) " + cmp
                     + " static_cast<std::ptrdiff_t>(_numkit_j) - _numkit_k) " + cppIdent(name)
                     + "[_numkit_i + _numkit_j * _numkit_r] = " + A.dataExpr + "[_numkit_i + _numkit_j * _numkit_r];");
                close();
                close();
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // 1-D SLICE read: y = x(a:b) / y = x(a:s:b) -> a sub-array copied into the
        // owned 1-D local y. The colon ranges over x's 1-based positions; `end`
        // inside it = x's length (pushed here, so x(2:end) works). count = the colon
        // count; y[k] = x[(start-1) + k*step] (0-based, forward or reverse step).
        // Bounds-checked against x's length (MATLAB errors on an out-of-range slice).
        // v1: x a 1-D DOUBLE array VAR, y an owned 1-D LOCAL.
        if (isArrayVar(name) && arrays_.at(name).isLocal && !arrays_.at(name).is2D
            && !arrays_.at(name).isND && rhs.type == NodeType::CALL && rhs.children.size() == 2
            && rhs.children[0]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[0]->strValue)
            && rhs.children[1]->type == NodeType::COLON_EXPR
            && !arrays_.at(rhs.children[0]->strValue).is2D
            && !arrays_.at(rhs.children[0]->strValue).isND
            && arrays_.at(rhs.children[0]->strValue).dtype == ValueType::DOUBLE
            && (rhs.children[1]->children.size() == 2 || rhs.children[1]->children.size() == 3)) {
            const ArrayInfo  &xa    = arrays_.at(rhs.children[0]->strValue);
            const ASTNode    &colon = *rhs.children[1];
            const bool        three = colon.children.size() == 3;
            endStack_.push_back(xa.lenVar);  // `end` inside the slice = x's length
            const std::string start = emitExpr(*colon.children[0]);
            const std::string step  = three ? emitExpr(*colon.children[1]) : std::string("1.0");
            const std::string stop  = emitExpr(*colon.children[three ? 2 : 1]);
            endStack_.pop_back();
            line("{");
            ++indent_;
            line("std::ptrdiff_t _numkit_s0 = 0, _numkit_st = 0;");
            line("const std::ptrdiff_t _numkit_cnt = numkit_rt::slice_bounds(" + start + ", "
                 + step + ", " + stop + ", " + xa.lenVar + ", _numkit_s0, _numkit_st);");
            line(cppIdent(name) + ".assign(static_cast<std::size_t>(_numkit_cnt), 0.0);");
            open("for (std::ptrdiff_t _numkit_k = 0; _numkit_k < _numkit_cnt; ++_numkit_k)");
            line(cppIdent(name) + "[static_cast<std::size_t>(_numkit_k)] = " + xa.dataExpr
                 + "[static_cast<std::size_t>(_numkit_s0 + _numkit_k * _numkit_st)];");
            close();
            --indent_;
            line("}");
            return;
        }
        // `s = size(A)` (no dim): fill a 1 x rank row with A's per-axis sizes.
        // rank is compile-time (2 for a scalar/vector/matrix; the array's rank
        // for N-D). Native + self-contained — the first array-RESULT-from-a-
        // builtin producer beyond zeros/ones. BEFORE the bridged path, so it
        // lowers natively instead of boxing into numkit::size.
        if (isArrayVar(name) && !arrays_.at(name).is2D && !arrays_.at(name).isND
            && rhs.type == NodeType::CALL && rhs.children.size() == 2
            && rhs.children[0]->type == NodeType::IDENTIFIER && rhs.children[0]->strValue == "size"
            && rhs.children[1]->type == NodeType::IDENTIFIER) {
            const std::string &opnd = rhs.children[1]->strValue;
            const ArrayInfo   &sai  = arrays_.at(name);
            if (isArrayVar(opnd)) {
                const ArrayInfo  &aai  = arrays_.at(opnd);
                const std::size_t rank = aai.isND ? aai.ndDims.size() : 2;
                if (sai.isLocal) line(cppIdent(name) + ".assign(" + std::to_string(rank) + ", 0.0);");
                for (std::size_t k = 0; k < rank; ++k)
                    line(sai.dataExpr + "[" + std::to_string(k) + "] = static_cast<double>("
                         + dimExpr(aai, k) + ");");
                types_.set(name, {InferredType::concrete(sai.dtype, Shape::rowVector()),
                                  ConstVal::unknown()});
                return;
            }
            // A scalar operand: size is [1 1] (rank 2). Native + self-contained.
            if (inferExpr(*rhs.children[1], types_, reg_, classes_).type.shape.kind
                == ShapeKind::Scalar) {
                if (sai.isLocal) line(cppIdent(name) + ".assign(2, 1.0);");
                else { line(sai.dataExpr + "[0] = 1.0;"); line(sai.dataExpr + "[1] = 1.0;"); }
                types_.set(name, {InferredType::concrete(sai.dtype, Shape::rowVector()),
                                  ConstVal::unknown()});
                return;
            }
        }
        // Transpose: y = x' (ctranspose) / y = x.' (transpose). A 1-D vector
        // flips orientation; a 2-D matrix swaps its dims (column-major). The
        // data is copied, and ctranspose (') additionally conjugates a complex
        // operand. The result shape comes from the transfer. N-D transpose is
        // undefined in MATLAB -> refused (an explicit boundary). Native +
        // self-contained.
        if (isArrayVar(name) && (arrays_.at(name).isOutput || arrays_.at(name).isLocal)
            && rhs.type == NodeType::UNARY_OP && (rhs.strValue == "'" || rhs.strValue == ".'")
            && rhs.children.size() == 1 && rhs.children[0]->type == NodeType::IDENTIFIER
            && isArrayVar(rhs.children[0]->strValue)) {
            const ArrayInfo &dst = arrays_.at(name);
            const ArrayInfo &src = arrays_.at(rhs.children[0]->strValue);
            // A runtime-dim 2-D matrix is an NDims rank-2 (ndRuntimeLocal); its dims live
            // in ndDims, not rowsVar/colsVar. Allow the rank-2 N-D case through; only a
            // true N-D (rank>=3) transpose is undefined in MATLAB -> refused.
            const bool srcRank2ND = src.isND && src.ndDims.size() == 2;
            const bool dstRank2ND = dst.isND && dst.ndDims.size() == 2;
            if ((src.isND && !srcRank2ND) || (dst.isND && !dstRank2ND))
                unsupported("N-D transpose (not defined for N-D arrays)");
            if (name == rhs.children[0]->strValue)
                unsupported("in-place transpose (y = y')");
            const bool conj = rhs.strValue == "'" && src.dtype == ValueType::COMPLEX;
            if (srcRank2ND || dstRank2ND) {
                // Runtime-dim 2-D transpose: y is n x m (= src cols x src rows), and
                // y(p,q) = A(q,p) -> y[p + q*yrows] = op(A[q + p*Arows]). Dims and the
                // buffer come from the ndDims companions; yrows = src cols = src.ndDims[1].
                if (!srcRank2ND || !dstRank2ND)
                    unsupported("transpose dest/source rank mismatch");
                line(dst.ndDims[0] + " = " + src.ndDims[1] + ";");  // dst rows = src cols
                line(dst.ndDims[1] + " = " + src.ndDims[0] + ";");  // dst cols = src rows
                line(cppIdent(name) + ".assign(" + dst.ndDims[0] + " * " + dst.ndDims[1] + ", "
                     + zeroLiteral(dst.dtype) + ");");
                open("for (std::size_t _numkit_j = 0; _numkit_j < " + dst.ndDims[1] + "; ++_numkit_j)");
                open("for (std::size_t _numkit_i = 0; _numkit_i < " + dst.ndDims[0] + "; ++_numkit_i)");
                const std::string rd = src.dataExpr + "[_numkit_j + _numkit_i * " + src.ndDims[0] + "]";
                line(cppIdent(name) + "[_numkit_i + _numkit_j * " + dst.ndDims[0] + "] = "
                     + (conj ? ("std::conj(" + rd + ")") : rd) + ";");
                close();
                close();
            } else if (src.is2D || dst.is2D) {
                // 2-D matrix transpose, column-major: y is n x m, A is m x n,
                // and y(p,q) = A(q,p) -> y[p + q*yrows] = op(A[q + p*Arows]).
                if (!src.is2D || !dst.is2D)
                    unsupported("transpose dest/source rank mismatch");
                if (dst.isLocal)
                    line(cppIdent(name) + ".assign(" + dst.rowsVar + " * " + dst.colsVar + ", "
                         + zeroLiteral(dst.dtype) + ");");
                open("for (std::size_t _numkit_j = 0; _numkit_j < " + dst.colsVar + "; ++_numkit_j)");
                open("for (std::size_t _numkit_i = 0; _numkit_i < " + dst.rowsVar + "; ++_numkit_i)");
                const std::string rd = src.dataExpr + "[_numkit_j + _numkit_i * " + src.rowsVar + "]";
                line(dst.dataExpr + "[_numkit_i + _numkit_j * " + dst.rowsVar + "] = "
                     + (conj ? ("std::conj(" + rd + ")") : rd) + ";");
                close();
                close();
            } else {
                // 1-D vector transpose: orientation flip, element-for-element
                // copy (ctranspose conjugates a complex operand).
                if (dst.isLocal) line(cppIdent(name) + ".resize(" + src.lenVar + ");");
                open("for (std::size_t _numkit_i = 0; _numkit_i < " + src.lenVar + "; ++_numkit_i)");
                const std::string rd = src.dataExpr + "[_numkit_i]";
                line(dst.dataExpr + "[_numkit_i] = " + (conj ? ("std::conj(" + rd + ")") : rd) + ";");
                close();
                if (src.orient == VecOrient::Row)
                    arrays_.at(name).orient = VecOrient::Col;
                else if (src.orient == VecOrient::Col)
                    arrays_.at(name).orient = VecOrient::Row;
            }
            types_.set(name, inferExpr(rhs, types_, reg_, classes_));
            return;
        }
        // Fused elementwise + transpose: `C = (A + B)'`, `C = (2 * A - 1).'`, etc.
        // Evaluates the pure-elementwise inner expression at source (j, i) and writes to
        // destination (i, j) (conjugating for `'` on complex) in a single pass with no temp.
        if (isArrayVar(name) && (arrays_.at(name).isOutput || arrays_.at(name).isLocal)
            && rhs.type == NodeType::UNARY_OP && (rhs.strValue == "'" || rhs.strValue == ".'")
            && rhs.children.size() == 1 && rhs.children[0]->type != NodeType::IDENTIFIER) {
            std::set<std::string> srcArrays;
            if (collectElementwise(*rhs.children[0], srcArrays) && !srcArrays.empty()
                && srcArrays.count(name) == 0) {
                const ArrayInfo    &dst  = arrays_.at(name);
                const bool          conj = rhs.strValue == "'" && dst.dtype == ValueType::COMPLEX;
                const AbstractValue res  = inferExpr(rhs, types_, reg_, classes_);
                bool all2D = true, all1D = true;
                for (const std::string &an : srcArrays) {
                    const ArrayInfo &sa = arrays_.at(an);
                    const bool       m2 = sa.is2D || (sa.isND && sa.ndDims.size() == 2);
                    if (!m2) all2D = false;
                    if (sa.is2D || sa.isND) all1D = false;
                }
                const bool dstMat = dst.is2D || (dst.isND && dst.ndDims.size() == 2);
                if (all2D && dstMat && res.type.isConcrete()) {
                    const ArrayInfo  &ref     = arrays_.at(*srcArrays.begin());
                    const std::string srcRows = dimExpr(ref, 0), srcCols = dimExpr(ref, 1);
                    std::string       guard;
                    for (const std::string &an : srcArrays) {
                        const ArrayInfo  &sa = arrays_.at(an);
                        const std::string sr = dimExpr(sa, 0), sc = dimExpr(sa, 1);
                        if (sr != srcRows || sc != srcCols)
                            guard += (guard.empty() ? "" : " || ")
                                     + ("(" + sr + " != " + srcRows + " || " + sc + " != " + srcCols + ")");
                    }
                    if (!guard.empty())
                        line("if (" + guard + ") throw std::out_of_range(\"numkit: array dimensions must match\");");
                    if (dst.isND && dst.ndRuntimeLocal) {
                        line(dst.ndDims[0] + " = " + srcCols + ";");
                        line(dst.ndDims[1] + " = " + srcRows + ";");
                        line(cppIdent(name) + ".assign(" + dst.ndDims[0] + " * " + dst.ndDims[1] + ", "
                             + zeroLiteral(dst.dtype) + ");");
                    } else if (dst.isLocal) {
                        line(cppIdent(name) + ".assign(" + dimExpr(dst, 0) + " * " + dimExpr(dst, 1) + ", "
                             + zeroLiteral(dst.dtype) + ");");
                    } else {
                        line("if (" + dimExpr(dst, 0) + " != " + srcCols + " || "
                             + dimExpr(dst, 1) + " != " + srcRows
                             + ") throw std::out_of_range(\"numkit: transpose dimensions must agree\");");
                    }
                    const std::string dstRows = dimExpr(dst, 0), dstCols = dimExpr(dst, 1);
                    elementCtx_  = "_numkit_j + _numkit_i * " + srcRows;
                    elementRows_ = srcRows;
                    elementCols_ = srcCols;
                    const std::string inner = emitExpr(*rhs.children[0]);
                    elementCtx_.clear();
                    elementRows_.clear();
                    elementCols_.clear();
                    open("for (std::size_t _numkit_j = 0; _numkit_j < " + dstCols + "; ++_numkit_j)");
                    open("for (std::size_t _numkit_i = 0; _numkit_i < " + dstRows + "; ++_numkit_i)");
                    line(dst.dataExpr + "[_numkit_i + _numkit_j * " + dstRows + "] = "
                         + (conj ? ("std::conj(" + inner + ")") : inner) + ";");
                    close();
                    close();
                    types_.set(name, res);
                    return;
                } else if (all1D && !dst.is2D && !dst.isND && res.type.isConcrete()) {
                    const ArrayInfo  &ref   = arrays_.at(*srcArrays.begin());
                    const std::string bound = dst.isLocal ? ref.lenVar : dst.lenVar;
                    std::string       guard;
                    for (const std::string &an : srcArrays) {
                        const std::string &alen = arrays_.at(an).lenVar;
                        if (alen != bound)
                            guard += (guard.empty() ? "" : " || ") + (alen + " != " + bound);
                    }
                    if (!guard.empty())
                        line("if (" + guard + ") throw std::out_of_range(\"numkit: array dimensions must match\");");
                    if (dst.isLocal) line(cppIdent(name) + ".resize(" + bound + ");");
                    elementCtx_  = "_numkit_i";
                    elementRows_ = ref.orient == VecOrient::Col ? bound : std::string("1");
                    elementCols_ = ref.orient == VecOrient::Row ? bound : std::string("1");
                    const std::string inner = emitExpr(*rhs.children[0]);
                    elementCtx_.clear();
                    elementRows_.clear();
                    elementCols_.clear();
                    open("for (std::size_t _numkit_i = 0; _numkit_i < " + bound + "; ++_numkit_i)");
                    line(dst.dataExpr + "[_numkit_i] = "
                         + (conj ? ("std::conj(" + inner + ")") : inner) + ";");
                    close();
                    if (ref.orient == VecOrient::Row)
                        arrays_.at(name).orient = VecOrient::Col;
                    else if (ref.orient == VecOrient::Col)
                        arrays_.at(name).orient = VecOrient::Row;
                    types_.set(name, res);
                    return;
                }
            }
        }
        // Matrix product: C = A * B (both 2-D). C is m x n (A is m x k, B is
        // k x n), C(i,j) = sum_l A(i,l)*B(l,j), column-major. The shared dim
        // must agree (runtime guard, MATLAB-like). With ops kernels enabled +
        // a DOUBLE result, lower to numkit::ops::matmulDouble (the SIMD kernel
        // ops owns); otherwise an inline triple loop (the deletable fallback +
        // the complex path — no complex ops kernel yet). (Scalar*X / X*scalar
        // are elementwise scaling, handled above; matrix*vector needs a 1-D
        // operand and is not yet lowered.)
        if (isArrayVar(name) && (arrays_.at(name).isOutput || arrays_.at(name).isLocal)
            && (arrays_.at(name).is2D
                || (arrays_.at(name).isND && arrays_.at(name).ndDims.size() == 2))
            && rhs.type == NodeType::BINARY_OP && rhs.strValue == "*"
            && rhs.children.size() == 2 && rhs.children[0]->type == NodeType::IDENTIFIER
            && isArrayVar(rhs.children[0]->strValue)
            && rhs.children[1]->type == NodeType::IDENTIFIER
            && isArrayVar(rhs.children[1]->strValue)) {
            const ArrayInfo &dst = arrays_.at(name);
            const ArrayInfo &A   = arrays_.at(rhs.children[0]->strValue);
            const ArrayInfo &B   = arrays_.at(rhs.children[1]->strValue);
            // A rank-2 operand is a matrix — KnownDims 2-D OR a runtime-dim 2-D (NDims
            // rank-2). Only matrix*matrix is lowered here; with a vector operand (a
            // matrix*vector, or a col*row outer product whose result is also 2-D) this
            // branch falls through to the matrix*vector / outer-product branches below.
            const bool aMatrix = A.is2D || (A.isND && A.ndDims.size() == 2);
            const bool bMatrix = B.is2D || (B.isND && B.ndDims.size() == 2);
            if (aMatrix && bMatrix) {
                if (name == rhs.children[0]->strValue || name == rhs.children[1]->strValue)
                    unsupported("in-place matrix product (C = C * B)");
                // dimExpr is rank-agnostic: a KnownDims matrix yields its rows/colsVar
                // literals, a runtime-dim 2-D yields its ndDims companions. So the same
                // lowering serves both (the runtime dst gets its companions set first).
                const std::string Arows = dimExpr(A, 0), Ak = dimExpr(A, 1);  // A is m x k
                const std::string Brows = dimExpr(B, 0), Bn = dimExpr(B, 1);  // B is k x n
                const bool        dstRuntime = dst.isND && dst.ndDims.size() == 2;
                line("if (" + Ak + " != " + Brows
                     + ") throw std::out_of_range(\"numkit: inner matrix dimensions must agree\");");
                if (dstRuntime) {
                    line(dst.ndDims[0] + " = " + Arows + ";");  // C rows = A rows (m)
                    line(dst.ndDims[1] + " = " + Bn + ";");     // C cols = B cols (n)
                }
                const std::string Crows = dimExpr(dst, 0), Ccols = dimExpr(dst, 1);
                if (opsKernels_
                    && (dst.dtype == ValueType::DOUBLE || dst.dtype == ValueType::COMPLEX)) {
                    // ops owns the kernel: M=C rows, N=C cols, K=A cols (==B rows,
                    // guarded). The kernel zeroes+accumulates; a LOCAL still needs its
                    // owned vector sized first. DOUBLE -> SIMD matmulDouble; COMPLEX ->
                    // portable matmulComplex (amortised over O(M·N·K), no per-elem cost).
                    const char *fn =
                        dst.dtype == ValueType::DOUBLE ? "matmulDouble" : "matmulComplex";
                    if (dst.isLocal)
                        line(cppIdent(name) + ".resize(" + Crows + " * " + Ccols + ");");
                    line("numkit::ops::" + std::string(fn) + "(" + A.dataExpr + ", "
                         + B.dataExpr + ", " + dst.dataExpr + ", " + Crows + ", " + Ccols
                         + ", " + Ak + ");");
                } else {
                    if (dst.isLocal)
                        line(cppIdent(name) + ".assign(" + Crows + " * " + Ccols + ", "
                             + zeroLiteral(dst.dtype) + ");");
                    open("for (std::size_t _numkit_j = 0; _numkit_j < " + Ccols + "; ++_numkit_j)");
                    open("for (std::size_t _numkit_i = 0; _numkit_i < " + Crows + "; ++_numkit_i)");
                    line(cppScalarType(dst.dtype) + " _numkit_acc = " + zeroLiteral(dst.dtype) + ";");
                    open("for (std::size_t _numkit_l = 0; _numkit_l < " + Ak + "; ++_numkit_l)");
                    line("_numkit_acc += " + A.dataExpr + "[_numkit_i + _numkit_l * " + Arows + "] * "
                         + B.dataExpr + "[_numkit_l + _numkit_j * " + Brows + "];");
                    close();
                    line(dst.dataExpr + "[_numkit_i + _numkit_j * " + Crows + "] = _numkit_acc;");
                    close();
                    close();
                }
                types_.set(name, inferExpr(rhs, types_, reg_, classes_));
                return;
            }
        }
        // Matrix * vector / vector * matrix -> a vector. A*x (A m x k, x k x 1)
        // -> y m x 1: y[i] = sum_l A[i+l*Arows]*x[l]. r*A (r 1 x k, A k x n)
        // -> y 1 x n: y[j] = sum_l r[l]*A[l+j*Arows]. Native double loop +
        // shared-dim guard. (vector*vector inner/outer products not yet
        // lowered.) Self-contained.
        if (isArrayVar(name) && (arrays_.at(name).isOutput || arrays_.at(name).isLocal)
            && !arrays_.at(name).is2D && !arrays_.at(name).isND
            && rhs.type == NodeType::BINARY_OP && rhs.strValue == "*"
            && rhs.children.size() == 2 && rhs.children[0]->type == NodeType::IDENTIFIER
            && isArrayVar(rhs.children[0]->strValue)
            && rhs.children[1]->type == NodeType::IDENTIFIER
            && isArrayVar(rhs.children[1]->strValue)) {
            const ArrayInfo &dst = arrays_.at(name);
            const ArrayInfo &L   = arrays_.at(rhs.children[0]->strValue);
            const ArrayInfo &R   = arrays_.at(rhs.children[1]->strValue);
            if (name == rhs.children[0]->strValue || name == rhs.children[1]->strValue)
                unsupported("in-place matrix*vector (y = y * x)");
            // A matrix operand is KnownDims 2-D OR a runtime-dim 2-D (NDims rank-2); a
            // vector is 1-D (neither). dimExpr reads the matrix dims rank-agnostically
            // (rows/colsVar literals or ndDims companions). A true N-D (rank>=3) operand
            // is refused. (matrix*matrix went to the matmul branch above; both-vector
            // falls to the outer-product / refusal below.)
            if ((L.isND && L.ndDims.size() != 2) || (R.isND && R.ndDims.size() != 2))
                unsupported("N-D operand in matrix*vector");
            const bool        lMat = L.is2D || (L.isND && L.ndDims.size() == 2);
            const bool        rMat = R.is2D || (R.isND && R.ndDims.size() == 2);
            const bool        lVec = !L.is2D && !L.isND;
            const bool        rVec = !R.is2D && !R.isND;
            const std::string ty   = cppScalarType(dst.dtype);
            const std::string zer  = zeroLiteral(dst.dtype);
            if (lMat && rVec) {  // A * x -> column vector
                const std::string Arows = dimExpr(L, 0), Acols = dimExpr(L, 1);
                line("if (" + Acols + " != " + R.lenVar + ") throw std::out_of_range(\""
                     "numkit: inner matrix dimensions must agree\");");
                if (dst.isLocal) line(cppIdent(name) + ".resize(" + Arows + ");");
                open("for (std::size_t _numkit_i = 0; _numkit_i < " + Arows + "; ++_numkit_i)");
                line(ty + " _numkit_acc = " + zer + ";");
                open("for (std::size_t _numkit_l = 0; _numkit_l < " + Acols + "; ++_numkit_l)");
                line("_numkit_acc += " + L.dataExpr + "[_numkit_i + _numkit_l * " + Arows + "] * "
                     + R.dataExpr + "[_numkit_l];");
                close();
                line(dst.dataExpr + "[_numkit_i] = _numkit_acc;");
                close();
            } else if (lVec && rMat) {  // r * A -> row vector
                const std::string Brows = dimExpr(R, 0), Bcols = dimExpr(R, 1);
                line("if (" + L.lenVar + " != " + Brows + ") throw std::out_of_range(\""
                     "numkit: inner matrix dimensions must agree\");");
                if (dst.isLocal) line(cppIdent(name) + ".resize(" + Bcols + ");");
                open("for (std::size_t _numkit_j = 0; _numkit_j < " + Bcols + "; ++_numkit_j)");
                line(ty + " _numkit_acc = " + zer + ";");
                open("for (std::size_t _numkit_l = 0; _numkit_l < " + Brows + "; ++_numkit_l)");
                line("_numkit_acc += " + L.dataExpr + "[_numkit_l] * " + R.dataExpr + "[_numkit_l + _numkit_j * "
                     + Brows + "];");
                close();
                line(dst.dataExpr + "[_numkit_j] = _numkit_acc;");
                close();
            } else {
                unsupported("vector*vector with a vector result (use the outer-product path)");
            }
            types_.set(name, inferExpr(rhs, types_, reg_, classes_));
            return;
        }
        // Outer product: y = c * r (c m x 1, r 1 x n) -> y m x n, column-major
        // y[i + j*m] = c[i]*r[j]. Works for both KnownDims 2-D and runtime-dim rank-2
        // destinations. A LOCAL sets its dim vars from the operand lengths + sizes its
        // vector; an OUTPUT guards its caller-passed dims. Complex accumulates in
        // std::complex. Native + self-contained.
        if (isArrayVar(name) && (arrays_.at(name).isOutput || arrays_.at(name).isLocal)
            && (arrays_.at(name).is2D || (arrays_.at(name).isND && arrays_.at(name).ndDims.size() == 2))
            && rhs.type == NodeType::BINARY_OP && rhs.strValue == "*"
            && rhs.children.size() == 2 && rhs.children[0]->type == NodeType::IDENTIFIER
            && isArrayVar(rhs.children[0]->strValue)
            && rhs.children[1]->type == NodeType::IDENTIFIER
            && isArrayVar(rhs.children[1]->strValue)) {
            const ArrayInfo &dst = arrays_.at(name);
            const ArrayInfo &c   = arrays_.at(rhs.children[0]->strValue);
            const ArrayInfo &r   = arrays_.at(rhs.children[1]->strValue);
            if (c.is2D || c.isND || r.is2D || r.isND)
                unsupported("outer product over a non-vector operand");
            if (name == rhs.children[0]->strValue || name == rhs.children[1]->strValue)
                unsupported("in-place outer product");
            const std::string d0 = dimExpr(dst, 0);  // rows = length(c)
            const std::string d1 = dimExpr(dst, 1);  // cols = length(r)
            if (dst.ndRuntimeLocal) {
                line(dst.ndDims[0] + " = " + c.lenVar + ";");
                line(dst.ndDims[1] + " = " + r.lenVar + ";");
                line(cppIdent(name) + ".assign(" + d0 + " * " + d1 + ", " + zeroLiteral(dst.dtype) + ");");
            } else if (dst.isLocal) {
                line(cppIdent(name) + ".assign(" + d0 + " * " + d1 + ", " + zeroLiteral(dst.dtype) + ");");
            } else {  // OUTPUT: caller-allocated + caller-passed dims; guard agreement
                line("if (" + d0 + " != " + c.lenVar + " || " + d1 + " != " + r.lenVar
                     + ") throw std::out_of_range(\"numkit: outer product output dimensions "
                       "must agree\");");
            }
            open("for (std::size_t _numkit_j = 0; _numkit_j < " + d1 + "; ++_numkit_j)");
            open("for (std::size_t _numkit_i = 0; _numkit_i < " + d0 + "; ++_numkit_i)");
            line(dst.dataExpr + "[_numkit_i + _numkit_j * " + d0 + "] = " + c.dataExpr + "[_numkit_i] * "
                 + r.dataExpr + "[_numkit_j];");
            close();
            close();
            types_.set(name, inferExpr(rhs, types_, reg_, classes_));
            return;
        }
        // Inner / dot product: s = r * c (r 1 x k, c k x 1) = sum_l r[l]*c[l].
        // The result is a SCALAR (s is not an array), so this is a reduction
        // loop into a scalar, not an array producer — placed where the scalar
        // LHS is handled. A complex product accumulates in std::complex.
        if (!isArrayVar(name) && rhs.type == NodeType::BINARY_OP && rhs.strValue == "*"
            && rhs.children.size() == 2 && rhs.children[0]->type == NodeType::IDENTIFIER
            && isArrayVar(rhs.children[0]->strValue)
            && rhs.children[1]->type == NodeType::IDENTIFIER
            && isArrayVar(rhs.children[1]->strValue)) {
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete() && res.type.shape.isScalar()) {
                const ArrayInfo &L = arrays_.at(rhs.children[0]->strValue);
                const ArrayInfo &R = arrays_.at(rhs.children[1]->strValue);
                if (L.is2D || R.is2D || L.isND || R.isND)
                    unsupported("inner product over a non-vector operand");
                line("{");
                ++indent_;
                line("if (" + L.lenVar + " != " + R.lenVar + ") throw std::out_of_range(\""
                     "numkit: inner matrix dimensions must agree\");");
                line(cppScalarType(res.type.dtype) + " _numkit_acc = "
                     + zeroLiteral(res.type.dtype) + ";");
                open("for (std::size_t _numkit_l = 0; _numkit_l < " + L.lenVar + "; ++_numkit_l)");
                line("_numkit_acc += " + L.dataExpr + "[_numkit_l] * " + R.dataExpr + "[_numkit_l];");
                close();
                line(cppIdent(name) + " = _numkit_acc;");
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native any(x) / all(x) -> LOGICAL scalar (self-contained; EXACT). An
        // inline short-circuit loop over a 1-D array (double or logical), so no
        // runtime is needed — preferred over the bridged reduction below. NaN is
        // nonzero (any([NaN])=true, all([NaN])=true), matching MATLAB. v1: a single
        // 1-D non-complex array arg, the result directly assigned to the scalar.
        if (!isArrayVar(name) && rhs.type == NodeType::CALL && rhs.children.size() == 2
            && rhs.children[0]->type == NodeType::IDENTIFIER
            && (rhs.children[0]->strValue == "any" || rhs.children[0]->strValue == "all")
            && rhs.children[1]->type == NodeType::IDENTIFIER
            && isArrayVar(rhs.children[1]->strValue)) {
            const ArrayInfo    &a   = arrays_.at(rhs.children[1]->strValue);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (!a.is2D && !a.isND && a.dtype != ValueType::COMPLEX && res.type.isConcrete()
                && res.type.shape.isScalar()) {
                const bool isAny = rhs.children[0]->strValue == "any";
                line("{");  // scope _numkit_acc so repeated any/all in one fn don't collide
                ++indent_;
                line(std::string("bool _numkit_acc = ") + (isAny ? "false;" : "true;"));
                open("for (std::size_t _numkit_i = 0; _numkit_i < " + a.lenVar + "; ++_numkit_i)");
                line(isAny ? ("if (" + a.dataExpr + "[_numkit_i] != 0) { _numkit_acc = true; break; }")
                           : ("if (" + a.dataExpr + "[_numkit_i] == 0) { _numkit_acc = false; break; }"));
                close();
                line(cppIdent(name) + " = _numkit_acc;");
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native min(x) / max(x) -> scalar (self-contained; EXACT, NaN-skipping
        // like MATLAB). An inline fold over a 1-D array, preferred over the bridged
        // reduction. Seed with element 0; for a float dtype `acc != acc` replaces a
        // NaN seed, and a NaN x[i] fails the compare so it is skipped
        // (max([1 NaN 3])=3, max([NaN NaN])=NaN). v1: a single 1-D non-complex
        // array arg (assumed non-empty -> a scalar), the result directly assigned.
        if (!isArrayVar(name) && rhs.type == NodeType::CALL && rhs.children.size() == 2
            && rhs.children[0]->type == NodeType::IDENTIFIER
            && (rhs.children[0]->strValue == "min" || rhs.children[0]->strValue == "max")
            && rhs.children[1]->type == NodeType::IDENTIFIER
            && isArrayVar(rhs.children[1]->strValue)) {
            const ArrayInfo    &a   = arrays_.at(rhs.children[1]->strValue);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (!a.is2D && !a.isND && a.dtype != ValueType::COMPLEX && res.type.isConcrete()
                && res.type.shape.isScalar()) {
                const std::string cmp = rhs.children[0]->strValue == "max" ? ">" : "<";
                const bool        isFloat =
                    a.dtype == ValueType::DOUBLE || a.dtype == ValueType::SINGLE;
                const std::string nanClause = isFloat ? " || _numkit_acc != _numkit_acc" : "";
                line("{");  // scope _numkit_acc so repeated reductions in one fn don't collide
                ++indent_;
                line("if (" + a.lenVar
                     + " == 0) throw std::out_of_range(\"numkit: min/max of empty array requires array output\");");
                line(cppScalarType(a.dtype) + " _numkit_acc = " + a.dataExpr + "[0];");
                open("for (std::size_t _numkit_i = 1; _numkit_i < " + a.lenVar + "; ++_numkit_i)");
                line("if (" + a.dataExpr + "[_numkit_i] " + cmp + " _numkit_acc" + nanClause + ") _numkit_acc = "
                     + a.dataExpr + "[_numkit_i];");
                close();
                line(cppIdent(name) + " = _numkit_acc;");
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native sum / prod / mean(x) -> scalar, but ONLY with no bridge. UNLIKE
        // min/max these are order-dependent: a sequential reduction can differ in
        // the last ULP from the runtime's order, so the bridged path stays the
        // EXACT tier when the bridge is on; native is the self-contained fallback.
        // sum: acc += (seed 0); prod: acc *= (seed 1); mean: sum / len. Empty input
        // matches MATLAB: sum([])=0, prod([])=1, mean([])=NaN (0/0). v1: a single
        // 1-D DOUBLE array arg (logical/int dtype subtleties -> bridged), direct
        // assign.
        if (!isArrayVar(name) && !bridge_ && rhs.type == NodeType::CALL
            && rhs.children.size() == 2 && rhs.children[0]->type == NodeType::IDENTIFIER
            && (rhs.children[0]->strValue == "sum" || rhs.children[0]->strValue == "prod"
                || rhs.children[0]->strValue == "mean")
            && rhs.children[1]->type == NodeType::IDENTIFIER
            && isArrayVar(rhs.children[1]->strValue)
            && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE) {
            const ArrayInfo    &a   = arrays_.at(rhs.children[1]->strValue);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (!a.is2D && !a.isND && res.type.isConcrete() && res.type.shape.isScalar()) {
                const std::string callee = rhs.children[0]->strValue;
                const bool        isProd = callee == "prod";
                line("{");  // scope _numkit_acc so repeated reductions in one fn don't collide
                ++indent_;
                line(std::string("double _numkit_acc = ") + (isProd ? "1.0;" : "0.0;"));
                open("for (std::size_t _numkit_i = 0; _numkit_i < " + a.lenVar + "; ++_numkit_i)");
                line(std::string("_numkit_acc ") + (isProd ? "*=" : "+=") + " " + a.dataExpr
                     + "[_numkit_i];");
                close();
                if (callee == "mean")
                    line(cppIdent(name) + " = _numkit_acc / static_cast<double>(" + a.lenVar + ");");
                else
                    line(cppIdent(name) + " = _numkit_acc;");
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native norm/var/std(x) scalar reductions (1-D DOUBLE), no-bridge tier
        // (gate !bridge_; the bridged path stays exact when the bridge is on). norm =
        // the 2-norm sqrt(sum x^2); var = sum((x-mean)^2)/(n-1) (MATLAB's sample
        // default; 0 for a scalar, NaN for empty); std = sqrt(var). Two passes for
        // var/std (mean, then sum of squared deviations). v1: a single 1-D DOUBLE
        // array var; the result a scalar.
        if (!isArrayVar(name) && !bridge_ && rhs.type == NodeType::CALL
            && rhs.children.size() == 2 && rhs.children[0]->type == NodeType::IDENTIFIER
            && (rhs.children[0]->strValue == "norm" || rhs.children[0]->strValue == "var"
                || rhs.children[0]->strValue == "std")
            && rhs.children[1]->type == NodeType::IDENTIFIER
            && isArrayVar(rhs.children[1]->strValue)
            && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE) {
            const ArrayInfo    &a   = arrays_.at(rhs.children[1]->strValue);
            const std::string  &fn  = rhs.children[0]->strValue;
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (!a.is2D && !a.isND && res.type.isConcrete() && res.type.shape.isScalar()) {
                line("{");
                ++indent_;
                line("const std::size_t _numkit_n = " + a.lenVar + ";");
                if (fn == "norm") {
                    line("double _numkit_ss = 0.0;");
                    open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_n; ++_numkit_i)");
                    line("_numkit_ss += " + a.dataExpr + "[_numkit_i] * " + a.dataExpr + "[_numkit_i];");
                    close();
                    line(cppIdent(name) + " = std::sqrt(_numkit_ss);");
                } else {
                    line("double _numkit_m = 0.0;");
                    open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_n; ++_numkit_i)");
                    line("_numkit_m += " + a.dataExpr + "[_numkit_i];");
                    close();
                    line("_numkit_m = _numkit_n ? _numkit_m / static_cast<double>(_numkit_n) : 0.0;");
                    line("double _numkit_ss = 0.0;");
                    open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_n; ++_numkit_i)");
                    line("const double _numkit_d = " + a.dataExpr + "[_numkit_i] - _numkit_m;");
                    line("_numkit_ss += _numkit_d * _numkit_d;");
                    close();
                    line("double _numkit_var;");
                    line("if (_numkit_n > 1) _numkit_var = _numkit_ss / static_cast<double>(_numkit_n - 1);");
                    line("else _numkit_var = _numkit_n == 1 ? 0.0 "
                         ": std::numeric_limits<double>::quiet_NaN();");
                    line(cppIdent(name) + (fn == "std" ? " = std::sqrt(_numkit_var);" : " = _numkit_var;"));
                }
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native trapz(y) -> trapezoidal integral with unit spacing, a DOUBLE scalar:
        // sum over i of (y[i]+y[i+1])/2. n<2 -> 0 (no interval). Gated !bridge_
        // (order-dependent summation -> the bridged path stays exact under the
        // bridge). v1: a single 1-D DOUBLE array var.
        if (!isArrayVar(name) && !bridge_ && rhs.type == NodeType::CALL
            && rhs.children.size() == 2 && rhs.children[0]->type == NodeType::IDENTIFIER
            && rhs.children[0]->strValue == "trapz"
            && rhs.children[1]->type == NodeType::IDENTIFIER
            && isArrayVar(rhs.children[1]->strValue)
            && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE) {
            const ArrayInfo    &a   = arrays_.at(rhs.children[1]->strValue);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (!a.is2D && !a.isND && res.type.isConcrete() && res.type.shape.isScalar()) {
                line("{");
                ++indent_;
                line("double _numkit_acc = 0.0;");
                open("for (std::size_t _numkit_i = 0; _numkit_i + 1 < " + a.lenVar + "; ++_numkit_i)");
                line("_numkit_acc += (" + a.dataExpr + "[_numkit_i] + " + a.dataExpr
                     + "[_numkit_i + 1]) * 0.5;");
                close();
                line(cppIdent(name) + " = _numkit_acc;");
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native trace(A) -> the sum of the diagonal of a 2-D matrix, a DOUBLE scalar.
        // Column-major: acc += A[i + i*rows] for i in [0, min(rows,cols)). Gated
        // !bridge_ (order-dependent sum). v1: A a 2-D DOUBLE matrix var.
        if (!isArrayVar(name) && !bridge_ && rhs.type == NodeType::CALL
            && rhs.children.size() == 2 && rhs.children[0]->type == NodeType::IDENTIFIER
            && rhs.children[0]->strValue == "trace"
            && rhs.children[1]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[1]->strValue)
            && arrays_.at(rhs.children[1]->strValue).is2D
            && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE) {
            const ArrayInfo    &A   = arrays_.at(rhs.children[1]->strValue);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete() && res.type.shape.isScalar()) {
                line("{");
                ++indent_;
                line("const std::size_t _numkit_d = " + A.rowsVar + " < " + A.colsVar + " ? " + A.rowsVar
                     + " : " + A.colsVar + ";");
                line("double _numkit_acc = 0.0;");
                open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_d; ++_numkit_i)");
                line("_numkit_acc += " + A.dataExpr + "[_numkit_i + _numkit_i * " + A.rowsVar + "];");
                close();
                line(cppIdent(name) + " = _numkit_acc;");
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native median(x) -> the middle of a sorted copy (no-bridge tier, !bridge_).
        // Sort a temp copy with the NaN-last comparator (a valid strict-weak-ordering).
        // MATLAB's default median propagates NaN (any NaN -> NaN): a NaN sorts last,
        // so a NaN present <=> the last post-sort element is NaN. empty -> NaN; n odd
        // -> tmp[n/2]; n even -> (tmp[n/2-1]+tmp[n/2])/2. v1: a 1-D DOUBLE array var.
        if (!isArrayVar(name) && !bridge_ && rhs.type == NodeType::CALL
            && rhs.children.size() == 2 && rhs.children[0]->type == NodeType::IDENTIFIER
            && rhs.children[0]->strValue == "median"
            && rhs.children[1]->type == NodeType::IDENTIFIER
            && isArrayVar(rhs.children[1]->strValue)
            && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE) {
            const ArrayInfo    &a   = arrays_.at(rhs.children[1]->strValue);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (!a.is2D && !a.isND && res.type.isConcrete() && res.type.shape.isScalar()) {
                line("{");
                ++indent_;
                line("std::vector<double> _numkit_t(" + a.dataExpr + ", " + a.dataExpr + " + "
                     + a.lenVar + ");");
                line("const std::size_t _numkit_n = _numkit_t.size();");
                line("std::sort(_numkit_t.begin(), _numkit_t.end(),");
                line("    [](double _a, double _b){ return _a < _b || (_b != _b && _a == _a); });");
                line("if (_numkit_n == 0 || _numkit_t[_numkit_n - 1] != _numkit_t[_numkit_n - 1])");
                line("    " + cppIdent(name) + " = std::numeric_limits<double>::quiet_NaN();");
                line("else if (_numkit_n % 2 == 1) " + cppIdent(name) + " = _numkit_t[_numkit_n / 2];");
                line("else " + cppIdent(name) + " = (_numkit_t[_numkit_n / 2 - 1] + _numkit_t[_numkit_n / 2]) / 2.0;");
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native median(A) on a 2-D MATRIX -> column-wise median -> a 1 x n ROW vector (a 1-D
        // LOCAL). Per column: copy the contiguous column block, sort (NaN-last), then the middle
        // (odd m) or the mean of the two middles (even m); any NaN in a column -> NaN (the sorted
        // column's last element is NaN). Mirrors the 1-D median, per column. !bridge_ guard. v1: a
        // DOUBLE matrix var; result a 1-D array LOCAL.
        if (isArrayVar(name) && arrays_.at(name).isLocal && !bridge_ && !arrays_.at(name).is2D
            && !arrays_.at(name).isND && rhs.type == NodeType::CALL && rhs.children.size() == 2
            && rhs.children[0]->type == NodeType::IDENTIFIER && rhs.children[0]->strValue == "median"
            && rhs.children[1]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[1]->strValue)
            && (arrays_.at(rhs.children[1]->strValue).is2D
                || (arrays_.at(rhs.children[1]->strValue).isND
                    && arrays_.at(rhs.children[1]->strValue).ndDims.size() == 2))
            && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE) {
            const ArrayInfo    &A   = arrays_.at(rhs.children[1]->strValue);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete() && !res.type.shape.isScalar()) {
                line("{");
                ++indent_;
                line("const std::size_t _numkit_m = " + dimExpr(A, 0) + ";");
                line("const std::size_t _numkit_n = " + dimExpr(A, 1) + ";");
                line("const double* _numkit_da = " + A.dataExpr + ";");
                line(cppIdent(name) + ".assign(_numkit_n, 0.0);");
                open("for (std::size_t _numkit_j = 0; _numkit_j < _numkit_n; ++_numkit_j)");
                line("std::vector<double> _numkit_t(_numkit_da + _numkit_j * _numkit_m, _numkit_da + (_numkit_j + 1) * "
                     "_numkit_m);");
                line("std::sort(_numkit_t.begin(), _numkit_t.end(),");
                line("    [](double _a, double _b){ return _a < _b || (_b != _b && _a == _a); });");
                line("if (_numkit_m == 0 || _numkit_t[_numkit_m - 1] != _numkit_t[_numkit_m - 1]) " + cppIdent(name)
                     + "[_numkit_j] = std::numeric_limits<double>::quiet_NaN();");
                line("else if (_numkit_m % 2 == 1) " + cppIdent(name) + "[_numkit_j] = _numkit_t[_numkit_m / 2];");
                line("else " + cppIdent(name) + "[_numkit_j] = (_numkit_t[_numkit_m / 2 - 1] + _numkit_t[_numkit_m / 2]) / 2.0;");
                close();
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native mode(x) on a 1-D vector -> a scalar: the most frequent value, MATLAB tie -> the
        // SMALLEST (sort ascending NaN-last, then the first run whose length is STRICTLY greater
        // wins, so equal-length ties keep the smaller earlier value). NaN is IGNORED (the sorted NaN
        // tail is skipped via the v==v gate); all-NaN / empty -> NaN. !bridge_ guard. v1: a 1-D
        // DOUBLE array var; scalar result.
        if (!isArrayVar(name) && !bridge_ && rhs.type == NodeType::CALL && rhs.children.size() == 2
            && rhs.children[0]->type == NodeType::IDENTIFIER && rhs.children[0]->strValue == "mode"
            && rhs.children[1]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[1]->strValue)
            && !arrays_.at(rhs.children[1]->strValue).is2D
            && !arrays_.at(rhs.children[1]->strValue).isND
            && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE) {
            const ArrayInfo    &a   = arrays_.at(rhs.children[1]->strValue);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete() && res.type.shape.isScalar()) {
                line("{");
                ++indent_;
                line("std::vector<double> _numkit_t(" + a.dataExpr + ", " + a.dataExpr + " + "
                     + a.lenVar + ");");
                line("std::sort(_numkit_t.begin(), _numkit_t.end(),");
                line("    [](double _a, double _b){ return _a < _b || (_b != _b && _a == _a); });");
                line("double _numkit_bv = std::numeric_limits<double>::quiet_NaN();");
                line("std::size_t _numkit_bc = 0, _numkit_i = 0;");
                open("while (_numkit_i < _numkit_t.size() && _numkit_t[_numkit_i] == _numkit_t[_numkit_i])");
                line("std::size_t _numkit_j = _numkit_i;");
                line("while (_numkit_j < _numkit_t.size() && _numkit_t[_numkit_j] == _numkit_t[_numkit_i]) ++_numkit_j;");
                line("if (_numkit_j - _numkit_i > _numkit_bc) { _numkit_bc = _numkit_j - _numkit_i; _numkit_bv = _numkit_t[_numkit_i]; }");
                line("_numkit_i = _numkit_j;");
                close();
                line(cppIdent(name) + " = _numkit_bv;");
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native mode(A) on a 2-D MATRIX -> column-wise mode -> a 1 x n ROW vector (a 1-D LOCAL).
        // Per column: copy the contiguous block, sort (NaN-last), then the most-frequent value (tie
        // -> smallest, NaN ignored). Mirrors the 1-D mode, per column. !bridge_ guard. v1: a DOUBLE
        // matrix var; result a 1-D array LOCAL.
        if (isArrayVar(name) && arrays_.at(name).isLocal && !bridge_ && !arrays_.at(name).is2D
            && !arrays_.at(name).isND && rhs.type == NodeType::CALL && rhs.children.size() == 2
            && rhs.children[0]->type == NodeType::IDENTIFIER && rhs.children[0]->strValue == "mode"
            && rhs.children[1]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[1]->strValue)
            && (arrays_.at(rhs.children[1]->strValue).is2D
                || (arrays_.at(rhs.children[1]->strValue).isND
                    && arrays_.at(rhs.children[1]->strValue).ndDims.size() == 2))
            && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE) {
            const ArrayInfo    &A   = arrays_.at(rhs.children[1]->strValue);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete() && !res.type.shape.isScalar()) {
                line("{");
                ++indent_;
                line("const std::size_t _numkit_m = " + dimExpr(A, 0) + ";");
                line("const std::size_t _numkit_n = " + dimExpr(A, 1) + ";");
                line("const double* _numkit_da = " + A.dataExpr + ";");
                line(cppIdent(name) + ".assign(_numkit_n, 0.0);");
                open("for (std::size_t _numkit_jj = 0; _numkit_jj < _numkit_n; ++_numkit_jj)");
                line("std::vector<double> _numkit_t(_numkit_da + _numkit_jj * _numkit_m, _numkit_da + (_numkit_jj + 1) * "
                     "_numkit_m);");
                line("std::sort(_numkit_t.begin(), _numkit_t.end(),");
                line("    [](double _a, double _b){ return _a < _b || (_b != _b && _a == _a); });");
                line("double _numkit_bv = std::numeric_limits<double>::quiet_NaN();");
                line("std::size_t _numkit_bc = 0, _numkit_i = 0;");
                open("while (_numkit_i < _numkit_t.size() && _numkit_t[_numkit_i] == _numkit_t[_numkit_i])");
                line("std::size_t _numkit_j = _numkit_i;");
                line("while (_numkit_j < _numkit_t.size() && _numkit_t[_numkit_j] == _numkit_t[_numkit_i]) ++_numkit_j;");
                line("if (_numkit_j - _numkit_i > _numkit_bc) { _numkit_bc = _numkit_j - _numkit_i; _numkit_bv = _numkit_t[_numkit_i]; }");
                line("_numkit_i = _numkit_j;");
                close();
                line(cppIdent(name) + "[_numkit_jj] = _numkit_bv;");
                close();
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native dot(a,b) -> sum of a[i]*b[i] (real inner product), but ONLY with
        // no bridge (order-dependent like sum -> the bridged path stays exact when
        // the bridge is on). A length-match guard then an accumulation loop. v1:
        // two 1-D DOUBLE array vars. (Mirrors the inner-product `s = r*c` path for
        // the dot() builtin form, which works for any vector orientation.)
        if (!isArrayVar(name) && !bridge_ && rhs.type == NodeType::CALL
            && rhs.children.size() == 3 && rhs.children[0]->type == NodeType::IDENTIFIER
            && rhs.children[0]->strValue == "dot"
            && rhs.children[1]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[1]->strValue)
            && rhs.children[2]->type == NodeType::IDENTIFIER
            && isArrayVar(rhs.children[2]->strValue)) {
            const ArrayInfo    &A   = arrays_.at(rhs.children[1]->strValue);
            const ArrayInfo    &B   = arrays_.at(rhs.children[2]->strValue);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (!A.is2D && !A.isND && !B.is2D && !B.isND && A.dtype == ValueType::DOUBLE
                && B.dtype == ValueType::DOUBLE && res.type.isConcrete()
                && res.type.shape.isScalar()) {
                line("if (" + A.lenVar + " != " + B.lenVar
                     + ") throw std::out_of_range(\"numkit: dot inputs must be the same length\");");
                line("{");
                ++indent_;
                line("double _numkit_acc = 0.0;");
                open("for (std::size_t _numkit_i = 0; _numkit_i < " + A.lenVar + "; ++_numkit_i)");
                line("_numkit_acc += " + A.dataExpr + "[_numkit_i] * " + B.dataExpr + "[_numkit_i];");
                close();
                line(cppIdent(name) + " = _numkit_acc;");
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native strcmp(a,b) -> LOGICAL scalar: true iff the two arrays are equal
        // (same length AND elementwise equal). Self-contained; works for char
        // (uint16) and numeric arrays. v1: two 1-D array vars, result assigned.
        if (!isArrayVar(name) && rhs.type == NodeType::CALL && rhs.children.size() == 3
            && rhs.children[0]->type == NodeType::IDENTIFIER
            && rhs.children[0]->strValue == "strcmp"
            && rhs.children[1]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[1]->strValue)
            && rhs.children[2]->type == NodeType::IDENTIFIER
            && isArrayVar(rhs.children[2]->strValue)) {
            const ArrayInfo    &A   = arrays_.at(rhs.children[1]->strValue);
            const ArrayInfo    &B   = arrays_.at(rhs.children[2]->strValue);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (!A.is2D && !A.isND && !B.is2D && !B.isND && res.type.isConcrete()
                && res.type.shape.isScalar()) {
                line("{");
                ++indent_;
                line("bool _numkit_eq = (" + A.lenVar + " == " + B.lenVar + ");");
                open("if (_numkit_eq)");
                open("for (std::size_t _numkit_i = 0; _numkit_i < " + A.lenVar + "; ++_numkit_i)");
                line("if (" + A.dataExpr + "[_numkit_i] != " + B.dataExpr
                     + "[_numkit_i]) { _numkit_eq = false; break; }");
                close();
                close();
                line(cppIdent(name) + " = _numkit_eq;");
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Bridged array reduction -> scalar (opt-in): s = sum(x) / prod / mean /
        // max / min. The array arg can't be a scalar C++ expression, so box it
        // (like the bridged array-RESULT path) and call bridge_scalar_arr —
        // exact runtime result (summation order, NaN). Sound only when inference
        // proves a real (non-complex) scalar result AND an arg is an array (a
        // pure-scalar call uses the bridge_scalar path in emitBuiltinCall).
        if (!isArrayVar(name) && bridge_ && rhs.type == NodeType::CALL && !rhs.children.empty()
            && rhs.children[0]->type == NodeType::IDENTIFIER) {
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            bool                anyArrayArg = false;
            for (std::size_t i = 1; i < rhs.children.size(); ++i)
                if (rhs.children[i]->type == NodeType::IDENTIFIER
                    && isArrayVar(rhs.children[i]->strValue))
                    anyArrayArg = true;
            if (anyArrayArg && res.type.isUnboxableScalar()
                && res.type.dtype != ValueType::COMPLEX) {
                const std::string callee = rhs.children[0]->strValue;
                const std::size_t nargs  = rhs.children.size() - 1;
                std::string       boxed;
                for (std::size_t i = 1; i < rhs.children.size(); ++i) {
                    const ASTNode &arg = *rhs.children[i];
                    if (i > 1) boxed += ", ";
                    if (arg.type == NodeType::IDENTIFIER && isArrayVar(arg.strValue)) {
                        const ArrayInfo &aai = arrays_.at(arg.strValue);
                        if (aai.is2D || aai.isND)
                            unsupported("bridged reduction: 2-D/N-D array argument (v1)");
                        if (aai.dtype == ValueType::COMPLEX)
                            unsupported("bridged reduction: complex array argument (v1)");
                        boxed += "numkit_box_array(" + aai.dataExpr + ", " + aai.lenVar + ")";
                    } else {
                        if (inferExpr(arg, types_, reg_, classes_).type.dtype == ValueType::COMPLEX)
                            unsupported("bridged reduction: complex scalar argument (v1)");
                        boxed += "numkit_box_scalar(" + emitExpr(arg) + ")";
                    }
                }
                open("");  // fresh scope for the temporary arg array
                line("numkit_val _numkit_args[] = { " + boxed + " };");
                line(cppIdent(name) + " = numkit_rt::bridge_scalar_arr(\"" + callee + "\", _numkit_args, "
                     + std::to_string(nargs) + ");");
                close();
                types_.set(name, res);
                return;
            }
        }
        // Native find(m) -> the 1-based LINEAR (column-major) positions of the nonzero (true)
        // elements, a runtime-sized 1-D DOUBLE column vector (MATLAB find). A native filter
        // loop over the flat buffer pushing (i+1); EXACT (deterministic positions), preferred
        // over the bridged array-result path below. Works for a 1-D vector OR a 2-D / N-D
        // matrix arg (the flat index +1 is the MATLAB linear index) -> bound on NUMEL. v1: a
        // single array VARIABLE (double or logical); the result a 1-D array LOCAL (push_back).
        if (isArrayVar(name) && arrays_.at(name).isLocal && !arrays_.at(name).is2D
            && !arrays_.at(name).isND && rhs.type == NodeType::CALL
            && rhs.children.size() == 2 && rhs.children[0]->type == NodeType::IDENTIFIER
            && rhs.children[0]->strValue == "find"
            && rhs.children[1]->type == NodeType::IDENTIFIER
            && isArrayVar(rhs.children[1]->strValue)) {
            const ArrayInfo    &m   = arrays_.at(rhs.children[1]->strValue);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete() && !res.type.shape.isScalar()) {
                std::string numel;  // flat element count (1-D len / 2-D rows*cols / N-D prod)
                if (m.isND) {
                    numel = m.ndDims[0];
                    for (std::size_t i = 1; i < m.ndDims.size(); ++i) numel += " * " + m.ndDims[i];
                } else if (m.is2D) {
                    numel = m.rowsVar + " * " + m.colsVar;
                } else {
                    numel = m.lenVar;
                }
                line("{");
                ++indent_;
                line(cppIdent(name) + ".clear();");
                open("for (std::size_t _numkit_i = 0; _numkit_i < (" + numel + "); ++_numkit_i)");
                line("if (" + m.dataExpr + "[_numkit_i]) " + cppIdent(name)
                     + ".push_back(static_cast<double>(_numkit_i + 1));");
                close();
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native find(<expr>) with an INLINE elementwise expression -> the 1-based positions
        // where the per-element value is nonzero/true: find(x>0), find(A>lo & A<hi), find(x-3)
        // (x!=3), ... The inline sibling of find(VAR): the expression is FUSED into the filter
        // loop (no temp) -- elementCtx_ makes the whole array emit arr[_numkit_i] (flat), then for
        // each flat element i, if the value is nonzero push (i+1). For a 2-D/N-D operand the
        // flat index +1 IS MATLAB's column-major LINEAR index. Restricted to a SINGLE-array
        // pure-elementwise expression (collectElementwise srcArrays of size 1) so the bound is
        // that array's NUMEL and per-element emission is valid (a multi-array find(x>y) ->
        // bridged). Result a 1-D DOUBLE LOCAL (push_back). v1: 1-D or 2-D/N-D operand.
        if (isArrayVar(name) && arrays_.at(name).isLocal && !arrays_.at(name).is2D
            && !arrays_.at(name).isND && rhs.type == NodeType::CALL
            && rhs.children.size() == 2 && rhs.children[0]->type == NodeType::IDENTIFIER
            && rhs.children[0]->strValue == "find"
            && rhs.children[1]->type != NodeType::IDENTIFIER) {
            std::set<std::string> srcArrays;
            const bool            pureEw = collectElementwise(*rhs.children[1], srcArrays);
            const AbstractValue   maskAV = inferExpr(*rhs.children[1], types_, reg_, classes_);
            const AbstractValue   res    = inferExpr(rhs, types_, reg_, classes_);
            if (pureEw && srcArrays.size() == 1 && maskAV.type.isConcrete()
                && !maskAV.type.shape.isScalar()
                && res.type.isConcrete() && !res.type.shape.isScalar()) {
                const ArrayInfo &ba = arrays_.at(*srcArrays.begin());
                std::string      numel;  // flat element count (1-D len / 2-D rows*cols / N-D prod)
                if (ba.isND) {
                    numel = ba.ndDims[0];
                    for (std::size_t i = 1; i < ba.ndDims.size(); ++i) numel += " * " + ba.ndDims[i];
                } else if (ba.is2D) {
                    numel = ba.rowsVar + " * " + ba.colsVar;
                } else {
                    numel = ba.lenVar;
                }
                line("{");
                ++indent_;
                line(cppIdent(name) + ".clear();");
                elementCtx_ = "_numkit_i";  // whole array in the expr -> arr[_numkit_i] (flat)
                const std::string maskExpr = emitExpr(*rhs.children[1]);
                elementCtx_.clear();
                open("for (std::size_t _numkit_i = 0; _numkit_i < (" + numel + "); ++_numkit_i)");
                line("if (" + maskExpr + ") " + cppIdent(name)
                     + ".push_back(static_cast<double>(_numkit_i + 1));");
                close();
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native any/all on a 2-D MATRIX -> a LOGICAL 1-D vector (uint8 buffer). any(A)/all(A) and
        // any(A,1)/all(A,1) are column-wise (-> 1 x n); any(A,2)/all(A,2) are row-wise (-> m x 1).
        // result = whether the reduced line has ANY nonzero (any) / is ALL nonzero (all). NaN counts
        // as nonzero (NaN != 0), matching MATLAB. EXACT -> every tier. Empty line -> any false / all
        // true (the seeds). v1: a DOUBLE/LOGICAL matrix var; dim a literal 1|2.
        {
            const bool anyTwoArg =
                rhs.type == NodeType::CALL && rhs.children.size() == 3
                && rhs.children[0]->type == NodeType::IDENTIFIER
                && (rhs.children[0]->strValue == "any" || rhs.children[0]->strValue == "all")
                && rhs.children[2]->type == NodeType::NUMBER_LITERAL
                && (rhs.children[2]->numValue == 1.0 || rhs.children[2]->numValue == 2.0);
            if (isArrayVar(name) && arrays_.at(name).isLocal && !arrays_.at(name).is2D
                && !arrays_.at(name).isND && rhs.type == NodeType::CALL
                && (rhs.children.size() == 2 || anyTwoArg)
                && rhs.children[0]->type == NodeType::IDENTIFIER
                && (rhs.children[0]->strValue == "any" || rhs.children[0]->strValue == "all")
                && rhs.children[1]->type == NodeType::IDENTIFIER
                && isArrayVar(rhs.children[1]->strValue)
                && (arrays_.at(rhs.children[1]->strValue).is2D
                    || (arrays_.at(rhs.children[1]->strValue).isND
                        && arrays_.at(rhs.children[1]->strValue).ndDims.size() == 2))
                && (arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE
                    || arrays_.at(rhs.children[1]->strValue).dtype == ValueType::LOGICAL)) {
                const ArrayInfo    &A   = arrays_.at(rhs.children[1]->strValue);
                const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
                if (res.type.isConcrete() && !res.type.shape.isScalar()) {
                    const bool        isAny = rhs.children[0]->strValue == "any";
                    const bool        dim2  = anyTwoArg && rhs.children[2]->numValue == 2.0;
                    const std::string comb  = isAny ? "|| _numkit_v != 0.0;" : "&& _numkit_v != 0.0;";
                    line("{");
                    ++indent_;
                    line("const std::size_t _numkit_m = " + dimExpr(A, 0) + ";");
                    line("const std::size_t _numkit_n = " + dimExpr(A, 1) + ";");
                    if (dim2) {  // row-wise: result length m, reduce across columns
                        line(cppIdent(name) + ".assign(_numkit_m, 0);");
                        open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_m; ++_numkit_i)");
                        line(std::string("bool _numkit_acc = ") + (isAny ? "false;" : "true;"));
                        open("for (std::size_t _numkit_j = 0; _numkit_j < _numkit_n; ++_numkit_j)");
                        line("const double _numkit_v = static_cast<double>(" + A.dataExpr
                             + "[_numkit_i + _numkit_j * _numkit_m]);");
                        line("_numkit_acc = _numkit_acc " + comb);
                        close();
                        line(cppIdent(name) + "[_numkit_i] = _numkit_acc;");
                        close();
                    } else {  // dim 1 column-wise: result length n
                        line(cppIdent(name) + ".assign(_numkit_n, 0);");
                        open("for (std::size_t _numkit_j = 0; _numkit_j < _numkit_n; ++_numkit_j)");
                        line(std::string("bool _numkit_acc = ") + (isAny ? "false;" : "true;"));
                        open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_m; ++_numkit_i)");
                        line("const double _numkit_v = static_cast<double>(" + A.dataExpr
                             + "[_numkit_i + _numkit_j * _numkit_m]);");
                        line("_numkit_acc = _numkit_acc " + comb);
                        close();
                        line(cppIdent(name) + "[_numkit_j] = _numkit_acc;");
                        close();
                    }
                    --indent_;
                    line("}");
                    types_.set(name, res);
                    return;
                }
            }
        }
        // Native any(A,dim) / all(A,dim) reduction along a LITERAL dim 1|2|3 of a rank-3 array -> a
        // LOGICAL result with the dim collapsed: dim 1 -> [1,n,p], dim 2 -> [m,1,p] (rank-3, the
        // singleton is non-trailing, kept), dim 3 -> [m,n] (rank-2, the trailing dim is fully
        // removed). result = the reduced fiber has ANY nonzero (any) / is ALL nonzero (all); NaN
        // counts as nonzero (NaN != 0), per MATLAB. Empty fiber -> any false / all true (the seeds).
        // EXACT, every tier. Reuses the cumsum stride scheme: each fiber (stride sd, length ld)
        // reduces to out[b/ld + r]. v1: a DOUBLE/LOGICAL rank-3 var.
        if (isArrayVar(name) && arrays_.at(name).isLocal && arrays_.at(name).ndRuntimeLocal
            && (arrays_.at(name).ndDims.size() == 2 || arrays_.at(name).ndDims.size() == 3)
            && rhs.type == NodeType::CALL && rhs.children.size() == 3
            && rhs.children[0]->type == NodeType::IDENTIFIER
            && (rhs.children[0]->strValue == "any" || rhs.children[0]->strValue == "all")
            && rhs.children[2]->type == NodeType::NUMBER_LITERAL
            && (rhs.children[2]->numValue == 1.0 || rhs.children[2]->numValue == 2.0
                || rhs.children[2]->numValue == 3.0)
            && rhs.children[1]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[1]->strValue)
            && arrays_.at(rhs.children[1]->strValue).isND
            && arrays_.at(rhs.children[1]->strValue).ndDims.size() == 3
            && (arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE
                || arrays_.at(rhs.children[1]->strValue).dtype == ValueType::LOGICAL)) {
            const int dimv = static_cast<int>(rhs.children[2]->numValue);
            // the dst rank must match the dim: dim 3 -> rank 2, dim 1|2 -> rank 3.
            if ((dimv == 3) == (arrays_.at(name).ndDims.size() == 2)) {
                const ArrayInfo    &A   = arrays_.at(rhs.children[1]->strValue);
                const ArrayInfo    &M   = arrays_.at(name);
                const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
                if (res.type.isConcrete() && !res.type.shape.isScalar()) {
                    const bool        isAny = rhs.children[0]->strValue == "any";
                    const std::string comb  = isAny ? "|| _numkit_v != 0.0;" : "&& _numkit_v != 0.0;";
                    const std::string sd = dimv == 1 ? "1" : (dimv == 2 ? "_numkit_m" : "_numkit_m * _numkit_n");
                    const std::string ld = dimv == 1 ? "_numkit_m" : (dimv == 2 ? "_numkit_n" : "_numkit_p");
                    line("{");
                    ++indent_;
                    line("const std::size_t _numkit_m = " + dimExpr(A, 0) + ";");
                    line("const std::size_t _numkit_n = " + dimExpr(A, 1) + ";");
                    line("const std::size_t _numkit_p = " + dimExpr(A, 2) + ";");
                    line("const std::size_t _numkit_tot = _numkit_m * _numkit_n * _numkit_p;");
                    line("const std::size_t _numkit_sd = " + sd + ";");
                    line("const std::size_t _numkit_ld = " + ld + ";");
                    line("const std::size_t _numkit_blk = _numkit_sd * _numkit_ld;");
                    if (dimv == 1) {
                        line(M.ndDims[0] + " = 1;");
                        line(M.ndDims[1] + " = _numkit_n;");
                        line(M.ndDims[2] + " = _numkit_p;");
                    } else if (dimv == 2) {
                        line(M.ndDims[0] + " = _numkit_m;");
                        line(M.ndDims[1] + " = 1;");
                        line(M.ndDims[2] + " = _numkit_p;");
                    } else {  // dim 3 -> rank-2 [m, n]
                        line(M.ndDims[0] + " = _numkit_m;");
                        line(M.ndDims[1] + " = _numkit_n;");
                    }
                    line(cppIdent(name) + ".assign(_numkit_tot / _numkit_ld, 0);");
                    open("for (std::size_t _numkit_b = 0; _numkit_b < _numkit_tot; _numkit_b += _numkit_blk)");
                    open("for (std::size_t _numkit_r = 0; _numkit_r < _numkit_sd; ++_numkit_r)");
                    line(std::string("bool _numkit_acc = ") + (isAny ? "false;" : "true;"));
                    open("for (std::size_t _numkit_t = 0; _numkit_t < _numkit_ld; ++_numkit_t)");
                    line("const double _numkit_v = static_cast<double>(" + A.dataExpr
                         + "[_numkit_b + _numkit_r + _numkit_t * _numkit_sd]);");
                    line("_numkit_acc = _numkit_acc " + comb);
                    close();
                    line(cppIdent(name) + "[_numkit_b / _numkit_ld + _numkit_r] = _numkit_acc;");
                    close();
                    close();
                    --indent_;
                    line("}");
                    types_.set(name, res);
                    return;
                }
            }
        }
        // Native max(A, [], dim) / min(A, [], dim) on a 2-D MATRIX with a LITERAL dim 1|2 -> a 1-D
        // vector. dim 1 reduces each COLUMN (-> 1 x n, same as max(A)); dim 2 reduces each ROW
        // (-> m x 1). NaN-skipping (seed on the reduced line's first element). EXACT, every tier.
        // The middle arg must be an empty matrix []. v1: a DOUBLE matrix var.
        if (isArrayVar(name) && arrays_.at(name).isLocal && !arrays_.at(name).is2D
            && !arrays_.at(name).isND && rhs.type == NodeType::CALL && rhs.children.size() == 4
            && rhs.children[0]->type == NodeType::IDENTIFIER
            && (rhs.children[0]->strValue == "max" || rhs.children[0]->strValue == "min")
            && rhs.children[1]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[1]->strValue)
            && (arrays_.at(rhs.children[1]->strValue).is2D
                || (arrays_.at(rhs.children[1]->strValue).isND
                    && arrays_.at(rhs.children[1]->strValue).ndDims.size() == 2))
            && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE
            && rhs.children[2]->type == NodeType::MATRIX_LITERAL
            && rhs.children[2]->children.empty()  // the [] placeholder
            && rhs.children[3]->type == NodeType::NUMBER_LITERAL
            && (rhs.children[3]->numValue == 1.0 || rhs.children[3]->numValue == 2.0)) {
            const ArrayInfo    &A   = arrays_.at(rhs.children[1]->strValue);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete() && !res.type.shape.isScalar()) {
                const bool        dim2 = rhs.children[3]->numValue == 2.0;
                const std::string cmp  = rhs.children[0]->strValue == "max" ? ">" : "<";
                line("{");
                ++indent_;
                line("const std::size_t _numkit_m = " + dimExpr(A, 0) + ";");
                line("const std::size_t _numkit_n = " + dimExpr(A, 1) + ";");
                if (dim2) {  // row-wise: result length m, reduce across columns
                    line(cppIdent(name) + ".assign(_numkit_m, 0.0);");
                    open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_m; ++_numkit_i)");
                    line("double _numkit_acc = 0.0;");
                    open("for (std::size_t _numkit_j = 0; _numkit_j < _numkit_n; ++_numkit_j)");
                    line("const double _numkit_v = " + A.dataExpr + "[_numkit_i + _numkit_j * _numkit_m];");
                    line("if (_numkit_j == 0 || _numkit_v " + cmp + " _numkit_acc"
                         " || (_numkit_acc != _numkit_acc && _numkit_v == _numkit_v)) _numkit_acc = _numkit_v;");
                    close();
                    line(cppIdent(name) + "[_numkit_i] = _numkit_acc;");
                    close();
                } else {  // dim 1 column-wise: result length n, reduce down rows
                    line(cppIdent(name) + ".assign(_numkit_n, 0.0);");
                    open("for (std::size_t _numkit_j = 0; _numkit_j < _numkit_n; ++_numkit_j)");
                    line("double _numkit_acc = 0.0;");
                    open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_m; ++_numkit_i)");
                    line("const double _numkit_v = " + A.dataExpr + "[_numkit_i + _numkit_j * _numkit_m];");
                    line("if (_numkit_i == 0 || _numkit_v " + cmp + " _numkit_acc"
                         " || (_numkit_acc != _numkit_acc && _numkit_v == _numkit_v)) _numkit_acc = _numkit_v;");
                    close();
                    line(cppIdent(name) + "[_numkit_j] = _numkit_acc;");
                    close();
                }
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native max(A,[],dim) / min(A,[],dim) reduction along a LITERAL dim 1|2|3 of a rank-3 array
        // -> the dim collapses: dim 1 -> [1,n,p], dim 2 -> [m,1,p] (rank-3, the singleton is non-
        // trailing, kept), dim 3 -> [m,n] (rank-2, the trailing dim is fully removed). NaN-skipping
        // per the single-output max/min accumulator (seed on the fiber first element, update on a
        // strict cmp, or acc NaN & cand not). EXACT, every tier. Reuses the cumsum stride scheme:
        // each fiber (stride sd, length ld) reduces to out[b/ld + r]. children: [max, A, [], dim].
        // v1: a DOUBLE rank-3 var.
        if (isArrayVar(name) && arrays_.at(name).isLocal && arrays_.at(name).ndRuntimeLocal
            && (arrays_.at(name).ndDims.size() == 2 || arrays_.at(name).ndDims.size() == 3)
            && rhs.type == NodeType::CALL && rhs.children.size() == 4
            && rhs.children[0]->type == NodeType::IDENTIFIER
            && (rhs.children[0]->strValue == "max" || rhs.children[0]->strValue == "min")
            && rhs.children[1]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[1]->strValue)
            && arrays_.at(rhs.children[1]->strValue).isND
            && arrays_.at(rhs.children[1]->strValue).ndDims.size() == 3
            && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE
            && rhs.children[2]->type == NodeType::MATRIX_LITERAL && rhs.children[2]->children.empty()
            && rhs.children[3]->type == NodeType::NUMBER_LITERAL
            && (rhs.children[3]->numValue == 1.0 || rhs.children[3]->numValue == 2.0
                || rhs.children[3]->numValue == 3.0)) {
            const int dimv = static_cast<int>(rhs.children[3]->numValue);
            // the dst rank must match the dim: dim 3 -> rank 2, dim 1|2 -> rank 3.
            if ((dimv == 3) == (arrays_.at(name).ndDims.size() == 2)) {
                const ArrayInfo    &A   = arrays_.at(rhs.children[1]->strValue);
                const ArrayInfo    &M   = arrays_.at(name);
                const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
                if (res.type.isConcrete() && !res.type.shape.isScalar()) {
                    const std::string cmp = rhs.children[0]->strValue == "max" ? ">" : "<";
                    const std::string sd = dimv == 1 ? "1" : (dimv == 2 ? "_numkit_m" : "_numkit_m * _numkit_n");
                    const std::string ld = dimv == 1 ? "_numkit_m" : (dimv == 2 ? "_numkit_n" : "_numkit_p");
                    line("{");
                    ++indent_;
                    line("const std::size_t _numkit_m = " + dimExpr(A, 0) + ";");
                    line("const std::size_t _numkit_n = " + dimExpr(A, 1) + ";");
                    line("const std::size_t _numkit_p = " + dimExpr(A, 2) + ";");
                    line("const std::size_t _numkit_tot = _numkit_m * _numkit_n * _numkit_p;");
                    line("const std::size_t _numkit_sd = " + sd + ";");
                    line("const std::size_t _numkit_ld = " + ld + ";");
                    line("const std::size_t _numkit_blk = _numkit_sd * _numkit_ld;");
                    if (dimv == 1) {
                        line(M.ndDims[0] + " = 1;");
                        line(M.ndDims[1] + " = _numkit_n;");
                        line(M.ndDims[2] + " = _numkit_p;");
                    } else if (dimv == 2) {
                        line(M.ndDims[0] + " = _numkit_m;");
                        line(M.ndDims[1] + " = 1;");
                        line(M.ndDims[2] + " = _numkit_p;");
                    } else {  // dim 3 -> rank-2 [m, n]
                        line(M.ndDims[0] + " = _numkit_m;");
                        line(M.ndDims[1] + " = _numkit_n;");
                    }
                    line(cppIdent(name) + ".assign(_numkit_tot / _numkit_ld, 0.0);");
                    open("for (std::size_t _numkit_b = 0; _numkit_b < _numkit_tot; _numkit_b += _numkit_blk)");
                    open("for (std::size_t _numkit_r = 0; _numkit_r < _numkit_sd; ++_numkit_r)");
                    line("double _numkit_acc = 0.0;");
                    open("for (std::size_t _numkit_t = 0; _numkit_t < _numkit_ld; ++_numkit_t)");
                    line("const double _numkit_v = " + A.dataExpr + "[_numkit_b + _numkit_r + _numkit_t * _numkit_sd];");
                    line("if (_numkit_t == 0 || _numkit_v " + cmp + " _numkit_acc"
                         " || (_numkit_acc != _numkit_acc && _numkit_v == _numkit_v)) _numkit_acc = _numkit_v;");
                    close();
                    line(cppIdent(name) + "[_numkit_b / _numkit_ld + _numkit_r] = _numkit_acc;");
                    close();
                    close();
                    --indent_;
                    line("}");
                    types_.set(name, res);
                    return;
                }
            }
        }
        // Native max(A) / min(A) on a 2-D MATRIX -> column-wise reduction -> a 1 x n ROW vector
        // (a 1-D LOCAL). result(j) = the max/min over column j, NaN-skipping (seed on the column's
        // first element, update on a strict cmp OR when acc is NaN -> the first non-NaN seeds it;
        // an all-NaN column stays NaN). EXACT + order-independent -> runs in every tier, no bridge
        // guard. Column-major. v1: a single DOUBLE matrix var (KnownDims or runtime rank-2); the
        // result a 1-D array LOCAL.
        if (isArrayVar(name) && arrays_.at(name).isLocal && !arrays_.at(name).is2D
            && !arrays_.at(name).isND && rhs.type == NodeType::CALL && rhs.children.size() == 2
            && rhs.children[0]->type == NodeType::IDENTIFIER
            && (rhs.children[0]->strValue == "max" || rhs.children[0]->strValue == "min")
            && rhs.children[1]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[1]->strValue)
            && (arrays_.at(rhs.children[1]->strValue).is2D
                || (arrays_.at(rhs.children[1]->strValue).isND
                    && arrays_.at(rhs.children[1]->strValue).ndDims.size() == 2))
            && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE) {
            const ArrayInfo    &A   = arrays_.at(rhs.children[1]->strValue);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete() && !res.type.shape.isScalar()) {
                const std::string cmp = rhs.children[0]->strValue == "max" ? ">" : "<";
                line("{");
                ++indent_;
                line("const std::size_t _numkit_m = " + dimExpr(A, 0) + ";");
                line("const std::size_t _numkit_n = " + dimExpr(A, 1) + ";");
                line(cppIdent(name) + ".assign(_numkit_n, 0.0);");
                open("for (std::size_t _numkit_j = 0; _numkit_j < _numkit_n; ++_numkit_j)");
                line("double _numkit_acc = 0.0;");  // seeded on _numkit_i == 0 (safe when _numkit_m == 0)
                open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_m; ++_numkit_i)");
                line("const double _numkit_v = " + A.dataExpr + "[_numkit_i + _numkit_j * _numkit_m];");
                line("if (_numkit_i == 0 || _numkit_v " + cmp + " _numkit_acc"
                     " || (_numkit_acc != _numkit_acc && _numkit_v == _numkit_v)) _numkit_acc = _numkit_v;");
                close();
                line(cppIdent(name) + "[_numkit_j] = _numkit_acc;");
                close();
                --indent_;
                line("}");
                arrays_.at(name).orient = VecOrient::Row;
                types_.set(name, res);
                return;
            }
        }
        // Native sum(A[, dim]) / prod(A[, dim]) / mean(A[, dim]) on a 2-D MATRIX:
        // - dim = 1 (or 1-arg default): column-wise reduction -> 1 x n row vector.
        // - dim = 2: row-wise reduction -> m x 1 column vector.
        if (isArrayVar(name) && (arrays_.at(name).isLocal || arrays_.at(name).isOutput)
            && !arrays_.at(name).is2D && !arrays_.at(name).isND
            && rhs.type == NodeType::CALL && (rhs.children.size() == 2 || rhs.children.size() == 3)
            && rhs.children[0]->type == NodeType::IDENTIFIER
            && (rhs.children[0]->strValue == "sum" || rhs.children[0]->strValue == "prod"
                || rhs.children[0]->strValue == "mean")
            && rhs.children[1]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[1]->strValue)
            && (arrays_.at(rhs.children[1]->strValue).is2D
                || (arrays_.at(rhs.children[1]->strValue).isND
                    && arrays_.at(rhs.children[1]->strValue).ndDims.size() == 2))
            && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE
            && (rhs.children.size() == 2
                || (rhs.children[2]->type == NodeType::NUMBER_LITERAL
                    && (rhs.children[2]->numValue == 1.0 || rhs.children[2]->numValue == 2.0)))) {
            const ArrayInfo    &A   = arrays_.at(rhs.children[1]->strValue);
            const ArrayInfo    &dst = arrays_.at(name);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete() && !res.type.shape.isScalar()) {
                const std::string &fn   = rhs.children[0]->strValue;
                const int          dim  = rhs.children.size() == 3
                                              ? static_cast<int>(rhs.children[2]->numValue)
                                              : 1;
                const std::string  init = fn == "prod" ? "1.0" : "0.0";
                const std::string  op   = fn == "prod" ? "*=" : "+=";
                line("{");
                ++indent_;
                line("const std::size_t _numkit_m = " + dimExpr(A, 0) + ";");
                line("const std::size_t _numkit_n = " + dimExpr(A, 1) + ";");
                if (dim == 1) {
                    if (dst.isLocal)
                        line(cppIdent(name) + ".assign(_numkit_n, 0.0);");
                    else
                        line("if (" + dst.lenVar + " != _numkit_n) throw std::out_of_range(\""
                             "numkit: reduction output length mismatch\");");
                    open("for (std::size_t _numkit_j = 0; _numkit_j < _numkit_n; ++_numkit_j)");
                    line("double _numkit_acc = " + init + ";");
                    open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_m; ++_numkit_i)");
                    line("_numkit_acc " + op + " " + A.dataExpr + "[_numkit_i + _numkit_j * _numkit_m];");
                    close();
                    if (fn == "mean")
                        line(dst.dataExpr + "[_numkit_j] = _numkit_acc / static_cast<double>(_numkit_m);");
                    else
                        line(dst.dataExpr + "[_numkit_j] = _numkit_acc;");
                    close();
                    arrays_.at(name).orient = VecOrient::Row;
                } else {
                    if (dst.isLocal)
                        line(cppIdent(name) + ".assign(_numkit_m, 0.0);");
                    else
                        line("if (" + dst.lenVar + " != _numkit_m) throw std::out_of_range(\""
                             "numkit: reduction output length mismatch\");");
                    open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_m; ++_numkit_i)");
                    line("double _numkit_acc = " + init + ";");
                    open("for (std::size_t _numkit_j = 0; _numkit_j < _numkit_n; ++_numkit_j)");
                    line("_numkit_acc " + op + " " + A.dataExpr + "[_numkit_i + _numkit_j * _numkit_m];");
                    close();
                    if (fn == "mean")
                        line(dst.dataExpr + "[_numkit_i] = _numkit_acc / static_cast<double>(_numkit_n);");
                    else
                        line(dst.dataExpr + "[_numkit_i] = _numkit_acc;");
                    close();
                    arrays_.at(name).orient = VecOrient::Col;
                }
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native diff(A[, 1, dim]) on a 2-D MATRIX -> consecutive differences along dim 1 (down
        // columns -> (m-1) x n) or dim 2 (across rows -> m x (n-1)). EXACT (subtraction), no bridge
        // guard. diff(A) defaults to dim 1 ONLY when rows are statically > 1 (a KnownDims matrix);
        // a runtime-dim 2-D could be 1 x n (first-non-singleton-dim ambiguity) -> there an explicit
        // dim is required via the 3-arg diff(A, 1, dim) form. Column-major. v1: a DOUBLE matrix var
        // distinct from the dest; order 1 only.
        {
            const bool diff3Arg =
                rhs.type == NodeType::CALL && rhs.children.size() == 4
                && rhs.children[0]->type == NodeType::IDENTIFIER && rhs.children[0]->strValue == "diff"
                && rhs.children[2]->type == NodeType::NUMBER_LITERAL
                && rhs.children[2]->numValue == 1.0  // order 1
                && rhs.children[3]->type == NodeType::NUMBER_LITERAL
                && (rhs.children[3]->numValue == 1.0 || rhs.children[3]->numValue == 2.0);
            const bool diff1Arg = rhs.type == NodeType::CALL && rhs.children.size() == 2
                                  && rhs.children[0]->type == NodeType::IDENTIFIER
                                  && rhs.children[0]->strValue == "diff";
            if (isArrayVar(name) && arrays_.at(name).isLocal
                && (arrays_.at(name).is2D
                    || (arrays_.at(name).isND && arrays_.at(name).ndDims.size() == 2))
                && (diff1Arg || diff3Arg) && rhs.children[1]->type == NodeType::IDENTIFIER
                && isArrayVar(rhs.children[1]->strValue) && rhs.children[1]->strValue != name
                && (arrays_.at(rhs.children[1]->strValue).is2D
                    || (arrays_.at(rhs.children[1]->strValue).isND
                        && arrays_.at(rhs.children[1]->strValue).ndDims.size() == 2))
                && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE) {
                const ArrayInfo &A = arrays_.at(rhs.children[1]->strValue);
                // 1-arg default-dim diff needs rows statically > 1 (A.is2D = KnownDims matrix); a
                // runtime-dim 2-D 1-arg diff stays bridged (ambiguous first non-singleton dim).
                if (diff3Arg || A.is2D) {
                    const ArrayInfo    &M   = arrays_.at(name);
                    const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
                    if (res.type.isConcrete() && !res.type.shape.isScalar()) {
                        const bool dim2 = diff3Arg && rhs.children[3]->numValue == 2.0;
                        line("{");
                        ++indent_;
                        line("const std::size_t _numkit_m = " + dimExpr(A, 0) + ";");
                        line("const std::size_t _numkit_n = " + dimExpr(A, 1) + ";");
                        line(std::string("const std::size_t _numkit_rm = ")
                             + (dim2 ? "_numkit_m;" : "_numkit_m - 1;"));  // result rows
                        line(std::string("const std::size_t _numkit_rn = ")
                             + (dim2 ? "_numkit_n - 1;" : "_numkit_n;"));  // result cols
                        if (M.isND && M.ndDims.size() == 2) {
                            line(M.ndDims[0] + " = _numkit_rm;");
                            line(M.ndDims[1] + " = _numkit_rn;");
                        }
                        line(cppIdent(name) + ".assign(_numkit_rm * _numkit_rn, 0.0);");
                        open("for (std::size_t _numkit_j = 0; _numkit_j < _numkit_rn; ++_numkit_j)");
                        open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_rm; ++_numkit_i)");
                        if (dim2)
                            line(cppIdent(name) + "[_numkit_i + _numkit_j * _numkit_rm] = " + A.dataExpr
                                 + "[_numkit_i + (_numkit_j + 1) * _numkit_m] - " + A.dataExpr
                                 + "[_numkit_i + _numkit_j * _numkit_m];");
                        else
                            line(cppIdent(name) + "[_numkit_i + _numkit_j * _numkit_rm] = " + A.dataExpr
                                 + "[(_numkit_i + 1) + _numkit_j * _numkit_m] - " + A.dataExpr
                                 + "[_numkit_i + _numkit_j * _numkit_m];");
                        close();
                        close();
                        --indent_;
                        line("}");
                        types_.set(name, res);
                        return;
                    }
                }
            }
        }
        // Native diff(A, 1, dim) along dim 1 or 2 of a rank-3 array -> consecutive differences with
        // that dim reduced by one (dim 1 -> [m-1, n, p], dim 2 -> [m, n-1, p]). EXACT (subtraction),
        // no bridge guard. The reduced dim is NON-trailing so the rank is preserved (dim 3 is sound-
        // refused: differencing the trailing dim can collapse it to extent 1 = MATLAB rank-2). The
        // in/out shapes differ, so explicit (i,j,k) nests with the two layouts (out strides 1, rm,
        // rm*rn; A strides 1, m, m*n). order 1, dim 1|2 only. v1: a DOUBLE rank-3 var distinct from
        // the dest.
        if (isArrayVar(name) && arrays_.at(name).isLocal && arrays_.at(name).ndRuntimeLocal
            && arrays_.at(name).ndDims.size() == 3 && rhs.type == NodeType::CALL
            && rhs.children.size() == 4 && rhs.children[0]->type == NodeType::IDENTIFIER
            && rhs.children[0]->strValue == "diff"
            && rhs.children[2]->type == NodeType::NUMBER_LITERAL && rhs.children[2]->numValue == 1.0
            && rhs.children[3]->type == NodeType::NUMBER_LITERAL
            && (rhs.children[3]->numValue == 1.0 || rhs.children[3]->numValue == 2.0)
            && rhs.children[1]->type == NodeType::IDENTIFIER
            && isArrayVar(rhs.children[1]->strValue) && rhs.children[1]->strValue != name
            && arrays_.at(rhs.children[1]->strValue).isND
            && arrays_.at(rhs.children[1]->strValue).ndDims.size() == 3
            && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE) {
            const ArrayInfo    &A   = arrays_.at(rhs.children[1]->strValue);
            const ArrayInfo    &M   = arrays_.at(name);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete() && !res.type.shape.isScalar()) {
                const bool dim2 = rhs.children[3]->numValue == 2.0;
                line("{");
                ++indent_;
                line("const std::size_t _numkit_m = " + dimExpr(A, 0) + ";");
                line("const std::size_t _numkit_n = " + dimExpr(A, 1) + ";");
                line("const std::size_t _numkit_p = " + dimExpr(A, 2) + ";");
                line(std::string("const std::size_t _numkit_rm = ") + (dim2 ? "_numkit_m;" : "_numkit_m - 1;"));
                line(std::string("const std::size_t _numkit_rn = ") + (dim2 ? "_numkit_n - 1;" : "_numkit_n;"));
                line("const std::size_t _numkit_rmn = _numkit_rm * _numkit_rn;");
                line(M.ndDims[0] + " = _numkit_rm;");
                line(M.ndDims[1] + " = _numkit_rn;");
                line(M.ndDims[2] + " = _numkit_p;");
                line(cppIdent(name) + ".assign(_numkit_rmn * _numkit_p, 0.0);");
                open("for (std::size_t _numkit_k = 0; _numkit_k < _numkit_p; ++_numkit_k)");
                open("for (std::size_t _numkit_j = 0; _numkit_j < _numkit_rn; ++_numkit_j)");
                open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_rm; ++_numkit_i)");
                line("const std::size_t _numkit_pa = _numkit_k * _numkit_m * _numkit_n;");  // A page offset
                if (dim2)
                    line(cppIdent(name) + "[_numkit_i + _numkit_j * _numkit_rm + _numkit_k * _numkit_rmn] = " + A.dataExpr
                         + "[_numkit_i + (_numkit_j + 1) * _numkit_m + _numkit_pa] - " + A.dataExpr
                         + "[_numkit_i + _numkit_j * _numkit_m + _numkit_pa];");
                else
                    line(cppIdent(name) + "[_numkit_i + _numkit_j * _numkit_rm + _numkit_k * _numkit_rmn] = " + A.dataExpr
                         + "[(_numkit_i + 1) + _numkit_j * _numkit_m + _numkit_pa] - " + A.dataExpr
                         + "[_numkit_i + _numkit_j * _numkit_m + _numkit_pa];");
                close();
                close();
                close();
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native diff(x) -> consecutive differences (y[i] = x[i+1]-x[i], length
        // n-1), a runtime-sized 1-D LOCAL preserving x's dtype. EXACT (subtraction)
        // -> preferred over the bridged array-result path; works for double and
        // complex. v1: a single 1-D array arg that is a VARIABLE; result a LOCAL.
        if (isArrayVar(name) && arrays_.at(name).isLocal && !arrays_.at(name).is2D
            && !arrays_.at(name).isND && rhs.type == NodeType::CALL
            && rhs.children.size() == 2 && rhs.children[0]->type == NodeType::IDENTIFIER
            && rhs.children[0]->strValue == "diff"
            && rhs.children[1]->type == NodeType::IDENTIFIER
            && isArrayVar(rhs.children[1]->strValue)) {
            const ArrayInfo    &xa  = arrays_.at(rhs.children[1]->strValue);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (!xa.is2D && !xa.isND && res.type.isConcrete() && !res.type.shape.isScalar()) {
                line("{");
                ++indent_;
                line(cppIdent(name) + ".clear();");
                open("for (std::size_t _numkit_i = 1; _numkit_i < " + xa.lenVar + "; ++_numkit_i)");
                line(cppIdent(name) + ".push_back(" + xa.dataExpr + "[_numkit_i] - " + xa.dataExpr
                     + "[_numkit_i - 1]);");
                close();
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native cumsum/cumprod(A[, dim]) on a 2-D MATRIX -> a fresh same-shape matrix, running
        // accumulation along dim 1 (down each column, the default) or dim 2 (across each row).
        // Like the 1-D form, ONLY with no bridge (order-dependent rounding -> the bridged path
        // stays the exact tier when the bridge is on). Column-major per-column (dim 1) or per-row
        // (dim 2) prefix scan. 1-arg -> dim 1; 2-arg cumsum(A, dim) with a LITERAL dim 1|2. v1: a
        // DOUBLE matrix var distinct from the dest.
        {
            const bool cumTwoArg =
                rhs.type == NodeType::CALL && rhs.children.size() == 3
                && rhs.children[0]->type == NodeType::IDENTIFIER
                && (rhs.children[0]->strValue == "cumsum" || rhs.children[0]->strValue == "cumprod")
                && rhs.children[2]->type == NodeType::NUMBER_LITERAL
                && (rhs.children[2]->numValue == 1.0 || rhs.children[2]->numValue == 2.0);
            if (isArrayVar(name) && arrays_.at(name).isLocal && !bridge_
                && (arrays_.at(name).is2D
                    || (arrays_.at(name).isND && arrays_.at(name).ndDims.size() == 2))
                && rhs.type == NodeType::CALL && (rhs.children.size() == 2 || cumTwoArg)
                && rhs.children[0]->type == NodeType::IDENTIFIER
                && (rhs.children[0]->strValue == "cumsum" || rhs.children[0]->strValue == "cumprod")
                && rhs.children[1]->type == NodeType::IDENTIFIER
                && isArrayVar(rhs.children[1]->strValue) && rhs.children[1]->strValue != name
                && (arrays_.at(rhs.children[1]->strValue).is2D
                    || (arrays_.at(rhs.children[1]->strValue).isND
                        && arrays_.at(rhs.children[1]->strValue).ndDims.size() == 2))
                && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE) {
                const ArrayInfo    &A   = arrays_.at(rhs.children[1]->strValue);
                const ArrayInfo    &M   = arrays_.at(name);
                const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
                if (res.type.isConcrete() && !res.type.shape.isScalar()) {
                    const bool        isProd = rhs.children[0]->strValue == "cumprod";
                    const bool        dim2   = cumTwoArg && rhs.children[2]->numValue == 2.0;
                    const std::string seed   = isProd ? "1.0" : "0.0";
                    const std::string op     = isProd ? "*=" : "+=";
                    line("{");
                    ++indent_;
                    line("const std::size_t _numkit_m = " + dimExpr(A, 0) + ";");
                    line("const std::size_t _numkit_n = " + dimExpr(A, 1) + ";");
                    if (M.isND && M.ndDims.size() == 2) {  // runtime-dim dst: set its companions
                        line(M.ndDims[0] + " = _numkit_m;");
                        line(M.ndDims[1] + " = _numkit_n;");
                    }
                    line(cppIdent(name) + ".assign(_numkit_m * _numkit_n, 0.0);");
                    if (dim2) {  // dim 2: across each row -> outer over rows, inner over columns
                        open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_m; ++_numkit_i)");
                        line("double _numkit_acc = " + seed + ";");
                        open("for (std::size_t _numkit_j = 0; _numkit_j < _numkit_n; ++_numkit_j)");
                    } else {  // dim 1 (default): down each column -> outer over columns, inner rows
                        open("for (std::size_t _numkit_j = 0; _numkit_j < _numkit_n; ++_numkit_j)");
                        line("double _numkit_acc = " + seed + ";");
                        open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_m; ++_numkit_i)");
                    }
                    line("_numkit_acc " + op + " " + A.dataExpr + "[_numkit_i + _numkit_j * _numkit_m];");
                    line(cppIdent(name) + "[_numkit_i + _numkit_j * _numkit_m] = _numkit_acc;");
                    close();
                    close();
                    --indent_;
                    line("}");
                    types_.set(name, res);
                    return;
                }
            }
        }
        // Native cumsum/cumprod(A, dim) along ANY dim 1|2|3 of a rank-3 array -> a fresh same-shape
        // rank-3 array, running accumulation along each fiber in the scan dim. Like the 1-D/2-D
        // forms, ONLY with no bridge (order-dependent rounding -> the bridged array-result path
        // stays the exact tier when the bridge is on). Column-major (dims m,n,p): a scan along dim d
        // has element stride sd = prod(dims before d) and length ld = size(A,d) -- dim1 sd=1 ld=m
        // (contiguous columns), dim2 sd=m ld=n (per-row within a page), dim3 sd=m*n ld=p (across
        // pages). The fiber starts tile as blocks of sd*ld: idx = b + r + t*sd for block base b
        // (step sd*ld), residue r in [0,sd), position t in [0,ld). Shape-preserving (no rank drop).
        // The 2-arg form with a LITERAL dim 1|2|3 only (the 1-arg default dim is the first non-
        // singleton -- ambiguous on a runtime-dim rank-3 -> stays bridged). v1: a DOUBLE rank-3 var
        // distinct from dst.
        if (isArrayVar(name) && arrays_.at(name).isLocal && arrays_.at(name).ndRuntimeLocal
            && !bridge_ && arrays_.at(name).ndDims.size() == 3 && rhs.type == NodeType::CALL
            && rhs.children.size() == 3 && rhs.children[0]->type == NodeType::IDENTIFIER
            && (rhs.children[0]->strValue == "cumsum" || rhs.children[0]->strValue == "cumprod")
            && rhs.children[2]->type == NodeType::NUMBER_LITERAL
            && (rhs.children[2]->numValue == 1.0 || rhs.children[2]->numValue == 2.0
                || rhs.children[2]->numValue == 3.0)
            && rhs.children[1]->type == NodeType::IDENTIFIER
            && isArrayVar(rhs.children[1]->strValue) && rhs.children[1]->strValue != name
            && arrays_.at(rhs.children[1]->strValue).isND
            && arrays_.at(rhs.children[1]->strValue).ndDims.size() == 3
            && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE) {
            const ArrayInfo    &A   = arrays_.at(rhs.children[1]->strValue);
            const ArrayInfo    &M   = arrays_.at(name);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete() && !res.type.shape.isScalar()) {
                const bool        isProd = rhs.children[0]->strValue == "cumprod";
                const std::string seed   = isProd ? "1.0" : "0.0";
                const std::string op     = isProd ? "*=" : "+=";
                const int         dimv   = static_cast<int>(rhs.children[2]->numValue);
                // scan-dim element stride sd and length ld in terms of m,n,p
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
                open("for (std::size_t _numkit_b = 0; _numkit_b < _numkit_tot; _numkit_b += _numkit_blk)");
                open("for (std::size_t _numkit_r = 0; _numkit_r < _numkit_sd; ++_numkit_r)");
                line("double _numkit_acc = " + seed + ";");
                open("for (std::size_t _numkit_t = 0; _numkit_t < _numkit_ld; ++_numkit_t)");
                line("const std::size_t _numkit_idx = _numkit_b + _numkit_r + _numkit_t * _numkit_sd;");
                line("_numkit_acc " + op + " " + A.dataExpr + "[_numkit_idx];");
                line(cppIdent(name) + "[_numkit_idx] = _numkit_acc;");
                close();
                close();
                close();
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native cumsum/cumprod(x) -> running accumulation, a SAME-LENGTH 1-D
        // result, but ONLY with no bridge (order-dependent rounding -> the bridged
        // array-result path stays the exact tier when the bridge is on). cumsum:
        // acc += (seed 0); cumprod: acc *= (seed 1). v1: a single 1-D DOUBLE array
        // arg that is a VARIABLE; the result a 1-D array LOCAL (push_back).
        if (isArrayVar(name) && arrays_.at(name).isLocal && !bridge_ && !arrays_.at(name).is2D
            && !arrays_.at(name).isND && rhs.type == NodeType::CALL
            && rhs.children.size() == 2 && rhs.children[0]->type == NodeType::IDENTIFIER
            && (rhs.children[0]->strValue == "cumsum" || rhs.children[0]->strValue == "cumprod")
            && rhs.children[1]->type == NodeType::IDENTIFIER
            && isArrayVar(rhs.children[1]->strValue)
            && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE) {
            const ArrayInfo    &xa  = arrays_.at(rhs.children[1]->strValue);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (!xa.is2D && !xa.isND && res.type.isConcrete() && !res.type.shape.isScalar()) {
                const bool isProd = rhs.children[0]->strValue == "cumprod";
                line("{");
                ++indent_;
                line(std::string("double _numkit_acc = ") + (isProd ? "1.0;" : "0.0;"));
                line(cppIdent(name) + ".clear();");
                open("for (std::size_t _numkit_i = 0; _numkit_i < " + xa.lenVar + "; ++_numkit_i)");
                line(std::string("_numkit_acc ") + (isProd ? "*=" : "+=") + " " + xa.dataExpr
                     + "[_numkit_i];");
                line(cppIdent(name) + ".push_back(_numkit_acc);");
                close();
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native cummax/cummin(A[, dim]) on a 2-D MATRIX -> a fresh same-shape matrix, running
        // max/min along dim 1 (down each column, the default) or dim 2 (across each row). EXACT
        // (no rounding), so no bridge guard. Per scan-line the accumulator uses the single-output
        // max/min NaN logic (seed on the line's first element, update on a strict cmp, or when acc
        // is NaN and the candidate is not -> the first non-NaN seeds it). 1-arg -> dim 1; 2-arg
        // cummax(A, dim) with a LITERAL dim 1|2. v1: a DOUBLE matrix var distinct from the dest.
        {
            const bool cmTwoArg =
                rhs.type == NodeType::CALL && rhs.children.size() == 3
                && rhs.children[0]->type == NodeType::IDENTIFIER
                && (rhs.children[0]->strValue == "cummax" || rhs.children[0]->strValue == "cummin")
                && rhs.children[2]->type == NodeType::NUMBER_LITERAL
                && (rhs.children[2]->numValue == 1.0 || rhs.children[2]->numValue == 2.0);
            if (isArrayVar(name) && arrays_.at(name).isLocal
                && (arrays_.at(name).is2D
                    || (arrays_.at(name).isND && arrays_.at(name).ndDims.size() == 2))
                && rhs.type == NodeType::CALL && (rhs.children.size() == 2 || cmTwoArg)
                && rhs.children[0]->type == NodeType::IDENTIFIER
                && (rhs.children[0]->strValue == "cummax" || rhs.children[0]->strValue == "cummin")
                && rhs.children[1]->type == NodeType::IDENTIFIER
                && isArrayVar(rhs.children[1]->strValue) && rhs.children[1]->strValue != name
                && (arrays_.at(rhs.children[1]->strValue).is2D
                    || (arrays_.at(rhs.children[1]->strValue).isND
                        && arrays_.at(rhs.children[1]->strValue).ndDims.size() == 2))
                && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE) {
                const ArrayInfo    &A   = arrays_.at(rhs.children[1]->strValue);
                const ArrayInfo    &M   = arrays_.at(name);
                const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
                if (res.type.isConcrete() && !res.type.shape.isScalar()) {
                    const std::string cmp  = rhs.children[0]->strValue == "cummax" ? ">" : "<";
                    const bool        dim2 = cmTwoArg && rhs.children[2]->numValue == 2.0;
                    line("{");
                    ++indent_;
                    line("const std::size_t _numkit_m = " + dimExpr(A, 0) + ";");
                    line("const std::size_t _numkit_n = " + dimExpr(A, 1) + ";");
                    if (M.isND && M.ndDims.size() == 2) {  // runtime-dim dst: set its companions
                        line(M.ndDims[0] + " = _numkit_m;");
                        line(M.ndDims[1] + " = _numkit_n;");
                    }
                    line(cppIdent(name) + ".assign(_numkit_m * _numkit_n, 0.0);");
                    // `first` is the line's leading element: i==0 for dim 1, j==0 for dim 2.
                    const std::string first = dim2 ? "_numkit_j == 0" : "_numkit_i == 0";
                    if (dim2) {  // dim 2: across each row -> outer over rows, inner over columns
                        open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_m; ++_numkit_i)");
                        line("double _numkit_acc = 0.0;");
                        open("for (std::size_t _numkit_j = 0; _numkit_j < _numkit_n; ++_numkit_j)");
                    } else {  // dim 1 (default): down each column -> outer over columns, inner rows
                        open("for (std::size_t _numkit_j = 0; _numkit_j < _numkit_n; ++_numkit_j)");
                        line("double _numkit_acc = 0.0;");
                        open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_m; ++_numkit_i)");
                    }
                    line("const double _numkit_v = " + A.dataExpr + "[_numkit_i + _numkit_j * _numkit_m];");
                    line("if (" + first + " || _numkit_v " + cmp + " _numkit_acc"
                         " || (_numkit_acc != _numkit_acc && _numkit_v == _numkit_v)) _numkit_acc = _numkit_v;");
                    line(cppIdent(name) + "[_numkit_i + _numkit_j * _numkit_m] = _numkit_acc;");
                    close();
                    close();
                    --indent_;
                    line("}");
                    types_.set(name, res);
                    return;
                }
            }
        }
        // Native cummax/cummin(A, dim) along ANY dim 1|2|3 of a rank-3 array -> a fresh same-shape
        // rank-3 array, running max/min along each fiber in the scan dim. EXACT (no rounding), so no
        // bridge guard. Reuses the cumsum/cumprod stride/length scheme (dim d: element stride sd =
        // prod(dims before d), length ld = size(A,d); fiber starts tile as blocks of sd*ld, idx =
        // b + r + t*sd), with the single-output max/min NaN accumulator (seed on the fiber's first
        // element t==0, update on a strict cmp, or when acc is NaN and the candidate is not -> the
        // first non-NaN seeds it). 2-arg LITERAL dim 1|2|3 only (the 1-arg default dim is the first
        // non-singleton -- ambiguous on a runtime-dim rank-3 -> stays bridged). v1: a DOUBLE rank-3
        // var distinct from dst.
        if (isArrayVar(name) && arrays_.at(name).isLocal && arrays_.at(name).ndRuntimeLocal
            && arrays_.at(name).ndDims.size() == 3 && rhs.type == NodeType::CALL
            && rhs.children.size() == 3 && rhs.children[0]->type == NodeType::IDENTIFIER
            && (rhs.children[0]->strValue == "cummax" || rhs.children[0]->strValue == "cummin")
            && rhs.children[2]->type == NodeType::NUMBER_LITERAL
            && (rhs.children[2]->numValue == 1.0 || rhs.children[2]->numValue == 2.0
                || rhs.children[2]->numValue == 3.0)
            && rhs.children[1]->type == NodeType::IDENTIFIER
            && isArrayVar(rhs.children[1]->strValue) && rhs.children[1]->strValue != name
            && arrays_.at(rhs.children[1]->strValue).isND
            && arrays_.at(rhs.children[1]->strValue).ndDims.size() == 3
            && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE) {
            const ArrayInfo    &A   = arrays_.at(rhs.children[1]->strValue);
            const ArrayInfo    &M   = arrays_.at(name);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete() && !res.type.shape.isScalar()) {
                const std::string cmp  = rhs.children[0]->strValue == "cummax" ? ">" : "<";
                const int         dimv = static_cast<int>(rhs.children[2]->numValue);
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
                open("for (std::size_t _numkit_b = 0; _numkit_b < _numkit_tot; _numkit_b += _numkit_blk)");
                open("for (std::size_t _numkit_r = 0; _numkit_r < _numkit_sd; ++_numkit_r)");
                line("double _numkit_acc = 0.0;");
                open("for (std::size_t _numkit_t = 0; _numkit_t < _numkit_ld; ++_numkit_t)");
                line("const std::size_t _numkit_idx = _numkit_b + _numkit_r + _numkit_t * _numkit_sd;");
                line("const double _numkit_v = " + A.dataExpr + "[_numkit_idx];");
                line("if (_numkit_t == 0 || _numkit_v " + cmp + " _numkit_acc"
                     " || (_numkit_acc != _numkit_acc && _numkit_v == _numkit_v)) _numkit_acc = _numkit_v;");
                line(cppIdent(name) + "[_numkit_idx] = _numkit_acc;");
                close();
                close();
                close();
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native cummax/cummin(x) -> running max/min, a SAME-LENGTH 1-D LOCAL. Exact
        // (no rounding), so it runs in every tier. The running accumulator uses the
        // single-output max/min NaN logic (update on the first element, on a strict
        // cmp, or when acc is NaN and the candidate isn't -> the first non-NaN seeds
        // it), so NaN is ignored mid-stream but a LEADING NaN is preserved (MATLAB
        // cummax([1 NaN 3]) = [1 1 3]; cummax([NaN 3]) = [NaN 3]). v1: a single 1-D
        // DOUBLE array var; the result a 1-D array LOCAL (push_back).
        if (isArrayVar(name) && arrays_.at(name).isLocal && !arrays_.at(name).is2D
            && !arrays_.at(name).isND && rhs.type == NodeType::CALL && rhs.children.size() == 2
            && rhs.children[0]->type == NodeType::IDENTIFIER
            && (rhs.children[0]->strValue == "cummax" || rhs.children[0]->strValue == "cummin")
            && rhs.children[1]->type == NodeType::IDENTIFIER
            && isArrayVar(rhs.children[1]->strValue)
            && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE) {
            const ArrayInfo    &xa  = arrays_.at(rhs.children[1]->strValue);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (!xa.is2D && !xa.isND && res.type.isConcrete() && !res.type.shape.isScalar()) {
                const std::string cmp = rhs.children[0]->strValue == "cummax" ? ">" : "<";
                line("{");
                ++indent_;
                line(cppIdent(name) + ".clear();");
                line("double _numkit_acc = 0.0;");
                open("for (std::size_t _numkit_i = 0; _numkit_i < " + xa.lenVar + "; ++_numkit_i)");
                line("const double _numkit_v = " + xa.dataExpr + "[_numkit_i];");
                line("if (_numkit_i == 0 || _numkit_v " + cmp + " _numkit_acc"
                     " || (_numkit_acc != _numkit_acc && _numkit_v == _numkit_v)) _numkit_acc = _numkit_v;");
                line(cppIdent(name) + ".push_back(_numkit_acc);");
                close();
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native circshift(x, k) -> a circularly-shifted copy, SAME shape as x. A
        // positive k shifts toward higher indices (MATLAB circshift([1 2 3 4],1) =
        // [4 1 2 3]); out[i] = x[((i-k) mod n + n) mod n], which is well-defined for
        // any k (negative / larger than n). dtype-general (a plain element copy).
        // v1: a 1-D array var + a scalar integer shift.
        if (isArrayVar(name) && arrays_.at(name).isLocal && !arrays_.at(name).is2D
            && !arrays_.at(name).isND && rhs.type == NodeType::CALL && rhs.children.size() == 3
            && rhs.children[0]->type == NodeType::IDENTIFIER
            && rhs.children[0]->strValue == "circshift"
            && rhs.children[1]->type == NodeType::IDENTIFIER
            && isArrayVar(rhs.children[1]->strValue)
            && !arrays_.at(rhs.children[1]->strValue).is2D
            && !arrays_.at(rhs.children[1]->strValue).isND) {
            const ArrayInfo    &xa  = arrays_.at(rhs.children[1]->strValue);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete() && !res.type.shape.isScalar()
                && inferExpr(*rhs.children[2], types_, reg_, classes_).type.shape.isScalar()) {
                const std::string k = emitExpr(*rhs.children[2]);
                line("{");
                ++indent_;
                line("const std::ptrdiff_t _numkit_n = static_cast<std::ptrdiff_t>(" + xa.lenVar + ");");
                line("const std::ptrdiff_t _numkit_k = static_cast<std::ptrdiff_t>(" + k + ");");
                line(cppIdent(name) + ".assign(static_cast<std::size_t>(_numkit_n), " + zeroLiteral(xa.dtype)
                     + ");");
                open("for (std::ptrdiff_t _numkit_i = 0; _numkit_i < _numkit_n; ++_numkit_i)");
                line("const std::ptrdiff_t _numkit_s = ((_numkit_i - _numkit_k) % _numkit_n + _numkit_n) % _numkit_n;");
                line(cppIdent(name) + "[static_cast<std::size_t>(_numkit_i)] = " + xa.dataExpr
                     + "[static_cast<std::size_t>(_numkit_s)];");
                close();
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native permute(A, [literal perm]) -> reorder A's dimensions (phase N1, the defining
        // N-D op). B's dim k = A's dim perm[k] (1-based); flat over the column-major buffers --
        // for each output o, decompose into B's multi-index (j_k = (o / Bstride_k) % Bdim_k) and
        // gather A at sum_k j_k * Astride_{perm[k]-1}. perm must be a LITERAL permutation of
        // 1..rank (read here; a runtime perm hits the scalar/refuse path). Runtime dims via
        // dimExpr; B a fresh rank-r ndRuntimeLocal; built into a temp so an in-place
        // A = permute(A, perm) is aliasing-safe. v1: DOUBLE A (NDims or KnownDims-2-D), rank 2-4.
        if (isArrayVar(name) && arrays_.at(name).isLocal && arrays_.at(name).ndRuntimeLocal
            && rhs.type == NodeType::CALL && rhs.children.size() == 3
            && rhs.children[0]->type == NodeType::IDENTIFIER
            && rhs.children[0]->strValue == "permute"
            && rhs.children[1]->type == NodeType::IDENTIFIER
            && isArrayVar(rhs.children[1]->strValue)
            && rhs.children[2]->type == NodeType::MATRIX_LITERAL) {
            const ArrayInfo  &A = arrays_.at(rhs.children[1]->strValue);
            const ArrayInfo  &B = arrays_.at(name);
            const std::size_t r = A.isND ? A.ndDims.size() : (A.is2D ? 2u : 0u);
            const ASTNode    &pn = *rhs.children[2];
            // The literal perm: a single row of r integer literals, a permutation of 1..r.
            std::vector<std::size_t> perm;
            if (r >= 2 && r <= 4 && B.ndDims.size() == r && A.dtype == ValueType::DOUBLE
                && pn.children.size() == 1 && pn.children[0]
                && pn.children[0]->children.size() == r) {
                bool              permOk = true;
                std::vector<bool> seen(r, false);
                for (const auto &el : pn.children[0]->children) {
                    if (!el || el->type != NodeType::NUMBER_LITERAL
                        || el->numValue != std::floor(el->numValue) || el->numValue < 1.0
                        || el->numValue > static_cast<double>(r)) {
                        permOk = false;
                        break;
                    }
                    const std::size_t v = static_cast<std::size_t>(el->numValue);  // 1-based
                    if (seen[v - 1]) {
                        permOk = false;  // a repeated axis -> not a permutation
                        break;
                    }
                    seen[v - 1] = true;
                    perm.push_back(v);
                }
                const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
                if (permOk && perm.size() == r && res.type.isConcrete()
                    && !res.type.shape.isScalar()) {
                    line("{");
                    ++indent_;
                    line("const std::size_t _numkit_As0 = 1;");  // A column-major strides (runtime)
                    for (std::size_t d = 1; d < r; ++d)
                        line("const std::size_t _numkit_As" + std::to_string(d) + " = _numkit_As"
                             + std::to_string(d - 1) + " * (" + dimExpr(A, d - 1) + ");");
                    for (std::size_t k = 0; k < r; ++k)  // B dims = A dims permuted
                        line("const std::size_t _numkit_Bd" + std::to_string(k) + " = ("
                             + dimExpr(A, perm[k] - 1) + ");");
                    line("const std::size_t _numkit_Bs0 = 1;");  // B strides
                    for (std::size_t k = 1; k < r; ++k)
                        line("const std::size_t _numkit_Bs" + std::to_string(k) + " = _numkit_Bs"
                             + std::to_string(k - 1) + " * _numkit_Bd" + std::to_string(k - 1) + ";");
                    std::string numel = "_numkit_Bd0";
                    for (std::size_t k = 1; k < r; ++k) numel += " * _numkit_Bd" + std::to_string(k);
                    for (std::size_t k = 0; k < r; ++k)  // set B's dim companions (permuted)
                        line(B.ndDims[k] + " = _numkit_Bd" + std::to_string(k) + ";");
                    line("std::vector<double> _numkit_out(" + numel + ");");
                    open("for (std::size_t _numkit_o = 0; _numkit_o < (" + numel + "); ++_numkit_o)");
                    line("std::size_t _numkit_af = 0;");
                    for (std::size_t k = 0; k < r; ++k)
                        line("_numkit_af += ((_numkit_o / _numkit_Bs" + std::to_string(k) + ") % _numkit_Bd"
                             + std::to_string(k) + ") * _numkit_As" + std::to_string(perm[k] - 1) + ";");
                    line("_numkit_out[_numkit_o] = " + A.dataExpr + "[_numkit_af];");
                    close();
                    line(cppIdent(name) + ".assign(_numkit_out.begin(), _numkit_out.end());");
                    --indent_;
                    line("}");
                    types_.set(name, res);
                    return;
                }
            }
        }
        // Native circshift(A, k[, dim]) on a 2-D MATRIX with a SCALAR shift, MATLAB semantics.
        // dim 1 (default, rows): each column is circularly shifted, B(i,j)=A(mod(i-k,m),j); the
        // column index passes through. dim 2 (columns): B(i,j)=A(i,mod(j-k,n)); the row index
        // passes through. SAME shape as A. Runtime-dim 2-D operand (dims via dimExpr); a fresh
        // rank-2 ndRuntimeLocal result. dtype-general. Built into a temp first, so an in-place
        // A = circshift(A, k[, dim]) is aliasing-safe. v1: a scalar shift; a LITERAL dim in
        // {1,2} (the transfer keeps a runtime / other dim Dynamic -> bridged); KnownDims-2-D
        // deferred. children: [circshift, A, k] (dim 1) or [circshift, A, k, dim].
        if (isArrayVar(name) && arrays_.at(name).isLocal && arrays_.at(name).ndRuntimeLocal
            && arrays_.at(name).ndDims.size() == 2 && rhs.type == NodeType::CALL
            && (rhs.children.size() == 3
                || (rhs.children.size() == 4 && rhs.children[3]->type == NodeType::NUMBER_LITERAL
                    && (rhs.children[3]->numValue == 1.0 || rhs.children[3]->numValue == 2.0)))
            && rhs.children[0]->type == NodeType::IDENTIFIER
            && rhs.children[0]->strValue == "circshift"
            && rhs.children[1]->type == NodeType::IDENTIFIER
            && isArrayVar(rhs.children[1]->strValue)
            && (arrays_.at(rhs.children[1]->strValue).is2D
                || (arrays_.at(rhs.children[1]->strValue).isND
                    && arrays_.at(rhs.children[1]->strValue).ndDims.size() == 2))) {
            const ArrayInfo    &A   = arrays_.at(rhs.children[1]->strValue);
            const ArrayInfo    &B   = arrays_.at(name);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete() && !res.type.shape.isScalar()
                && inferExpr(*rhs.children[2], types_, reg_, classes_).type.shape.isScalar()) {
                const long dimv =
                    rhs.children.size() == 4 ? static_cast<long>(rhs.children[3]->numValue) : 1;
                const std::string ct = cppScalarType(B.dtype);
                const std::string m = dimExpr(A, 0), n = dimExpr(A, 1);
                const std::string k = emitExpr(*rhs.children[2]);
                line("{");
                ++indent_;
                line("const std::ptrdiff_t _numkit_m = static_cast<std::ptrdiff_t>(" + m + ");");
                line("const std::ptrdiff_t _numkit_n = static_cast<std::ptrdiff_t>(" + n + ");");
                line("const std::ptrdiff_t _numkit_k = static_cast<std::ptrdiff_t>(" + k + ");");
                line("std::vector<" + ct + "> _numkit_out(static_cast<std::size_t>(_numkit_m * _numkit_n));");
                open("for (std::ptrdiff_t _numkit_j = 0; _numkit_j < _numkit_n; ++_numkit_j)");
                open("for (std::ptrdiff_t _numkit_i = 0; _numkit_i < _numkit_m; ++_numkit_i)");
                if (dimv == 1) {
                    line("const std::ptrdiff_t _numkit_ri = ((_numkit_i - _numkit_k) % _numkit_m + _numkit_m) % _numkit_m;");
                    line("_numkit_out[static_cast<std::size_t>(_numkit_i + _numkit_j * _numkit_m)] = " + A.dataExpr
                         + "[static_cast<std::size_t>(_numkit_ri + _numkit_j * _numkit_m)];");
                } else {  // dim 2: wrap the COLUMN index, the row passes through
                    line("const std::ptrdiff_t _numkit_rj = ((_numkit_j - _numkit_k) % _numkit_n + _numkit_n) % _numkit_n;");
                    line("_numkit_out[static_cast<std::size_t>(_numkit_i + _numkit_j * _numkit_m)] = " + A.dataExpr
                         + "[static_cast<std::size_t>(_numkit_i + _numkit_rj * _numkit_m)];");
                }
                close();
                close();
                line(B.ndDims[0] + " = static_cast<std::size_t>(_numkit_m);");
                line(B.ndDims[1] + " = static_cast<std::size_t>(_numkit_n);");
                line(cppIdent(name) + ".assign(_numkit_out.begin(), _numkit_out.end());");
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native circshift(A, k, dim) on a rank-3 array with a SCALAR shift along a LITERAL dim
        // 1|2|3 -> a fresh same-shape rank-3 array, each fiber along the scan dim circularly shifted
        // (a positive k toward higher indices, MATLAB: out[t] = A[mod(t-k, ld)]). Reuses the cumsum
        // stride scheme (dim d: element stride sd, length ld; fiber starts tile as blocks of sd*ld,
        // idx = b + r + t*sd). SAME shape (no rank change). Built into a temp first so an in-place
        // A = circshift(A,k,dim) is aliasing-safe. The explicit 3-arg form only (the 2-arg default
        // dim is the first non-singleton -- ambiguous on a runtime-dim rank-3 -> bridged). v1: a
        // DOUBLE rank-3 array var, scalar k.
        if (isArrayVar(name) && arrays_.at(name).isLocal && arrays_.at(name).ndRuntimeLocal
            && arrays_.at(name).ndDims.size() == 3 && rhs.type == NodeType::CALL
            && rhs.children.size() == 4 && rhs.children[0]->type == NodeType::IDENTIFIER
            && rhs.children[0]->strValue == "circshift"
            && rhs.children[3]->type == NodeType::NUMBER_LITERAL
            && (rhs.children[3]->numValue == 1.0 || rhs.children[3]->numValue == 2.0
                || rhs.children[3]->numValue == 3.0)
            && rhs.children[1]->type == NodeType::IDENTIFIER
            && isArrayVar(rhs.children[1]->strValue)
            && arrays_.at(rhs.children[1]->strValue).isND
            && arrays_.at(rhs.children[1]->strValue).ndDims.size() == 3
            && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE) {
            const ArrayInfo    &A   = arrays_.at(rhs.children[1]->strValue);
            const ArrayInfo    &B   = arrays_.at(name);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete() && !res.type.shape.isScalar()
                && inferExpr(*rhs.children[2], types_, reg_, classes_).type.shape.isScalar()) {
                const int         dimv = static_cast<int>(rhs.children[3]->numValue);
                const std::string sd = dimv == 1 ? "1" : (dimv == 2 ? "_numkit_m" : "_numkit_m * _numkit_n");
                const std::string ld = dimv == 1 ? "_numkit_m" : (dimv == 2 ? "_numkit_n" : "_numkit_p");
                const std::string k  = emitExpr(*rhs.children[2]);
                line("{");
                ++indent_;
                line("const std::size_t _numkit_m = " + dimExpr(A, 0) + ";");
                line("const std::size_t _numkit_n = " + dimExpr(A, 1) + ";");
                line("const std::size_t _numkit_p = " + dimExpr(A, 2) + ";");
                line("const std::size_t _numkit_tot = _numkit_m * _numkit_n * _numkit_p;");
                line("const std::size_t _numkit_sd = " + sd + ";");
                line("const std::size_t _numkit_ld = " + ld + ";");
                line("const std::size_t _numkit_blk = _numkit_sd * _numkit_ld;");
                line("const std::ptrdiff_t _numkit_L = static_cast<std::ptrdiff_t>(_numkit_ld);");
                line("const std::ptrdiff_t _numkit_k = static_cast<std::ptrdiff_t>(" + k + ");");
                line("std::vector<double> _numkit_out(_numkit_tot);");
                open("for (std::size_t _numkit_b = 0; _numkit_b < _numkit_tot; _numkit_b += _numkit_blk)");
                open("for (std::size_t _numkit_r = 0; _numkit_r < _numkit_sd; ++_numkit_r)");
                open("for (std::size_t _numkit_t = 0; _numkit_t < _numkit_ld; ++_numkit_t)");
                line("const std::size_t _numkit_ts = static_cast<std::size_t>("
                     "((static_cast<std::ptrdiff_t>(_numkit_t) - _numkit_k) % _numkit_L + _numkit_L) % _numkit_L);");
                line("_numkit_out[_numkit_b + _numkit_r + _numkit_t * _numkit_sd] = " + A.dataExpr
                     + "[_numkit_b + _numkit_r + _numkit_ts * _numkit_sd];");
                close();
                close();
                close();
                line(B.ndDims[0] + " = _numkit_m;");
                line(B.ndDims[1] + " = _numkit_n;");
                line(B.ndDims[2] + " = _numkit_p;");
                line(cppIdent(name) + ".assign(_numkit_out.begin(), _numkit_out.end());");
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native gradient(y) -> numerical gradient, unit spacing, SAME length as y.
        // Edge points use one-sided differences, the interior centered: g[0]=y[1]-y[0];
        // g[i]=(y[i+1]-y[i-1])/2; g[n-1]=y[n-1]-y[n-2]; n==1 -> {0}; n==0 -> empty.
        // Exact -> every tier. v1: a single 1-D DOUBLE array var; result a 1-D LOCAL.
        if (isArrayVar(name) && arrays_.at(name).isLocal && !arrays_.at(name).is2D
            && !arrays_.at(name).isND && rhs.type == NodeType::CALL && rhs.children.size() == 2
            && rhs.children[0]->type == NodeType::IDENTIFIER
            && rhs.children[0]->strValue == "gradient"
            && rhs.children[1]->type == NodeType::IDENTIFIER
            && isArrayVar(rhs.children[1]->strValue)
            && !arrays_.at(rhs.children[1]->strValue).is2D
            && !arrays_.at(rhs.children[1]->strValue).isND
            && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE) {
            const ArrayInfo    &y   = arrays_.at(rhs.children[1]->strValue);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete() && !res.type.shape.isScalar()) {
                const std::string d = y.dataExpr;
                line("{");
                ++indent_;
                line("const std::size_t _numkit_n = " + y.lenVar + ";");
                line(cppIdent(name) + ".assign(_numkit_n, 0.0);");
                open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_n; ++_numkit_i)");
                line("if (_numkit_n == 1) " + cppIdent(name) + "[_numkit_i] = 0.0;");
                line("else if (_numkit_i == 0) " + cppIdent(name) + "[_numkit_i] = " + d + "[1] - " + d + "[0];");
                line("else if (_numkit_i + 1 == _numkit_n) " + cppIdent(name) + "[_numkit_i] = " + d + "[_numkit_i] - " + d
                     + "[_numkit_i - 1];");
                line("else " + cppIdent(name) + "[_numkit_i] = (" + d + "[_numkit_i + 1] - " + d
                     + "[_numkit_i - 1]) * 0.5;");
                close();
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native flip/fliplr/flipud(x) on a 1-D vector -> a fresh same-length 1-D
        // LOCAL. flip reverses a vector; fliplr reverses along columns (a ROW is
        // reversed, a column is unchanged); flipud reverses along rows (a COLUMN is
        // reversed, a row is unchanged). Whether a reversal happens is decided from
        // the operand's orientation; fliplr/flipud on an ERASED-orientation 1-D
        // buffer can't be decided -> fall through (refuse). dtype-general (a plain
        // element copy). v1: a single 1-D array VAR; the result a 1-D array LOCAL.
        if (isArrayVar(name) && arrays_.at(name).isLocal && !arrays_.at(name).is2D
            && !arrays_.at(name).isND && rhs.type == NodeType::CALL && rhs.children.size() == 2
            && rhs.children[0]->type == NodeType::IDENTIFIER
            && (rhs.children[0]->strValue == "flip" || rhs.children[0]->strValue == "fliplr"
                || rhs.children[0]->strValue == "flipud")
            && rhs.children[1]->type == NodeType::IDENTIFIER
            && isArrayVar(rhs.children[1]->strValue)
            && !arrays_.at(rhs.children[1]->strValue).is2D
            && !arrays_.at(rhs.children[1]->strValue).isND) {
            const ArrayInfo    &xa  = arrays_.at(rhs.children[1]->strValue);
            const std::string  &fn  = rhs.children[0]->strValue;
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            bool                reverse   = true;
            bool                decidable = true;
            if (fn == "flip")
                reverse = true;  // a vector flip reverses regardless of orientation
            else if (xa.orient == VecOrient::Unknown)
                decidable = false;  // fliplr/flipud need a known orientation
            else if (fn == "fliplr")
                reverse = xa.orient == VecOrient::Row;
            else  // flipud
                reverse = xa.orient == VecOrient::Col;
            if (decidable && res.type.isConcrete() && !res.type.shape.isScalar()) {
                line("{");
                ++indent_;
                line(cppIdent(name) + ".assign(" + xa.lenVar + ", " + zeroLiteral(xa.dtype) + ");");
                open("for (std::size_t _numkit_i = 0; _numkit_i < " + xa.lenVar + "; ++_numkit_i)");
                line(cppIdent(name) + "[_numkit_i] = " + xa.dataExpr + "["
                     + (reverse ? (xa.lenVar + " - 1 - _numkit_i") : std::string("_numkit_i")) + "];");
                close();
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native flip/fliplr/flipud(A) on a 2-D MATRIX (KnownDims or runtime-dim) -> a fresh
        // same-shape matrix. fliplr reverses the COLUMN order, flipud (and flip's default
        // dim-1) reverses the ROW order. The 2-arg flip(A, dim) form with a LITERAL dim 1|2
        // is also handled: dim 1 reverses rows (= flipud), dim 2 reverses columns (= fliplr).
        // Column-major, both tiers via dimExpr: fliplr M[i+j*m] = A[i+(n-1-j)*m]; flipud
        // M[i+j*m] = A[(m-1-i)+j*m]. v1: a DOUBLE matrix var distinct from the dest (an in-place
        // B=flip(B) would self-alias the read).
        {
            const bool flipTwoArg =
                rhs.type == NodeType::CALL && rhs.children.size() == 3
                && rhs.children[0]->type == NodeType::IDENTIFIER
                && rhs.children[0]->strValue == "flip"
                && rhs.children[2]->type == NodeType::NUMBER_LITERAL
                && (rhs.children[2]->numValue == 1.0 || rhs.children[2]->numValue == 2.0);
            if (isArrayVar(name) && arrays_.at(name).isLocal
                && (arrays_.at(name).is2D
                    || (arrays_.at(name).isND && arrays_.at(name).ndDims.size() == 2))
                && rhs.type == NodeType::CALL && (rhs.children.size() == 2 || flipTwoArg)
                && rhs.children[0]->type == NodeType::IDENTIFIER
                && (rhs.children[0]->strValue == "flip" || rhs.children[0]->strValue == "fliplr"
                    || rhs.children[0]->strValue == "flipud")
                && rhs.children[1]->type == NodeType::IDENTIFIER
                && isArrayVar(rhs.children[1]->strValue) && rhs.children[1]->strValue != name
                && (arrays_.at(rhs.children[1]->strValue).is2D
                    || (arrays_.at(rhs.children[1]->strValue).isND
                        && arrays_.at(rhs.children[1]->strValue).ndDims.size() == 2))) {
            const ArrayInfo    &A   = arrays_.at(rhs.children[1]->strValue);
            const ArrayInfo    &M   = arrays_.at(name);
            const std::string  &fn  = rhs.children[0]->strValue;
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete()) {
                const bool flipRows = flipTwoArg ? (rhs.children[2]->numValue == 1.0)
                                                 : (fn == "flipud" || fn == "flip");  // dim 1 = rows
                line("{");
                ++indent_;
                line("const std::size_t _numkit_m = " + dimExpr(A, 0) + ";");
                line("const std::size_t _numkit_n = " + dimExpr(A, 1) + ";");
                if (M.isND && M.ndDims.size() == 2) {  // runtime-dim dst: set its companions
                    line(M.ndDims[0] + " = _numkit_m;");
                    line(M.ndDims[1] + " = _numkit_n;");
                }
                line(cppIdent(name) + ".assign(_numkit_m * _numkit_n, " + zeroLiteral(M.dtype) + ");");
                open("for (std::size_t _numkit_j = 0; _numkit_j < _numkit_n; ++_numkit_j)");
                open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_m; ++_numkit_i)");
                line(cppIdent(name) + "[_numkit_i + _numkit_j * _numkit_m] = " + A.dataExpr + "["
                     + (flipRows ? "(_numkit_m - 1 - _numkit_i) + _numkit_j * _numkit_m"
                                 : "_numkit_i + (_numkit_n - 1 - _numkit_j) * _numkit_m")
                     + "];");
                close();
                close();
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
            }
        }
        // Native flip(A, dim) / fliplr(A) / flipud(A) on a rank-3 ARRAY -> a fresh same-shape rank-3
        // array, reversing along a dim: dim 1 reverses rows within each page, dim 2 reverses
        // columns, dim 3 reverses the page order. flip(A,dim) takes a LITERAL dim 1|2|3; fliplr(A)
        // is dim 2 and flipud(A) is dim 1 (both UNAMBIGUOUS on a rank-3 -- no first-non-singleton
        // issue, so the 1-arg forms are native here). Column-major: B[i + j*m + k*m*n] =
        // A[i' + j'*m + k'*m*n] with the chosen axis coordinate reflected (m-1-i / n-1-j / p-1-k). A
        // runtime-dim rank-3 ndRuntimeLocal dest. v1: a DOUBLE rank-3 var distinct from the dest.
        // (1-arg flip(A) on a rank-3 stays bridged -- first-non-singleton-dim ambiguity.)
        {
            const std::string fn = (rhs.type == NodeType::CALL && !rhs.children.empty()
                                    && rhs.children[0]->type == NodeType::IDENTIFIER)
                                       ? rhs.children[0]->strValue
                                       : std::string();
            const bool flip3 = rhs.type == NodeType::CALL && rhs.children.size() == 3 && fn == "flip"
                               && rhs.children[2]->type == NodeType::NUMBER_LITERAL
                               && (rhs.children[2]->numValue == 1.0 || rhs.children[2]->numValue == 2.0
                                   || rhs.children[2]->numValue == 3.0);
            const bool lr2 = rhs.type == NodeType::CALL && rhs.children.size() == 2 && fn == "fliplr";
            const bool ud2 = rhs.type == NodeType::CALL && rhs.children.size() == 2 && fn == "flipud";
            if (isArrayVar(name) && arrays_.at(name).isLocal && arrays_.at(name).ndRuntimeLocal
                && arrays_.at(name).ndDims.size() == 3 && (flip3 || lr2 || ud2)
                && rhs.children[1]->type == NodeType::IDENTIFIER
                && isArrayVar(rhs.children[1]->strValue) && rhs.children[1]->strValue != name
                && arrays_.at(rhs.children[1]->strValue).isND
                && arrays_.at(rhs.children[1]->strValue).ndDims.size() == 3
                && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE) {
            const ArrayInfo    &A   = arrays_.at(rhs.children[1]->strValue);
            const ArrayInfo    &M   = arrays_.at(name);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete() && !res.type.shape.isScalar()) {
                const int d = flip3 ? static_cast<int>(rhs.children[2]->numValue)
                                    : (lr2 ? 2 : 1);  // fliplr -> dim 2, flipud -> dim 1
                line("{");
                ++indent_;
                line("const std::size_t _numkit_m = " + dimExpr(A, 0) + ";");
                line("const std::size_t _numkit_n = " + dimExpr(A, 1) + ";");
                line("const std::size_t _numkit_p = " + dimExpr(A, 2) + ";");
                line(M.ndDims[0] + " = _numkit_m;");
                line(M.ndDims[1] + " = _numkit_n;");
                line(M.ndDims[2] + " = _numkit_p;");
                line(cppIdent(name) + ".assign(_numkit_m * _numkit_n * _numkit_p, 0.0);");
                const std::string ip = d == 1 ? "(_numkit_m - 1 - _numkit_i)" : "_numkit_i";
                const std::string jp = d == 2 ? "(_numkit_n - 1 - _numkit_j)" : "_numkit_j";
                const std::string kp = d == 3 ? "(_numkit_p - 1 - _numkit_k)" : "_numkit_k";
                open("for (std::size_t _numkit_k = 0; _numkit_k < _numkit_p; ++_numkit_k)");
                open("for (std::size_t _numkit_j = 0; _numkit_j < _numkit_n; ++_numkit_j)");
                open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_m; ++_numkit_i)");
                line(cppIdent(name) + "[_numkit_i + _numkit_j * _numkit_m + _numkit_k * _numkit_m * _numkit_n] = " + A.dataExpr + "["
                     + ip + " + " + jp + " * _numkit_m + " + kp + " * _numkit_m * _numkit_n];");
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
        // Native rot90(A, k) on a 2-D matrix with a LITERAL non-negative k -> rotate k*90deg CCW.
        // By k%4: 0 = identity (m x n, B[i+j*m]=A[i+j*m]); 1 = 90 CCW (n x m, B[i+j*n]=A[j+(n-1-i)*m]);
        // 2 = 180 (m x n, B[i+j*m]=A[(m-1-i)+(n-1-j)*m]); 3 = 270 CCW / 90 CW (n x m, B[i+j*n]=
        // A[(m-1-j)+i*m]). Column-major, both tiers via dimExpr; a runtime dst gets its companions
        // set. v1: a DOUBLE matrix var distinct from the dest. (Placed before the 1-arg form.)
        if (isArrayVar(name) && arrays_.at(name).isLocal
            && (arrays_.at(name).is2D
                || (arrays_.at(name).isND && arrays_.at(name).ndDims.size() == 2))
            && rhs.type == NodeType::CALL && rhs.children.size() == 3
            && rhs.children[0]->type == NodeType::IDENTIFIER && rhs.children[0]->strValue == "rot90"
            && rhs.children[1]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[1]->strValue)
            && rhs.children[1]->strValue != name
            && (arrays_.at(rhs.children[1]->strValue).is2D
                || (arrays_.at(rhs.children[1]->strValue).isND
                    && arrays_.at(rhs.children[1]->strValue).ndDims.size() == 2))
            && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE
            && rhs.children[2]->type == NodeType::NUMBER_LITERAL
            && rhs.children[2]->numValue >= 0.0) {
            const ArrayInfo    &A   = arrays_.at(rhs.children[1]->strValue);
            const ArrayInfo    &M   = arrays_.at(name);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete() && !res.type.shape.isScalar()) {
                const int  k    = static_cast<int>(rhs.children[2]->numValue) % 4;
                const bool swap = (k == 1 || k == 3);  // result dims swap for 90/270
                line("{");
                ++indent_;
                line("const std::size_t _numkit_m = " + dimExpr(A, 0) + ";");  // A rows
                line("const std::size_t _numkit_n = " + dimExpr(A, 1) + ";");  // A cols
                // result rows R / cols C: swapped (n x m) for 90/270, else m x n
                const std::string R = swap ? "_numkit_n" : "_numkit_m";
                const std::string C = swap ? "_numkit_m" : "_numkit_n";
                if (M.isND && M.ndDims.size() == 2) {
                    line(M.ndDims[0] + " = " + R + ";");
                    line(M.ndDims[1] + " = " + C + ";");
                }
                line(cppIdent(name) + ".assign(" + R + " * " + C + ", 0.0);");
                // srcIdx into A (column-major, A is m x n) for result element (i, j):
                std::string srcIdx;
                if (k == 0)
                    srcIdx = "_numkit_i + _numkit_j * _numkit_m";
                else if (k == 1)
                    srcIdx = "_numkit_j + (_numkit_n - 1 - _numkit_i) * _numkit_m";
                else if (k == 2)
                    srcIdx = "(_numkit_m - 1 - _numkit_i) + (_numkit_n - 1 - _numkit_j) * _numkit_m";
                else  // k == 3
                    srcIdx = "(_numkit_m - 1 - _numkit_j) + _numkit_i * _numkit_m";
                open("for (std::size_t _numkit_j = 0; _numkit_j < " + C + "; ++_numkit_j)");
                open("for (std::size_t _numkit_i = 0; _numkit_i < " + R + "; ++_numkit_i)");
                line(cppIdent(name) + "[_numkit_i + _numkit_j * " + R + "] = " + A.dataExpr + "[" + srcIdx + "];");
                close();
                close();
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native rot90(A) on a 2-D matrix -> a 90deg-CCW rotation, dims swapped (A m x n ->
        // B n x m). Column-major: B[i + j*n] = A[j + (n-1-i)*m] (B(i,j) = A(j, n-1-i)). Both
        // tiers via dimExpr; a runtime dst gets its companions set (rows=n, cols=m). v1: the
        // 1-arg form, a DOUBLE matrix var distinct from the dest. (rot90(A,k) / vector rot90
        // stay bridged.)
        if (isArrayVar(name) && arrays_.at(name).isLocal
            && (arrays_.at(name).is2D
                || (arrays_.at(name).isND && arrays_.at(name).ndDims.size() == 2))
            && rhs.type == NodeType::CALL && rhs.children.size() == 2
            && rhs.children[0]->type == NodeType::IDENTIFIER && rhs.children[0]->strValue == "rot90"
            && rhs.children[1]->type == NodeType::IDENTIFIER
            && isArrayVar(rhs.children[1]->strValue) && rhs.children[1]->strValue != name
            && (arrays_.at(rhs.children[1]->strValue).is2D
                || (arrays_.at(rhs.children[1]->strValue).isND
                    && arrays_.at(rhs.children[1]->strValue).ndDims.size() == 2))
            && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE) {
            const ArrayInfo    &A   = arrays_.at(rhs.children[1]->strValue);
            const ArrayInfo    &M   = arrays_.at(name);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete()) {
                line("{");
                ++indent_;
                line("const std::size_t _numkit_m = " + dimExpr(A, 0) + ";");  // A rows
                line("const std::size_t _numkit_n = " + dimExpr(A, 1) + ";");  // A cols
                if (M.isND && M.ndDims.size() == 2) {  // result is n x m
                    line(M.ndDims[0] + " = _numkit_n;");
                    line(M.ndDims[1] + " = _numkit_m;");
                }
                line(cppIdent(name) + ".assign(_numkit_n * _numkit_m, 0.0);");
                open("for (std::size_t _numkit_j = 0; _numkit_j < _numkit_m; ++_numkit_j)");  // B cols
                open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_n; ++_numkit_i)");  // B rows
                line(cppIdent(name) + "[_numkit_i + _numkit_j * _numkit_n] = " + A.dataExpr
                     + "[_numkit_j + (_numkit_n - 1 - _numkit_i) * _numkit_m];");
                close();
                close();
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        // Native rot90(A) on a rank-3 ARRAY -> rotate each page 90deg CCW in the (dim1,dim2) plane,
        // dim 3 fixed: A m x n x p -> B n x m x p, B(:,:,k) = rot90(A(:,:,k)). Column-major per page
        // (equal page size m*n): B[base + i + j*n] = A[base + j + (n-1-i)*m], base = k*m*n. A
        // runtime-dim rank-3 ndRuntimeLocal dest. v1: the 1-arg form, a DOUBLE rank-3 var distinct
        // from the dest. (rot90(A,k) stays bridged.)
        if (isArrayVar(name) && arrays_.at(name).isLocal && arrays_.at(name).ndRuntimeLocal
            && arrays_.at(name).ndDims.size() == 3 && rhs.type == NodeType::CALL
            && rhs.children.size() == 2 && rhs.children[0]->type == NodeType::IDENTIFIER
            && rhs.children[0]->strValue == "rot90"
            && rhs.children[1]->type == NodeType::IDENTIFIER && isArrayVar(rhs.children[1]->strValue)
            && rhs.children[1]->strValue != name && arrays_.at(rhs.children[1]->strValue).isND
            && arrays_.at(rhs.children[1]->strValue).ndDims.size() == 3
            && arrays_.at(rhs.children[1]->strValue).dtype == ValueType::DOUBLE) {
            const ArrayInfo    &A   = arrays_.at(rhs.children[1]->strValue);
            const ArrayInfo    &M   = arrays_.at(name);
            const AbstractValue res = inferExpr(rhs, types_, reg_, classes_);
            if (res.type.isConcrete() && !res.type.shape.isScalar()) {
                line("{");
                ++indent_;
                line("const std::size_t _numkit_m = " + dimExpr(A, 0) + ";");
                line("const std::size_t _numkit_n = " + dimExpr(A, 1) + ";");
                line("const std::size_t _numkit_p = " + dimExpr(A, 2) + ";");
                line(M.ndDims[0] + " = _numkit_n;");  // result rows = A cols
                line(M.ndDims[1] + " = _numkit_m;");  // result cols = A rows
                line(M.ndDims[2] + " = _numkit_p;");
                line(cppIdent(name) + ".assign(_numkit_n * _numkit_m * _numkit_p, 0.0);");
                line("const std::size_t _numkit_pg = _numkit_m * _numkit_n;");  // page size (A and B equal)
                open("for (std::size_t _numkit_k = 0; _numkit_k < _numkit_p; ++_numkit_k)");
                line("const std::size_t _numkit_b = _numkit_k * _numkit_pg;");
                open("for (std::size_t _numkit_j = 0; _numkit_j < _numkit_m; ++_numkit_j)");  // B cols
                open("for (std::size_t _numkit_i = 0; _numkit_i < _numkit_n; ++_numkit_i)");  // B rows
                line(cppIdent(name) + "[_numkit_b + _numkit_i + _numkit_j * _numkit_n] = " + A.dataExpr
                     + "[_numkit_b + _numkit_j + (_numkit_n - 1 - _numkit_i) * _numkit_m];");
                close();
                close();
                close();
                --indent_;
                line("}");
                types_.set(name, res);
                return;
            }
        }
        emitAssignIdentifierTail(lhs, rhs, name, structWriteFld);
        return;
    }

    if (lhs.type == NodeType::FIELD_ACCESS
        || lhs.type == NodeType::DYNAMIC_FIELD_ACCESS) {  // obj.field = rhs / s(i).a = rhs
        if (lhs.children.empty()) unsupported("field write arity");
        // s(i).a = v : struct-ARRAY field write (SoA, G2.5) -> _numkit_fld_s_a(i) = v, GROWABLE
        // (index_set_grow), so s(i).a=v in a loop builds the field-array. s is not a value;
        // only its field-arrays exist -> matched before inferring the base / structFieldLocal.
        if (lhs.children[0]->type == NodeType::CALL && lhs.children[0]->children.size() == 2
            && lhs.children[0]->children[0]->type == NodeType::IDENTIFIER) {
            const std::string sv  = lhs.children[0]->children[0]->strValue;
            const std::string fld = "_numkit_fld_" + sv + "_" + lhs.strValue;
            if (isArrayVar(fld) && !types_.has(sv) && arrays_.at(fld).isLocal
                && !arrays_.at(fld).is2D && !arrays_.at(fld).isND) {
                const ArrayInfo &fa = arrays_.at(fld);
                endStack_.push_back(fa.lenVar);
                const std::string idx = emitExpr(*lhs.children[0]->children[1]);
                endStack_.pop_back();
                line("numkit_rt::index_set_grow(" + fld + ", " + idx + ", " + emitExpr(rhs) + ");");
                return;
            }
        }
        const AbstractValue base = inferExpr(*lhs.children[0], types_, reg_, classes_);
        if (!base.type.isObject() || !classes_)
            unsupported("field write on a non-object value");
        const ClassInfo *ci = classes_->byId(base.type.classId);
        if (!ci || !ci->field(lhs.strValue))
            unsupported("unknown field '" + lhs.strValue + "'");
        line(emitExpr(*lhs.children[0]) + (ci->isHandle ? "->" : ".") + cppIdent(lhs.strValue)
             + " = " + emitExpr(rhs) + ";");
        return;
    }

    if ((lhs.type == NodeType::CALL || lhs.type == NodeType::INDEX)
        && !lhs.children.empty()
        && (lhs.children[0]->type == NodeType::IDENTIFIER
            || !resolveArrayVar(*lhs.children[0]).empty())) {
        emitIndexWrite(lhs, rhs);
        if (lhs.children[0]->type == NodeType::FIELD_ACCESS
            || lhs.children[0]->type == NodeType::DYNAMIC_FIELD_ACCESS) {
            const std::string fld = resolveArrayVar(*lhs.children[0]);
            if (!fld.empty()) {
                const ArrayInfo &ai = arrays_.at(fld);
                const InferredType ft =
                    types_.has(fld) ? types_.get(fld).type
                                    : InferredType::concrete(ai.dtype, Shape::rowVector());
                recordStructFieldWrite(*lhs.children[0], fld, ft);
            }
        }
        return;
    }

    // c{i} = rhs / c{i,j} = rhs : cell content store on a Dynamic cell local (G4b).
    // v1: one or two unboxed-scalar subscripts on a Dynamic VARIABLE; the runtime
    // grows the cell when a subscript is past the end (MATLAB auto-grow), so
    // c={}; c{1}=..; c{2}=.. (or c{1,1}=..; c{2,2}=..) builds a cell imperatively.
    // A CSL c{:} / >2 subscripts / non-Dynamic target refuses -> interpreted (sound).
    if (lhs.type == NodeType::CELL_INDEX && !lhs.children.empty()
        && lhs.children[0]->type == NodeType::IDENTIFIER
        && (types_.has(lhs.children[0]->strValue)
            || dynamicLocals_.count(lhs.children[0]->strValue))
        && (dynamicLocals_.count(lhs.children[0]->strValue)
            || inferExpr(*lhs.children[0], types_, reg_, classes_).type.isDynamic())) {
        const std::size_t nsub = lhs.children.size() - 1;
        if (nsub != 1 && nsub != 2)
            unsupported("Dynamic tier: cell content store supports 1 or 2 subscripts (v1)");
        for (std::size_t k = 1; k <= nsub; ++k)
            if (!isUnboxableScalarType(inferExpr(*lhs.children[k], types_, reg_, classes_).type))
                unsupported("Dynamic tier: cell content store index must be unboxed scalar(s) (v1)");
        const std::string cvar = cppIdent(lhs.children[0]->strValue);
        if (nsub == 1)
            line("numkit_rt::cell_set(" + cvar + ", " + emitExpr(*lhs.children[1]) + ", "
                 + emitDynamicExpr(rhs) + ");");
        else
            line("numkit_rt::cell_set_2d(" + cvar + ", " + emitExpr(*lhs.children[1]) + ", "
                 + emitExpr(*lhs.children[2]) + ", " + emitDynamicExpr(rhs) + ");");
        return;
    }
    unsupported("assignment lhs kind");
}

} // namespace numkit::codegen::detail
