// codegen/src/emitter_stmt.cpp — control-flow and statement lowering.

#include "emitter_internal.hpp"

namespace numkit::codegen::detail {

void Emitter::emitFor(const ASTNode &s)
{
    if (s.children.size() != 2) unsupported("for arity");
    const ASTNode &range = *s.children[0];
    if (range.type != NodeType::COLON_EXPR)
        unsupported("for range must be a colon expression a:b / a:s:b");

    const std::string &loopVar  = s.strValue;
    const std::string  cLoopVar = cppIdent(loopVar);
    const ASTNode     &body     = *s.children[1];

    // Settle body types (fixpoint) so indexing / disambiguation inside the
    // loop sees loop-carried types. The loop var takes the range element
    // type (a colon range is always double scalar). The env only widens
    // between iterations, so this terminates.
    const AbstractValue iterVal{InferredType::scalar(ValueType::DOUBLE),
                                ConstVal::unknown()};
    TypeEnv cur       = types_;
    bool    converged = false;
    for (int i = 0; i < kMaxFixpoint; ++i) {
        TypeEnv bodyEnv = cur;
        bodyEnv.set(loopVar, iterVal);
        inferStmt(body, bodyEnv, reg_, nullptr, classes_);
        TypeEnv next = joinEnv(cur, bodyEnv);
        if (next == cur) {
            converged = true;
            break;
        }
        cur = next;
    }
    while (!converged) {
        TypeEnv bodyEnv = cur;
        bodyEnv.set(loopVar, iterVal);
        inferStmt(body, bodyEnv, reg_, nullptr, classes_);
        TypeEnv next = joinEnv(cur, bodyEnv);
        if (next == cur) {
            converged = true;
            break;
        }
        for (const auto &[k, av] : next.entries())
            if (!cur.has(k) || av != cur.get(k))
                cur.set(k, AbstractValue::dynamic());
    }
    cur.set(loopVar, iterVal);
    types_ = cur;

    // brick 6: clean-index promotion -> 0-based size_t counter (no double
    // loop var, no bounds check). Gated on analyzeOptimizations(); absent
    // the fact this falls through to the always-correct checked form.
    if (opt_.promotedLoops.count(&s)) {
        const std::string         *saved = promotedCounter_;
        promotedCounter_                 = &loopVar;
        open("for (std::size_t " + cLoopVar + " = 0; " + cLoopVar + " < "
             + opt_.loopBound.at(&s) + "; ++" + cLoopVar + ")");
        emitStmt(body);
        close();
        promotedCounter_ = saved;
        types_           = cur;
        return;
    }

    // Default (always correct): a double loop variable counting the colon
    // range; index sites stay bounds-checked.
    const std::string startE = emitExpr(*range.children[0]);
    std::string       stopE, stepE, cond;
    if (range.children.size() == 2) {
        stopE = emitExpr(*range.children[1]);
        stepE = "1.0";
        cond  = "<=";
    } else if (range.children.size() == 3) {
        const ConstVal sc = inferExpr(*range.children[1], types_, reg_, classes_).constant;
        if (!sc.isKnown() || sc.value == 0.0)
            unsupported("for range step must be a known non-zero constant");
        stepE = emitExpr(*range.children[1]);
        stopE = emitExpr(*range.children[2]);
        cond  = sc.value > 0.0 ? "<=" : ">=";
    } else {
        unsupported("malformed colon range");
    }

    open("for (" + cLoopVar + " = " + startE + "; " + cLoopVar + " " + cond + " " + stopE
         + "; " + cLoopVar + " += " + stepE + ")");
    emitStmt(body);
    close();
    types_ = cur;
}

void Emitter::emitWhile(const ASTNode &s)
{
    if (s.children.size() != 2) unsupported("while arity");
    const ASTNode &body = *s.children[1];

    TypeEnv cur       = types_;
    bool    converged = false;
    for (int i = 0; i < kMaxFixpoint; ++i) {
        TypeEnv bodyEnv = cur;
        inferStmt(body, bodyEnv, reg_, nullptr, classes_);
        TypeEnv next = joinEnv(cur, bodyEnv);
        if (next == cur) {
            converged = true;
            break;
        }
        cur = next;
    }
    while (!converged) {
        TypeEnv bodyEnv = cur;
        inferStmt(body, bodyEnv, reg_, nullptr, classes_);
        TypeEnv next = joinEnv(cur, bodyEnv);
        if (next == cur) {
            converged = true;
            break;
        }
        for (const auto &[k, av] : next.entries())
            if (!cur.has(k) || av != cur.get(k))
                cur.set(k, AbstractValue::dynamic());
    }
    types_ = cur;

    open("while (" + emitCondition(*s.children[0]) + ")");
    emitStmt(body);
    close();
    types_ = cur;
}

void Emitter::emitIf(const ASTNode &s)
{
    const TypeEnv entry = types_;
    TypeEnv merged;
    bool    have = false;
    auto    mergeIn = [&](const TypeEnv &e) {
        merged = have ? joinEnv(merged, e) : e;
        have   = true;
    };

    for (std::size_t i = 0; i < s.branches.size(); ++i) {
        types_ = entry;  // condition + body typed from the incoming env
        const std::string cond = emitCondition(*s.branches[i].first);
        open((i == 0 ? "if (" : "else if (") + cond + ")");
        if (s.branches[i].second) emitStmt(*s.branches[i].second);
        close();
        mergeIn(types_);
    }
    if (s.elseBranch) {
        types_ = entry;
        open("else");
        emitStmt(*s.elseBranch);
        close();
        mergeIn(types_);
    } else {
        mergeIn(entry);  // no branch taken — fall-through
    }
    types_ = have ? merged : entry;
}

void Emitter::emitSwitch(const ASTNode &s)
{
    // switch: evaluate the selector ONCE, then take the first case whose
    // value — or ANY element of a `case {a,b}` cell-list — equals it (isequal),
    // else `otherwise`. Lowered to an if-else chain over a selector temp. v1: an
    // unboxed-scalar selector + unboxed-scalar case values, so isequal reduces to
    // `==` (NaN matches nothing). A string/array selector or case
    // value (full isequal), or a 2-D cell case-list, is refused.
    if (s.children.empty()) unsupported("switch with no selector expression");
    const ASTNode      &sel  = *s.children[0];
    const AbstractValue selv = inferExpr(sel, types_, reg_, classes_);
    if (!isUnboxableScalarType(selv.type))
        unsupported("switch selector must be an unboxed scalar (v1)");
    const std::string tmp = "_numkit_switch" + std::to_string(switchCounter_++);

    const TypeEnv entry = types_;
    TypeEnv       merged;
    bool          have = false;
    auto mergeIn = [&](const TypeEnv &e) { merged = have ? joinEnv(merged, e) : e; have = true; };

    open("");  // scope the selector temp (no leak / nested-switch name collision)
    line("const " + cppScalarType(selv.type.dtype) + " " + tmp + " = " + emitExpr(sel) + ";");

    auto eqTerm = [&](const ASTNode &v) -> std::string {
        const AbstractValue vv = inferExpr(v, types_, reg_, classes_);
        if (!isUnboxableScalarType(vv.type))
            unsupported("switch case value must be an unboxed scalar (v1)");
        return "(" + tmp + " == " + emitExpr(v) + ")";
    };

    for (std::size_t i = 0; i < s.branches.size(); ++i) {
        types_ = entry;  // each case body typed from the incoming env
        const ASTNode &caseVal = *s.branches[i].first;
        std::string    cond;
        if (caseVal.type == NodeType::CELL_LITERAL) {
            // `case {a,b,...}` matches the selector against ANY element. A cell
            // literal stores each row as a BLOCK child (a 1-row cell -> one row
            // block), so descend a row block to reach the elements; flatten all
            // rows (matching any element is rank-agnostic).
            auto addElem = [&](const ASTNode &el) {
                cond += (cond.empty() ? "" : " || ") + eqTerm(el);
            };
            for (const auto &child : caseVal.children) {
                if (!child) continue;
                if (child->type == NodeType::BLOCK)
                    for (const auto &el : child->children) { if (el) addElem(*el); }
                else
                    addElem(*child);
            }
            if (cond.empty()) cond = "false";  // `case {}` matches nothing
        } else {
            cond = eqTerm(caseVal);
        }
        open((i == 0 ? "if (" : "else if (") + cond + ")");
        if (s.branches[i].second) emitStmt(*s.branches[i].second);
        close();
        mergeIn(types_);
    }
    if (s.elseBranch) {
        types_ = entry;
        open(s.branches.empty() ? "" : "else");  // otherwise; a bare block if no cases
        emitStmt(*s.elseBranch);
        close();
        mergeIn(types_);
    } else {
        mergeIn(entry);  // no case matched — fall-through
    }
    close();  // selector-temp scope
    types_ = have ? merged : entry;
}

void Emitter::emitTry(const ASTNode &s)
{
    // try/catch -> C++ `try { <body> } catch (...) { <handler> }`. A runtime
    // error surfaces as a C++ exception, so `catch (...)` catches it. The inference
    // marks try/catch-assigned vars Dynamic (a throw mid-try leaves them uncertain
    // — sound over-approximation), so the bodies ride the Dynamic tier (bridge).
    // v1: the catch's bound exception variable (an MException object)
    // is NOT represented -> refuse a `catch err` form; a bare `catch`, or
    // `try ... end` with no catch (which silently swallows), is supported.
    if (s.children.empty()) unsupported("try with no body");
    if (!s.strValue.empty())
        unsupported("try/catch with a bound exception variable '" + s.strValue
                    + "' (MException object not represented; v1)");

    const TypeEnv entry = types_;
    TypeEnv       merged;
    bool          have = false;
    auto mergeIn = [&](const TypeEnv &e) { merged = have ? joinEnv(merged, e) : e; have = true; };

    open("try");
    emitStmt(*s.children[0]);  // try body
    close();
    mergeIn(types_);  // try ran to completion
    mergeIn(entry);   // try threw before assigning -> the incoming types

    types_ = entry;   // catch body typed from the incoming env
    open("catch (...)");
    if (s.children.size() > 1 && s.children[1]) emitStmt(*s.children[1]);
    close();
    mergeIn(types_);

    types_ = have ? merged : entry;
}

void Emitter::emitStmt(const ASTNode &s)
{
    switch (s.type) {
    case NodeType::BLOCK:
        for (const auto &c : s.children)
            if (c) emitStmt(*c);
        return;
    case NodeType::ASSIGN:        emitAssign(s);      return;
    case NodeType::MULTI_ASSIGN:  emitMultiAssign(s); return;
    case NodeType::FOR_STMT:   emitFor(s);    return;
    case NodeType::WHILE_STMT: emitWhile(s);  return;
    case NodeType::IF_STMT:     emitIf(s);     return;
    case NodeType::SWITCH_STMT: emitSwitch(s); return;
    case NodeType::TRY_STMT:    emitTry(s);    return;
    // break / continue lower directly to the C++ loop-control keywords. They only
    // ever appear inside a loop body (language semantics + the parser enforce it,
    // and the emitted code mirrors that structure), so the keyword lands inside
    // the corresponding emitted for/while. No effect on the index/bound analyses
    // (those are per-expression, not control-flow sensitive); an early exit just
    // leaves arrays filled up to the exit point.
    case NodeType::BREAK_STMT:    line("break;");    return;
    case NodeType::CONTINUE_STMT: line("continue;"); return;
    case NodeType::RETURN_STMT:
        // Early return -> the SAME form as the end-of-function return (set by
        // emitOneFunction via setReturnInfo). A value function returns its output
        // var (boxed via .take() for a Dynamic result); a void / out-param /
        // multi-ref function emits a bare `return;`.
        emitEpilogueAssigns();
        if (!returnsValue_)      line("return;");
        else if (returnDynamic_) emitReturnDynamic(returnName_);
        else                     emitReturnScalar(returnName_);
        return;
    case NodeType::COMMAND_CALL:
        if (s.strValue == "clear" || s.strValue == "clc" || s.strValue == "close")
            return;
        unsupported("command call '" + s.strValue + "'");
    case NodeType::EXPR_STMT:
        if (!s.children.empty() && s.children[0]) {
            const ASTNode &e = *s.children[0];
            if (e.type == NodeType::IDENTIFIER && !types_.has(e.strValue)
                && (e.strValue == "clear" || e.strValue == "clc" || e.strValue == "close")) {
                return;
            }
            if (e.type == NodeType::CALL && !e.children.empty()
                && e.children[0]->type == NodeType::IDENTIFIER) {
                const std::string &fn = e.children[0]->strValue;
                const bool shadowed = types_.has(fn)
                                      || (ctx_ && ctx_->funcs && ctx_->funcs->has(fn));
                if (!shadowed && fn == "disp" && e.children.size() == 2) {
                    const ASTNode &arg = *e.children[1];
                    if (arg.type == NodeType::STRING_LITERAL
                        || arg.type == NodeType::DQSTRING_LITERAL) {
                        bool ok = false;
                        const std::string esc = cEscape(arg.strValue, ok);
                        if (!ok) unsupported("disp string literal contains non-printable bytes");
                        line("numkit_rt::disp(\"" + esc + "\");");
                        return;
                    }
                    if (arg.type == NodeType::IDENTIFIER && isArrayVar(arg.strValue)) {
                        const ArrayInfo &ai = arrays_.at(arg.strValue);
                        if (ai.dtype == ValueType::DOUBLE) {
                            if (ai.is2D || (ai.isND && ai.ndDims.size() == 2)) {
                                line("numkit_rt::disp_mat(" + ai.dataExpr + ", " + dimExpr(ai, 0) + ", "
                                     + dimExpr(ai, 1) + ");");
                                return;
                            }
                            if (!ai.is2D && !ai.isND) {
                                line("numkit_rt::disp_vec(" + ai.dataExpr + ", " + ai.lenVar + ");");
                                return;
                            }
                        }
                        if (bridge_) {
                            line("numkit_rt::call_void(\"disp\", {" + emitDynamicExpr(arg) + "});");
                            return;
                        }
                    }
                    const AbstractValue argAv = inferExpr(arg, types_, reg_, classes_);
                    if (bridge_ && !isUnboxableScalarType(argAv.type)) {
                        line("numkit_rt::call_void(\"disp\", {" + emitDynamicExpr(arg) + "});");
                        return;
                    }
                    line("numkit_rt::disp(" + emitExpr(arg) + ");");
                    return;
                }
                if (!shadowed && fn == "fprintf" && e.children.size() >= 2) {
                    std::size_t fmtIdx = 1;
                    std::string stream = "stdout";
                    if (e.children.size() >= 3
                        && e.children[1]->type == NodeType::NUMBER_LITERAL
                        && (e.children[1]->numValue == 1.0 || e.children[1]->numValue == 2.0)
                        && (e.children[2]->type == NodeType::STRING_LITERAL
                            || e.children[2]->type == NodeType::DQSTRING_LITERAL)) {
                        fmtIdx = 2;
                        stream = (e.children[1]->numValue == 2.0) ? "stderr" : "stdout";
                    }
                    if (e.children[fmtIdx]->type == NodeType::STRING_LITERAL
                        || e.children[fmtIdx]->type == NodeType::DQSTRING_LITERAL) {
                        const std::string &rawFmt = e.children[fmtIdx]->strValue;
                        std::string cppFmt;
                        std::vector<char> specKinds;
                        for (std::size_t i = 0; i < rawFmt.size(); ++i) {
                            const char c = rawFmt[i];
                            if (c == '%') {
                                if (i + 1 < rawFmt.size() && rawFmt[i + 1] == '%') {
                                    cppFmt += "%%";
                                    ++i;
                                    continue;
                                }
                                cppFmt += '%';
                                ++i;
                                while (i < rawFmt.size()
                                       && (rawFmt[i] == '-' || rawFmt[i] == '+' || rawFmt[i] == ' '
                                           || rawFmt[i] == '#' || rawFmt[i] == '0'
                                           || rawFmt[i] == '.'
                                           || (rawFmt[i] >= '0' && rawFmt[i] <= '9'))) {
                                    cppFmt += rawFmt[i++];
                                }
                                if (i >= rawFmt.size())
                                    unsupported("fprintf format ends with incomplete '%'");
                                const char conv = rawFmt[i];
                                if (conv == 'd' || conv == 'i') {
                                    cppFmt += "lld";
                                    specKinds.push_back('i');
                                } else if (conv == 'u' || conv == 'x' || conv == 'X'
                                           || conv == 'o') {
                                    cppFmt += "ll";
                                    cppFmt += conv;
                                    specKinds.push_back('u');
                                } else if (conv == 'f' || conv == 'F' || conv == 'e' || conv == 'E'
                                           || conv == 'g' || conv == 'G') {
                                    cppFmt += conv;
                                    specKinds.push_back('f');
                                } else if (conv == 's') {
                                    cppFmt += 's';
                                    specKinds.push_back('s');
                                } else if (conv == 'c') {
                                    cppFmt += 'c';
                                    specKinds.push_back('c');
                                } else {
                                    unsupported(std::string("fprintf format specifier '%") + conv
                                                + "'");
                                }
                            } else if (c == '\\') {
                                if (i + 1 < rawFmt.size()) {
                                    const char nx = rawFmt[i + 1];
                                    if (nx == 'n' || nx == 't' || nx == 'r' || nx == '\\'
                                        || nx == '\'' || nx == '"') {
                                        cppFmt += '\\';
                                        cppFmt += nx;
                                        ++i;
                                        continue;
                                    }
                                }
                                cppFmt += "\\\\";
                            } else if (c == '"') {
                                cppFmt += "\\\"";
                            } else if (c == '\n') {
                                cppFmt += "\\n";
                            } else if (c == '\r') {
                                cppFmt += "\\r";
                            } else if (c == '\t') {
                                cppFmt += "\\t";
                            } else {
                                cppFmt += c;
                            }
                        }
                        const std::size_t nArgs = e.children.size() - (fmtIdx + 1);
                        if (nArgs != specKinds.size())
                            unsupported("fprintf argument count does not match format specifiers");
                        std::string callStr = "std::fprintf(" + stream + ", \"" + cppFmt + "\"";
                        for (std::size_t k = 0; k < nArgs; ++k) {
                            const ASTNode &a = *e.children[fmtIdx + 1 + k];
                            const char     sk = specKinds[k];
                            callStr += ", ";
                            if (sk == 's') {
                                if (a.type != NodeType::STRING_LITERAL
                                    && a.type != NodeType::DQSTRING_LITERAL)
                                    unsupported("fprintf %s requires a string literal argument");
                                bool ok = false;
                                const std::string esc = cEscape(a.strValue, ok);
                                if (!ok) unsupported("fprintf %s literal has non-printable bytes");
                                callStr += "\"" + esc + "\"";
                            } else if (sk == 'i') {
                                callStr += "static_cast<long long>(" + emitExpr(a) + ")";
                            } else if (sk == 'u') {
                                callStr += "static_cast<unsigned long long>(" + emitExpr(a) + ")";
                            } else if (sk == 'c') {
                                callStr += "static_cast<int>(" + emitExpr(a) + ")";
                            } else {
                                callStr += "static_cast<double>(" + emitExpr(a) + ")";
                            }
                        }
                        callStr += ");";
                        line(callStr);
                        return;
                    }
                }
            }
            // A call evaluated for effect — a void method/function (e.g. a
            // handle class's in-place mutator), result discarded.
            if (e.type == NodeType::CALL) {
                try {
                    line(emitExpr(e) + ";");
                    return;
                } catch (const std::runtime_error &) {
                    if (bridge_ && !e.children.empty()
                        && e.children[0]->type == NodeType::IDENTIFIER) {
                        const std::string &fn = e.children[0]->strValue;
                        const bool shadowed = types_.has(fn)
                                              || (ctx_ && ctx_->funcs && ctx_->funcs->has(fn));
                        if (!shadowed) {
                            std::string args;
                            for (std::size_t k = 1; k < e.children.size(); ++k) {
                                args += (k > 1 ? ", " : "") + emitDynamicExpr(*e.children[k]);
                            }
                            line("numkit_rt::call_void(\"" + fn + "\", {" + args + "});");
                            return;
                        }
                    }
                    throw;
                }
            }
        }
        unsupported("expression statement (only a void call is supported)");
    default:
        unsupported("statement node kind");
    }
}

} // namespace numkit::codegen::detail
