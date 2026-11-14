// codegen/tests/wave5_codegen_safety_test.cpp
//
// Wave 5 safety, soundness, and AOT hardening regression suite:
//   1. Strict 1-based integer & bounds checks on 1-D, growable 1-D, 2-D, N-D,
//      and slice reads/writes (0, negative, fractional, NaN, Inf, >dim).
//   2. Call-site input/output array aliasing protection when an output vector
//      reallocates before the callee finishes reading an aliased input buffer.
//   3. Logical array (`std::vector<std::uint8_t>`) interprocedural & multi-out
//      returns.
//   4. Empty-vector guard (`std::out_of_range`) on inline `min(v)` / `max(v)`
//      in both statement and expression (IIFE) positions.
//   5. C++ reserved keywords as function, class, field, parameter, and local
//      identifiers.
//   6. `ShapeKind::Vector1D` vs rank-unknown `ShapeKind::Unknown`: joining
//      incompatible 2-D shapes refuses standalone miscompilation and executes
//      soundly via the bridged Dynamic tier.
//   7. Guaranteed loop post-fixpoint widening (`> kMaxFixpoint` shift chain)
//      preventing silent scalar mis-typing of unconverged loop variables.
//   8. POSIX shell quoting (spaces + single quotes), `WEXITSTATUS` decoding in
//      `aot::runExecutable`, and `-D` define validation in `aot::compile*`.

#include <numkit/codegen/aot.hpp>
#include <numkit/codegen/classinfo.hpp>
#include <numkit/codegen/emitter.hpp>
#include <numkit/codegen/inference.hpp>
#include <numkit/codegen/monomorphize.hpp>
#include <numkit/codegen/transfer.hpp>
#include <numkit/codegen/type_lattice.hpp>

#include <numkit/core/lexer.hpp>
#include <numkit/core/parser.hpp>
#include <numkit/core/standard_engine.hpp>
#include <numkit/value/value.hpp>

#include "numkit_codegen_test_paths.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using numkit::ValueType;
using namespace numkit::codegen;

namespace {

std::filesystem::path wave5TmpDir()
{
    auto p = std::filesystem::temp_directory_path() / "numkit_wave5_codegen_safety";
    std::filesystem::create_directories(p);
    return p;
}

std::unique_ptr<numkit::ASTNode> parseSrc(const std::string &src)
{
    numkit::Lexer  lex(src);
    numkit::Parser parser(lex.tokenize());
    return parser.parse();
}

const numkit::ASTNode *firstFunc(const numkit::ASTNode &root)
{
    for (const auto &c : root.children)
        if (c && c->type == numkit::NodeType::FUNCTION_DEF) return c.get();
    return nullptr;
}

}  // namespace

// ── 1. Strict runtime subscript & slice bounds checking ──────────────────
TEST(Wave5CodegenSafety, BoundsChecksRejectZeroNegativeNanInfFractionalAndSliceOob)
{
    if (!aot::available())
        GTEST_SKIP() << "no external compiler configured for this build";

    const char *src =
        "function y = read1d(x, idx)\n"
        "  y = x(idx);\n"
        "end\n"
        "function s = write1d(idx)\n"
        "  y = zeros(1, 3);\n"
        "  y(idx) = 9;\n"
        "  s = y(idx);\n"
        "end\n"
        "function s = grow1d(idx)\n"
        "  v = [1, 2];\n"
        "  v(idx) = 7;\n"
        "  s = numel(v) + v(idx);\n"
        "end\n"
        "function y = read2d(A, r, c)\n"
        "  y = A(r, c);\n"
        "end\n"
        "function s = nd3(i, j, k)\n"
        "  T = zeros(2, 2, 2);\n"
        "  T(i, j, k) = 5;\n"
        "  s = T(i, j, k);\n"
        "end\n"
        "function s = colslice(A, c)\n"
        "  v = A(:, c);\n"
        "  s = v(1) + v(2);\n"
        "end\n"
        "function s = rowslice(A, r)\n"
        "  v = A(r, :);\n"
        "  s = v(1) + v(2) + v(3);\n"
        "end\n"
        "function s = subslice(A, j1, j2)\n"
        "  B = A(:, j1:j2);\n"
        "  s = numel(B);\n"
        "end\n"
        "function s = writeslice1d(a, b)\n"
        "  y = [10, 20, 30, 40];\n"
        "  v = [7, 8];\n"
        "  y(a:b) = v;\n"
        "  s = y(1) + y(2) + y(3) + y(4);\n"
        "end\n"
        "function s = writeslicecol(c)\n"
        "  A = zeros(2, 3);\n"
        "  v = [4; 5];\n"
        "  A(:, c) = v;\n"
        "  s = A(1, c) + A(2, c);\n"
        "end\n"
        "function out = dispatch_bounds(mode, a, b)\n"
        "  x = [10, 20, 30];\n"
        "  M = [1, 2, 3; 4, 5, 6];\n"
        "  if mode == 1\n"
        "    out = read1d(x, a);\n"
        "  elseif mode == 2\n"
        "    out = write1d(a);\n"
        "  elseif mode == 3\n"
        "    out = grow1d(a);\n"
        "  elseif mode == 4\n"
        "    out = read2d(M, a, b);\n"
        "  elseif mode == 5\n"
        "    out = nd3(1, a, 1);\n"
        "  elseif mode == 6\n"
        "    out = colslice(M, a);\n"
        "  elseif mode == 7\n"
        "    out = writeslicecol(a);\n"
        "  elseif mode == 8\n"
        "    out = rowslice(M, a);\n"
        "  elseif mode == 9\n"
        "    out = subslice(M, a, b);\n"
        "  else\n"
        "    out = writeslice1d(a, b);\n"
        "  end\n"
        "end\n";

    TransferRegistry reg;
    registerStandardTransfers(reg);
    auto          root = parseSrc(src);
    FunctionTable table;
    collectFunctions(*root, table);
    registerUserFunctions(reg, table);

    const numkit::ASTNode *entryFn = table.find("dispatch_bounds");
    ASSERT_NE(entryFn, nullptr);
    const EmittedFunction ef = emitProgram(
        *entryFn,
        {{"mode", InferredType::scalar(ValueType::DOUBLE)},
         {"a", InferredType::scalar(ValueType::DOUBLE)},
         {"b", InferredType::scalar(ValueType::DOUBLE)}},
        table, reg);

    auto                   fixedRoot = parseSrc(
        "function y = fixed_out(idx)\n"
        "  y = zeros(1, 3);\n"
        "  y(idx) = 9;\n"
        "end\n");
    const EmittedFunction fixedEf = emitFunction(
        *firstFunc(*fixedRoot), {{"idx", InferredType::scalar(ValueType::DOUBLE)}}, reg);
    // Strip the duplicate prelude by taking fixedEf.signature + body after last `namespace numkit_rt` close,
    // or simply find `fixedEf.signature` in `fixedEf.source`.
    const auto sigPos = fixedEf.source.find(fixedEf.signature);
    ASSERT_NE(sigPos, std::string::npos);
    const std::string fixedDef = fixedEf.source.substr(sigPos);

    std::string program = ef.source + "\n" + fixedDef + R"CPP(
#include <cmath>
#include <limits>
#include <stdexcept>

template <typename Fn>
bool throwsOob(Fn&& fn) {
    try {
        fn();
        return false;
    } catch (const std::out_of_range&) {
        return true;
    }
}

int main() {
    auto call = [](double mode, double a, double b = 1.0) {
        return )CPP" + ef.name + R"CPP((mode, a, b);
    };
    const double nanv = std::numeric_limits<double>::quiet_NaN();
    const double infv = std::numeric_limits<double>::infinity();
    const double bads[] = {0.0, -1.0, 1.5, nanv, infv, -infv, 4.0};
    double outBuf[3] = {};

    // Valid baseline calls
    if (call(1, 2.0) != 20.0) return 1;
    if (call(2, 2.0) != 9.0) return 2;
    if (call(3, 5.0) != 12.0) return 3;
    if (call(4, 2.0, 3.0) != 6.0) return 4;
    if (call(5, 2.0) != 5.0) return 5;
    if (call(6, 2.0) != 7.0) return 6;
    if (call(7, 2.0) != 9.0) return 7;
    if (call(8, 2.0) != 15.0) return 8;
    if (call(9, 1.0, 2.0) != 4.0) return 9;
    fixed_out(2.0, outBuf, 3);
    if (outBuf[1] != 9.0) return 27;

    // 1-D read & fixed RawBuffer output write reject 0, <0, fractional, NaN, Inf, >len
    for (double b : bads) {
        if (!throwsOob([&]{ return call(1, b); })) return 10;
        if (!throwsOob([&]{ fixed_out(b, outBuf, 3); })) return 11;
    }
    // Growable 1-D locals reject <=0, fractional, NaN, Inf, while allowing integer >len
    if (call(2, 4.0) != 9.0) return 12;
    for (double b : {0.0, -2.0, 2.25, nanv, infv, -infv}) {
        if (!throwsOob([&]{ return call(2, b); })) return 28;
        if (!throwsOob([&]{ return call(3, b); })) return 13;
    }
    // 2-D and 3-D indexing
    for (double b : {0.0, -1.0, 1.2, nanv, infv, 3.0}) {
        if (!throwsOob([&]{ return call(4, b, 1.0); })) return 14;
        if (!throwsOob([&]{ return call(5, b); })) return 15;
    }
    if (!throwsOob([&]{ return call(4, 1.0, 4.0); })) return 16;

    // Slice reads & writes
    for (double b : {0.0, -1.0, 1.5, nanv, infv, 4.0}) {
        if (!throwsOob([&]{ return call(6, b); })) return 17;
        if (!throwsOob([&]{ return call(7, b); })) return 18;
    }
    for (double b : {0.0, -1.0, 1.5, nanv, infv, 3.0}) {
        if (!throwsOob([&]{ return call(8, b); })) return 19;
    }
    // Inverted / OOB / fractional 2-D and 1-D slices throw std::out_of_range.
    if (!throwsOob([&]{ return call(9, 3.0, 2.0); })) return 20;
    if (!throwsOob([&]{ return call(9, 0.0, 2.0); })) return 21;
    if (!throwsOob([&]{ return call(9, 1.0, 4.0); })) return 22;
    if (!throwsOob([&]{ return call(9, 1.5, 2.5); })) return 23;
    if (!throwsOob([&]{ return call(10, 0.0, 1.0); })) return 24;
    if (!throwsOob([&]{ return call(10, 3.0, 5.0); })) return 25;
    if (call(10, 2.0, 3.0) != (10.0 + 7.0 + 8.0 + 40.0)) return 26;
    return 0;
}
)CPP";

    const std::string exe = (wave5TmpDir() / "wave5_bounds").string();
    const auto        cr  = aot::compileToExecutable(program, exe);
    ASSERT_EQ(cr.status, aot::CompileStatus::Ok) << cr.log << "\n" << program;
    EXPECT_EQ(aot::runExecutable(exe).exitCode, 0);
}

// ── 2. Call-site input/output array aliasing safety ──────────────────────
TEST(Wave5CodegenSafety, CallInputOutputArrayAliasingSafeOnRealloc)
{
    if (!aot::available())
        GTEST_SKIP() << "no external compiler configured for this build";

    const char *src =
        "function [s, out] = resize_and_sum(x)\n"
        "  n = numel(x);\n"
        "  out = zeros(1, 8192);\n"
        "  s = 0;\n"
        "  for k = 1:n\n"
        "    s = s + x(k);\n"
        "    out(k) = x(k) * 2;\n"
        "  end\n"
        "end\n"
        "function sout = mut_struct(sin)\n"
        "  n = numel(sin.v);\n"
        "  sout.v = zeros(1, 8192);\n"
        "  sout.tot = 0;\n"
        "  for k = 1:n\n"
        "    sout.tot = sout.tot + sin.v(k);\n"
        "    sout.v(k) = sin.v(k) + 1;\n"
        "  end\n"
        "end\n"
        "function res = caller(seed)\n"
        "  a = [seed, seed * 2, seed * 3, seed * 4];\n"
        "  [s1, a] = resize_and_sum(a);\n"
        "  st.v = [seed, seed * 2, seed * 3, seed * 4];\n"
        "  st.tot = 0;\n"
        "  st = mut_struct(st);\n"
        "  res = s1 + a(1) + a(4) + st.tot + st.v(1) + st.v(4) + numel(a) + numel(st.v);\n"
        "end\n";

    TransferRegistry reg;
    registerStandardTransfers(reg);
    auto          root = parseSrc(src);
    FunctionTable table;
    collectFunctions(*root, table);
    registerUserFunctions(reg, table);

    const numkit::ASTNode *callerFn = table.find("caller");
    ASSERT_NE(callerFn, nullptr);
    const EmittedFunction ef =
        emitProgram(*callerFn, {{"seed", InferredType::scalar(ValueType::DOUBLE)}}, table, reg);

    // seed = 10 -> a_in = [10, 20, 30, 40]:
    //   s1 = 100, a(1) = 20, a(4) = 80, numel(a) = 8192
    //   st.tot = 100, st.v(1) = 11, st.v(4) = 41, numel(st.v) = 8192
    //   total = 100 + 20 + 80 + 100 + 11 + 41 + 8192 + 8192 = 16736
    std::string program = ef.source + "\nint main() {\n    const double got = " + ef.name
                        + "(10.0);\n    return (got == 16736.0) ? 0 : 1;\n}\n";

    const std::string exe = (wave5TmpDir() / "wave5_alias").string();
    const auto        cr  = aot::compileToExecutable(program, exe);
    ASSERT_EQ(cr.status, aot::CompileStatus::Ok) << cr.log << "\n" << program;
    EXPECT_EQ(aot::runExecutable(exe).exitCode, 0);
}

// ── 3. Logical array interproc & multi-output return ─────────────────────
TEST(Wave5CodegenSafety, LogicalArrayInterprocAndMultiOutputReturn)
{
    if (!aot::available())
        GTEST_SKIP() << "no external compiler configured for this build";

    const char *src =
        "function m = pos_mask(x)\n"
        "  m = x > 0;\n"
        "end\n"
        "function [m, cnt] = mask_and_count(x)\n"
        "  m = pos_mask(x);\n"
        "  cnt = 0;\n"
        "  for k = 1:numel(m)\n"
        "    if m(k)\n"
        "      cnt = cnt + 1;\n"
        "    end\n"
        "  end\n"
        "end\n"
        "function score = run_masks(x)\n"
        "  [m, cnt] = mask_and_count(x);\n"
        "  score = cnt * 100;\n"
        "  if m(2) && ~m(1) && m(4)\n"
        "    score = score + 7;\n"
        "  end\n"
        "end\n";

    TransferRegistry reg;
    registerStandardTransfers(reg);
    auto          root = parseSrc(src);
    FunctionTable table;
    collectFunctions(*root, table);
    registerUserFunctions(reg, table);

    const numkit::ASTNode *fn = table.find("run_masks");
    ASSERT_NE(fn, nullptr);
    const EmittedFunction ef = emitProgram(
        *fn, {{"x", InferredType::concrete(ValueType::DOUBLE, Shape::rowVector())}}, table, reg);

    std::string program = ef.source + "\nint main() {\n"
                          "    const double x[4] = {-3.0, 5.0, 0.0, 2.0};\n"
                          "    const double got = " + ef.name + "(x, 4);\n"
                          "    return (got == 207.0) ? 0 : 1;\n}\n";

    const std::string exe = (wave5TmpDir() / "wave5_logical_ret").string();
    const auto        cr  = aot::compileToExecutable(program, exe);
    ASSERT_EQ(cr.status, aot::CompileStatus::Ok) << cr.log << "\n" << program;
    EXPECT_EQ(aot::runExecutable(exe).exitCode, 0);
}

// ── 4. Empty-vector guard on inline min(v) / max(v) ──────────────────────
TEST(Wave5CodegenSafety, EmptyVectorMinMaxThrowsOutOfRange)
{
    if (!aot::available())
        GTEST_SKIP() << "no external compiler configured for this build";

    const char *src =
        "function r = stmt_min(x, thresh)\n"
        "  v = x(x > thresh);\n"
        "  r = min(v);\n"
        "end\n"
        "function r = stmt_max(x, thresh)\n"
        "  v = x(x > thresh);\n"
        "  r = max(v);\n"
        "end\n"
        "function r = expr_min(x, thresh)\n"
        "  v = x(x > thresh);\n"
        "  r = 10 + min(v);\n"
        "end\n"
        "function r = expr_max(x, thresh)\n"
        "  v = x(x > thresh);\n"
        "  r = 10 + max(v);\n"
        "end\n"
        "function s = dispatch_minmax(mode, x, thresh)\n"
        "  if mode == 1\n"
        "    s = stmt_min(x, thresh);\n"
        "  elseif mode == 2\n"
        "    s = stmt_max(x, thresh);\n"
        "  elseif mode == 3\n"
        "    s = expr_min(x, thresh);\n"
        "  elseif mode == 4\n"
        "    s = expr_max(x, thresh);\n"
        "  else\n"
        "    s = stmt_min(x, thresh) + stmt_max(x, thresh) + expr_min(x, thresh) + expr_max(x, thresh);\n"
        "  end\n"
        "end\n";

    TransferRegistry reg;
    registerStandardTransfers(reg);
    auto          root = parseSrc(src);
    FunctionTable table;
    collectFunctions(*root, table);
    registerUserFunctions(reg, table);

    const EmittedFunction ef = emitProgram(
        *table.find("dispatch_minmax"),
        {{"mode", InferredType::scalar(ValueType::DOUBLE)},
         {"x", InferredType::concrete(ValueType::DOUBLE, Shape::rowVector())},
         {"thresh", InferredType::scalar(ValueType::DOUBLE)}},
        table, reg);

    std::string program = ef.source + R"CPP(
#include <stdexcept>

template <typename Fn>
bool throwsOutOfRange(Fn&& fn) {
    try {
        fn();
        return false;
    } catch (const std::out_of_range&) {
        return true;
    }
}

int main() {
    const double x[3] = {2.0, 5.0, 9.0};
    auto call = [&](double mode, double thresh) {
        return )CPP" + ef.name + R"CPP((mode, x, 3, thresh);
    };
    // Non-empty filtered vector [5, 9]: min=5, max=9, 10+min=15, 10+max=19 -> sum = 48.
    if (call(0.0, 3.0) != 48.0) return 1;

    // Empty filtered vector (thresh = 100): all four forms MUST throw std::out_of_range.
    if (!throwsOutOfRange([&]{ return call(1.0, 100.0); })) return 2;
    if (!throwsOutOfRange([&]{ return call(2.0, 100.0); })) return 3;
    if (!throwsOutOfRange([&]{ return call(3.0, 100.0); })) return 4;
    if (!throwsOutOfRange([&]{ return call(4.0, 100.0); })) return 5;
    return 0;
}
)CPP";

    const std::string exe = (wave5TmpDir() / "wave5_empty_minmax").string();
    const auto        cr  = aot::compileToExecutable(program, exe);
    ASSERT_EQ(cr.status, aot::CompileStatus::Ok) << cr.log << "\n" << program;
    EXPECT_EQ(aot::runExecutable(exe).exitCode, 0);
}

// ── 5. C++ reserved keywords as function, class, param, and local names ──
TEST(Wave5CodegenSafety, CppReservedKeywordsAsParamsLocalsFunctionsAndClasses)
{
    if (!aot::available())
        GTEST_SKIP() << "no external compiler configured for this build";

    const char *src =
        "function operator = namespace(class, default, and)\n"
        "  new = zeros(1, 3);\n"
        "  for template = 1:3\n"
        "    new(template) = class + default * template;\n"
        "  end\n"
        "  delete = sum(new) + and;\n"
        "  private = new(2);\n"
        "  operator = delete + private;\n"
        "end\n";

    TransferRegistry reg;
    registerStandardTransfers(reg);
    auto                   root = parseSrc(src);
    const numkit::ASTNode *fn   = firstFunc(*root);
    ASSERT_NE(fn, nullptr);

    ClassRegistry classes;
    ClassInfo     ci;
    ci.name   = "register";
    ci.fields = {{"virtual", InferredType::scalar(ValueType::DOUBLE), nullptr},
                 {"explicit", InferredType::scalar(ValueType::DOUBLE), nullptr}};
    classes.add(ci);

    const EmittedFunction ef = emitFunction(
        *fn,
        {{"class", InferredType::scalar(ValueType::DOUBLE)},
         {"default", InferredType::scalar(ValueType::DOUBLE)},
         {"and", InferredType::scalar(ValueType::DOUBLE)}},
        reg, &classes);

    EXPECT_EQ(ef.name, "numkit_kw_namespace");

    // class = 10, default = 2, and = 5:
    //   new = [12, 14, 16], sum(new) = 42, delete = 47, private = 14 -> operator = 61
    std::string program = ef.source + R"CPP(
int main() {
    numkit_kw_register obj{};
    obj._numkit_kw_virtual = 3.0;
    obj._numkit_kw_explicit = 4.0;
    const double r = numkit_kw_namespace(10.0, 2.0, 5.0)
                   + obj._numkit_kw_virtual + obj._numkit_kw_explicit;
    return (r == 68.0) ? 0 : 1;
}
)CPP";

    const std::string exe = (wave5TmpDir() / "wave5_keywords").string();
    const auto        cr  = aot::compileToExecutable(program, exe);
    ASSERT_EQ(cr.status, aot::CompileStatus::Ok) << cr.log << "\n" << program;
    EXPECT_EQ(aot::runExecutable(exe).exitCode, 0);
}

// ── 6. Vector1D vs rank-unknown ShapeKind::Unknown branch merge ──────────
TEST(Wave5CodegenSafety, RankUnknownMatrixBranchMergeRefusesWithoutBridgeAndRunsBridged)
{
    const Shape v1d = joinShape(Shape::rowVector(), Shape::colVector());
    EXPECT_EQ(v1d.kind, ShapeKind::Vector1D);

    const Shape unk = joinShape(Shape::dims(2, 3), Shape::dims(4, 5));
    EXPECT_EQ(unk.kind, ShapeKind::Unknown);

    const char *src =
        "function y = branch_sum(flag)\n"
        "  if flag > 0\n"
        "    A = ones(2, 3);\n"
        "  else\n"
        "    A = ones(4, 5) * 2;\n"
        "  end\n"
        "  y = sum(A);\n"
        "end\n";

    TransferRegistry reg;
    registerStandardTransfers(reg);
    auto                   root = parseSrc(src);
    const numkit::ASTNode *fn   = firstFunc(*root);
    ASSERT_NE(fn, nullptr);

    const std::vector<ParamSpec> params = {{"flag", InferredType::scalar(ValueType::DOUBLE)}};

    // Without bridge: MUST refuse cleanly instead of mis-typing A as a 1-D vector buffer!
    EXPECT_THROW(emitFunction(*fn, params, reg, nullptr, BridgeOptions{}), std::runtime_error);

    if (!aot::available())
        GTEST_SKIP() << "no external compiler configured for bridged execution";

    // With bridge: lowers via Dynamic tier and produces exact 1x3 [2,2,2] and 1x5 [8,8,8,8,8].
    BridgeOptions bridgeOpts;
    bridgeOpts.enabled       = true;
    bridgeOpts.runtimeHeader = "numkit_codegen_rt.h";
    const EmittedFunction ef = emitFunction(*fn, params, reg, nullptr, bridgeOpts);

    std::string program = ef.source + R"CPP(
int main() {
    const long long live0 = numkit_debug_live_handles();
    numkit_val r1 = branch_sum(1.0);
    if (!r1 || numkit_numel(r1) != 3) return 1;
    double b1[3] = {};
    numkit_unbox_array(r1, b1, 3);
    numkit_release(r1);
    for (int i = 0; i < 3; ++i)
        if (b1[i] != 2.0) return 2;

    numkit_val r0 = branch_sum(0.0);
    if (!r0 || numkit_numel(r0) != 5) return 3;
    double b0[5] = {};
    numkit_unbox_array(r0, b0, 5);
    numkit_release(r0);
    for (int i = 0; i < 5; ++i)
        if (b0[i] != 8.0) return 4;

    if (numkit_debug_live_handles() != live0) return 5;
    return 0;
}
)CPP";

    const std::string exe = (wave5TmpDir() / "wave5_rank_unknown_bridged").string();
    aot::CompileOptions opts;
    opts.includeDirs = {NUMKIT_BRIDGE_DIR};
    opts.defines     = {"NUMKIT_RT_USE_DLL"};
    opts.linkLibs    = {NUMKIT_RT_IMPORT_LIB};
    const auto cr = aot::compileToExecutable(program, exe, opts);
    ASSERT_EQ(cr.status, aot::CompileStatus::Ok) << cr.log << "\n" << program;
    EXPECT_EQ(aot::runExecutable(exe).exitCode, 0);
}

// ── 7. Unconverged loop fixpoint (>16 chain) widens to Dynamic ───────────
TEST(Wave5CodegenSafety, UnconvergedLoopFixpointWidensToDynamicInsteadOfMiscompilation)
{
    std::ostringstream oss;
    oss << "function y = shift18(seed)\n";
    oss << "  v0 = [seed, seed + 1, seed + 2];\n";
    for (int i = 1; i <= 17; ++i)
        oss << "  v" << i << " = 0;\n";
    oss << "  for k = 1:20\n";
    for (int i = 17; i >= 1; --i)
        oss << "    v" << i << " = v" << (i - 1) << ";\n";
    oss << "  end\n";
    oss << "  y = numel(v17);\n";
    oss << "end\n";
    const std::string src = oss.str();

    TransferRegistry reg;
    registerStandardTransfers(reg);
    auto                   root = parseSrc(src);
    const numkit::ASTNode *fn   = firstFunc(*root);
    ASSERT_NE(fn, nullptr);

    // 1. Verify inference widens v17 to Dynamic (NOT Concrete scalar!).
    TypeEnv env;
    env.set("seed", {InferredType::scalar(ValueType::DOUBLE), ConstVal::unknown()});
    inferStmt(*fn->children[0], env, reg);
    ASSERT_TRUE(env.has("v17"));
    EXPECT_TRUE(env.get("v17").type.isDynamic()) << "actual type: " << env.get("v17").type.str();

    const std::vector<ParamSpec> params = {{"seed", InferredType::scalar(ValueType::DOUBLE)}};

    // 2. Standalone emitFunction MUST refuse cleanly rather than miscompiling y = 1.0!
    EXPECT_THROW(emitFunction(*fn, params, reg, nullptr, BridgeOptions{}), std::runtime_error);

    if (!aot::available())
        GTEST_SKIP() << "no external compiler configured for bridged execution";

    // 3. Bridged emitFunction executes via Dynamic tier and returns exact 3.0 (matching engine).
    numkit::StandardEngine engine;
    engine.eval(src, true);
    const double expected = engine.eval("shift18(5)", true).toScalar();
    ASSERT_DOUBLE_EQ(expected, 3.0);

    BridgeOptions bridgeOpts;
    bridgeOpts.enabled       = true;
    bridgeOpts.runtimeHeader = "numkit_codegen_rt.h";
    const EmittedFunction ef = emitFunction(*fn, params, reg, nullptr, bridgeOpts);

    std::string program = ef.source + R"CPP(
int main() {
    const long long live0 = numkit_debug_live_handles();
    const double got = shift18(5.0);
    if (numkit_debug_live_handles() != live0) return 3;
    return (got == 3.0) ? 0 : 2;
}
)CPP";

    const std::string exe = (wave5TmpDir() / "wave5_loop_widen").string();
    aot::CompileOptions opts;
    opts.includeDirs = {NUMKIT_BRIDGE_DIR};
    opts.defines     = {"NUMKIT_RT_USE_DLL"};
    opts.linkLibs    = {NUMKIT_RT_IMPORT_LIB};
    const auto cr = aot::compileToExecutable(program, exe, opts);
    ASSERT_EQ(cr.status, aot::CompileStatus::Ok) << cr.log << "\n" << program;
    EXPECT_EQ(aot::runExecutable(exe).exitCode, 0);
}

// ── 8. AOT shell quoting, WEXITSTATUS decoding & define validation ───────
TEST(Wave5CodegenSafety, AotShellQuotingWaitStatusAndDefineValidation)
{
    if (!aot::available())
        GTEST_SKIP() << "no external compiler configured for this build";

    {
        aot::CompileOptions badOpts;
        badOpts.defines = {"SAFE=1", "INJECT=1\necho pwned"};
        const std::string exe = (wave5TmpDir() / "wave5_never_built").string();
        const auto        r   = aot::compileToExecutable("int main(){return 0;}\n", exe, badOpts);
        EXPECT_EQ(r.status, aot::CompileStatus::CompileError);
        EXPECT_FALSE(std::filesystem::exists(exe));
    }
    {
        aot::CompileOptions badOpts;
        badOpts.defines = {"123BAD"};
        const std::string exe = (wave5TmpDir() / "wave5_never_built2").string();
        const auto        r   = aot::compileToExecutable("int main(){return 0;}\n", exe, badOpts);
        EXPECT_EQ(r.status, aot::CompileStatus::CompileError);
    }

    const auto trickyDir = wave5TmpDir() / "dir with space and 'quote'";
    std::filesystem::create_directories(trickyDir);
    const std::string exe = (trickyDir / "ret42 bin").string();

    aot::CompileOptions okOpts;
    okOpts.defines = {"RET_CODE=42"};
    const auto cr = aot::compileToExecutable(
        "int main() { return RET_CODE; }\n", exe, okOpts);
    ASSERT_EQ(cr.status, aot::CompileStatus::Ok) << cr.log << "\ncmd: " << cr.command;
    EXPECT_EQ(aot::runExecutable(exe).exitCode, 42);
}
