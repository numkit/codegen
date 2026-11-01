// codegen/tests/inference_test.cpp
//
// End-to-end inference over REAL parsed code: lex+parse a snippet, run
// the straight-line inference driver, and assert the inferred type of a
// variable. This is where the lattice + ConstVal + transfer registry
// come together into actual type inference — including the two wins that
// matter: constant propagation into shape (n=5 -> zeros(n) is 5x5) and
// scalar element access (x(i) is an unboxed scalar).

#include <numkit/codegen/classinfo.hpp>
#include <numkit/codegen/inference.hpp>
#include <numkit/codegen/monomorphize.hpp>
#include <numkit/codegen/transfer.hpp>

#include <numkit/core/lexer.hpp>
#include <numkit/core/parser.hpp>

#include <gtest/gtest.h>

using numkit::ValueType;
using namespace numkit::codegen;

namespace {

// Lex + parse + infer a snippet; return the final environment.
TypeEnv inferSrc(const std::string &src)
{
    numkit::Lexer  lex(src);
    auto           tokens = lex.tokenize();
    numkit::Parser parser(tokens);
    auto           root = parser.parse();

    TransferRegistry reg;
    registerStandardTransfers(reg);
    return inferProgram(*root, reg);
}

}  // namespace

// A bare scalar literal binding: scalar double, unboxable, const known.
TEST(Inference, ScalarLiteral)
{
    const auto env = inferSrc("a = 3.0;");
    const auto a = env.get("a");
    EXPECT_TRUE(a.type.isUnboxableScalar());
    EXPECT_EQ(a.type.dtype, ValueType::DOUBLE);
    EXPECT_TRUE(a.constant.isKnown());
    EXPECT_DOUBLE_EQ(a.constant.value, 3.0);
}

// linspace with literal length -> double 1 x 64, inferred through the
// transfer registry on real parsed code.
TEST(Inference, LinspaceLiteral)
{
    const auto env = inferSrc("y = linspace(0, 1, 64);");
    const auto y = env.get("y");
    ASSERT_TRUE(y.type.isConcrete());
    EXPECT_EQ(y.type.dtype, ValueType::DOUBLE);
    EXPECT_EQ(y.type.shape.kind, ShapeKind::KnownDims);
    EXPECT_EQ(y.type.shape.rows, 1u);
    EXPECT_EQ(y.type.shape.cols, 64u);
}

// THE killer demo: a literal constant flows through a variable binding
// into a size constructor's shape. n=5 ; zeros(n) -> 5x5.
TEST(Inference, ConstantPropagatesIntoShape)
{
    const auto env = inferSrc("n = 5; z = zeros(n);");
    const auto z = env.get("z");
    ASSERT_TRUE(z.type.isConcrete());
    EXPECT_EQ(z.type.dtype, ValueType::DOUBLE);
    EXPECT_EQ(z.type.shape.kind, ShapeKind::KnownDims);
    EXPECT_EQ(z.type.shape.rows, 5u);
    EXPECT_EQ(z.type.shape.cols, 5u);
}

// Scalar element access: x(3) on a double array is an unboxed scalar —
// the hot x(i)-in-a-loop case.
TEST(Inference, ScalarIndexIsUnboxableScalar)
{
    const auto env = inferSrc("x = zeros(1, 10); v = x(3);");
    const auto v = env.get("v");
    ASSERT_TRUE(v.type.isConcrete());
    EXPECT_TRUE(v.type.isUnboxableScalar());
    EXPECT_EQ(v.type.dtype, ValueType::DOUBLE);
}

// A use of an unbound variable is conservatively Dynamic.
TEST(Inference, UnknownVariableIsDynamic)
{
    const auto env = inferSrc("q = nope;");
    EXPECT_TRUE(env.get("q").type.isDynamic());
}

// A variable assigned only inside an `if` (no else, not defined before)
// may be undefined on the fall-through path -> Dynamic at the merge.
TEST(Inference, IfAssignedVarWithoutElseIsMaybeUndefined)
{
    const auto env = inferSrc("if true\n w = 5;\n end");
    EXPECT_TRUE(env.get("w").type.isDynamic());
}

// if/else that defines the variable on BOTH branches with the same type
// joins to that precise type.
TEST(Inference, IfElseBothBranchesJoin)
{
    const auto env = inferSrc("if c\n a = 5;\n else\n a = 10;\n end");
    const auto a = env.get("a");
    EXPECT_TRUE(a.type.isUnboxableScalar());
    EXPECT_EQ(a.type.dtype, ValueType::DOUBLE);
}

// A variable defined before the `if` and conditionally reassigned with
// the same type stays precisely typed (branch joins fall-through).
TEST(Inference, IfWithPriorDefinitionStaysTyped)
{
    const auto env = inferSrc("a = 1; if c\n a = 2;\n end");
    EXPECT_TRUE(env.get("a").type.isUnboxableScalar());
}

// Type-unstable across branches (double vs char) -> Dynamic.
TEST(Inference, TypeUnstableAcrossBranchesIsDynamic)
{
    const auto env = inferSrc("a = 1; if c\n a = 'x';\n end");
    EXPECT_TRUE(env.get("a").type.isDynamic());
}

// for n = 1:10 -> n is a scalar double (a colon range iterates scalars).
TEST(Inference, ForLoopVariableIsScalarDouble)
{
    const auto env = inferSrc("for n = 1:10\n end");
    const auto n = env.get("n");
    EXPECT_TRUE(n.type.isUnboxableScalar());
    EXPECT_EQ(n.type.dtype, ValueType::DOUBLE);
}

// while-loop fixpoint: loop-carried scalars stay precisely typed.
TEST(Inference, WhileFixpointTypesLoopCarried)
{
    const auto env = inferSrc("s = 0; k = 1; while k < 10\n s = s + 1; k = k + 1;\n end");
    EXPECT_TRUE(env.get("s").type.isUnboxableScalar());
    EXPECT_TRUE(env.get("k").type.isUnboxableScalar());
}

// M1 capstone — the whole biquad loop types: the loop variable is a
// scalar double, the loop-carried filter state stays a scalar double
// (fixpoint), and the output stays a double array. This is the loop M0
// measured; with this, the transpiler can emit it fully unboxed.
TEST(Inference, BiquadLoopFullyTyped)
{
    const auto env = inferSrc(
        "N = 64; x = zeros(1, N); y = zeros(1, N);"
        "b0=0.0675; b1=0.1349; b2=0.0675; a1=-1.1430; a2=0.4128;"
        "x1=0; x2=0; y1=0; y2=0;"
        "for n = 1:N\n"
        "  xn = x(n);\n"
        "  yn = b0*xn + b1*x1 + b2*x2 - a1*y1 - a2*y2;\n"
        "  y(n) = yn;\n"
        "  x2 = x1; x1 = xn; y2 = y1; y1 = yn;\n"
        "end");

    // loop variable
    EXPECT_TRUE(env.get("n").type.isUnboxableScalar());
    // loop-carried state survives the fixpoint as a scalar double
    EXPECT_TRUE(env.get("x1").type.isUnboxableScalar());
    EXPECT_TRUE(env.get("y1").type.isUnboxableScalar());
    // the output is a typed double array (shape unknown after indexed writes)
    const auto y = env.get("y");
    ASSERT_TRUE(y.type.isConcrete());
    EXPECT_EQ(y.type.dtype, ValueType::DOUBLE);
}

// Scalar arithmetic now types through the elementwise family: a + 1 is
// an unboxed scalar double.
TEST(Inference, ScalarArithmeticIsTypedDouble)
{
    const auto env = inferSrc("a = 3; b = a + 1;");
    EXPECT_TRUE(env.get("a").type.isUnboxableScalar());
    const auto b = env.get("b");
    EXPECT_TRUE(b.type.isUnboxableScalar());
    EXPECT_EQ(b.type.dtype, ValueType::DOUBLE);
}

// The biquad inner expression: every operand is a scalar double, so the
// whole Direct-Form-I update types to an unboxed scalar double — the
// case M0 measured at ~97x the VM.
TEST(Inference, BiquadBodyIsUnboxableScalar)
{
    const auto env = inferSrc(
        "b0=0.0675; b1=0.1349; b2=0.0675; a1=-1.1430; a2=0.4128;"
        "xn=0.5; x1=0; x2=0; y1=0; y2=0;"
        "yn = b0*xn + b1*x1 + b2*x2 - a1*y1 - a2*y2;");
    const auto yn = env.get("yn");
    EXPECT_TRUE(yn.type.isUnboxableScalar());
    EXPECT_EQ(yn.type.dtype, ValueType::DOUBLE);
}

// A comparison yields a logical scalar.
TEST(Inference, ComparisonIsLogical)
{
    const auto env = inferSrc("t = 2 < 3;");
    const auto t = env.get("t");
    ASSERT_TRUE(t.type.isConcrete());
    EXPECT_EQ(t.type.dtype, ValueType::LOGICAL);
    EXPECT_TRUE(t.type.shape.isScalar());
}

// Elementwise math on a typed array preserves dtype and shape:
// sin(zeros(1,8)) is a double 1x8.
TEST(Inference, ElementwiseMathPreservesArrayShape)
{
    const auto env = inferSrc("x = zeros(1, 8); s = sin(x);");
    const auto s = env.get("s");
    ASSERT_TRUE(s.type.isConcrete());
    EXPECT_EQ(s.type.dtype, ValueType::DOUBLE);
    EXPECT_EQ(s.type.shape.kind, ShapeKind::KnownDims);
    EXPECT_EQ(s.type.shape.rows, 1u);
    EXPECT_EQ(s.type.shape.cols, 8u);
}

// Nested and non-scalar struct fields: local writes, reads, array element
// indexing, and whole-struct copies preserve nested struct layouts and
// 1-D / 2-D / N-D array shapes.
TEST(Inference, NestedAndArrayStructFields)
{
    const auto env = inferSrc(
        "s.scalar = 42;\n"
        "s.vec = linspace(0, 1, 8);\n"
        "s.mat = zeros(2, 3);\n"
        "s.inner.cube = ones(2, 3, 4);\n"
        "s.inner.deep.val = 7;\n"
        "elem1 = s.vec(3);\n"
        "elem2 = s.mat(1, 2);\n"
        "elem3 = s.inner.cube(1, 2, 3);\n"
        "deepv = s.inner.deep.val;\n"
        "copy = s;\n"
        "copy_elem = copy.inner.cube(2, 1, 1);\n");

    const auto sType = env.get("s").type;
    ASSERT_TRUE(sType.isStruct());
    ASSERT_NE(sType.structLayout, nullptr);
    EXPECT_TRUE(sType.structLayout->field("scalar") != nullptr);
    EXPECT_TRUE(sType.structLayout->field("vec") != nullptr);
    EXPECT_TRUE(sType.structLayout->field("mat") != nullptr);
    EXPECT_TRUE(sType.structLayout->field("inner") != nullptr);

    const InferredType *vecT = sType.structLayout->field("vec");
    ASSERT_NE(vecT, nullptr);
    EXPECT_EQ(vecT->shape.kind, ShapeKind::KnownDims);
    EXPECT_EQ(vecT->shape.rows, 1u);
    EXPECT_EQ(vecT->shape.cols, 8u);

    const InferredType *matT = sType.structLayout->field("mat");
    ASSERT_NE(matT, nullptr);
    EXPECT_EQ(matT->shape.kind, ShapeKind::KnownDims);
    EXPECT_EQ(matT->shape.rows, 2u);
    EXPECT_EQ(matT->shape.cols, 3u);

    const InferredType *innerT = sType.structLayout->field("inner");
    ASSERT_NE(innerT, nullptr);
    ASSERT_TRUE(innerT->isStruct());
    ASSERT_NE(innerT->structLayout, nullptr);
    const InferredType *cubeT = innerT->structLayout->field("cube");
    ASSERT_NE(cubeT, nullptr);
    EXPECT_TRUE(cubeT->shape.isNDims());
    ASSERT_EQ(cubeT->shape.nd.size(), 3u);
    EXPECT_EQ(cubeT->shape.nd[0], 2u);
    EXPECT_EQ(cubeT->shape.nd[1], 3u);
    EXPECT_EQ(cubeT->shape.nd[2], 4u);

    EXPECT_TRUE(env.get("elem1").type.isUnboxableScalar());
    EXPECT_TRUE(env.get("elem2").type.isUnboxableScalar());
    EXPECT_TRUE(env.get("elem3").type.isUnboxableScalar());
    EXPECT_TRUE(env.get("deepv").type.isUnboxableScalar());
    EXPECT_TRUE(env.get("copy").type.isStruct());
    EXPECT_TRUE(env.get("copy_elem").type.isUnboxableScalar());
}

// Multi-output user functions and class methods infer across multi-assign,
// single-LHS assignment, and expression contexts.
TEST(Inference, MultiOutputFunctionAndMethodContexts)
{
    const std::string src =
        "classdef Counter\n"
        "  properties\n"
        "    x = 0\n"
        "  end\n"
        "  methods\n"
        "    function obj = Counter(v)\n"
        "      obj.x = v;\n"
        "    end\n"
        "    function [obj, prev] = step(obj, dx)\n"
        "      prev = obj.x;\n"
        "      obj.x = obj.x + dx;\n"
        "    end\n"
        "    function [val, obj] = readAndStep(obj, dx)\n"
        "      val = obj.x;\n"
        "      obj.x = obj.x + dx;\n"
        "    end\n"
        "  end\n"
        "end\n"
        "function [s, d] = sumdiff(a, b)\n"
        "  s = a + b;\n"
        "  d = a - b;\n"
        "end\n"
        "function y = main()\n"
        "  [u, v] = sumdiff(5, 2);\n"
        "  first_fn = sumdiff(5, 2);\n"
        "  expr_fn = sumdiff(5, 2) + 10;\n"
        "  c = Counter(3);\n"
        "  [c, p] = c.step(4);\n"
        "  c2 = c.step(1);\n"
        "  expr_m = c.readAndStep(2) * 3;\n"
        "  y = u + v + first_fn + expr_fn + p + c2.x + expr_m;\n"
        "end\n";

    numkit::Lexer  lex(src);
    auto           tokens = lex.tokenize();
    numkit::Parser parser(tokens);
    auto           root = parser.parse();

    TransferRegistry reg;
    registerStandardTransfers(reg);
    ClassRegistry classes;
    collectClasses(*root, classes, reg);
    FunctionTable funcs;
    collectFunctions(*root, funcs);
    registerClassConstructors(reg, classes);
    registerClassMethods(reg, classes);
    registerUserFunctions(reg, funcs);

    const numkit::ASTNode *mainFn = funcs.find("main");
    ASSERT_NE(mainFn, nullptr);
    ASSERT_GE(mainFn->children.size(), 1u);

    TypeEnv env;
    inferStmt(*mainFn->children[0], env, reg, nullptr, &classes);

    EXPECT_TRUE(env.get("u").type.isUnboxableScalar());
    EXPECT_TRUE(env.get("v").type.isUnboxableScalar());
    EXPECT_TRUE(env.get("first_fn").type.isUnboxableScalar());
    EXPECT_TRUE(env.get("expr_fn").type.isUnboxableScalar());
    EXPECT_TRUE(env.get("c").type.isObject());
    EXPECT_TRUE(env.get("p").type.isUnboxableScalar());
    EXPECT_TRUE(env.get("c2").type.isObject());
    EXPECT_TRUE(env.get("expr_m").type.isUnboxableScalar());
    EXPECT_TRUE(env.get("y").type.isUnboxableScalar());
}

