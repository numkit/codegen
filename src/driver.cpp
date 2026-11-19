// codegen/src/driver.cpp — see driver.hpp.

#include <numkit/codegen/driver.hpp>

#include <numkit/codegen/classinfo.hpp>
#include <numkit/codegen/monomorphize.hpp>
#include <numkit/codegen/transfer.hpp>

#include <numkit/core/ast.hpp>
#include <numkit/core/lexer.hpp>
#include <numkit/core/parser.hpp>

#include <keel/format.h>
#include <keel/smart_ptr.h>

#include <cctype>
#include <stdexcept>

namespace numkit::codegen::driver {

namespace {

keel::StringRef trim(keel::StringRef s)
{
    keel::usize a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

ValueType dtypeFromString(keel::StringRef name)
{
    if (name == "double")  return ValueType::DOUBLE;
    if (name == "single")  return ValueType::SINGLE;
    if (name == "complex") return ValueType::COMPLEX;
    if (name == "logical") return ValueType::LOGICAL;
    if (name == "int8")    return ValueType::INT8;
    if (name == "int16")   return ValueType::INT16;
    if (name == "int32")   return ValueType::INT32;
    if (name == "int64")   return ValueType::INT64;
    if (name == "uint8")   return ValueType::UINT8;
    if (name == "uint16")  return ValueType::UINT16;
    if (name == "uint32")  return ValueType::UINT32;
    if (name == "uint64")  return ValueType::UINT64;
    throw std::runtime_error(keel::format("numkit build: unknown argument type '{}'", name).c_str());
}

} // namespace

keel::Array<InferredType> parseTypeSpec(keel::StringRef spec)
{
    keel::Array<InferredType> out;
    const keel::StringRef     s = trim(spec);
    if (s.empty()) return out;  // nullary entry

    keel::usize pos = 0;
    while (pos <= s.size()) {
        const keel::usize comma = s.find(',', pos);
        keel::StringRef   tok   = trim(s.substr(pos, comma == keel::kInvalidIndex
                                                         ? keel::kInvalidIndex
                                                         : comma - pos));
        if (tok.empty())
            throw std::runtime_error("numkit build: empty argument type in spec");

        bool isArray = false;
        if (tok.size() >= 2 && tok.substr(tok.size() - 2) == "[]") {
            isArray = true;
            tok     = trim(tok.substr(0, tok.size() - 2));
        }
        const ValueType dt = dtypeFromString(tok);
        out.push_back(isArray ? InferredType::concrete(dt, Shape::rowVector())
                              : InferredType::scalar(dt));

        if (comma == keel::kInvalidIndex) break;
        pos = comma + 1;
    }
    return out;
}

namespace {

// Parse `source`, register transfers/user-functions/classes, resolve the entry
// (named or sole), arity-check, build the ParamSpec list, then invoke `emit`
// WHILE everything is still alive. Nothing borrowing escapes — the registry's
// user-function transfers reference these locals, so emission must run inside
// this scope (shared by both transpileSource and transpileToPlugin).
template <class Emit>
auto withPrepared(keel::StringRef source, keel::StringRef entry,
                  keel::ArrayRef<InferredType> paramTypes, Emit &&emit)
{
    Lexer  lex(source);
    Parser parser(lex.tokenize());
    auto   root = parser.parse();
    if (!root) throw std::runtime_error("numkit build: failed to parse source");

    TransferRegistry reg;
    registerStandardTransfers(reg);
    ClassRegistry classes;
    collectClasses(*root, classes, reg);
    FunctionTable table;
    collectFunctions(*root, table);
    registerUserFunctions(reg, table);
    if (classes.size() > 0) {
        registerClassMethods(reg, classes);
        registerClassConstructors(reg, classes);
    }
    const ClassRegistry *classesPtr = classes.size() > 0 ? &classes : nullptr;

    ASTNode syntheticScriptFn;
    if (entry.empty() && root->type == NodeType::BLOCK) {
        auto scriptBody = keel::make_unique<ASTNode>();
        scriptBody->type = NodeType::BLOCK;
        for (auto &c : root->children) {
            if (c && c->type != NodeType::FUNCTION_DEF && c->type != NodeType::CLASSDEF_DEF) {
                scriptBody->children.push_back(keel::move(c));
            }
        }
        if (!scriptBody->children.empty()) {
            syntheticScriptFn.type = NodeType::FUNCTION_DEF;
            syntheticScriptFn.strValue = keel::String::from("numkit_script_main");
            syntheticScriptFn.children.push_back(keel::move(scriptBody));
        }
    }

    const ASTNode *fn = nullptr;
    if (!entry.empty()) {
        fn = table.find(entry);
        if (!fn)
            throw std::runtime_error(
                keel::format("numkit build: no function named '{}'", entry).c_str());
    } else if (!syntheticScriptFn.children.empty()) {
        fn = &syntheticScriptFn;
    } else if (table.size() == 1) {
        for (auto [k, v] : table.entries()) {
            (void)k;
            fn = v;
            break;
        }
    } else {
        throw std::runtime_error(
            keel::format("numkit build: source has {} functions — specify the entry with --entry",
                         table.size()).c_str());
    }

    if (paramTypes.size() != fn->paramNames.size())
        throw std::runtime_error(
            keel::format("numkit build: entry '{}' takes {} argument(s) but the type spec has {}",
                         fn->strValue, fn->paramNames.size(), paramTypes.size()).c_str());

    keel::Array<ParamSpec> params;
    params.reserve(paramTypes.size());
    for (keel::usize i = 0; i < paramTypes.size(); ++i)
        params.push_back(ParamSpec(fn->paramNames[i], paramTypes[i]));

    return emit(*fn, params, reg, table, classesPtr);
}

} // namespace

EmittedFunction transpileSource(keel::StringRef source, keel::StringRef entry,
                                keel::ArrayRef<InferredType> paramTypes,
                                const BridgeOptions &bridge,
                                const OpsKernelOptions &opsKernels)
{
    return withPrepared(source, entry, paramTypes,
                        [&](const ASTNode &fn, keel::ArrayRef<ParamSpec> params,
                            const TransferRegistry &reg, const FunctionTable &table,
                            const ClassRegistry *classes) {
                            return emitProgram(fn, params, table, reg, classes, bridge, opsKernels);
                        });
}

keel::String transpileToPlugin(keel::StringRef source, keel::StringRef entry,
                               keel::ArrayRef<InferredType> paramTypes,
                               keel::StringRef exportName, keel::StringRef abiHeaderPath)
{
    return withPrepared(source, entry, paramTypes,
                        [&](const ASTNode &fn, keel::ArrayRef<ParamSpec> params,
                            const TransferRegistry &reg, const FunctionTable &,
                            const ClassRegistry *classes) {
                            return emitScalarPlugin(fn, params, reg, exportName, abiHeaderPath,
                                                    classes);
                        });
}

// Parse the return type out of an EmittedFunction signature of the form
// `<retty> <name>(<params>)`. The emitter uses fixed-width / single-token
// types (double, float, std::complex<double>, bool, int8_t..int64_t,
// uint8_t..uint64_t, void) — none contain spaces — so the return type is the
// substring before the last space that precedes the first `(`.
namespace {
keel::StringRef parseRetType(keel::StringRef signature)
{
    const keel::usize paren = signature.find('(');
    if (paren == keel::kInvalidIndex) return {};
    const keel::StringRef head = signature.substr(0, paren);  // "<retty> <name>"
    const keel::usize     sp   = head.rfind(' ');
    if (sp == keel::kInvalidIndex) return {};
    return head.substr(0, sp);
}
} // namespace

keel::String buildHarnessMain(const EmittedFunction &em)
{
    const keel::StringRef retty = parseRetType(em.signature);
    keel::String          body;
    if (retty == "void") {
        body = keel::format("    {}();\n", em.name);
    } else if (retty == "double" || retty == "float" || retty == "bool" ||
               retty == "int8_t" || retty == "int16_t" || retty == "int32_t" ||
               retty == "int64_t" || retty == "uint8_t" || retty == "uint16_t" ||
               retty == "uint32_t" || retty == "uint64_t") {
        body = keel::format("    {} _r = {}();\n"
                            "    std::printf(\"%.17g\\n\", static_cast<double>(_r));\n",
                            retty, em.name);
    } else if (retty == "std::complex<double>") {
        body = keel::format("    std::complex<double> _r = {}();\n"
                            "    std::printf(\"%.17g%+.17gi\\n\", _r.real(), _r.imag());\n",
                            em.name);
    } else {
        throw std::runtime_error(
            keel::format("numkit build: cannot build a run harness for return type '{}' "
                         "(v1 supports only scalar / complex / void returns). Use `coder` / "
                         "`-o` to get the C++ and write your own main().",
                         retty).c_str());
    }
    return keel::format("#include <cstdio>\n"
                        "#include <complex>\n"
                        "int main() {\n"
                        "{}"
                        "    return 0;\n"
                        "}\n",
                        body);
}

} // namespace numkit::codegen::driver
