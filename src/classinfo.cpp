// codegen/src/classinfo.cpp — see classinfo.hpp.

#include <numkit/codegen/classinfo.hpp>

#include <numkit/codegen/inference.hpp>

#include <keel/format.h>

#include <stdexcept>

namespace numkit::codegen {

const ClassField *ClassInfo::field(keel::StringRef n) const
{
    for (const auto &f : fields)
        if (f.name == n) return &f;
    return nullptr;
}

const ASTNode *ClassInfo::method(keel::StringRef n) const
{
    const ASTNode *const *p = methods.get(n);
    return p ? *p : nullptr;
}

int ClassRegistry::add(ClassInfo info)
{
    info.id = static_cast<int>(classes_.size());
    byName_.put(info.name.clone(), info.id);
    classes_.push_back(keel::move(info));
    return classes_.back().id;
}

const ClassInfo *ClassRegistry::byId(int id) const
{
    if (id < 0 || id >= static_cast<int>(classes_.size())) return nullptr;
    return &classes_[static_cast<keel::usize>(id)];
}

int ClassRegistry::idOf(keel::StringRef name) const
{
    const int *p = byName_.get(name);
    return p ? *p : -1;
}

const ClassInfo *ClassRegistry::byName(keel::StringRef name) const
{
    return byId(idOf(name));
}

keel::String classRefusalReason(const ASTNode &classDef)
{
    if (classDef.type != NodeType::CLASSDEF_DEF) return keel::String::from("not a classdef");

    // Inheritance beyond the `handle` mixin is out of the v1 subset.
    for (const auto &super : classDef.paramNames)
        if (super != "handle")
            return keel::format("inheritance from '{}' not supported (v1)", super);

    for (const auto &c : classDef.children) {
        if (!c) continue;
        if (c->type == NodeType::FUNCTION_DEF
            && c->strValue.find('.') != keel::kInvalidIndex)
            return keel::format("property accessor '{}' not supported (v1)", c->strValue);
        if (c->type == NodeType::CLASSDEF_PROPERTY)
            for (const auto &a : c->classAttrs)
                if (a == "Dependent" || a == "Constant" || a == "Abstract")
                    return keel::format("property attribute '{}' not supported (v1)", a);
        if (c->type == NodeType::CLASSDEF_ENUM_MEMBER)
            return keel::String::from("enumeration classes not supported (v1)");
    }
    return {};
}

ClassInfo buildClassInfo(const ASTNode &classDef, int id, const TransferRegistry &reg)
{
    if (const keel::String why = classRefusalReason(classDef); !why.empty())
        throw std::runtime_error(keel::format("class '{}': {}", classDef.strValue, why).c_str());

    ClassInfo ci;
    ci.name = classDef.strValue.clone();
    ci.id   = id;
    ci.def  = &classDef;
    for (const auto &super : classDef.paramNames)
        if (super == "handle") ci.isHandle = true;

    for (const auto &c : classDef.children) {
        if (!c) continue;
        if (c->type == NodeType::CLASSDEF_PROPERTY) {
            if (c->children.empty() || !c->children[0])
                throw std::runtime_error(
                    keel::format("class '{}': property '{}' needs a concrete-typed default (v1)",
                                 ci.name, c->strValue).c_str());
            TypeEnv            empty;
            const InferredType ft = inferExpr(*c->children[0], empty, reg).type;
            if (!ft.isConcrete())
                throw std::runtime_error(
                    keel::format("class '{}': property '{}' default is not concretely typed ({})",
                                 ci.name, c->strValue, ft.str()).c_str());
            ci.fields.push_back(ClassField(c->strValue, ft, c->children[0].get()));
        } else if (c->type == NodeType::FUNCTION_DEF) {
            ci.methods.put(c->strValue.clone(), c.get());
        }
    }
    return ci;
}

keel::usize collectClasses(const ASTNode &root, ClassRegistry &reg,
                           const TransferRegistry &reg2)
{
    keel::usize added = 0;
    if (root.type == NodeType::CLASSDEF_DEF) {
        reg.add(buildClassInfo(root, static_cast<int>(reg.size()), reg2));
        ++added;
    }
    for (const auto &c : root.children)
        if (c) added += collectClasses(*c, reg, reg2);
    return added;
}

} // namespace numkit::codegen
