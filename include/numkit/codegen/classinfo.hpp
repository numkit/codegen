// codegen/include/numkit/codegen/classinfo.hpp
//
// Class table (build plan §12, brick 4): parse a CLASSDEF_DEF into a
// ClassInfo (fields with inferred types, methods, value-vs-handle) and a
// ClassRegistry that hands out the classId carried by InferredType::object.
//
// v1 subset (DESIGN.md §7a): a value class (or a handle class via `<
// handle`), plain stored properties each with a concrete-typed default,
// and methods called monomorphically. Inheritance (beyond `handle`),
// property accessors (`get.`/`set.`), and Dependent/Constant/Abstract
// properties are REFUSED with a diagnostic — never miscompiled.

#pragma once

#include <numkit/codegen/transfer.hpp>
#include <numkit/codegen/type_lattice.hpp>

#include <numkit/core/ast.hpp>

#include <keel/array.h>
#include <keel/core.h>
#include <keel/hash_map.h>
#include <keel/ref.h>
#include <keel/string.h>

namespace numkit::codegen {

struct ClassField {
    keel::String   name;
    InferredType   type;                  // inferred from the default expression
    const ASTNode *defaultExpr = nullptr;  // the property's default value expression

    ClassField() noexcept = default;
    ClassField(keel::StringRef n, InferredType t, const ASTNode *def = nullptr)
        : name(keel::String::from(n)), type(keel::move(t)), defaultExpr(def)
    {}
    ClassField(const ClassField &o)
        : name(o.name.clone()), type(o.type.clone()), defaultExpr(o.defaultExpr)
    {}
    ClassField &operator=(const ClassField &o)
    {
        if (this != &o) {
            name        = o.name.clone();
            type        = o.type.clone();
            defaultExpr = o.defaultExpr;
        }
        return *this;
    }
    ClassField(ClassField &&) noexcept = default;
    ClassField &operator=(ClassField &&) noexcept = default;
    ~ClassField() noexcept = default;

    ClassField clone() const { return *this; }
};

struct ClassInfo {
    keel::String                                name;
    bool                                        isHandle = false;
    int                                         id       = -1;
    keel::Array<ClassField>                     fields;
    keel::HashMap<keel::String, const ASTNode *> methods;  // name -> FUNCTION_DEF
    const ASTNode                              *def = nullptr;  // the CLASSDEF_DEF

    ClassInfo() noexcept = default;
    ClassInfo(const ClassInfo &o)
        : name(o.name.clone()), isHandle(o.isHandle), id(o.id), fields(o.fields.clone()),
          methods(o.methods.clone()), def(o.def)
    {}
    ClassInfo &operator=(const ClassInfo &o)
    {
        if (this != &o) {
            name     = o.name.clone();
            isHandle = o.isHandle;
            id       = o.id;
            fields   = o.fields.clone();
            methods  = o.methods.clone();
            def      = o.def;
        }
        return *this;
    }
    ClassInfo(ClassInfo &&) noexcept = default;
    ClassInfo &operator=(ClassInfo &&) noexcept = default;
    ~ClassInfo() noexcept = default;

    ClassInfo clone() const { return *this; }

    const ClassField *field(keel::StringRef n) const;
    const ASTNode    *method(keel::StringRef n) const;
};

// classId -> ClassInfo (id == insertion index). Borrows AST nodes (owned by
// the caller's parse tree, which must outlive the registry).
class ClassRegistry {
public:
    ClassRegistry() noexcept = default;
    ClassRegistry(ClassRegistry &&) noexcept = default;
    ClassRegistry &operator=(ClassRegistry &&) noexcept = default;
    ~ClassRegistry() noexcept = default;

    int              add(ClassInfo info);  // assigns + returns the classId
    const ClassInfo *byId(int id) const;
    const ClassInfo *byName(keel::StringRef name) const;
    int              idOf(keel::StringRef name) const;  // -1 if absent
    bool             has(keel::StringRef name) const { return idOf(name) >= 0; }
    keel::usize      size() const { return classes_.size(); }

private:
    keel::Array<ClassInfo>              classes_;
    keel::HashMap<keel::String, int>    byName_;
};

// Why `classDef` is outside the v1 compilable subset; empty string =
// accepted. (Inheritance beyond `handle`, `get.`/`set.` accessors,
// Dependent/Constant/Abstract properties.)
keel::String classRefusalReason(const ASTNode &classDef);

// Build a ClassInfo from a CLASSDEF_DEF: `id` is the classId it will hold;
// field types are inferred from property defaults under `reg`. Throws
// std::runtime_error when the class is refused (classRefusalReason) or a
// property lacks a concrete-typed default.
ClassInfo buildClassInfo(const ASTNode &classDef, int id, const TransferRegistry &reg);

// Collect every CLASSDEF_DEF reachable in `root` into `reg`. Throws on a
// refused class (loud, per §7a). Returns the number added.
keel::usize collectClasses(const ASTNode &root, ClassRegistry &reg,
                           const TransferRegistry &reg2);

} // namespace numkit::codegen
