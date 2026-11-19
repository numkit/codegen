// codegen/include/numkit/codegen/aot.hpp
//
// The AOT compile harness (build plan brick 4): take a self-contained C++
// source string produced by emitFunction() and compile it to a native
// executable with the external compiler captured at CMake configure time
// (aot_config.hpp). This is the bridge from "we emitted C++" to "we ran
// it" — the end-to-end value-differential gate (brick 5) builds on it.
//
// Design: the harness NEVER fails the test suite when no compiler is
// configured — it returns Unavailable and the caller skips. When a
// compiler IS configured, a non-zero exit (or a missing artifact) is a
// CompileError with the captured compiler log. MSVC is driven through its
// vcvars environment via a generated .bat (so INCLUDE/LIB are set without
// the test process inheriting a developer prompt).

#pragma once

#include <keel/array.h>
#include <keel/ref.h>
#include <keel/string.h>

namespace numkit::codegen::aot {

// Extra compile/link inputs. Empty (the default) reproduces the original
// self-contained, stdlib-only compile. A BRIDGED artifact (DESIGN.md §6a)
// supplies the bridge include dir, NUMKIT_RT_USE_DLL, and the numkit_codegen_rt
// import library so its numkit_* calls resolve.
struct CompileOptions {
    keel::Array<keel::String> includeDirs;  // added as /I or -I
    keel::Array<keel::String> defines;      // added as /D or -D (NAME or NAME=VALUE)
    keel::Array<keel::String> linkLibs;     // full library paths added to the link

    CompileOptions() noexcept = default;
    CompileOptions(const CompileOptions &o)
        : includeDirs(o.includeDirs.clone()), defines(o.defines.clone()),
          linkLibs(o.linkLibs.clone())
    {}
    CompileOptions &operator=(const CompileOptions &o)
    {
        if (this != &o) {
            includeDirs = o.includeDirs.clone();
            defines     = o.defines.clone();
            linkLibs    = o.linkLibs.clone();
        }
        return *this;
    }
    CompileOptions(CompileOptions &&) noexcept = default;
    CompileOptions &operator=(CompileOptions &&) noexcept = default;
    ~CompileOptions() noexcept = default;

    CompileOptions clone() const { return *this; }
};

enum class CompileStatus {
    Ok,            // compiled; the executable exists at exePath
    CompileError,  // the compiler ran but failed (see CompileResult::log)
    Unavailable,   // no external compiler configured for this build
};

struct CompileResult {
    CompileStatus status = CompileStatus::Unavailable;
    keel::String  log;      // compiler stdout+stderr (best effort)
    keel::String  command;  // the command line used (for diagnostics)

    CompileResult() noexcept = default;
    CompileResult(CompileStatus st, keel::String l = {}, keel::String cmd = {}) noexcept
        : status(st), log(keel::move(l)), command(keel::move(cmd))
    {}
    CompileResult(const CompileResult &o)
        : status(o.status), log(o.log.clone()), command(o.command.clone())
    {}
    CompileResult &operator=(const CompileResult &o)
    {
        if (this != &o) {
            status  = o.status;
            log     = o.log.clone();
            command = o.command.clone();
        }
        return *this;
    }
    CompileResult(CompileResult &&) noexcept = default;
    CompileResult &operator=(CompileResult &&) noexcept = default;
    ~CompileResult() noexcept = default;

    CompileResult clone() const { return *this; }

    bool ok() const { return status == CompileStatus::Ok; }
};

// True when an external C++ compiler was configured at CMake-configure
// time (so compileToExecutable can actually build something).
bool available();

// Build default CompileOptions (include directory + static runtime libraries)
// for compiling a bridged C++ translation unit into a standalone executable.
CompileOptions defaultBridgeCompileOptions();

// Compile `cppSource` (a self-contained TU) into a native executable at
// `exePath`. Writes intermediate files (`<exePath>.cpp`, a build script,
// a log) next to `exePath`. Synchronous. Returns Unavailable without
// touching the filesystem when no compiler is configured.
CompileResult compileToExecutable(keel::StringRef cppSource,
                                  keel::StringRef exePath,
                                  const CompileOptions &opts = {});

// As above but emits a shared library (DLL / .so) at `libPath` — for
// loading the transpiler's output and benchmarking it against the
// hand-written reference (brick 7). The source should export a symbol
// (e.g. extern "C" __declspec(dllexport) on Windows).
CompileResult compileToSharedLibrary(keel::StringRef cppSource,
                                      keel::StringRef libPath,
                                      const CompileOptions &opts = {});

// Run the executable at `exePath` (typically produced by compileToExecutable)
// and capture its combined stdout+stderr. Synchronous. Returns the exit code
// (0 on success) and the captured output. Native-only — under Emscripten a
// process spawn is meaningless, so this returns -1 with an explanatory `log`.
struct RunResult {
    int          exitCode = -1;
    keel::String log;      // captured stdout+stderr

    RunResult() noexcept = default;
    RunResult(int ec, keel::String l = {}) noexcept : exitCode(ec), log(keel::move(l)) {}
    RunResult(const RunResult &o) : exitCode(o.exitCode), log(o.log.clone()) {}
    RunResult &operator=(const RunResult &o)
    {
        if (this != &o) {
            exitCode = o.exitCode;
            log      = o.log.clone();
        }
        return *this;
    }
    RunResult(RunResult &&) noexcept = default;
    RunResult &operator=(RunResult &&) noexcept = default;
    ~RunResult() noexcept = default;

    RunResult clone() const { return *this; }
};
RunResult runExecutable(keel::StringRef exePath);

} // namespace numkit::codegen::aot
