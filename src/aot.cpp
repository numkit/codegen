// codegen/src/aot.cpp — see aot.hpp.

#include <numkit/codegen/aot.hpp>
#include <numkit/codegen/aot_config.hpp>
#include <numkit/sys/toolchain.hpp>
#include <numkit/sys/env.hpp>

#include <keel/format.h>
#include <keel/fs.h>
#include <keel/hash_set.h>
#include <keel/process.h>

#include <cstdlib>

namespace numkit::codegen::aot {

namespace {

keel::String toNative(keel::StringRef s)
{
    keel::String out = keel::String::from(s);
#ifdef _WIN32
    for (keel::usize i = 0; i < out.size(); ++i)
        if (out[i] == '/') out[i] = '\\';
#endif
    return out;
}

keel::String shellQuotePosix(keel::StringRef s)
{
    keel::String out = keel::String::from("'");
    for (char c : s) {
        if (c == '\'') out.append("'\\''");
        else out.push_back(c);
    }
    out.push_back('\'');
    return out;
}

bool isValidDefine(keel::StringRef d)
{
    if (d.empty()) return false;
    for (char ch : d) {
        const keel::u8 c = keel::as<keel::u8>(ch);
        if (c < 0x20 || c == 0x7f) return false;
#ifdef _WIN32
        if (c == '"' || c == '&' || c == '|' || c == '<' || c == '>' || c == '^' || c == '%')
            return false;
#endif
    }
    return true;
}

bool isPosixSharedLibraryPath(keel::StringRef p)
{
    if (p.empty() || p.back() == '/' || p.back() == '\\') return false;
    if (p.find('/') == keel::kInvalidIndex && p.find('\\') == keel::kInvalidIndex) return false;
    const keel::StringRef fn = keel::path_basename(p);
    if (fn.ends_with(".so")) return true;
    if (fn.contains(".so.")) return true;
    if (fn.ends_with(".dylib")) return true;
    return false;
}

bool writeFile(keel::StringRef path, keel::StringRef content)
{
    return keel::fs_write_text(path, content).is_ok();
}

bool fileExists(keel::StringRef path)
{
    return keel::fs_is_file(path);
}

// Quote a compiler specification that may carry a trailing subcommand such as
// `zig c++` or `C:\Users\...\.numkit\toolchain\zig\zig.exe c++`.
keel::String quoteCompilerCmd(keel::StringRef cxx, bool winQuote)
{
    keel::StringRef exe = cxx;
    keel::StringRef subcmd;
    if (exe.size() > 4 && exe.ends_with(" c++")) {
        subcmd = " c++";
        exe    = exe.substr(0, exe.size() - 4);
    } else if (exe.size() > 3 && exe.ends_with(" cc")) {
        subcmd = " cc";
        exe    = exe.substr(0, exe.size() - 3);
    }
    if (winQuote) {
        return keel::format("\"{}\"{}", toNative(exe), subcmd);
    }
    return keel::format("{}{}", shellQuotePosix(exe), subcmd);
}

// The C++ compiler used at runtime:
//   1. NUMKIT_CXX env override (highest priority)
//   2. NUMKIT_USE_ZIG=1 -> portable/system `zig c++` via numkit::sys::findZig()
//   3. Build-time compiler (NUMKIT_CODEGEN_AOT_CXX) when still available on disk
//   4. Runtime toolchain discovery (`~/.numkit/toolchain/zig`, clang++, g++)
keel::String resolveCxx()
{
    keel::String env = sys::envGet("NUMKIT_CXX");
    if (!env.empty())
        return env;

    if (sys::envGet("NUMKIT_USE_ZIG") == "1") {
        auto zig = sys::findZig();
        if (zig.found)
            return keel::format("{} c++", zig.path);
    }

    if (NUMKIT_CODEGEN_AOT_AVAILABLE != 0) {
        if (NUMKIT_CODEGEN_AOT_IS_MSVC != 0) {
            if (fileExists(NUMKIT_CODEGEN_AOT_VCVARS))
                return keel::String::from(NUMKIT_CODEGEN_AOT_CXX);
        } else if (!keel::StringRef(NUMKIT_CODEGEN_AOT_CXX).empty()) {
            return keel::String::from(NUMKIT_CODEGEN_AOT_CXX);
        }
    }

    auto discovered = sys::findCxxCompiler();
    if (discovered.found && discovered.name != "msvc") {
        if (!discovered.extraArgs.empty())
            return keel::format("{} {}", discovered.path, discovered.extraArgs);
        return keel::move(discovered.path);
    }
    return {};
}

// True when the build-time compiler is MSVC AND we are using that exact compiler.
bool useMsvcVcvars(keel::StringRef cxx)
{
    return NUMKIT_CODEGEN_AOT_IS_MSVC != 0
        && cxx == NUMKIT_CODEGEN_AOT_CXX
        && fileExists(NUMKIT_CODEGEN_AOT_VCVARS);
}

} // namespace

bool available() { return !resolveCxx().empty(); }

namespace {

keel::String findStaticLibInRoots(keel::ArrayRef<keel::String> roots,
                                  keel::StringRef baseName)
{
    const keel::String winName   = keel::format("{}.lib", baseName);
    const keel::String posixName = keel::format("lib{}.a", baseName);

    for (const auto &root : roots) {
        if (root.empty() || !keel::fs_exists(root)) continue;
        const keel::String directCandidates[] = {
            keel::path_join(root, "Release", winName),
            keel::path_join(root, winName),
            keel::path_join(root, "Release", posixName),
            keel::path_join(root, posixName)
        };
        for (const auto &p : directCandidates) {
            if (keel::fs_is_file(p))
                return p.clone();
        }
        keel::String match;
        (void)keel::fs_walk_dir(root, [&](const keel::DirEntry &entry) -> bool {
            if (!entry.is_file) return true;
            if (entry.name == winName || entry.name == posixName) {
                match = keel::String::from(entry.full_path);
                return false;
            }
            return true;
        });
        if (!match.empty())
            return match;
    }
    return {};
}

} // namespace

CompileOptions defaultBridgeCompileOptions()
{
    CompileOptions opts;

    auto homeRes = keel::fs_home_dir();
    const keel::StringRef home = (homeRes.is_ok() && !homeRes->empty()) ? homeRes->as_ref() : keel::StringRef{};
    keel::String envPrefix = sys::envGet("NUMKIT_PREFIX");
    const keel::String prefix = envPrefix.empty()
                                    ? (home.empty() ? keel::String::from(".numkit")
                                                    : keel::path_join(home, ".numkit"))
                                    : keel::move(envPrefix);

    // 1. Locate bridge header directory (numkit_codegen_rt.h)
    const keel::String bridgeCandidates[] = {
#ifdef NUMKIT_CODEGEN_BRIDGE_DIR
        keel::String::from(NUMKIT_CODEGEN_BRIDGE_DIR),
#endif
        keel::path_join(prefix, "src", "codegen", "bridge")
    };
    for (const auto &b : bridgeCandidates) {
        if (!b.empty() && keel::fs_exists(keel::path_join(b, "numkit_codegen_rt.h"))) {
            opts.includeDirs.push_back(b.clone());
            break;
        }
    }

#ifdef NUMKIT_CODEGEN_BRIDGE_DIR
    const keel::StringRef bridgeDirRef(NUMKIT_CODEGEN_BRIDGE_DIR);
    const keel::StringRef bridgeUp2 = bridgeDirRef.empty()
                                          ? keel::StringRef{}
                                          : keel::path_dirname(keel::path_dirname(bridgeDirRef));
    const keel::StringRef bridgeUp3 = bridgeUp2.empty() ? keel::StringRef{} : keel::path_dirname(bridgeUp2);
    const keel::StringRef bridgeUp4 = bridgeUp3.empty() ? keel::StringRef{} : keel::path_dirname(bridgeUp3);
#endif

    // 1b. Locate ops kernels header directory (numkit/ops/kernels.hpp)
    const keel::String opsCandidates[] = {
#ifdef NUMKIT_CODEGEN_BRIDGE_DIR
        bridgeUp2.empty() ? keel::String{} : keel::path_join(bridgeUp2, "ops", "include"),
        bridgeUp2.empty() ? keel::String{}
                          : keel::path_join(bridgeUp2, "numkit/build/_deps/numkit_ops-src/include"),
#endif
        keel::path_join(prefix, "src", "ops", "include"),
        keel::path_join(prefix, "src/numkit/build/_deps/numkit_ops-src/include")
    };
    for (const auto &o : opsCandidates) {
        if (!o.empty() && keel::fs_exists(keel::path_join(o, "numkit", "ops", "kernels.hpp"))) {
            opts.includeDirs.push_back(o.clone());
            break;
        }
    }

    // 1c. Locate keel header directory (keel/core.h, required by <numkit/ops/kernels.hpp>)
    const keel::String keelCandidates[] = {
#ifdef NUMKIT_CODEGEN_KEEL_INCLUDE_DIR
        keel::String::from(NUMKIT_CODEGEN_KEEL_INCLUDE_DIR),
#endif
#ifdef NUMKIT_CODEGEN_BRIDGE_DIR
        bridgeUp4.empty() ? keel::String{} : keel::path_join(bridgeUp4, "keelcpp", "keel", "include"),
        bridgeUp3.empty() ? keel::String{} : keel::path_join(bridgeUp3, "keelcpp", "keel", "include"),
#endif
        keel::path_join(prefix, "include")
    };
    for (const auto &k : keelCandidates) {
        if (!k.empty() && keel::fs_exists(keel::path_join(k, "keel", "core.h"))) {
            opts.includeDirs.push_back(k.clone());
            break;
        }
    }

    // 2. Locate static runtime libraries
    keel::Array<keel::String> searchRoots;
#ifdef NUMKIT_CODEGEN_BUILD_DIR
    if (!keel::StringRef(NUMKIT_CODEGEN_BUILD_DIR).empty()) {
        const keel::StringRef bdir(NUMKIT_CODEGEN_BUILD_DIR);
        searchRoots.push_back(keel::String::from(bdir));
        const keel::StringRef bparent = keel::path_dirname(bdir);
        if (bparent != "." && bparent != bdir && keel::path_basename(bparent) == "modules") {
            searchRoots.push_back(keel::String::from(keel::path_dirname(bparent)));
        }
    }
#endif
#ifdef NUMKIT_CODEGEN_PREBUILT_DIR
    if (!keel::StringRef(NUMKIT_CODEGEN_PREBUILT_DIR).empty()) {
        searchRoots.push_back(keel::String::from(NUMKIT_CODEGEN_PREBUILT_DIR));
    }
#endif
    searchRoots.push_back(keel::path_join(prefix, "src", "codegen", "build"));
    searchRoots.push_back(keel::path_join(prefix, "src", "numkit", "build"));

    const char *requiredLibs[] = {
        "numkit_codegen_core",
        "numkit",
        "numkit_ops",
        "numkit_sys",
        "numkit_value",
        "keel_fft",
        "keel_image",
        "keel_compress",
        "keel",
        "hwy",
        "hwy_contrib"
    };
    for (const char *libName : requiredLibs) {
        keel::String found = findStaticLibInRoots(searchRoots, libName);
        if (!found.empty()) {
            opts.linkLibs.push_back(keel::move(found));
        }
    }
#ifdef _WIN32
    opts.linkLibs.push_back(keel::String::from("user32.lib"));
    opts.linkLibs.push_back(keel::String::from("gdi32.lib"));
#endif
    return opts;
}

namespace {

// Shared compile core for the executable and shared-library forms; the
// only difference is the output kind (an extra /LD for MSVC, -shared
// -fPIC otherwise).
CompileResult compileImpl(keel::StringRef cppSource, keel::StringRef outPath,
                          bool sharedLib, const CompileOptions &opts)
{
    CompileResult r;
    const keel::String cxx = resolveCxx();
    if (cxx.empty()) {
        r.status = CompileStatus::Unavailable;
        return r;
    }

    const keel::String src = keel::format("{}.cpp", outPath);
    for (const auto &d : opts.defines) {
        if (!isValidDefine(d)) {
            r.status = CompileStatus::CompileError;
            r.log    = keel::format("aot: invalid preprocessor define: {}", d);
            return r;
        }
    }
    if (!writeFile(src, cppSource)) {
        r.status = CompileStatus::CompileError;
        r.log    = keel::format("aot: failed to write source file {}", src);
        return r;
    }

    const bool   isMsvc = useMsvcVcvars(cxx);
    keel::String cmd;

#ifdef _WIN32
    // A .bat carries the (optional) vcvars call + compile, so the test
    // process needs no developer-prompt environment and we dodge cmd's
    // nested-quote hazards.
    const keel::String bat = keel::format("{}.build.bat", outPath);
    keel::String       b   = keel::String::from("@echo off\r\n");
    if (isMsvc) {
        keel::String inc, def, libs;
        for (const auto &d : opts.includeDirs) keel::format_into(inc, " /I\"{}\"", toNative(d));
        for (const auto &d : opts.defines) keel::format_into(def, " /D{}", d);
        for (const auto &l : opts.linkLibs) keel::format_into(libs, " \"{}\"", toNative(l));
        keel::format_into(b, "call \"{}\" >nul 2>&1\r\n", toNative(NUMKIT_CODEGEN_AOT_VCVARS));
        keel::format_into(b, "cl /nologo /O2 /EHsc /MD /std:c++20 {}{}{} \"{}\" /Fe:\"{}\" /Fo:\"{}.obj\"",
                          sharedLib ? "/LD " : "", inc, def, toNative(src), toNative(outPath),
                          toNative(outPath));
        if (!libs.empty()) keel::format_into(b, " /link{}", libs);
        b.append("\r\n");
    } else {
        keel::String inc, def, libs;
        for (const auto &d : opts.includeDirs) keel::format_into(inc, " -I\"{}\"", toNative(d));
        for (const auto &d : opts.defines) keel::format_into(def, " -D{}", d);
        for (const auto &l : opts.linkLibs) keel::format_into(libs, " \"{}\"", toNative(l));
        keel::format_into(b, "{} -O2 -std=c++20 {}{}{} \"{}\" -o \"{}\"{}\r\n",
                          quoteCompilerCmd(cxx, /*winQuote=*/true),
                          sharedLib ? "-shared -fPIC " : "", inc, def, toNative(src),
                          toNative(outPath), libs);
    }
    b.append("exit /b %ERRORLEVEL%\r\n");
    if (!writeFile(bat, b)) {
        r.status = CompileStatus::CompileError;
        r.log    = keel::format("aot: failed to write build script {}", bat);
        return r;
    }
    cmd = keel::format("\"{}\"", toNative(bat));
#else
    (void)isMsvc;
    keel::String inc, def, libs;
    for (const auto &d : opts.includeDirs) keel::format_into(inc, " -I{}", shellQuotePosix(d));
    for (const auto &d : opts.defines) {
        const keel::String defFlag = keel::format("-D{}", d);
        keel::format_into(def, " {}", shellQuotePosix(defFlag));
    }
    keel::HashSet<keel::String> rpaths;
    for (const auto &l : opts.linkLibs) {
        keel::format_into(libs, " {}", shellQuotePosix(l));
#  ifndef __EMSCRIPTEN__
        if (isPosixSharedLibraryPath(l)) {
            const keel::StringRef dir = keel::path_dirname(l);
            if (!dir.empty() && rpaths.insert(keel::String::from(dir))) {
                const keel::String rpathFlag = keel::format("-Wl,-rpath,{}", dir);
                keel::format_into(libs, " {}", shellQuotePosix(rpathFlag));
            }
        }
#  endif
    }
    cmd = keel::format("{} -O2 -std=c++20 {}{}{} {} -o {}{}",
                       quoteCompilerCmd(cxx, /*winQuote=*/false),
                       sharedLib ? "-shared -fPIC " : "", inc, def, shellQuotePosix(src),
                       shellQuotePosix(outPath), libs);
#endif

    r.command = cmd.clone();
    keel::ProcessOptions procOpts;
    procOpts.capture_stdout = true;
    procOpts.capture_stderr = true;
    procOpts.merge_stderr = true;
    auto procRes = keel::process_run_shell(cmd, procOpts);
    if (procRes.is_err()) {
        r.log = keel::format("aot: compiler launch failed: {}", keel::error_message(procRes.error()));
        r.status = CompileStatus::CompileError;
        return r;
    }
    r.log = keel::move(procRes->stdout_text);
    r.status = (procRes->ok() && fileExists(outPath)) ? CompileStatus::Ok
                                                      : CompileStatus::CompileError;
    return r;
}

} // namespace

CompileResult compileToExecutable(keel::StringRef cppSource, keel::StringRef exePath,
                                  const CompileOptions &opts)
{
    return compileImpl(cppSource, exePath, /*sharedLib=*/false, opts);
}

CompileResult compileToSharedLibrary(keel::StringRef cppSource, keel::StringRef libPath,
                                      const CompileOptions &opts)
{
    return compileImpl(cppSource, libPath, /*sharedLib=*/true, opts);
}

RunResult runExecutable(keel::StringRef exePath)
{
    RunResult r;
#ifdef __EMSCRIPTEN__
    (void)exePath;
    r.log = keel::String::from("aot: runExecutable is not available under Emscripten — a native "
                               "process cannot be spawned from a WASM module. Use the desktop IDE "
                               "or the native `nk` CLI for compile-and-run.");
    return r;
#else
    if (exePath.empty() || !fileExists(exePath)) {
        r.log = keel::format("aot: executable not found: {}", exePath);
        return r;
    }
    const keel::StringRef argv[] = {exePath};
    keel::ProcessOptions opts;
    opts.capture_stdout = true;
    opts.capture_stderr = true;
    opts.merge_stderr = true;
    auto res = keel::process_run(keel::ArrayRef<keel::StringRef>(argv, 1), opts);
    if (res.is_err()) {
        r.log = keel::format("aot: failed to launch executable: {} ({})",
                             exePath, keel::error_message(res.error()));
        return r;
    }
    r.exitCode = res->exit_code;
    r.log = keel::move(res->stdout_text);
    return r;
#endif
}

} // namespace numkit::codegen::aot
