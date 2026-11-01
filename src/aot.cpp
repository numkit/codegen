// codegen/src/aot.cpp — see aot.hpp.

#include <numkit/codegen/aot.hpp>
#include <numkit/codegen/aot_config.hpp>
#include <numkit/sys/toolchain.hpp>
#include <numkit/sys/env.hpp>

#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace numkit::codegen::aot {

namespace {

std::string toNative(std::string s)
{
#ifdef _WIN32
    for (char &c : s)
        if (c == '/') c = '\\';
#endif
    return s;
}

bool writeFile(const std::string &path, const std::string &content)
{
    std::ofstream os(path, std::ios::binary);
    if (!os) return false;
    os.write(content.data(), static_cast<std::streamsize>(content.size()));
    return static_cast<bool>(os);
}

std::string readFile(const std::string &path)
{
    std::ifstream is(path, std::ios::binary);
    if (!is) return {};
    std::ostringstream ss;
    ss << is.rdbuf();
    return ss.str();
}

bool fileExists(const std::string &path)
{
    std::ifstream is(path, std::ios::binary);
    return static_cast<bool>(is);
}

// Quote a compiler specification that may carry a trailing subcommand such as
// `zig c++` or `C:\Users\...\.numkit\toolchain\zig\zig.exe c++`.
std::string quoteCompilerCmd(const std::string &cxx, bool winQuote)
{
    std::string exe = cxx;
    std::string subcmd;
    if (exe.size() > 4 && exe.compare(exe.size() - 4, 4, " c++") == 0) {
        subcmd = " c++";
        exe.resize(exe.size() - 4);
    } else if (exe.size() > 3 && exe.compare(exe.size() - 3, 3, " cc") == 0) {
        subcmd = " cc";
        exe.resize(exe.size() - 3);
    }
    if (winQuote) {
        return "\"" + toNative(exe) + "\"" + subcmd;
    }
    if (!subcmd.empty()) {
        return "'" + exe + "'" + subcmd;
    }
    return exe;
}

// The C++ compiler used at runtime:
//   1. NUMKIT_CXX env override (highest priority)
//   2. NUMKIT_USE_ZIG=1 -> portable/system `zig c++` via numkit::sys::findZig()
//   3. Build-time compiler (NUMKIT_CODEGEN_AOT_CXX) when still available on disk
//   4. Runtime toolchain discovery (`~/.numkit/toolchain/zig`, clang++, g++)
std::string resolveCxx()
{
    std::string env = sys::envGet("NUMKIT_CXX");
    if (!env.empty())
        return env;

    if (sys::envGet("NUMKIT_USE_ZIG") == "1") {
        auto zig = sys::findZig();
        if (zig.found)
            return zig.path + " c++";
    }

    if (NUMKIT_CODEGEN_AOT_AVAILABLE != 0) {
        if (NUMKIT_CODEGEN_AOT_IS_MSVC != 0) {
            if (fileExists(NUMKIT_CODEGEN_AOT_VCVARS))
                return NUMKIT_CODEGEN_AOT_CXX;
        } else if (std::string(NUMKIT_CODEGEN_AOT_CXX).size() > 0) {
            return NUMKIT_CODEGEN_AOT_CXX;
        }
    }

    auto discovered = sys::findCxxCompiler();
    if (discovered.found && discovered.name != "msvc") {
        if (!discovered.extraArgs.empty())
            return discovered.path + " " + discovered.extraArgs;
        return discovered.path;
    }
    return {};
}

// True when the build-time compiler is MSVC AND we are using that exact compiler.
bool useMsvcVcvars(const std::string &cxx)
{
    return NUMKIT_CODEGEN_AOT_IS_MSVC != 0
        && cxx == NUMKIT_CODEGEN_AOT_CXX
        && fileExists(NUMKIT_CODEGEN_AOT_VCVARS);
}

} // namespace

bool available() { return !resolveCxx().empty(); }

namespace {

std::string findStaticLibInRoots(const std::vector<std::filesystem::path> &roots,
                                 const std::string &baseName)
{
    namespace fs = std::filesystem;
    std::error_code ec;
    const std::string winName   = baseName + ".lib";
    const std::string posixName = "lib" + baseName + ".a";

    for (const auto &root : roots) {
        if (root.empty() || !fs::exists(root, ec)) continue;
        const fs::path directCandidates[] = {
            root / "Release" / winName,
            root / winName,
            root / "Release" / posixName,
            root / posixName
        };
        for (const auto &p : directCandidates) {
            if (fs::exists(p, ec) && fs::is_regular_file(p, ec))
                return p.string();
        }
        for (const auto &entry : fs::recursive_directory_iterator(root, ec)) {
            if (!entry.is_regular_file(ec)) continue;
            const std::string fn = entry.path().filename().string();
            if (fn == winName || fn == posixName)
                return entry.path().string();
        }
    }
    return {};
}

} // namespace

CompileOptions defaultBridgeCompileOptions()
{
    namespace fs = std::filesystem;
    CompileOptions opts;
    std::error_code ec;

#ifdef _WIN32
    std::string home = sys::envGet("USERPROFILE");
#else
    std::string home = sys::envGet("HOME");
#endif
    fs::path prefix = sys::envGet("NUMKIT_PREFIX").empty()
                          ? (home.empty() ? fs::path(".numkit") : (fs::path(home) / ".numkit"))
                          : fs::path(sys::envGet("NUMKIT_PREFIX"));

    // 1. Locate bridge header directory (numkit_codegen_rt.h)
    const fs::path bridgeCandidates[] = {
#ifdef NUMKIT_CODEGEN_BRIDGE_DIR
        fs::path(NUMKIT_CODEGEN_BRIDGE_DIR),
#endif
        prefix / "src" / "codegen" / "bridge"
    };
    for (const auto &b : bridgeCandidates) {
        if (!b.empty() && fs::exists(b / "numkit_codegen_rt.h", ec)) {
            opts.includeDirs.push_back(b.string());
            break;
        }
    }

    // 2. Locate static runtime libraries
    std::vector<fs::path> searchRoots;
#ifdef NUMKIT_CODEGEN_BUILD_DIR
    if (std::string(NUMKIT_CODEGEN_BUILD_DIR).size() > 0) {
        fs::path bdir(NUMKIT_CODEGEN_BUILD_DIR);
        searchRoots.push_back(bdir);
        if (bdir.has_parent_path() && bdir.parent_path().filename() == "modules") {
            searchRoots.push_back(bdir.parent_path().parent_path());
        }
    }
#endif
#ifdef NUMKIT_CODEGEN_PREBUILT_DIR
    if (std::string(NUMKIT_CODEGEN_PREBUILT_DIR).size() > 0) {
        searchRoots.push_back(fs::path(NUMKIT_CODEGEN_PREBUILT_DIR));
    }
#endif
    searchRoots.push_back(prefix / "src" / "codegen" / "build");
    searchRoots.push_back(prefix / "src" / "numkit" / "build");

    const char *requiredLibs[] = {
        "numkit_codegen_core",
        "numkit",
        "numkit_ops",
        "numkit_sys",
        "numkit_value",
        "hwy",
        "hwy_contrib"
    };
    for (const char *libName : requiredLibs) {
        std::string found = findStaticLibInRoots(searchRoots, libName);
        if (!found.empty()) {
            opts.linkLibs.push_back(found);
        }
    }
#ifdef _WIN32
    opts.linkLibs.push_back("user32.lib");
    opts.linkLibs.push_back("gdi32.lib");
#endif
    return opts;
}

namespace {

// Shared compile core for the executable and shared-library forms; the
// only difference is the output kind (an extra /LD for MSVC, -shared
// -fPIC otherwise).
CompileResult compileImpl(const std::string &cppSource, const std::string &outPath,
                          bool sharedLib, const CompileOptions &opts)
{
    CompileResult r;
    const std::string cxx = resolveCxx();
    if (cxx.empty()) {
        r.status = CompileStatus::Unavailable;
        return r;
    }

    const std::string src = outPath + ".cpp";
    const std::string log = outPath + ".log";
    if (!writeFile(src, cppSource)) {
        r.status = CompileStatus::CompileError;
        r.log    = "aot: failed to write source file " + src;
        return r;
    }

    const bool  isMsvc = useMsvcVcvars(cxx);
    std::string cmd;

#ifdef _WIN32
    // A .bat carries the (optional) vcvars call + compile, so the test
    // process needs no developer-prompt environment and we dodge cmd's
    // nested-quote hazards. Redirect the whole script to the log file.
    const std::string bat = outPath + ".build.bat";
    std::string       b   = "@echo off\r\n";
    if (isMsvc) {
        std::string inc, def, libs;
        for (const auto &d : opts.includeDirs) inc += " /I\"" + toNative(d) + "\"";
        for (const auto &d : opts.defines) def += " /D" + d;
        for (const auto &l : opts.linkLibs) libs += " \"" + toNative(l) + "\"";
        b += "call \"" + toNative(NUMKIT_CODEGEN_AOT_VCVARS) + "\" >nul 2>&1\r\n";
        b += std::string("cl /nologo /O2 /EHsc /MD /std:c++17 ") + (sharedLib ? "/LD " : "") + inc
             + def + " \"" + toNative(src) + "\" /Fe:\"" + toNative(outPath) + "\" /Fo:\""
             + toNative(outPath) + ".obj\"" + (libs.empty() ? "" : " /link" + libs) + "\r\n";
    } else {
        std::string inc, def, libs;
        for (const auto &d : opts.includeDirs) inc += " -I\"" + toNative(d) + "\"";
        for (const auto &d : opts.defines) def += " -D" + d;
        for (const auto &l : opts.linkLibs) libs += " \"" + toNative(l) + "\"";
        b += quoteCompilerCmd(cxx, /*winQuote=*/true) + " -O2 -std=c++17 "
             + (sharedLib ? "-shared -fPIC " : "") + inc + def + " \"" + toNative(src) + "\" -o \""
             + toNative(outPath) + "\"" + libs + "\r\n";
    }
    b += "exit /b %ERRORLEVEL%\r\n";
    if (!writeFile(bat, b)) {
        r.status = CompileStatus::CompileError;
        r.log    = "aot: failed to write build script " + bat;
        return r;
    }
    // `cmd /c` (what std::system uses) strips the first and last quote of
    // the whole line, which would mangle a command that itself begins and
    // ends with a quoted path. Wrap the entire command in one extra pair
    // so the inner quoting survives.
    cmd = "\"\"" + toNative(bat) + "\" > \"" + toNative(log) + "\" 2>&1\"";
#else
    std::string inc, def, libs;
    for (const auto &d : opts.includeDirs) inc += " -I'" + d + "'";
    for (const auto &d : opts.defines) def += " -D" + d;
    for (const auto &l : opts.linkLibs) libs += " '" + l + "'";
    cmd = quoteCompilerCmd(cxx, /*winQuote=*/false) + " -O2 -std=c++17 "
          + (sharedLib ? "-shared -fPIC " : "") + inc + def + " '" + src + "' -o '" + outPath
          + "'" + libs + " > '" + log + "' 2>&1";
#endif

    r.command    = cmd;
    const int rc = std::system(cmd.c_str());
    r.log        = readFile(log);
    r.status = (rc == 0 && fileExists(outPath)) ? CompileStatus::Ok
                                                 : CompileStatus::CompileError;
    return r;
}

} // namespace

CompileResult compileToExecutable(const std::string &cppSource, const std::string &exePath,
                                  const CompileOptions &opts)
{
    return compileImpl(cppSource, exePath, /*sharedLib=*/false, opts);
}

CompileResult compileToSharedLibrary(const std::string &cppSource, const std::string &libPath,
                                      const CompileOptions &opts)
{
    return compileImpl(cppSource, libPath, /*sharedLib=*/true, opts);
}

RunResult runExecutable(const std::string &exePath)
{
    RunResult r;
#ifdef __EMSCRIPTEN__
    (void)exePath;
    r.log = "aot: runExecutable is not available under Emscripten — a native "
            "process cannot be spawned from a WASM module. Use the desktop IDE "
            "or the native `nk` CLI for compile-and-run.";
    return r;
#else
    if (exePath.empty() || !fileExists(exePath)) {
        r.log = "aot: executable not found: " + exePath;
        return r;
    }
    // Capture combined stdout+stderr via a pipe. popen returns the child's
    // stdout; `2>&1` merges stderr into that stream. Quote the exe path on
    // Windows (cmd /c strips the outer quotes, so wrap once more — same
    // trick as compileImpl).
    std::string cmd;
#  ifdef _WIN32
    cmd = "\"\"" + toNative(exePath) + "\" 2>&1\"";
#  else
    cmd = "'" + exePath + "' 2>&1";
#  endif
#  ifdef _WIN32
    FILE *pipe = _popen(cmd.c_str(), "rt");
#  else
    FILE *pipe = ::popen(cmd.c_str(), "r");
#  endif
    if (!pipe) {
        r.log = "aot: failed to launch executable: " + exePath;
        return r;
    }
    std::ostringstream ss;
    char buf[4096];
    while (std::fgets(buf, sizeof(buf), pipe))
        ss << buf;
#  ifdef _WIN32
    r.exitCode = _pclose(pipe);
#  else
    r.exitCode = ::pclose(pipe);
#  endif
    r.log = ss.str();
    return r;
#endif
}

} // namespace numkit::codegen::aot
