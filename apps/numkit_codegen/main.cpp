// apps/numkit_codegen/numkit_codegen_main.cpp
//
// `numkit_codegen` — the CLI front end for the codegen driver (DESIGN.md §8
// M4). It transpiles a nk source file's entry function to C++ (the
// MATLAB-Coder-style "generate code" step). By default the emitted TU is
// written to stdout / -o; with --run it is additionally AOT-compiled with
// the external C++ compiler (env NUMKIT_CXX first, build-time fallback) and
// executed, with the program's stdout forwarded. A bridged artifact
// additionally links the numkit_codegen_rt runtime; see DESIGN.md §6a — the e2e
// tests exercise that path with the build-captured runtime paths.
//
// Usage:
//   numkit_codegen <file.m> [options]
//     --args "<spec>"   entry parameter types, MATLAB-Coder -args style,
//                       comma-separated; each a dtype with optional [] for a
//                       row vector, e.g. "double[], double, double".
//     --entry <name>    entry function (default: the file's sole function)
//     --bridge          emit C-ABI calls for builtins the emitter can't lower
//     --run             transpile → AOT-compile → run; print program stdout.
//                       The compiler is NUMKIT_CXX (env) or the build-time
//                       default; the artifact is a temp file (not -o).
//     -o <file>         write the generated C++ here (default: stdout).
//                       Ignored with --run unless --emit is also given.

#include <numkit/codegen/aot.hpp>
#include <numkit/codegen/driver.hpp>
#include <numkit/sys/env.hpp>
#include <numkit/sys/toolchain.hpp>

#include <keel/format.h>
#include <keel/exception.h>
#include <keel/fs.h>
#include <keel/process.h>
#include <keel/string.h>

#include <cstdio>
#include <cstdlib>

namespace {

[[noreturn]] void usage(const char *argv0, keel::StringRef err)
{
    keel::eprintln("error: {}\n", err);
    keel::eprintln("usage: {} <file.m> [--args \"<spec>\"] [--entry <name>]"
                   " [--bridge | --plugin <name>] [--ops-kernels] [--build] [--run] [-o <file>]",
                   argv0);
    std::exit(2);
}

bool hasBinaryExtension(keel::StringRef p)
{
    if (p.size() >= 4) {
        const keel::StringRef ext = p.substr(p.size() - 4);
        if (ext == ".exe" || ext == ".EXE" || ext == ".out") return true;
    }
    return false;
}

} // namespace

int main(int argc, char **argv)
{
    keel::String file, argsSpec, entry, outPath, pluginExport;
    bool         bridge = false, opsKernels = false, runFlag = false, buildFlag = false;

    for (int i = 1; i < argc; ++i) {
        const keel::StringRef a(argv[i]);
        auto                  next = [&](const char *flag) -> keel::String {
            if (i + 1 >= argc) usage(argv[0], keel::format("missing value for {}", flag));
            return keel::String::from(argv[++i]);
        };
        if (a == "--args") argsSpec = next("--args");
        else if (a == "--entry") entry = next("--entry");
        else if (a == "-o") outPath = next("-o");
        else if (a == "--bridge") bridge = true;
        else if (a == "--ops-kernels") opsKernels = true;
        else if (a == "--plugin") pluginExport = next("--plugin");
        else if (a == "--build") buildFlag = true;
        else if (a == "--run") runFlag = true;
        else if (a == "-h" || a == "--help") usage(argv[0], "help");
        else if (!a.empty() && a[0] == '-') usage(argv[0], keel::format("unknown option {}", a));
        else if (file.empty()) file = keel::String::from(a);
        else usage(argv[0], keel::format("unexpected extra argument {}", a));
    }
    if (file.empty()) usage(argv[0], "no input file");
    if (!outPath.empty() && hasBinaryExtension(outPath)) buildFlag = true;
    if ((runFlag || buildFlag) && !pluginExport.empty())
        usage(argv[0], "--build/--run and --plugin are mutually exclusive (a plugin TU is "
                       "a shared lib, not a runnable program)");

    auto srcRes = keel::fs_read_text(file);
    if (srcRes.is_err()) {
        keel::eprintln("error: cannot read {}", file);
        return 1;
    }
    const keel::String source = keel::move(srcRes.value());

    if (bridge && !pluginExport.empty())
        usage(argv[0], "--plugin and --bridge are mutually exclusive (a plugin TU is "
                       "self-contained-compiled, not bridged)");

    try {
        namespace cg = numkit::codegen;
        const auto  paramTypes = cg::driver::parseTypeSpec(argsSpec);

        keel::String        generated;  // the C++ TU to write
        keel::String        info;       // a one-line note to stderr (keeps -o clean)
        cg::EmittedFunction emittedFn;  // kept for --run's harness (non-plugin path)
        bool usedBridge = bridge;
        if (!pluginExport.empty()) {
            generated = cg::driver::transpileToPlugin(source, entry, paramTypes, pluginExport,
                                                      "numkit_plugin.h");
            info      = keel::format("emitted plugin registering '{}'", pluginExport);
        } else {
            cg::BridgeOptions bridgeOpts;
            bridgeOpts.enabled       = bridge;
            bridgeOpts.runtimeHeader = keel::String::from("numkit_codegen_rt.h");
            cg::OpsKernelOptions opsOpts;
            opsOpts.enabled = opsKernels;
            try {
                emittedFn = cg::driver::transpileSource(source, entry, paramTypes, bridgeOpts, opsOpts);
            } catch (const keel::Exception &) {
                if ((buildFlag || runFlag) && !bridge) {
                    bridgeOpts.enabled = true;
                    usedBridge         = true;
                    emittedFn = cg::driver::transpileSource(source, entry, paramTypes, bridgeOpts, opsOpts);
                } else {
                    throw;
                }
            }
            generated = emittedFn.source.clone();
            info      = keel::format("emitted entry '{}': {}", emittedFn.name, emittedFn.signature);
        }

        // --build / --run: wrap the emitted TU in a main() harness, AOT-compile
        // it, and optionally execute it.
        if (buildFlag || runFlag) {
            if (!usedBridge && !opsKernels && numkit::sys::envGet("NUMKIT_CXX").empty() &&
                numkit::sys::findZig().found) {
                numkit::sys::envSet("NUMKIT_USE_ZIG", "1");
            }
            if (!cg::aot::available())
                throw keel::Exception(
                    "no C++ compiler configured — set NUMKIT_CXX or rebuild with a "
                    "build-time compiler (see DESIGN.md §8)");
            if (!pluginExport.empty())
                throw keel::Exception("--build/--run is incompatible with --plugin");
            if (!paramTypes.empty())
                throw keel::Exception(
                    "--build/--run: v1 supports only a nullary entry (no --args). For "
                    "functions with args, emit the C++ with -o and write your "
                    "own main() harness.");
            keel::String program = generated.clone();
            program.append(cg::driver::buildHarnessMain(emittedFn).as_ref());

            struct TempDirGuard {
                keel::String dir;
                ~TempDirGuard() {
                    if (!dir.empty()) {
                        (void)keel::fs_remove_all(dir);
                    }
                }
            } tempRunDir;
            keel::String exe;
            if (buildFlag) {
                if (!outPath.empty()) {
                    exe = outPath.clone();
                    if (keel::path_extension(exe).empty() && !keel::process_exe_extension().empty())
                        exe.append(keel::process_exe_extension());
                } else {
                    exe = keel::format("{}{}", keel::path_stem(file), keel::process_exe_extension());
                }
            } else {
                auto tmpRes = keel::fs_create_temp_dir("numkit_codegen_run_");
                if (tmpRes.is_err())
                    throw keel::Exception("failed to create temporary run directory");
                tempRunDir.dir = keel::move(*tmpRes);
                exe = keel::path_join(tempRunDir.dir,
                                      keel::format("aot{}", keel::process_exe_extension()));
            }
            const auto compileOpts = (usedBridge || opsKernels)
                                         ? cg::aot::defaultBridgeCompileOptions()
                                         : cg::aot::CompileOptions{};
            const auto cr = cg::aot::compileToExecutable(program, exe, compileOpts);
            (void)keel::fs_remove(keel::format("{}.cpp", exe));
            (void)keel::fs_remove(keel::format("{}.obj", exe));
            (void)keel::fs_remove(keel::format("{}.build.bat", exe));
            if (!cr.ok())
                throw keel::Exception(
                    keel::format("compilation failed.\n--- compiler log ---\n{}\n--- end log ---",
                                 cr.log).c_str());
            keel::eprintln("numkit-codegen: compiled -> {}", exe);
            if (!runFlag) return 0;

            const auto rr = cg::aot::runExecutable(exe);
            if (!buildFlag) {
                (void)keel::fs_remove(exe);
            }
            if (!rr.log.empty())
                std::fwrite(rr.log.data(), 1, rr.log.size(), stdout);
            if (rr.exitCode != 0) {
                keel::eprintln("numkit-codegen: program exited with code {}", rr.exitCode);
                return rr.exitCode;
            }
            return 0;
        }

        if (outPath.empty()) {
            if (!generated.empty())
                std::fwrite(generated.data(), 1, generated.size(), stdout);
        } else {
            if (keel::fs_write_text(outPath, generated).is_err()) {
                keel::eprintln("error: failed writing {}", outPath);
                return 1;
            }
        }
        keel::eprintln("numkit-codegen: {}", info);  // stderr, so -o stays clean
        return 0;
    } catch (const std::exception &e) {
        keel::eprintln("error: {}", e.what());
        return 1;
    }
}
