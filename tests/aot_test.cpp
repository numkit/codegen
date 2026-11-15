// codegen/tests/aot_test.cpp
//
// Brick 4: the AOT compile harness. When an external compiler is
// configured (it is on this MSVC build), a self-contained C++ source
// string compiles to a runnable executable; a broken source reports a
// CompileError with a log. When no compiler is configured the harness
// returns Unavailable and these tests skip (the green baseline is never
// broken by a missing toolchain).

#include <numkit/codegen/aot.hpp>

#include <gtest/gtest.h>

#include <keel/fs.h>
#include <keel/interop_std.h>

#include <cstdlib>
#include <string>

namespace {

keel::String tmpBase()
{
    auto tmp = keel::fs_temp_dir();
    keel::String p = keel::path_join(
        tmp.is_ok() ? tmp.value().as_ref() : keel::StringRef("/tmp"), "numkit_codegen_aot");
    (void)keel::fs_create_dirs(p);
    return p;
}

}  // namespace

TEST(Aot, TrivialCompileAndRun)
{
    if (!numkit::codegen::aot::available())
        GTEST_SKIP() << "no external compiler configured for this build";

    const std::string exe = keel::to_std_string(keel::path_join(tmpBase(), "numkit_trivial.exe"));
    const char       *src =
        "#include <cstdio>\n"
        "int main() { std::printf(\"ok\\n\"); return 0; }\n";

    const auto r = numkit::codegen::aot::compileToExecutable(src, exe);
    ASSERT_EQ(r.status, numkit::codegen::aot::CompileStatus::Ok)
        << "log:\n" << r.log << "\ncmd: " << r.command;
    ASSERT_TRUE(keel::fs_exists(exe));
    EXPECT_EQ(std::system(("\"" + exe + "\"").c_str()), 0);
}

TEST(Aot, CompileErrorReported)
{
    if (!numkit::codegen::aot::available())
        GTEST_SKIP() << "no external compiler configured for this build";

    const std::string exe = keel::to_std_string(keel::path_join(tmpBase(), "numkit_broken.exe"));
    const auto        r =
        numkit::codegen::aot::compileToExecutable("@ this is not valid c++ @\n", exe);

    EXPECT_EQ(r.status, numkit::codegen::aot::CompileStatus::CompileError);
    EXPECT_FALSE(r.log.empty());
}
