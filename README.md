# numkit/codegen

Ahead-of-time (AOT) C++ transpiler and static type-inference engine for [NumKit](https://github.com/numkit/numkit).

## Overview

`numkit/codegen` translates `.m` functions and classes into standalone C++17 translation units, native plugins, or executables:

- **Static Type Lattice & Abstract Interpretation** (`include/numkit/codegen/type_lattice.hpp`, `inference.hpp`, `transfer.hpp`, `monomorphize.hpp`, `classinfo.hpp`): interprocedural value-type, shape, and class inference with call-site specialization.
- **C++17 Emitter** (`include/numkit/codegen/emitter.hpp`, `indexing.hpp`): lowers typed ASTs to self-contained C++17 (raw column-major buffers, SIMD ops kernels via `numkit/ops`, or C-ABI runtime bridge calls).
- **AOT Compiler Driver & CLI** (`include/numkit/codegen/driver.hpp`, `aot.hpp`, `apps/numkit_codegen/main.cpp`): `numkit-codegen` command-line tool supporting top-level scripts and functions with `--args`, `--entry`, `--bridge`, `--plugin`, `--build`, and `--run`.
- **Opaque C-ABI Runtime Bridge & Tiered Plugin ABI** (`bridge/numkit_codegen_rt.h`, `bridge/numkit_plugin.h`): zero-header-leak C ABI for calling back into the NumKit interpreter or loading compiled functions as native runtime plugins.

See [DESIGN.md](DESIGN.md) for the full architecture and lowering contracts.

## Building

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

## License

BSD Zero Clause License ([0BSD](LICENSE)).
