/* numkit_codegen_rt.h — the Value-ABI bridge (DESIGN.md §6a).
 *
 * A minimal, stable, OPAQUE C ABI that lets AOT-compiled nk code call
 * uncompiled builtins / library functions and hold Dynamic values, WITHOUT
 * depending on nk's C++ Value type, headers, binary layout, or its
 * transitive dependency graph. The implementation (numkit_codegen_rt.cpp) owns
 * nk (Value + a private StandardEngine) entirely behind this boundary.
 *
 * Ownership: box_* / eval / call return OWNED handles the caller must release
 * with numkit_release — and ONLY with numkit_release. A handle's storage belongs to
 * this runtime (the DLL when built as one); never free/delete it yourself or
 * across a different allocator/CRT. call BORROWS its argument handles (caller
 * still owns them). A handle is opaque — never dereference it.
 *
 * Threading: NOT thread-safe. All entry points share one process-wide engine
 * (+ caches); serialise calls, or give each thread its own process. (The
 * underlying interpreter is single-threaded.)
 */
#ifndef NUMKIT_CODEGEN_RT_H
#define NUMKIT_CODEGEN_RT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Linkage. Default (static / same TU — e.g. compiled into a test, the driver,
 * or a self-contained artifact): plain. Building the numkit_codegen_rt shared
 * lib: dllexport. A consumer linking that shared lib (a bridged AOT artifact):
 * define NUMKIT_RT_USE_DLL -> dllimport. No effect off Windows. */
#if defined(_WIN32) && defined(NUMKIT_RT_BUILDING_DLL)
#  define NUMKIT_RT_API __declspec(dllexport)
#elif defined(_WIN32) && defined(NUMKIT_RT_USE_DLL)
#  define NUMKIT_RT_API __declspec(dllimport)
#else
#  define NUMKIT_RT_API
#endif

typedef struct numkit_val_s *numkit_val;  /* opaque boxed nk value */

/* Error channel. code 0 == success. A C++ exception never crosses the C
 * boundary — it is caught and reported here (DESIGN.md §6b). */
typedef struct numkit_error {
    int  code;            /* 0 = ok, non-zero = failure */
    char message[256];    /* NUL-terminated diagnostic (empty on success) */
} numkit_error;

/* Box an unboxed value into a handle (boundary only — never a hot loop). */
NUMKIT_RT_API numkit_val numkit_box_scalar(double v);
NUMKIT_RT_API numkit_val numkit_box_array(const double *p, size_t len);  /* copies the data in */
/* Box a NUL-terminated C string as a 1 x len CHAR row (one byte per code unit) -- for a
 * bridged char ARG such as a sprintf format string. */
NUMKIT_RT_API numkit_val numkit_box_string(const char *s);
/* Cell construction: build a 1 x n cell from n boxed elements (BORROWED; each
 * element is deep-copied into the cell). For a brace literal {e1,...,en} in
 * compiled code. Returns owned; NULL on failure. */
NUMKIT_RT_API numkit_val numkit_box_cell(const numkit_val *elems, size_t n);
/* Build an R x C cell from rows*cols boxed elements given in ROW-MAJOR (source)
 * order; stored column-major (transposed in, mirroring the interpreter's
 * execCellLiteral). Borrowed elements deep-copied. For a 2-D brace literal
 * {a,b; c,d}. Returns owned; NULL on failure. */
NUMKIT_RT_API numkit_val numkit_box_cell_2d(const numkit_val *elems, size_t rows, size_t cols);
/* Content extraction c{idx} -- a 1-based linear index passed as a double to
 * preserve the interpreter's integer/range check. Returns the content of cell
 * `idx` (owned). SINGLE element only: a CSL c{:} / c{vec} is multi-value and the
 * codegen refuses it. Non-cell / out-of-range / non-integer idx -> NULL + *err. */
NUMKIT_RT_API numkit_val numkit_cell_get(numkit_val c, double idx1, numkit_error *err);
/* 2-D content extraction c{i,j} -- 1-based (row, col) subscripts as doubles
 * (sub2indChecked bounds-checks). Returns the content (owned). SINGLE element
 * only. Non-cell / out-of-range / non-integer subscript -> NULL + *err. */
NUMKIT_RT_API numkit_val numkit_cell_get_2d(numkit_val c, double i1, double j1, numkit_error *err);
/* Content store c{idx} = v -- a 1-based linear index as a double. GROWS the cell
 * if idx is past the end and coerces an empty/unset receiver to a cell (MATLAB
 * auto-grow), so `c={}; c{1}=..; c{2}=..` builds a cell imperatively. Mutates the
 * cell handle in place; `v` is BORROWED (deep-copied in). SINGLE element only.
 * Non-integer idx / non-cell receiver -> *err set (no throw across the C frame). */
NUMKIT_RT_API void numkit_cell_set(numkit_val c, double idx1, numkit_val v, numkit_error *err);
/* 2-D content store c{i,j} = v -- 1-based (row, col) subscripts as doubles. GROWS
 * the cell to fit (i,j) and coerces an empty/unset receiver to a cell (MATLAB
 * auto-grow), so c={}; c{1,1}=..; c{2,2}=.. builds a 2-D cell. Mutates in place;
 * `v` borrowed (deep-copied in). Non-integer subscript / non-cell -> *err set. */
NUMKIT_RT_API void numkit_cell_set_2d(numkit_val c, double i1, double j1, numkit_val v, numkit_error *err);
/* Build a named function handle @name (a handle to a builtin / engine-registered
 * function). Returns owned; NULL + *err on failure. The name resolves in the
 * runtime registry -- a co-compiled user spec is NOT there, so the codegen must
 * not pass one (it refuses @userFn at emit time). For @funcName in compiled code. */
NUMKIT_RT_API numkit_val numkit_make_handle(const char *name, numkit_error *err);
/* Build an anonymous closure @(...)... that CAPTURES variables by value: each
 * names[i] is bound to vals[i] (borrowed), then `src` (the full @(...)... source)
 * is evaluated so the closure snapshots them -- the codegen Dynamic-tier capturing-
 * closure path. Returns owned; NULL + *err on failure. (The bindings are removed
 * after, so the runtime workspace is left clean.) */
NUMKIT_RT_API numkit_val numkit_make_closure_captured(const char *src, const char **names,
                                          const numkit_val *vals, size_t n, numkit_error *err);
/* Complex array: `p` is interleaved re,im — `len` complex elements = 2*len
 * doubles, matching std::complex<double>[len] (codegen passes a
 * reinterpret_cast<const double*> of its std::complex buffer). */
NUMKIT_RT_API numkit_val numkit_box_complex_array(const double *p, size_t len);
/* Shape-preserving boxers for a Dynamic ARRAY operand (column-major, matching
 * the codegen RawBuffer layout): a 2-D rows×cols matrix, and an N-D array whose
 * `nd` dims are `dims[0..nd)`. `p` holds rows*cols (resp. prod(dims)) doubles. */
NUMKIT_RT_API numkit_val numkit_box_matrix(const double *p, size_t rows, size_t cols);
NUMKIT_RT_API numkit_val numkit_box_array_nd(const double *p, const size_t *dims, int nd);
/* Complex counterparts: `p` is interleaved re,im (2*numel doubles, column-major),
 * matching std::complex<double>[numel]. 2-D rows×cols, and N-D `dims[0..nd)`. */
NUMKIT_RT_API numkit_val numkit_box_complex_matrix(const double *p, size_t rows, size_t cols);
NUMKIT_RT_API numkit_val numkit_box_complex_array_nd(const double *p, const size_t *dims, int nd);

/* Evaluate nk source `code` in the runtime's persistent workspace and
 * return the last expression's value (owned; an empty handle for pure
 * statements). State persists across calls — numkit_eval("x=5;",0) then
 * numkit_eval("x+1",0) yields 6. On failure returns NULL and sets *err. This is
 * the host embedding entry point (DESIGN.md §6b). */
NUMKIT_RT_API numkit_val numkit_eval(const char *code, numkit_error *err);

/* Invoke builtin/registered function `name` on `args` (borrowed), producing
 * `nargout` results. Returns the first (owned); results [1..nargout-1] are
 * written (owned) into extra_outs[0..nargout-2] (extra_outs may be null when
 * nargout <= 1). On failure returns NULL and sets *err (if non-null);
 * never throws across this boundary. */
NUMKIT_RT_API numkit_val numkit_call(const char *name, const numkit_val *args, size_t nargs,
                         size_t nargout, numkit_val *extra_outs, numkit_error *err);

/* ---- Dynamic-tier value operations (DESIGN.md §10 C1) ----------------------
 *
 * Apply an operator to boxed operands, identical to the interpreter (overload-
 * aware): the codegen Dynamic tier emits these where a value's type could not
 * be inferred, so a typed inline op is unavailable. Operands are BORROWED;
 * the result is OWNED (release with numkit_release). On failure return NULL and
 * set *err. `op` is the nk source token. */
NUMKIT_RT_API numkit_val numkit_binop(const char *op, numkit_val a, numkit_val b, numkit_error *err);
/* Unary op: token "-", "+", "~", "'", ".'". Borrowed operand, owned result. */
NUMKIT_RT_API numkit_val numkit_unop(const char *op, numkit_val a, numkit_error *err);
/* MATLAB truthiness of `v` (an `if` / `while` condition): 1 iff `v` is
 * non-empty and EVERY element is non-zero, else 0. On failure returns 0 and
 * sets *err. (`v` is borrowed.) */
NUMKIT_RT_API int numkit_truth(numkit_val v, numkit_error *err);
/* Apply subscripts to a boxed value: a(subs) — resolved EXACTLY as the
 * interpreter (`value(subs)` is index/call-ambiguous): a function handle /
 * closure is CALLED, an object dispatches to its subsref, otherwise it is
 * array indexing. v1 supports a single subscript (the subscript may be a
 * vector → a sub-array). `subs` borrowed; returns an OWNED result; on failure
 * NULL + *err. (DESIGN.md §10 C1 A4.) */
NUMKIT_RT_API numkit_val numkit_index(numkit_val a, const numkit_val *subs, size_t nsubs, numkit_error *err);

/* Unbox. */
NUMKIT_RT_API double numkit_unbox_scalar(numkit_val v);
NUMKIT_RT_API void   numkit_unbox_array(numkit_val v, double *out, size_t len);  /* copies min(len,numel) */
/* Complex unbox: `out` is interleaved re,im (len complex = 2*len doubles).
 * Handles a REAL result too (imag = 0) — nk may narrow a zero-imag complex
 * result back to a real array. */
NUMKIT_RT_API void   numkit_unbox_complex_array(numkit_val v, double *out, size_t len);
/* Char-array unbox: copies the result's char CODE UNITS (one per element) into
 * `out` (a uint16 buffer, the codegen's char width); copies min(len,numel). For a
 * CHAR result of num2str/sprintf/... — the runtime owns the formatting. */
NUMKIT_RT_API void   numkit_unbox_char_array(numkit_val v, uint16_t *out, size_t len);
NUMKIT_RT_API size_t numkit_numel(numkit_val v);

/* Duplicate a handle: a new OWNED handle holding a copy of `v`'s value (the
 * Dynamic tier's val wrapper is copyable, matching MATLAB value semantics).
 * Returns NULL if `v` is null. (`v` is borrowed.) */
NUMKIT_RT_API numkit_val numkit_clone(numkit_val v);

/* Free an owned handle. */
NUMKIT_RT_API void numkit_release(numkit_val v);

/* The number of owned handles currently outstanding (created by a box / eval /
 * call entry, not yet numkit_release'd). For leak testing: snapshot it around a
 * sequence and assert it returns to baseline. */
NUMKIT_RT_API long long numkit_debug_live_handles(void);

/* ---- Plugin / extension ABI (DESIGN.md §6b) -------------------------------
 *
 * A plugin PROVIDES functions with this signature — the mirror image of
 * numkit_call. It receives borrowed `args` (do not release them), must produce
 * `nargout` results: return the first (OWNED — the runtime releases it) and
 * write results [1..nargout-1] (OWNED) into extra_outs[0..nargout-2]
 * (extra_outs is sized nargout-1, valid only when nargout > 1). To report a
 * failure, set err->code non-zero with a message and return NULL; the runtime
 * raises it as a nk error at the call site. A plugin function must NEVER
 * throw a C++ exception across this boundary. */
typedef numkit_val (*numkit_fn)(const numkit_val *args, size_t nargs, size_t nargout,
                        numkit_val *extra_outs, numkit_error *err);

/* Register `fn` under `name` into the nk runtime. Thereafter `name` is
 * resolvable from nk source, the embedding API, and numkit_call — exactly
 * like a builtin. Returns 0 on success, non-zero on failure — including if
 * `name` is already registered (the runtime rejects duplicates; there is no
 * override without an unregister API). */
NUMKIT_RT_API int numkit_register_fn(const char *name, numkit_fn fn);

/* Load a plugin shared library from `path`, verify its ABI version, and run
 * its registration hook (which registers the plugin's functions into this
 * runtime via the host API; see numkit_plugin.h). Returns 0 on success; on
 * failure returns non-zero and sets *err. On success the library stays loaded
 * for the process lifetime — its registered function pointers remain live.
 * Idempotent per path: loading an already-loaded plugin is a no-op success.
 * (DESIGN.md §6b) */
NUMKIT_RT_API int numkit_load_plugin(const char *path, numkit_error *err);

#ifdef __cplusplus
}
#endif

#endif /* NUMKIT_CODEGEN_RT_H */
