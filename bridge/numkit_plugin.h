/* numkit_plugin.h — the nk plugin-author ABI (DESIGN.md §6b).
 *
 * A plugin is a shared library (.dll / .so / .dylib) that exports two C entry
 * points. The host (nk runtime) loads it via numkit_load_plugin (see
 * numkit_codegen_rt.h), checks the ABI version, then calls numkit_plugin_register,
 * passing a table of host functions (numkit_host_api). The plugin uses that table
 * to register its own functions and to box / unbox / call values.
 *
 * The plugin links against NOTHING from the host — it only includes this
 * header and uses the passed-in function pointers. No import library, no
 * symbol or binary-layout coupling: the ABI is a hard, versioned boundary.
 */
#ifndef NUMKIT_PLUGIN_H
#define NUMKIT_PLUGIN_H

#include "numkit_codegen_rt.h" /* numkit_val, numkit_error, numkit_fn */

#ifdef __cplusplus
extern "C" {
#endif

/* Bump on ANY incompatible change to numkit_host_api or the value ABI. The host
 * refuses to load a plugin built against a different version. */
#define NUMKIT_PLUGIN_ABI_VERSION 1

/* Export macro for the plugin's two entry points. */
#if defined(_WIN32)
#  define NUMKIT_PLUGIN_EXPORT __declspec(dllexport)
#else
#  define NUMKIT_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

/* The table of host functions handed to the plugin at registration time.
 * Mirrors the free functions in numkit_codegen_rt.h, but passed by pointer so the
 * plugin needs no import library and no link against the host. */
typedef struct numkit_host_api {
    int abi_version; /* == NUMKIT_PLUGIN_ABI_VERSION the host was built with */
    int (*register_fn)(const char *name, numkit_fn fn);
    numkit_val (*box_scalar)(double v);
    numkit_val (*box_array)(const double *p, size_t len);
    numkit_val (*call)(const char *name, const numkit_val *args, size_t nargs,
                   size_t nargout, numkit_val *extra_outs, numkit_error *err);
    double (*unbox_scalar)(numkit_val v);
    void (*unbox_array)(numkit_val v, double *out, size_t len);
    size_t (*numel)(numkit_val v);
    void (*release)(numkit_val v);
} numkit_host_api;

/* The two entry points a plugin MUST export.
 *   numkit_plugin_abi_version — return NUMKIT_PLUGIN_ABI_VERSION (the value you
 *                           compiled against); the host refuses a mismatch.
 *   numkit_plugin_register    — register the plugin's functions via host->register_fn;
 *                           return 0 on success, non-zero to abort the load.
 *
 * Lifetime: the `host` table pointer stays valid for the process lifetime, so
 * a plugin MAY store it and use it later from its registered functions (as the
 * sample plugin does). */
NUMKIT_PLUGIN_EXPORT int numkit_plugin_abi_version(void);
NUMKIT_PLUGIN_EXPORT int numkit_plugin_register(const numkit_host_api *host);

#ifdef __cplusplus
}
#endif

#endif /* NUMKIT_PLUGIN_H */
