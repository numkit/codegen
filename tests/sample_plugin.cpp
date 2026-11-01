// sample_plugin.cpp — a minimal nk plugin (DESIGN.md §6b), built as a
// standalone shared library and loaded at runtime by the plugin e2e test.
//
// It links against NOTHING from the host: it includes only numkit_plugin.h and
// uses the host function table passed to numkit_plugin_register. It registers one
// function, numkit_sample_triple, which triples its scalar argument.

#include "numkit_plugin.h"

namespace {

// Saved host API table — the plugin's registered function reaches box/unbox
// (and could reach call/numel/...) through it, never through a linked symbol.
const numkit_host_api *g_host = nullptr;

numkit_val sample_triple(const numkit_val *args, size_t nargs, size_t nargout,
                     numkit_val *extra_outs, numkit_error *err)
{
    (void)nargout;
    (void)extra_outs;
    // A plugin function should validate its inputs and report failure via
    // numkit_error (never assume nargs / dereference a missing arg).
    if (!g_host || nargs < 1 || !args) {
        if (err) {
            err->code = 1;
            err->message[0] = '\0';  // (a real plugin would set a diagnostic)
        }
        return nullptr;
    }
    return g_host->box_scalar(g_host->unbox_scalar(args[0]) * 3.0);
}

} // namespace

extern "C" {

NUMKIT_PLUGIN_EXPORT int numkit_plugin_abi_version(void) { return NUMKIT_PLUGIN_ABI_VERSION; }

NUMKIT_PLUGIN_EXPORT int numkit_plugin_register(const numkit_host_api *host)
{
    if (!host || host->abi_version != NUMKIT_PLUGIN_ABI_VERSION) return 1;
    g_host = host;
    return host->register_fn("numkit_sample_triple", &sample_triple);
}

} // extern "C"
