// NCC distinguishes C and C++ callback function types more strictly than GCC.
// Preserve the public C API and forward to the existing context implementation.
#include "llama-context.h"
#include "llama.h"
using ve_cpp_abort_callback = bool (*)(void *);
extern "C" void llama_set_abort_callback(llama_context *ctx, ggml_abort_callback callback, void *data) {
    ctx->set_abort_callback(reinterpret_cast<ve_cpp_abort_callback>(callback), data);
}
