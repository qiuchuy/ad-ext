// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Top-level compiler embedding API. This API is stable and intended to provide
// ABI stability across releases. It should be sufficient to construct tools
// and JITs which operate at the level of granularity of input and output
// artifacts.
//
// See the bottom of the file for unstable entrypoints which allow conversion
// to/from corresponding entities in the MLIR C API. These can be used for
// tools that hard-link to the compiler but are not available via runtime
// loaded stubs.

#ifndef IREE_COMPILER_EMBEDDING_API_H
#define IREE_COMPILER_EMBEDDING_API_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "iree/compiler/api_support.h"

#ifdef __cplusplus
extern "C" {
#endif

// Opaque structures.
typedef struct iree_compiler_error_t iree_compiler_error_t;
typedef struct iree_compiler_session_t iree_compiler_session_t;
typedef struct iree_compiler_invocation_t iree_compiler_invocation_t;
typedef struct iree_compiler_source_t iree_compiler_source_t;
typedef struct iree_compiler_output_t iree_compiler_output_t;

//===----------------------------------------------------------------------===//
// Errors.
// Most compilation errors are routed through the diagnostic engine, but
// some errors are returned immediately. In this case, an API function will
// return an `iree_compiler_error_t*` which will be nullptr on success and
// an error on failure. When a non-null error is returned, it must be freed
// via `ireeCompilerErrorDestroy()`.
//===----------------------------------------------------------------------===//

// Destroys an error. Only non-nullptr errors must be destroyed, but it is
// legal to destroy nullptr.
IREE_EMBED_EXPORTED void ireeCompilerErrorDestroy(iree_compiler_error_t *error);

// Gets the message associated with the error as a C-string. The string will be
// valid until the error is destroyed.
IREE_EMBED_EXPORTED const char *
ireeCompilerErrorGetMessage(iree_compiler_error_t *error);

//===----------------------------------------------------------------------===//
// Global initialization.
//===----------------------------------------------------------------------===//

// Gets the version of the API that this compiler exports.
// The version is encoded with the lower 16 bits containing the minor version
// and upper bits containing the major version.
// The compiler API is versioned. Within a major version, symbols may be
// added, but existing symbols must not be removed or changed to alter
// previously exposed functionality. A major version bump implies an API
// break and no forward or backward compatibility is assumed across major
// versions.
IREE_EMBED_EXPORTED int ireeCompilerGetAPIVersion();

// The compiler must be globally initialized before further use.
// It is intended to be called as part of the hosting process's startup
// sequence. Any failures that it can encounter are fatal and will abort
// the process.
// It is legal to call this multiple times, and each call must be balanced
// by a call to |ireeCompilerGlobalShutdown|. The final shutdown call will
// permanently disable the compiler for the process and subsequent calls
// to initialize will fail/abort. If this is not desirable, some higher level
// code must hold initialization open with its own call.
IREE_EMBED_EXPORTED void ireeCompilerGlobalInitialize();

// Gets the build revision of the IREE compiler. In official releases, this
// will be a string with the build tag. In development builds, it may be an
// empty string. The returned is valid for as long as the compiler is
// initialized.
// Available since: 1.1
IREE_EMBED_EXPORTED const char *ireeCompilerGetRevision();

// Processes an argc/argv from main() using platform specific dark magic,
// possibly updating them in-place to cleaned up values. On most systems,
// this is a no-op. However, on Windows, this will do various processing
// to extract the original command line and decode it properly as UTF-8.
// It is therefore essential that this be passed the actual argc/argv to main
// and not some permutation of it. Any updated argv will persist until a
// final call to |ireeCompilerGlobalShutdown|. This must be called after
// |ireeCompilerGlobalInitialize|.
//
// We're really sorry about this. Old platforms are annoying.
// This API is not yet considered version-stable. If using out of tree, please
// contact the developers.
IREE_EMBED_EXPORTED void ireeCompilerGetProcessCLArgs(int *argc,
                                                      const char ***argv);

// Initializes the command line environment from an explicit argc/argv
// (typically the result of a prior call to ireeCompilerGetProcessCLArgs).
// This uses dark magic to setup the usual array of expected signal handlers.
// This API is not yet considered version-stable. If using out of tree, please
// contact the developers.
//
// Note that there is as yet no facility to register new global command line
// options from the C API. However, this facility should be sufficient for
// subordinating builtin command line options to a higher level integration
// by tunneling global options into the initialization sequence.
IREE_EMBED_EXPORTED void ireeCompilerSetupGlobalCL(int argc, const char **argv,
                                                   const char *banner,
                                                   bool installSignalHandlers);

// Destroys any process level resources that the compiler may have created.
// This must be called prior to library unloading.
IREE_EMBED_EXPORTED void ireeCompilerGlobalShutdown();

// Invokes a callback with each registered HAL target backend.
//
// This is really only suitable for global CLI-like tools, as plugins may
// register target backends and plugins are activated at the session level.
IREE_EMBED_EXPORTED void ireeCompilerEnumerateRegisteredHALTargetBackends(
    void (*callback)(const char *backend, void *userData), void *userData);

// Enumerates all plugins that are linked into the compiler.
// Available since: 1.2
IREE_EMBED_EXPORTED void ireeCompilerEnumeratePlugins(
    void (*callback)(const char *pluginName, void *userData), void *userData);

//===----------------------------------------------------------------------===//
// Session management.
// A session represents a scope where one or more runs can be executed.
// Internally, it consists of an MLIRContext and a private set of session
// options. If the CL environment was initialized, session options will be
// bootstrapped from global flags.
//
// Session creation cannot fail in a non-fatal way.
//===----------------------------------------------------------------------===//

// Creates a new session (which must be destroyed by
// ireeCompilerSessionDestroy).
IREE_EMBED_EXPORTED iree_compiler_session_t *ireeCompilerSessionCreate();

// Destroys a session.
IREE_EMBED_EXPORTED void
ireeCompilerSessionDestroy(iree_compiler_session_t *session);

// Sets session-local flags. These are a subset of flags supported by CLI
// tools and are privately scoped.
IREE_EMBED_EXPORTED iree_compiler_error_t *
ireeCompilerSessionSetFlags(iree_compiler_session_t *session, int argc,
                            const char *const *argv);

// Gets textual flags actually in effect from any source. Optionally, only
// calls back for non-default valued flags.
IREE_EMBED_EXPORTED void ireeCompilerSessionGetFlags(
    iree_compiler_session_t *session, bool nonDefaultOnly,
    void (*onFlag)(const char *flag, size_t length, void *), void *userData);

//===----------------------------------------------------------------------===//
// Run management.
// Runs execute against a session and represent a discrete invocation of the
// compiler.
//===----------------------------------------------------------------------===//

enum iree_compiler_diagnostic_severity_t {
  IREE_COMPILER_DIAGNOSTIC_SEVERITY_NOTE = 0,
  IREE_COMPILER_DIAGNOSTIC_SEVERITY_WARNING = 1,
  IREE_COMPILER_DIAGNOSTIC_SEVERITY_ERROR = 2,
  IREE_COMPILER_DIAGNOSTIC_SEVERITY_REMARK = 3,
};

IREE_EMBED_EXPORTED iree_compiler_invocation_t *
ireeCompilerInvocationCreate(iree_compiler_session_t *session);

// Enables a callback to receive diagnostics. This is targeted at API use of
// the compiler, allowing fine grained collection of formatted diagnostic
// records. It is not completely identical to
// |ireeCompilerInvocationEnableConsoleDiagnostics| which produces output
// suitable for an interactive stream (including color detection, etc) and has
// additional features for reading source files, etc. With default flags, no
// system state outside of the session will be used (i.e. no debug information
// loaded from files, etc).
// The |flags| parameter is reserved for the future and must be 0.
// The |callback| may be invoked from any thread at any time prior to
// destruction of the invocation. The callback should not make any calls back
// into compiler APIs.
// The |message| passes to the callback is only valid for the duration of
// the callback and the |messageSize| does not include a terminator nul.
IREE_EMBED_EXPORTED void ireeCompilerInvocationEnableCallbackDiagnostics(
    iree_compiler_invocation_t *inv, int flags,
    void (*callback)(enum iree_compiler_diagnostic_severity_t severity,
                     const char *message, size_t messageSize, void *userData),
    void *userData);

// Enables default, pretty-printed diagnostics to the console. This is usually
// the right thing to do for command-line tools, but other mechanisms are
// preferred for library use.
IREE_EMBED_EXPORTED void
ireeCompilerInvocationEnableConsoleDiagnostics(iree_compiler_invocation_t *inv);

// Destroys a run.
IREE_EMBED_EXPORTED void
ireeCompilerInvocationDestroy(iree_compiler_invocation_t *inv);

// Sets a crash handler on the invocation. In the event of a crash, the callback
// will be invoked to create an output which will receive the crash dump.
// The callback should either set |*outOutput| to a new |iree_compiler_output_t|
// or return an error. Ownership of the output is passed to the caller.
// The implementation implicitly calls |ireeCompilerOutputKeep| on the
// output.
IREE_EMBED_EXPORTED void ireeCompilerInvocationSetCrashHandler(
    iree_compiler_invocation_t *inv, bool genLocalReproducer,
    iree_compiler_error_t *(*onCrashCallback)(
        iree_compiler_output_t **outOutput, void *userData),
    void *userData);

// Parses a source into this instance in preparation for performing a
// compilation action.
// Returns false and emits diagnostics on failure.
IREE_EMBED_EXPORTED bool
ireeCompilerInvocationParseSource(iree_compiler_invocation_t *inv,
                                  iree_compiler_source_t *source);

// Sets a mnemonic phase name to run compilation from. Default is "input".
// The meaning of this is pipeline specific. See IREEVMPipelinePhase
// for the standard pipeline.
// Available since: 1.3
IREE_EMBED_EXPORTED void
ireeCompilerInvocationSetCompileFromPhase(iree_compiler_invocation_t *inv,
                                          const char *phase);

// Sets a mnemonic phase name to run compilation to. Default is "end".
// The meaning of this is pipeline specific. See IREEVMPipelinePhase
// for the standard pipeline.
IREE_EMBED_EXPORTED void
ireeCompilerInvocationSetCompileToPhase(iree_compiler_invocation_t *inv,
                                        const char *phase);

// Enables/disables verification of IR after each pass. Defaults to enabled.
IREE_EMBED_EXPORTED void
ireeCompilerInvocationSetVerifyIR(iree_compiler_invocation_t *inv, bool enable);

// Runs a compilation pipeline.
// Returns false and emits diagnostics on failure.
enum iree_compiler_pipeline_t {
  // IREE's full compilation pipeline.
  IREE_COMPILER_PIPELINE_STD = 0,
  // Pipeline to translate a single hal.executable into a target-specific
  // binary form (such as an ELF file or a flatbuffer containing a SPIR-V
  // blob).
  IREE_COMPILER_PIPELINE_HAL_EXECUTABLE = 1,
  // IREE's precompilation pipeline, which does input preprocessing and
  // pre-fusion global optimization.
  // This is experimental and this should be changed as we move to a more
  // cohesive approach for managing compilation phases.
  IREE_COMPILER_PIPELINE_PRECOMPILE = 2,
};
IREE_EMBED_EXPORTED bool
ireeCompilerInvocationPipeline(iree_compiler_invocation_t *inv,
                               enum iree_compiler_pipeline_t pipeline);

// Runs an arbitrary pass pipeline.
// Returns false and emits diagnostics on failure.
// Available since: 1.4
IREE_EMBED_EXPORTED bool
ireeCompilerInvocationRunPassPipeline(iree_compiler_invocation_t *inv,
                                      const char *textPassPipeline);

// Outputs the current compiler state as textual IR to the output.
IREE_EMBED_EXPORTED iree_compiler_error_t *
ireeCompilerInvocationOutputIR(iree_compiler_invocation_t *inv,
                               iree_compiler_output_t *output);

// Outputs the current compiler state as bytecode IR to the output.
// Emits as the given bytecode version or most recent if -1.
// Available since: 1.4
IREE_EMBED_EXPORTED iree_compiler_error_t *
ireeCompilerInvocationOutputIRBytecode(iree_compiler_invocation_t *inv,
                                       iree_compiler_output_t *output,
                                       int bytecodeVersion);

// Assuming that the compiler has produced VM IR, converts it to bytecode
// and outputs it. This is a valid next step after running the
// IREE_COMPILER_PIPELINE_STD pipeline.
IREE_EMBED_EXPORTED iree_compiler_error_t *
ireeCompilerInvocationOutputVMBytecode(iree_compiler_invocation_t *inv,
                                       iree_compiler_output_t *output);

// Assuming that the compiler has produced VM IR, converts it to textual
// C source and output it. This is a valid next step after running the
// IREE_COMPILER_PIPELINE_STD pipeline.
IREE_EMBED_EXPORTED iree_compiler_error_t *
ireeCompilerInvocationOutputVMCSource(iree_compiler_invocation_t *inv,
                                      iree_compiler_output_t *output);

// Outputs the contents of a single HAL executable as binary data.
// This is a valid next step after running the
// IREE_COMPILER_PIPELINE_HAL_EXECUTABLE pipeline.
IREE_EMBED_EXPORTED iree_compiler_error_t *
ireeCompilerInvocationOutputHALExecutable(iree_compiler_invocation_t *inv,
                                          iree_compiler_output_t *output);

//===----------------------------------------------------------------------===//
// Sources.
// Compilation sources are loaded into iree_compiler_source_t instances. These
// instances reference an originating session and may contain a concrete
// buffer of memory. Generally, when processing a source, its backing buffer
// will be transferred out from under it (i.e. sources are single-use).
// The actual source instance must be kept live until all processing is
// completed as some methods of loading a source involve maintaining
// references to backing resources.
//===----------------------------------------------------------------------===//

// Destroy source instances.
IREE_EMBED_EXPORTED void
ireeCompilerSourceDestroy(iree_compiler_source_t *source);

// Opens the source from a file. This is used for normal text assembly file
// sources.
// Must be destroyed with ireeCompilerSourceDestroy().
IREE_EMBED_EXPORTED iree_compiler_error_t *
ireeCompilerSourceOpenFile(iree_compiler_session_t *session,
                           const char *filePath,
                           iree_compiler_source_t **out_source);

// Wraps an existing buffer in memory.
// If |isNullTerminated| is true, then the null must be accounted for in the
// length. This is required for text buffers and it is permitted for binary
// buffers.
// Must be destroyed with ireeCompilerSourceDestroy().
IREE_EMBED_EXPORTED iree_compiler_error_t *
ireeCompilerSourceWrapBuffer(iree_compiler_session_t *session,
                             const char *bufferName, const char *buffer,
                             size_t length, bool isNullTerminated,
                             iree_compiler_source_t **out_source);

// Splits the current source buffer, invoking a callback for each "split"
// within it. This is per the usual MLIR split rules (see
// splitAndProcessBuffer): which split on `// -----`.
// Both the original source and all yielded sources must be destroyed by the
// caller eventually (split buffers are allowed to escape the callback).
IREE_EMBED_EXPORTED iree_compiler_error_t *ireeCompilerSourceSplit(
    iree_compiler_source_t *source,
    void (*callback)(iree_compiler_source_t *source, void *userData),
    void *userData);

//===----------------------------------------------------------------------===//
// Outputs.
// Compilation outputs are standalone instances that are used to collect
// final compilation artifacts. In their most basic form, they are just
// wrappers around an output stream over some file. However, more advanced
// things can be enabled via additional APIs (i.e. allocating efficient
// temporary file handles, etc).
//
// Outputs are not bound to a session as they can outlive it or be disconnected
// from the actual process of compilation in arbitrary ways.
//===----------------------------------------------------------------------===//

// Destroy output instances.
IREE_EMBED_EXPORTED void
ireeCompilerOutputDestroy(iree_compiler_output_t *output);

// Opens a file for the output.
// Must be destroyed via ireeCompilerOutputDestroy().
IREE_EMBED_EXPORTED iree_compiler_error_t *
ireeCompilerOutputOpenFile(const char *filePath,
                           iree_compiler_output_t **out_output);

// Opens a file descriptor for output.
// Must be destroyed via ireeCompilerOutputDestroy().
IREE_EMBED_EXPORTED iree_compiler_error_t *
ireeCompilerOutputOpenFD(int fd, iree_compiler_output_t **out_output);

// Opens an output to in-memory storage. Use the API
// |ireeCompilerOutputMapMemory| to access the mapped contents once all
// output has been written.
IREE_EMBED_EXPORTED iree_compiler_error_t *
ireeCompilerOutputOpenMembuffer(iree_compiler_output_t **out_output);

// Maps the contents of a compiler output opened via
// |ireeCompilerOutputOpenMembuffer|. This may be something obtained via
// mmap or a more ordinary temporary buffer. This may fail in platform
// specific ways unless if the output was created via
// |ireeCompilerOutputOpenMembuffer|.
IREE_EMBED_EXPORTED iree_compiler_error_t *
ireeCompilerOutputMapMemory(iree_compiler_output_t *output, void **contents,
                            uint64_t *size);

// For file or other persistent outputs, by default they will be deleted on
// |ireeCompilerOutputDestroy| (or exit). It is necessary to call
// |ireeCompilerOutputKeep| in order to have them committed to their accessible
// place.
IREE_EMBED_EXPORTED void ireeCompilerOutputKeep(iree_compiler_output_t *output);

// Writes arbitrary data to the output.
IREE_EMBED_EXPORTED iree_compiler_error_t *
ireeCompilerOutputWrite(iree_compiler_output_t *output, const void *data,
                        size_t length);

#ifdef __cplusplus
}
#endif

#endif // IREE_COMPILER_EMBEDDING_API_H
