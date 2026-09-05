"""Build the small Issue #3 native search kernel.

The extension deliberately does not link against Faiss or a BLAS.  It consumes
canonical packed buffers through pybind11 and uses the host C++ compiler's
ordinary IEEE operations under the explicit non-fast-math flags below.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

import pybind11


NATIVE_COMPILE_FLAGS = [
    "-O3",
    "-std=c++17",
    "-fno-fast-math",
    "-ffp-contract=off",
]
NATIVE_SOURCE = Path("src/searchability/_native_kernel.cpp")


def _macro_string(value: str) -> str:
    """Render one build-provenance value as a portable C string literal."""

    return json.dumps(value)


class NativeBuildExt(build_ext):
    """Embed the compiler driver/flags selected by setuptools into the binary."""

    def build_extensions(self) -> None:
        compiler_so = list(getattr(self.compiler, "compiler_so", ()) or ())
        compiler_cxx = list(getattr(self.compiler, "compiler_cxx", ()) or ())
        if not compiler_so and not compiler_cxx:
            raise RuntimeError("setuptools did not expose a native compiler command")
        # UnixCCompiler replaces compiler_so[0] with compiler_cxx[0] for a .cpp
        # source.  Capture exactly that selected driver and all driver/default
        # flags before setuptools appends per-source -I/-D/-c/-o arguments.
        driver = str((compiler_cxx or compiler_so)[0])
        default_flags = [str(value) for value in (compiler_so[1:] if compiler_so else [])]
        source_sha256 = hashlib.sha256(NATIVE_SOURCE.read_bytes()).hexdigest()
        for extension in self.extensions:
            extra_flags = [str(value) for value in extension.extra_compile_args or ()]
            driver_and_flags = [driver, *default_flags, *extra_flags]
            extension.define_macros = list(extension.define_macros or ()) + [
                (
                    "SEARCHABILITY_NATIVE_SOURCE_SHA256",
                    _macro_string(source_sha256),
                ),
                (
                    "SEARCHABILITY_COMPILER_EXECUTABLE",
                    _macro_string(driver),
                ),
                (
                    "SEARCHABILITY_COMPILER_COMMAND",
                    _macro_string(shlex.join(driver_and_flags)),
                ),
                (
                    "SEARCHABILITY_COMPILER_COMMAND_SOURCE",
                    _macro_string(
                        "setuptools build_ext: compiler_cxx[0] plus "
                        "compiler_so[1:] plus Extension.extra_compile_args"
                    ),
                ),
            ]
        super().build_extensions()


setup(
    cmdclass={"build_ext": NativeBuildExt},
    ext_modules=[
        Extension(
            "searchability._native_kernel",
            [str(NATIVE_SOURCE)],
            include_dirs=[pybind11.get_include()],
            language="c++",
            extra_compile_args=NATIVE_COMPILE_FLAGS,
        )
    ]
)
