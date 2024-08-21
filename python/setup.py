import os 
import re
import shutil
import subprocess
import sys
import sysconfig

from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py

from typing import NamedTuple

def check_env_flag(name: str, default: str = "") -> bool:
    return os.getenv(name, default).upper() in ["ON", "1", "YES", "TRUE", "Y"]

def get_build_type():
    if check_env_flag("DEBUG"):
        return "Debug"
    else:
        return "Release"

# ---- cmake extension ----
def get_base_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def get_cmake_dir():
    plat_name = sysconfig.get_platform()
    python_version = sysconfig.get_python_version()
    dir_name = f"cmake.{plat_name}-{sys.implementation.name}-{python_version}"
    cmake_dir = Path(get_base_dir()) / "python" / "build" / dir_name
    cmake_dir.mkdir(parents=True, exist_ok=True)
    return cmake_dir

class CMakeBuildPy(build_py):

    def run(self) -> None:
        self.run_command('build_ext')
        return super().run()


class CMakeExtension(Extension):

    def __init__(self, name, path, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)
        self.path = path


class CMakeBuild(build_ext):

    user_options = build_ext.user_options + \
        [('base-dir=', None, 'base directory of AILang')]

    def initialize_options(self):
        build_ext.initialize_options(self)
        self.base_dir = get_base_dir()

    def finalize_options(self):
        build_ext.finalize_options(self)

    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        match = re.search(r"version\s*(?P<major>\d+)\.(?P<minor>\d+)([\d.]+)?", out.decode())
        cmake_major, cmake_minor = int(match.group("major")), int(match.group("minor"))
        if (cmake_major, cmake_minor) < (3, 18):
            raise RuntimeError("CMake >= 3.18.0 is required")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        ninja_dir = shutil.which('ninja')
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.path)))
        # create build directories
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        # third party packages
        thirdparty_cmake_args = get_thirdparty_packages([get_llvm_package_info()])
        cmake_args = [
            "-G", "Ninja",  # Ninja is much faster than make
            "-DCMAKE_MAKE_PROGRAM=" +
            ninja_dir,  # Pass explicit path to ninja otherwise cmake may cache a temporary path
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", "-DLLVM_ENABLE_WERROR=ON",
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir, 
            "-DAILANG_BUILD_PYTHON_MODULE=ON", "-DPython3_EXECUTABLE:FILEPATH=" + sys.executable,
            "-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON",  
        ]
        cmake_args.extend(thirdparty_cmake_args)

        # configuration
        cfg = get_build_type()
        build_args = ["--config", cfg]

        cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
        max_jobs = os.getenv("MAX_JOBS", str(2 * os.cpu_count()))
        build_args += ['-j' + max_jobs]

        if check_env_flag("AILANG_BUILD_WITH_CLANG_LLD"):
            cmake_args += [
                "-DCMAKE_C_COMPILER=clang",
                "-DCMAKE_CXX_COMPILER=clang++",
                "-DCMAKE_LINKER=lld",
                "-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld",
                "-DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld",
                "-DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld",
            ]

        if check_env_flag("AILANG_BUILD_WITH_CCACHE"):
            cmake_args += [
                "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
            ]

        env = os.environ.copy()
        cmake_dir = get_cmake_dir()
        subprocess.check_call(["git", "submodule", "update", "--init", "--recursive"], cwd=cmake_dir, env=env)
        subprocess.check_call(["cmake", self.base_dir] + cmake_args, cwd=cmake_dir, env=env)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=cmake_dir)
        subprocess.check_call(["cmake", "--build", ".", "--target", "mlir-doc"], cwd=cmake_dir)

# ---- package data ---
class Package(NamedTuple):
    package: str
    include_flag: str
    lib_flag: str
    syspath_var_name: str

# llvm
def get_llvm_package_info():
    # system_suffix = 'ubuntu-x64'
    # llvm_hash_path = os.path.join(get_base_dir(), "cmake", "llvm-hash.txt")
    # with open(llvm_hash_path, "r") as llvm_hash_file:
    #     rev = llvm_hash_file.read(8)
    return Package("llvm-project", "LLVM_INCLUDE_DIRS", "LLVM_LIBRARY_DIR", "LLVM_SYSPATH")

def get_ailang_cache_path():
    return os.path.join(get_base_dir(), "../third_party")

def get_thirdparty_packages(packages: list):
    ailang_cache_path = get_ailang_cache_path()
    thirdparty_cmake_args = []
    for p in packages:
        package_root_dir = os.path.join(ailang_cache_path, p.package)
        package_dir = os.path.join(package_root_dir, p.package)
        if os.environ.get(p.syspath_var_name):
            package_dir = os.environ[p.syspath_var_name]
        if p.include_flag:
            thirdparty_cmake_args.append(f"-D{p.include_flag}={package_dir}/include")
        if p.lib_flag:
            thirdparty_cmake_args.append(f"-D{p.lib_flag}={package_dir}/lib")
    return thirdparty_cmake_args

def get_packages():
    packages = [
        "ailang",
        "ailang/_C",
        "ailang/compiler",
    ]
    return packages


def get_entry_points():
    entry_points = {}
    return entry_points


setup(
    name="ailang",
    version="0.1",
    description="A language and compiler for custom Deep Learning operations",
    long_description="",
    packages=get_packages(),
    entry_points=get_entry_points(),
    ext_modules=[CMakeExtension("ailang", "ailang/_C/")],
    cmdclass={
        "build_ext": CMakeBuild,
        "build_py": CMakeBuildPy,
    },
)