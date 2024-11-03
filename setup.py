import sysconfig
import sys
import os
from setuptools import setup, find_packages, Extension
from pathlib import Path
from typing import List
from setuptools.command.build_ext import build_ext
import subprocess
import re
import shutil

def get_torch_root():
    try:
        import torch

        return str(Path(torch.__file__).parent)
    except ImportError:
        return None

def use_cxx11_abi():
    try:
        import torch

        return torch._C._GLIBCXX_USE_CXX11_ABI
    except ImportError:
        return False

def get_cmake_dir():
    plat_name = sysconfig.get_platform()
    python_version = sysconfig.get_python_version().replace(".", "")
    dir_name = f"cmake.{plat_name}-{sys.implementation.name}-{python_version}"
    cmake_dir = Path(get_base_dir()) / "build" / dir_name
    cmake_dir.mkdir(parents=True, exist_ok=True)
    return cmake_dir

def get_version() -> str:
    version = os.getenv("DXZ_VERSION")
    if not version:
        with open("version.txt", "r") as f:
            version = f.read().strip()
    
    if version and version.startswith("v"):
        version = version[1:]
    
    if not version:
        raise RuntimeError("Unable to find version string.")
    
    version_suffix = os.getenv("DXZ_VERSION_SUFFIX")
    if version_suffix:
        version += version_suffix
    return version

def get_base_dir():
    return os.path.abspath(os.path.dirname(__file__))

def join_path(*paths):
    return os.path.join(get_base_dir(), *paths)

def read_readme() -> str:
    p = join_path("README.md")
    with open(p, 'r', encoding='utf-8') as f:
        readme = f.read()
    return readme

def read_requirements() -> List[str]:
    p = join_path("requirements.txt")
    with open(p) as f:
        requirements = f.read().splitlines()
    return requirements

class CMakeExtension(Extension):
    def __init__(self, name: str, path: str):
        super().__init__(name=name, sources=[]) # the name and sources is used by build_ext
        self.path = path
class CMakeBuild(build_ext):
    user_options = build_ext.user_options + [
        ("base-dir=", None, "base directory of project"),
    ]

    def initialize_options(self):
        build_ext.initialize_options(self)
        self.base_dir = get_base_dir()
    
    def finalize_options(self):
        build_ext.finalize_options(self)
    
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                f"CMake must be installed to build the following extensions: {', '.join(e.name for e in self.extensions) }"
            )
        match = re.search(
            r"version\s*(?P<major>\d+)\.(?P<minor>\d+)([\d.]+)?", out.decode()
        )
        cmake_major, cmake_minor = int(match.group("major")), int(match.group("minor"))
        if (cmake_major, cmake_minor) < (3, 18):
            raise RuntimeError("CMake >= 3.18.0 is required")
        for ext in self.extensions:
            self.build_extension(ext)
    
    def build_extension(self, ext: CMakeExtension):
        print('start build ============================================================')
        print(f'ext.path      : {ext.path}')

        ninja_dir = shutil.which("ninja")
        print(f'ninja_dir: {ninja_dir}')

        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.path)))
        print(f'extdir   : {extdir}')

        # create build directory
        print(f'self.build_temp {self.build_temp}')
        os.makedirs(self.build_temp, exist_ok=True)

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        build_type = "Debug" if debug else "Release"
        print(f'build_type {build_type}')

        so_output_path = join_path(extdir, "dxz", "_C", "kernel")
        print(f'so_output_path {so_output_path}')
        cuda_architectures = "80;89;90"
        cmake_args = [
            "-G",
            "Ninja",  # Ninja is much faster than make
            f"-DCMAKE_MAKE_PROGRAM={ninja_dir}",  # pass in the ninja build path
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={so_output_path}",
            "-DUSE_CCACHE=ON",  # use ccache if available
            "-DUSE_MANYLINUX:BOOL=ON",  # use manylinux settings
            f"-DPython_EXECUTABLE:FILEPATH={sys.executable}",
            f"-DCMAKE_CUDA_ARCHITECTURES={cuda_architectures}",
            f"-DCMAKE_BUILD_TYPE={build_type}",  # not used on MSVC, but no harm
        ]
        # check if torch binary is built with cxx11 abi
        if use_cxx11_abi():
            cmake_args += ["-DUSE_CXX11_ABI=ON"]
        else:
            cmake_args += ["-DUSE_CXX11_ABI=OFF"]


        env = os.environ.copy()
        LIBTORCH_ROOT = get_torch_root()
        if LIBTORCH_ROOT is None:
            raise RuntimeError(
                "Please install requirements first, pip install -r requirements.txt"
            )
        print(f'LIBTORCH_ROOT {LIBTORCH_ROOT}')
        env["LIBTORCH_ROOT"] = LIBTORCH_ROOT

        print("CMake Args: ", cmake_args)
        print("Env: ", env)

        cmake_dir = get_cmake_dir()
        print(f'cmake_dir {cmake_dir}')

        subprocess.check_call(
            ["cmake", self.base_dir] + cmake_args, cwd=cmake_dir, env=env
        ) # this command means first cd cmake_dir then export env cmake self.base_dir cmake_args

        build_args = ["--config", build_type]
        max_jobs = os.getenv("MAX_JOBS", str(os.cpu_count()))
        build_args += ["-j" + max_jobs]
        # add build target to speed up the build process
        build_args += ["--target", "flash_attn", "flash_infer", "kv_cache_kernels"]
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=cmake_dir)

        print('end build ============================================================')
            


if __name__ == '__main__':
    setup(
        name='dxz',
        version=get_version(),
        license="Apache 2.0",
        author="Xianzhe Dong",
        author_email="dongxianzhe2019@gmail.com",
        description="a llm inference engine for academic research",
        long_description=read_readme(), 
        long_description_content_type="text/markdown",
        url="https://github.com/dongxianzhe/dxz",
        project_url={
            "Homepage": "https://github.com/dongxianzhe/dxz",
        },
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Programming Language :: C++",
            "Programming Language :: Python :: 3 :: Only",
            "Programming Language :: Python :: 3.10",
            "Environment :: GPU :: NVIDIA CUDA",
            "Operating System :: POSIX",
            "License :: OSI Approved :: Apache Software License",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ], # classifiers will be displayed in PyPi website
        packages = find_packages() + ['dxz._C.kernel.flash_attn', 'dxz._C.kernel.flash_infer', 'dxz._C.kernel.kv_cache_kernels'], # find all directory with __init__.py
        ext_modules=[CMakeExtension('_C', "csrc")],
        cmdclass={"build_ext": CMakeBuild}, 
        zip_safe=False,  # package is not safe with zip format
        python_requires=">=3.10", # requires python version >= 3.10
        install_requires=read_requirements(),  # pip install will automatically install requirements.txt
    )