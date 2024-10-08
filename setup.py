import io
import os
import re
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import List
from jinja2 import Template

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


def use_cxx11_abi():
    try:
        import torch

        return torch._C._GLIBCXX_USE_CXX11_ABI
    except ImportError:
        return False


def get_torch_root():
    try:
        import torch
        return str(Path(torch.__file__).parent)
    except ImportError:
        return None

class CMakeExtension(Extension):
    def __init__(self, name: str) -> None: # name is cmake target
        super().__init__(name, sources=[])

class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext: CMakeExtension):
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=/home/xzd/projects/dxz/build/lib.linux-x86_64-cpython-310", 
            "-DUSE_CXX11_ABI=ON" if use_cxx11_abi() else "-DUSE_CXX11_ABI=OFF"]

        env = os.environ.copy()
        LIBTORCH_ROOT = get_torch_root()
        if LIBTORCH_ROOT is None:
            raise RuntimeError("Please install requirements first, pip install -r requirements.txt")
        env["LIBTORCH_ROOT"] = LIBTORCH_ROOT
        
        cmake_dir = "/home/xzd/projects/dxz/build/cmake.linux-x86_64-cpython-310"
        os.makedirs(cmake_dir, exist_ok=True)
        subprocess.check_call(["cmake", '../../'] + cmake_args, cwd=cmake_dir, env=env)
        subprocess.check_call(["cmake", "--build", ".", "--target", f"{ext.name}"], cwd=cmake_dir)

if __name__ == "__main__":
    setup(
        name="dxz",
        version="1.0",
        license="Apache 2.0",
        author="dongxianzhe",
        author_email="dongxianzhe2019@gmail.com",
        description="a llm inference engine for academic research",
        packages=["dxz", "dxz/engine", "dxz/", "dxz/entrypoint", "dxz/model"], # setup tools will copy the file in the directory to .whl package
        ext_modules=[CMakeExtension("gemm")], 
        cmdclass={"build_ext": CMakeBuild},
        python_requires=">=3.8",
    )