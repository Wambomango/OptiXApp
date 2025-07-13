from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension
from torch.utils.cpp_extension import include_paths, library_paths


include_dirs=include_paths()
library_dirs=library_paths()
libraries=["c10", "torch", "torch_cpu", "torch_python", "nvrtc"]

include_dirs.append("src")
libraries.append("MWIR")
library_dirs.append("build/src/mwir")
extra_objects=["build/src/mwir/libMWIR.a"]

ext_modules = [
    Pybind11Extension(
        "ManyWorldsInverseRadar",
        ["src/mwir_python/many_worlds_inverse_radar.cpp"],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        extra_objects=extra_objects,
    ),
]

setup(
    name="ManyWorldsInverseRadar",
    version="0.0.1",
    description="ManyWorldsInverseRadar python module",
    ext_modules=ext_modules
)


