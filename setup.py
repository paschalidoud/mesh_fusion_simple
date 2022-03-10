#!/usr/bin/env python
"""Setup mesh_fusion."""

from itertools import dropwhile
from setuptools import find_packages, setup
from os import path

import numpy as np

from Cython.Build import cythonize
from distutils.extension import Extension


def collect_docstring(lines):
    """Return document docstring if it exists"""
    lines = dropwhile(lambda x: not x.startswith('"""'), lines)
    doc = ""
    for line in lines:
        doc += line
        if doc.endswith('"""\n'):
            break

    return doc[3:-4].replace("\r", "").replace("\n", " ")


def collect_metadata():
    meta = {}
    with open(path.join("mesh_fusion", "__init__.py")) as f:
        lines = iter(f)
        meta["description"] = collect_docstring(lines)
        for line in lines:
            if line.startswith("__"):
                key, value = map(lambda x: x.strip(), line.split("="))
                meta[key[2:-2]] = value[1:-1]

    return meta


def get_extensions():
    extra_compile_args = [
        "-ffast-math",
        "-msse",
        "-msse2",
        "-msse3",
        "-msse4.2",
        "-O4",
        "-fopenmp"
    ]
    extra_link_args = [
        "-lGLEW",
        "-lglut",
        "-lGL",
        "-lGLU",
        "-fopenmp"
    ]
    return cythonize([
        Extension(
            "mesh_fusion.external.libmesh.triangle_hash",
            sources=["mesh_fusion/external/libmesh/triangle_hash.pyx"],
            include_dirs=[np.get_include()],
            libraries=["m"]  # Unix-like specific
        ),
        Extension(
            "mesh_fusion.external.librender.pyrender",
            sources=[
                "mesh_fusion/external/librender/pyrender.pyx",
                "mesh_fusion/external/librender/offscreen.cpp"
            ],
            language="c++",
            include_dirs=[np.get_include()],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            libraries=["m"]  # Unix-like specific
        ),
        Extension(
            "mesh_fusion.external.libmcubes.mcubes",
            sources=[
                "mesh_fusion/external/libmcubes/mcubes.pyx",
                "mesh_fusion/external/libmcubes/pywrapper.cpp",
                "mesh_fusion/external/libmcubes/marchingcubes.cpp"
            ],
            language="c++",
            include_dirs=[np.get_include()],
            extra_compile_args=["-std=c++11"],
            libraries=["m"]  # Unix-like specific
        ),
        Extension(
            "mesh_fusion.external.libfusioncpu.cyfusion",
            sources=[
                "mesh_fusion/external/libfusioncpu/cyfusion.pyx",
            ],
            language="c++",
            library_dirs=["mesh_fusion/external/libfusioncpu/build/"],
            libraries=["m", "fusion_cpu"],
            include_dirs=[np.get_include()],
            extra_compile_args=[
                "-ffast-math", "-msse", "-msse2", "-msse3", "-msse4.2"
            ]
        ),
    ])


def get_install_requirements():
    return [
        "numpy",
        "cython",
        "pycollada",
        "Pillow",
        "trimesh",
        "tqdm"
    ]

def setup_package():
    with open("README.md") as f:
        long_description = f.read()
    meta = collect_metadata()
    setup(
        name="mesh_fusion",
        version=meta["version"],
        description=meta["description"],
        long_description=long_description,
        maintainer=meta["maintainer"],
        maintainer_email=meta["email"],
        url=meta["url"],
        license=meta["license"],
        classifiers=[
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: MIT License",
            "Topic :: Scientific/Engineering",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
        ],
        packages=find_packages(exclude=["scripts"]),
        install_requires=get_install_requirements(),
        ext_modules=get_extensions()
    )


if __name__ == "__main__":
    setup_package()

