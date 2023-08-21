#!/usr/bin/env python
"""Setup watertight_transformer."""

from itertools import dropwhile
from setuptools import find_packages, setup
from os import path

import numpy as np

from Cython.Build import cythonize
from Cython.Distutils import build_ext
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
    with open(path.join("watertight_transformer", "__init__.py")) as f:
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
            "watertight_transformer.external.libmesh.triangle_hash",
            sources=["watertight_transformer/external/libmesh/triangle_hash.pyx"],
            include_dirs=[np.get_include()],
            libraries=["m"]  # Unix-like specific
        ),
        Extension(
            "watertight_transformer.external.librender.pyrender",
            sources=[
                "watertight_transformer/external/librender/pyrender.pyx",
                "watertight_transformer/external/librender/offscreen.cpp"
            ],
            language="c++",
            include_dirs=[np.get_include()],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            libraries=["m"]  # Unix-like specific
        ),
        Extension(
            "watertight_transformer.external.libmcubes.mcubes",
            sources=[
                "watertight_transformer/external/libmcubes/mcubes.pyx",
                "watertight_transformer/external/libmcubes/pywrapper.cpp",
                "watertight_transformer/external/libmcubes/marchingcubes.cpp"
            ],
            language="c++",
            include_dirs=[np.get_include()],
            extra_compile_args=["-std=c++11"],
            libraries=["m"]  # Unix-like specific
        ),
        Extension(
            "watertight_transformer.external.libfusioncpu.cyfusion",
            sources=[
                "watertight_transformer/external/libfusioncpu/cyfusion.pyx",
                "watertight_transformer/external/libfusioncpu/fusion.cpp"
            ],
            language="c++",
            libraries=["m"],
            include_dirs=[np.get_include()],
            extra_compile_args=[
                "-fopenmp", "-ffast-math", "-msse", "-msse2", "-msse3", "-msse4.2"
            ],
            extra_link_args=["-fopenmp"]
        ),
        #Extension(
        #    "watertight_transformer.external.libfusiongpu.cyfusion",
        #    sources=[
        #        "watertight_transformer/external/libfusiongpu/cyfusion.pyx",
        #    ],
        #    language="c++",
        #    library_dirs=["watertight_transformer/external/libfusiongpu/build/"],
        #    libraries=["m", "fusion_gpu"],
        #    include_dirs=[np.get_include()],
        #    extra_compile_args=[
        #        "-ffast-math", "-msse", "-msse2", "-msse3", "-msse4.2"
        #    ]
        #),
    ])


def get_install_requirements():
    return [
        "numpy",
        "scipy",
        "cython",
        "pycollada",
        "Pillow",
        "trimesh",
        "tqdm",
        "h5py",
        "pymeshlab"
    ]

def setup_package():
    with open("README.md") as f:
        long_description = f.read()
    meta = collect_metadata()
    setup(
        name="watertight_transformer",
        cmdclass={"build_ext": build_ext},
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
