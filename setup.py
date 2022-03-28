from setuptools import find_packages, setup

install_requires = (
    "transformers",
    "pandas",
    "numpy",
    "tqdm",
)

excludes = (
    "*test*",
    "*local_settings*",
)

setup(
    name="lmtools",
    version="0.1",
    license="MIT",
    description="Tools for OTR with language models.",
    author="Taylor Sorensen",
    author_email="tsor1313@gmail.com",
    url="http://github.com/tsor13/lmtools",
    packages=find_packages(exclude=excludes),
    entry_points={
        "console_scripts": [
            "lm-pipeline = lmtools.pipeline:main",
            "lm-postprocess = lmtools.postprocessor:main",
        ]
    },
    install_requires=install_requires,
)
