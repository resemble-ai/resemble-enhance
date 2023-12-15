import subprocess
from datetime import datetime, timezone
from pathlib import Path

from setuptools import find_packages, setup


def shell(*args):
    out = subprocess.check_output(args)
    return out.decode("ascii").strip()


def write_version(version_core, pre_release=True):
    if pre_release:
        last_commit_time = shell("git", "log", "-1", "--format=%cd", "--date=iso-strict")
        last_commit_time = datetime.strptime(last_commit_time, "%Y-%m-%dT%H:%M:%S%z")
        last_commit_time = last_commit_time.astimezone(timezone.utc)
        last_commit_time = last_commit_time.strftime("%y%m%d%H%M%S")
        version = f"{version_core}-dev{last_commit_time}"
    else:
        version = version_core

    with open(Path("resemble_enhance", "version.py"), "w") as f:
        f.write('__version__ = "{}"\n'.format(version))

    return version


with open("README.md", "r") as f:
    long_description = f.read()


with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="resemble-enhance",
    python_requires=">=3.10",
    version=write_version("0.0.2", pre_release=True),
    description="Speech denoising and enhancement with deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=requirements,
    url="https://github.com/resemble-ai/resemble-enhance",
    author="Resemble AI",
    author_email="team@resemble.ai",
    entry_points={
        "console_scripts": [
            "resemble-enhance=resemble_enhance.enhancer.__main__:main",
        ]
    },
)
