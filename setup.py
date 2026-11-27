"""
Setup script for the LLM-RAG system for Indian jurisprudence.
"""

from setuptools import setup, find_packages


def read_requirements(path):
    with open(path, "r", encoding="utf-8") as fh:
        return [line.split("#")[0].strip() for line in fh
                if line.strip() and not line.startswith(("#", "-r"))]


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="legal-llm-rag",
    version="1.1.0",
    description="LLM + RAG legal assistant for Indian law (BNS/BNSS-aware)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.9",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "train": read_requirements("requirements-train.txt"),
        "dev": ["black>=23.0", "flake8>=6.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    include_package_data=True,
    package_data={"": ["*.yaml", "*.json"]},
)
