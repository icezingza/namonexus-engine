"""
NamoNexus Fusion Engine — Package Setup
Patent-Pending Technology | NamoNexus Research Team
"""

from setuptools import setup, find_packages
import os

# Read README for long description
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="namonexus-fusion",
    version="4.0.0",
    description=(
        "Patent-pending multimodal Bayesian fusion engine anchored on the Golden Ratio. "
        "Supports temporal decay, empirical prior learning, modality auto-calibration, "
        "sensor trust scoring, drift detection, streaming inference, Shapley XAI, "
        "and hierarchical Bayesian federated learning."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="NamoNexus Research Team",
    author_email="research@namonexus.ai",
    url="https://github.com/namonexus/namonexus-fusion",
    license="Proprietary",
    license_files=["LICENSE", "NOTICE.txt"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.9",
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    install_requires=[
        "numpy>=1.26.0,<3.0",
        "scipy>=1.11.0,<2.0",
        "numba>=0.59.0,<1.0",
        "cryptography>=43.0.1,<47.0.0",
        "aiofiles>=23.2.1,<25.0.0",
        "pydantic>=2.9.0,<3.0.0",
        "pydantic-settings>=2.2.1,<3.0.0",
        "cachetools>=5.3.0,<7.0.0",
        "fastapi>=0.116.0,<1.0.0",
        "starlette>=0.47.2,<1.0.0",
        "uvicorn>=0.30.0,<1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-asyncio>=0.21",
            "pytest-cov>=4.0",
            "black>=23.0",
            "isort>=5.12",
            "mypy>=1.5",
            "ruff>=0.1",
            "pip-audit>=2.7",
        ],
        "streaming": [
            "kafka-python>=2.0",
            "websockets>=11.0",
        ],
        "all": [
            "kafka-python>=2.0",
            "websockets>=11.0",
        ],
    },
    entry_points={
        "console_scripts": [
            # Optional CLI for quick testing
            # "namonexus=namonexus_fusion.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
