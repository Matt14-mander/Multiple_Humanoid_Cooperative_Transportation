# Internal Force Suppression Module - Setup

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "src" / "internal_force_suppression" / "README.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")

setup(
    name="internal_force_suppression",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Internal Force Suppression Module for Dual-Robot Cooperative Manipulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/internal-force-suppression",

    # Package configuration
    package_dir={"": "src"},
    packages=find_packages(where="src"),

    # Dependencies
    install_requires=[
        "numpy>=1.20.0",
        "pyyaml>=5.4.0",
        # Note: pinocchio needs to be installed via conda
        # conda install pinocchio -c conda-forge
    ],

    # Optional dependencies
    extras_require={
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
        ],
        "viz": [
            "matplotlib>=3.5.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "matplotlib>=3.5.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },

    # Package data
    package_data={
        "internal_force_suppression": [
            "config/*.yaml",
        ],
    },

    # Metadata
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Robotics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],

    python_requires=">=3.8",
)
