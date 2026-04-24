from pathlib import Path
import re

from setuptools import setup, find_packages


def get_version() -> str:
    version_file = Path(__file__).resolve().parent / "darkbottomline" / "_version.py"
    match = re.search(r'__version__\s*=\s*"([^"]+)"', version_file.read_text())
    if not match:
        raise RuntimeError(f"Unable to read version from {version_file}")
    return match.group(1)

# Normalize version format for setuptools compatibility
setup(
    name="darkbottomline",
    version=get_version(),
    description="Modular Coffea-based analysis framework for CMS Run 3 bbMET analysis",
    author="DarkBottomLine Team",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        # Core scientific
        "numpy>=1.24.3",
        "scipy>=1.10.0",
        "matplotlib>=3.8.4",
        "pandas>=2.0.1",
        # Physics analysis
        "awkward>=2.8.11",
        "uproot>=5.6.9",
        "correctionlib>=2.6.4",
        "coffea>=2025.12.0",
        "fsspec-xrootd>=0.2.2",
        "xrootd>=5.7.2",
        # Distributed computing
        "dask[complete]>=2024.4.2",
        "distributed>=2024.4.2",
        # Machine learning
        "torch>=2.0.0",
        "scikit-learn>=1.2.2",
        # Histogramming and plotting
        "hist>=2.7.2",
        "plotly>=5.0.0",
        # Data handling
        "pyarrow>=17.0.0",
        # Configuration
        "pyyaml>=6.0.1",
        # Utilities
        "tqdm>=4.67.1",
        "memory-profiler>=0.60.0",
    ],
    entry_points={
        "console_scripts": [
            "darkbottomline=darkbottomline.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
