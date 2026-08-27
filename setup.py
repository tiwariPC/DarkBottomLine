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
        "numpy==2.2.6",
        "scipy==1.16.3",
        "matplotlib==3.10.8",
        "pandas==2.2.3",
        # Physics analysis
        "awkward==2.8.9",
        "uproot==5.6.0",
        "correctionlib==2.6.4",
        "coffea==2025.12.0",
        "fsspec-xrootd==0.5.1",
        # xrootd is intentionally NOT listed here: it's provided as a
        # conda-forge binary via environment.yml (xrootd=5.9.1), which
        # doesn't register as pip-visible, so pip would otherwise try to
        # build it from source (requires cmake) and fail. See local_setup.sh.
        # Distributed computing
        "dask==2025.2.0",
        "distributed==2025.2.0",
        # Machine learning
        "torch==2.10.0",
        "scikit-learn==1.8.0",
        # Histogramming and plotting
        "hist==2.9.0",
        "plotly==5.16.1",
        # Data handling
        "pyarrow==20.0.0",
        # Configuration
        "pyyaml==6.0.2",
        # Utilities
        "tqdm==4.67.1",
        "memory-profiler==0.61.0",
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
        "Programming Language :: Python :: 3.13",
    ],
)
