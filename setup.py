"""Setup configuration for LLM_Criteria_Gemma."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="llm-criteria-gemma",
    version="0.1.0",
    author="Oscar Tsao",
    author_email="",
    description="Gemma Encoder for DSM-5 Criteria Matching using ReDSM5 Dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/OscarTsao/LLM_Criteria_Gemma",
    project_urls={
        "Bug Tracker": "https://github.com/OscarTsao/LLM_Criteria_Gemma/issues",
        "Documentation": "https://github.com/OscarTsao/LLM_Criteria_Gemma/blob/main/README.md",
        "Source Code": "https://github.com/OscarTsao/LLM_Criteria_Gemma",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-xdist>=3.3.0",
            "black>=23.0.0",
            "flake8>=6.1.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
        ],
        "optim": [
            "optuna>=3.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gemma-train=training.train_gemma_hydra:main",
            "gemma-eval=training.evaluate:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
