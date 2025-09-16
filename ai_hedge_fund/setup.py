"""
Setup configuration for AI Hedge Fund System
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    requirements = []
    with open(requirements_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                requirements.append(line)
else:
    requirements = []

setup(
    name="ai-hedge-fund",
    version="1.0.0",
    description="Professional AI hedge fund system using AutoGen framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="AI Hedge Fund Team",
    author_email="team@aihedgefund.com",
    url="https://github.com/your-org/ai-hedge-fund",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=23.7.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.11.0",
        ],
        "ml": [
            "tensorflow>=2.13.0",
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "scikit-learn>=1.3.0",
        ],
        "backtesting": [
            "backtrader>=1.9.78.123",
            "zipline-reloaded>=2.2.0",
            "pyfolio>=0.9.2",
            "empyrical>=0.5.5",
        ],
        "data": [
            "yfinance>=0.2.18",
            "alpha-vantage>=2.3.1",
            "polygon-api-client>=1.10.0",
            "pandas-datareader>=0.10.0",
            "tweepy>=4.14.0",
        ],
        "visualization": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.15.0",
            "jupyter>=1.0.0",
        ],
        "production": [
            "gunicorn>=21.2.0",
            "docker>=6.1.0",
            "sentry-sdk>=1.29.0",
            "prometheus-client>=0.17.0",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="hedge fund, algorithmic trading, artificial intelligence, autogen, quantitative finance",
    entry_points={
        "console_scripts": [
            "ai-hedge-fund=ai_hedge_fund.cli:main",
            "hedge-fund-server=ai_hedge_fund.server:main",
            "hedge-fund-backtest=ai_hedge_fund.backtesting.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "ai_hedge_fund": [
            "config/*.yaml",
            "strategies/*/*.json",
            "strategies/*/prompts/*.txt",
            "notebooks/*.ipynb",
        ],
    },
    zip_safe=False,
    project_urls={
        "Bug Reports": "https://github.com/your-org/ai-hedge-fund/issues",
        "Source": "https://github.com/your-org/ai-hedge-fund",
        "Documentation": "https://ai-hedge-fund.readthedocs.io/",
    },
)