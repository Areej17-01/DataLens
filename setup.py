from setuptools import setup, find_packages

setup(
    name="agentpro",
    version="0.1.0",
    description="AgentPro: Traversaal Agentic Framework",
    author="Traversaal AI",
    author_email="contact@traversaal.ai",
    url="https://github.com/traversaal-ai/AgentPro",
    packages=find_packages(),  # automatically finds all modules in agentpro/
    install_requires=[
        "openai",
        "youtube_transcript_api",
        "duckduckgo-search",
        "requests",
        "python-pptx",
        "pydantic",
        "python-dotenv",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "openpyxl",
        "pyarrow",
        "scikit-learn",
        "yfinance",
        "litellm"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
