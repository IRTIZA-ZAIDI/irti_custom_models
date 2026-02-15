from setuptools import setup, find_packages

setup(
    name="irti_custom",
    version="0.1.0",
    description="Custom HF-compatible AutoModelForMaskedSeqModeling",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers>=4.35.0",
    ],
    python_requires=">=3.9",
)
