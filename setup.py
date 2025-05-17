import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

VERSION = "0.1.0"

INSTALL_REQUIRES = [
    "torch>=2.6.0+cu124",
    "datasets>=3.6.0",
    "transformers>=4.51.3",
    "huggingface_hub>=0.31.1",
    "wandb>=0.19.11",
    "accelerate>=1.6.0",
    "bitsandbytes>=0.45.5"
]

EXTRAS_REQUIRE = {
    "flash_attn": ["flash-attn>=2.7.4"]
}


setuptools.setup(
    name="adaptive-trainer",
    version=VERSION,
    author="Samarth Pusalkar",
    author_email="samarthpusalkar@gmail.com",
    description="A Hugging Face Trainer extension for adaptive loss and ideas learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/samarthpusalkar/Adaptive_Trainer",
    project_urls={
        "Bug Tracker": "https://github.com/samarthpusalkar/Adaptive_Trainer/issues",
    },
    license=None,
    packages=setuptools.find_packages(
        exclude=["tests*", "examples*"]
    ),
    classifiers=[
        "Development Status :: 0 - Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: Not mentioned yet please contact at samarthpusalkar@gmail.com",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8", # Minimum Python version
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    keywords="nlp, machine learning, deep learning, transformers, huggingface, trainer, adaptive loss, ideas learning",
)
