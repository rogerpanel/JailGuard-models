from pathlib import Path
from setuptools import find_packages, setup

ROOT = Path(__file__).parent

setup(
    name="ct_dgnn_jailguard",
    version="1.0.0",
    description=(
        "CT-DGNN-JailGuard: Continuous-Time Dynamic Graph Neural Networks with "
        "Certified Robustness for Real-Time Detection of LLM Jailbreak Campaigns "
        "via API Interaction Graphs"
    ),
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Roger Nick Anaedevha",
    license="MIT",
    url="https://github.com/rogerpanel/CV/tree/main/ct_dgnn_jailguard",
    python_requires=">=3.10",
    packages=find_packages(exclude=("tests", "notebooks", "scripts")),
    install_requires=(ROOT / "requirements.txt").read_text().splitlines(),
    entry_points={
        "console_scripts": [
            "ctdgnn-train=scripts.train:main",
            "ctdgnn-eval=scripts.evaluate:main",
            "ctdgnn-certify=scripts.certify:main",
            "ctdgnn-build-jailcampaign=scripts.build_jail_campaign:main",
        ],
    },
)
