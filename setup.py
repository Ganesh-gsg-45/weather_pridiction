from setuptools import setup, find_packages

def get_requirements(file_path: str) -> list[str]:
    """Read requirements.txt and return list of dependencies."""
    requirements = []
    with open(file_path) as f:
        for line in f:
            req = line.strip()
            # Skip empty lines, comments, and editable installs
            if req and not req.startswith("#") and req != "-e .":
                requirements.append(req)
    return requirements


setup(
    name="weatherprediction",
    version="0.0.1",
    author="T Ganesh",
    description="Weather Prediction MLOps Pipeline — Rain Tomorrow Classifier",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=get_requirements("requirements.txt"),
    python_requires=">=3.8",
)
