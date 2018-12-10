import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Bambanta",
    version="0.0.4",
    author="Karina Huang, Rong Liu, Rory Maizels",
    author_email="qhuang@g.harvard.edu, rongliu@g.harvard.edu, rorymaizels@g.harvard.edu",
    description="An automatic differentiation package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CS207-Project-Group-9/cs207-FinalProject",
    packages=setuptools.find_packages("Bambanta"),
    package_dir = {"":"Bambanta"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
