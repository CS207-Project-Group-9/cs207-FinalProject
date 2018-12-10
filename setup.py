import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(name='Bambanta',
	version='0.1.1',
	description='An automatic differentiation package',
	author='Karina Huang, Rong Liu, Rory Maizels',
	long_description=long_description,
	long_description_content_type="text/markdown",
	packages=['Bambanta'],
	install_requires=[
	  'numpy',
	  'math',
	],
	classifiers=[
		"Development Status :: 5 - Production/Stable",
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
	],
	url="https://github.com/CS207-Project-Group-9/cs207-FinalProject",
	license="MIT",
)
