from setuptools import setup

with open("README.md", "r") as fh:
    readme = fh.read()

with open("LICENSE", "r") as fh:
    license = fh.read()

setup(
        name='sparsely_activated_networks_pytorch',
        version='0.1',
        description='Sparsely activated networks for Pytorch',
        long_description=readme,
        long_description_content_type="text/markdown",
        license=license,
        url='https://github.com/pbizopoulos/sparsely_activated_networks_pytorch',
        author='Paschalis Bizopoulos',
        author_email='pbizopoulos@protonmail.com',
        classifiers=[
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3',
            ],
        py_modules=['sparsely_activated_networks_pytorch'],
        python_requires='>=3.6',
        install_requires=['torch'],
        )
