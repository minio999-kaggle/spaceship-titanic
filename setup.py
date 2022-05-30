from setuptools import setup, find_packages

with open('requirements.txt') as file:
    required = file.read().splitlines()

with open("LICENSE") as file:
    license = file.read() 


setup(
    name="app",
    version='0.1.0',
    description='Predict which passengers are transported to an alternate dimension',
    author='Dominik Mińkowski',
    author_email='minkowskidominik03@gmail.com',
    url='',
    license=license,
    include_package_data=True,
    package_dir={"":"src", "tests":"tests"},
    packages=find_packages(),
    install_requires=required
)