from setuptools import setup, find_packages
from typing import List

def get_requirements() -> List[str]:
    """
    This function reads a requirements file and returns a list of requirements.
    """
    try:
        with open("requirements.txt",'r') as file:
            requirements = file.readlines()
    
        # Remove any leading/trailing whitespace characters
        requirements = [req.strip() for req in requirements]
    
        # Remove any version specifiers (e.g., 'package==1.0.0')
        requirements = [req.split('==')[0] for req in requirements if req and not req.startswith('#')]

        # Remove empty lines and -e .
        requirements = [req for req in requirements if req and not req.startswith('-e .')]
    except FileNotFoundError:
        print("requirements.txt file not found. Please ensure it exists in the current directory.")
    
    return requirements
print(get_requirements())
setup(
    name='NetworkSecurity',
    version='0.0.1',
    author='Lakshya',
    author_email='lakshyabhatnagar1@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements()
)
