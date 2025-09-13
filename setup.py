from setuptools import find_packages,setup
from typing import List

ignoreln = '-e .'
def get_requirements(filepath:str)->List[str]:
    requirements = []
    with open(filepath) as file1:
        requirements = file1.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        
        if ignoreln in requirements:
            requirements.remove(ignoreln)
    return requirements

setup(
name='stroke-prediction',
version='0.0.1',
author='Sreerag',
author_email='sreerag3ngu@gmail.com',
packages=find_packages(),
install_required=get_requirements('requirements.txt')
)