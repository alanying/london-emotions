from setuptools import find_packages
from setuptools import setup

# with open('requirements.txt') as f:
#     content = f.readlines()
# requirements = [x.strip() for x in content if 'git+' not in x]

REQUIRED_PACKAGES = [
      'pip>=9',
      'setuptools>=26',
      'wheel>=0.29',
      'nltk==3.5',
      'pytest',
      'coverage',
      'flake8',
      'black',
      'yapf',
      'python-gitlab',
      'twine',
      'mlflow',
      'memoized_property',
      'google-cloud-storage==1.26.0',
      'pandas==0.24.2',
      'pandas-profiling==2.9.0',
      'scikit-image==0.17.2',
      'scikit-learn==0.20.4',
      'gcsfs==0.6.0',
      'six==1.13.0',
      'joblib==0.14.1',
      'numpy==1.18.4',
      'gensim',
      'python-dotenv'
]

setup(name='london-emotions',
      version="1.0",
      description="Project Description",
      install_requires=REQUIRED_PACKAGES,
      packages=find_packages(),
      test_suite = 'tests',
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      scripts=['scripts/london-emotions-run'],
      zip_safe=False)
