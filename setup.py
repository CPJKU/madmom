from setuptools import setup

setup(name='python_cpjku',
      version='0.01',
      description='Python package used at cp.jku.at',
      long_description=open('README').read(),
      #url='http://github.com/CPJKU/python_cpjku',
      author='Department of Computational Perception, Johannes Kepler University, Linz, Austria',
      author_email='python_cpjku@jku.at',
      license='BSD',
      packages=['cp'],
      install_requires=[
          'numpy',
          'scipy'
      ],
      zip_safe=False)
