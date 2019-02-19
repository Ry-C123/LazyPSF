from setuptools import setup

setup(name='lazypsf',
      version='3.0.1',
      description='A code for modelling PSFs and injecting fake sources with given flux distributions',
      url= 'https://github.com/ryanc123/LazyPSF' ,
      author='Ryan Cutter',
      author_email='R.Cutter@wawrcik.ac.uk',
      license='GNU',
      packages=['lazypsf'],
      package_data={'lazypsf': ['config/*.txt']},
      include_package_data=True,
      zip_safe=False)
