from setuptools import setup
from hebel.version import version

try:
    from pypandoc import convert
    read_md = lambda f: convert(f, 'rst')
except ImportError:
    print("warning: pypandoc module not found, could not convert Markdown to RST")
    read_md = lambda f: open(f, 'r').read()

setup( name='Hebel',
       version=version,
       description='GPU-Accelerated '
       'Deep Learning Library in Python',
       long_description=read_md('README.md'),
       keywords='cuda gpu machine-learning deep-learning neural-networks',
       classifiers=[
           'Development Status :: 3 - Alpha',
           'Intended Audience :: Science/Research',
           'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
           'Programming Language :: C',
           'Programming Language :: Python :: 2.7',
           'Topic :: Scientific/Engineering :: Artificial Intelligence',
           'Topic :: Scientific/Engineering :: Image Recognition'
       ],
       url='https://github.com/hannes-brt/hebel',
       author='Hannes Bretschneider',
       author_email='hannes@psi.utoronto.ca',
       license='GPLv2',
       packages=['hebel',
                 'hebel.models',
                 'hebel.layers',
                 'hebel.utils',
                 'hebel.pycuda_ops'],
       install_requires=[
           'pycuda',
           'numpy',
           'pyyaml',
           'skdata'
       ],
       test_suite='nose.collector',
       tests_require=['nose'],
       scripts=['train_model.py'],
       include_package_data=True,
       zip_safe=False
)
