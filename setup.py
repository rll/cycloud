from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [
    Extension(
        "cycloud",
        ["src/cycloud.pyx"],
        language="c++",
        include_dirs=[numpy.get_include()],
    )
]

setup(
  name = 'cycloud',
  version='0.0.1',
  cmdclass = {'build_ext': build_ext},
  ext_package = 'cycloud',
  ext_modules = ext_modules,
  packages= ['cycloud'],
)
