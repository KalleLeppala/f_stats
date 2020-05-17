#   @copyright: 2019 by Kalle Leppälä
#   @license: MIT <http://www.opensource.org/licenses/mit-license.php>


if __name__ == '__main__':
    import doctest
    import f_stats
    import numpy

    # by importing these here, there might be some import errors left..
    globs = {
    'f_stats': f_stats,
    'numpy': numpy

#    'estimator_covariance': f_stats.estimator_covariance,
#    'sample': f_stats.sample,
#    'jackknife': f_stats.jackknife,
#    'frequencies': f_stats.frequencies,
#    'f2_basis': f_stats.f2_basis,
#    'f2_all': f_stats.f2_all,
#    'f3_all': f_stats.f3_all,
#    'f4_basis': f_stats.f4_basis,
#    'f4_all': f_stats.f4_all,
#    'index': f_stats.index
    }
    

    doctest.testfile(filename="f_stats.py", module_relative=True, package=f_stats, globs=globs)
    doctest.testfile(filename="__init__.py", module_relative=True, package=f_stats, globs=globs)
