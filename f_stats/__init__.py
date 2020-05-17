#   @copyright: 2019 by Kalle Leppälä
#   @license: MIT <http://www.opensource.org/licenses/mit-license.php>

"""
I'm sorry python is so stupid there is no multiline commenting. It makes these examples pretty hard to run.
        
Example 1)
Just a basic usage of the main function estimator_covariance using fake frequency data made with independent uniform distributions.
        
        >>> numpy.random.seed(10)
        >>> table = numpy.random.rand(100000, 5)
        >>> pops = numpy.array(["bear", "wolf", "lynx", "wolverine", "fox"])


Let's make a basis of f4-statistics.
The default setting would generate a basis of all f-statistics, consisting of f2- and f3-statistics.
We set the output_type to 3 in order to obtain a matrix directly applicable to the R package admixturegraph, and save the results
in a text files starting with the word "beast".

        >>> f_stats.estimator_covariance(table, pops, 1000, "f4_basis", output_type = 3, save = "beast")
        Guessing the table uses 1 chromosome per individual or already contains allele frquencies.
        (array([ 9.45092659e-04, -1.08584670e-03,  3.18334213e-04, -3.49745712e-04,
                9.61636557e-05]), array([[-2227.89184692,   517.28533574,  1298.32849716,   133.09029429,
                 -871.99995308],
               [-1722.80345344, -1414.29745722,   902.37191474,   731.94816561,
                  730.47257376],
               [  968.0740968 , -2467.8394992 ,  1055.48995949,  -783.4636571 ,
                 -166.87557781],
               [ 2868.09950545,   300.91108024,  1128.59256605,   835.38712129,
                 -208.23568014],
               [  126.03251267,  1919.40799433,  1495.14320313,  -634.82752598,
                  591.33679581]]), array(['wolverine bear fox lynx', 'wolverine bear lynx wolf',
               'bear fox lynx wolf', 'bear fox wolf wolverine',
               'fox lynx wolf wolverine'], dtype=object))


Example 2)
This sanity check demontrates that when the data is independet samples from a multivariate source with non-trivial correlation matrix,
the chosen window size doesn't matter much.
If the data was from independent multivariate source with independent coordinates, the window size would make a difference.
The real genetic data would not be independent samples, but big enough window size should help with that.


        >>> source_mean = (- 2, - 1, 0, 1, 2)
        >>> source_cov = [[1, 2, 3, 4, 5], [2, 4, 6, 8, 10], [3, 6, 9, 12, 15], [4, 8, 12, 16, 20], [5, 10, 15, 20, 25]]
        >>> source = numpy.random.multivariate_normal(source_mean, source_cov, 100000)


We want to be inside the unit interval, so let's apply the logistic function.


        >>> table = 1/(numpy.exp(source) + 1)
        >>> pops = numpy.array(["bear", "wolf", "lynx", "wolverine", "fox"])


Take a look at the first row of the covariance matrix computed with different widow sizes:


        >>> print(f_stats.estimator_covariance(table, pops, 1)[1][:, 0])
        Guessing the table uses 1 chromosome per individual or already contains allele frquencies.
        [6.87888520e-07 3.85801345e-07 1.10683117e-07 5.92509744e-07
         5.95751011e-08 2.78731120e-07 4.86631281e-09 8.89074617e-08
         2.96124142e-08 2.12392853e-07]


        >>> print(f_stats.estimator_covariance(table, pops, 10)[1][:, 0])
        Guessing the table uses 1 chromosome per individual or already contains allele frquencies.
        [6.86567629e-07 3.85025364e-07 1.10619772e-07 5.91833121e-07
         5.90749239e-08 2.78221084e-07 4.69081745e-09 8.88579686e-08
         2.95768470e-08 2.12462023e-07]

        >>> print(f_stats.estimator_covariance(table, pops, 100)[1][:, 0])
        Guessing the table uses 1 chromosome per individual or already contains allele frquencies.
        [6.23573430e-07 3.48324008e-07 1.00385300e-07 5.34366670e-07
         5.62068600e-08 2.53581056e-07 5.21477086e-09 7.96215949e-08
         2.70114133e-08 1.90905155e-07]

        >>> print(f_stats.estimator_covariance(table, pops, 1000)[1][:, 0])
        Guessing the table uses 1 chromosome per individual or already contains allele frquencies.
        [7.93405713e-07 4.51121144e-07 1.33592828e-07 6.88201573e-07
         6.52297148e-08 3.15692748e-07 4.27025726e-09 1.01593199e-07
         3.39012353e-08 2.43332849e-07]


        >>> print(f_stats.estimator_covariance(table, pops, 10000)[1][:, 0])
        Guessing the table uses 1 chromosome per individual or already contains allele frquencies.
        [ 1.27359998e-06  9.61452894e-07  4.01481312e-07  1.21567119e-06
         2.71093923e-08  3.07828028e-07 -4.03754283e-09  1.45656235e-07
         2.31736359e-08  2.77397903e-07]


Example 3)
This second sanity check shows that the resampling functions trurly recover the covariance matrix of the sample mean,
which equals the covariance matrix of the underlying random variable divided by the sample size.


        >>> mean = (- 1, 0, 1)
        >>> cov = [[1, 2, 3], [2, 4, 6], [3, 6, 9]]
        >>> sample = numpy.random.multivariate_normal(mean, cov, 1000).T
        >>> print(f_stats.jackknife(sample, cov))


Example 4)
Next we check that the function frequencies is working as intended by testing it with two diploid populations of different sample
sizes and missing data.


        >>> table = numpy.random.randint(2, size = [10, 3]) + numpy.random.randint(2, size = [10, 3]) - numpy.random.randint(2, size = [10, 3])
        >>> pops = numpy.array(["bear", "bear", "wolf"])
        >>> print(table)
        [[ 1 -1  0]
         [ 0  1  0]
         [ 1  2  0]
         [ 1  0 -1]
         [ 1  1  1]
         [-1 -1  0]
         [ 1  0  1]
         [ 1  1  1]
         [ 0  1 -1]
         [ 0  1  0]]


        >>> f_stats.frequencies(table, pops, 2)
        (array([[ 0.5 ,  0.  ],
               [ 0.25,  0.  ],
               [ 0.75,  0.  ],
               [ 0.25, -1.  ],
               [ 0.5 ,  0.5 ],
               [-1.  ,  0.  ],
               [ 0.25,  0.5 ],
               [ 0.5 ,  0.5 ],
               [ 0.25, -1.  ],
               [ 0.25,  0.  ]]), ['bear', 'wolf'])



"""

__all__ = [
    'estimator_covariance',
    'sample',
    'jackknife',
    'frequencies',
    'f2_basis',
    'f2_all',
    'f3_all',
    'f4_basis',
    'f4_all',
    'index']

from .f_stats import (
    estimator_covariance,
    sample,
    jackknife,
    frequencies,
    f2_basis,
    f2_all,
    f3_all,
    f4_basis,
    f4_all,
    index)