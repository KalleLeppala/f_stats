# f_stats

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




## estimator_covariance
The function estimator_covariance is the main function using the other functions. Given either allele counts of individuals or allele
frequencies of populations, it will compute the sample mean of a selected collection of second degree homogeneous polynomials, like some
f2-, f3- or f4-statistics, and use block jackknife to estimate the covariance matrix of this sample mean.
table          A two dimensional numpy array of values between zero and input_type, and missing values represented by negative numbers.
               These values can be allele counts/dosages of individuals from input_type chromosomes, but also allele frequencies of
               populations when input_type = 1. Rows are SNPs, columns are individuals or populations.
populations    A one dimensional numpy array containing the names of the populations each column of the table belongs to, in the same 
               order. Thus, each population name can appear once or many times.
window         The size of the blocks used by the resampling methods in SNPs. The sesampling requires i.i.d. random variables but the
               independence of consecutive samples is not very well satisfied, using blocks helps with that. I don't have a very good
               idea on what would be a justified block size. Any data at the end of table that doesn't fill a whole block is discarded.
statistic      The name of the preset statistic collection. Overruled if user defined statistics are given by M and names.
               Let P be number of populations and x_1, x_2, ..., x_P the allele frequencies of a SNP in those populations.
               By a statistic I mean (the expected value over all SNPs of) a homogeneous second degree polynomial on the x_i.
               The dimension of the linear space spanned by all the possible statistics is P(P + 1)/2, as the space is spanned by
               the linearly independent polynomials x_1^2, x_1x_2, ..., x_1x_P, x_2^2, x_2x_3, x_2x_4, ..., x_2x_P, ..., x_P^2.
               The f2- and the f3-statistics span the same (P - 1)P/2-dimensional subspace, which contains the (P - 3)P/2-dimensional
               subspace spanned by the f4-statistics.
               "f2_basis"  The default value. All the (P - 1)P/2 f2-statistics of the form (x_i - x_j)^2 (i < j). They are a linearly
                           independent set that spans the whole subspace generated by all the f2-statistics.
               "f2_all"    All the (P - 1)P f2-statistics (x_i - x_j)^2 (i not j). They do not form a linearly independent set, and so
                           the covariance matrix of the sample mean will be only positive semidefinite, and thus singular.
                           That's why you cannot select output_type = 3.
               "f3_all"    All the (P - 2)(P - 1)P f3-statistics (x_i - x_j)(x_i - x_k) (i, j and k different). They do not form a
                           linearly independent set, and so the covariance matrix of the sample mean will be only positive semidefinite,
                           and thus singular. That's why you cannot select output_type = 3. (There exists no nice basis made of f3-
                           statistics but the space is the same as the space of f2-statistics.)
               "f4_basis"  A collection of (P - 3)P/2 linearly independent f4-statistics (x_i - x_(i + 1))(x_j - x_(j + 1)) (addition of
                           indices done modulo P, i - j not - 1, 0 or 1). They span the whole (P - 3)P/2-dimensional subspace generated
                           by all the f4-statistics.
               "f4_all"    All the (P - 3)(P - 2)(P - 1)P f4-statistics (x_i - x_j)(x_k - x_l) (i, j, k and l different). They do not
                           form a linearly independent set, and so the covariance matrix of the sample mean will be only positive
                           semidefinite, and thus singular. That's why you cannot select output_type = 3.
               The statistic naming convention in all of the preset settings is that the name of f4(W, X; Y, Z), where W, X, Y and Z
               are some populations (corresponds to the homogeneous polynomial (w - x)(y - z), where w, x, y and z are allele
               frequencies of a SNP in populations W, X, Y and Z, respectively) is "W X Y Z". (This is the format the R package
               admixturegraph uses.) Note that f2(X, Y) = f4(X, Y; X, Y) and f3(X; Y, Z) = f4(X, Y; X, Z).
M              An optional parameter for overruling preset statistics (together with names). A two dimensional numpy array that codes
               the homogeneous polynomials corresponding to your desired statistics as columns. More precisely w = vM, where
               v = [x_1^2, x_1x_2, ..., x_1x_P, x_2^2, x_2x_3, x_2x_4, ..., x_2x_P, ..., x_P^2] and w is a row vector of your S new
               statistics. Note that if deg(M) < S, the new statistics are linearly dependent, the covariance matrix of the sample mean
               will be only positive semidefinite and thus singular, and so you cannot select output_type = 3.
names          An optional parameter for overruling preset statistics (together with M). A one dimensional numpy array containing the
               names for the S new statistics coded by M.
input_type     Number of chromosomes in table. Automatically guessed by estimator_covariance when using the default value input_type = 0.
output_type    An optional parameter deciding how estimator_covariance returns the covariance matrix (second element of the output).
               1 The default value. Gives an estimate for the covariance matrix of the sample mean, let's call it S.
               2 Gives a matrix s such that s^Ts = S.
               3 Gives the inverse of s. Note that if the statistic collection is not linearly independent, the inverse does not
                 exist and you cannot select output_type = 3. This matrix directly applicable to the R package admixturegraph.
save           An optional string, if present will save the output in text files. The file "<save>_sm.txt" contains the sample means
               of the statistics along with their names, and the file "<save>_covariance_<output_type>.txt" contains the matrix
               decided by output_type (the second element of the output, the covariance matrix of the sample means by default).
The function returns a list of three things:
[0] The sample mean of the collection of statistics used (either preset or user defined).
[1] What output_type dictates. Defaults at the covariance matrix of the sample mean in [0].
[2] The names of the collection of statistics used, either preset or user defined.
    
## sample
Given either allele counts of individuals or allele frequencies of populations, the function sample will compute sample means of all
the second degree homogeneous monomials of allele frequencies in blocks.

table          A two dimensional numpy array of values between zero and input_type, and missing values represented by negative numbers.
               These values can be allele counts/dosages of individuals from input_type chromosomes, but also allele frequencies of
               populations when input_type = 1. Rows are SNPs, columns are individuals or populations.
populations    A one dimensional numpy array containing the names of the populations each column of the table belongs to, in the same 
               order. Thus, each population name can appear once or many times.
window         The size of the blocks the sample means are computed over. Any data at the end of table that doesn't fill a whole block
               is discarded.
input_type     Number of chromosomes in table. Automatically guessed by estimator_covariance when using the default value input_type = 0.
The function returns a list of three things:
[0] A two dimensional numpy array consisting of some number of sample means of the P(P + 1)/2 homoheneous second degree monomials
    x_1^2, x_1x_2, ... , x_1x_P, x_2^2, x_2x_3, ..., x_2x_P, ..., x_P^2, where x_i is the allele frequency of a SNP in the i:th
    population and P is the number of populations. Rows are monomials, columns are samples.
    The size of each sample block is determined by the parameter window; SNPs at the end that do not fill a whole window are discarded.
[1] A coverage matrix describing the proportion of non-missingness for each pair of monomials on a window.
[2] The population names (which could be new information after applying the function frequencies).
    
## jackknife
Given samples of a (multidimensional) i.i.d. random variable, the function jackknife computes the sample mean and the covariance
matrix of the sample mean (not the random variable) using the jackknife method.
table       A two dimensional numpy array containing samples of an i.i.d. random variable.
            Rows are coordinates and columns are iterations. The first coordinate of the output of the function sample is suitable.
coverage    An array of matrices describing the proportion of non-missingness between pairs of variables on each window the samples
            are summarizing. Used for weighing down samples summarized from windows of incomplete data.
The output is a list of two things:
[0] A one dimensional numpy array containing the sample mean of the (multidimensional) random variable.
[1] A two dimensional numpy array containing the covariance matrix of that sample mean.
    
## frequencies
The function frequencies calculate a table allele frequencies from allele count/dosage data.
table          A two dimensional numpy array containing allele counts/dosages/frequencies from input_type chromosomes. Missing data
               indicated by a negative number.
populations    A one dimensional numpy array containing the names of the populations each column of the table belongs to, in the same
               order. Thus, each population name can appear once or many times.
input_type     The number of chromosomes.
The function returns a list of two things:
[0] A two dimensional numpy array containing the allele frequencies on the population level. Rows are SNPs, columns are populations.
[1] A one dimensional numpy array containing the names of the populations, in the order they appear as the columns of table.
    The new populations are sorted(list(set(populations))).
    
## f2_basis
The function f2_basis creates the parameters M and names used by estimator_covariance with the default preset statistics "f2_basis".
populations    A one dimensional numpy array of populations.
The function returns a list of two things:
[0] The matrix M corresponding to the preset statistic setting "f2_basis", see estimator_covariance.
[1] The array of statistic names corresponding to the preset statistic setting "f2_basis", see estimator_covariance.
    
## f2_all
The function f2_all creates the parameters M and names used by estimator_covariance with the preset statistics "f2_all".
populations    A one dimensional numpy array of populations.
The function returns a list of two things:
[0] The matrix M corresponding to the preset statistic setting "f2_all", see estimator_covariance.
[1] The array of statistic names corresponding to the preset statistic setting "f2_all", see estimator_covariance.
    
## f3_all
The function f3_all creates the parameters M and names used by estimator_covariance with the preset statistics "f3_all".
populations    A one dimensional numpy array of populations.
The function returns a list of two things:
[0] The matrix M corresponding to the preset statistic setting "f3_all", see estimator_covariance.
[1] The array of statistic names corresponding to the preset statistic setting "f3_all", see estimator_covariance.
    
## f4_basis
The function f4_basis creates the parameters M and names used by estimator_covariance with the preset statistics "f4_basis".
populations    A one dimensional numpy array of populations.
The function returns a list of two things:
[0] The matrix M corresponding to the preset statistic setting "f4_basis", see estimator_covariance.
[1] The array of statistic names corresponding to the preset statistic setting "f4_basis", see estimator_covariance.
    
## f4_all
The function f4_all creates the parameters M and names used by estimator_covariance with the preset statistics "f4_all".
populations    A one dimensional numpy array of populations.
The function returns a list of two things:
[0] The matrix M corresponding to the preset statistic setting "f4_all", see estimator_covariance.
[1] The array of statistic names corresponding to the preset statistic setting "f4_all", see estimator_covariance.
    
## index
The function index returns the ordinal of an element (r, r + d), d >= 0, from the diagonal or the upper triangle of a PxP-table,
when ordered first by rows and then by columns.
P    The size of the table.
r    The row of the element in question.
d    The diagional (the column minus the row) of the element in question.
The function returns the ordinal of the element in question when the table is ordered by rows first, then columns.
    