import numpy

########## EXAMPLES #######################################################################################################################

# I'm sorry python is so stupid there is no multiline commenting. It makes these examples pretty hard to run.

# Example 1)
# Just a basic usage of the main function estimator_covariance using fake frequency data made with independent uniform distrihbutions.
#
# table = numpy.random.rand(100000, 5)
# pops = numpy.array(["bear", "wolf", "lynx", "wolverine", "fox"])
# Let's make a basis of f4-statistics.
# The default setting would generate a basis of all f-statistics, consisting of f2- and f3-statistics.
# We set the output_type to 3 in order to obtain a matrix directly applicable to the R package admixturegraph, and save the results
# in a text files starting with the word "beast".
# f_stats.estimator_covariance(table, pops, 1000, "f4", output_type = 3, save = "beast")

# Example 2)
# This sanity check demontrates that when the data is independet samples from a multivariate source with nonm-trivial correlation matrix,
# the chosen window size doesn't matter much.
# If the data was from independent multivariate source with independent coordinates, the window size would make a difference.
# The real genetic data would not be independent samples, but big enough window size (around 10000?) should help with that.
#
# source_mean = (- 2, - 1, 0, 1, 2)
# source_cov = [[1, 2, 3, 4, 5], [2, 4, 6, 8, 10], [3, 6, 9, 12, 15], [4, 8, 12, 16, 20], [5, 10, 15, 20, 25]]
# source = numpy.random.multivariate_normal(source_mean, source_cov, 100000)
# We want to be inside the unit interval, so let's apply the logistic function.
# table = 1/(numpy.exp(source) + 1)
# pops = numpy.array(["bear", "wolf", "lynx", "wolverine", "fox"])
# Take a look at the first row of the covariance matrix computed with different widow sizes:
# print(f_stats.estimator_covariance(table, pops, 1)[1][:, 0])
# print(f_stats.estimator_covariance(table, pops, 10)[1][:, 0])
# print(f_stats.estimator_covariance(table, pops, 100)[1][:, 0])
# print(f_stats.estimator_covariance(table, pops, 1000)[1][:, 0])
# print(f_stats.estimator_covariance(table, pops, 10000)[1][:, 0])

# Example 3)
# This second sanity check shows that the resampling functions trurly recover the covariance matrix of the sample mean,
# which equals the covariance matrix of the underlying random variable divided by the sample size.
#
# mean = (- 1, 0, 1)
# cov = [[1, 2, 3], [2, 4, 6], [3, 6, 9]]
# sample = numpy.random.multivariate_normal(mean, cov, 1000).T
# print(f_stats.jackknife(sample))
# print(f_stats.bootstrap(sample))

########## THE MAIN FUNCTION  #############################################################################################################

# estimator_covariance
# The function estimator_covariance is the main function using the other functions. Given either allele counts of individuals or allele
# frequencies of populations, it will compute the sample mean of a selected collection of second degree homogeneous polynomials, like some
# f2-, f3- or f4-statistics, and use block resampling methods to estimate the covariance matrix of this sample mean (NOT the covariance
# matrix of the underlying multidimensional random variable, which is the covariance of the sample mean times the sample size).
#
# table          There are two input types, usually automatically recognized by estimator_covariance:
#                1) A two dimensional numpy array containing the allele counts on the individual level. Rows are SNPs, columns are
#                   individuals. If the allele count is nowhere 2, you need to use the optional parameter input_type = 1.
#                2) A two dimensional numpy array containing the allele frequencies on the population level. Rows are SNPs, columns are
#                   populations. If the allele frequency is always either 0 or 1, you need to use the optional parameter input_type = 2.
#                You can use the function counts_to_frequencies to transmute table and populations from input type 1) to input type 2).
# populations    There are two input types, usually automatically recognized by estimator_covariance:
#                1) A one dimensional numpy array containing the name of the population each individual belongs to, in the order the
#                   individuals appear as the columns of table. Thus, each population name can appear once or many times.
#                2) A one dimensional numpy array containing the names of the populations, in the order they appear as the columns of
#                   table.
#                You can use the function counts_to_frequencies to transmute table and populations from input type 1) to input type 2).
# window         The size of the blocks used by the resampling methods in SNPs. The sesampling requires i.i.d. random variables but the
#                independence of consecutive samples is not very well satisfied, using blocks helps with that. I don't have a very good
#                idea on what would be a justified block size. Any data at the end of table that doesn't fill a whole block is discarded.
# statistic      The name of the preset statistic collection. Overruled if user defined statistics are given by M and names.
#                Let P be number of populations and x_1, x_2, ..., x_P the allele frequencies of a SNP in those populations.
#                By a statistic I mean (the expected value over all SNPs of) a homogeneous second degree polynomial on the x_i.
#                The dimension of the linear space spanned by all the possible statistics is P(P + 1)/2, as the space is spanned by
#                the linearly independent polynomials x_1^2, x_1x_2, ..., x_1x_P, x_2^2, x_2x_3, x_2x_4, ..., x_2x_P, ..., x_P^2.
#                "basis"     The default value. Consists of all the (P - 1)P/2 f2-statistics (x_i - x_j)^2 (i not j) and P f3-statistics
#                            of the form (x_i - x_(i - 1))(x_i - x_(i + 1)) (addition of indices done modulo P).
#                            In total that's P(P + 1)/2 statistics that are linearly independent and span the whole statistic space.
#                "f2_basis"  All the (P - 1)P/2 f2-statistics of the form (x_i - x_j)^2 (i < j). They are a linearly independent set that
#                            spans the whole subspace generated by all the f2-statistics.
#                "f2_all"    All the (P - 1)P f2-statistics (x_i - x_j)^2 (i not j). They do not form a linearly independent set, and so
#                            the covariance matrix of the sample mean will be only positive semidefinite, and thus singular.
#                            That's why you cannot select output_type = 3.
#                "f3_subset" There exists no nice basis made of f3-statistics for the P(P + 1)/2-dimensional subspace spanned by the
#                            f3-statistics. This is just the P linearly independent f3-statistics (x_i - x_(i - 1))(x_i - x_(i + 1))
#                            (addition of indices done modulo P). I use this to make "basis", by itself probably not interesting.
#                "f3_all"    All the (P - 2)(P - 1)P f3-statistics (x_i - x_j)(x_i - x_k) (i, j and k different). They do not form a
#                            linearly independent set, and so the covariance matrix of the sample mean will be only positive semidefinite,
#                            and thus singular. That's why you cannot select output_type = 3.
#                "f4_basis"  A collection of (P - 3)P/2 linearly independent f4-statistics (x_i - x_(i + 1))(x_j - x_(j + 1)) (addition of
#                            indices done modulo P, i - j not - 1, 0 or 1). They span the whole (P - 3)P/2-dimensional subspace generated
#                            by all the f4-statistics.
#                "f4_all"    All the (P - 3)(P - 2)(P - 1)P f4-statistics (x_i - x_j)(x_k - x_l) (i, j, k and l different). They do not
#                            form a linearly independent set, and so the covariance matrix of the sample mean will be only positive
#                            semidefinite, and thus singular. That's why you cannot select output_type = 3.
#                The statistic naming convention in all of the preset settings is that the name of f4(W, X; Y, Z), where W, X, Y and Z
#                are some populations (corresponds to the homogeneous polynomial (w - x)(y - z), where w, x, y and z are allele
#                frequencies of a SNP in populations W, X, Y and Z, respectively) is "W X Y Z". (This is the format the R package
#                admixturegraph uses.) Note that f2(X, Y) = f4(X, Y; X, Y) and f3(X; Y, Z) = f4(X, Y; X, Z).
# M              An optional parameter for overruling preset statistics (together with names). A two dimensional numpy array that codes
#                the homogeneous polynomials corresponding to your desired statistics as columns. More precisely w = vM, where
#                v = [x_1^2, x_1x_2, ..., x_1x_P, x_2^2, x_2x_3, x_2x_4, ..., x_2x_P, ..., x_P^2] and w is a row vector of your S new
#                statistics. Note that if deg(M) < S, the new statistics are linearly dependent, the covariance matrix of the sample mean
#                will be only positive semidefinite and thus singular, and so you cannot select output_type = 3.
# names          An optional parameter for overruling preset statistics (together with M). A one dimensional numpy array containing the
#                names for the S new statistics coded by M.
# resampling     The resampling method used for estimating the covariance of the sample mean of the statistic collection.
#                "jackknife" The default value. Using the (block) jackknife method.
#                "bootstrap" (Actually any other value than "jackknife".) Using the (block) bootsrap method with B bootstrap resamples.
# B              An optional parameter for the number of resamples when using the bootstrap method. Defaults at the number of blocks.
# input_type     An optional parameter for helping estimator_covariance to recognize whether it was given allele counts or allele
#                frequencies. Only needed when table has only zeros or ones, then use input_type = 1 for input type 1) (allele counts)
#                and input_type = 2 for input type 2) (allele frequencies). Defaults at input_type = 0, which is attempting to recognize
#                the input type automatically.
# output_type    An optional parameter deciding how estimator_covariance returns the covariance matrix (second element of the output).
#                1 The default value. Gives an estimate for the covariance matrix of the sample mean, let's call it S.
#                2 Gives a matrix s such that s^Ts = S.
#                3 Gives the inverse of s. Nothe that if the statistic collection is not linearly independent, the inverse does not
#                  exist and you cannot select output_type = 3. This matrix directly applicable to the R package admixturegraph.
# save           An optional string, if present will save the output in text files. The file "<save>_sm.txt" contains the sample means
#                of the statistics along with their names, and the file "<save>_covariance_<output_type>.txt" contains the matrix
#                decided by output_type (the second element of the output, the covariance matrix of the sample means by default).
#
# The function returns a list of three things:
# [0] The sample mean of the collection of statistics used (either preset or user defined).
# [1] What output_type dictates. Defaults at the covariance matrix of the sample mean in [0].
# [2] The names of the collection of statistics used, either preset or user defined.
def estimator_covariance(table,
                         populations,
                         window,
                         statistic = "basis",
                         M = None,
                         names = None,
                         resampling = "jackknife",
                         B = None,
                         input_type = 0,
                         output_type = 1,
                         save = ""):
    # Before proceeding, let's see if there will be problems with output_type = 3.
    if output_type == 3:
        if numpy.all(M) == None: # Using preset statistics.
            if statistic == "f2_all":
                print("You cannot choose output_type = 3 when using the preset statistics 'f2_all' because the statistics are not linearly independent and so the covariance matrix of the sample mean is not invertible.")
                return()
            if statistic == "f3_all":
                print("You cannot choose output_type = 3 when using the preset statistics 'f3_all' because the statistics are not linearly independent and so the covariance matrix of the sample mean is not invertible.")
                return()
            if statistic == "f4_all":
                print("You cannot choose output_type = 3 when using the preset statistics 'f4_all' because the statistics are not linearly independent and so the covariance matrix of the sample mean is not invertible.")
                return()
        else: # Using user defined statistics.
            if numpy.linalg.matrix_rank(M) < M.shape[1]:
                print("You cannot choose output_type = 3 because the statistics coded by the matrix M you gave are not linearly independent and so the covariance matrix of the sample mean is not invertible.")
                return()
    # Start by sampling in windows of size window, using the function sample.
    (table, populations) = sample(table, populations, window, input_type)
    # Now approximate the covariance matrix of the sample mean using either jackknifing or bootstrapping.
    if resampling == "jackknife":
        resample = jackknife(table)
    else:
        resample = bootstrap(table, B)
    (sm, cov) = resample
    # For covariances between linear combinations of the original statistics it's convenient to split the covariance matrix into a
    # product of a matrix and its transpose.
    ec = numpy.linalg.eigh(cov)
    half = numpy.diag(numpy.sqrt(numpy.absolute(ec[0])))@ec[1].T # The half transposed times half is cov.
    # Transform the covariance matrix to concern the statistics the user is interested in.
    if numpy.all(M) == None: # Using preset statistics.
        if statistic == "basis":
            (M, names) = basis(populations)
        elif statistic == "f2_basis":
            (M, names) = f2_basis(populations)
        elif statistic == "f2_all":
            (M, names) = f2_all(populations)
        elif statistic == "f3_subset":
            (M, names) = f3_subset(populations)
        elif statistic == "f3_all":
            (M, names) = f3_all(populations)
        elif statistic == "f4_basis":
            (M, names) = f4_basis(populations)
        elif statistic == "f4_all":
            (M, names) = f4_all(populations)
        else:
            print("Invalid preset statistics, use 'basis', 'f2_basis', 'f2_all', 'f3_subset', 'f3_all', 'f4_basis', 'f4_all', or your own parameters M and names.")
            return()
    sm = M.T@sm
    half = half@M
    cov = half.T@half
    if output_type == 1:
        result = cov
    if output_type == 2:
        ec = numpy.linalg.eigh(cov)
        half = numpy.diag(numpy.sqrt(numpy.absolute(ec[0])))@ec[1].T
        result = half
    if output_type == 3:
        ec = numpy.linalg.eigh(cov)
        half = numpy.diag(numpy.sqrt(numpy.absolute(ec[0])))@ec[1].T
        result = numpy.linalg.inv(half)
    if save != "":
        sm_filename = save + "_sm.txt"
        with open(sm_filename, "w") as f:
            for i in range(0, len(sm)):
                f.write(names[i])
                f.write(" ")
                f.write(str(sm[i, 0]))
                if i < len(sm) - 1:
                    f.write("\n")
        result_filename = save + "_covariance_" + str(output_type) + ".txt"
        with open(result_filename, "w") as f:
            for i in range(result.shape[0]):
                for j in range(result.shape[1]):
                    f.write(str(result[i, j]))
                    if j < result.shape[1] - 1:
                        f.write(" ")
                if i < result.shape[0] - 1:
                    f.write("\n")
    return(sm, result, names)
    
########## THE SAMPLER FUNCTION  ##########################################################################################################

# sample
# Given either allele counts of individuals or allele frequencies of populations, the function sample will compute sample means of all
# the second degree homogeneous monomials of allele frequencies in blocks. 
# 
# table          There are two input types, usually automatically recognized by sample:
#                1) A two dimensional numpy array containing the allele counts on the individual level. Rows are SNPs, columns are
#                   individuals. If the allele count is nowhere 2, you need to use the optional parameter input_type = 1.
#                2) A two dimensional numpy array containing the allele frequencies on the population level. Rows are SNPs, columns are
#                   populations. If the allele frequency is always either 0 or 1, you need to use the optional parameter input_type = 2.
#                You can use the function counts_to_frequencies to transmute table and populations from input type 1) to input type 2).
# populations    There are two input types, usually automatically recognized by sample:
#                1) A one dimensional numpy array containing the name of the population each individual belongs to, in the order the
#                   individuals appear as the columns of table. Thus, each population name can appear once or many times.
#                2) A one dimensional numpy array containing the names of the populations, in the order they appear as the columns of
#                   table.
#                You can use the function counts_to_frequencies to transmute table and populations from input type 1) to input type 2).
# window         The size of the blocks the sample means are computed over. Any data at the end of table that doesn't fill a whole block
#                is discarded.
# input_type     An optional parameter for helping sample to recognize whether it was given allele counts or allele frequencies.
#                Only needed when table has only zeros or ones, then use input_type = 1 for input type 1) (allele counts) and
#                input_type = 2 for input type 2) (allele frequencies). Defaults at input_type = 0, which is attempting to recognize
#                the input type automatically.
#
# The function returns a list of two things:
# [0] A two dimensional numpy array consisting of some number of sample means of the P(P + 1)/2 homoheneous second degree monomials
#     x_1^2, x_1x_2, ... , x_1x_P, x_2^2, x_2x_3, ..., x_2x_P, ..., x_P^2, where x_i is the allele frequency of a SNP in the i:th
#     population and P is the number of populations. Rows are monomials, columns are samples.
#     The size of each sample block is determined by the parameter window; SNPs at the end that do not fill a whole window are discarded.
# [1] The population names (which could be new information if using the input type 1)).
def sample(table, populations, window, input_type = 0):
    # Let's do what we can to detect the input type:
    if input_type == 0:
        for i in range(0, table.shape[0]):
            for j in range(0, table.shape[1]):
                if table[i, j] == 2:
                    input_type = 1 # Input type is definitely 1).
                    break
                if 0 < table[i, j] < 1:
                    input_type = 2 # Input type is definitely 2).
                    break
            if input_type != 0:
                break
    if input_type == 0:
        print("I don't know whether your input is allele counts of individuals (use parameter input_type = 1) or allele frequencies of populations (use parameter input_type = 2)")
        return()
    if input_type == 1: # Assuming 1).
        (table, populations) = counts_to_frequencies(table, populations)
        input_type = 2
    if input_type == 2: # Assuming 2).
        # Start by creating an output file of the right size.
        P = len(populations) # Number of populations.
        S = int(P*(P + 1)/2) # The number of second degree terms.
        W = int(table.shape[0]/window) # Number of windows.
        sample = numpy.empty([S, W]) # The result, contains what ever for now.
        # Then we proceed one window at a time, recording the sample mean of each second degree term.
        for w in range(W):
            t = 0 # A counter.
            for i in range(0, P):
                A = table[range(w*window, (w + 1)*window), i].T
                for j in range(i, P):
                    B = table[range(w*window, (w + 1)*window), j]
                    sample[t, w] = A@B/window
                    t += 1
    return(sample, populations)

########## RESAMPLING FUNCTIONS ###########################################################################################################

# jackknife
# Given samples of a (multidimensional) i.i.d. random variable, the function jackknife computes the sample mean and the covariance
# matrix of the sample mean (not the random variable) using the jackknife method.
#
# table    A two dimensional numpy array containing samples of an i.i.d. random variable.
#          Rows are coordinates and columns are iterations. The first coordinate of the output of the function sample is suitable.
#
# The output is a list of two things:
# [0] A one dimensional numpy array containing the sample mean of the (multidimensional) random variable.
# [1] A two dimensional numpy array containing the covariance matrix of that sample mean.
def jackknife(table):
    B = table.shape[1] # The number of resamplings.
    # Start by computing the sample mean.
    sm = table@numpy.ones([B, 1])/B
    # Then compute the covariance matrix of the sample mean.
    cov = numpy.zeros([table.shape[0], table.shape[0]])
    for i in range(0, B):
        # What I'm doing might not look like jackknifing but it is equivalent.
        v = numpy.asarray(numpy.asmatrix(table)[:, i]) - sm
        cov += v@v.T
    cov = cov/((B - 1)*B)
    return(sm, cov)

# bootstrap
# Given samples of a (multidimensional) i.i.d. random variable, the function bootstrap computes the sample mean and the covariance
# matrix of the sample mean (not the random variable) using the bootstrap method.
# This is slower than jackknife but should also be better with large values of B.
# If you are interested in the sample mean only and not its covariance matrix, you can use this function with B = 2 (for speed).
#
# table    A two dimensional numpy array containing samples of an i.i.d. random variable.
#          Rows are coordinates and columns are iterations. The first coordinate of the output of the function sample is suitable.
# B        An optional parameter for the number of resamples. Defaults at the number of original samples.
#
# The output is a list of two things:
# [0] A one dimensional numpy array containing the sample mean of the (multidimensional) random variable.
# [1] A two dimensional numpy array containing the covariance matrix of that sample mean.
def bootstrap(table, B = None):
    if B == None:
        B = table.shape[1]
    # Start by computing the sample mean.
    sm = table@numpy.ones([table.shape[1], 1])/table.shape[1]
    # Perform the bootstrap resampling.
    boots = numpy.empty([table.shape[0], B])
    for i in range(0, B):
        u = bootstrap_vector(table.shape[1])
        boots[:, i] = (table@u/table.shape[1]).T
    # Next we need the sample mean of the bootstrap resamples.
    bsm = boots@numpy.ones([B, 1])/B
    # Then compute the covariance matrix of the sample mean.
    cov = numpy.zeros([table.shape[0], table.shape[0]])
    for i in range(0, B):
        v = numpy.asarray(numpy.asmatrix(boots)[:, i]) - bsm
        cov += v@v.T
    cov = cov/(B - 1)
    return(sm, cov)

########### AUXILIARY FUNCTIONS ###########################################################################################################

# counts_to_frequencies
# The function counts_to_frequencies is used to convert table and populations of input type 1) (allele counts) into input type 2) 
# (ellele frequencies) for the functions estimator_covariance and sample.
#
# table          A two dimensional numpy array containing the allele counts on the individual level. Rows are SNPs, columns are
#                individuals.
# populations    A one dimensional numpy array containing the name of the population each individual belongs to, in the order the
#                individuals appear as the columns of table. Thus, each population name can appear once or many times.
#
# The function returns a list of two things:
# [0] A two dimensional numpy array containing the allele frequencies on the population level. Rows are SNPs, columns are populations.
# [1] A one dimensional numpy array containing the names of the populations, in the order they appear as the columns of table.
#     The new populations are list(set(populations)), so the original order af appearance from the input might not be kept. 
def counts_to_frequencies(table, populations):
    new_populations = list(set(populations))
    new_table = numpy.empty([table.shape[0], len(new_populations)])
    for p in range(0, len(new_populations)):
        v = numpy.where(populations == new_populations[p])[0]
        new_table[:, p] = (table[:, v]@numpy.ones((len(v), 1))/len(v))[:, 0]
    return(new_table, new_populations)

# basis
# The function basis creates the parameters M and names used by estimator_covariance with the default preset statistics "basis".
#
# populations    A one dimensional numpy array of populations.
#
# The function returns a list of two things:
# [0] The matrix M corresponding to the defaut preset statistic setting "basis", see estimator_covariance.
# [1] The array of statistic names corresponding to the default preset statistic setting "basis", see estimator_covariance.
def basis(populations):
    (M1, names1) = f2_basis(populations)
    (M2, names2) = f3_subset(populations)
    names = numpy.concatenate((names1, names2), 0)
    M = numpy.concatenate((M1, M2), 1)
    return(M, names)

# f2_basis
# The function f2_basis creates the parameters M and names used by estimator_covariance with the preset statistics "f2_basis".
#
# populations    A one dimensional numpy array of populations.
#
# The function returns a list of two things:
# [0] The matrix M corresponding to the preset statistic setting "f2_basis", see estimator_covariance.
# [1] The array of statistic names corresponding to the preset statistic setting "f2_basis", see estimator_covariance.
def f2_basis(populations):
    P = len(populations) # The number of populations.
    S = int(P*(P + 1)/2) # The number of second degree terms.
    F = int((P - 1)*P/2) # The number of statistics.
    M = numpy.zeros((S, F)) # The matrix coding the statistics in terms of the second degree terms.
    names = numpy.empty(F, dtype = object) # An empty array for the statistic names.
    t = 0 # A counter.
    for i in range(0, P - 1):
        for j in range(i + 1, P):
            M[index(P, i, 0), t] = 1
            M[index(P, j, 0), t] = 1
            M[index(P, i, j - i), t] = - 2
            names[t] = populations[i] + " " + populations[j] + " " + populations[i] + " " + populations[j]
            t += 1
    return(M, names)

# f2_all
# The function f2_all creates the parameters M and names used by estimator_covariance with the preset statistics "f2_all".
#
# populations    A one dimensional numpy array of populations.
#
# The function returns a list of two things:
# [0] The matrix M corresponding to the preset statistic setting "f2_all", see estimator_covariance.
# [1] The array of statistic names corresponding to the preset statistic setting "f2_all", see estimator_covariance.
def f2_all(populations):
    P = len(populations) # The number of populations.
    S = int(P*(P + 1)/2) # The number of second degree terms.
    F = int((P - 1)*P) # The number of statistics.
    M = numpy.zeros((S, F)) # The matrix coding the statistics in terms of the second degree terms.
    names = numpy.empty(F, dtype = object) # An empty array for the statistic names.
    t = 0 # A counter.
    for i in range(0, P):
        for j in range(0, P):
            if j != i:
                M[index(P, i, 0), t] = 1
                M[index(P, j, 0), t] = 1
                M[index(P, numpy.minimum(i, j), numpy.absolute(i - j)), t] = - 2
                names[t] = populations[i] + " " + populations[j] + " " + populations[i] + " " + populations[j]
                t += 1
    return(M, names)

# f3_subset
# The function f3_subset creates the parameters M and names used by estimator_covariance with the preset statistics "f3_subset".
#
# populations    A one dimensional numpy array of populations.
#
# The function returns a list of two things:
# [0] The matrix M corresponding to the preset statistic setting "f3_subset", see estimator_covariance.
# [1] The array of statistic names corresponding to the preset statistic setting "f3_subset", see estimator_covariance.
def f3_subset(populations):
    P = len(populations) # The number of populations.
    S = int(P*(P + 1)/2) # The number of second degree terms.
    F = P # The number of statistics.
    M = numpy.zeros((S, F)) # The matrix coding the statistics in terms of the second degree terms.
    names = numpy.empty(F, dtype = object) # An empty array for the statistic names.
    t = 0 # A counter.
    # The case i = 0 is dealt with separately.
    M[index(P, 0, 0), t] = 1
    M[index(P, 1, P - 2), t] = 1
    M[index(P, 0, P - 1), t] = - 1
    M[index(P, 0, 1), t] = - 1
    names[t] = populations[0] + " " + populations[P - 1] + " " + populations[0] + " " + populations[1]
    t += 1
    for i in range(1, P - 1):
        M[index(P, i, 0), t] = 1
        M[index(P, i - 1, 2), t] = 1
        M[index(P, i - 1, 1), t] = - 1
        M[index(P, i, 1), t] = - 1
        names[t] = populations[i] + " " + populations[i - 1] + " " + populations[i] + " " + populations[i + 1] + " "
        t += 1
    # The case i = P - 1 is dealt with separately.
    M[index(P, P - 1, 0), t] = 1
    M[index(P, 0, P - 2), t] = 1
    M[index(P, P - 2, 1), t] = - 1
    M[index(P, 0, P - 1), t] = - 1
    names[t] = populations[P - 1] + " " + populations[P - 2] + " " + populations[P - 1] + " " + populations[0]
    t += 1
    return(M, names)

# f3_all
# The function f3_all creates the parameters M and names used by estimator_covariance with the preset statistics "f3_all".
#
# populations    A one dimensional numpy array of populations.
#
# The function returns a list of two things:
# [0] The matrix M corresponding to the preset statistic setting "f3_all", see estimator_covariance.
# [1] The array of statistic names corresponding to the preset statistic setting "f3_all", see estimator_covariance.
def f3_all(populations):
    P = len(populations) # The number of populations.
    S = int(P*(P + 1)/2) # The number of second degree terms.
    F = int((P - 2)*(P - 1)*P) # The number of statistics.
    M = numpy.zeros((S, F)) # The matrix coding the statistics in terms of the second degree terms.
    names = numpy.empty(F, dtype = object) # An empty array for the statistic names.
    t = 0 # A counter.
    for i in range(0, P):
        for j in range(0, P):
            if j != i:
                for k in range(0, P):
                    if k != i:
                        if k != j:
                            M[index(P, i, 0), t] = 1
                            M[index(P, numpy.minimum(j, k), numpy.absolute(j - k)), t] = 1
                            M[index(P, numpy.minimum(i, j), numpy.absolute(i - j)), t] = - 1
                            M[index(P, numpy.minimum(i, k), numpy.absolute(i - k)), t] = - 1
                            names[t] = populations[i] + " " + populations[j] + " " + populations[i] + " " + populations[k]
                            t += 1
    return(M, names)
                    
# f4_basis
# The function f4_basis creates the parameters M and names used by estimator_covariance with the preset statistics "f4_basis".
#
# populations    A one dimensional numpy array of populations.
#
# The function returns a list of two things:
# [0] The matrix M corresponding to the preset statistic setting "f4_basis", see estimator_covariance.
# [1] The array of statistic names corresponding to the preset statistic setting "f4_basis", see estimator_covariance.
def f4_basis(populations):
    P = len(populations) # The number of populations.
    S = int(P*(P + 1)/2) # The number of second degree terms.
    F = int((P - 3)*P/2) # The number of statistics.
    M = numpy.zeros((S, F)) # The matrix coding the statistics in terms of the second degree terms.
    names = numpy.empty(F, dtype = object) # An empty array for the statistic names.
    t = 0 # A counter.
    # The case i = P - 1 is dealt with separately.
    for j in range(1, P - 2):
        M[index(P, 0, j + 1), t] = 1
        M[index(P, j, P - j - 1), t] = 1
        M[index(P, 0, j), t] = - 1
        M[index(P, j + 1, P - j - 2), t] = - 1
        names[t] = populations[P - 1] + " " + populations[0] + " " + populations[j] + " " + populations[j + 1]
        t += 1
    for i in range(0, P - 1):
        for j in range(i + 2, P - 1):
            M[index(P, i, j - i), t] = 1
            M[index(P, i + 1, j - i), t] = 1
            M[index(P, i, j - i + 1), t] = - 1
            M[index(P, i + 1, j - i - 1), t] = - 1
            names[t] = populations[i] + " " + populations[i + 1] + " " + populations[j] + " " + populations[j + 1]
            t += 1
    return(M, names)

# f4_all
# The function f4_all creates the parameters M and names used by estimator_covariance with the preset statistics "f4_all".
#
# populations    A one dimensional numpy array of populations.
#
# The function returns a list of two things:
# [0] The matrix M corresponding to the preset statistic setting "f4_all", see estimator_covariance.
# [1] The array of statistic names corresponding to the preset statistic setting "f4_all", see estimator_covariance.
def f4_all(populations):
    P = len(populations) # The number of populations.
    S = int(P*(P + 1)/2) # The number of second degree terms.
    F = int((P - 3)*(P - 2)*(P - 1)*P) # The number of statistics.
    M = numpy.zeros((S, F)) # The matrix coding the statistics in terms of the second degree terms.
    names = numpy.empty(F, dtype = object) # An empty array for the statistic names.
    t = 0 # A counter.
    for i in range(0, P):
        for j in range(0, P):
            if j != i:
                for k in range(0, P):
                    if k != i:
                        if k != j:
                            for l in range(0, P):
                                if l != i:
                                    if l != j:
                                        if l != k:
                                            M[index(P, numpy.minimum(i, k), numpy.absolute(i - k)), t] = 1
                                            M[index(P, numpy.minimum(j, l), numpy.absolute(j - l)), t] = 1
                                            M[index(P, numpy.minimum(i, l), numpy.absolute(i - l)), t] = - 1
                                            M[index(P, numpy.minimum(j, k), numpy.absolute(j - k)), t] = - 1
                                            names[t] = populations[i] + " " + populations[j] + " " + populations[k] + " " + populations[l]
                                            t += 1
    return(M, names)

# index
# The function index returns the ordinal of an element (r, r + d), d >= 0, from the diagonal or the upper triangle of a PxP-table,
# when ordered first by rows and then by columns.
#
# P    The size of the table.
# r    The row of the element in question.
# d    The diagional (the column minus the row) of the element in question.
#
# The function returns the ordinal of the element in question when the table is ordered by rows first, then columns.
def index(P, r, d):
    return(int(d + r*(P - (r - 1)/2)))

# bootsrap_vector
# The function bootsrap_vector returns the sum of n vectors uniformly randomly chosen from the elementary basis vectors of length n.
#
# n    It's a number. Beginning to think that too strict consistency in documenting style is not worth it :-P
#
# The function returns the sum of n vectors uniformly randomly chosen from the elementary basis vectors of length n.
def bootstrap_vector(n):
    u = numpy.zeros(n)
    for i in range(0, n):
        u[numpy.random.randint(0, n)] += 1
    return(numpy.asarray(numpy.asmatrix(u).T))