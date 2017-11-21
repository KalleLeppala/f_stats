import numpy

########## EXAMPLES #######################################################################################################################

# I'm sorry python is so stupid there is no multiline commenting. It makes these examples pretty hard to run.

# Example 1)
# Just a basic usage of the main function estimator_covariance using fake frequency data made with independent uniform distrihbutions.
# table = numpy.random.rand(100000, 5)
# pops = numpy.array(["bear", "wolf", "lynx", "wolverine", "fox"])
# Let's make a basis of f4-statistics.
# The default setting would generate a basis of all f-statistics, consisting of f2- and f3-statistics.
# We set the output_type to 3 in order to obtain a matrix directly applicable to the R package admixturegraph, and save the results
# in a text files starting with the word "beast".
# f-stats.estimator_covariance(table, pops, 1000, "f4", output_type = 3, save = "beast")

# Example 2)
# This sanity check demontrates that when the data is independet samples from a multivariate source with nonm-trivial correlation matrix,
# the chosen window size doesn't matter much.
# If the data was from independent multivariate source with independent coordinates, the window size would make a difference.
# The real genetic data would not be independent samples, but big enough window size (around 10000?) should help with that.
# source_mean = (- 2, - 1, 0, 1, 2)
# source_cov = [[1, 2, 3, 4, 5], [2, 4, 6, 8, 10], [3, 6, 9, 12, 15], [4, 8, 12, 16, 20], [5, 10, 15, 20, 25]]
# source = numpy.random.multivariate_normal(source_mean, source_cov, 100000)
# We want to be inside the unit interval, so let's apply the logistic function.
# table = 1/(numpy.exp(source) + 1)
# pops = numpy.array(["bear", "wolf", "lynx", "wolverine", "fox"])
# Take a look at the first row of the covariance matrix computed with different widow sizes:
# print(f-stats.estimator_covariance(table, pops, 1)[1][:, 0])
# print(f-stats.estimator_covariance(table, pops, 10)[1][:, 0])
# print(f-stats.estimator_covariance(table, pops, 100)[1][:, 0])
# print(f-stats.estimator_covariance(table, pops, 1000)[1][:, 0])
# print(f-stats.estimator_covariance(table, pops, 10000)[1][:, 0])

# Example 3)
# This second sanity check shows that the resampling functions trurly recover the covariance matrix of the sample mean,
# which equals the covariance matrix of the variable divided by the sample size.
# mean = (- 1, 0, 1)
# cov = [[1, 2, 3], [2, 4, 6], [3, 6, 9]]
# sample = numpy.random.multivariate_normal(mean, cov, 1000).T
# print(f-stats.jackknife(sample))
# print(f-stats.bootstrap(sample))

########## THE MAIN FUNCTION  #############################################################################################################

# I'm documenting this maybe later today, and also maybe make a clearer documentation for the parameters of everything.
# Just in case I won't, it's time to push it already.
def estimator_covariance(table,
                         populations,
                         window,
                         statistic = "base",
                         M = None,
                         names = None,
                         resampling = "jackknife",
                         B = None,
                         input_type = 0,
                         output_type = 1,
                         save = ""):
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
    P = len(populations) # Number of populations.
    if M == None: # Using preset statistics.
        if statistic == "f2":
            (M, names) = f2_base(P, populations)
        elif statistic == "f3":
            (M, names) = f3_base(P, populations)
        elif statistic == "f4":
            (M, names) = f4_base(P, populations)
        elif statistic == "f3_subset":
            (M, names) = f3_subset(P, populations)
        elif statistic == "base":
            (M, names) = base(P, populations)
        else:
            print("Invalid preset statistics, use 'f2', 'f3', 'f4', f3_subset, 'all', or the default value '' with your own parameters M and names.")
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
        if output_type == 1:
            result_filename = save + "_covariance.txt"
        if output_type == 2:
            result_filename = save + "_inverse_concentration.txt"
        if output_type == 3:
            result_filename = save + "_concentration.txt"
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

# Accepts two different input types. In most cases the type is detected automatically, but there is one pathological case where they
# can't be told apart; optional parameter input_type can be then used.
#
# 1)
# The parameter table is a two dimensional numpy array containing the allele counts on the individual level.
# Rows are SNPs, columns are individuals.
# The parameter populations is a one dimensional numpy array containing the name of the population each individual belongs to,
# in the order they appear as the columns of table. Thus, each population name can appear once or many times.
# You can use the function counts_to_frequencies to transmute table and populations from input type 1) to input type 2).
# The parameter window tells how many SNPs are collapsed together in the output of this function.
# You can use the optional parameter input_type = 1, but it's only necessary if the counted allele is never homozygous.
#
# 2)
# The parameter table is a two dimensional numpy array containing the allele frequencies on the population level.
# Rows are SNPs, columns are populations.
# The parameter populations is a one dimensional numpy array containing the names of the populations,
# in the order they appear as the columns of table.
# You can use the function counts_to_frequencies to transmute table and populations from input type 1) to input type 2).
# The parameter window tells how many SNPs are collapsed together in the output of this function.
# You can use the optional parameter input_type = 2, but it's only necessary if all alleles are fixed in all populations.
#
# The function returns a list of two things:
# [0] A one dimensional numpy array of sample means of x_1^2, x_1*x_2, ... , x_1*x_P, x_2^2, x_2*x_3, ..., x_2*x_P, ..., x_P^2,
#     where x_i is the allele frequency of a SNP in population X_i and there is P the number of populations.
#     The size of each sample is determined by the parameter window; SNPs at the end that do not fill a whole window are discarded.
#     Rows are different second degree terms, columns are windows.
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

# Given samples of a (multidimensional) i.i.d. random variable, the function jackknife computes the sample mean and the covariance
# matrix of the sample mean (not the random variable) using the jackknife method.
#
# The parameter table is two dimensional numpy array containing samples of an i.i.d. random variable.
# Rows are coordinates and columns are iterations. The first coordinate of the output of the function sample is suitable.
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

# Given samples of a (multidimensional) i.i.d. random variable, the function bootstrap computes the sample mean and the covariance
# matrix of the sample mean (not the random variable) using the bootstrap method.
# This is slower than jackknife but should also better with large values of B.
# If you are interested in the sample mean only and not its covariance matrix, you can use this function with B = 2 (for speed).
#
# The parameter table is two dimensional numpy array containing samples of an i.i.d. random variable.
# Rows are coordinates and columns are iterations. The first coordinate of the output of the function sample is suitable.
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

# Given a table of allele counts, returns a table of allele frequencies.
# The numpy array populations divides the columns of table into groups of one or more individuals.
# The new populations are list(set(populations)), so the original order af appearance might not be kept.
#
# Used to convert table and populations of input type 1) into input type 2) for functions f2, f3 and f4. 
def counts_to_frequencies(table, populations):
    new_populations = list(set(populations))
    new_table = numpy.empty([table.shape[0], len(new_populations)])
    for p in range(0, len(new_populations)):
        v = numpy.where(populations == new_populations[p])[0]
        new_table[:, p] = (table[:, v]@numpy.ones((len(v), 1))/len(v))[:, 0]
    return(new_table, new_populations)

# Creates a matrix that codes a basis of f2-statistics in terms of the second degree terms, and names for those statistics.
def f2_base(P, populations):
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

# Creates a matrix that codes a basis of f4-statistics in terms of the second degree terms, and names for those statistics.
def f4_base(P, populations):
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

# Creates a matrix that codes a certain subset of f3-statistics in terms of the second degree terms, and names for those statistics.
def f3_subset(P, populations):
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
    names[t] = populations[P - 1] + " " + populations[P - 2] + " " + populations[P - 1] + " " + populations[1]
    t += 1
    return(M, names)

# Creates a matrix that codes a basis of all f-statistics in terms of the second degree terms, and names for those statistics.
def base(P, populations):
    (M1, names1) = f2_base(P, populations)
    (M2, names2) = f3_subset(P, populations)
    names = numpy.concatenate((names1, names2), 0)
    M = numpy.concatenate((M1, M2), 1)
    return(M, names)

# Returns the ordinal of an element (r, c), r >= c, from the diagonal or the upper triangle of a PxP-table, when ordered first by rows
# and then by columns.
# You don't need to call this ever.
def index(P, r, c):
    return(int(c + r*(P - (r - 1)/2)))

# Returns the sum of n vectors uniformly randomly chosen from the elementary basis vectors of length n.
# You don't need to call this ever.
def bootstrap_vector(n):
    u = numpy.zeros(n)
    for i in range(0, n):
        u[numpy.random.randint(0, n)] += 1
    return(numpy.asarray(numpy.asmatrix(u).T))