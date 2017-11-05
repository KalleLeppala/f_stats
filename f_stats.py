import numpy

########## EXAMPLES ######################################################################################################################

########## STATISTIC SAMPLER FUNCTIONS ###################################################################################################

# The function f4 accepts three different input types. In most cases the type is detected automatically, but there is one pathological
# case where types 1) and 2) can't be told apart; optional parameter input_type can be then used.
#
# 1)
# The parameter table is a two dimensional numpy array containing the allele counts on the individual level.
# Rows are SNPs, columns are individuals.
# The parameter populations is a one dimensional numpy array containing the name of the population each individual belongs to,
# in the order they appear as the columns of table. Thus, each population name can appear once or many times.
# You can use the function counts_to_frequencies to transmute table and populations from input type 1) to input type 2).
# The parameter window tells how many SNPs are collapsed together in the output of this function.
# You can use the optional parameter input_type = 1, but it's only necessary if the minor allele is never homozygous.
#
# 2)
# The parameter table is a two dimensional numpy array containing the allele frequencies on the population level.
# Rows are SNPs, columns are populations.
# The parameter populations is a one dimensional numpy array containing the names of the populations,
# in the order they appear as the columns of table.
# You can use the function counts_to_frequencies to transmute table and populations from input type 1) to input type 2).
# You can use the function frequencies_to_differences to transmute table from input type 2) to input type 3).
# The parameter window tells how many SNPs are collapsed together in the output of this function.
# You can use the optional parameter input_type = 2, but it's only necessary if all alleles are fixed in all populations.
#
# 3)
# The parameter table is a two dimensional numpy array containing the allele frequency differencies on the pairs-of-populations -level.
# Rows are SNPs, columns are population pairs in certain order (only create such table using the function frequencies_to_differences).
# The parameter populations is a one dimensional numpy array containing the names of the populations, in an order realted to table.
# You can use the function frequencies_to_differences to transmute table from input type 2) to input type 3).
# The parameter window tells how many SNPs are collapsed together in the output of this function.
# You can use the optional parameter input_type = 3, but it's never necessary.
#
# The reason for this flexibility is the balancing between saving storage space and avoiding recalculating things.
# All the steps have to be done eventually but there might be no reason to ever save the full table 3) unless one intends to compute
# the f2- and f3-statistics as well (which use the same steps).
#
# The function returns a list of three things.
# [0] A two dimensional numpy array of sample means of (x - y)*(z - w), where x, y, z and w are allele frequencies of a SNP in
#     populations X, Y, Z and W, respectively, in a certain order. The mean of this variable is f4(X, Y; Z, W).
#     Statistics that are just other statistics negated are omitted.
#     The size of each sample is determined by the parameter window; SNPs at the end that do not fill a whole window are discarded.
#     Rows are statistics, columns are windows.
# [1] A one dimensional numpy array of names of the f4 statistics in the order [0] has them.
# [2] The window size used.
def f4(table, populations, window, input_type = 0):
    # Let's do what we can to detect the input type:
    if input_type == 0:
        if len(populations) < table.shape[1]:
            input_type = 3 # Input type is definitely 3).
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
        # I believe it's best to start by creating an output file of the right size.
        # Then we can avoid the unnecessary copying caused by the usage of append, concatenate or column_stack.
        P = len(populations) # Number of populations.
        F = int(P*(P - 3)/2) # Number of f4-statistics.
        W = int(table.shape[0]/window) # Number of windows.
        f4 = numpy.empty([F, W]) # The result, contains what ever for now.
        # Create the matrix that transforms allele frequencies into allele frequency differences.
        diff = numpy.zeros((P, P))
        for a in range(0, P - 1):
            diff[a, a] = 1
            diff[a + 1, a] = - 1
        diff[P - 1, P - 1] = 1
        diff[0, P - 1] = - 1
        # Then we proceed one window at a time, recording the sample mean of each f4-statistic.
        for w in range(W):
            diff_table = table[range(w*window, (w + 1)*window), ]@diff
            t = 0
            # The case a = P - 1 is dealt separately.
            for b in range(1, P - 2):
                f4[t, w] = diff_table[:, P - 1].T@diff_table[:, b]/window
                t += 1
            for a in range(0, P - 1):
                for b in range(a + 2, P - 1):
                    f4[t, w] = diff_table[:, a].T@diff_table[:, b]/window
                    t += 1
    if input_type == 3: # Assuming 3).
        # I believe it's best to start by creating an output file of the right size.
        # Then we can avoid the unnecessary copying caused by the usage of append, concatenate or column_stack.
        P = len(populations) # Number of populations.
        F = int(P*(P - 3)/2) # Number of f4-statistics.
        W = int(table.shape[0]/window) # Number of windows.
        f4 = numpy.empty([F, W]) # The result, contains what ever for now.
        # We proceed one window at a time, recording the sample mean of each f4-statistic.
        for w in range(W):
            t = 0
             # The case a = P - 1 is dealt separately.
            for b in range(1, P - 2):
                f4[t, w] = table[range(w*window, (w + 1)*window), P - 1].T@table[range(w*window, (w + 1)*window), b]/window
                t += 1
            for a in range(0, P - 1):
                for b in range(a + 2, P - 1):
                    f4[t, w] = table[range(w*window, (w + 1)*window), a].T@table[range(w*window, (w + 1)*window), b]/window
                    t += 1
    return(f4, f4_names(populations), window)

########## RESAMPLING FUNCTIONS ###########################################################################################################

# Given samples of a (multidimensional) i.i.d. random variable, the function jackknife computes the sample mean and the covariance
# matrix of the sample mean (not the random variable) using the jackknife method.
#
# The parameter triplet is (typically) an output of the function f2, f3 or f4. Triplet is a list of three things:
# [0] A two dimensional numpy array containing samples of an i.i.d. random variable. Rows are coordinates and columns are iterations.
# [1] A one dimensional numpy array containing names for the rows of [0].
# [2] A numeric multiplier for the covariance matrix of the sample mean (for when [0] is actually computed from blocks of the size [2]).
# The parameter save decides whether the results will be saved as txt files (save)_sm.txt for the sample mean and
# (save)_concentration.txt for the Cholesky decomposition of the inverted covariance matrix.
# The empty word (default) means no txt files is created.
#
# The output is a list of three things (and possibly some txt files created).
# [0] A one dimensional numpy array containing the sample mean of the (multidimensional) random variable.
# [1] A two dimensional numpy array containing the covariance matrix of that sample mean.
# [2] A one dimensional numpy array containing names of the coordinates.
def jackknife(triplet, save = ""):
    table = triplet[0]
    B = table.shape[1] # The number of resamplings.
    # Start by computing the sample mean.
    sm = table@numpy.ones([B, 1])/B
    # Then compute the covariance matrix of the sample mean. Not entirely sure about the scaling coefficient, though.
    cov = numpy.zeros([table.shape[0], table.shape[0]])
    for i in range(0, B):
        # What I'm doing might not look like jackknifing but it should be equivalent.
        # I'm actually computing the sample variance here... I guess jackknifing makes better sense when the statistic of interest
        # is something more complicated that the simple sample mean.
        v = numpy.asarray(numpy.asmatrix(table)[:, i]) - sm
        cov += v@v.T
    cov = triplet[2]*(B - 1)*cov/B
    if save != "":
        sm_filename = save + "_sm.txt"
        with open(sm_filename, "w") as f:
            for i in range(0, len(sm)):
                f.write(triplet[1][i])
                f.write(str(sm[i, 0]))
                f.write("\n")
        concentration = numpy.linalg.inv(numpy.linalg.cholesky(cov))
        concentration_filename = save + "_concentration.txt"
        with open(concentration_filename, "w") as f:
            for i in range(concentration.shape[0]):
                for j in range(concentration.shape[0]):
                    f.write(str(concentration[i, j]))
                    f.write(" ")
                f.write("\n")
    return(sm, cov, triplet[1])

# Given samples of a (multidimensional) i.i.d. random variable, the function jackknife computes the sample mean and the covariance
# matrix of the sample mean (not the random variable) using the bootstrap method.
#
# The parameter triplet is (typically) an output of the function f2, f3 or f4. Triplet is a list of three things:
# [0] A two dimensional numpy array containing samples of an i.i.d. random variable. Rows are coordinates and columns are iterations.
# [1] A one dimensional numpy array containing names for the rows of [0].
# [2] A numeric multiplier for the covariance matrix of the sample mean (for when [0] is actually computed from blocks of the size [2]).
# The parameter B is the number of bootstrap resamplings.
# The parameter save decides whether the results will be saved as txt files (save)_sm.txt for the sample mean and
# (save)_concentration.txt for the Cholesky decomposition of the inverted covariance matrix.
# The empty word (default) means no txt files is created.
#
# The output is a list of three things (and possibly some txt files created).
# [0] A one dimensional numpy array containing the sample mean of the (multidimensional) random variable.
# [1] A two dimensional numpy array containing the covariance matrix of that sample mean.
# [2] A one dimensional numpy array containing names of the coordinates.
def bootstrap(triplet, B = None, save = False):
    table = triplet[0]
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
    # Then compute the covariance matrix of the sample mean. Not entirely sure about the scaling coefficient, though.
    # At least it's converging when B tends to infinity, which is a good sign.
    cov = numpy.zeros([table.shape[0], table.shape[0]])
    for i in range(0, B):
        v = numpy.asarray(numpy.asmatrix(boots)[:, i]) - bsm
        cov += v@v.T
    cov = triplet[2]*(table.shape[1] - 1)*cov/B
    if save != "":
    sm_filename = save + "_sm.txt"
    with open(sm_filename, "w") as f:
        for i in range(0, len(sm)):
            f.write(triplet[1][i])
            f.write(str(sm[i, 0]))
            f.write("\n")
    concentration = numpy.linalg.inv(numpy.linalg.cholesky(cov))
    concentration_filename = save + "_concentration.txt"
    with open(concentration_filename, "w") as f:
        for i in range(concentration.shape[0]):
            for j in range(concentration.shape[0]):
                f.write(str(concentration[i, j]))
                f.write(" ")
            f.write("\n")
    return(sm, cov, triplet[1])

########### AUXILIARY FUNCTIONS ###########################################################################################################

# Given a table of allele counts, returns a table of allele frequencies.
# The numpy array populations divides the columns of table into groups of one or more individuals.
# The new populations are list(set(populations)).
#
# Used to convert table and populations of input type 1) into input type 2) for functions f2, f3 and f4. 
def counts_to_frequencies(table, populations):
    new_populations = list(set(populations))
    new_table = numpy.empty([table.shape[0], len(new_populations)])
    for p in range(0, len(new_populations)):
        v = numpy.where(populations == new_populations[p])[0]
        new_table[:, p] = (table[:, v]@numpy.ones((len(v), 1))/len(v))[:, 0]
    return(new_table, new_populations)

# Given a table of allele frequencies, returns a table of allele frequency differences.
# This function doesn't need to know the populations.
#
# Used to convert table of input type 2) into input type 3) for functions f2, f3 and f4. 
def frequencies_to_differences(table):
    P = table.shape[1] # Number of populations.
    D = int(P*(P - 1)/2) # Number of pairs of populations.
    # Create the matrix that transforms allele frequencies into allele frequency differences.
    diff = numpy.zeros((P, D))
    t = 0
    for a in range(0, P - 1):
        for b in range(a + 1, P):
            diff[a, t] = 1
            diff[b, t] = - 1
            t += 1
    return(table@diff)

# The function f4_names takes a numpy array of population names and returns a numpy array of f4-statistic names in the order
# the function f4 uses.
def f4_names(populations):
    P = len(populations) # Number of populations
    F = int(P*(P - 3)/2) # Number of f4-statistics.
    names = numpy.empty(F, dtype = object) # An empty array.
    t = 0 # A counter.
    # The case a = P - 1 is dealt separately.
    for b in range(1, P - 2):
        names[t] = populations[P - 1] + " " + populations[0] + " " + populations[b] + " " + populations[b + 1] + " "
        t += 1
    for a in range(0, P - 1):
        for b in range(a + 2, P - 1):
            names[t] = populations[a] + " " + populations[a + 1] + " " + populations[b] + " " + populations[b + 1] + " "
            t += 1
    return(names)

# The function bootstrap_vector returns the sum of n vectors uniformly randomly chosen from the elementary basis vectors of length n.
# You don't need to call this ever.
def bootstrap_vector(n):
    u = numpy.zeros(n)
    for i in range(0, n):
        u[numpy.random.randint(0, n)] += 1
    return(numpy.asarray(numpy.asmatrix(u).T))