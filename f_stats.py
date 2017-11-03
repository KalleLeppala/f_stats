# Given a table of allele counts, returns a table of allele frequencies.
# The list populations divides the columns of table into groups of one or more individuals.
# The new populations are list(set(populations)).
def counts_to_frequencies(table, populations):
    new_populations = list(set(populations))
    new_table = numpy.empty([table.shape[0], len(new_populations)])
    for p in range(0, len(new_populations)):
        v = numpy.where(populations == new_populations[p])[0]
        new_table[:, p] = (table[:, v]@numpy.ones((len(v), 1))/len(v))[:, 0]
    return(new_table, new_populations)

# Given a table of allele frequencies, returns a table of allele frequency differences.
# This function doesn't need to know the populations.
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

# The function f4 accepts three different tables as an input.
# 1) The table of allele counts on the individual level.
# 2) The table of allele frequencies on the population level.
# 3) The table of allele frequency differencies on the pairs-of-populations -level.
# The reason for this flexibility is the balancing between saving storage space and avoiding recalculating things.
# All the steps have to be done eventually but there might be no reason to ever save the full table 3) unless one intends to compute
# the f2- and f3-statistics as well (which use the same steps).
# The function returns a table of sample means of f4-statistics, where the size of each sample is determined by the parameter window.
# We will discard the data rows at the end that do not fill a whole window.
# Further one can then use the function ??? to estimate both the mean and variance of the f4-statistics.
# The numpy array populations contains the population names in the right order, this could be included in the table when I swith to
# using hdf5.
def f4(table, window, populations, mode = 2):
    # I could detect the mode automatically, getting back to this soon.
    if mode == 1: # Assuming 1).
        (table, populations) = counts_to_frequencies(table, populations)
        mode = 2
    if mode == 2: # Assuming 2).
        # I believe it's best to start by creating an output file of the right size.
        # Then we can avoid the unnecessary copying caused by the usage of append, concatenate or column_stack.
        P = len(populations) # Number of populations.
        D = int(P*(P - 1)/2) # Number of pairs of populations.
        F = int(P*(P - 1)*(P - 2)*(P - 3)/8) # Number of f4-statistics.
        W = int(table.shape[0]/window) # Number of windows.
        f4 = numpy.empty([F, W]) # The result, contains what ever for now.
        # Create the matrix that transforms allele frequencies into allele frequency differences.
        diff = numpy.zeros((P, D))
        t = 0
        for a in range(0, P - 1):
            for b in range(a + 1, P):
                diff[a, t] = 1
                diff[b, t] = - 1
                t += 1
        # Then we proceed one window at a time, recording the sample mean of each f4-statistic.
        for w in range(W):
            diff_table = table[range(w*window, (w + 1)*window), ]@diff
            t = 0
            for a in range(0, P - 3):
                for b in range(a + 1, P - 2):
                    for c in range(b + 1, P - 1):
                        for d in range(c + 1, P):
                            u = int(a*P - a*(a + 3)/2 + b - 1)
                            v = int(c*P - c*(c + 3)/2 + d - 1)
                            f4[t, w] = diff_table[:, u].T@diff_table[:, v]/window
                            t += 1
                            u = int(a*P - a*(a + 3)/2 + c - 1)
                            v = int(b*P - b*(b + 3)/2 + d - 1)
                            f4[t, w] = diff_table[:, u].T@diff_table[:, v]/window
                            t += 1
                            u = int(a*P - a*(a + 3)/2 + d - 1)
                            v = int(b*P - b*(b + 3)/2 + c - 1)
                            f4[t, w] = diff_table[:, u].T@diff_table[:, v]/window
                            t += 1
    if mode == 3: # Assuming 3).
        # I believe it's best to start by creating an output file of the right size.
        # Then we can avoid the unnecessary copying caused by the usage of append, concatenate or column_stack.
        P = len(populations) # Number of populations.
        D = int(P*(P - 1)/2) # Number of pairs of populations.
        F = int(P*(P - 1)*(P - 2)*(P - 3)/8) # Number of f4-statistics.
        W = int(table.shape[0]/window) # Number of windows.
        f4 = numpy.empty([F, W]) # The result, contains what ever for now.
        # We proceed one window at a time, recording the sample mean of each f4-statistic.
        for w in range(W):
            t = 0
            for a in range(0, P - 3):
                for b in range(a + 1, P - 2):
                    for c in range(b + 1, P - 1):
                        for d in range(c + 1, P):
                            u = int(a*P - a*(a + 3)/2 + b - 1)
                            v = int(c*P - c*(c + 3)/2 + d - 1)
                            f4[t, w] = table[range(w*window, (w + 1)*window), u].T@table[range(w*window, (w + 1)*window), v]/window
                            t += 1
                            u = int(a*P - a*(a + 3)/2 + c - 1)
                            v = int(b*P - b*(b + 3)/2 + d - 1)
                            f4[t, w] = table[range(w*window, (w + 1)*window), u].T@table[range(w*window, (w + 1)*window), v]/window
                            t += 1
                            u = int(a*P - a*(a + 3)/2 + d - 1)
                            v = int(b*P - b*(b + 3)/2 + c - 1)
                            f4[t, w] = table[range(w*window, (w + 1)*window), u].T@table[range(w*window, (w + 1)*window), v]/window
                            t += 1
    return(f4, populations)

# The function jackknife takes a table of samples from multidimensional i.i.d. random variable, each sample being a column,
# and computes the sample mean and the covariance matrix of the sample mean (not the random variable) using the jackknife method.
# Each sample could be a block of samples collapsed together in fact, to achieve better independence, but this function need not know
# about it (it might still scale the covariance matrix of the sample mean somehow, need to study this more).
def jackknife(table):
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
    return(sm, (B - 1)*cov/B)

# The function bootstrap takes a table of samples from multidimensional i.i.d. random variable, each sample being a column,
# and computes the sample mean and the covariance matrix of the sample mean (not the random variable) using the bootstrap method.
# Each sample could be a block of samples collapsed together in fact, to achieve better independence, but this function need not know
# about it (it might still scale the covariance matrix of the sample mean somehow, need to study this more).
# The parameter B is the number of bootstrap resamplings.
def bootstrap(table, B = table.shape[1]):
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
    return(sm, (table.shape[1] - 1)*cov/B)

# The function bootstrap_vector returns the sum of n vectors uniformly randomly chosen from the elementary basis vectors of length n. 
def bootstrap_vector(n):
    u = numpy.zeros(n)
    for i in range(0, n):
        u[numpy.random.randint(0, n)] += 1
    return(numpy.asarray(numpy.asmatrix(u).T))

def f4_quadruples(populations):
    P = len(populations)
    for a in range(0, P - 3):
        for b in range(a + 1, P - 2):
            for c in range(b + 1, P - 1):
                for d in range(c + 1, P):
                    print("(" + populations[a] + ", " + populations[b] + "; " + populations[c] + ", " + populations[d] + ")")
                    print("(" + populations[a] + ", " + populations[c] + "; " + populations[b] + ", " + populations[d] + ")")
                    print("(" + populations[a] + ", " + populations[d] + "; " + populations[b] + ", " + populations[c] + ")")
