import numpy as np
from copy import copy, deepcopy
from time import time, sleep
from itertools import permutations
import matplotlib.pyplot as plt
import pandas as pd
from memory_profiler import profile


# Helper functions for testing

# Returns time elapsed and solution
def compute_instance(n_vec, bins=None, orders=None, type=None):
    if not bins:
        bins = [[1/float(sum(n_vec)) for _ in range(len(orders))] for _ in range(len(n_vec))]
    a = time()
    jocdf_object = None
    if type == "Unconditional":
        jocdf_object = Unconditional(n_vec, orders, bins)
    elif type == "Conditional":
        jocdf_object = Conditional(n_vec, orders, bins)
    elif type == "Spillover":
        jocdf_object = Spillover(n_vec, orders, bins)
    elif type == "Boncelet":
        jocdf_object = Boncelet(n_vec, orders, bins)
    elif type == "BapatBeg":
        jocdf_object = BapatBeg(n_vec, orders, bins)
    elif type == "Glueck":
        if len(n_vec) == 1:
            jocdf_object = GlueckSingle(n_vec, orders, bins)
        elif len(n_vec) == 2:
            jocdf_object = GlueckDouble(n_vec, orders, bins)
        else:
            print("Glueck writeup cannot hold up for m > 2")
            return 0, 0
    result = jocdf_object.compute()
    b = time()
    return b - a, result


# Create instance of single or multi distribution while looking at bottom d order statistics using uniform distribution
def test_bottom_d(n_vec, d, type):
    n = sum(n_vec)
    m = len(n_vec)
    bins = [[1/float(n) for _ in range(1, d + 1)] for _ in range(m)]
    orders = [i for i in range(1, d+1)]
    return compute_instance(n_vec, bins, orders, type)


# Compute the test multiple times to get time median for order input going from {1}, {1, 2}, ..., {1, 2, ... , d}
def multi_test(n_vec, d_range, type, num_trials):
    # n_vec - length m list of positive integers
    # d_range - positive integers of values that d can take on
    # type - can be string "Boncelet", "Unconditional", "Conditional", "BapatBeg", "Glueck", "Spillover"
    # num_trials - positive integer, number of times to run the trials
    times_elapsed = [[] for _ in d_range]
    results = [0 for _ in d_range]
    for trial in range(num_trials):
        for d in d_range:
            time_elapsed, result = test_bottom_d(n_vec, d, type)
            times_elapsed[d - d_range[0]].append(time_elapsed)
            results[d - d_range[0]] = result
    median_times = [np.median(times_elapsed[i]) for i in range(d_range[-1] - d_range[0] + 1)]
    return median_times, results


# Compute the test multiple times to get time median with fixed d
def multi_test_same(n_vec, order_input, type_alg, num_trials):
    # n_vec - length m list of positive integers
    # d_range - positive integers of values that d can take on
    # type - can be string "Boncelet", "Unconditional", "Conditional", "BapatBeg", "Glueck", "Spillover"
    # num_trials - positive integer, number of times to run the trials
    times_elapsed = []
    for trial in range(num_trials):
        time_elapsed, result = compute_instance(n_vec, bins=None, orders=order_input, type=type_alg)
        times_elapsed.append(time_elapsed)
    median_time = np.median(times_elapsed)
    return median_time


n_vec_input = ToyExample8_n_vec
bins_input = ToyExample8_bins
order_input = ToyExample8_order

BapatBegFunc = lambda n_vec, order, bins: BapatBeg(n_vec, order, bins)
BonceletFunc = lambda n_vec, order, bins: Boncelet(n_vec, order, bins)
GlueckSingleFunc = lambda n_vec, order, bins: GlueckSingle(n_vec, order, bins)
GlueckDoubleFunc = lambda n_vec, order, bins: GlueckDouble(n_vec, order, bins)
SpilloverFunc = lambda n_vec, order, bins: Spillover(n_vec, order, bins)
UnconditionalFunc = lambda n_vec, order, bins: Unconditional(n_vec, order, bins)
ConditionalFunc = lambda n_vec, order, bins: Conditional(n_vec, order, bins)

lambda_test_func = []
lambda_test_func.append(BapatBegFunc)
lambda_test_func.append(BonceletFunc)
lambda_test_func.append(GlueckSingleFunc)
lambda_test_func.append(GlueckDoubleFunc)
lambda_test_func.append(SpilloverFunc)
lambda_test_func.append(UnconditionalFunc)
lambda_test_func.append(ConditionalFunc)

# Test case 1 breaks:
# Test case 2 breaks:
# Test case 3 breaks: GlueckDouble runs into divide by 0
# Test case 4 breaks: GlueckDouble runs into divide by 0
# Test case 5 breaks:
# Test case 6 breaks:
# Test case 7 breaks: GlueckDouble is incorrect
# Test case 8 breaks: GlueckDouble is incorrect

index = 1
for lambda_func in lambda_test_func:
    print("Index: {}".format(index))
    t1 = time()
    instance = lambda_func(deepcopy(n_vec_input), deepcopy(order_input), deepcopy(bins_input))
    final_value = instance.compute()
    t2 = time()
    print("Total time elapsed: {}".format(str(t2-t1)))
    print("n_vec: ", n_vec_input)
    print("Bin probabilities: ", bins_input)
    print("Selected Order Statistics: ", order_input)
    print("Bin probability condition fulfill: ", final_value)
    print("\n")
    sleep(0.1)
    index += 1



# First set of tests, might include in the paper? Compares the performance degradation of Conditional algorithm and Glueck

n_vecs = []
n_vecs.append([10])
n_vecs.append([5, 5])
num_selected = [i for i in range(1, 10)]
num_trials = 10

jocdf5_times, jocdf5_results = [None for _ in range(len(n_vecs))], [None for _ in range(len(n_vecs))]
jocdf6_times, jocdf6_results = [None for _ in range(len(n_vecs))], [None for _ in range(len(n_vecs))]
boncelet_times, boncelet_results = [None for _ in range(len(n_vecs))], [None for _ in range(len(n_vecs))]
bonceletimproved_times, bonceletimproved_results = [None for _ in range(len(n_vecs))], [None for _ in range(len(n_vecs))]
bapatbeg_times, bapatbeg_results = [None for _ in range(len(n_vecs))], [None for _ in range(len(n_vecs))]
glueck_times, glueck_results = [None for _ in range(len(n_vecs))], [None for _ in range(len(n_vecs))]


for i in range(len(n_vecs)):
    n_vec = n_vecs[i]
    jocdf5_times[i], jocdf5_results[i] = multi_test(n_vec, num_selected, "Unconditional", num_trials)
    print("----------------")
    print("Unconditional")
    print("----------------")
    print("Times:", jocdf5_times[i])
    print("Results:", jocdf5_results[i])
    jocdf6_times[i], jocdf6_results[i] = multi_test(n_vec, num_selected, "Conditional", num_trials)
    print("----------------")
    print("Conditional")
    print("----------------")
    print("Times:", jocdf6_times[i])
    print("Results:", jocdf6_results[i])
    boncelet_times[i], boncelet_results[i] = multi_test(n_vec, num_selected, "Boncelet", num_trials)
    print("----------------")
    print("Boncelet")
    print("----------------")
    print("Times:", boncelet_times[i])
    print("Results:", boncelet_results[i])
    bonceletimproved_times[i], bonceletimproved_results[i] = multi_test(n_vec, num_selected, "Spillover", num_trials)
    print("----------------")
    print("Spillover")
    print("----------------")
    print("Times:", bonceletimproved_times[i])
    print("Results:", bonceletimproved_results[i])
    bapatbeg_times[i], bapatbeg_results[i] = multi_test(n_vec, num_selected, "BapatBeg", num_trials)
    print("----------------")
    print("BapatBeg")
    print("----------------")
    print("Times:", bapatbeg_times[i])
    print("Results:", bapatbeg_results[i])
    glueck_times[i], glueck_results[i] = multi_test(n_vec, num_selected, "Glueck", num_trials)
    print("----------------")
    print("Glueck")
    print("----------------")
    print("Times:", glueck_times[i])
    print("Results:", glueck_results[i])


jocdf5_times_df, jocdf5_results_df = pd.DataFrame(jocdf5_times), pd.DataFrame(jocdf5_results)
jocdf6_times_df, jocdf6_results_df = pd.DataFrame(jocdf6_times), pd.DataFrame(jocdf6_results)
boncelet_times_df, boncelet_results_df = pd.DataFrame(boncelet_times), pd.DataFrame(boncelet_results)
bonceletimproved_times_df, bonceletimproved_results_df = pd.DataFrame(bonceletimproved_times), pd.DataFrame(bonceletimproved_results)
bapatbeg_times_df, bapatbeg_results_df = pd.DataFrame(bapatbeg_times), pd.DataFrame(bapatbeg_results)
glueck_times_df, glueck_results_df = pd.DataFrame(glueck_times), pd.DataFrame(glueck_results)

jocdf5_times_df.to_csv('jocdf5_times_df_exp_1.csv')
jocdf5_results_df.to_csv('jocdf5_results_df_exp_1.csv')
jocdf6_times_df.to_csv('jocdf6_times_df_exp_1.csv')
jocdf6_results_df.to_csv('jocdf6_results_df_exp_1.csv')
boncelet_times_df.to_csv('boncelet_times_df_exp_1.csv')
boncelet_results_df.to_csv('boncelet_results_df_exp_1.csv')
bonceletimproved_times_df.to_csv('bonceletimproved_times_df_exp_1.csv')
bonceletimproved_results_df.to_csv('bonceletimproved_results_df_exp_1.csv')
bapatbeg_times_df.to_csv('bapatbeg_times_df_exp_1.csv')
bapatbeg_results_df.to_csv('bapatbeg_results_df_exp_1.csv')
glueck_times_df.to_csv('glueck_times_df_exp_1.csv')
glueck_results_df.to_csv('glueck_results_df_exp_1.csv')



# Plots for the first set of experiments as above
for i in range(len(n_vecs)):
    n_vec = n_vecs[i]
    plt.figure(figsize=(8,6))
    # plt.plot(num_selected, jocdf5_times[i], label="Unconditional", marker='o', fillstyle='none')
    plt.plot(num_selected, jocdf6_times[i], label="Conditional", marker='v', fillstyle='none')
    plt.plot(num_selected, boncelet_times[i], label="Boncelet", marker='^', fillstyle='none')
    # plt.plot(num_selected, bonceletimproved_times[i], label="Spillover", marker='<', fillstyle='none')
    plt.plot(num_selected, bapatbeg_times[i], label="Bapat-Beg", marker='>', fillstyle='none')
    plt.plot(num_selected, glueck_times[i], label="Glueck", marker='D', fillstyle='none')
    plt.title("Time Elapsed for n = {}".format(n_vec))
    plt.xlabel("d")
    plt.ylabel("Time Elapsed (seconds)")
    plt.legend()
    plt.show()

    plt.figure(figsize=(16, 8))
    plt.plot(num_selected, jocdf5_results[i], label="Unconditional", marker='o', fillstyle='none')
    plt.plot(num_selected, jocdf6_results[i], label="Conditional", marker='v', fillstyle='none')
    plt.plot(num_selected, boncelet_results[i], label="Boncelet", marker='^', fillstyle='none')
    plt.plot(num_selected, bonceletimproved_results[i], label="Spillover", marker='<', fillstyle='none')
    plt.plot(num_selected, bapatbeg_results[i], label="Bapat-Beg", marker='>', fillstyle='none')
    plt.plot(num_selected, glueck_results[i], label="Glueck", marker='D', fillstyle='none')
    plt.title("Result for n_vec = {}".format(n_vec))
    plt.xlabel("d")
    plt.ylabel("Probability")
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(8, 6))
    plt.plot(num_selected, jocdf5_results[i], label="Unconditional", marker='o', fillstyle='none')
    plt.plot(num_selected, jocdf6_results[i], label="Conditional", marker='v', fillstyle='none')
    plt.plot(num_selected, boncelet_results[i], label="Boncelet", marker='^', fillstyle='none')
    plt.plot(num_selected, bonceletimproved_results[i], label="Spillover", marker='<', fillstyle='none')
    plt.plot(num_selected, bapatbeg_results[i], label="Bapat-Beg", marker='>', fillstyle='none')
    #plt.plot(num_selected, glueck_results, label="Glueck", marker='D', fillstyle='none')
    plt.title("Result for n_vec = {}".format(n_vec))
    plt.xlabel("d")
    plt.ylabel("Probability")
    plt.legend()
    plt.show()


# Second set of tests, definitely will include in the paper if Takehiro, Zach, and Professor Greenwald like this test
# This one produces a table where the output are the times, the x columns are varying n from 5, 10, 15, 20, 25
# and the y columns are varying C = {1}, {1, 2}, {1, 2, 3}, {1, 2, 3, 4}, {1, 2, 3, 4, 5}

# Warning, Bapat Beg will NOT finish running in a reasonable amount of time. Cut off the tests after a minute of running

# Also, the values for this test for Boncelet, Spillover, BapatBeg should be close to the same for [4] and [2, 2],
# the same for [8] and [4, 4], ..., and the same for [20] and [10, 10]. If they're not, please let me know
n_vecs = [[6], [12], [18], [24], [30], [3, 3], [6, 6], [9, 9], [12, 12], [15, 15]]
order_inputs = [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]]
num_trials = 10

# Change this line to change which algorithm is running, alg in ("BapatBeg", "Glueck", "Boncelet", "Spillover", "Unconditional", "Conditional")
alg = "Conditional"
# Again warning, BapatBeg I'm certain will time out, so might have to do a smaller test for that or just fill in the values
# manually by printing them out
my_table_of_runtimes = [[0.0 for _ in range(len(n_vecs))] for _ in range(len(order_inputs))]

for i in range(len(order_inputs)):
    for j in range(len(n_vecs)):
        alg_time = multi_test_same(deepcopy(n_vecs[j]), deepcopy(order_inputs[i]), alg, num_trials)
        print("For order statistics {} and n_vec {}, elapsed runtime is {}".format(order_inputs[i], n_vecs[j], alg_time))
        my_table_of_runtimes[i][j] = alg_time

my_table_pd = dict()
# Copies to clipboard so you can paste into excel
my_table_pd[alg] = pd.DataFrame(my_table_of_runtimes)
my_table_pd[alg].to_csv(alg + '_experiment_2.csv')# Fourth set of tests, only for the new algorithm to empirically determine complexity with regards to n

d_max, num_trials = 200, 5 # To decrease wobliness, increase range of d, to increase trial size, increase d_max
d_flattened = [i for i in range(1, d_max+1)]
n_vec = [200]
d_vec = [lambda n, d: [i for i in range(1, d+1)], lambda n, d: [i for i in range(n-d+1, n+1)], lambda n, d: [int((i+1)*n/float(d)) for i in range(d)]]
d_times = [[0.00 for _ in range(d_max)] for _ in range(len(d_vec))]
plt.figure(figsize=(12,8))
plt.title("d vs Run-time for Conditional Algorithm")
plt.xlabel("log d")
plt.ylabel("Time elpased (log seconds)")
for d_idx in range(len(d_vec)):
    d_func = d_vec[d_idx]
    for d_prop in range(198, d_max+1):
        c_vec = d_func(n_vec[0], d_prop)
        print(c_vec)
        alg_time = multi_test_same(deepcopy(n_vec), deepcopy(c_vec), "Conditional", num_trials)
        d_times[d_idx][d_prop-1] = alg_time
    print(d_times[d_idx])
d_times_df = pd.DataFrame(d_times)
d_times_df.to_csv('d_times_df_exp_4_extended.csv')