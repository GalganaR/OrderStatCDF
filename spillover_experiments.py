import numpy as np
from copy import copy, deepcopy
from time import time, sleep
from itertools import permutations
import matplotlib.pyplot as plt
import pandas as pd
from memory_profiler import profile
import csv
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


# Helper functions for testing

# Returns time elapsed and solution
def compute_instance(n_vec_i, bins_i=None, orders_i=None, type=None):
    n_vec = deepcopy(n_vec_i)
    bins = deepcopy(bins_i)
    orders = deepcopy(orders_i)
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


# Compute the test multiple times to get time median with fixed d
def multi_test_same(n_vec, order_input, type_alg, num_trials):
    # n_vec - length m list of positive integers
    # type - can be string "Boncelet", "Unconditional", "Conditional", "BapatBeg", "Glueck", "Spillover"
    # num_trials - positive integer, number of times to run the trials
    times_elapsed = []
    for trial in range(num_trials):
        time_elapsed, result = compute_instance(n_vec_i = n_vec, bins_i=None, orders_i=order_input, type=type_alg)
        times_elapsed.append(time_elapsed)
    median_time = np.median(times_elapsed)
    return median_time



# EXPERIMENT 2, n = 100, d in \{1,...,10\} for C = (1,...,d)

final_experiment_2 = True

if final_experiment_2:
    d_min, d_max = 3, 10
    n_list = [[10], [20]]
    c_list = [list(range(1,i+1)) for i in range(d_min, d_max)]
    times_exp2_spillover, times_exp2_boncelet = [[] for _ in n_list], [[] for _ in n_list]
    num_trials = 11
else:
    d_min, d_max = 4, 7
    n_list = [[7], [14]]
    c_list = [list(range(1,i+1)) for i in range(d_min, d_max)]
    times_exp2_spillover, times_exp2_boncelet = [[] for _ in n_list], [[] for _ in n_list]
    num_trials = 1
    
a, b = time(), time()
for j in range(len(n_list)):
    for i in range(len(c_list)):
        orders = c_list[i]
        n = n_list[j]
        print(n, orders)
        times_exp2_spillover[j].append(multi_test_same(n, orders, 'Spillover', num_trials))
        times_exp2_boncelet[j].append(multi_test_same(n, orders, 'Boncelet', num_trials))
        b = time()
        print(b - a)

plt.title(r"Spillover vs Boncelet Algorithm, Varying $d, \mathcal{C}$")
for i in range(len(n_list)):
    
    x_row = np.log2([x for x in range(d_min,d_max)])
    x_row_reshape = x_row.reshape(-1, 1)
    times_exp2_spillover_reshape = np.log2(times_exp2_spillover[i]).reshape(-1, 1)
    times_exp2_boncelet_reshape = np.log2(times_exp2_boncelet[i]).reshape(-1, 1)

    regr1 = linear_model.LinearRegression()
    regr1.fit(x_row_reshape, times_exp2_spillover_reshape)
    print('Coefficients: \n', regr1.coef_)

    regr2 = linear_model.LinearRegression()
    regr2.fit(x_row_reshape, times_exp2_boncelet_reshape)
    print('Coefficients: \n', regr2.coef_)
    
    regr_1_coef = int(regr1.coef_ * 100)/100.0
    regr_2_coef = int(regr2.coef_ * 100)/100.0
    
    plt.plot(np.log2([x for x in range(d_min,d_max)]), np.log2(times_exp2_spillover[i]), colors[i], label='Spillover, n={}, slope={}'.format(n_list[i], regr_1_coef))
    plt.plot(np.log2([x for x in range(d_min,d_max)]), np.log2(times_exp2_boncelet[i]), colors[i]+"--", label='Boncelet, n={}, slope={}'.format(n_list[i], regr_2_coef))

colors = ['b', 'r', 'g']
plt.xlabel(r"$\log d$")
plt.ylabel("Time Elapsed (log2 seconds)")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True)

d_2_spillover_times_df = pd.DataFrame(times_exp2_spillover)
d_2_boncelet_times_df = pd.DataFrame(times_exp2_boncelet)
d_2_spillover_times_df.to_csv('d_2_spillover_times_df.csv')
d_2_boncelet_times_df.to_csv('d_2_boncelet_times_df.csv')


# EXPERIMENT 3, n = 100, d in \{1,...,10\} for C = (1,...,d)

final_experiment_3 = True

if final_experiment_3:
    n_list = [[8*int(10*2**(i/4))] for i in range(15)] + [[1000]]
    print(n_list)
    times_exp3_spillover, times_exp3_boncelet = [[] for _ in range(2)], [[] for _ in range(2)]
    num_trials = 11
else:
    n_list = [[10*int(10*2**(i/4))] for i in range(12)]
    print(n_list)
    times_exp3_spillover, times_exp3_boncelet = [[] for _ in range(2)], [[] for _ in range(2)]
    num_trials = 1
    
a, b = time(), time()
for j in range(len(n_list)):
    n = n_list[j]
    orders = [1]
    n = n_list[j]
    print(n, orders)
    times_exp3_spillover[0].append(multi_test_same(n, orders, 'Spillover', num_trials))
    times_exp3_boncelet[0].append(multi_test_same(n, orders, 'Boncelet', num_trials))

    c = [n_list[j][0]]
    orders = c
    n = n_list[j]
    print(n, orders)
    times_exp3_spillover[1].append(multi_test_same(n, orders, 'Spillover', num_trials))
    times_exp3_boncelet[1].append(multi_test_same(n, orders, 'Boncelet', num_trials))

plt.title(r"Spillover vs Boncelet Algorithm, Varying $n$")

colors = ['b', 'r', 'g']
x_row = np.log2([n[0] for n in n_list])
x_row_reshape = x_row.reshape(-1, 1)

print(x_row)
print(times_exp3_spillover[0])
print(times_exp3_boncelet[0])

times_exp3_spillover_reshape = np.log2(times_exp3_spillover[0]).reshape(-1, 1)
times_exp3_boncelet_reshape = np.log2(times_exp3_boncelet[0]).reshape(-1, 1)

regr1 = linear_model.LinearRegression()
regr1.fit(x_row_reshape, times_exp3_spillover_reshape)
print('Coefficients: \n', regr1.coef_)

regr2 = linear_model.LinearRegression()
regr2.fit(x_row_reshape, times_exp3_boncelet_reshape)
print('Coefficients: \n', regr2.coef_)

regr_1_coef = int(regr1.coef_ * 100)/100.0
regr_2_coef = int(regr2.coef_ * 100)/100.0

plt.plot(np.log2([n[0] for n in n_list]), np.log2(times_exp3_spillover[0]), colors[0], label=r'Spillover, $c_1=1$, slope={}'.format(regr_1_coef))
plt.plot(np.log2([n[0] for n in n_list]), np.log2(times_exp3_boncelet[0]), colors[0]+"--", label='Boncelet, $c_1=1$, slope={}'.format(regr_2_coef))


x_row = np.log2([n[0] for n in n_list])
x_row_reshape = x_row.reshape(-1, 1)
times_exp3_spillover_reshape = np.log2(times_exp3_spillover[1]).reshape(-1, 1)
times_exp3_boncelet_reshape = np.log2(times_exp3_boncelet[1]).reshape(-1, 1)

regr1 = linear_model.LinearRegression()
regr1.fit(x_row_reshape, times_exp3_spillover_reshape)
print('Coefficients: \n', regr1.coef_)

regr2 = linear_model.LinearRegression()
regr2.fit(x_row_reshape, times_exp3_boncelet_reshape)
print('Coefficients: \n', regr2.coef_)

regr_1_coef = int(regr1.coef_ * 100)/100.0
regr_2_coef = int(regr2.coef_ * 100)/100.0

plt.plot(np.log2([n[0] for n in n_list]), np.log2(times_exp3_spillover[1]), colors[1], label=r'Spillover, $c_1=n$, slope={}'.format(regr_1_coef))
plt.plot(np.log2([n[0] for n in n_list]), np.log2(times_exp3_boncelet[1]), colors[1]+"--", label=r'Boncelet, $c_1=n$, slope={}'.format(regr_2_coef))



plt.xlabel(r"$n$")
plt.ylabel("Time Elapsed (log2 seconds)")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True)

d_3_spillover_times_df = pd.DataFrame(times_exp3_spillover)
d_3_boncelet_times_df = pd.DataFrame(times_exp3_boncelet)
d_3_spillover_times_df.to_csv('d_3_spillover_times_df.csv')
d_3_boncelet_times_df.to_csv('d_3_boncelet_times_df.csv')



# Second set of tests, definitely will include in the paper if Takehiro, Zach, and Professor Greenwald like this test
# This one produces a table where the output are the times, the x columns are varying n from 5, 10, 15, 20, 25
# and the y columns are varying C = {1}, {1, 2}, {1, 2, 3}, {1, 2, 3, 4}, {1, 2, 3, 4, 5}

# Warning, Bapat Beg will NOT finish running in a reasonable amount of time. Cut off the tests after a minute of running

# Also, the values for this test for Boncelet, Spillover, BapatBeg should be close to the same for [4] and [2, 2],
# the same for [8] and [4, 4], ..., and the same for [20] and [10, 10]. If they're not, please let me know
n_vecs = [[6], [12], [18], [24], [30]]
order_inputs = [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]]
num_trials = 5

# Change this line to change which algorithm is running, alg in ("BapatBeg", "Glueck", "Boncelet", "Spillover", "Unconditional", "Conditional")
alg = "Boncelet"
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
my_table_pd[alg].to_csv(alg + '_experiment_4.csv')



# Second set of tests, definitely will include in the paper if Takehiro, Zach, and Professor Greenwald like this test
# This one produces a table where the output are the times, the x columns are varying n from 5, 10, 15, 20, 25
# and the y columns are varying C = {1}, {1, 2}, {1, 2, 3}, {1, 2, 3, 4}, {1, 2, 3, 4, 5}

# Warning, Bapat Beg will NOT finish running in a reasonable amount of time. Cut off the tests after a minute of running

# Also, the values for this test for Boncelet, Spillover, BapatBeg should be close to the same for [4] and [2, 2],
# the same for [8] and [4, 4], ..., and the same for [20] and [10, 10]. If they're not, please let me know
n_vecs = [[6], [12], [18], [24], [30]]
order_inputs = [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]]
num_trials = 5

# Change this line to change which algorithm is running, alg in ("BapatBeg", "Glueck", "Boncelet", "Spillover", "Unconditional", "Conditional")
alg = "Spillover"
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
my_table_pd[alg].to_csv(alg + '_experiment_4.csv')