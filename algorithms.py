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

class BapatBeg:

    def __init__(self, n_vec, orders, bins):
        print(len(orders))
        self.m = len(n_vec)
        self.n = sum(n_vec)
        self.n_vec = n_vec
        if orders:
            self.orders = orders
        else:
            self.orders = [i + 1 for i in range(self.n)]
        self.bins = bins
        self.d = len(bins[0])

        for bin in self.bins:
            bin.append(1 - sum(bin))

        self.n_set_perm = [[]]
        for i in range(self.n):
            new_n_set_perm = []
            for perm in self.n_set_perm:
                new_n_set_perm.append(perm)
                new_n_set_perm.append(perm + [i])
            self.n_set_perm = new_n_set_perm

        self.summation_indices = []
        for j in range(self.orders[0], self.n + 1):
            self.summation_indices.append([j])
        for i in range(1, self.d):
            new_indices = []
            for index in self.summation_indices:
                for j in range(max(index[-1], self.orders[i]), self.n + 1):
                    new_indices.append(index + [j])
            self.summation_indices = new_indices

        self.log_factorial_table = [0]
        for i in range(1, self.n + 1):
            self.log_factorial_table.append(self.log_factorial_table[-1] + np.log(i))

    def construct_row(self, ball_type, indices):
        block_row = []
        for j in range(indices[0]):
            block_row.append(self.bins[ball_type][0])
        for i in range(1, self.d):
            for j in range(indices[i] - indices[i-1]):
                block_row.append(self.bins[ball_type][i])
        for j in range(self.n - indices[-1]):
            block_row.append(self.bins[ball_type][-1])
        return block_row

    def block_matrix(self, indices):
        block_matrix = []
        for ball_type in range(self.m):
            block_row = self.construct_row(ball_type, indices)
            for j in range(self.n_vec[ball_type]):
                block_matrix.append(block_row)
        return block_matrix

    def lazy_method(self, A):
        total_sum = 0
        n = len(A)
        for n_perm in list(permutations([i for i in range(n)])):
            helper_product = 1
            for i in range(n):
                helper_product *= A[i][n_perm[i]]
            total_sum += helper_product
        return total_sum

    def ryser_method(self, A):
        n = len(A)
        total_sum = 0
        for n_perm in self.n_set_perm:
            helper_sum = (-1) ** len(n_perm)
            for i in range(self.n):
                helper_sum *= sum([A[i][j] for j in n_perm])
            total_sum += helper_sum
        return (-1) ** n * total_sum

    def compute(self):
        running_sum = 0
        for indices in self.summation_indices:
            helper_product = self.log_factorial_table[indices[0]]
            for i in range(1, self.d):
                helper_product += self.log_factorial_table[indices[i] - indices[i-1]]
            helper_product += self.log_factorial_table[self.n - indices[-1]]
            running_sum += np.exp(np.log(max(10 ** -15, self.ryser_method(self.block_matrix(indices)))) - helper_product)
        return running_sum



class GlueckSingle:

    def __init__(self, n_vec, orders, bins):
        self.m = len(n_vec)
        self.n = sum(n_vec)
        self.n_vec = n_vec
        if orders:
            self.orders = orders
        else:
            self.orders = [i + 1 for i in range(self.n)]
        self.bins = bins
        self.d = len(bins[0])

        for bin in self.bins:
            bin.append(1 - sum(bin))

        self.summation_indices = []
        for j in range(self.orders[0], self.n + 1):
            self.summation_indices.append([j])
        for i in range(1, self.d):
            new_indices = []
            for index in self.summation_indices:
                for j in range(max(index[-1], self.orders[i]), self.n + 1):
                    new_indices.append(index + [j])
            self.summation_indices = new_indices
        for index in self.summation_indices:
            index.insert(0, 0)
            index.append(self.n)

        self.log_factorial_table = [0]
        for i in range(1, self.n + 1):
            self.log_factorial_table.append(self.log_factorial_table[-1] + np.log(i))


    def compute(self):
        # total = np.exp(self.log_factorial_table[self.n])
        total = 0
        for summation_index in self.summation_indices:
            total += np.exp(self.log_factorial_table[self.n] + np.sum([np.log(self.bins[0][j]) *
            (summation_index[j+1] - summation_index[j]) - self.log_factorial_table[summation_index[j+1] - summation_index[j]]
            for j in range(0, self.d+1)]))
        return total



class GlueckDouble:

    def __init__(self, n_vec, orders, bins):
        self.m = len(n_vec)
        self.n = sum(n_vec)
        self.n_vec = n_vec
        if orders:
            self.orders = orders
        else:
            self.orders = [i + 1 for i in range(self.n)]
        self.bins = bins
        self.d = len(bins[0])

        for bin in self.bins:
            bin.append(1 - sum(bin))

        self.summation_indices = []
        for j in range(self.orders[0], self.n + 1):
            self.summation_indices.append([j])
        for i in range(1, self.d):
            new_indices = []
            for index in self.summation_indices:
                for j in range(max(index[-1], self.orders[i]), self.n + 1):
                    new_indices.append(index + [j])
            self.summation_indices = new_indices
        for index in self.summation_indices:
            index.insert(0, 0)
            index.append(self.n)

        self.log_factorial_table = [0]
        for i in range(1, self.n + 1):
            self.log_factorial_table.append(self.log_factorial_table[-1] + np.log(i))

    def compute(self):
        total = 0
        static_multipliers = self.log_factorial_table[self.n_vec[0]] + self.log_factorial_table[self.n_vec[1]]
        for summation_index in self.summation_indices:
            lambda_vectors = compute_lambda_vector(summation_index, self.d, self.n_vec[0])
            for lambda_vec in lambda_vectors:
                factorial_multipliers = [self.log_factorial_table[lambda_vec[j]] - self.log_factorial_table[summation_index[j+1] - summation_index[j] - lambda_vec[j]] for j in range(self.d + 1)]
                probability_multipliers = [lambda_vec[j] * np.log(self.bins[0][j]) + (summation_index[j+1] - summation_index[j] - lambda_vec[j]) * np.log(self.bins[1][j]) for j in range(self.d + 1)]
                total += np.exp(sum(probability_multipliers) + sum(factorial_multipliers) + static_multipliers)
        return total


# summation_indices will transform into [i_{j+1} - i_{j} for j in range(len(orders) + 1)]
# target is the desired number
def compute_lambda_vector(summation_indices, order_len, target):
    differences = [summation_indices[j+1] - summation_indices[j] for j in range(order_len + 1)]
    summed_differences = [sum(differences[i:]) for i in range(len(differences))]
    lambda_vectors, total_remaining = [], []
    # print("Summed differences: ")
    # print(summed_differences[1])
    for i in range(max(0, target - summed_differences[1]), min(target + 1, differences[0] + 1)):
        lambda_vectors.append([i])
        total_remaining.append(target - i)
    for i in range(1, len(differences) - 1):
        new_vectors, new_remaining = [], []
        for j in range(len(lambda_vectors)):
            vector, remaining = lambda_vectors[j], total_remaining[j]
            for k in range(max(0, remaining - summed_differences[i + 1]), min(remaining + 1, differences[i] + 1)):
                new_vectors.append(vector + [k])
                new_remaining.append(remaining - k)
        lambda_vectors, total_remaining = new_vectors, new_remaining
    for j in range(len(lambda_vectors)):
        vector, remaining = lambda_vectors[j], total_remaining[j]
        vector.append(remaining)
    #print(lambda_vectors)
    return lambda_vectors



class Unconditional:

    def __init__(self, n_vec, orders, bins):
        # Input:          n_vec, a length m list of the number of observations from each type
        # Input:          orders, a maximally length n list of the order statistics
        #                 to be computed for. If None, this is initialized for all order statistics.
        #                 orders must be the same length as bins[0].
        # Input:          bins, a size m by d matrix of bin probabilities
        self.n_vec = n_vec
        self.n = sum(n_vec)
        self.m = len(n_vec)
        self.s = len(bins[0])
        self.bins = None
        self.log_table = [0 for _ in range(self.n+1)]
        # In case the problem has already been loaded, don't load again
        self.loaded = False

        # Pre-processing step; if selected orders, append 0 at front, else we care about all order statistics 1:n
        if orders:
            self.orders = [0] + orders
        else:
            self.orders = [i for i in range(self.n + 1)]

        self.bins = [[0 for _ in range(self.n)] for _ in range(self.m)]
        for i in range(self.m):
            for j in range(self.s):
                self.bins[i][self.orders[j]] = bins[i][j]

        # g_set is a list such that g_set[i] is the set of all non negative integer valued vectors of length m
        # such that the elements sum to i and the entries are upper bounded by n_vec[i]
        self.g_set = [[] for _ in range(self.n + 1)]
        self.p = np.zeros([self.m, self.n, self.n])  # Used to construct the p_1 probabilities
        self.augmented_n_vec = [i + 1 for i in self.n_vec]  # Helper variable to construct probability table
        self.T = np.zeros(self.augmented_n_vec + [self.n + 1])  # Dynamic programming table
        self.T[tuple(np.zeros(self.m + 1).astype(int))] = 1  # Initialize 0's entry to 1

    # Clears dynamic programming table without clearing loaded p, bins, g_set, and factorial tables
    def clear(self):
        self.T = np.zeros(self.augmented_n_vec + [self.n + 1])  # Dynamic programming table
        self.T[tuple(np.zeros(self.m + 1).astype(int))] = 1

    # Construct all of the children of a given vector
    @staticmethod
    def children(c_vector_init):
        c_vector = deepcopy(c_vector_init)
        for i in range(len(c_vector)):
            c_vector[i] += 1
        children_i = [[i] for i in range(c_vector.pop(0))]
        for c in c_vector:
            test_vector = []
            for count in range(c):
                for child in children_i:
                    test_vector.append(child + [count])
            children_i = test_vector
        return children_i[1:]

    # Load the g_set
    def compute_g_set_dict(self):
        all_children = self.children(self.n_vec)
        self.g_set[0] = [np.zeros(self.m).astype(int)]
        for child in all_children:
            self.g_set[sum(child)].append(child)

    # Pre compute log table
    def compute_logs(self):
        for i in range(1, self.n+1):
            self.log_table[i] = np.log(i)

    # Compute the probability table to be used in the recursive sub problems
    def make_p(self):
        for h in range(self.m):
            for i in range(self.n):
                for k in range(i+1):
                    if np.sum(self.bins[h][i-k:]) > 0 and np.sum(self.bins[h][:i-k]) < 1:
                        self.p[h][i][k] = np.sum(self.bins[h][i-k:i+1]) / \
                                          (1 - np.sum(self.bins[h][:i-k]))
                    elif np.sum(self.bins[h][i-k:]) <= 0:
                        self.p[h][i][k] = 0
                    elif np.sum(self.bins[h][:i-k]) >= 1:
                        self.p[h][i][k] = 1

    # Load all of the data if not yet loaded
    def load(self):
        if not self.loaded:
            self.make_p()
            self.compute_logs()
            self.compute_g_set_dict()
            self.loaded = True

    # Main updating step in DP algorithm, using tree to speed up computation
    # i_vec is a vector of the first m entries in the indices of the table entry to be updated
    # k is the number of bins in the super-bin
    # min_bound is a helper number to speed up computation by terminating calculation of irrelevant table entries
    def update_single_log(self, i_vec, k, min_bound):
        my_tree = []
        i = sum(i_vec)
        include_first = False
        # Compute the no-move/0-ball initial probability
        if float(self.T[tuple(list(i_vec) + [k])]) != 0:
            initial_probability = np.log(float(self.T[tuple(list(i_vec) + [k])]))
            for j in range(self.m):
                if float(self.p[j][i][k]) < 1:
                    initial_probability += (self.n_vec[j] - i_vec[j]) * np.log(float((1 - self.p[j][i][k])))
                else:
                    include_first = True
            my_tree.append([-1, initial_probability])
            # Find the immediate children
            for j in range(self.m):
                new_tree = []
                bin_1 = float(self.p[j][i][k])
                # If probability is 0 or negative due to floating point errors, move on without increasing i_vec[j]
                if bin_1 <= 0:
                    for child in my_tree:
                        log_probability = float(child.pop())
                        new_k = child.pop()
                        new_tree.append(child + [i_vec[j], new_k, log_probability])
                # If probability is 1 or larger due to floating point errors, put all samples into current bin
                elif bin_1 >= 1:
                    for child in my_tree:
                        log_probability = float(child.pop())
                        new_k = child.pop()
                        new_tree.append(child + [self.n_vec[j], new_k + self.n_vec[j] - i_vec[j], log_probability])
                # Otherwise, iterate through probabilities of throwing some number of balls into the current bin
                else:
                    for child in my_tree:
                        log_probability = float(child.pop())
                        new_k = child.pop()
                        new_tree.append(child + [i_vec[j], new_k, log_probability])
                        for l in range(1, self.n_vec[j] - i_vec[j] + 1):
                            log_probability += np.log((self.n_vec[j] - i_vec[j] + 1 - l)) - np.log(l) + \
                                           np.log(bin_1) - np.log(1 - bin_1)
                            new_tree.append(child + [i_vec[j] + l, new_k + l, log_probability])
                my_tree = deepcopy(new_tree)

            if include_first:
                for child in my_tree:
                    if child[-2] >= min_bound - 1:
                        self.T[tuple(child[:-1])] += np.exp(child[-1])
            else:
                # Update according to the probabilities
                for child in my_tree[1:]:
                    if child[-2] >= min_bound - 1:
                        self.T[tuple(child[:-1])] += np.exp(child[-1])

    # Main updating algorithm
    def update(self):
        self.update_single_log([0 for _ in range(self.m)], 0, self.orders[1])
        if len(self.orders) > 2:
            counter = 2
            min_bound = self.orders[counter]
            for i in range(self.orders[1], self.orders[-1]):
                i_vec_set = self.g_set[i]
                if i == min_bound and i != self.orders[-1]:
                    counter += 1
                    min_bound = self.orders[counter]
                for i_vec in i_vec_set:
                    for k in range(i):
                        self.update_single_log(i_vec, k, min_bound - i)

    # Computes and returns the final probability of bin condition fulfillment
    def compute(self):
        self.load()
        self.update()
        final_probability = 0
        for i in range(self.orders[-1], self.n + 1):
            for child in self.g_set[i]:
                final_probability += np.sum(self.T[tuple(child)])
        self.clear()
        return final_probability



class Conditional:

    def __init__(self, n_vec, orders, bins):
        # Input:          n_vec, a length m list of the number of observations from each type
        # Input:          orders, a maximally length n list of the order statistics
        #                 to be computed for. If left blank, this is initialized for all order statistics.
        #                 orders must be the same length as bins[0]. Orders is of length d.
        # Input:          bins, a size m by d matrix of bin probabilities
        self.n_vec = n_vec
        self.n = sum(n_vec)
        self.m = len(n_vec)
        self.bins = bins

        # Pre-processing step; if selected orders, append 0 at front, else we care about all order statistics 1:n
        if orders:
            self.orders = orders
        else:
            self.orders = [i for i in range(1, self.n + 1)]

        self.d = len(self.orders)

        # g_set is a list such that g_set[i] is the set of all non negative integer valued vectors of length m
        # such that the elements sum to i and the entries are upper bounded by n_vec[i]
        self.g_set = [[] for _ in range(self.n + 1)]
        self.compute_g_set()

        # first_index is a map from number of balls placed to first unfulfilled bin condition
        self.first_index = None
        self.bin_index_map()

        self.log_table = [0 for _ in range(self.n + 1)]  # To speed up factorial operations
        self.compute_logs()

        self.p = None  # Used to construct the p_1 probabilities
        self.compute_p()

        self.augmented_n_vec = [i + 1 for i in self.n_vec]  # Helper variable to construct dynamic programming table
        self.T = np.zeros(self.augmented_n_vec + [self.d + 1])  # Dynamic programming table
        for k in range(self.orders[-1], self.n + 1):
            for B in self.g_set[k]:
                self.T[tuple(B)] = [1 for _ in range(self.d + 1)]  # Initial conditions

    def compute_logs(self):
        for i in range(1, self.n+1):
            self.log_table[i] = np.log(i)

    # Construct all of the children of a given vector
    @staticmethod
    def children(c_vector_init):
        c_vector = deepcopy(c_vector_init)
        for i in range(len(c_vector)):
            c_vector[i] += 1
        children_i = [[i] for i in range(c_vector.pop(0))]
        for c in c_vector:
            test_vector = []
            for count in range(c):
                for child in children_i:
                    test_vector.append(child + [count])
            children_i = test_vector
        return children_i[1:]

    # Load the g_set
    def compute_g_set(self):
        all_children = self.children(self.n_vec)
        self.g_set[0] = [np.zeros(self.m).astype(int)]
        for child in all_children:
            self.g_set[sum(child)].append(child)

    # Obtains the bin index of the first unfulfilled bin condition
    def find_bin_index(self, B):
        for i in range(self.d):
            if self.orders[i] > B:
                return i
        return self.d

    def bin_index_map(self):
        self.first_index = []
        for i in range(self.n + 1):
            self.first_index.append(self.find_bin_index(i))

    # Finds the first non-zero index, will be useful for finding children probabilities
    def find_positive_index(self, input_list):
        for i in range(self.m):
            if input_list[i] > 0:
                return i
        return 0

    # Obtain the unfulfilled bin condition mapping to initial bin probability
    def compute_p(self):
        # Map between first unfulfilled bin condition, new first unfulfilled bin condition, and type to bin probability
        # self.p = np.zeros([self.m, self.n + 1, self.n + 1])
        self.p = np.zeros([self.m, self.d, self.d + 1])
        for i in range(self.m):
            for j in range(self.d):
                for k in range(self.d + 1):
                    # self.p[i][j][k] = max(0, min(sum(self.bins[i][self.first_index[j]:self.first_index[k]])/
                    #                              (1 - sum(self.bins[i][:self.first_index[j]])), 1))
                    # print((i,j,k))
                    # print(max(0, min(sum(self.bins[i][j:k])/(1 - sum(self.bins[i][:j])), 1)))
                    self.p[i][j][k] = max(0, min(sum(self.bins[i][j:k])/(1 - sum(self.bins[i][:j])), 1))

    # Compute transition probability for a single input vector
    def compute_single(self, input_vector):
        n_vec = input_vector[:-1]
        n_vec_sum = sum(n_vec)
        k = input_vector[-1]
        final_probability = 0
        first_bin_index = self.first_index[n_vec_sum]
        num_required = self.orders[first_bin_index] - n_vec_sum
        p_vec = [self.p[i][first_bin_index - k + 1][first_bin_index + 1] for i in range(self.m)]
        first_n_vec = copy(n_vec)
        first_probability = 0
        for i in range(self.m):
            if p_vec[i] >= 1:
                first_n_vec[i] = self.n_vec[i]
            else:
                first_probability += (self.n_vec[i] - n_vec[i]) * np.log(1 - p_vec[i])
        remaining = [self.n_vec[i] - first_n_vec[i] for i in range(self.m)]
        first_n_vec.append(0)
        initial_tree = [[first_n_vec, first_probability, sum(first_n_vec)]]
        for i in range(self.m):
            tree_size = len(initial_tree)
            if remaining[i] == 0:
                continue
            elif p_vec[i] <= 0:
                continue
            elif p_vec[i] >= 1:
                continue
            else:
                for element in initial_tree[:tree_size]:
                    helper_probability = element[1]
                    for j in range(1, remaining[i] + 1):
                        helper_probability += np.log(p_vec[i]) - np.log((1 - p_vec[i])) - self.log_table[j] + self.log_table[1 + remaining[i] - j]
                        element[0][i] += j
                        initial_tree.append([copy(element[0]), helper_probability, element[2] + j])
                        element[0][i] -= j
        for element in initial_tree:
            if element[2] - n_vec_sum >= num_required:
                element[0][-1] = self.first_index[element[2]] - self.first_index[n_vec_sum]
                final_probability += np.exp(element[1]) * self.T[tuple(element[0])]
        self.T[tuple(input_vector)] = final_probability
        return 0

    # Compute everything
    def compute(self):
        for i in range(self.orders[-1] - 1, self.orders[0] - 1, -1):
            for child in self.g_set[i]:
                # print("CHILD: {}".format(child))
                for k in range(1, 1 + self.first_index[sum(child)]):
                    self.compute_single(child + [k])
        self.compute_single([0 for _ in range(self.m)] + [1])
        return self.T[tuple([0 for _ in range(self.m)] + [1])]



class Spillover:

    def __init__(self, n_vec, orders, bins):
        self.m = len(n_vec)
        self.n = sum(n_vec)
        self.n_vec = n_vec
        if orders:
            self.orders = orders
        else:
            self.orders = [i + 1 for i in range(self.n)]
        self.bins = bins
        self.d = len(bins[0])

        for bin in self.bins:
            bin.append(1 - sum(bin))

        self.order_diff = None
        self.T = None
        self.g_dict = None
        self.g_prune = None
        self.g_children = None
        self.g_indices = None
        self.one_list = None

    def construct_g_dict(self):
        if self.g_indices:
            pass
        order_list = [[]]
        for i in range(self.d):
            new_list = []
            for list in order_list:
                for k in range(self.order_diff[i] + 1):
                    new_list.append(list + [k])
            order_list = new_list

        # Set the key to the number of balls in all of the bins so far
        self.g_dict = dict()
        # Set the key to prune bin combinations when they are infeasible
        self.g_prune = dict()  # If remaining balls < g_prune[list], prune
        for i in range(self.n + 1):
            self.g_dict[i] = []
        for list in order_list:
            tuple_list = tuple(list)
            self.g_dict[int(sum(tuple_list))].append(tuple_list)
            self.g_prune[tuple_list] = max([self.order_diff[i] - list[i] for i in range(self.d)])

        # Construct the direct children of each of the tuples
        self.g_children = dict()
        # Construct the indices for which balls were added or removed
        self.g_indices = dict()
        for list in order_list:
            tuple_list = tuple(list)
            self.g_children[tuple_list] = []
            self.g_indices[tuple_list] = []
            self.g_children[tuple_list].append(tuple_list)
            index_list = [self.d]
            for j in range(self.d - 1, -1, -1):
                if list[j] == self.order_diff[j]:
                    index_list.append(j)
                else:
                    break
            self.g_indices[tuple_list].append(index_list)
            for i in range(self.d):
                if list[i] > 0:
                    list[i] -= 1
                    self.g_children[tuple_list].append(tuple(list))
                    list[i] += 1
                    index_list = [i]
                    for j in range(i-1, -1, -1):
                        if list[j] == self.order_diff[j]:
                            index_list.append(j)
                        else:
                            break
                    self.g_indices[tuple_list].append(index_list)

    def compute(self):
        self.order_diff = [self.orders[0]]
        for i in range(self.d - 1):
            self.order_diff.append(self.orders[i+1] - self.orders[i])
        self.T = np.zeros([1 + self.order_diff[i] for i in range(self.d)])  # Dynamic programming table
        self.T[tuple(np.zeros(self.d).astype(int))] = 1
        self.construct_g_dict()
        count = 0
        for i in range(self.m):  # Iterate over ball types
            num_type = self.n_vec[i]
            bins = self.bins[i]
            for j in range(num_type):  # Iterate over number of a ball type
                for k in range(min(self.n, count + 1), -1, -1):  # Iterate over table in decreasing order of number of balls thrown
                    for list in self.g_dict[k]:  # Iterate over each of the tuples in which l balls have been thrown
                        tuple_list = tuple(list)
                        if self.n - count >= self.g_prune[tuple_list]:  # Only select valid entries to update
                            helper_value = 0
                            for l in range(len(self.g_indices[tuple_list])):
                                base_probability = self.T[self.g_children[tuple_list][l]]
                                transition_probability = sum([bins[i] for i in self.g_indices[tuple_list][l]])
                                helper_value += base_probability * transition_probability
                            self.T[tuple_list] = helper_value
                count += 1
        result = self.T[tuple(self.order_diff)]
        return result



class Boncelet:

    def __init__(self, n_vec, orders, bins):
        self.m = len(n_vec)
        self.n = sum(n_vec)
        self.n_vec = n_vec
        if orders:
            self.orders = orders
        else:
            self.orders = [i + 1 for i in range(self.n)]
        self.bins = bins
        self.d = len(bins[0])

        for bin in self.bins:
            bin.append(1 - sum(bin))
            
        self.T = dict()

        self.one_zero_array = [np.array([0 for _ in range(i)] + [1 for _ in range(i, self.d)]) for i in range(self.d + 1)]
        self.successful_throws = set()
        order_list = [[0]]
        for i in range(1, self.n + 1):
            order_list.append([i])
        for i in range(self.d - 1):
            new_list = []
            for sublist in order_list:
                for j in range(sublist[-1], self.n + 1):
                    new_list.append(sublist + [j])
            order_list = new_list
        for sublist in order_list:
            self.T[tuple(sublist)] = 0
            if all([sublist[i] >= self.orders[i] for i in range(self.d)]):
                self.successful_throws.add(tuple(sublist))

        # Set the key to the number of balls in all of the bins so far
        self.g_dict = dict()
        for i in range(self.n + 1):
            self.g_dict[i] = []
        for list in order_list:
            tuple_list = tuple(list)
            self.g_dict[tuple_list[-1]].append(tuple_list)

        self.T[tuple(np.zeros(self.d).astype(int))] = 1

    def compute(self):
        count = 0
        for i in range(self.m):  # Iterate over ball types
            num_type = self.n_vec[i]
            bins = self.bins[i]
            for j in range(num_type):  # Iterate over number of a ball type
                for k in range(count, -1, -1):  # Iterate over table in decreasing order of number of balls thrown
                    for list in self.g_dict[k]:  # Iterate over each of the tuples in which l balls have been thrown
                        tuple_list = tuple(list)
                        base_probability = self.T[tuple_list]
                        for i in range(self.d):
                            self.T[tuple(self.one_zero_array[i] + np.array(list))] += base_probability * bins[i]
                        self.T[tuple_list] = base_probability * bins[self.d]
                count += 1
        result = sum([self.T[successful_throw] for successful_throw in self.successful_throws])
        return result



