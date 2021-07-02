import numpy as np
from itertools import combinations
from math import log2,log,sqrt
import pdb
# VFDT node class
class VfdtNode:
    def __init__(self, possible_split_features,possible_split_featuretypes = None,parent = None):
        """
        nijk: statistics of feature i, value j, class
        possible_split_features: features list
        """
        self.parent = parent
        self.left_child = None
        self.right_child = None
        self.split_feature = None
        self.split_value = None  # both continuous and discrete value
        self.new_examples_seen = 0
        self.total_examples_seen = 0
        self.class_frequency = {}
        self.nijk = {f: {} for f in possible_split_features}
        self.possible_split_features = possible_split_features
        self.possible_split_featuretypes = possible_split_featuretypes
    def is_leaf(self):
        return self.left_child is None and self.right_child is None

    # recursively trace down the tree
    # to distribute data examples to corresponding leaves
    def sort_example(self, x):
        if self.is_leaf():
            return self
        else:
            if isinstance(self.split_value, list):  # discrete value
                return (self.left_child.sort_example(x) if x[self.possible_split_features.index(self.split_feature)] in self.split_value[0] else self.right_child.sort_example(x))
            else:  # continuous value
                return (self.left_child.sort_example(x) if x[self.possible_split_features.index(self.split_feature)] <= self.split_value else self.right_child.sort_example(x))

    # the most frequent class
    def most_frequent(self):
        try:
            return max(self.class_frequency,key = self.class_frequency.get)
        except ValueError:
            # if self.class_frequency dict is empty, go back to parent
            return self.parent.most_frequent()

    # update leaf stats in order to calculate gini
    def update_stats(self, x, y):
        for i in [f for f in self.possible_split_features if f is not None]:
            value = x[self.possible_split_features.index(i)]
#             pdb.set_trace()
            if value not in self.nijk[i]:
                self.nijk[i][value] = {y: 1}
            else:
                try:
                    self.nijk[i][value][y] += 1
                except KeyError:
                    self.nijk[i][value][y] = 1

        self.total_examples_seen += 1
        self.new_examples_seen += 1
        try:
            self.class_frequency[y] += 1
        except KeyError:
            self.class_frequency[y] = 1

    def check_not_splitting(self):
        # compute gini index for not splitting
        class_frequency = np.array(list(self.class_frequency.values()))
        return 1 - np.sum((class_frequency / np.sum(class_frequency)) ** 2)

    def hoeffding_bound(self, delta):
        return sqrt(-log(len(self.class_frequency)) **2  * log(delta) / (2 * self.total_examples_seen))

    def gini(self, njk, class_frequency,feature_type):
        D = self.total_examples_seen
        m1 = 1  # minimum gini
        Xa_value = None
        feature_values = list(njk.keys())  # list() is essential
        if feature_type == 'continuous':  # numeric  feature values
            sort = np.array(sorted(feature_values))
            # vectorized computation, like in R
            split = (sort[0:-1] + sort[1:])/2

            D1_class_frequency = {j: 0 for j in class_frequency.keys()}
            for index in range(len(split)):
                nk = njk[sort[index]]
                for j in nk:
                    D1_class_frequency[j] += nk[j]
                D1 = sum(D1_class_frequency.values())
                D2 = D - D1
                g_d1 = 1
                g_d2 = 1

                D2_class_frequency = {}
                for key, value in class_frequency.items():
                    if key in D1_class_frequency:
                        D2_class_frequency[key] = value - \
                            D1_class_frequency[key]
                    else:
                        D2_class_frequency[key] = value

                for key, v in D1_class_frequency.items():
                    g_d1 -= (v/D1)**2
                for key, v in D2_class_frequency.items():
                    g_d2 -= (v/D2)**2
                g = g_d1*D1/D + g_d2*D2/D
                if g < m1:
                    m1 = g
                    Xa_value = split[index]
                # elif m1 < g < m2:
                    # m2 = g
            return [m1, Xa_value]

        else:  # discrete feature_values
            length = len(njk)
            if length > 10:  # too many discrete feature values, estimate
                for j, k in njk.items():
                    D1 = sum(k.values())
                    D2 = D - D1
                    g_d1 = 1
                    g_d2 = 1

                    D2_class_frequency = {}
                    for key, value in class_frequency.items():
                        if key in k:
                            D2_class_frequency[key] = value - k[key]
                        else:
                            D2_class_frequency[key] = value
                    for key, v in k.items():
                        g_d1 -= (v/D1) ** 2

                    if D2 != 0:
                        for key, v in D2_class_frequency.items():
                            g_d2 -= (v/D2) ** 2
                    g = g_d1 * D1/D + g_d2 * D2/D
                    if g < m1:
                        m1 = g
                        Xa_value = [j]
                right = list(np.setdiff1d(feature_values, Xa_value))

            else:  # fewer discrete feature values, get combinations
                for i in self.select_combinations(feature_values):
                    left = list(i)
                    D1_class_frequency = {
                        key: 0 for key in class_frequency.keys()}
                    D2_class_frequency = {
                        key: 0 for key in class_frequency.keys()}
                    for j, k in njk.items():
                        for key, value in class_frequency.items():
                            if j in left:
                                if key in k:
                                    D1_class_frequency[key] += k[key]
                            else:
                                if key in k:
                                    D2_class_frequency[key] += k[key]
                    g_d1 = 1
                    g_d2 = 1
                    D1 = sum(D1_class_frequency.values())
                    D2 = D - D1
                    for key, v in D1_class_frequency.items():
                        g_d1 -= (v/D1) ** 2
                    for key, v in D2_class_frequency.items():
                        g_d2 -= (v/D2) ** 2
                    g = g_d1 * D/D + g_d2 * D2/D
                    if g < m1:
                        m1 = g
                        Xa_value = left
                    # elif m1 < g < m2:
                        # m2 = g
                right = list(np.setdiff1d(feature_values, Xa_value))
            return [m1, [Xa_value, right]]

    # divide values into two groups, return the combination of left groups
    def select_combinations(self, feature_values):
        combination = []
        e = len(feature_values)
        if e % 2 == 0:
            end = int(e/2)
            for i in range(1, end+1):
                if i == end:
                    cmb = list(combinations(feature_values, i))
                    enough = int(len(cmb)/2)
                    combination.extend(cmb[:enough])
                else:
                    combination.extend(combinations(feature_values, i))
        else:
            end = int((e-1)/2)
            for i in range(1, end + 1):
                combination.extend(combinations(feature_values, i))
        return combination