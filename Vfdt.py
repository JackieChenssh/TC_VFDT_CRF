import numpy as np
from VfdtNode import VfdtNode
# very fast decision tree class, i.e. hoeffding tree
class Vfdt:
    def __init__(self, features, feature_types, delta=0.01, nmin=100, tau=0.1, using_nmins_adaptation = False):
        """
        :features: list of data features
        :delta: used to compute hoeffding bound, error rate
        :nmin: to limit the G computations
        :tau: to deal with ties
        """
        self.features = features
        self.delta = delta
        self.nmin = nmin
        self.tau = tau
        self.root = VfdtNode(features,feature_types)
        self.n_examples_processed = 0
        self.using_nmins_adaptation = using_nmins_adaptation
        # print(self.features, self.delta, self.tau, self.n_examples_processed)
        # self.print_tree()
        # print("--- / __init__ ---")

    # update the tree by adding one or many training example(s)
    def update(self, X, y):
        from tqdm.notebook import tqdm
        # X, y = check_X_y(X, y)
        for x, _y in tqdm(list(zip(X, y))):
            self._update(x, _y)
            
    def _update(self, x, _y):                
        self.n_examples_processed += 1
        node = self.root.sort_example(x)
        node.update_stats(x, _y)

        if node.new_examples_seen > self.nmin and len(node.class_frequency) != 1:
            node.new_examples_seen = 0  # reset
            nijk = node.nijk
            first_min,second_min = 1,1

            split_feature = ''
            split_value = None

            for feature,feature_type in [f for f in zip(node.possible_split_features,node.possible_split_featuretypes) if f[0] is not None]:
                njk = nijk[feature]
                gini, value = node.gini(njk, node.class_frequency,feature_type)
                if gini < first_min:
                    first_min = gini
                    split_feature,split_value = feature,value
                elif gini < second_min:
                    second_min = gini
#                 pdb.set_trace()
            epsilon = node.hoeffding_bound(self.delta)
            class_frequency = np.array(list(node.class_frequency.values()))
            g_X0 = 1 - np.sum((class_frequency / np.sum(class_frequency)) ** 2)

            delta_gini = second_min - first_min
            if (first_min < g_X0) and (delta_gini > epsilon or epsilon < self.tau):
                left = VfdtNode(node.possible_split_features,node.possible_split_featuretypes,node)
                right = VfdtNode(node.possible_split_features,node.possible_split_featuretypes,node)

                node.split_feature = split_feature
                node.split_value = split_value
                node.left_child = left
                node.right_child = right

                node.nijk.clear()  # reset stats

                if isinstance(split_value, list):
                    # discrete split value list's length = 1, stop splitting
                    new_features = [None if f == split_feature else f for f in node.possible_split_features]
#                         pdb.set_trace()
                    if len(split_value[0]) <= 1:
                        left.possible_split_features = new_features
                    if len(split_value[1]) <= 1:
                        right.possible_split_features = new_features

            elif self.using_nmins_adaptation:
                self.nmins = sqrt(-log(len(node.class_frequency)) ** 2 * log(self.delta) / (2 * max(self.tau,delta_gini) ** 2))

    # predict test example's classification
    def predict(self, X):
        from tqdm.notebook import tqdm
        return [self.root.sort_example(x).most_frequent() for x in tqdm(X)]

    def toList(self):
        node = self.root
        node_dict = {id(node) : 0}
        node_stack = []
        node_list = []

        while node_stack or not node.is_leaf():
            while not node.is_leaf():
                node_list.append(['root' if not node.parent else 'left_branch' if id(node) == id(node.parent.left_child) else 'right_branch',
                                  node_dict[id(node.parent)] if node.parent else -1,node])

                if node.left_child and node.right_child:
                    node_stack.append(node)        
                node = (node.left_child if node.left_child else node.right_child)
                node_dict[id(node)] = len(node_list)

            node_list.append(['left_leaf' if id(node) == id(node.parent.left_child) else 'right_leaf',node_dict[id(node.parent)],node])

            if node_stack:
                node = node_stack[-1].right_child
                node_dict[id(node)] = len(node_list)
                del node_stack[-1]

        node_list.append(['root' if not node.parent else 'left_leaf' if id(node) == id(node.parent.left_child) else 'right_leaf',
            node_dict[id(node.parent)] if node.parent else -1,node])  
    
        return node_list
    
    def toGraph(self,max_node = None):
        branch_template = '%d -> %d [labeldistance=1.5, labelangle=%s, headlabel="%s"] ;'
        node_template = '%d [label=<%s>, fillcolor="#e5813900"] ;'
        leaf_label = 'main_class:%s'
        branch_label = 'split_feature:%s<br />split_value:%s'

        lsDot = ['digraph Tree {',
                 'node [shape=box, style="filled, rounded", color="black", fontname=helvetica] ;',
                 'edge [fontname=helvetica] ;']

        node_list = self.toList()
        
        if max_node:
            if isinstance(max_node,int):
                node_list = node_list[:max_node]
            else:
                raise TypeError('max_node must be int')
        
        if len(node_list) == 1:
            lsDot.append(node_template % (0,leaf_label % node_list[0][-1].most_frequent()))
        else:
            for element in node_list:
                if(element[0][-4:] == 'leaf'):    
                    lsDot.append(node_template % (node_list.index(element),leaf_label % element[-1].most_frequent()))
                else:
                    lsDot.append(node_template % (node_list.index(element),branch_label % (element[-1].split_feature,element[-1].split_value)))

                if(element[0][:4] == 'left'):
                    lsDot.append(branch_template % (element[1],node_list.index(element),45,'True'))
                elif (element[0][:5] == 'right'):
                    lsDot.append(branch_template % (element[1],node_list.index(element),45,'False'))

        lsDot.append('}')       
        return '\n'.join(lsDot)
    
    def print_tree(self, node=None):
        if node is None:
            self.print_tree(self.root)
        elif node.is_leaf():
            print('Leaf')
        else:
            print(node.split_feature)
            self.print_tree(node.left_child)
            self.print_tree(node.right_child)