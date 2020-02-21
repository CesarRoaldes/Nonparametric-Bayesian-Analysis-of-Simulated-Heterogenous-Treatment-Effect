import os
import pydotplus
import time
import pickle
import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor

dir_path = os.path.dirname(os.path.realpath(__file__))

##############
# Require the part1.R script to be run in order to generate data
df = pd.read_csv(dir_path+'/../data/simulated_data.csv', sep=',')
df.drop(['inter'], axis=1, inplace=True)
df.rename({"X_9": "X_sex", "X_10": "X_new"}, axis=1, inplace=True)


##############
# Compute the y* transformed reponse for TOT
q = df['t'].mean()
df['y_tot'] = df['y_hete'] * (df['t'] - q) / (q * (1 - q))

df.drop(['y_hete'], axis=1, inplace=True)
pos_y = df.columns.get_loc("y_tot")
pos_x_sex = df.columns.get_loc("X_sex")
pos_x_new = df.columns.get_loc("X_new")


###############################################
###############################################
#           BAYESIAN BOOTSTRAP
###############################################
###############################################

##############
# Fitting N trees to study the uncertainty associated with the TOT trees
N = 1000
bf = RandomForestRegressor(n_estimators=N,
                           max_depth=5,
                           min_samples_leaf=10000,
                           bootstrap=False,
                           bayesian_bootstrap=True,
                           n_jobs=-1,
                           random_state=1)
bf.fit(df.iloc[:, 1:pos_y].values, df.iloc[:, pos_y].values)

# Saving the model
pickle.dump(bf, open(dir_path+'/../pickle/bf_tot.pkl', 'wb'))
bf = pickle.load(open(dir_path+'/../pickle/bf_tot.pkl', 'rb'))


##############
# Predictions on m observations
m = 10
for i in range(m):
    pred = bf.predict(df.iloc[i, 1:pos_y].values.reshape(1, -1))
    print("#%s obs. with X_sex=%s and X_new=%s :"
          "\n\t -> Treatment effect = %s"
          % (i, df.iloc[i, pos_x_sex], df.iloc[i, pos_x_new], pred[0]))

# Store the N trees
tot_trees = bf.estimators_
# Matrix storing the probability of occurence of a feature (by columns) at each
# depth 1:5 (by rows)
features_incertainty = np.zeros((5, 10))


for estimator in tot_trees:

    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold

    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth

    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    # For each feature, the minimum depth is found and the uncertainty is
    # updated
    for k in range(10):
        tmp = []
        for i in range(n_nodes):
            if feature[i] == k:
                tmp.append(node_depth[i])
        if len(tmp) > 0:
            features_incertainty[min(tmp):, k] += 1/N

# Displaying results
print('Posterior probability of the TOT algorithm splitting at or above '
      'depths 1 to 5\n')
for k in range(10):
    print('\t* On variable %s :' % df.iloc[:, 1:pos_y].columns[k])
    for j in range(5):
        print('\t\tdepth = %s : %s' % (j, round(features_incertainty[j][k], 2)))


###########
# Study of the trunk of a tree
N = 1
bf = RandomForestRegressor(n_estimators=N,
                           max_depth=3,
                           min_samples_leaf=100000,
                           bootstrap=False,
                           bayesian_bootstrap=True,
                           n_jobs=-1,
                           random_state=1)
bf.fit(df.iloc[:, 1:pos_y].values, df.iloc[:, pos_y].values)

tot_trees = bf.estimators_
arbre_test = tot_trees[0]
dot_data = sklearn.tree.export_graphviz(arbre_test, out_file=None,
                                        feature_names=df.columns[1:-1])
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf(dir_path+'/../graphs/trunk.pdf')
