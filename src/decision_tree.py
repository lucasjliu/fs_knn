'''
1. remove feature_set. make select_feature_set() in fs.py
	make preprocessing.py
2. classifier.py: fit()=0, predict()=0, accuracy() k_fold()
3. class knn_classifier and decision_tree.py
'''
#====================cross validation=======================
import numpy as np
import random
from math import ceil
from numpy import s_

def loocv(data, objective_func):
	accuracy_sum = 0
	n = len(data)
	for i in range(n):
		testing_set = np.copy([data[i]])
		training_set = np.delete(data, i, axis=0)
		accuracy_sum += objective_func(training_set, testing_set)
	return accuracy_sum / n

def k_fold_cv(k, data, objective_func):
	accuracy_sum = 0
	n = len(data)
	random.shuffle(data)
	l = ceil(n / k)
	for j in range(k):
		begin = j * l
		end = min(n, (j+1)*l)
		test_set = np.copy(data[begin:end,:])
		train_set = np.concatenate((data[:begin,:], data[end:,:]), axis=0)
		accuracy_sum += objective_func(train_set, test_set)
	return accuracy_sum / k

#=====================decision tree=========================
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree
from heapq import heappush, heappop

DecisionTree = DecisionTreeClassifier

def dt_build(training_set):
	dt = DecisionTree(criterion="entropy")
	dt.fit(training_set[:,1:], training_set[:,0])
	return dt

def dt_accuracy(dt, testing_set):
	if len(testing_set) == 0:
		return 0
	correct_count = 0
	predict = dt.predict(testing_set[:,1:])
	for i in range(len(testing_set)):
		if testing_set[i][0] == predict[i]:
			correct_count += 1
	return correct_count / len(testing_set)

def dt_topk_features(dt, k):
	heap = []
	importance = dt.feature_importances_
	for i in range(len(importance)):
		heappush(heap, (-importance[i], i+1))
	topk = []
	for x in range(k):
		topk.append(heappop(heap)[1])
	return topk

def print_tree(tree, feature_names):
	tree_ = tree.tree_
	feature_name = [
		feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
		for i in tree_.feature
	]
	print("def tree({}):".format(", ".join(feature_names)))

	def recurse(node, depth):
		indent = "    " * depth
		if tree_.feature[node] != _tree.TREE_UNDEFINED:
			name = feature_name[node]
			threshold = tree_.threshold[node]
			print("{}if {} <= {}:".format(indent, name, "{0:.2f}".format(threshold)))
			recurse(tree_.children_left[node], depth + 1)
			print("{}else:  # if {} > {}".format(indent, name, "{0:.2f}".format(threshold)))
			recurse(tree_.children_right[node], depth + 1)
		else:
			nums = list(tree_.value[node][0])
			mode = max(nums)
			print("{}class {}".format(indent, nums.index(mode)) + 
				" conf {0:.0f}%".format(mode * 100.0 / sum(nums)))

	recurse(0, 1)

def test():
	label = np.array([0.,0.,0.,1.,1.,1.])
	data = np.array([[1,1,1], [1,2,1], [0,0,0], [-1,-1,-1], [-1,1,2], [0,-1,-1]])

	test_data = np.array([[2,2,2], [-2,2,2], [-2,-2,2], [-2,-2,-2]])
	test_data = np.insert(test_data, 0, np.array([0.,1.,1.,1.]), axis=1)

	data = np.insert(data, 0, label, axis=1)

	dt = dt_build(data)
	print_tree(dt, ['a0', 'a1', 'a2'])
	print(dt.predict(test_data[:,1:]))
	print(dt_accuracy(dt, test_data))

#=====================experiment=========================
from preprocessing import no_norm
from preprocessing import zero_one_norm
from preprocessing import z_norm

def load_txt(file, d = ' '):
	return np.genfromtxt(file, delimiter = d) #TODO: handle invalid file name

def no_format(data, norm):
		return data

def format_leaf(data, norm):
	data = np.delete(data, 1, axis=1)
	return norm(data)

DTCLF = (lambda train_set, test_set: dt_accuracy(dt_build(train_set), test_set))
LOOCV = lambda data: loocv(data, DTCLF)
FOLDCV = lambda data: k_fold_cv(2, data, DTCLF)

def print_exp_result(train_file, test_file, format=no_format, norm=no_norm, exp_id='exp', feature_select=None):
	train_set = format(load_txt(train_file, ","), norm)
	test_set = format(load_txt(test_file, ","), norm)
	print("Dataset: " + str(exp_id))
	feature_names = ['Eccentricity', 'AspectRatio', 'Elongation', 'Solidity', 'StochasticConvexity',
		'IsoperimetricFactor', 'MaximalIndentation', 'Lobedness', 'AverageIntensity',
		'AverageContrast', 'Smoothness', 'ThirdMoment', 'Uniformity', 'Entropy']
	print("LOOCV: " + str(LOOCV(train_set)))
	print("2FOLD_CV: " + str(FOLDCV(train_set)))
	dt = dt_build(train_set)
	print(dt.feature_importances_)
	print(dt_topk_features(dt, 11))
	#print_tree(dt, feature_names)
	print("\n")

def do_exp():
	print_exp_result("Leaf/leaf.csv", "Leaf/leaf.csv", format_leaf, no_norm, "leaf_no_norm")
	print_exp_result("Leaf/leaf.csv", "Leaf/leaf.csv", format_leaf, zero_one_norm, "leaf_zero_one_norm")
	print_exp_result("Leaf/leaf.csv", "Leaf/leaf.csv", format_leaf, z_norm, "leaf_z_norm")

if __name__ == "__main__":
	do_exp()
