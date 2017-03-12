import numpy as np
import math
from heapq import heappush, heappop
import operator
from scipy.spatial import distance

def sqeuclidean(sample1, sample2):
	'''
	distance = 0
	for i in range(len(sample1)):
		distance += pow((sample1[i] - sample2[i]), 2)
	return distance
	'''
	return distance.sqeuclidean(sample1, sample2)

def knn_get_n(training_set, test_sample, k):
	dist_heap = []
	for sample in training_set:
		dist = sqeuclidean(test_sample[1:], sample[1:])
		label = sample[0]
		heappush(dist_heap, (dist, label))
	neighbors = []
	for x in range(k):
		neighbors.append(heappop(dist_heap)[1])
	return neighbors

def knn_vote(neighbors):
	votes = {}
	for x in neighbors:
		label = x
		if label in votes:
			votes[label] += 1
		else:
			votes[label] = 1
	max_count = 0
	max_vote = None
	for label, count in votes.items():
		if count > max_count:
			max_vote = label
			max_count = count
	return max_vote

def knn_predict(training_set, test_sample, k):
	return knn_vote(knn_get_n(training_set, test_sample, k))

def knn_accuracy(training_set, testing_set, k):
	if len(testing_set) == 0:
		return 0
	correct_count = 0
	for sample in testing_set:
		if sample[0] == knn_predict(training_set, sample, k):
			correct_count += 1
	return correct_count / len(testing_set)

KNN = (lambda k: lambda training_set, testing_set:
		knn_accuracy(training_set, testing_set, k))