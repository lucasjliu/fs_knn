import numpy as np
import math
from heapq import heappush, heappop
import operator
from sklearn import preprocessing

def euclidean(sample1, sample2, feature_set):
    distance = 0
    for x in feature_set:
        distance += pow((sample1[x] - sample2[x]), 2)
    return distance
 
def knn_get_n(training_set, test_sample, k, feature_set):
    dist_heap = []
    for sample in training_set:
        dist = euclidean(test_sample, sample, feature_set)
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

def knn_predict(training_set, test_sample, k, feature_set):
    return knn_vote(knn_get_n(training_set, test_sample, k, feature_set))

def knn_accuracy(training_set, testing_set, k, feature_set):
    correct_count = 0
    for sample in testing_set:
        if sample[0] == knn_predict(training_set, sample, k, feature_set):
            correct_count += 1
    return correct_count# / len(testing_set)

def loocv(data, feature_set, objective_func):
    accuracy_sum = 0
    n = len(data)
    for i in range(n):
        testing_set = [data[i]]
        training_set = data[0:i]
        training_set.extend(data[i+1:n])
        accuracy_sum += objective_func(training_set, testing_set, feature_set)
    return accuracy_sum / n

KNN = (lambda k: lambda training_set, testing_set, feature_set:
        knn_accuracy(training_set, testing_set, k, feature_set))

LOOCV = lambda data, feature_set: loocv(data, feature_set, KNN(1))

def select_best(data, curr_set, search_set, objective_func):
    opt = None
    opt_val = 0
    temp_set = list(curr_set)
    for x in search_set:
        temp_set.append(x)
        temp_val = objective_func(data, temp_set)
        print('\tUsing feature(s)', temp_set, "accuracy is", temp_val)
        if opt is None or temp_val > opt_val:
            opt = x
            opt_val = temp_val
        temp_set.pop()
    return (opt, opt_val)

def select_worst(data, curr_set, search_set, objective_func):
    opt_elem = None
    opt_val = 0
    for x in search_set:
        temp_set = list(curr_set)
        temp_set.remove(x)
        temp_val = objective_func(data, temp_set)
        print('\tUsing feature(s)', temp_set, "accuracy is", temp_val)
        if opt_elem is None or temp_val > opt_val:
            opt_val = temp_val
            opt_elem = x
    return (opt_elem, opt_val)

def forward_select(data):
    nsample = len(data)
    nfeature = len(data[0]) - 1
    best_accuracy = 0
    best_set = []
    curr_set = []
    track = []
    for i in range(nfeature):
        search_set = []
        for j in range(nfeature):
            if j + 1 not in curr_set:
                search_set.append(j + 1)
        to_add = select_best(data, curr_set, search_set, LOOCV)
        curr_set.append(to_add[0])
        track.append((to_add[1], list(curr_set)))
        if to_add[1] > best_accuracy:
            best_accuracy = to_add[1]
            best_set = list(curr_set)
        print("\nFeature set", best_set, "was best, accuracy is", best_accuracy, "\n")
    print("Finished search!! The best feature subset is", best_set, ", which has an accuracy of", best_accuracy)
    track_acc = []
    for step in track:
        print(step)
        track_acc.append(step[0])
    print(track_acc)
            
def backward_select(data):
    nsample = len(data)
    nfeature = len(data[0]) - 1
    best_accuracy = 0
    best_set = []
    curr_set = []
    track = []
    for i in range(nfeature):
        curr_set.append(i + 1)
    for i in range(nfeature):
        to_sub = select_worst(data, curr_set, curr_set, LOOCV)
        curr_set.remove(to_sub[0])
        track.append((to_sub[1], list(curr_set)))
        if to_sub[1] > best_accuracy:
            best_accuracy = to_sub[1]
            best_set = list(curr_set)
        print("\nFeature set", best_set, "was best, accuracy is", best_accuracy, "\n")
    print("Finished search!! The best feature subset is", best_set, ", which has an accuracy of", best_accuracy)
    track_acc = []
    for step in track:
        print(step)
        track_acc.append(step[0])
    print(track_acc)

def forward_floating(data):
    curr_set = []
    curr_accuracy = 0
    best_set = []
    best_accuracy = 0
    nfeature = len(data[0]) - 1
    track = []
    for i in range(nfeature):
        search_set = []
        for j in range(nfeature):
            if j + 1 not in curr_set:
                search_set.append(j + 1)
        to_add = select_best(data, curr_set, search_set, LOOCV)
        curr_set.append(to_add[0])
        curr_accuracy = to_add[1]
        to_sub = select_worst(data, curr_set, curr_set, LOOCV)
        if to_sub[0] != to_add[0]:
            while to_sub[1] > curr_accuracy:
                curr_set.remove(to_sub[0])
                curr_accuracy = to_sub[1]
                to_sub = select_worst(data, curr_set, curr_set, LOOCV)
        track.append((curr_accuracy, list(curr_set)))
        if curr_accuracy > best_accuracy:
            best_accuracy = curr_accuracy
            best_set = list(curr_set)
        #print("\nCurrent feature set is", curr_set, ", accuracy is", curr_accuracy)
        print("\nFeature set", best_set, "was best, accuracy is", best_accuracy, "\n")
    print("Finished search!! The best feature subset is", best_set, ", which has an accuracy of", best_accuracy)
    track_acc = []
    for step in track:
        print(step)
        track_acc.append(step[0])
    print(track_acc)

def bidirection_select(data):
    nfeature = len(data[0]) - 1
    forward_set = []
    backward_set = []
    best_set = []
    best_accuracy = 0
    forward_search_set = []
    backward_search_set = []
    forward_track = []
    backward_track = []
    for i in range(nfeature):
        backward_set.append(i + 1)
    forward_search_set = list(backward_set)
    backward_search_set = list(backward_set)
    while forward_set != backward_set and (len(forward_search_set) > 0 or len(backward_search_set) > 0):
        to_add = select_best(data, forward_set, forward_search_set, LOOCV)
        forward_set.append(to_add[0])
        forward_search_set.remove(to_add[0])
        backward_search_set.remove(to_add[0])
        forward_track.append((to_add[1], list(forward_set)))
        if to_add[1] > best_accuracy:
            best_accuracy = to_add[1]
            best_set = list(forward_set)
        print("\nFeature set", best_set, "was best, accuracy is", best_accuracy, "\n")
        
        if forward_set == backward_set:
            break
        if len(backward_search_set) == 0:
            continue
        
        to_sub = select_worst(data, backward_set, backward_search_set, LOOCV)
        backward_set.remove(to_sub[0])
        backward_search_set.remove(to_sub[0])
        forward_search_set.remove(to_sub[0])
        backward_track.insert(0, (to_sub[1], list(backward_set)))
        if to_sub[1] > best_accuracy:
            best_accuracy = to_sub[1]
            best_set = list(backward_set)
        print("\nFeature set", best_set, "was best, accuracy is", best_accuracy, "\n")
    print("Finished search!! The best feature subset is", best_set, ", which has an accuracy of", best_accuracy)
    track = list(forward_track)
    track.extend(backward_track)
    track_acc = []
    for step in track:
        print(step)
        track_acc.append(step[0])
    print(track_acc)

def load_seed_dataset(data):
    label = []
    for i in range(len(data)):
        label.append(data[i][-1])
    data = preprocessing.normalize(data)
    for i in range(len(data)):
        data[i][-1] = label[i]
        
    lists = []
    for row in data:
        temp = [row[-1]]
        temp.extend(row[0:len(row)-1])
        lists.append(temp)
    return lists

def main():
    print("Welcome to My Feature Selection Algorithm.")
    file = input("Type the name of the file to test: ")
    data = np.genfromtxt(file)
    
##    label = []
##    for i in range(len(data)):
##        label.append(data[i][0])
##    data = preprocessing.normalize(data)
##    for i in range(len(data)):
##        data[i][0] = label[i]
        
    #lists = []
    #for row in data:
    #    lists.append(list(row[:]))
    lists = load_seed_dataset(data)
    
    alg = input('''
Type the number of the algorithm you want to run.\n
    1) Forward Selection
    2) Backward Selection
    3) Forward Floating Selection
    4) Bidirection Selection

                                  ''')
    print("\nThis dataset has", len(lists[0])-1, "features (not including the class attribute) with", len(lists), "instances\n")
    print("Beginning search.\n")
    
    if alg == '1':
        forward_select(lists)
    elif alg == '2':
        backward_select(lists)
    elif alg == '3':
        forward_floating(lists)
    elif alg == '4':
        bidirection_select(lists)

if __name__ == "__main__":
    main()
