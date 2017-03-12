import numpy as np

LOG_LEVEL = 0

def select_best(data, curr_set, search_set, objective_func):
	opt = None
	opt_val = 0
	temp_set = list(curr_set)
	for x in search_set:
		temp_set.append(x)
		temp_set.insert(0, 0)
		temp_val = objective_func(data[:, temp_set])
		temp_set = temp_set[1:]
		if LOG_LEVEL >= 2:
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
		temp_set.insert(0, 0)
		temp_val = objective_func(data[:, temp_set])
		temp_set = temp_set[1:]
		if LOG_LEVEL >= 2:
			print('\tUsing feature(s)', temp_set, "accuracy is", temp_val)
		if opt_elem is None or temp_val > opt_val:
			opt_val = temp_val
			opt_elem = x
	return (opt_elem, opt_val)

def print_track(track):	
	track_acc = []
	for step in track:
		print('\t', end='')
		print(step)
		track_acc.append(step[0])
	print('\t', end='')
	print(track_acc)

def forward_select(data, objective_func):
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
		to_add = select_best(data, curr_set, search_set, objective_func)
		curr_set.append(to_add[0])
		track.append((to_add[1], list(curr_set)))
		if to_add[1] > best_accuracy:
			best_accuracy = to_add[1]
			best_set = list(curr_set)
	if LOG_LEVEL >= 1:
		print_track(track)
	return best_set
			
def backward_select(data, objective_func):
	nsample = len(data)
	nfeature = len(data[0]) - 1
	best_accuracy = 0
	best_set = []
	curr_set = []
	track = []
	for i in range(nfeature):
		curr_set.append(i + 1)
	for i in range(nfeature):
		to_sub = select_worst(data, curr_set, curr_set, objective_func)
		curr_set.remove(to_sub[0])
		track.append((to_sub[1], list(curr_set)))
		if to_sub[1] > best_accuracy:
			best_accuracy = to_sub[1]
			best_set = list(curr_set)
	if LOG_LEVEL >= 1:
		print_track(track)
	return best_set

def forward_floating(data, objective_func):
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
		to_add = select_best(data, curr_set, search_set, objective_func)
		curr_set.append(to_add[0])
		curr_accuracy = to_add[1]
		to_sub = select_worst(data, curr_set, curr_set, objective_func)
		if to_sub[0] != to_add[0]:
			while to_sub[1] > curr_accuracy:
				curr_set.remove(to_sub[0])
				curr_accuracy = to_sub[1]
				to_sub = select_worst(data, curr_set, curr_set, objective_func)
		track.append((curr_accuracy, list(curr_set)))
		if curr_accuracy > best_accuracy:
			best_accuracy = curr_accuracy
			best_set = list(curr_set)
	if LOG_LEVEL >= 1:
		print_track(track)
	return best_set

def bidirection_select(data, objective_func):
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
		to_add = select_best(data, forward_set, forward_search_set, objective_func)
		forward_set.append(to_add[0])
		forward_search_set.remove(to_add[0])
		backward_search_set.remove(to_add[0])
		forward_track.append((to_add[1], list(forward_set)))
		if to_add[1] > best_accuracy:
			best_accuracy = to_add[1]
			best_set = list(forward_set)
		
		if forward_set == backward_set:
			break
		if len(backward_search_set) == 0:
			continue
		
		to_sub = select_worst(data, backward_set, backward_search_set, objective_func)
		backward_set.remove(to_sub[0])
		backward_search_set.remove(to_sub[0])
		forward_search_set.remove(to_sub[0])
		backward_track.insert(0, (to_sub[1], list(backward_set)))
		if to_sub[1] > best_accuracy:
			best_accuracy = to_sub[1]
			best_set = list(backward_set)
	if LOG_LEVEL >= 1:
		track = list(forward_track)
		track.extend(backward_track)
		print_track(track)
	return best_set

def main():
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
