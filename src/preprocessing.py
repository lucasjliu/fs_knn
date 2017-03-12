from sklearn import preprocessing
from scipy import stats

def no_norm(data):
	return data

def zero_one_norm(data):
	for i in range(1, data.shape[1]):
		data[:,i] -= min(data[:,i])
	data[:,1:] = preprocessing.normalize(data[:,1:], norm='max', axis=0)
	return data

def z_norm(data):
	data[:,1:data.shape[1]] = stats.zscore(data[:,1:data.shape[1]], axis=0)
	return data