#Contains routines for data manipulation s that are different from other modules


def prob2Onehot(prob_dist):
	import numpy as np
	ret_vec = []
	for probs in prob_dist:
		probs_list = list(probs)
		temp = [0]*len(probs_list)
		temp[probs_list.index(max(probs_list))] = 1
		ret_vec.append(temp)
	return np.array(ret_vec)
