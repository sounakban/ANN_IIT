#Progrm to perform data transformation to directly feed into the neural network
#This prgram only creates models for trigger detection

def oneHot_to_standard(doc_vectors, trig_vectors, tags, sent_len=10, posit_window_len=5):

	if posit_window_len%2 == 0:
		raise ValueError("Please enter a  odd number as windows size, to ensure even distribution around target word.")
	doc_split, trig_split, tag_split, posit_vecs = split(doc_vectors, trig_vectors, tags, posit_window_len, sent_len)
	ret_doc = []
	ret_trig = []
	#print("Length : ", len(doc_split), len(trig_split), len(posit_vecs))
	if not len(trig_split) == len(doc_split):
		raise IndexError("Length of sentence splits for words and triger labels do not match!!!")
	for doc, trig, posit_vec in zip(doc_split, trig_split, posit_vecs):
		ret_doc.append(doc)
		if trig[posit_vec.index([-2, -1, 0, 1, 2])] == 1:
			ret_trig.append([1,0])
		else:
			ret_trig.append([0,1])

	#include_posit=True

	return (ret_doc, ret_trig, tag_split, posit_vecs)




def split(doc_vectors, trig_vectors, tags, posit_window_len, sent_len):

	doc_split = []
	trig_split = []
	tag_split = []
	posit_vecs = []
	"""
	while len(doc_vectors) < sent_len:
		doc_vectors.append(0)
		trig_vectors.append(0)
		tags.append("Empty")
		print("Entering padding in Data_Standardization :: split")
	"""
	doc_len = len(doc_vectors)
	for i in range(doc_len):
		#For padded vriables
		if doc_vectors[i] == 0:
			break
		#print("Doc len , i : ", doc_len, i)
		start = 0 if (i - (sent_len/2)) < 0 else int(i - (sent_len/2))
		end = start + sent_len
		#print("Start,  End : ", start, end)
		if end > doc_len:
			start = start - (end - doc_len)
			#end = doc_len - 1
		#print("Start,  End : ", start, end)
		doc_split.append(doc_vectors[start:end])
		trig_split.append(trig_vectors[start:end])
		tag_split.append(tags[start:end])
		relative_posit = i - start	#Get relative position of word based on the window
		#print("Target posit : ", i)
		#print("Relative posit : ", relative_posit)
		posit_vecs.append(create_position_vectors(sent_len, posit_window_len, relative_posit))
	return (doc_split, trig_split, tag_split, posit_vecs)


def create_position_vectors(sent_len, posit_window_len, target_posit):

	##Loop to calculate the vector, For example if 5 is the sentence length the loop will
	#run from -2 to 2
	vector_loop = list( range(-int(posit_window_len/2), int(posit_window_len/2)+1) )
	posit_vec = []
	for i in range(sent_len):
		distance = i - target_posit
		posit_vec.append([distance + j for j in vector_loop])
	#print(posit_vec)

	return posit_vec
