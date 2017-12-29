#Progrm to perform data transformation to directly feed into the neural network
#This prgram only creates models for trigger detection

def oneHot_to_standard(doc_vectors, trig_vectors, tags, sent_len=10, posit_window_len=5, append_posit_vec=True):

	if posit_window_len%2 == 0:
		raise ValueError("Please enter a  odd number as windows size, to ensure even distribution around target word.")
	doc_split, trig_split, tag_split = split(doc_vectors, trig_vectors, tags, sent_len)
	ret_doc = []
	ret_trig = []
	if not len(trig_split) == len(doc_split):
		raise IndexError("Length of sentence splits for words and triger labels do not match!!!")
	for doc, trig, indx in zip(doc_split, trig_split, range(len(doc_split))):
		if indx > sent_len:
			indx = indx - (sent_len/2)	#Get relative position
		if append_posit_vec==True:
			posit_vec = create_position_vectors(sent_len, posit_window_len, indx)
			ret_doc.append(doc.extend(posit_vec))
		else:
			ret_doc.append(doc)
		if trig[indx] == 1:
			ret_trig.append([1,0])
		else:
			ret_trig.append([0,1])

	#include_posit=True

	return (ret_doc, ret_trig, tag_split)




def split(doc_vectors, trig_vectors, tags, sent_len):

	if len(doc_vectors) > sent_len:
		candidate_indices = []
		doc_split = []
		trig_split = []
		tag_split = []
		for i in range(len(doc_vectors)):
			start = 0 if (i - (sent_len/2)) < 0 else int(i - (sent_len/2))
			end = start + sent_len
			#print(start, end)
			doc_split.append(doc_vectors[start:end])
			trig_split.append(trig_vectors[start:end])
			tag_split.append(tags[start:end])
			candidate_indices.append(i if start==0 else i-start)
		return (doc_split, trig_split, tag_split)
	else:
		return ([doc_vectors], [trig_vectors], [tag_split])


def create_position_vectors(sent_len, posit_window_len, target_posit):

	posit_vec = []
	for i in range(sent_len):
		posit_vec.append([(i-target_posit)-(j-(posit_window_len/2)) for j in reversed(range(posit_window_len))])

	return posit_vec
