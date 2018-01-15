#Progrm to create a python format data model to directly feed into the neural network
#This prgram only creates odels for trigger detection

class processed_data:

	def __init__(self, sourceDir=""):
		#For reading
		import csv
		import os
		sourceDir = "../../../Resources/ACE Corpus/fp2_apf_extracted/"

		self.docs = []
		self.triggers = []
		self.evType = []

		for path, subdirs, files in os.walk(sourceDir):
			for name in files:
				if name.endswith(".csv"):
					with open(os.path.join(path, name)) as inFile:
						content = csv.reader(inFile, delimiter = '\t')
						for row in content:
							if len(row[1].split(' ')) > 1:
								continue
							docs.append(row[0])
							triggers.append(row[1])
							evType.append(row[2].split('|')[0])

		print("Reading Docs complete")


	def get_Data_Embeddings(self):
		from Load_Embedings import Embeddings
		model_generator = Embeddings()

		doc_vec, embeddings, trig_vec, maxLen, POS_labels, _ = model_generator.GoogleVecs_POS_triggerVecs(self.docs, self.triggers)
		del model_generator

		print("Document & Trigger modelling complete")
		return (doc_vec, embeddings, trig_vec, maxLen, POS_labels)


def tagMatrix2Embeddings(tag_Matrix, embd_size=50):
	if not type(tag_Matrix) == type([]):
		raise ValueError("\'tag_Matrix\' should be of type, \'list of lists\'")

	import itertools
	all_tags = list(set(itertools.chain.from_iterable(tag_Matrix)))

	tag_vectors = []
	for doc in tag_Matrix:
		tag_temp = []
		for tag in doc:
			tag_temp.append(all_tags.index(tag))
		tag_vectors.append(tag_temp)

	import numpy as np
	tag_emdeddings = np.random.uniform(size = (len(all_tags), embd_size) )

	return (tag_vectors, tag_emdeddings, all_tags)







######################################## Deprecated Functions ########################################

def labelMatrix2OneHot(label_Matrix):
if not type(label_Matrix) == type([]):
	raise ValueError("\'label_Matrix\' should be of type, \'list of lists\'")

import itertools
all_labels = list(set(itertools.chain.from_iterable(label_Matrix)))

label2Vactor_Map = {}
size = len(all_labels)
for i in range(size):
	temp = [0]*size
	temp[i] = 1
	label2Vactor_Map[all_labels[i]] = temp

vector_Matrix = []
for i in range(len(label_Matrix)):
	temp = []
	for j in range(len(label_Matrix[i])):
		temp.append(label2Vactor_Map[label_Matrix[i][j]])
	vector_Matrix.append(temp)

return (vector_Matrix, label2Vactor_Map)


def pad_sequences_3D(POS_vectors, maxlen, value):
	for i in range(len(POS_vectors)):
		while len(POS_vectors[i]) < maxlen:
			POS_vectors[i].append(value)
	return POS_vectors


def concat_3Dvectors(vecA, vecB):
	if (not len(vecA) == len(vecB)) or (not len(vecA[0]) == len(vecB[0])):
		raise ValueError("The dimentions of inut vectors donot match in the fist two axes")

	for i in range(len(vecA)):
		for j in range(len(vecA[i])):
			vecA[i][j].extend(vecB[i][j])
	return vecA


def concat_2Dvectors(vecA, vecB):
	print(len(vecA), len(vecB))
	if not len(vecA) == len(vecB):
		raise ValueError("The dimentions of inut vectors donot match")

	for i in range(len(vecA)):
		vecA[i].extend(vecB[i])
	return vecA


def Flatten_3Dto2D(vec):
	ret_vec = []
	for i in range(len(vec)):
		ret_vec.extend(vec[i])
	return ret_vec
