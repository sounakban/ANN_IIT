#Progrm to create a python format data model to directly feed into the neural network
#This prgram only creates odels for trigger detection

class processed_data:

	def read_data(self):
		#For reading
		import csv
		import os
		sourceDir = "/home/sounak/Resources/Data/ACE Dataset/ace_2005_td_v7/data/English/nw/fp2_apf_extracted/"

		docs = []
		triggers = []
		evType = []
		for path, subdirs, files in os.walk(sourceDir):
			for name in files:
				if name.endswith(".csv"):
					with open(os.path.join(path, name)) as inFile:
						#print('###############################################################################################')
						content = csv.reader(inFile, delimiter = '\t')
						for row in content:
							if len(row[1].split(' ')) > 1:
								continue
								"""
								print(name)
								print("1 : "+row[0])
								print("2 : "+row[1])
								print("3 : "+row[2])
								"""
							docs.append(row[0])
							triggers.append(row[1])
							evType.append(row[2].split('|')[0])

		print("Reading Docs complete")
		return (docs, triggers, evType)


	def get_Data_Vectors(self):
		docs, triggers, _ = self.read_data()
		from Vector_Generator import Vector_Generator
		#from Vector_Generator_concat import Vector_Generator
		model_generator = Vector_Generator()

		doc_vec, maxLen, vocabSize, POS_labels = model_generator.create_lexicon(docs)
		trig_vec = model_generator.get_triggerVec(triggers)

		"""
		for trig in range(len(triggers)):
		if 'sickened w' == triggers[trig]:
		print(str(trig)+" : "+triggers[trigor word in tokens:])
		print(str(trig)+" : "+docs[trig])
		"""

		"""
		print('###############################################################################################')
		print(docs[10])
		print(triggers[10])
		print(new[10])
		print(new2[10])
		print(len(new[10]))
		print(len(new2[10]))
		print('###############################################################################################')
		print(docs[16])
		print(triggers[16])
		print(new[16])
		print(new2[16])
		print(len(new[16]))
		print(len(new2[16]))
		print('###############################################################################################')
		print(len(docs))
		print(len(triggers))
		print(len(evType))
		print('###############################################################################################')
		"""

		print("Document & Trigger modelling complete")
		return (doc_vec, trig_vec, maxLen, vocabSize, POS_labels)


	def get_Data_Embeddings(self):
		docs, triggers, _ = self.read_data()
		from Load_Embedings import Embeddings
		model_generator = Embeddings()

		doc_vec, embeddings, trig_vec, maxLen, POS_labels = model_generator.GoogleVecs_POS_triggerVecs(docs, triggers)
		#trig_vec = model_generator.get_triggerVec(triggers)
		del model_generator

		print("Document & Trigger modelling complete")
		return (doc_vec, embeddings, trig_vec, maxLen, POS_labels)


	def getSplit_DocTrigger(self):
		doc_vec, trig_vec, maxLen, vocabSize = self.getVectors_DocTrigger()
		#Finish program, perform train-test split


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
		for j in range(len(POS_vectors[i]), maxlen):
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



#obj = processed_data()
#obj.getVectors_doc_trigger()
