#Includes functions to import word vectors

class Embeddings:

	def __init__(self, sent_len = 11):
		import gensim
		self.model = gensim.models.KeyedVectors.load_word2vec_format('../../../Resources/GoogleNews-vectors-negative300.bin', binary=True)
		self.doc_vectors = []
		self.embeddings = []
		#To keep the 0th element an empty vector [to map the padded elements]
		self.embeddings.append([0.]*len(list(self.model["hello"])))
		self.maxSize = 0
		self.POS_labels = []
		#To keep the 0th element an empty vector [to map the padded elemnts]
		self.POS_labels.append(["."])
		self.num_of_words = 1	#0 is reserved for unknown words
		self.word2vec_Map = {}
		self.sent_len = sent_len

	def GoogleVecs_POS_triggerVecs(self, corpus, trigger_list):

		from Data_Standardization import oneHot_to_standard

		from nltk.tokenize import RegexpTokenizer
		tokenizer = RegexpTokenizer(r'\w+')
		from nltk.corpus import stopwords
		stop_words = set(stopwords.words('english'))
		#stop_words = set([])
		from nltk.stem import WordNetLemmatizer
		wordnet_lemmatizer = WordNetLemmatizer()
		from nltk import pos_tag
		from difflib import get_close_matches

		trig_vectors = []
		not_in_vocab = 0
		trig_not_in_vocab = 0
		tot_padding = 0
		for doc_num in range(len(corpus)):
			trig = tokenizer.tokenize(trigger_list[doc_num])
			trig = [wordnet_lemmatizer.lemmatize(t) for t in trig]
			tokens = tokenizer.tokenize(corpus[doc_num])
			tokens = [wordnet_lemmatizer.lemmatize(word) for word in tokens if not word in stop_words]
			tags = pos_tag(tokens)
			tags = [x[1] for x in tags]

			doc_temp = []
			trig_temp = []
			for word in tokens:
				if word in self.model.vocab:
					doc_temp.append(self.num_of_words)
					self.embeddings.append(list(self.model[word]))
					self.num_of_words += 1
					if word in trig:
						trig_temp.append([1,0])
					else:
						trig_temp.append([0,1])
				else:
					not_in_vocab += 1
					#Not set to 0 in order to retain atleast the POS tags for the terms
					doc_temp.append(self.num_of_words)
					self.embeddings.append([0.]*len(list(self.model["hello"])))
					self.num_of_words += 1
					if word in trig:
						trig_not_in_vocab += 1
						trig_temp.append([1,0])
					else:
						trig_temp.append([0,1])

			#doc_temp, trig_temp, _ = oneHot_to_standard(doc_temp, trig_temp, tags)
			if len(doc_temp[0]) > self.maxSize:
				self.maxSize = len(doc_temp[0])
			self.doc_vectors.append(doc_temp)
			trig_vectors.append(trig_temp)
			self.POS_labels.append(tags)

		print("Load_Embedings :: GoogleVecs_POS_triggerVecs")
		print("Num of Docs : ", len(corpus))
		print("Total Doc Len : ", self.num_of_words)
		print("Words not found in embeddings : ", not_in_vocab)
		print("Triggers not found in embeddings : ", trig_not_in_vocab)
		print("Total Padding : ", tot_padding)
		print("Load_Embedings :: GoogleVecs_POS_triggerVecs")

		del self.model

		return (self.doc_vectors, self.embeddings, trig_vectors, self.maxSize, self.POS_labels)
