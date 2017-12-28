class Vector_Generator:

	def __init__(self):
		self.word2index_dict = {}
		self.word_list = []
		self.doc_vectors = []
		self.num_of_words = 0
		self.maxSize = 0
		self.POS_labels = []


	def create_lexicon(self, corpus):
		#from nltk.tokenize import word_tokenize
		from nltk.tokenize import RegexpTokenizer
		tokenizer = RegexpTokenizer(r'\w+')
		from nltk.corpus import stopwords
		stop_words = set(stopwords.words('english'))
		#stop_words = set([])
		from nltk.stem import WordNetLemmatizer
		wordnet_lemmatizer = WordNetLemmatizer()
		from nltk import pos_tag

		for doc in corpus:
			temp = []
			#tokens = word_tokenize(doc)
			tokens = tokenizer.tokenize(doc)
			tags = pos_tag(tokens)
			self.POS_labels.append([x[1] for x in tags])
			#[Figure out better solution!] because otherwise binarization will be maunal in 3D
			for word in tokens:
				if not word in stop_words:
					word = wordnet_lemmatizer.lemmatize(word)
					if not word in self.word_list:
						self.word2index_dict[word] = self.num_of_words
						self.word_list.append(word)
						self.num_of_words = self.num_of_words + 1
					temp.append(self.word2index_dict[word])
			if len(temp) > self.maxSize:
				self.maxSize = len(temp)
			self.doc_vectors.append(temp)

		return (self.doc_vectors, self.maxSize, self.num_of_words, self.POS_labels)


	def get_triggerVec(self, trigger_list):

		if self.maxSize == 0:
			raise RuntimeError("Make sure you run 'create_lexicon' function before calling 'get_triggerVec'")

		from nltk.tokenize import RegexpTokenizer
		tokenizer = RegexpTokenizer(r'\w+')
		from nltk.stem import WordNetLemmatizer
		wordnet_lemmatizer = WordNetLemmatizer()
		from nltk.corpus import stopwords
		stop_words = set(stopwords.words('english'))

		#Returns a binary vector for the position of the trigger in the sentence
		trig_vectors = []
		for doc_num in range(len(trigger_list)):
			temp = [0]*len(self.doc_vectors[doc_num])

			#Creates a trigger-vector
			trig_ind = []
			#trig = trigger_list[doc_num].split(" ")
			trig = tokenizer.tokenize(trigger_list[doc_num])
			for t in trig:
				if len(t) > 1 and not t in stop_words:
					trig_ind.append(self.word2index_dict[wordnet_lemmatizer.lemmatize(t)])
				"""
				if wordnet_lemmatizer.lemmatize(t) == 'w':
					print("result : " + trigger_list[doc_num])
				"""

			#returns a list of tuples (fisrt_index, last_index) where the sequence of trigger words occur in the document.
			indx_range = [(pos, pos+len(trig_ind)) for pos in range(len(self.doc_vectors[doc_num])) if self.doc_vectors[doc_num][pos:pos+len(trig_ind)] == trig_ind]

			if len(indx_range) == 1:
				if indx_range[0][0]+1 == indx_range[0][1]:
					temp[indx_range[0][0]] = 1
				else:
					for i in range(indx_range[0][0], indx_range[0][1]):
						temp[i] = 1
			#Needs to e handled better [This is a temporary fix where, if a trigger is found in multiple locations, all will be marked]
			elif len(indx_range) > 1:
				for j in range(len(indx_range)):
					if indx_range[j][0]+1 == indx_range[j][1]:
						temp[indx_range[j][0]] = 1
					else:
						for i in range(indx_range[j][0], indx_range[j][1]):
							temp[i] = 1
			else:
				#raise value error(trigger word not found in span, make sure sequence of both lists match)
				pass

			trig_vectors.append(temp)

		return trig_vectors
