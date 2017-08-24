from sklearn import svm
from sklearn.model_selection import GridSearchCV
import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer

import re
import os
path= os.getcwd()
vectorizer = CountVectorizer(min_df=1)

#traing the model with labeled data
def train():
	train_labels = []
	train_data = []
	for file in os.listdir(path+"/trainData"):
		text = open(path+"/trainData/"+file).read()
		for labeledque in text.split("\n"):
			if labeledque:
				que=re.split(" ,,, ",labeledque)
				train_data.append(que[0])
				train_labels.append(que[1])
	train_vectors=vectorizer.fit_transform(train_data)

	parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
	svr = svm.SVC()
	svm_grid_model = GridSearchCV(svr, parameters)
	svm_grid_model.fit(train_vectors,train_labels)
	modelname = 'svm_grid_model.pkl'
	pickle.dump(svm_grid_model, open(modelname, 'wb'))
	loaded_model = pickle.load(open(modelname, 'rb'))
	return loaded_model

#testing a new query using the generated model
def testing(loaded_model, query):
	test_data=[query]
	test_vector=vectorizer.transform(test_data)
	print loaded_model.predict(test_vector)
	

if __name__ == '__main__':
	loaded_model=train()
	query=raw_input("Write a query to test?:\n")
	while query != 'exit':
		testing(loaded_model,query)
		query=raw_input("Write a query to test?:\n")
