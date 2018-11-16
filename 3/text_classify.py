from sklearn import model_selection, preprocessing, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas
import numpy as np

# load the dataset
data = open('./data/train.dat').read()
labels, texts = [], []
for i, line in enumerate(data.split("\n")):
    if not line:
        continue
    content = line.split()
    labels.append(content[-1])
    texts.append(line.rsplit(' ', 1)[0])

# create a dataframe using texts and lables
trainDF = pandas.DataFrame()
trainDF['text'] = texts
trainDF['label'] = labels

# split the dataset into training and validation datasets
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])

# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(trainDF['text'])
xtrain_tfidf = tfidf_vect.transform(train_x)
xvalid_tfidf = tfidf_vect.transform(valid_x)

# fit the training dataset on the classifier
classifier = svm.LinearSVC()
classifier.fit(xtrain_tfidf, train_y)

# predict the labels on validation dataset
predictions = classifier.predict(xvalid_tfidf)

# training accuracy measure
accuracy = metrics.accuracy_score(predictions, valid_y)
print "The training accuracy:", accuracy

# load the test dataset
data = open('./data/test.dat').read()
test_texts = []
for i, line in enumerate(data.split("\n")):
    if line:
        test_texts.append(line)

testDF = pandas.DataFrame()
testDF['test_txt'] = test_texts

test_tfidf = tfidf_vect.transform(testDF['test_txt'])
predictions = classifier.predict(test_tfidf)

# output the real test result into result.txt
with open('result.txt', 'wb') as f:
    np.savetxt(f, predictions, fmt='%s')

data = open('./data/answer.txt').read()
answer = []
for i, line in enumerate(data.split("\n")):
    if line:
        answer.append(line)
answerDF = pandas.DataFrame()
answerDF['answer'] = answer
accuracy = metrics.accuracy_score(predictions, answerDF['answer'])
print "The real accuracy:", accuracy