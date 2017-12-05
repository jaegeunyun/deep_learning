###vocabulary = {}
###with open('SMSSpamCollection') as file_handle:
###  for line in file_handle:
###    splits = line.split()
###    label = splits[0]
###    text = splits[1:]
###    for word in text:
###      lower = word.lower()
###      if not lower in vocabulary:
###        vocabulary[lower] = len(vocabulary)
###print('# of voca:', len(vocabulary))
###
###import numpy as np
###features = []
###with open('SMSSpamCollection') as file_handle:
###  for line in file_handle:
###    feature = np.zeros(len(vocabulary))
###    splits = line.split()
###    text = splits[1:]
###    for word in text:
###      lower = word.lower()
###      feature[vocabulary[lower]] += 1
###    feature = feature / sum(feature)
###    features.append(feature)
####print(features)
###
###
###with open('SMSSpamCollection') as file_handle:
###  for line in file_handle:
###    splits = line.split()
###    label = splits[0]
###    if label == 'spam':
###      labels.append(1)
###    else:
###      labels.append(0)
###

#-------------- save pickle ---------------------------------------
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

spam_header = 'spam\t'
no_spam_header = 'ham\t'
documents = []
labels = []

with open('SMSSpamCollection') as file_handle:
  for line in file_handle:
    if line.startswith(spam_header):
      labels.append(1)
      documents.append(line[len(spam_header):])
    elif line.startswith(no_spam_header):
      labels.append(0)
      documents.append(line[len(no_spam_header):])
vectorizer = CountVectorizer()
term_counts = vectorizer.fit_transform(documents)
vocabulary = vectorizer.get_feature_names()

tf_transformer = TfidfTransformer(use_idf=False).fit(term_counts)
features = tf_transformer.transform(term_counts)

with open('processed.pickle', 'wb') as file_handle:
  pickle.dump((vocabulary, features, labels), file_handle)

#-------------- use pickle ---------------------------------------
import pickle
from sklearn.linear_model import LogisticRegression
with open('processed.pickle', 'rb') as file_handle:
  vocabulary, features, labels = pickle.load(file_handle)
total_number = len(labels)
middle_index = total_number//2
train_features = features[:middle_index,:]
train_labels = labels[:middle_index]
test_features = features[middle_index:,:]
test_labels = labels[middle_index:]

classifier = LogisticRegression()
classifier.fit(train_features, train_labels)
print('train accuracy: %4.4f' % classifier.score(train_features, train_labels))
print('test accuracy: %4.4f' % classifier.score(test_features, test_labels))

#-------------- review ---------------------------------------
weights = classifier.coef_[0,:]
pairs = []
for index, value in enumerate(weights):
    pairs.append((abs(value), vocabulary[index]))
pairs.sort(key=lambda x: x[0], reverse=True)
for pair in pairs[:20]:
    print('score %4.4f word: %s' %pair)


