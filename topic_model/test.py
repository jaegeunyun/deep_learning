from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

spam_header = 'span\t'
no_spam_header = 'ham\t'
documents = []

with open('../sms_spam/SMSSpamCollection') as file_handle:
    for line in file_handle:
        if line.startswith(spam_header):
            documents.append(line[len(spam_header):])
        elif line.startswith(no_spam_header):
            documents.append(line[len(no_spam_header):])

vectorizer = CountVectorizer(stop_words='english', max_features=2000)
term_counts = vectorizer.fit_transform(documents)
vocabulary = vectorizer.get_feature_names()

topic_model = LatentDirichletAllocation(n_topics=10)
topic_model.fit(term_counts)

topics = topic_model.components_
for topic_id, weights in enumerate(topics):
    print('topic %d' % topic_id, end=': ')
    pairs = []
    for term_id, value in enumerate(weights):
        pairs.append((abs(value), vocabulary[term_id]))
    pairs.sort(key=lambda x: x[0], reverse=True)
    for pair in pairs[:10]:
        print(pair[1], end=',')
    print()
