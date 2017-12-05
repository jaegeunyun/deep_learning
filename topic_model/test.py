from sklearn.docomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

spam_header = 'span\t'
no_spam_header = 'ham\t'
documents = []
