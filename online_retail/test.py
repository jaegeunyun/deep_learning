import time
from scipy import stats

user_product_dic = {}
product_user_dic = {}
product_id_name_dic = {}

#------------------- import ----------------------
for line in open('online_retail_utf_tab.csv'):
  line_items = line.strip().split('\t')
  #print (line_items)
  user_code = line_items[6]
  product_id = line_items[1]
  product_name = line_items[2]
  if len(user_code) == 0:
    continue
  country = line_items[7]
  if country != '"United Kingdom"':
    continue
  try:
    invoice_year = time.strptime(line_items[4], '%m/%d/%Y %H:%M').tm_year
  except ValueError:
    continue

  if invoice_year != 2011:
    continue
  #print (user_code, product_id, product_name, country, invoice_year)

  user_product_dic.setdefault(user_code, set())
  user_product_dic[user_code].add(product_id)
  product_user_dic.setdefault(product_id, set())
  product_user_dic[product_id].add(user_code)
  product_id_name_dic[product_id] = product_name


####------------------- plot ----------------------
###from collections import Counter
import matplotlib.pyplot as plt
###
###plot_data_all = Counter(product_per_user_li)
###plot_data_x = list(plot_data_all.keys())
###plot_data_y = list(plot_data_all.values())
###plt.xlabel('uniq king of product')
###plt.ylabel('# of user')
###plt.scatter(plot_data_x, plot_data_y, marker='x')
###plt.show()

#------------------- remove noise ----------------------
product_per_user_li = [len(x) for x in user_product_dic.values()]
print('# of    users:', len(user_product_dic))
print('# of products:', len(product_user_dic))
print(stats.describe(product_per_user_li))

min_product_user_li = [k for k,v in user_product_dic.items() if len(v)==1]
max_product_user_li = [k for k,v in user_product_dic.items() if len(v)>=600]
print("# of users purchased 1 product:%d" % (len(min_product_user_li)))
print("# of users purchased more than 600 product:%d" % (len(max_product_user_li)))

user_product_dic = {k:v for k,v in user_product_dic.items() if 1<len(v)<=600}
print("# of left user:%d" % (len(user_product_dic)))

id_product_dic = {}
for product_set_li in user_product_dic.values():
  for x in product_set_li:
    if x in id_product_dic:
      product_id = id_product_dic[x]
    else:
      id_product_dic.setdefault(x, len(id_product_dic))
print("# of left items:%d" % (len(id_product_dic)))


#------------------- one hot encoding ----------------------
id_user_dic = {}
user_product_vec_li = []
all_product_count = len(id_product_dic)
for user_code, product_per_user_set in user_product_dic.items():
  user_product_vec = [0] * all_product_count
  id_user_dic[len(id_user_dic)] = user_code
  for product_name in product_per_user_set:
    user_product_vec[id_product_dic[product_name]] = 1
  user_product_vec_li.append(user_product_vec)
print(id_user_dic[0])
print(user_product_dic[id_user_dic[0]])
#print(user_product_vec_li[0])
print(len(user_product_vec_li[0]))

#------------------- k-means clustering ----------------------
from sklearn.cluster import KMeans
import random
from random import shuffle

random.shuffle(user_product_vec_li)
train_data = user_product_vec_li[:2500]
test_data = user_product_vec_li[2500:]
print("#of train data:%d, #of test data:%d" % (len(train_data), len(test_data)))
km_predict = KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=20).fit(train_data)
km_predict_result = km_predict.predict(test_data)
print(km_predict_result)

####------------------- # of cluster ----------------------
###import numpy as np
###from sklearn.metrics import silhouette_score
###test_data = np.array(user_product_vec_li)
###for k in range(2,9):
###  km = KMeans(n_clusters=k).fit(test_data)
###  print("score for %d clusters:%.3f" % (k, silhouette_score(test_data, km.labels_)))

####------------------- within sum of squares ----------------------
###ssw_dic = {}
###for k in range(1,8):
###  km = KMeans(n_clusters=4).fit(test_data)
###  ssw_dic[k] = km.inertia_
###
###plot_data_x = list(ssw_dic.keys())
###plot_data_y = list(ssw_dic.values())
###plt.xlabel("# of clusters")
###plt.ylabel("within ss")
###plt.plot(plot_data_x, plot_data_y, linestyle="-", marker="o")
###plt.show()

#------------------- analyze_cluster_keywords ----------------------
from collections import Counter
def analyze_cluster_keywords(labels, product_id_name_dic, user_product_dic, id_user_dic):
  print(Counter(labels))
  cluster_item = {}
  for i in range(len(labels)):
    cluster_item.setdefault(labels[i], [])
    print('1:', labels)
    #print('2:', product_id_name_dic)
    #cluster_item[labels[i]].extend([product_id_name_dic[i]])
  for x in user_product_dic[id_user_dic[i]]:
    for cluster_id, product_name in cluster_item.items():
      bigram = []
      product_name_keyword = (' ').join(product_name).replace(' OF ', ' ').split()
      for i in range(0,len(product_name_keyword) -1):
        bigram.append(' '.join(product_name_keyword[i:i+2]))
      print('cluster_id:', cluster_id)
      print(Counter(bigram).most_common(20))

def analyze_clusters_product_count(labels, user_product_dic, id_user_dic):
  product_len_dic = {}
  for i in range(0, len(labels)):
    product_len_dic.setdefault(labels[i], [])
    product_len_dic[labels[i]].append(len(user_product_dic[id_user_dic[i]]))
  for k,v in product_len_dic.items():
    print('cluster:', k)
    print(stats.describe(v))

###km=KMeans(n_clusters=2, n_init=10, max_iter=20)
###km.fit(test_data)
####analyze_cluster_keywords(km.labels_, product_id_name_dic, user_product_dic, id_user_dic)
###analyze_clusters_product_count(km.labels_, user_product_dic, id_user_dic)

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram

row_clusters = linkage(test_data, method='complete', metric='euclidean')
tmp_label=[]
for i in range(len(id_user_dic)):
  tmp_label.append(id_user_dic[i])

row_denr = dendrogram(row_clusters, labels=tmp_label)
plt.tight_layout()
plt.ylabel('euclid')
plt.show()


