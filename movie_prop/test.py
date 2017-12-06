#-------------- import file and make metrix ----------------------------
import codecs
def read_data(fin, delim):
    info_li = []
    for line in codecs.open(fin, "r", encoding="latin-1"):
        line_items = line.strip().split(delim)
        key = int(line_items[0])
        if(len(info_li) + 1) != key:
            print('errors at data_id')
            exit(0)
        info_li.append(line_items[1:])
    print('rows in %s: %d' % (fin, len(info_li)))
    return(info_li)

fin_user = './ml-100k/u.user'
fin_movie = './ml-100k/u.item'
fin_data = './ml-100k/u.data'
user_info_dic = read_data(fin_user, '|')
movie_info_dic = read_data(fin_movie, '|')

import numpy as np
R = np.zeros((len(user_info_dic), len(movie_info_dic)), dtype=np.float64)
for line in codecs.open(fin_data, 'r', encoding='latin-1'):
    user, movie, rating, date = line.strip().split('\t')
    user_index = int(user) -1
    movie_index = int(movie) -1
    R[user_index, movie_index] = float(rating)

print(R)
print(R[0,10])
print(R.shape[0])

#-------------- stat ----------------------------
from scipy import stats

user_mean_li=[]
for i in range(0, R.shape[0]):
    user_rating = [x for x in R[i] if x>0.0]
    #print(stats.describe(user_rating).mean)
    user_mean_li.append(stats.describe(user_rating).mean)
print(stats.describe(user_mean_li))

movie_mean_li=[]
for i in range(0,R.shape[1]):
    R_T = R.T
    movie_rating = [x for x in R_T[i] if x>0.0]
    movie_mean_li.append(stats.describe(movie_rating).mean)
print(stats.describe(movie_mean_li))

import requests
import json
response = requests.get('http://us.imdb.com/M/title-exact?Toy%20Story%20%281995%29')
print('imdb url: %s' % (response.url))
imdb_id = response.url.split('/')[-2]
movie_plot_response = requests.get('http://www.omdbapi.com/?i='+imdb_id+'&plot=full&r=json')
print([x for x in movie_plot_response])
