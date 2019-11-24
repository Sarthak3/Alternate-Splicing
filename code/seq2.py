import scipy.io as sio
import pandas as pd
import scipy
import numpy as np
import scipy.sparse
import scipy.stats
import random
import operator
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
import math

from preprocess import *

import pickle
with open('../results/embed', 'rb') as f:
	embed=pickle.load(f)
for i in range(len(embed)):
	print(np.linalg.norm(embed[i]))
	embed[i]=embed[i].tolist()
# embed=[[],[],[],[]]
# embed[0]=[1,0,0,0]
# embed[1]=[0,1,0,0]
# embed[2]=[0,0,1,0]
# embed[3]=[0,0,0,1]


def list_from_threshold(trainx, trainy, k):
	list_of_seq=[]
	for i in range(len(trainx)):
		if trainy[i][k]>0.8:
			list_of_seq.append(trainx[i])
	return list_of_seq

def list_to_vec(lst, list_of_seq, lenpat):
	for i in range(lenpat):
		for j in range(len(list_of_seq)):
			x=[]
			if list_of_seq[j][i]==0.0:
				x=embed[0]
			elif list_of_seq[j][i]==1.0:
				x=embed[1]
			elif list_of_seq[j][i]==2.0:
				x=embed[2]
			else: x=embed[3]
			# print(j)
			# print(x)
			# print(lst[j])
			lst[j].extend(x)
			# print(lst[j])
	# print(lst[0])
	return lst

def clus(lst, lenpat, list_of_seq):
	km=KMeans()
	clust=km.fit(lst)
	# print(km.cluster_centers_)
	ans=km.cluster_centers_
	for i in range(len(ans)):
		pat=""
		for j in range(lenpat):
			ind=len(embed)*j
			pat+="("
			# if ans[i][ind]>0.5:
			# 	pat+="A"
			# ind+=1
			# if ans[i][ind]>0.5:
			# 	pat+="T"
			# ind+=1
			# if ans[i][ind]>0.5:
			# 	pat+="G"
			# ind+=1
			# if ans[i][ind]>0.5:
			# 	pat+="C"
			if i==0  and j==0:
				# print(np.linalg.norm(ans[i][ind:ind+len(embed)]) - embed[0])
				# print(np.linalg.norm(ans[i][ind:ind+len(embed)]) - embed[1])
				# print(np.linalg.norm(ans[i][ind:ind+len(embed)]) - embed[2])
				# print(np.linalg.norm(ans[i][ind:ind+len(embed)]) - embed[3])
				# math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))
				print(euclidean_distances(embed[0].reshape(-1,1), ans[i][ind:ind+len(embed)].reshape(-1,1))[0][0])
				print(euclidean_distances(embed[1].reshape(-1,1), ans[i][ind:ind+len(embed)].reshape(-1,1))[0][0])
				# print(math.sqrt(sum([(a - b) ** 2 for a, b in zip(embed[1], ans[i][ind:ind+len(embed)])])))
				print(euclidean_distances(embed[2].reshape(-1,1), ans[i][ind:ind+len(embed)].reshape(-1,1))[0][0])
				print(euclidean_distances(embed[3].reshape(-1,1), ans[i][ind:ind+len(embed)].reshape(-1,1))[0][0])

			if cosine_similarity(embed[0].reshape(-1,1), ans[i][ind:ind+len(embed)].reshape(-1,1))[0][0] > 0.85:
				pat+="A"
			if cosine_similarity(embed[1].reshape(-1,1), ans[i][ind:ind+len(embed)].reshape(-1,1))[0][0] > 0.85:
				pat+="T"
			if cosine_similarity(embed[2].reshape(-1,1), ans[i][ind:ind+len(embed)].reshape(-1,1))[0][0] > 0.85:
				pat+="G"
			if cosine_similarity(embed[3].reshape(-1,1), ans[i][ind:ind+len(embed)].reshape(-1,1))[0][0] > 0.85:
				pat+="C"
			pat+=")"
		print(pat)
	print()


	i=lenpat
	while(i<len(list_of_seq[0])):
		for j in range(len(lst)):
			lst[j]=lst[j][len(embed):]
			x=[]
			if list_of_seq[j][i]==0.0:
				x=embed[0]
			elif list_of_seq[j][i]==1.0:
				x=embed[1]
			elif list_of_seq[j][i]==2.0:
				x=embed[2]
			else: x=embed[3]
			lst[j].extend(x)

		km=KMeans()
		clust=km.fit(lst)
		# print(km.cluster_centers_)
		ans=km.cluster_centers_
		for a in range(len(ans)):
			pat=""
			for b in range(lenpat):
				ind=len(embed)*b
				pat+="("
				# if ans[a][ind]>0.5:
				# 	pat+="A"
				# ind+=1
				# if ans[a][ind]>0.5:
				# 	pat+="T"
				# ind+=1
				# if ans[a][ind]>0.5:
				# 	pat+="G"
				# ind+=1
				# if ans[a][ind]>0.5:
				# 	pat+="C"
				if (np.array_equal(embed[0], embed[1])):
					print("fuuuuck")
				if cosine_similarity(embed[0].reshape(-1,1), ans[a][ind:ind+len(embed)].reshape(-1,1))[0][0] > 0.85:
					pat+="A"
				if cosine_similarity(embed[1].reshape(-1,1), ans[a][ind:ind+len(embed)].reshape(-1,1))[0][0] > 0.85:
					pat+="T"
				if cosine_similarity(embed[2].reshape(-1,1), ans[a][ind:ind+len(embed)].reshape(-1,1))[0][0] > 0.85:
					pat+="G"
				if cosine_similarity(embed[3].reshape(-1,1), ans[a][ind:ind+len(embed)].reshape(-1,1))[0][0] > 0.85:
					pat+="C"
				pat+=")"
			print(pat)
		print()

		i+=1

def cluspatpos(trainx, trainy, k, lenpat):
	print("thresh to list")
	list_of_seq=list_from_threshold(trainx, trainy, k)
	
	lst=[]
	for _ in list_of_seq:
		lst.append([])

	print("to vec")
	lst=list_to_vec(lst, list_of_seq, lenpat)

	print("clustering")
	clus(lst, lenpat, list_of_seq)

def dataz():
	data=sio.loadmat('../data_gz/Reads.mat')
	a5d=data['A5SS']
	# a5d=data['A3SS']
	a5d=np.array(a5d.todense())
	# a5d=a5d[:999940,:]
	a5n=np.array(a5d.sum(axis=1)>0).flatten()	
	a5d=a5d[a5n]
	
	five_splicingdist=convert_out_to_five_splicing_sites_for_5ss(a5d)
	a5seqs=pd.read_csv('../data_gz/A5SS_Seqs.csv', index_col=0).Seq[a5n]
	all_data_feat_lab=combine_feat_lab_and_normalize_distr(a5seqs, five_splicingdist)

	random.shuffle(all_data_feat_lab)
	train_set = all_data_feat_lab[:int(len(all_data_feat_lab))]
	trainx, trainy=convert_data_to_feat_lab(train_set)

	lenpats=[5, 10, 15, 20]
	
	for i in range(5):
		print(i)
		print()
		for lenpat in lenpats:
			print(lenpat)
			print()
			cluspatpos(trainx, trainy, i, lenpat)

	print()
	print()
	print()
	print()
	print()
	print()
	print()

	data=sio.loadmat('../data_gz/Reads.mat')
	a5d=data['A3SS']
	a5d=np.array(a5d.todense())
	a5d=a5d[:999940,:]
	a5n=np.array(a5d.sum(axis=1)>0).flatten()	
	a5d=a5d[a5n]
	
	five_splicingdist=convert_out_to_five_splicing_sites_for_3ss(a5d)
	a5seqs=pd.read_csv('../data_gz/A3SS_Seqs.csv', index_col=0).Seq[:999940][a5n]
	all_data_feat_lab=combine_feat_lab_and_normalize_distr(a5seqs, five_splicingdist)

	random.shuffle(all_data_feat_lab)
	train_set = all_data_feat_lab[:int(len(all_data_feat_lab))]
	trainx, trainy=convert_data_to_feat_lab(train_set)

	for i in range(5):
		print(i)
		print()
		for lenpat in lenpats:
			print(lenpat)
			print()
			cluspatpos(trainx, trainy, i, lenpat)

dataz()


# TGCTTGGTGAAGACAGAGAAAGAGAACCAAAAGGTCGACCCAGGTTCGTGAATCCGGTAACGCGGAGAGAATACAGAGGTATTCTTATCACCTTCGTGGCT




















# patterns like ATGGTCGATAT..... - [0.2, 0.3, 0.3, 0.2]
# ATGG - [ , , , ] TGGT - [ , , , ]