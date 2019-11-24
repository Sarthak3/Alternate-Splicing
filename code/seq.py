import scipy.io as sio
import pandas as pd
import scipy
import numpy as np
import scipy.sparse
import scipy.stats
import random
import operator
from preprocess import *



# splits the data which has features and labels and makes them separate
def convert_data_to_feat_lab(data):
	datafeat=[]
	datalab=[]
	for x in data:
		d=(list(x[0]))
		datafeat.append(d)
		z=[]
		z.append(x[1])
		z.append(x[2])
		z.append(x[3])
		z.append(x[4])
		z.append(x[5])
		datalab.append(z)
	return np.array(datafeat), np.array(datalab)

def findpat(trainx, trainy, lenpat, k):
	cnt=0
	dic={}
	for i in range(len(trainx)):
		if trainy[i][k]>0.8:
			cnt+=1
			pat=""
			stri=trainx[i]
			for j in range(lenpat):
				pat+=stri[j]
			if pat in dic: dic[pat]+=1
			else: dic[pat]=1 
			j=lenpat
			while(j<len(stri)):
				pat=pat[1:]
				pat+=stri[j]
				if pat in dic: dic[pat]+=1
				else: dic[pat]=1 
				j+=1
	dic=sorted(dic.items(), key=operator.itemgetter(1), reverse=True)
	print(dic[:10])
	print(len(dic), cnt)
	print()
	print()

def dataz():
	data=sio.loadmat('../data_gz/Reads.mat')
	# a5d=data['A5SS']
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

	
	lenpats=[5, 10, 15, 20]
	
	for lenpat in lenpats:
		print(lenpat)
		findpat(trainx, trainy, lenpat, 0)
		findpat(trainx, trainy, lenpat, 1)
		findpat(trainx, trainy, lenpat, 2)
		findpat(trainx, trainy, lenpat, 3)
		findpat(trainx, trainy, lenpat, 4)

	print()
	print()
	print()
	print()

	a5d=data['A5SS']
	a5d=np.array(a5d.todense())
	a5n=np.array(a5d.sum(axis=1)>0).flatten()	
	a5d=a5d[a5n]
	
	five_splicingdist=convert_out_to_five_splicing_sites_for_5ss(a5d)
	a5seqs=pd.read_csv('../data_gz/A5SS_Seqs.csv', index_col=0).Seq[a5n]
	all_data_feat_lab=combine_feat_lab_and_normalize_distr(a5seqs, five_splicingdist)

	random.shuffle(all_data_feat_lab)
	train_set = all_data_feat_lab[:int(len(all_data_feat_lab))]
	trainx, trainy=convert_data_to_feat_lab(train_set)

	
	for lenpat in lenpats:
		print(lenpat)
		findpat(trainx, trainy, lenpat, 0)
		findpat(trainx, trainy, lenpat, 1)
		findpat(trainx, trainy, lenpat, 2)
		findpat(trainx, trainy, lenpat, 3)
		findpat(trainx, trainy, lenpat, 4)

	print()
	print()
	print()
	print()
	
	
dataz()


# TGCTTGGTGAAGACAGAGAAAGAGAACCAAAAGGTCGACCCAGGTTCGTGAATCCGGTAACGCGGAGAGAATACAGAGGTATTCTTATCACCTTCGTGGCT




















# patterns like ATGGTCGATAT..... - [0.2, 0.3, 0.3, 0.2]
# ATGG - [ , , , ] TGGT - [ , , , ]