import scipy.io as sio
import pandas as pd
import scipy
import numpy as np
import scipy.sparse
import scipy.stats
import random



# converts output or labels which contain the distribution per splicing sites
# to have just five splicing sites 1, 2, N1, N2 and crypt  for the sequence

# this function is for Alternate 5'splicing
# GCTTGGNNNNNNNNNNNNNNNNNNNNNNNNNGGTCGACCCAGGTTCGTGNNNNNNNNNNNNNNNNNNNNNNNNNGAGGTATTCTTATCACCTTCGTGGCT
# [SD1, SN1, SD2, SDN2, SDCrypt]
def convert_out_to_five_splicing_sites_for_5ss(splicingdist):
	cnt=0
	five_splicingdist=[]
	for x in splicingdist:
		s=0
		vec=[]
		for i in range(len(x)):
			# SD1
			if i==0:
				vec.append(x[i])

			# SD2, SDCrpt
			elif i==44 or i==79 :
				vec.append(s)
				vec.append(x[i])
				s=0
			# SDN1 and SDN2
			elif (i>6 and i<32) or (i>49 and i<75):
				s+=x[i]

		five_splicingdist.append(vec)

	return five_splicingdist

# this function is for Alternate 3' splicing
# gtaagttatcaccttcgtggctacagagtttccttatttgtctctgttgccggcttatatggacaagcatatcacagccatttatcggagcgcctccgtacacgctattatcggacgcctcgcgagatcaatacgtataccagctgccctcgatacatgtcttggacggggtcggtgttgatatcgtatNNNNNNNNNNNNNNNNNNNNNNNNNGCTTGGATCTGATCTCAACAGGGTNNNNNNNNNNNNNNNNNNNNNNNNNatgattacacatatagacacgcgagcacccatcttttatagaatgggtagaacccgtcctaaggactcagattgagcatcgtttgcttctcgagtactacctggtacagatgtctcttcaaacaggacggcagcgtgcagctcgccgaccactaccagcagaacacccccatcggcgacggccccgtgctgctgcccgacaaccactacctgagctaccagtccgccctgagcaaagaccccaacgagaagcgcgatcacatggtcctgctggagttcgtgaccgccgccgggatcactctcggcatggacgagctgtacaaggactgatagtaaggcccattacctgcNNNNNNNNNNNNNNNNNNNNGCAGAACACAGCGGTTCGACCTGCGTGATATCTCGTATGCCGTCTTCTGCTTG
# [SN1, SA1, SN2, SACrypt, SA2]
def convert_out_to_five_splicing_sites_for_3ss(splicingdist):
	cnt=0
	five_splicingdist=[]
	for x in splicingdist:
		s=0
		vec=[]
		for i in range(len(x)):
			# SA2
			if i==388:
				vec.append(x[i])

			# SACrypt, SA1
			elif i==372 or i==238 :
				vec.append(s)
				vec.append(x[i])
				s=0
			# SAN1 and SAN2
			elif (i>188 and i<214) or (i>238 and i<264):
				s+=x[i]

		five_splicingdist.append(vec)

	return five_splicingdist

# combines the features and labels andnormalizes the distribution at 5 splicing sites
# if the there is no splicing it is ignored
def combine_feat_lab_and_normalize_distr(a5seqs, five_splicingdist):
	all_data_feat_lab=[]
	for i in range(len(a5seqs)):
		z=[]
		z.append(a5seqs[i])
		s=0.0
		for x in five_splicingdist[i]:
			s+=x
		if s==0:
			continue
		for x in five_splicingdist[i]:
			z.append(x/s)
		all_data_feat_lab.append(z)
	return all_data_feat_lab

# splits the data which has features and labels and makes them separate
def convert_data_to_feat_lab(data):
	datafeat=[]
	datalab=[]
	for x in data:
		d=(list(x[0]))
		e=[]
		for c in d:
			if c=='A':
				e.append(0.0)
			elif c=='T':
				e.append(1.0)
			elif c=='G':
				e.append(2.0)
			elif c=='C':
				e.append(3.0)
		datafeat.append(e)
		z=[]
		z.append(x[1])
		z.append(x[2])
		z.append(x[3])
		z.append(x[4])
		z.append(x[5])
		datalab.append(z)
	return np.array(datafeat), np.array(datalab)


def dataz5ss():
	data=sio.loadmat('../data_gz/Reads.mat')
	a5d=data['A5SS']
	a5d=np.array(a5d.todense())
	a5n=np.array(a5d.sum(axis=1)>0).flatten()
	a5d=a5d[a5n]
	
	# a5d=a5d/a5d.sum(axis=1)[:, np.newaxis]

	five_splicingdist=convert_out_to_five_splicing_sites_for_5ss(a5d)
	a5seqs=pd.read_csv('../data_gz/A5SS_Seqs.csv', index_col=0).Seq[a5n]
	all_data_feat_lab=combine_feat_lab_and_normalize_distr(a5seqs, five_splicingdist)


	random.shuffle(all_data_feat_lab)
	train_set = all_data_feat_lab[:int(len(all_data_feat_lab)*0.9)]
	test_set = all_data_feat_lab[int(len(all_data_feat_lab)*0.9):]


	trainx, trainy=convert_data_to_feat_lab(train_set)
	testx, testy=convert_data_to_feat_lab(test_set)


	# trainx=[' '.join(i) for i in trainx]
	print(trainx[0])
	print(trainy[0])
	# print(testx[0])
	# print(testy[0])

	print("Returning Variables")
	return trainx, trainy, testx, testy


def dataz3ss():
	data=sio.loadmat('../data_gz/Reads.mat')
	a3d=data['A3SS']
	a3d=np.array(a3d.todense())

	splitlen=800000
	a3d=a3d[:splitlen,:]
	
	a3n=np.array(a3d.sum(axis=1)>0).flatten()
	a3d=a3d[a3n]
	
	# a3d=a3d/a3d.sum(axis=1)[:, np.newaxis]

	five_splicingdist=convert_out_to_five_splicing_sites_for_3ss(a3d)
	a3seqs=pd.read_csv('../data_gz/A3SS_Seqs.csv', index_col=0).Seq[:splitlen][a3n]
	all_data_feat_lab=combine_feat_lab_and_normalize_distr(a3seqs, five_splicingdist)


	random.shuffle(all_data_feat_lab)
	train_set = all_data_feat_lab[:int(len(all_data_feat_lab)*0.9)]
	test_set = all_data_feat_lab[int(len(all_data_feat_lab)*0.9):]


	trainx, trainy=convert_data_to_feat_lab(train_set)
	testx, testy=convert_data_to_feat_lab(test_set)


	# trainx=[' '.join(i) for i in trainx]
	print(trainx[0])
	print(trainy[0])
	# print(testx[0])
	# print(testy[0])

	print("Returning Variables")
	return trainx, trainy, testx, testy

# dataz3ss()