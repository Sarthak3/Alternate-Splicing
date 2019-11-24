import scipy.io as sio
import pandas as pd
import scipy
import numpy as np
import scipy.sparse
import scipy.stats
import random

import seaborn
import matplotlib.pyplot as plt
plt.style.use('seaborn-ticks')
from matplotlib import transforms
import matplotlib.patheffects
import numpy as np

COLOR_SCHEME = {'G': 'orange', 
                'A': 'red', 
                'C': 'blue', 
                'T': 'darkgreen'}
BASES = list(COLOR_SCHEME.keys())

class Scale(matplotlib.patheffects.RendererBase):
    def __init__(self, sx, sy=None):
        self._sx = sx
        self._sy = sy

    def draw_path(self, renderer, gc, tpath, affine, rgbFace):
        affine = affine.identity().scale(self._sx, self._sy)+affine
        renderer.draw_path(gc, tpath, affine, rgbFace)

def draw_logo(all_scores):
    fig = plt.figure()
    fig.set_size_inches(len(all_scores),2.5)
    ax = fig.add_subplot(111)
    ax.set_xticks(range(len(all_scores)))

    xshift = 0
    trans_offset = transforms.offset_copy(ax.transAxes, 
                                      fig=fig, 
                                      x=0, 
                                      y=0, 
                                      units='points')

   
    for scores in all_scores:
        yshift = 0
        for base, score in scores:
            txt = ax.text(0, 
                          0, 
                          base, 
                          transform=trans_offset,
                          fontsize=80, 
                          color=COLOR_SCHEME[base],
                          weight='bold',
                          ha='center',
                          family='sans-serif'
                          )
            txt.set_clip_on(False) 
            txt.set_path_effects([Scale(1.0, score)])
            fig.canvas.draw()
            window_ext = txt.get_window_extent(txt._renderer)
            yshift = window_ext.height*score
            trans_offset = transforms.offset_copy(txt._transform, fig=fig, y=yshift, units='points')
        xshift += window_ext.width
        trans_offset = transforms.offset_copy(ax.transAxes, fig=fig, x=xshift, units='points')


    ax.set_yticks(range(0,3))


    seaborn.despine(ax=ax, offset=30, trim=True)
    ax.set_xticklabels(range(1,len(all_scores)+1), rotation=90)
    ax.set_yticklabels(np.arange(0,3,1))
    # plt.show()
    plt.savefig('../results/yeah.png')

def freqatpos(seq):
	x=[[item[i] for item in seq] for i in range(len(seq[0]))]
	list_of_dic=[]
	for s in x:
		dic={}
		dic['A']=0.0
		dic['T']=0.0
		dic['G']=0.0
		dic['C']=0.0
		for t in s:
			dic[t]+=1
		dic['A']=dic['A']/len(seq)
		dic['T']=dic['T']/len(seq)
		dic['G']=dic['G']/len(seq)
		dic['C']=dic['C']/len(seq)
		# list_of_dic.append(dic)
		list_of_dic.append([(k,v) for k,v in dic.items()] )
	return list_of_dic

data=sio.loadmat('../data_gz/Reads.mat')
# a5d=data['A5SS']
# a5d=np.array(a5d.todense())
# a5n=np.array(a5d.sum(axis=1)>0).flatten()
# a5d=a5d[a5n]

# # a5d=a5d/a5d.sum(axis=1)[:, np.newaxis]

# a5seqs=pd.read_csv('../data_gz/A5SS_Seqs.csv', index_col=0).Seq[a5n]

# list_of_tup=freqatpos(a5seqs)

# # print(list_of_tup[:3])

# draw_logo(list_of_tup)

a3d=data['A3SS']
a3d=np.array(a3d.todense())

splitlen=800000
a3d=a3d[:splitlen,:]

a3n=np.array(a3d.sum(axis=1)>0).flatten()
a3d=a3d[a3n]

# a3d=a3d/a3d.sum(axis=1)[:, np.newaxis]

# five_splicingdist=convert_out_to_five_splicing_sites_for_3ss(a3d)
a3seqs=pd.read_csv('../data_gz/A3SS_Seqs.csv', index_col=0).Seq[:splitlen][a3n]

list_of_tup=freqatpos(a3seqs)

# print(list_of_tup[:3])

draw_logo(list_of_tup)
