# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 18:28:06 2013

@author: rami999999999
"""
import cria_dados
import Rgrad as grad 
import scipy.sparse as sparse
import numpy as np
import pdb
import iteracoes as it
import pickle
import Rgrad_old_step as grad_old
import Lasso as lasso

'''
Xtotal,Ytotal,W=cria_dados.XYartificiais(5,2,1000,0.1)

file=open("../dados_priberam/dados_X_random.pkl","wb")
pickle.dump(Xtotal,file)
file.close()

file=open("../dados_priberam/dados_Y_random.pkl","wb")
pickle.dump(Ytotal,file)
file.close()

file=open("../dados_priberam/dados_W_random.pkl","wb")
pickle.dump(W,file)
file.close()
'''
file=open("../dados_priberam/dados_X_random.pkl","rb")
Xtotal=pickle.load(file)

file=open("../dados_priberam/dados_Y_random.pkl","rb")
Ytotal=pickle.load(file)

file=open("../dados_priberam/dados_W_random.pkl","rb")
W=pickle.load(file)



trainsize=int(Xtotal.shape[0]*0.8)
devsize=int(Xtotal.shape[0]*0.1)+trainsize
lamb=10
max_iter=10000
vec=sparse.csr_matrix([0 for i in xrange(Xtotal.shape[1])])
vec=vec.transpose()
#pdb.set_trace()
error=Xtotal[:trainsize,:]*W - Ytotal[:trainsize,:]
custo=it.get_func(error,W,lamb)
print custo
#w_estimado=grad.grad(Xtotal[:trainsize,:],Ytotal[0:trainsize,0],vec,Xtotal[devsize:,:],Ytotal[devsize:,0],Xtotal[trainsize:devsize,:],Ytotal[trainsize:devsize,0],custo,lamb)
#w_estimado,yy1,xx1=grad_old.grad(Xtotal,Ytotal,vec,Xtotal,Ytotal,Xtotal,Ytotal,lamb,max_iter,True)
#w_estimado,yy2,xx2=grad_old.grad(Xtotal,Ytotal,vec,Xtotal,Ytotal,Xtotal,Ytotal,lamb,max_iter,False)
#w_estimado,yy3,xx3=grad.grad(Xtotal,Ytotal,vec,Xtotal,Ytotal,Xtotal,Ytotal,max_iter)
w0=[]
w1=[]
w2=[]
w3=[]
w4=[]
lambs=[0.1,1,10,10e2,10e3,10e4,10e5,10e6,10e7,10e8,10e9,10e10,10e11]
xg=[np.log10(x) for x in lambs]
'''
for  lamb in lambs:
    w_estimado,yy3,xx3=grad.grad(Xtotal,Ytotal,vec,Xtotal,Ytotal,Xtotal,Ytotal,max_iter,lamb)
    w0.append(w_estimado.todense()[0,0])
    w1.append(w_estimado.todense()[1,0])
    w2.append(w_estimado.todense()[2,0])
    w3.append(w_estimado.todense()[3,0])
    w4.append(w_estimado.todense()[4,0])
    print w_estimado
'''
for  lamb in lambs:
    w_estimado,yy3,xx3=lasso.lasso(Xtotal,Ytotal,vec,Xtotal,Ytotal,Xtotal,Ytotal,max_iter,lamb)
    w0.append(w_estimado[0,0])
    w1.append(w_estimado[1,0])
    w2.append(w_estimado[2,0])
    w3.append(w_estimado[3,0])
    w4.append(w_estimado[4,0])
    print w_estimado

import pylab
import matplotlib.pyplot as plt
plt.figure(1)
'''
plt.title("Comparacao entre os 3 metodos")
plt.plot(xx1,yy1,"b",xx2,yy2,"g",xx3,yy3,"r",xx1,[custo for cenas in xrange(len(xx1))],"k--")
pylab.ylim([custo-1e14,custo+4e14])
plt.show()
'''

plt.plot(xg,w0,"r",xg,w1,"k",xg,w2,"b",xg,w3,"g",xg,w4,"y")
plt.show()
'''
#pdb.set_trace()
print "X=>",X.data
print "Y=>",Y.data
print "W=>",W.data
vec=sparse.csr_matrix([1000000 for i in xrange(X.shape[1])])
vec=vec.transpose()
grad.grad(X,Y,vec)
'''

