# -*- coding: utf-8 -*-
"""
Created on Mon May  6 14:51:53 2013

@author: rami999999999
"""
import sys
sys.path.append("../")
import scipy.sparse as sparse
import numpy as np
import pdb
import cria_dados
import iteracoes as it
import itertools
X,Y,W=cria_dados.XYlasso(10,10,10,10)

#########################calculaW formula#######################
print "********FORMULA************"
#calcula o W usando os valores de X e Y de train e de dev
finalF=[-1,0,0]
print "A fazer calculos para obter o W"
#print "A calcular transposta de X"
#print "Transposta calculada, a outros calculos"
X=sparse.csc_matrix(X)
Xdev=X[0:3,:]
Xtrain=X[3:7,:]
Xtest=X[7:9,:]

Ydev=Y[0:3,0]
Ytrain=Y[3:7,0]
Ytest=Y[7:9,0]

vec=sparse.csr_matrix([5 for i in xrange(X.shape[1])])
w0=vec.transpose()

import matplotlib.pyplot as plt   
print "A fazer calculos..." 
w_new=np.zeros(w0.shape)
final=[0,0]
sigma=0.01
sigma2=0.01
max_iter = 1000
yyy=[]
xxx=[]
soft=it.soft
gmedia=[]
for lamb in [10**x for x in xrange(-3,6)]:
#for lamb in [10**5]:
    w_old_old=w0
    w_old=w0
    yy=[]
    xx=[]
    print "calculos terminados, a fazer iterecoes"
    for i in xrange(max_iter):
        error=(X*w_old)-Y
        #print i
        if i==0:
            step_size=0.001
        else:
            error_old=error=(X*w_old)-Y
            error_old_old=(X*w_old_old)-Y
            alpha=it.get_step(w_old,w_old_old,X,error_old,error_old_old,lamb)
            if alpha==0:
                print "**ERRO**"
                print "aplha=0, impossivel continuar o algorimto"
                break
            step_size=sigma/alpha
        error=(X*w_old)-Y    
        grad1=it.GetGradL(error,X)
        U=it.getU(w_old,step_size,grad1).todense()
        w_new=list(itertools.imap(soft,U.tolist(),[step_size*lamb for x in xrange(len(U.tolist()))]))
        w_new=np.matrix(w_new)
        w_new=w_new.transpose()
        dif=w_new-w_old
        dif=dif.transpose()*dif
        w_new=sparse.csr_matrix(w_new)
        error=(X*w_new)-Y
        y_new=it.get_func(error,w_new,lamb) #funcao de erro        
        count=0            
        if i!=0:
            while y_new>=y_old-sigma2*alpha*dif[0,0]:
                print "A diminuir step:",i
                step_size=step_size/10
                U=it.getU(w_old,step_size,grad1).todense()
                w_new=list(itertools.imap(soft,U.tolist(),[step_size*lamb for x in xrange(len(U.tolist()))]))
                w_new=np.matrix(w_new)
                w_new=w_new.transpose()
                error=(X*w_new)-Y
                dif=w_new-w_old
                dif=dif.transpose()*dif
                y_new=it.get_func(error,w_new,lamb) #funcao de erro
                count=count+1
                if count==40:
                    break
        if count ==40:
            print "****A SAIR****\nProvavelmente o sparsa chegou ao minimo antes de terminar o numero de iteracoes"
            break
        y_old=y_new
        #print y_new
        w_old_old=w_old
        w_old = sparse.csr_matrix(w_new)
        yy.append(y_new)
        xx.append(i)
    print "exceeded maximum number of iterations, leaving"
    error=Xdev*w_new
    media=np.abs(error-Y[0:3,0])
    media=sparse.csr_matrix(media)
    media=sum(media[:,0].transpose().todense().tolist()[0])/media.shape[0]
    print lamb,media
    gmedia.append(media)
    if final[0]>media or final[0]==0:
        final[0]=media
        final[1]=lamb
        graphFinal=yy
        finalxx=xx
        Wfinal=w_new
    zero=0.0
    for J in xrange(w_new.shape[0]):
        if w_new[J,0]==0:
            zero=zero+1.0
        #print "zero->",zero
    sp=(zero/w_new.shape[0])*100
    print "percentagem:",sp        
    yyy.append(sp)
    xxx.append(lamb)
plt.figure(1)
plt.subplot(311)
plt.title("Funcao de custo")
plt.plot(finalxx,graphFinal,"r")
plt.subplot(312)
plt.title("Percentagem de W com valor =0")
import pylab    
print yyy
#pylab.ylim([0,100])
plt.plot(yyy,"b",yyy,"ro")    
plt.subplot(313)
plt.title("Evolucao do erro ao longo dos lambdas")
plt.plot(gmedia,"b",gmedia,"ro")
plt.tight_layout()
plt.show()   

print "\n\n********RESULTADOS************"
#print "custo grad: ", graphyy[len(graphyy)-1]

error=Xtest*Wfinal
error=np.abs(error-Y[7:9,0])

#error=sum(error[:,0].transpose().tolist()[0])/error.shape[0]
#print "ERRO GRAD:\n",error

print "_________ W calculado"
print Wfinal


print "_________ W original"
print W



