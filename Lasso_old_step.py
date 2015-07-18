# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:17:27 2013

@author: rami999999999
"""

import Rfechado as RRegression_beta
import pdb 
import numpy as np
import scipy.sparse as sparse
import itertools
import iteracoes as it
import cria_dados
import pickle
from copy import copy, deepcopy

        
    
def lasso(X,Y,w0,Xteste,Yteste,Xdev,Ydev):
    import matplotlib.pyplot as plt   
    w_new=np.zeros(w0.shape)
    final=[0,0,0]
    sigma=1
    sigma2=0.0000001
    max_iter =2000
    yyy=[]
    xxx=[]
    gmedia=[]
    gmediatest=[]
    w0=w0.todense()
    soft=it.soft
    lambs=[10**x for x in xrange(0,8)]
    xg=map(lambda x: np.log10(x),lambs)
    #lambdas=100*X.shape[0]
    #lambs=[lambdas]
    #xg=[2]
    for lamb in lambs:
        print "A iniciar Iteracoes para lambda=",lamb
        w_old_old=w0
        w_old=w0
        yy=[]
        xx=[]
        i=0
        while i<max_iter:
            #print "iteracao",i
            if i==0:
                step_size=0.0000001
            else:
                error_old=(X*w_old)-Y
                error_old_old=(X*w_old_old)-Y
                alpha=it.get_step(w_old,w_old_old,X,error_old,error_old_old,lamb)
                if alpha!=0:
                    step_size=sigma/alpha
                else:
                    i=max_iter
            error=(X*w_old)-Y
            grad1=it.GetGradL(error,X)
            U=it.getU(w_old,step_size,grad1)
            #pdb.set_trace()
            #w_new=[soft(x,step_size*lamb) for x in U.transpose().tolist()[0]]
            #w_new=sparse.csr_matrix(w_new)
            #w_new=w_new.transpose()
            w_new=it.softt(U,step_size*lamb)
            
            #pdb.set_trace()
            dif=w_new-w_old
        
            
            dif=dif.transpose()*dif
            error=(X*w_new)-Y
            y_new=it.get_func_lasso(error,np.matrix(w_new).transpose(),lamb) #funcao de erro        
            count=0            
            if i!=0:
                w_temp=w_new
                while y_new>=y_old-sigma2*alpha*dif[0,0] and i<max_iter:
                #while y_new>=y_old:
                    
                    #print "A diminuir step:",i
                    step_size=step_size/2
                    U=it.getU(w_old,step_size,grad1)
                    #w_new=[soft(x,step_size*lamb) for x in U.transpose().tolist()[0]]
                    #w_new=sparse.csr_matrix(w_new)
                    #w_new=w_new.transpose().transpose())
                    w_new=it.softt(U,step_size*lamb)
                    error=(X*w_new)-Y
                    dif=w_new-w_old
                    dif=dif.transpose()*dif
                    y_new=it.get_func_lasso(error,w_new,lamb) #funcao de custo
                    count=count+1
                    i=i+1
   
                    if count==10:
                        break
                else:
                    if i==max_iter:
                        w_new=w_temp
                 
            #if count ==5000:
                #print "****A SAIR****\nProvavelmente o sparsa chegou ao minimo antes de terminar o numero de iteracoes"
                #break
            #elif i==max_iter:
                #pass
                #print "Fim das interacoes"
            
            i=i+1

            y_old=y_new
            w_old_old=w_old
            w_old=w_new
            yy.append(y_new)
            xx.append(i)
        errod=RRegression_beta.erro(Xdev,Ydev,w_new)
        gmedia.append(errod)
        errot=RRegression_beta.erro(Xteste,Yteste,w_new)
        gmediatest.append(errot)
        if final[0]>errod or final[0]==0:
            final[0]=errod
            final[1]=lamb
            final[2]=errot
            graphFinal=deepcopy(yy)
            wfinal=deepcopy(w_new)
            yfinal=deepcopy(y_new)
            finalxx=deepcopy(xx)

        zero=0.0
        for J in xrange(w_new.shape[0]):
            if w_new[J,0]==0:
                zero=zero+1.0
        sp=(zero/w_new.shape[0])*100
        print "percentagem:",sp        
        yyy.append(sp)
        xxx.append(lamb)

    plt.figure(1)
    plt.subplot(221)
    plt.title("Funcao de custo")
    plt.plot(finalxx,graphFinal,"r",finalxx,[1.1959e15 for lolol in xrange(len(finalxx))])
    plt.subplot(222)
    plt.title("Percentagem de W com valor =0")
    import pylab    
    #print yyy
    #pylab.ylim([0,100])
    plt.plot(xg,yyy,"b",xg,yyy,"ro")
    plt.subplot(223)
    plt.title("Erro DEV ao longo dos lambdas")
    plt.plot(xg,gmedia,"b",xg,gmedia,"ro")
    plt.subplot(224)
    plt.title("Erro teste ao longo dos lambdas")
    plt.plot(xg,gmediatest,"b",xg,gmediatest,"ro")
    #pylab.savefig("lasso_beta.png")
    plt.show()  
    return wfinal,yfinal,final[1]

    
    



if __name__ == '__main__':
    f="../le_ficheiro/someta"
    dictionary,total,y=cria_dados.read_output(f+"train.txt")
    #X,Y,mediaY,stdY,mediaX=cria_dados.criaXY(dictionary,total,y,True)
    X,Y=cria_dados.criaXY(dictionary,total,y,False)
    dictionary,temp,y=cria_dados.read_output(f+"test.txt")
    Xteste,Yteste=cria_dados.criaXY(dictionary,total,y,False)
    dictionary,temp,y=cria_dados.read_output(f+"dev.txt")
    Xdev,Ydev=cria_dados.criaXY(dictionary,total,y,False)

    mediaX=0.0
    mediaY=0.0
    stdY=1.0
    #X,total=cria_dados.delcomun(X,total)
    vec=sparse.csr_matrix([0.0 for i in xrange(X.shape[1])])
    vec=vec.transpose()
    ficheiro=open("data.pkl","rb")
    #vec=pickle.load(ficheiro)
    W,F,lambd=lasso(X,Y,vec,Xteste,Yteste,Xdev,Ydev)
    error=X*W-Y
    print "OBJECTIVO",F
    print "OBJECTIVO_CALC:", it.get_func_lasso(error,W,lambd)
    print "----------erro---------"   
    print "LAMBDA:",lambd
    print "TESTE",RRegression_beta.erro(f+"test.txt",W,total,mediaY,stdY,mediaX)
    print "-----------------------"
    #print "TRAIN",RRegression_beta.erro("../le_ficheiro/train_meta.txt",W,total)
    #print "-----------------------"
    #print "DEV",RRegression_beta.erro("../le_ficheiro/dev_meta.txt",W,total)
