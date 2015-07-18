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

        
    
def lasso(X,Y,w0,Xteste,Yteste,Xdev,Ydev,max_iter,lamb):
    import matplotlib.pyplot as plt   
    w_new=np.zeros(w0.shape)
    zeros=np.zeros(w0.shape)
    final=[0,0,0]
    sigma=1
    sigma2=0.1
    if max_iter==False:
        max_iter=6000
    
    yyy=[]
    xxx=[]
    gmedia=[]
    gmediatest=[]
    w0=w0.todense()
    softt=it.softt
    get_step=it.get_step
    getU=it.getU
    get_func_lasso=it.get_func_lasso
    GetGradL=it.GetGradL
    #lambdas=100*X.shape[0]
    if lamb==False:
        lambs=[10**x for x in xrange(0,10)]
        xg=map(lambda x: np.log10(x),lambs)
    else:
        #lambs=[lamb*X.shape[0]]
        lambs=[lamb]
        xg=[2]
    for lamb in lambs:
        print "A iniciar Iteracoes para lambda=",lamb
        w_old_old=w0
        w_old=w0
        yy=[]
        xx=[]
        i=0
        while i<max_iter:
            #pdb.set_trace()
            #print "iteracao",i
            if i==0:
                step_size=0.0000001
            else:
                #pdb.set_trace()

                error_old=(X*w_old)-Y
                error_old_old=(X*w_old_old)-Y
                alpha=get_step(w_old,w_old_old,X,error_old,error_old_old,lamb)
                if alpha!=0:
                    step_size=sigma/alpha
                else:
                    i=max_iter
            error=(X*w_old)-Y
            grad1=GetGradL(error,X)
            U=getU(w_old,step_size,grad1)
            #pdb.set_trace()
            #w_new=[soft(x,step_size*lamb) for x in U.transpose().tolist()[0]]
            #w_new=sparse.csr_matrix(w_new)
            #w_new=w_new.transpose()
            w_new=softt(U,step_size*lamb,zeros)
            
            #pdb.set_trace()
            dif=w_new-w_old
        
            
            dif=dif.transpose()*dif
            error=(X*w_new)-Y
            #pdb.set_trace()
            y_new=get_func_lasso(error,w_new,lamb) #funcao de erro        
            #print i,"->",y_new
            count=0            
            if i!=0:
                w_temp=w_new
                while y_new>=y_old-sigma2*alpha*dif[0,0] and i<max_iter:
                #while y_new>=y_old:
                    
                    #print "A diminuir step:",i
                    step_size=step_size/2
                    U=getU(w_old,step_size,grad1)
                    w_new=softt(U,step_size*lamb,zeros)
                    error=(X*w_new)-Y
                    dif=w_new-w_old
                    dif=dif.transpose()*dif
                    y_new=get_func_lasso(error,w_new,lamb) #funcao de custo
                    count=count+1
                    i=i+1
   
                    if count==500:
                        print "nao consegue dar um passo pequeno o sufeciente"
                        w_new=w_old
                        break
                if count ==500:
                    #print "****A SAIR****\nProvavelmente o sparsa chegou ao minimo antes de terminar o numero de iteracoes"
                    break
            

          
                 
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
        print "fim do algoritmo"
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
    import pylab
    pylab.ylim([0e16,7e16])
    plt.plot(finalxx,graphFinal,"r",finalxx)
    plt.subplot(222)
    plt.title("Percentagem de W com valor =0")
        
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
    f="../le_ficheiro/meta+palavra1"
    print "FICHEIRO:",f,"\n"
    
    dictionary,total,y=cria_dados.read_output(f+"train.txt")
    #X,Y,mediaY,stdY,mediaX=cria_dados.criaXY(dictionary,total,y,True)
    X,Y=cria_dados.criaXY(dictionary,total,y,False)
    pdb.set_trace() 
    X,total=cria_dados.delstopword(X,total,True)    
    dictionary,temp,y=cria_dados.read_output(f+"test.txt")
    Xteste,Yteste=cria_dados.criaXY(dictionary,total,y,False)
    dictionary,temp,y=cria_dados.read_output(f+"dev.txt")
    Xdev,Ydev=cria_dados.criaXY(dictionary,total,y,False)
    
    #Y_train=Y.shape[0]
    #Y_test=Yteste.shape[0]
    #Y_dev=Ydev.shape[0]
    #Ytotal=np.vstack((Y.toarray(),Yteste.toarray(),Ydev.toarray()))
    #maxi=max(np.abs(Ytotal))
    #Ytotal=Ytotal-np.ones_like(Ytotal)*maxi    
    #Y=sparse.csr_matrix(Ytotal)[0:Y_train,0]
    #Yteste=sparse.csr_matrix(Ytotal)[Y_train:Y_test+Y_train,0]
    #Ydev=sparse.csr_matrix(Ytotal)[Y_test+Y_train:,0]

    vec=sparse.csr_matrix([0.0 for i in xrange(X.shape[1])])
    vec=vec.transpose()
    #ficheiro=open(".pkl","rb")
    #vec=pickle.load(ficheiro)
    W,F,lambd=lasso(X,Y,vec,Xteste,Yteste,Xdev,Ydev,False,False)
    print "OBJECTIVO:",F
    error=X*W-Y
    #print "OBJECTIVO",F
    #print "OBJECTIVO_CALC:", it.get_func_lasso(error,W,lambd)
    print "----------erro---------"   
    print "LAMBDA:",lambd
    print "TESTE",RRegression_beta.erro(Xteste,Yteste,W)
    print "-----------------------"
   
    print "PIORES 10"
    for coiso in sorted(np.array(W))[:10]:
        i=0
        while W[i,0] != coiso:
            i+=1
        else:
            for cenas in total:
                if total[cenas]==i:
                    print cenas , coiso, "->",i
                    break
    print "TOP 10"
    for coiso in sorted(np.array(W))[W.shape[1]-11:]:
        i=0
        while W[i,0] != coiso:
            i+=1
        else:
            for cenas in total:
                if total[cenas]==i:
                    print cenas , coiso, "->",i
                    break
    
    #print "TRAIN",RRegression_beta.erro("../le_ficheiro/train_meta.txt",W,total)
    #print "-----------------------"
    #print "DEV",RRegression_beta.erro("../le_ficheiro/dev_meta.txt",W,total)
