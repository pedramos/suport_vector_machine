# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:17:27 2013

@author: rami999999999
"""

import Rfechado as RRegression_beta
import Rfechado as Rfechado
import pdb 
import numpy as np
import scipy.sparse as sparse
import itertools
import iteracoes as it
import cria_dados
from copy import copy, deepcopy

        

def elastic_net(X,Y,w0,Xteste,Yteste,Xdev,Ydev,alpha_en):
    import matplotlib.pyplot as plt   
    w_new=np.zeros(w0.shape)
    zeros=np.zeros(w0.shape)
    final=[0,0]
    sigma=1
    sigma2=0.1
    max_iter =40000
    yyy=[]
    xxx=[]
    gmedia=[]
    gmediatest=[]
    w0=w0.todense()
    softt=it.softt
    getU=it.getU
    get_step=it.get_step
    get_func_elastic=it.get_func_elastic
    GetGradL=it.GetGradL
    lambs=[10**x for x in xrange(0,10)]
    xg=map(lambda x: np.log10(x),lambs)
    
    #lambs=[100]
    #lambs=[1000*X.shape[0]]
    #xg=[7]
    for lamb in lambs:

        print "A iniciar Iteracoes para lambda=",lamb
        w_old_old=w0
        w_old=w0
        yy=[]
        xx=[]
        i=0
        while i<max_iter:
            #print "iteracao",i

            error=(X*w_old)-Y
            if i==0:
                step_size=0.0000001
            else:

                #pdb.set_trace()
                error_old=error=(X*w_old)-Y
                error_old_old=(X*w_old_old)-Y
                alpha=get_step(w_old,w_old_old,X,error_old,error_old_old,lamb)
                if alpha==0:
                    #print "**ERRO**"
                    #print "aplha=0, impossivel continuar o algorimto"
                    break
                step_size=sigma/alpha
            error=(X*w_old)-Y    
            grad1=GetGradL(error,X)
            U=getU(w_old,step_size,grad1)
            K=1+step_size*lamb*(1-alpha_en)
            w_new=softt(U,step_size*lamb*alpha_en/K,zeros)
            dif=w_new-w_old
            dif=dif.transpose()*dif
            error=(X*w_new)-Y
            y_new=get_func_elastic(error,w_new,lamb,alpha_en) #funcao de erro   
            #print i,"->",y_new     
            count=0            
            if i!=0:
                while y_new>=y_old-sigma2*alpha*dif[0,0] and i<max_iter:
                    #print "A diminuir step:",i
                    step_size=step_size/2
                    U=getU(w_old,step_size,grad1)
                    K=1+step_size*lamb*(1-alpha_en)
                    w_new=softt(U,step_size*lamb*alpha_en/K,zeros)
                    error=(X*w_new)-Y
                    dif=w_new-w_old
                    dif=dif.transpose()*dif
                    y_new=get_func_elastic(error,w_new,lamb,alpha_en) #funcao de erro
                    count=count+1
                    i=i+1
                    if count==10000:
                        break
            if count ==10000:
                #print "****A SAIR****\nProvavelmente o sparsa chegou ao minimo antes de terminar o numero de iteracoes"
                break
          
           
            i=i+1

            y_old=y_new
            w_old_old=w_old
            w_old = w_new
            yy.append(y_new)
            xx.append(i)
        media=RRegression_beta.erro(Xdev,Ydev,w_new)
        gmedia.append(media)
        gmediatest.append(RRegression_beta.erro(Xteste,Yteste,w_new))
        if final[0]>media or final[0]==0:
            final[0]=media
            final[1]=lamb
            graphFinal=deepcopy(yy)
            wfinal=w_new
            yfinal=y_new
            finalxx=deepcopy(xx)

        zero=0.0
        for J in xrange(w_new.shape[0]):
            if w_new[J,0]==0:
                zero=zero+1.0
        sp=(zero/w_new.shape[0])*100
        #print "percentagem:",sp        
        yyy.append(sp)
        xxx.append(lamb)
    '''
    plt.figure(1)
    plt.subplot(221)
    plt.title("Funcao de custo")
    plt.plot(finalxx,graphFinal,"r")
    plt.subplot(222)
    plt.title("Percentagem de W com valor =0")
    import pylab    
    #print yyy
    #pylab.ylim([0,100])
    plt.plot(xg,yyy,"b",xg,yyy,"ro")
    plt.subplot(223)
    plt.title("Evolucao do erro DEV ao longo dos lambdas")
    plt.plot(xg,gmedia,"b",xg,gmedia,"ro")
    plt.subplot(224)
    plt.title("Evolucao do erro teste ao longo dos lambdas")
    plt.plot(xg,gmediatest,"b",xg,gmediatest,"ro")
    plt.tight_layout()
    #pylab.savefig("elastic_beta.png")
    plt.show()
    '''  
    return wfinal,yfinal,final[1]

    
    



if __name__ == '__main__':
    f="../le_ficheiro/someta"
    print "FICHEIRO:",f,"\n"

    
    dictionary,total,y=cria_dados.read_output(f+"train.txt")
    #X,Y,mediaY,stdY,mediaX=cria_dados.criaXY(dictionary,total,y,True)
    X,Y=cria_dados.criaXY(dictionary,total,y,False)
    #X,total=cria_dados.delstopword(X,total,True)    
    dictionary,temp,y=cria_dados.read_output(f+"test.txt")
    Xteste,Yteste=cria_dados.criaXY(dictionary,total,y,False)
    dictionary,temp,y=cria_dados.read_output(f+"dev.txt")
    Xdev,Ydev=cria_dados.criaXY(dictionary,total,y,False)
    #X,total=cria_dados.delcomun(X,total)
    vec=sparse.csr_matrix([0.0 for i in xrange(X.shape[1])])
    vec=vec.transpose()
    '''
    W,F=elastic_net(X,Y,vec,Xteste,Yteste,Xdev,Ydev,alpha_en)
    print "----------erro---------"    
    print "TESTE",RRegression_beta.erro(f+"train.txt",W,total)
    print "-----------------------"
    '''
    print "-----------------------------"

    #W,F,lamb=elastic_net(X,Y,vec,Xteste,Yteste,Xdev,Ydev,0.0)
    #print "ALPHA", alpha
    #print "LAMBDA",lamb
    #print "ERRO_DEV:",Rfechado.erro(Xdev,Ydev,W)    #W=elastic_net.elastic_net(Xtrain,Ytrain.transpose(),vec,Xdev,Ydev.transpose(),Xtest,Ytest.transpose(),0.5)
    #print "ERRO_TEST:",Rfechado.erro(Xteste,Yteste,W) 
    #error=X*W-Y
    #print "OBJECTIVO:", it.get_func_elastic(error,W,1000*X.shape[0],1) 
    #print "OBJECTIVO:", F 

    
    for alpha in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
        print "------------------------------"
        W,F,lamb=elastic_net(X,Y,vec,Xteste,Yteste,Xdev,Ydev,alpha)
        print "ALPHA", alpha
        print "LAMBDA",lamb
        print "ERRO_DEV:",Rfechado.erro(Xdev,Ydev,W)    #W=elastic_net.elastic_net(Xtrain,Ytrain.transpose(),vec,Xdev,Ydev.transpose(),Xtest,Ytest.transpose(),0.5)
        print "ERRO_TEST:",Rfechado.erro(Xteste,Yteste,W) 
        
    #print "TRAIN",RRegression_beta.erro("../le_ficheiro/train_meta.txt",W,total)
    #print "-----------------------"
    #print "DEV",RRegression_beta.erro("../le_ficheiro/dev_meta.txt",W,total)
