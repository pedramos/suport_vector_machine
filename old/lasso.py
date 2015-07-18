# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:17:27 2013

@author: rami999999999
"""

import RRegression_beta
import grad_beta
import pdb
import numpy as np
import scipy.sparse as sparse
import itertools
import iteracoes as it
import cria_dados



        
    
def lasso(X,Y,w0,total):
    import matplotlib.pyplot as plt   
    print "A fazer calculos..." 
    w_new=np.zeros(w0.shape)
    Y=sparse.csr_matrix(np.log2(Y.todense()))
    final=[0,0]

    #Use a fixed small step size 
    step_size = 0.00000002
    #max iterations
    max_iter = 1000
    yyy=[]
    xxx=[]
    it.soft=soft
    for lamb in [10**x for x in xrange(-12,-9)]:
    #for lamb in [0.000001]:
        w_old=w0
        yy=[]
        
        xx=[]
        
               
        print "calculos terminados, a fazer iterecoes"
        for i in xrange(max_iter):
            
            error=(X*w_old)-Y    
            grad1=it.GetGradL(error,X)
            U=it.getU(w_old,step_size,grad1).todense()
            w_new=list(itertools.imap(soft,U.tolist(),[step_size*lamb for x in xrange(len(U.tolist()))]))
            w_new=np.matrix(w_new)
            w_new=w_new.transpose()
            error=(X*w_new)-Y
            y_new=it.get_func(error,w_old,lamb) #funcao de erro
            y_old=y_new
            w_old = sparse.csr_matrix(w_new)
            yy.append(y_new)
            xx.append(i)
            
        print "exceeded maximum number of iterations, leaving"
        
        media=RRegression_beta.erro("../le_ficheiro/dev.txt",w_new,total)
        
        print lamb,media
        if final[0]>media or final[0]==0:
            final[0]=media
            final[1]=lamb
            graphFinal=yy
    
        
        
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
    plt.subplot(211)
    plt.title("Funcao de custo")
    plt.plot(xx,graphFinal,"r")
    plt.subplot(212)
    plt.title("Percentagem de W com valor =0")
    import pylab    
    print yyy
    #pylab.ylim([0,100])
    plt.plot(yyy,"b",yyy,"ro")    
    plt.show()  
    return w_new,y_new

    
    



if __name__ == '__main__':
    dictionary,total,y=cria_dados.read_output("../le_ficheiro/train_meta.txt")
    X,Y=cria_dados.criaXY(dictionary,total,y)
    X,total=cria_dados.delcomun(X,total)

    Y=sparse.csr_matrix(Y)
    X=sparse.csr_matrix(X)
    
    vec=sparse.csr_matrix([5 for i in xrange(X.shape[1])])
    vec=vec.transpose()
    W,F=lasso(X,Y,vec,total)
    print "----------erro---------"    
    print "TESTE",RRegression_beta.erro("../le_ficheiro/test_meta.txt",W,total)
    print "-----------------------"
    print "TRAIN",RRegression_beta.erro("../le_ficheiro/train_meta.txt",W,total)
    print "-----------------------"
    print "DEV",RRegression_beta.erro("../le_ficheiro/dev_meta.txt",W,total)
