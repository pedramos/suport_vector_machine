#-*- coding: utf-8 -*-
"""
Created on Mon Mar 11 18:49:56 2013

@author: rami999999999
"""



import pdb
import gc
import numpy as np
import scipy.sparse as sparse
import iteracoes as it
import cria_dados
import Rfechado as RR

def grad(X,Y,W0,Xteste,Yteste,Xdev,Ydev,arg_lamb, max_iter,fixo):   #funcao que implementa o algoritmo gradient descent, Y nao normalizado
    from copy import copy, deepcopy
    w_old=W0
    import numpy as np
    import scipy.sparse as sparse
    import iteracoes as it
    import cria_dados
    import Rfechado as RR

    if arg_lamb == False:
        #lambs=[10**x for x in xrange(-2,10)]
        lambs=[10]
        xg=[1]
    else:
        lambs=[arg_lamb]
        xg=[np.log10(arg_lamb)]
    
    #max_iter =30000000
    gc.collect()
    #############SIGMA######################
    #                                      #
    #********Para lambda>1 funciona********#    
    sigma=1                       #
    sigma2=0.0000001                       #
    #                                      #
    #********Para lambda <1 funciona*******#
    #sigma=0.0001                          #
    #sigma2=0.000001                       #
    #                                      #    
    ########################################
    errorg=[]
    errortg=[]
    final=[-1,0,0]
    
    
    for lamb in lambs:
        print "A iniciar iteracoes para lambda=", lamb
        xx=[]
        y_old=0
        yy=[]
        w_old=W0
        i=0
        
        for i in xrange(max_iter):
            if fixo==True:
                step_size=1e-10
            else:
                step=1e-9
                step_size=step/np.sqrt(i+1)
            error=(X*w_old)-Y        
            grad1=it.get_gradient(error,X,w_old,lamb)        
            w_new = w_old - (step_size * grad1)
            w_new=w_new.tocsr()
            error=(X*w_new)-Y
            #dif=w_new-w_old
            #dif=dif.transpose()*dif
            y_new=it.get_func(error,w_new,lamb) #funcao de erro
            
            '''
            count=0
            if i!=0:
                while y_new>=y_old-sigma2*alpha*dif[0,0]:
                    #print "A diminuir step:",i
                    step_size=step_size/2
                    w_new = w_old - (step_size * grad1)
                    w_new=w_new.tocsr()
                    error=(X*w_new)-Y
                    dif=w_new-w_old
                    dif=dif.transpose()*dif
                    y_new=it.get_func(error,w_new,lamb) #funcao de erro
                    count=count+1
                    #y_new=y_new[0,0]
                    if count==1000:
                        break
                if count ==1000:
                    #print "****A SAIR****\nO sparsa encontrou o minimo"
                    break
                
            '''
            y_old=y_new
            #print y_new
            w_old = w_new
            #print "y_new:",y_new
            yy.append(y_new)
            xx.append(i)
            #yyy.append(((grad1.transpose()*grad1).data)[0])
            #i=i+1
        
        #pdb.set_trace()
        error=RR.erro(Xdev,Ydev,w_new)
        errorg.append(error)
        errortg.append(RR.erro(Xteste,Yteste,w_new))
        if final[0]>error or final[0]==-1:
            final[0]=error
            final[1]=lamb
            final[2]=w_new
            yyg=deepcopy(yy)
            xxg=deepcopy(xx)

    print final[1] 
    ''' 
    import os
    import pylab
    os.environ['DISPLAY']
    import matplotlib.pyplot as plt
    plt.figure(1)
    
    plt.title("n = 1e-8")
    plt.plot(xxg,yyg,"b")
    plt.plot(xxg,yyg,"b",xxg,[custo for cenas in xrange(len(xxg))],"r--")
    pylab.ylim([custo-1e90,custo+8e90])
    '''
    '''
    plt.subplot(311)
    plt.title("Funcao de custo ao longo das iteracoes")
    plt.plot(xxg,yyg,"b")
    #plt.plot(xxg,yyg,"b",xxg,[9.42247e+15 for cenas in xrange(len(xxg))],"r--")
    #pylab.ylim([4.082103e+15,15.082103e+15])
    
    plt.subplot(312)
    #plt.title("Modulo do gradiente")
    #plt.plot(xx,yyy,"b")
    plt.title("Erro Test para os varios lambdas")
    plt.plot(xg,errortg,"b",xg,errortg,"ro")
    
    plt.subplot(313)
    plt.title("Erro Dev para os varios lambdas")
    plt.plot(xg,errorg,"b",xg,errorg,"ro")
    '''
    print "FUNCAO DE CUSTO"
    #print yyg[len(yyg)-1]
    #plt.show()
    
    return final[2],yyg,xxg

    
    

def main():
    media=0
    maximo=1
    f="../le_ficheiro/someta"
    dictionary,total,y=cria_dados.read_output(f+"train.txt")
    X,Y=cria_dados.criaXY(dictionary,total,y,False)
    #X,total=cria_dados.delcomun(X,total)
    dictionary,temp,y=cria_dados.read_output(f+"test.txt")
    Xteste,Yteste=cria_dados.criaXY(dictionary,total,y,False)
    dictionary,temp,y=cria_dados.read_output(f+"dev.txt")
    Xdev,Ydev=cria_dados.criaXY(dictionary,total,y,False)
 
    dictionaty=[]
    y=[]
    gc.collect()
    vec=sparse.csr_matrix([0 for i in xrange(X.shape[1])])
    vec=vec.transpose()
    W=grad(X,Y,vec,Xteste,Yteste,Xdev,Ydev)
    print "W:"
    print W.todense()
    
    print "------------------------------"
    print "MSE(teste):\n",RR.erro(f+"test.txt",W,total,media,maximo)    
    print "------------------------------"
        
    print "------------------------------"
    #print "MSE(train):\n",erro(f+"train.txt",W,total)
    #print "------------------------------"
    print "MSE(dev):\n",RR.erro(f+"dev.txt",W,total,media,maximo)
     
    #error=(X*W)-Y
    #print "Funcao de custo:",get_func(error,W,10)

if __name__=="__main__":
    main()    
        
