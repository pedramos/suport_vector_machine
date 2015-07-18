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

def grad(X,Y,W0,Xteste,Yteste,Xdev,Ydev,max_iter,lamb):   #funcao que implementa o algoritmo gradient descent, Y nao normalizado
    from copy import copy, deepcopy
    w_old=W0
    import numpy as np
    import scipy.sparse as sparse
    import iteracoes as it
    import cria_dados
    import Rfechado as RR

    
    if max_iter==False:
        max_iter =2000
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
    if lamb == False:
        lambs=[10**x for x in xrange(0,10)]
        xg=map(lambda x: np.log10(x),lambs)
    else:
        lambs=[lamb]
        xg=[1]
    for lamb in lambs:
        print "A iniciar iteracoes para lambda=", lamb
        xx=[]
        y_old=0
        yy=[]
        w_old_old=W0
        i=0
        for i in xrange(max_iter):
        #while True:
            
            #step_size=step/np.sqrt(i+1)
            error=(X*w_old)-Y
            
            if i==0:
                step_size=0.0000001
            else:
                error_old=error=(X*w_old)-Y
                error_old_old=(X*w_old_old)-Y
                alpha=it.get_step(w_old,w_old_old,X,error_old,error_old_old,lamb)
                if alpha==0:
                    print "**ERRO**"
                    print "aplha=0, impossivel continuar o algorimto"
                    break
                step_size=sigma/alpha
            grad1=it.get_gradient(error,X,w_old,lamb)        
            w_new = w_old - (step_size * grad1)
            w_new=w_new.tocsr()
            error=(X*w_new)-Y
            dif=w_new-w_old
            dif=dif.transpose()*dif
            y_new=it.get_func(error,w_new,lamb) #funcao de erro
            
            
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
                    if count==5000:
                        break
                if count ==5000:
                    #print "****A SAIR****\nO sparsa encontrou o minimo"
                    break
                
            
            y_old=y_new
            #print y_new
            w_old_old=w_old
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
     
    
    import pylab
    
    import matplotlib.pyplot as plt
    plt.figure(1)
    
    plt.subplot(311)
    plt.title("Evolucao da funcao objectivo com o uso do SpaRSA")
    plt.plot(xxg,yyg,"b")
    #plt.plot(xxg,yyg,"b",xxg,[455532161.458 for cenas in xrange(len(xxg))],"r--")
    #pylab.ylim([4.5e8-1e10,9e+10])
    '''
    '''
    plt.subplot(312)
    #plt.title("Modulo do gradiente")
    #plt.plot(xx,yyy,"b")
    plt.title("Erro Test para os varios lambdas")
    plt.plot(xg,errortg,"b",xg,errortg,"ro")
    
    plt.subplot(313)
    plt.title("Erro Dev para os varios lambdas")
    plt.plot(xg,errorg,"b",xg,errorg,"ro")
    
    #print "FUNCAO DE CUSTO"
    #print yyg[len(yyg)-1]
    #plt.savefig("cenasquedemoram.png")
    #plt.show()
    
    return final[2], yyg,xxg

    
    

def main():
    f="../le_ficheiro/palavra1"
    print "FICHEIRO:",f,"\n"
    dictionary,total,y=cria_dados.read_output(f+"train.txt")
    X,Y=cria_dados.criaXY(dictionary,total,y,False)
    #X,total=cria_dados.delcomun(X,200,total)
    X,total=cria_dados.delstopword(X,total,True) 
    dictionary,temp,y=cria_dados.read_output(f+"test.txt")
    Xteste,Yteste=cria_dados.criaXY(dictionary,total,y,False)
    dictionary,temp,y=cria_dados.read_output(f+"dev.txt")
    Xdev,Ydev=cria_dados.criaXY(dictionary,total,y,False)
 
    dictionaty=[]
    y=[]
    gc.collect()
    vec=sparse.csr_matrix([0 for i in xrange(X.shape[1])])
    vec=vec.transpose()
    W,y,x=grad(X,Y,vec,Xteste,Yteste,Xdev,Ydev,False,False)
    #pdb.set_trace()
    error=(Xteste*W-Yteste)
    print "----------erro---------"   
    print "TESTE",RR.erro(Xteste,Yteste,W)
    print "-----------------------"
    
    print "OBJECTIVO:", it.get_func(error,W,100)
    print "PIORES 10"
    for coiso in sorted(W.toarray())[:10]:
        i=0
        while W[i,0] != coiso:
            i+=1
        else:
            for cenas in total:
                if total[cenas]==i:
                    print cenas , coiso, "->",i
                    break
    print "TOP 10"
    for coiso in sorted(W.toarray())[W.shape[1]-11:]:
        i=0
        while W[i,0] != coiso:
            i+=1
        else:
            for cenas in total:
                if total[cenas]==i:
                    print cenas , coiso, "->",i
                    break
    

    '''
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
'''
if __name__=="__main__":
    main()    
        
