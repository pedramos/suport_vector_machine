# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 18:49:56 2013

@author: rami999999999
"""



import RRegression
import pdb
import gc
import numpy as np
import scipy.sparse as sparse


    

def get_func(error,W,lamb): #retorna o valor da funcao para um determinado X,Y,W
    temp=error.transpose()*error
    temp=temp+(W.transpose()*W)*lamb
    return temp
    
def XYartificiais(F,N,sigmaXY,sigmaN):
     
    W=np.abs(np.random.normal(0,sigmaXY,(F,1)))
    print W.shape
    X=np.abs(np.random.normal(0,sigmaXY,(N,F)))
    print X.shape
    Y=np.zeros((N,1))
    print Y.shape
    
    for i in xrange(N):
                
        x=X[i,:]
        x.shape=(F,1)
        n=sigmaN*np.random.normal(0,sigmaXY,(1,1))
        Y[i,0]=((W.transpose()).dot(x))+n
    X=sparse.csr_matrix(X)
    Y=sparse.csr_matrix(Y)
    W=sparse.csr_matrix(W)
    return X,Y,W
    
def get_grad2(Y,W,X): #gradiente calculado pela defenicao
   end=np.zeros(W.shape)
   error=(X*W)-Y
   for k in range(W.shape[0]):
       temp=get_func(error,W,0.001)
       #pdb.set_trace()       
       vec=[0 for i in xrange(W.shape[0])]
       #pdb.set_trace()
       vec[k]=0.000001
       vec=sparse.csr_matrix(vec)
       vec=vec.transpose()
       W0=W+vec
       temp1=get_func(error,W0,0.001)
       temp=((temp-temp1)/0.000001)
    
       
       end[k][0]=temp.todense()
       print end[k][0]
      
   return sparse.csr_matrix(end)

def get_grad2_graph(X,Y,W):
    import matplotlib.pyplot as plt
    xx=[]
    yy=[]
    vec=[0 for i in xrange(W.shape[0])]
    
    
    
    
    for i in [10**k for k in xrange(-7,0)]:
        print i        
        vec=[0 for j in xrange(W.shape[0])]
        vec[0]= i           
        vec=sparse.csr_matrix(vec) 
        vec=vec.transpose()
        temp=get_func(X,Y,W)
        #pdb.set_trace()
        Wtemp=W+vec
        temp1=get_func(X,Y,Wtemp)
        temp=((temp1-temp)/i)
        yy.append(temp[0][0].data)
        xx.append(i)
   
    plt.plot(xx,yy)
    plt.show()
    
    
def get_gradient(error,X,W,lamb):  
    
    error=X.transpose()*error
    
    error=error+lamb*W
    
    #print temp
    
    return error
    
def get_step(w_new,w_old,x,error,lamb):
    grad_new=get_gradient(error,x,w_new,lamb)
    grad_old=get_gradient(error,x,w_old,lamb)
    R=grad_new-grad_old
    S=w_new-w_old
    pdb.set_trace()
    xpto=(S.transpose()*R)/(S.transpose()*S)
    xpto=xpto.todense()[0,0]
    return xpto


def grad(X,Y,W0):   #funcao que implementa o algoritmo gradient descent, Y nao normalizado
    import matplotlib.pyplot as plt   
    print "A fazer calculos..." 
    Y=sparse.csr_matrix(np.log10(Y.todense()))
    w_old=W0
    '''
    trainShort.txt => step= 0.0000001 e max_iter=10000
    train.txt => max_iter=300000      
    '''
    # Precision of the solution
    prec = 500
    #Use a fixed small step size 
    step = 0.000001
    
    max_iter =100000
    yy=[]
    yyy=[]
    xx=[]
    gc.collect()    
    lamb=1000 #valor retirado do RRegression    
    w_old_old=0
    print "calulos terminados, a fazer iterecoes"
    for i in xrange(max_iter):
        step_size=step/np.sqrt(i+1)
        error=(X*w_old)-Y
        '''
        #usar em caso de sparsa
        if i==0:
            step_size=0.0000000000001
        else:
            step_size=get_step(w_old,w_old_old,X,error,lamb)
        '''
        grad1=get_gradient(error,X,w_old,lamb)
        w_new = w_old - (step_size * grad1)
        y_new=get_func(error,w_new,lamb) #funcao de erro
        y_old=y_new
        w_old_old=w_old
        w_old = w_new
        yy.append(y_new.data)
        xx.append(i)
        yyy.append(((grad1.transpose()*grad1).data)[0])        
        if( abs(y_new - y_old) < prec):
            print "change in function values too small, leaving" 
            return w_new
    print "exceeded maximum number of iterations, leaving" 
    plt.figure(1)
    plt.subplot(211)
    plt.title("Funcao de custo")
    plt.plot(xx,yy,"b",xx,[ 494.60545179 for cenas in xrange(len(yy))],"r--")
    import pylab    
    pylab.ylim([0,10000])    
    plt.subplot(212)
    plt.title("Modulo do gradiente")
    plt.plot(xx,yyy,"b")    
    plt.show()
    plt.savefig("plot.pdf")    
    return w_new,y_new

    
    

def main():
    dictionary,total,y=RRegression.read_output("../le_ficheiro/train.txt")
    X,Y=RRegression.criaXY(dictionary,total,y)
    X,total=RRegression.delcomun(X,total)
    X=sparse.csr_matrix(X)
    Y=sparse.csr_matrix(Y)
    
    gc.collect()
    vec=sparse.csr_matrix([0 for i in xrange(X.shape[1])])
    vec=vec.transpose()
    W,F=grad(X,Y,vec)
    #print W.todense()
    #print "___________________"
    #print F.todense()
    print "MSE:",RRegression.erro("../le_ficheiro/test.txt",W,total)    


if __name__=="__main__":
    main()    
        
