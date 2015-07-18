# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 17:23:33 2013

@author: rami999999999
"""

import grad_beta
import RRegression_beta
import scipy.sparse as sparse
import numpy as np
import pdb
lambdaa=0.001

X,Y,W=grad_beta.XYartificiais(10,10,10,10)

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
   
Xt=Xtrain.transpose()
print "A testar lambdas:"

#for lamb in [10**x for x in xrange(-6,6)]:
print "_____________________________________"
#for lamb in [10**x for x in xrange(-2,5)]:
for lamb in [lambdaa]:    
    
    #calculo do w para cada lambda        
    
    
    temp=Xtrain*Xt
     
    shape=temp.shape        
    lambd=sparse.identity(shape[0])*lamb
    temp=temp+lambd
    temp=temp.todense()
    temp=np.linalg.inv(temp)
    temp=Xt*temp
    temp=temp*Y[3:7,0]
    #fim do calculo
    
    '''
    ############CALCULO DO ERRO#############
    error=erro("../le_ficheiro/dev.txt",temp,total)
    '''    
    error=Xdev*temp
   
    error=np.abs(error-Y[0:3,0])
    error=sum(error[:,0].transpose().tolist()[0])/error.shape[0]
    print lamb,error
    
    if finalF[0]>error or finalF[0]==-1:
        finalF[0]=error
        finalF[1]=lamb
        finalF[2]=temp
        
ppp=(Xtrain*finalF[2])-Y[3:7,0]
print "funcao de custo:"
custo= grad_beta.get_func(ppp,finalF[2],finalF[1])

print custo
print "_________W"
print finalF[2].tolist()
print "_____________________________\n\n"
#############################CALCULO DO ERRO#####################
error=Xtest*finalF[2]   
error=np.abs(error-Y[7:9,0])
finalF[0]=sum(error[:,0].transpose().tolist()[0])/error.shape[0]
print "ERRO FORMULA:\n",finalF[0]


####################GRAD#######################
print "\n\n********GRADIENTE************"
finalG=[-1,0,0]
W0=sparse.csr_matrix([0 for i in xrange(X.shape[1])])
W0=W0.transpose()




import matplotlib.pyplot as plt
      
print "A fazer calculos..." 
#Y=sparse.csr_matrix(np.log10(Y.todense()))
w_old=W0
'''
trainShort.txt => step= 0.0000001 e max_iter=10000
train.txt => max_iter=300000      

##########COM SPARSA########
train.txt => sigma=0.0001 com ciclo (0.1 na condicao)

'''



def get_step(w_new,w_old,x,error_old,error_old_old,lamb):
    grad_new=grad_beta.get_gradient(error_old,x,w_new,lamb)
    grad_old=grad_beta.get_gradient(error_old,x,w_old,lamb)
    R=grad_new-grad_old
    S=w_new-w_old
    #pdb.set_trace()
    A=R.transpose()*R
    B=R.transpose()*S
    if A==0 or B==0:
        return 0
    else:
        xpto=(A)/(B)
        xpto=xpto.todense()[0,0]     
        return xpto
# Precision of the solution

#Use a fixed small step size 
#step = 0.000001
#for lamb in [10**x for x in xrange(-2,5)]:
for lamb in [lambdaa]:
    
    max_iter =10000
    yy=[]
    yyy=[]
    xx=[]
    
    #gc.collect()
    sigma1=0.0000001
    sigma2=0.001    
    #lamb=final[1] #valor retirado do RRegression    
    w_old_old=0
    #print "calulos terminados, a fazer iterecoes"
    for i in xrange(max_iter):
        #print i
       
        #step_size=step/np.sqrt(i+1)
        
        
        error=(Xtrain*w_old)-Y[3:7,0]
        if i==0:
            step_size=0.0000001
        else:
            error_old=error=(Xtrain*w_old)-Y[3:7,0]
            error_old_old=(Xtrain*w_old_old)-Y[3:7,0]
            alpha=get_step(w_old,w_old_old,Xtrain,error_old,error_old_old,lamb)
            if alpha==0:
                print "**ERRO**"
                print "aplha=0, impossivel continuar o algorimto"
                break
            step_size=sigma/alpha
        #print "fora:",step_size
        grad1=grad_beta.get_gradient(error,Xtrain,w_old,lamb)
        
        w_new = w_old - (step_size * grad1)
        error=(Xtrain*w_new)-Y[3:7,0]
        dif=w_new-w_old
        dif=dif.transpose()*dif
        y_new=grad_beta.get_func(error,w_new,lamb)[0,0] #funcao de erro
        
        
        count=0
        if i!=0:
            
            while y_new>=y_old-sigma2*alpha*dif[0,0]:
                count=count+1
			
			
                #print "A diminuir step:",i
                step_size=step_size/10
                #print "dentro:",step_size
                #print "w_new:",w_new
                #print "w_old:",w_old
                #print "step_size:",step_size
                #print "grad1",grad1
                w_new = w_old - (step_size * grad1)
                error=(Xtrain*w_new)-Y[3:7,0]
                dif=w_new-w_old
                dif=dif.transpose()*dif
            
                y_new=grad_beta.get_func(error,w_new,lamb) #funcao de erro
                if count==20:                
                    break
            
        if count==20:
			break
        
        y_old=y_new
        w_old_old=w_old
        w_old = w_new
        #print "y_new:",y_new
        yy.append(y_new)
        xx.append(i)
        yyy.append(((grad1.transpose()*grad1).data)[0]) 
        
    
    error=Xdev*w_new
   
    error=np.abs(error-Y[0:3,0])
    error=sum(error[:,0].transpose().todense().tolist()[0])/error.shape[0]
    print lamb,error    
    if finalG[0]>error or finalG[0]==-1:
        finalG[0]=error
        finalG[1]=lamb
        finalG[2]=w_new
        graphyy=yy
        graphyyy=yyy
        graphxx=xx


plt.figure(1)
plt.subplot(211)

print "lamb:",finalG[1]
plt.title("Funcao de custo")

plt.plot(graphxx,graphyy,"b",graphxx,[ custo[0,0] for cenas in xrange(len(graphxx))],"r--")
import pylab    
pylab.ylim([custo[0,0]-500,custo[0,0]+2000])    
plt.subplot(212)
plt.title("Modulo do gradiente")
plt.plot(graphxx,graphyyy,"b")

plt.show()
plt.savefig("plot.png")
    
#############################CALCULO DO ERRO#####################
print "\n\n********RESULTADOS************"
print "custo grad: ", graphyy[len(graphyy)-1]
error=Xtest*finalG[2]
error=np.abs(error-Y[7:9,0])

error=sum(error[:,0].transpose().todense().tolist()[0])/error.shape[0]
print "ERRO GRAD:\n",error

print "_________"
print finalF[2]
print "_________"
print finalG[2]

