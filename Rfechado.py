# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 15:43:17 2013

@author: rami999999999
"""

import sys
import pdb
import numpy as np
import scipy.sparse as sparse
import iteracoes as it
import cria_dados
import scipy.sparse.linalg as linalg

reload(sys)
sys.setdefaultencoding("utf-8")


    


def calculaW(X,Y,Xteste,Yteste,Xdev,Ydev): #funcao nao normaliza Y
    errorg=[]
    errortg=[]
    #calcula o W usando os valores de X e Y de train e de dev
    final=[-1,0,0]
    print "A fazer calculos para obter o W"
    #print "A calcular transposta de X"
    #print "Transposta calculada, a outros calculos"

    Xt=X.transpose()
    print "A testar lambdas:"
    lambs=[10**x for x in xrange(-3,8)]
    xg=map(lambda x: np.log10(x),lambs)
    #lambs=[100]
    #xg=[2]
    for lamb in lambs:
    #for lamb in [0.0000000000000000000000000001]:
        
        
        #calculo do w para cada lambda        
        

        temp=Xt*X
         
        shape=temp.shape        
        lambd=sparse.identity(shape[0])*lamb
        temp=temp+lambd
        
        temp=linalg.inv(temp)
        temp=temp*Xt
        temp=temp*Y
        #fim do calculo
        
        error=erro(Xdev,Ydev,temp)
        errortg.append(erro(Xteste,Yteste,temp))
        print lamb,error
        errorg.append(error) 
        if final[0]>error or final[0]==-1:
            final[0]=error
            final[1]=lamb
            final[2]=temp
    
    #valores returnados: temp->W final, final[0]->erro medio associado a esse valor, final[1]-> lambda
    try:
        import os
        os.environ['DISPLAY']
        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.subplot(211)
        plt.title("Evolucao do erro Dev")
        plt.plot(xg,errorg,"b",xg,errorg,"ro")
        plt.subplot(212)
        plt.title("Evolucao do erro Teste")
        plt.plot(xg,errortg,"b",xg,errortg,"ro")
        #plt.savefig("formula.png")
        plt.show()
    except :
        import os
        try:
            del os.environ['DISPLAY']
        except:
            pass
        print "Nao foi encontrado display\nO grafico vai ser escrito num ficheiro"
        print "Os dados usados sao:"+f
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.subplot(211)
        plt.title("Evolucao do erro Dev")
        plt.plot(xg,errorg,"b",xg,errorg,"ro")
        plt.subplot(212)
        plt.title("Evolucao do erro Teste")
        plt.plot(xg,errortg,"b",xg,errortg,"ro")
        plt.savefig("formula.png")

    return final[2],final[0],final[1]

def erro(X,Y,W):
    #dictionary,temp,y=cria_dados.read_output(ficheiro)
    #X,Y,cena1,cena2,cena3=cria_dados.criaXY(dictionary,total,y,True)
    #X,Y=cria_dados.criaXY(dictionary,total,y,False)

    #for indexe in xrange(len(mediaX)):
            #for indexe_linha in xrange(X.shape[0]):
                #X[indexe_linha,indexe]-=mediaX[indexe]
    Y_estimativa=X*W
 
    #Y_estimativa=sparse.csr_matrix(Y_estimativa.todok()*stdY)
    #Y_estimativa=sparse.csr_matrix(Y_estimativa.todok()+mediaY)
    erroRR=np.abs(Y-Y_estimativa)
    try:
        mediaRR=erroRR.sum()/erroRR.shape[0] 
    except:   
        mediaRR=sum(np.asarray(erroRR))/erroRR.shape[0]
    
        
    '''
    erroRR=erroRR.transpose()
    erroRR=erroRR.tolist()[0]
    for i in erroRR:
        i=i**2
    mediaRR=(sum(erroRR)/len(erroRR))
    mediaRR=np.sqrt(mediaRR)
    '''    
    '''    
    estimativa=sum(Y)/len(Y)
    for i in Y:
         i=(i-estimativa)**2
    media=sum(Y)/len(Y)
    '''    
    return mediaRR#,media       #mediaRR -> MSE || media->erro de algoritmo parvo
    
 
def errop(ficheiro,W,total):
    dictionary,temp,y=cria_dados.read_output(ficheiro)
    X,Y=cria_dados.criaXY(dictionary,total,y)
    Y_estimativa=X*W
    #Y_estimativa=map(lambda temp: 2**temp[0],Y_estimativa.tolist())
    
    erroRR=np.abs((Y-Y_estimativa)/Y)
    
    mediaRR=sum(erroRR.transpose().toarray()[0])/erroRR.shape[0]

        
    '''
    erroRR=erroRR.transpose()
    erroRR=erroRR.tolist()[0]
    for i in erroRR:
        i=i**2
    mediaRR=(sum(erroRR)/len(erroRR))
    mediaRR=np.sqrt(mediaRR)
    '''    
    '''    
    estimativa=sum(Y)/len(Y)
    for i in Y:
         i=(i-estimativa)**2
    media=sum(Y)/len(Y)
    '''    
    return mediaRR#,media       #mediaRR -> MSE || media->erro de algoritmo parvo
    
   
    
         
if __name__ == '__main__':
    f="../le_ficheiro/someta"
    dictionary,total,y=cria_dados.read_output(f+"train.txt")
    X,Y=cria_dados.criaXY(dictionary,total,y,False)
    dictionary,temp,y=cria_dados.read_output(f+"test.txt")
    Xteste,Yteste=cria_dados.criaXY(dictionary,total,y,False)
    dictionary,temp,y=cria_dados.read_output(f+"dev.txt")
    Xdev,Ydev=cria_dados.criaXY(dictionary,total,y,False)

    dictionary=[]
    y=[]
    #X,total=cria_dados.delcomun(X,total)

   

    ##Y=np.log2(Y)
    
    #dictionaryDev,totalDev,yDev=read_output("../le_ficheiro/dev.txt")
    #XDev,YDev=criaXY(dictionaryDev,total,yDev)
    
    #YDev=YDev/maxi
    #YDev=np.log10(YDev)
        
    W,erro_dev,lamb=calculaW(X,Y,Xteste,Yteste,Xdev,Ydev)
    error=(X*W-Y)
    print "OBJECTIVO:", it.get_func(error,W,lamb)
    '''
    print "___________________________________________________"
    print "___________________________________________________\n\n"
    print "lambda:",lamb
    print "erro do teste-----------------------\n", erro(f+"test.txt",W,total)
    #print "erro do treino-----------------------\n", erro("../le_ficheiro/train.txt",W,total)
    #print "erro do dev-----------------------\n", erro("../le_ficheiro/dev.txt",W,total)
    print "__________________________"
    print "W:"
    print W
    error=(X*W)-Y
    print "----------------------------------"
    print "funcao de custo:"
    print it.get_func(error,W,lamb)
    #print "----------------------------------"
        
'''