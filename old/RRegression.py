# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 15:43:17 2013

@author: rami999999999
"""

import sys
import pdb
import numpy as np
import scipy.sparse as sparse

import grad


reload(sys)
sys.setdefaultencoding("utf-8")

def delcomun(X,total):
    print "A apagar palavras mais comuns"
    import scipy as sc
    import operator
    cut=[]
    count={}
    for palavra in total:
        count[palavra]=sum(X[:,total[palavra]])
    
    for i in xrange(25):
        maxi=max(count.iteritems(), key=operator.itemgetter(1))[0]
                
        
        x=0
        for cutted in cut:
            if total[maxi]>cutted:
                x+=1
       
        X=sc.delete(X,total[maxi]-x,1)
       
        cut.append(total[maxi])
        del total[maxi]
        del count[maxi]
    
    for indice,palavra in enumerate(total):
        total[palavra]=indice
            
    return X,total
    

def read_output(local):
    print "A ler ficheiro:",local
    #interpreta os ficheiros de texto com a contagem das palavras das review, devolve o dictionary, total e y
    import codecs
    
    from collections import deque
    from collections import OrderedDict
    total=OrderedDict() #dicionario com todas as palavras presentes no ficheiro e o seu indice na matriz X
    dictionary={} #dicionario onde sao guardadas as palavras presentes em cada filme 
    y={} #dicionário com as receitas geradas por cada filme
    dados=deque()
    linha=1
    index=0
    file=codecs.open(local,"r","utf-8")
    for Nmovie,movie in enumerate(file):
        #print "A processar linha %i" %linha
        linha=linha+1
        
        #print movie
        dados=movie.split("\t")
        dictionary[dados[0]]={}
        y[dados[0]]=dados[1]
        filme=dados[0]
        
        for Npalavra,palavra in enumerate(dados[2:]):
            #print palavra
            
            if len(palavra)==0:
                #print palavra
                pass
            elif Npalavra%2==0:
                feature=palavra
                if feature not in total:
                    total[feature]=index
                    index+=1
            else:
                
                dictionary[filme][feature]=int(palavra)
    total["_____"]=index
    return dictionary,total,y

def criaXY(dictionary,total,y):   
    print "A criar X e Y"
    #calcula o X e Y para as palavras encontradas na review (dictionary), total para indexar as palavras na matriz X(as palavras presentes em dictionary que nao estejam em total nao sao consideradas) e y o lucro de cada filme          
    #print "A criar estrutura "
    X=np.zeros((len(dictionary),len(total)),dtype=np.int32)
    #pdb.set_trace()    
    #print "matriz criada"
    Y=np.zeros((len(y),1))
    
    for indice,movie in enumerate(dictionary):
        #print "Filme:", indice,movie
        for palavra in dictionary[movie]:
                  
            '''            
            print movie
            print "indice:",indice, "\n"
            print "palavra",palavra, "\n"
            print "toltal.index",total.index(palavra),"\n"
            
            print dictionary[movie][palavra], "\n"
            '''
            if palavra not in total:
                pass
            else:
                X[indice][total[palavra]]=int(dictionary[movie][palavra])
        
        Y[indice][0]=float(y[movie])
        X[indice][total["_____"]]=1
        #print "Processamento concluido", indice,movie
	#print X
	#print total
     
    return X,Y #X e Y sao matrizes numpy
        
        
def calculaW(X,Y,Xdev,Ydev): #funcao nao normaliza Y
    
    #calcula o W usando os valores de X e Y de train e de dev
    final=[0,0]
    print "A fazer calculos para obter o W"
    #print "A calcular transposta de X"
    #print "Transposta calculada, a outros calculos"
    X=sparse.csc_matrix(X)

    Xdev=sparse.csc_matrix(Xdev)
    Xt=X.transpose()
    print "A testar lambdas:"
    for lamb in [10**x for x in xrange(3,5)]:
        
        
        #calculo do w para cada lambda        
        

        temp=X*Xt
         
        shape=temp.shape        
        lambd=sparse.identity(shape[0])*lamb
        temp=lambd+temp
        temp=temp.todense()
        temp=np.linalg.inv(temp)
        temp=Xt*temp
        temp=temp*Y
        #fim do calculo
        Y_estimativa=Xdev*temp
        

        #MAE
        
        #Ydev1=map(lambda temp: 10**temp[0],Ydev.tolist())    
        #Ydev1=np.matrix(Ydev1).transpose()       
        #Y_estimativa=map(lambda temp: 10**temp[0],Y_estimativa.tolist())
        #Y_estimativa=np.matrix(Y_estimativa).transpose()
        erro=np.abs(Ydev-Y_estimativa)   
        media=sum(erro[:,0].transpose().tolist()[0])/erro.shape[0]
        
        '''
        #MSE
        erro=Ydev-Y_estimativa 
        media=erro.transpose()*erro        
        '''        
        
        print lamb,media
        if final[0]>media or final[0]==0:
            final[0]=media
            final[1]=lamb
    temp=X*Xt
         
    shape=temp.shape        
    lambd=sparse.identity(shape[0])*final[1]
    temp=lambd+temp
    temp=temp.todense()
    temp=np.linalg.inv(temp)
    temp=Xt*temp
    temp=temp*Y
  
    #valores returnados: temp->W final, final[0]->erro medio associado a esse valor, final[1]-> lambda
        
    return temp,final[0],final[1]

def erro(ficheiro,w,total):
    dictionary,temp,y=read_output(ficheiro)
    X,Y=criaXY(dictionary,total,y)
       
    
    Y_estimativa=X*w
    Y_estimativa=map(lambda temp: 10**temp[0],Y_estimativa.tolist())
    
    Y_estimativa=np.matrix(Y_estimativa).transpose()
    erroRR=np.abs(Y-Y_estimativa)
    
    #erroRR=sparse.csr_matrix(erroRR)
    #mediaRR=erroRR.transpose()*erroRR
        
    mediaRR=sum(erroRR)/erroRR.shape[0]

        
        
    '''
    for i in erroRR:
        i=i**2
    mediaRR=(sum(erroRR)/len(erroRR))
    '''
    '''    
    estimativa=sum(Y)/len(Y)
    for i in Y:
         i=(i-estimativa)**2
    media=sum(Y)/len(Y)
    '''    
    return mediaRR#,media #mediaRR -> MSE || media->erro de algoritmo parvo
    
    
    
         
if __name__ == '__main__':
    dictionary,total,y=read_output("../le_ficheiro/train.txt")
    
    X,Y=criaXY(dictionary,total,y)
    
    X,total=delcomun(X,total)
   
    #maxi=max(Y)
    #Y=Y/maxi
    Y=np.log10(Y)
    
    dictionaryDev,totalDev,yDev=read_output("../le_ficheiro/dev.txt")
    XDev,YDev=criaXY(dictionaryDev,total,yDev)
    
    #YDev=YDev/maxi
    YDev=np.log10(YDev)
        
    W,erro_dev,lamb=calculaW(X,Y,XDev,YDev)
    print erro("../le_ficheiro/test.txt",W,total)
    #print "lambda:",lamb
    #print "W:"
    #print W
    error=(X*W)-Y
    print "----------------------------------"
    print "funcao de custo:"
    print grad.get_func(error,W,lamb)
    #print "----------------------------------"
    print "lambda:",lamb
    print "erro do teste-----------------------\n", erro("../le_ficheiro/test.txt",W,total)
    print "erro do treino-----------------------\n", erro("../le_ficheiro/train.txt",W,total)
    print "erro do dev-----------------------\n", erro("../le_ficheiro/dev.txt",W,total)
    print "__________________________"
    print "W:"
    print W
    error=(X*W)-Y
    print "----------------------------------"
    print "funcao de custo:"
    print grad.get_func(error,W,lamb)
        
