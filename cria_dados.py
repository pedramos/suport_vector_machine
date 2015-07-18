import numpy as np
import scipy.sparse as sparse
import pdb

#####################################################
# DESCRICAO: le o ficheiro "loca" e cria uma        #
#           estrutura que contem a informacao do    #
#           do ficheiro                             #
#              local->ficheiro a ser lido           #
#####################################################

def read_output(local):
    #print "A ler ficheiro:",local
    #interpreta os ficheiros de texto com a contagem das palavras das review, devolve o dictionary, total e y
    import codecs
    
    from collections import deque
    from collections import OrderedDict
    total=OrderedDict() #dicionario com todas as palavras presentes no ficheiro e o seu indice na matriz X
    dictionary={} #dicionario onde sao guardadas as palavras presentes em cada filme 
    y={} #dicionario com as receitas geradas por cada filme
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


#####################################################
# DESCRICAO: utilisa a informacao retornada de      #
#           read_output e cria a matrix X e Y       #
#              dictionary->contem a informacao de X #
#              total->indice das palavras           #
#              y->incoming dos filmes               #
#              standard->boolean, normaizar dados   #
#####################################################

def criaXY(dictionary,total,y,standard):   
    #print "A criar X e Y"
    #calcula o X e Y para as palavras encontradas na review (dictionary), total para indexar as palavras na matriz X(as palavras presentes em dictionary que nao estejam em total nao sao consideradas) e y o lucro de cada filme          
    #print "A criar estrutura "
    X=sparse.dok_matrix((len(dictionary),len(total)),dtype=np.int32)
    #pdb.set_trace()    
    #print "matriz criada"
    Y=sparse.dok_matrix((len(y),1))
    maximo=0
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
                X[indice,total[palavra]]=int(dictionary[movie][palavra])
        
        Y[indice,0]=float(y[movie])
        if np.abs(Y[indice,0])>maximo:
            maximo=np.abs(Y[indice,0])
        X[indice,total["_____"]]=1
        #print "Processamento concluido", indice,movie
	#print X
	#print total
    if standard==True:
        from sklearn import preprocessing


        #Y_test = preprocessing.scale(Y.todense())
        mediaY=sum(Y.toarray())/Y.shape[0]
        Y=Y-mediaY[0]*np.ones_like(Y.todense())
        Y=sparse.csr_matrix(Y)
        
        #std=np.std(Y.toarray())
        #Y=Y/std
        #Y=sparse.dok_matrix(Y)
        #stdY=1
        #X = as_float_array(X, copy)
        #X_mean = sparse.csr_matrix(X,dtype="float").mean(axis=0)
        #X_mean=np.array(X_mean)[0]
        #X=sparse.dok_matrix(X,dtype="float")
        #for indexe in xrange(len(X_mean)):
            #for indexe_linha in xrange(X.shape[0]):
                #X[indexe_linha,indexe]-=X_mean[indexe]
        return X.tocsr(),Y.tocsr() #,mediaY[0] #,mediaY,stdY,X_mean #X e Y sao matrizes esparcas
    else:
         return X.tocsr(),Y.tocsr() #X e Y sao matrizes esparcas

#####################################################
# DESCRICAO: cria uma matrix X e Y atificias e      #
#           aleatorias                              #
#              F->n de features                     #
#              N->n de filmes                       #
#              sigmaXY->desvio padrao da gaussiana  #
#                       usada para criar X e Y      #
#               sigmaN->desvio padrao do ruio usado #
#####################################################

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

#####################################################
# DESCRICAO: cria uma matrix X e Y atificias e      #
#           aleatorias                              #
#              F->n de features (tem que ser pelo   #
#                  menos 10)                        #
#              N->n de filmes                       #
#              sigmaXY->desvio padrao da gaussiana  #
#                       usada para criar X e Y      #
#               sigmaN->desvio padrao do ruio usado #
#####################################################


def XYlasso(F,N,sigmaXY,sigmaN):
     
    W=np.abs(np.random.normal(0,sigmaXY,(F,1)))
    W[5]=0
    W[7]=0
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


#####################################################
# DESCRICAO: apaga as 25 features mais comuns       #
#              X->Matrix X                          #
#              total->indice dos  filmes            #
#####################################################

    
def delcomun(X,n,total):
    #print "A apagar palavras mais comuns"
    import scipy as sc
    import operator
    import collections as cl
    cut=[]
    count=cl.OrderedDict()
    
    for palavra in xrange(X.shape[1]):
        count[palavra]=sum(X[:,palavra].data)

    if n==-1:
        
        pass
    else:
             
        for i in xrange(n):
            maxi=max(count.iteritems(), key=operator.itemgetter(1))[0]
                    
            #pdb.set_trace()
            X=sc.sparse.hstack([X[:,:maxi-1],X[:,maxi:]])
            X=X.tocsc()
            '''
            for cutted in cut:
                if total[maxi]>cutted:
                    x+=1
            X=sc.delete(X,total[maxi]-x,1)
           
            cut.append(total[maxi])
            del count[maxi]
            ''' 
        indices=[]
        apagar=[]
        #sorted(X,reverse=True)[:n]   
        #pdb.set_trace()
        for coiso in sorted(count.itervalues(),reverse=True)[:n]:
            i=0
            for feature in count:
                i+=1
                if count[feature]==coiso:
                    if i not in indices:
                        indices.append(i)
                        break
        for total_i in total:
            for lol in indices:
                if total[total_i]>lol:
                    total[total_i]-=1
                elif total[total_i]==lol:
                    apagar.append(total_i)
        for toremove in apagar:
            del total[toremove]   
    return X,total


#lingua=True -> ingles, False-> portuges
def delstopword(X,indices,lingua):
    from nltk.corpus import stopwords
    import scipy as sc
    print "A RETIRAR STOPWORDS"
    if lingua==True:
        stop=stopwords.words("english")
    else:
        stop=stopwords.words("portuguese")
    apagar=[]
    for word in stop:
       if word in indices:
            apagar.append(word)
            #apagar.append(word,indices(word))
            X=sc.sparse.hstack([X[:,:indices[word]-1],X[:,indices[word]:]])
            X=X.tocsc()
            for palavra in indices:
                if indices[palavra]>indices[word]:
                    indices[palavra]-=1
    for word in apagar:
        del indices[word]
    return X, indices
