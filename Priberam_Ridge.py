def separaXY(X,Y):
    import scipy.sparse as sparse
    i=0
    Xtrain=sparse.csc_matrix((1,1))
    Ytrain=sparse.csr_matrix((1,1))
    Xtest=sparse.csc_matrix((1,1))
    Ytest=sparse.csr_matrix((1,1))
    Xdev=sparse.csc_matrix((1,1))
    Ydev=sparse.csr_matrix((1,1))
    for x in xrange(X.shape[0]):
        
        for xx in xrange(8):
            #pdb.set_trace()
            if Xtrain.shape==(1,1):
                try:
                    Xtrain=X[0,:]
                    Ytrain=Y[0,0]
                except:
                    break
            else:
                try:
                    Xtrain=sparse.vstack((Xtrain,X[xx+i*10,:]))
                    Ytrain=sparse.hstack((Ytrain,Y[xx+i*10,0]))
                except:
                    break
        if Xtest.shape==(1,1):
            Xtest=X[8,:]
            Ytest=Y[8,0]
        else :
            try:
                Xtest=sparse.vstack((Xtest,X[8+i*10,:]))
                Ytest=sparse.hstack((Ytest,Y[8+i*10,0]))
            except:
                break
        
        if Xdev.shape==(1,1):
            Xdev=X[9,:]
            Ydev=Y[9,0]
        else :
            try:
                Xdev=sparse.vstack((Xdev,X[9+i*10,:]))
                Ydev=sparse.hstack((Ydev,Y[9+i*10,0]))
            except:
                break
        i=i+1
    return Xtrain,Ytrain,Xtest,Ytest,Xdev,Ydev

def tira_meta(X,indices):
    import scipy as sc
    import pdb
    print "A RETIRAR OS METADADOS"
    i=0
    apagar=[]
    for feature in indices:
        if feature[len(feature)-1]=="]":
            apagar.append(feature)
    for feature in apagar:
            X=sc.sparse.hstack([X[:,:indices[feature]-1],X[:,indices[feature]:]])
            X=X.tocsc()            
            for indices_i in indices:
                if indices[indices_i]>indices[feature]:
                    indices[indices_i]-=1
    for coisas in apagar:
        del indices[coisas]
    return X, indices


def repara_texto(X,indices):
    import scipy as sc
    X=sc.sparse.hstack([X[:,:indices[">"]-1],X[:,indices[">"]:]])
    del indices[">"]
    X=X.tocsc()            
    X=sc.sparse.hstack([X[:,:indices["["]-1],X[:,indices["["]:]])
    X=X.tocsc()
    del indices["["]
    X=sc.sparse.hstack([X[:,:indices["<"]-1],X[:,indices["<"]:]])
    X=X.tocsc()
    del indices["<"]
    X=sc.sparse.hstack([X[:,:indices["]"]-1],X[:,indices["]"]:]])
    X=X.tocsc()
    del indices["]"]            
    if "[url]" in indices:
        X=sc.sparse.hstack([X[:,:indices["[url]"]-1],X[:,indices["[url]"]:]])
        X=X.tocsc()
        del indices["[url]"]            


    for K in indices:
        if indices[K]>indices[">"]:
            indices[K]-=1            
        if indices[K]>indices["["]:
            indices[K]-=1
        if indices[K]>indices["<"]:
            indices[K]-=1            
        if indices[K]>indices["]"]:
            indices[K]-=1            
        if "[url]" in indices and indices[K]>indices["[url]"]:
            indices[K]-=1            

    return X,indices         


if __name__=="__main__":
    import pickle
    import Rgrad
    import scipy.sparse as sparse
    import pdb
    import cria_dados
    import Rfechado
    import numpy as np
    

    file=open("../dados_priberam/dados_X_corrigido.pkl","rb")
    Xtotal=pickle.load(file)
    file=open("../dados_priberam/dados_Y_corrigido.pkl","rb")
    Ytotal=pickle.load(file)
    file=open("../dados_priberam/dados_indice_corrigido.pkl","rb")
    indices=pickle.load(file)
    trainsize=int(Xtotal.shape[0]*0.8)
    devsize=int(Xtotal.shape[0]*0.1)+trainsize
    #Xtrain,Ytrain,Xtest,Ytest,Xdev,Ydev=separaXY(Xtotal,Ytotal)
    #train_index=xrange(trainsize)
    #dev_index=xrange(trainsize,devsize)
    #test_index=xrange(devsize,Xtotal.shape[0])
    Xtotal=Xtotal.tocsc()
    Xtotal,indices=cria_dados.delstopword(Xtotal,indices,False)
    Xtotal,indices=tira_meta(Xtotal,indices)
    #Xtotal,indices=repara(Xtotal,indices)
    Xtotal=Xtotal.tocsc()
    Ytotal=Ytotal.tocsc()
    #vec=sparse.csr_matrix([0 for i in xrange(Xtrain.shape[1])])
    vec=sparse.csr_matrix([0 for i in xrange(Xtotal[:trainsize,:].shape[1])])
    vec=vec.transpose()
    W,lol1,lol2=Rgrad.grad(Xtotal[:trainsize,:],Ytotal[0:trainsize,0],vec,Xtotal[devsize:,:],Ytotal[devsize:,0],Xtotal[trainsize:devsize,:],Ytotal[trainsize:devsize,0],False,False)
    #W=Rgrad.grad(Xtrain,Ytrain.transpose(),vec,Xdev,Ydev.transpose(),Xtest,Ytest.transpose(),False,False)
    #pdb.set_trace()
    #print "ERRO:",Rfechado.erro(Xtest,Ytest,W)
    print "ERRO:", Rfechado.erro(Xtotal[trainsize:devsize,:],Ytotal[trainsize:devsize,:],W)
    print "__________________"
    print "PIORES 10"
    pdb.set_trace()
    for coiso in sorted(W.toarray())[:10]:
        i=0
        while W[i,0] != coiso:
            i+=1
        else:
            for cenas in indices:
                if indices[cenas]==i:
                    print cenas , coiso, "->",i
                    break
    print "TOP 10"
    for coiso in sorted(W.toarray())[W.shape[1]-11:]:
        i=0
        while W[i,0] != coiso:
            i+=1
        else:
            for cenas in indices:
                if indices[cenas]==i:
                    print cenas , coiso, "->",i
                    break

