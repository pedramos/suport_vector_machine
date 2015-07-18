import pdb
import Priberam_Ridge

if __name__=="__main__":
    import pickle
    import Lasso
    import scipy.sparse as sparse
    import Priberam_Ridge
    import Rfechado
    import numpy as np
    import cria_dados

    file=open("../dados_priberam/dados_X_corrigido.pkl","rb")
    Xtotal=pickle.load(file)
    file=open("../dados_priberam/dados_Y_corrigido.pkl","rb")
    Ytotal=pickle.load(file)
    file=open("../dados_priberam/dados_indice_corrigido.pkl","rb")
    indices=pickle.load(file)
    #Xtotal,indices=cria_dados.delstopword(Xtotal,indices,False)    
    #Xtotal,indices=Priberam_Ridge.tira_meta(Xtotal,indices)
    pdb.set_trace()
    trainsize=int(Xtotal.shape[0]*0.8)
    devsize=int(Xtotal.shape[0]*0.1)+trainsize
    Xtotal=Xtotal.tocsc()
    #Xtotal,indices=Priberam_Ridge.repara(Xtotal,indices)
    #train_index=xrange(trainsize)
    #dev_index=xrange(trainsize,devsize)
    #test_index=xrange(devsize,Xtotal.shape[0])
    
    #Xtrain,Ytrain,Xtest,Ytest,Xdev,Ydev=Priberam_Ridge.separaXY(Xtotal,Ytotal)
    vec=sparse.csr_matrix([0 for i in xrange(Xtotal.shape[1])])
    vec=vec.transpose()
   
    Ytotal=Ytotal.tocsc()
    W,F,lamb=Lasso.lasso(Xtotal[:trainsize,:],Ytotal[0:trainsize,0],vec,Xtotal[devsize:,:],Ytotal[devsize:,0],Xtotal[trainsize:devsize,:],Ytotal[trainsize:devsize,0],False,False)
    #W,F,lamb=Lasso.lasso(Xtrain,Ytrain.transpose(),vec,Xdev,Ydev.transpose(),Xtest,Ytest.transpose())
    #print "ERRO:",Rfechado.erro(Xtest,Ytest,W)
    print "ERRO:", Rfechado.erro(Xtotal[trainsize:devsize,:],Ytotal[trainsize:devsize,:],W)
    '''
    for k in  indices:
        if W[indices[k]]==0:
            print k
    '''
    print "PIORES 10"
    for coiso in sorted(np.array(W))[:10]:
        i=0
        while W[i,0] != coiso[0]:
            i+=1
        else:
            for cenas in indices:
                if indices[cenas]==i:
                    print cenas , coiso, "->",i
                    break
    print "TOP 10"
    
    for coiso in sorted(np.array(W),reverse=True)[:10]:
        
        i=0
        while W[i,0] != coiso[0]:
            i+=1
        else:
            for cenas in indices:
                if indices[cenas]==i:
                    print cenas , coiso, "->",i
                    break
    
