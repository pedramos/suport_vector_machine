if __name__=="__main__":
    import pickle
    import Rgrad
    import scipy.sparse as sparse
    import pdb
    import cria_dados
    import Rfechado
    import numpy as np
    import elastic_net
    import Priberam_Ridge as PB

    file=open("../dados_priberam/dados_X_corrigido.pkl","rb")
    Xtotal=pickle.load(file)
    file=open("../dados_priberam/dados_Y_corrigido.pkl","rb")
    Ytotal=pickle.load(file)
    file=open("../dados_priberam/dados_indice_corrigido.pkl","rb")
    indices=pickle.load(file)
    
    trainsize=int(Xtotal.shape[0]*0.8)
    devsize=int(Xtotal.shape[0]*0.1)+trainsize
    
    #Xtrain,Ytrain,Xtest,Ytest,Xdev,Ydev=PB.separaXY(Xtotal,Ytotal)
    #train_index=xrange(trainsize)
    #dev_index=xrange(trainsize,devsize)
    #test_index=xrange(devsize,Xtotal.shape[0])
    
    Xtotal=Xtotal.tocsc()
    Ytotal=Ytotal.tocsc()
    
    Xtotal,indices=cria_dados.delstopword(Xtotal,indices,False)    
    #Xtotal,indices=PB.tira_meta(Xtotal,indices)
    
    #vec=sparse.csr_matrix([0 for i in xrange(Xtrain.shape[1])])
    vec=sparse.csr_matrix([0 for i in xrange(Xtotal[:trainsize,:].shape[1])])
    vec=vec.transpose()
    print "-----------------------------"
    for alpha in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
        print "------------------------------"
        W,F,lamb=elastic_net.elastic_net(Xtotal[:trainsize,:],Ytotal[0:trainsize,0],vec,Xtotal[devsize:,:],Ytotal[devsize:,0],Xtotal[trainsize:devsize,:],Ytotal[trainsize:devsize,0],alpha)
        print "ALPHA", alpha
        print "LAMBDA",lamb
        print "ERRO:",Rfechado.erro(Xtotal[trainsize:devsize,:],Ytotal[trainsize:devsize,:],W)    #W=elastic_net.elastic_net(Xtrain,Ytrain.transpose(),vec,Xdev,Ydev.transpose(),Xtest,Ytest.transpose(),0.5)
        print "ERRO_DEV:",Rfechado.erro(Xtotal[trainsize:devsize,:],Ytotal[trainsize:devsize,:],W)    #W=elastic_net.elastic_net(Xtrain,Ytrain.transpose(),vec,Xdev,Ydev.transpose(),Xtest,Ytest.transpose(),0.5)
        print "ERRO_TEST:",Rfechado.erro(Xtotal[trainsize:devsize,:],Ytotal[trainsize:devsize,:],W) 
 
    #print "ERRO:",Rfechado.erro(Xtest,Ytest,W)
    #print "ERRO:",Rfechado.erro(Xtotal[trainsize:devsize,:],Ytest[trainsize:devsize,:],W)
