if __name__=="__main__":
    import pickle
    import Rgrad
    import scipy.sparse as sparse
    import pdb
    import numpy as np
    file=open("../dados_priberam/dados_X_corrigido.pkl","rb")
    Xtotal=pickle.load(file)
    file=open("../dados_priberam/dados_Y_corrigido.pkl","rb")
    Ytotal=pickle.load(file)
    trainsize=int(Xtotal.shape[0]*0.8)
    devsize=int(Xtotal.shape[0]*0.1)+trainsize
    #train_index=xrange(trainsize)
    #dev_index=xrange(trainsize,devsize)
    #test_index=xrange(devsize,Xtotal.shape[0])
    Xtotal=Xtotal.tocsc()
    Ytotal=Ytotal.tocsc()
    coisas = sum(Ytotal[:trainsize,0].toarray())/Ytotal[:trainsize,0].shape[0]
    print coisas
    Y_estimativa= sparse.csr_matrix([coisas for i in xrange(Ytotal[devsize:,0].shape[0])]) #781
    
    
    erroRR=np.abs(Ytotal[devsize:,0]-Y_estimativa)
    print sum(erroRR.toarray())/erroRR.shape[0]
    #W=Rgrad.grad(Xtotal[:trainsize,:],Ytotal[0:trainsize,0],vec,Xtotal[devsize:,:],Ytotal[devsize:,0],Xtotal[trainsize:devsize,:],Ytotal[trainsize:devsize,0])
