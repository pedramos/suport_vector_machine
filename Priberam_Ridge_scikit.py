if __name__=="__main__":
    import pickle
    import Lasso
    import scipy.sparse as sparse
    import pdb
    import sklearn.linear_model as lm
    file=open("../dados_priberam/dados_X.pkl","rb")
    Xtotal=pickle.load(file)
    file=open("../dados_priberam/dados_Y.pkl","rb")
    Ytotal=pickle.load(file)
    trainsize=int(Xtotal.shape[0]*0.8)
    devsize=int(Xtotal.shape[0]*0.1)+trainsize
    #train_index=xrange(trainsize)
    #dev_index=xrange(trainsize,devsize)
    #test_index=xrange(devsize,Xtotal.shape[0])
    vec=sparse.csr_matrix([0 for i in xrange(Xtotal.shape[1])])
    vec=vec.transpose()
    Xtotal=Xtotal.tocsc()
    Ytotal=Ytotal.tocsc()
    print "A iniciar algoritmo"
    pdb.set_trace()
    clf=lm.Ridge(alpha=100,max_iter=100,fit_intercept=False,normalize=False)
    cena=clf.fit(Xtotal[:trainsize,:].todense(),Ytotal[0:trainsize,0].todense())
    YP=cena.predict(Xtotal[devsize:,0].todense())
    YP=sparse.csr_matrix(YP)
    YP=YP.transpose()
    erro=np.abs(YP-Ytotal[devsize:,0])
    media=sum(np.array(erro.todense()))/erro.shape[0]
    print "_____________________________________"
    print media
    print"______________________________________"

    
    #W,F,lamb=Lasso.lasso(Xtotal[:trainsize,:],Ytotal[0:trainsize,0],vec,Xtotal[devsize:,:],Ytotal[devsize:,0],Xtotal[trainsize:devsize,:],Ytotal[trainsize:devsize,0])
