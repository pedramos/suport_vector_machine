if __name__=="__main__":
    import pickle
    import Rgrad
    import scipy.sparse as sparse
    import pdb
    import matplotlib.pyplot as plt 
    import numpy as np
    file=open("../dados_priberam/dados_X.pkl","rb")
    Xtotal=pickle.load(file)
    file=open("../dados_priberam/dados_Y.pkl","rb")
    Ytotal=pickle.load(file)
    trainsize=int(Xtotal.shape[0]*0.8)
    devsize=int(Xtotal.shape[0]*0.1)+trainsize
    Ytotal=Ytotal.toarray()
    plt.hist(Ytotal,bins=600, normed=False, facecolor='g')
    plt.axis([0,2000, 0, 1200])
    plt.show()
