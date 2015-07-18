import scipy.io
import scipy.sparse as sparse
import sys
sys.path.append("../")
import cria_dados
import sklearn.linear_model as lm
import numpy as np
import pdb
f="../../le_ficheiro/someta"

dictionary, total, y =cria_dados.read_output(f+"train.txt")
X,Y=cria_dados.criaXY(dictionary,total,y,False)
W=scipy.io.loadmat("W.mat")
W=sparse.csr_matrix(W['W'][0]).transpose()
dictionary,temp,y=cria_dados.read_output(f+"test.txt")
Xteste,Yteste=cria_dados.criaXY(dictionary,total,y,False)
YP=Xteste*W
erro=np.abs(YP-Yteste)
media=sum(erro.toarray())/erro.shape[0]
print "_____________________________________"
print media
print"______________________________________"

