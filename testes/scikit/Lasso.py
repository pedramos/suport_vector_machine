import sys
sys.path.append("../../")
import cria_dados
import numpy as np
import pdb
import scipy.io
import coordinate_descent as lm
import sklearn.linear_model as lm

f="../../le_ficheiro/someta"

dictionary, total, y =cria_dados.read_output(f+"train.txt")
X,Y=cria_dados.criaXY(dictionary,total,y,False)
scipy.io.savemat("X.mat",mdict={'X':X})
scipy.io.savemat("Y.mat",mdict={'Y':Y})
clf=lm.Lasso(alpha=100)
cena=clf.fit(X.todense(),Y.todense())
dictionary,temp,y=cria_dados.read_output(f+"test.txt")
Xteste,Yteste=cria_dados.criaXY(dictionary,total,y,False)
YP=cena.predict(Xteste)
erro=np.abs(YP-Yteste)
media=sum(np.array(erro)[0])/erro.shape[0]
print "_____________________________________"
print media
print"______________________________________"
ww=cena.coef_
#print ww
#print ww.shape[0]-len(np.nonzero(ww)[0])

