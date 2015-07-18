import sys
sys.path.append("../")
import cria_dados
import sklearn.linear_model as lm
import numpy as np
import pdb
f="../../le_ficheiro/someta"

dictionary, total, y =cria_dados.read_output(f+"train.txt")
X,Y=cria_dados.criaXY(dictionary,total,y,False)
clf=lm.Ridge(alpha=0.1)
cena=clf.fit(X.todense(),Y.todense())
dictionary,temp,y=cria_dados.read_output(f+"test.txt")
Xteste,Yteste=cria_dados.criaXY(dictionary,total,y,False)
YP=cena.predict(Xteste)
erro=np.abs(YP-Yteste)
media=sum(np.array(erro.transpose())[0])/erro.shape[0]
print media
