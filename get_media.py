import Rfechado
import scipy.sparse as sparse
import numpy
import cria_dados
f="../le_ficheiro/palavra1"
print "FICHEIRO:",f,"\n"

dictionary,total,y=cria_dados.read_output(f+"train.txt")
#X,Y,mediaY,stdY,mediaX=cria_dados.criaXY(dictionary,total,y,True)
X,Y=cria_dados.criaXY(dictionary,total,y,False)

#X,total=cria_dados.delstopword(X,total,True)    
dictionary,temp,y=cria_dados.read_output(f+"test.txt")
Xteste,Yteste=cria_dados.criaXY(dictionary,total,y,False)
dictionary,temp,y=cria_dados.read_output(f+"dev.txt")
Xdev,Ydev=cria_dados.criaXY(dictionary,total,y,False)

media=Y.sum()/Y.shape[0]

print media

erro=numpy.abs(Yteste-media*numpy.ones(Yteste.shape))

print erro.sum()/erro.shape[0]
