import rpy2.robjects as robjects
import sys
sys.path.append("../")
import cria_dados
import sklearn.linear_model as lm
import numpy as np
import pdb
import scipy.io
f="../../le_ficheiro/someta"

dictionary, total, y =cria_dados.read_output(f+"train.txt")
X,Y,media,maximo=cria_dados.criaXY(dictionary,total,y,True)
robjects.r(library(glmnet))
robjects.r(cv.glmnet( X, Y, family="gaussian", alpha=alpha, nlambda=100))
#robjects.r(res <- predict(fits, s=l, type="coefficients"))
