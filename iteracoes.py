
import numpy as np
import scipy.sparse as sparse
import pdb

#####################################################
# DESCRICAO: calcula o valor da funcao de custo     #
#              error->|X*W-Y|                       #
#              W->                                  #
#              lamb->                               #
#####################################################

def get_func(error,W,lamb): #retorna o valor da funcao para um determinado X,Y,W
    temp=(error.transpose()*error)*0.5
    temp=temp+(W.transpose()*W)*lamb*0.5
    return temp[0,0]
    

#####################################################
# DESCRICAO: calcula o valor da funcao de custo     #
#              do lasso                             #
#              error->|X*W-Y|                       #
#              W->                                  #
#              lamb->                               #
#####################################################

def get_func_lasso(error,W,lamb): #retorna o valor da funcao para um determinado X,Y,W
    #pdb.set_trace()
    
    temp=(error.transpose()*error)*0.5
    #temp=(np.linalg.norm(error)**2)*0.5
    T=(np.abs(W)).sum()
    #print "erro quadratico:",temp
    #print "regularizador",T
    temp=temp[0,0]+T*lamb
    return temp
 
#####################################################
# DESCRICAO: calcula o valor da funcao de custo     #
#              do elastic net                       #
#              error->|X*W-Y|                       #
#              W->                                  #
#              lamb->                               #
#              alpha->percentagem de Ridge/Lasso    #
#####################################################

def get_func_elastic(error,W,lamb,alpha): #retorna o valor da funcao para um determinado X,Y,W
    #pdb.set_trace()
    #temp=(error.transpose()*error)*0.5
    temp=(np.linalg.norm(error)**2)*0.5
    lasso=(np.abs(W)).sum()*alpha
    ridge=(np.linalg.norm(W)**2)*0.5*(1.0-alpha)
    temp=temp+(lasso + ridge)*lamb
    return temp
 
#####################################################
# DESCRICAO: calcula o gradiente (inclui a          #
#           regolarizacao)                          #
#              error->|X*W-Y|                       #
#              X->                                  #
#              W->                                  #
#              lamb->                               #
#####################################################
  
def get_gradient(error,X,W,lamb):  
    
    error=X.transpose()*error
    
    error=error+lamb*W
    
    #print temp

   
    return error
    
    
#####################################################
# DESCRICAO: calcula o alpha usado no sparsa        #
#              w_new->w actual                      #
#              w_old->w da ultima iteracao          #
#              x->X                                 #
#              error->|x*w-Y|                       #
#              lamb->                               #
#####################################################

def get_step(w_new,w_old,x,error_old,error_old_old,lamb):
    '''
    grad_new=get_gradient(error_old,x,w_new,lamb)
    grad_old=get_gradient(error_old_old,x,w_old,lamb)
    R=grad_new-grad_old
    S=w_new-w_old
    A=R.transpose()*R
    B=R.transpose()*S
    if A==0:
        print "**AVISO**:Diferenca entre gradientes =0"
        return 0
    elif B==0:
        print "**AVISO**:Diferenca de gradientes vezes a diferenca de W's deu zero"
        return 0
    else:
        xpto=(A)/(B)
        return xpto[0,0]
        '''
    grad_new=get_gradient(error_old,x,w_new,lamb)
    grad_old=get_gradient(error_old_old,x,w_old,lamb)
    R=grad_new-grad_old
    S=w_new-w_old
    A=S.transpose()*R
    B=S.transpose()*S
    if A==0:
        print "**AVISO**:Diferenca entre gradientes =0"
        return 0
    elif B==0:
        print "**AVISO**:Diferenca de gradientes vezes a diferenca de W's deu zero"
        return 0
    else:
        xpto=(A)/(B)
        return xpto[0,0]

######################################LASSO##########################

#####################################################
# DESCRICAO: calcula o valor do gradiente           #
#           sem a regularizacao (usar no lass       #
#              error->|X*W-Y|                       #
#              X->                                  #
#####################################################

def GetGradL(error,X):
    return X.transpose()*error

#####################################################
# DESCRICAO: funcao soft threshold                  #
#              v-> U caldulado atraves do proximal  #
#                   gradient                        #
#              t->   step*lambda                    #
#####################################################

def soft(v,t):
    
    if v>t:
        return v-t
    elif v>(-t) and v<(t):
        return 0.0
    else:
        return v+t

def softt(U,threshold,zeros):
    Wabs=np.abs(U)
    Wsgn=np.sign(U)
    try:
        #pdb.set_trace()
        joined=np.hstack([zeros,Wabs-threshold])
        U=np.multiply(np.asarray(Wsgn),np.asarray(np.max(joined,axis=1)))
    except:
        pass
        #pdb.set_trace()
    return np.matrix(U)

#####################################################
# DESCRICAO: calcula o valor da funcao de custo     #
#           sem a regularizacao (usar no lass       #
#              error->|X*W-Y|                       #
#              X->                                  #
#####################################################


def getU(wt,step,gradL):
    
    return wt-step*gradL
