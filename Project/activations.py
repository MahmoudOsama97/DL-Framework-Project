def sigmoid(z):


    s = 1/(1+np.exp(-z))

    
    return s
    
def tanh(z):
    

    s = (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
    
    return s
    
def relu(x):
 
    s = np.maximum(0,x)
    
    return s
    
def sigmoid_derivative(x):

    s = sigmoid(x)
    ds = s*(1-s)
 
    
    return ds

def tanh_derivative(x):

    t = tanh(x)
    dt = 1-np.power(t,2)
 
     return dt
     
     

    
def softmax(x):


    x_exp = np.exp(x)

    x_sum = np.sum(x_exp, axis=1, keepdims=True)
 
    s = x_exp/x_sum

  return s
def softmax(X):
    exps = np.exp(X)
    return exps / np.sum(exps)