

class Dense():

    
    def __init__(self, input_units,output_units,activation="ReLU" ,learning_rate=0.01):
        # A dense layer is a layer which performs a learned affine transformation:
        # f(x) = <W*x> + b
        self.activation=activation
        self.learning_rate = learning_rate
        self.weights = np.random.normal(loc=0.0, 
                                        scale = np.sqrt(2/(input_units+output_units)), 
                                        size = (input_units,output_units))
        self.biases = np.zeros(output_units)
        
    def forward(self,input):
        if self.activation=="ReLU":
            forward=np.dot(input,self.weights) + self.biases
            return np.maximum(0,forward)

        elif self.activation=="sigmoid":
            forward=np.dot(input,self.weights) + self.biases
            return  1/(1 + np.exp(-forward))


        elif self.activation=="softmax":
            X=np.dot(input,self.weights) + self.biases

            exps=np.exp(X)
            return exps/np.sum(exps)
         

  #  def cost(self,labels):
   #     cost = -1/m * np.sum(np.multiply(np.log(self.forward(self,input)),labels) +  np.multiply(np.log(1-self.forward(self,input)), (1-labels)))
    
    def backward(self,grad_output,input):
        # compute d f / d x = d f / d dense * d dense / d x
        # where d dense/ d x = weights transposed
        grad_input = np.dot(self.weights,grad_output)
        
        # compute gradient w.r.t. weights and biases
        grad_weights=np.zeros((len(grad_output),len(input)))
        for i in range(len(grad_output)):
          grad_weights[i] =  grad_output[i]*input
        grad_biases = grad_output
        grad_weights=grad_weights.T


        assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape

        # Here we perform a stochastic gradient descent step. 
        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases
        
        return grad_input

class Conv():
    D=1
    def __init__(self,filters=256,n_prev=3,kernel_size=11, strides=1, padding="valid",activation="tanh",learning_rate=0.01):

        self.n_C=filters
        #self.weights = np.random.normal(loc=0.0, 
        #                                scale = np.sqrt(2/(input_units+output_units)), 
        #                                size = (input_units,output_units))
        self.W=np.random.random(size=(kernel_size,kernel_size,n_prev,filters))
        self.b=np.zeros(shape=(1,filters))
        self.stride=strides
        self.learning_rate=learning_rate
        if padding== "valid":
            self.pad=0
        elif padding=="same" : #find equation 
            self.pad=int((kernel_size-1)/2)
        self.f=kernel_size
        self.n_C_prev=n_prev
        self.activation=activation

    def sigmoid(self,z):


        s = 1/(1+np.exp(-z))

        
        return s
        
    def tanh(self,z):
        

        s = (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
        
        return s
        
    def relu(self,x):
     
        s = np.maximum(0,x)
        
        return s
    def sigmoid_derivative(self,x):

        s = self.sigmoid(x)
        ds = s*(1-s)
     
        
        return ds

    def tanh_derivative(self,x):

        t = self.tanh(x)
        dt = 1-np.power(t,2)
     
        return dt

    def relu_derivative(self,x):
     
        return x>0    

    def Activation(self,x):
        if self.activation=="relu":
            self.relu(x)
        if self.activation=="tanh":
            return np.tanh(x)    
        if self.activation=="sigmoid":
            self.sigmoid(x)


    def derivative(self,x):
        if self.activation=="relu":
          return  self.relu_derivative(x)
        if self.activation=="tanh":
           return self.tanh_derivative(x)    
        if self.activation=="sigmoid":
           return self.sigmoid_derivative(x)


    def conv_single_step(self,a_slice_prev, W, b):
    
      s = np.multiply(a_slice_prev, W) + b
      Z = np.sum(s)

      return Z



    def zero_pad(self,X):

        X_pad = np.pad(X, ((self.pad, self.pad), (self.pad,self.pad), (0, 0)), 'constant', constant_values=0)
    
        return X_pad



    def forward(self,A_prev):

        #(m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        ( n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        #(f, f, n_C_prev, n_C) = W.shape

        #stride = hparameters['stride']
        #pad = hparameters['pad']
        
        n_H = int((n_H_prev - self.f + 2 * self.pad) / self.stride) + 1
        n_W = int((n_W_prev - self.f + 2 * self.pad) / self.stride) + 1
        
        Z = np.zeros((n_H, n_W, self.n_C))
        #Z = np.zeros((m, n_H, n_W, n_C))
        A_prev_pad = np.pad(A_prev,((self.pad,self.pad),(self.pad,self.pad),(0,0)), 'constant', constant_values = (0,0))
    #for i in range(m): 
        #a_prev_pad = A_prev_pad[i]                                # loop over the batch of training examples
        a_prev_pad = A_prev_pad                     # Select ith training example's padded activation
        for h in range(n_H):                           # loop over vertical axis of the output volume
            for w in range(n_W):                       # loop over horizontal axis of the output volume
                for c in range(self.n_C):                   # loop over channels (= #filters) of the output volume
                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = h * self.stride
                    vert_end = vert_start + self.f
                    horiz_start = w * self.stride
                    horiz_end = horiz_start + self.f
                    # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    # Convolve the (3D)11 slice with the correct filter W and bias b, to get back one output neuron. (≈1 line)
                    Z[ h, w, c] = self.conv_single_step(a_slice_prev, self.W[:,:,:,c], self.b[:,c])
                    #Z[i, h, w, c] = conv_single_step(a_slice_prev, W[...,c], b[...,c])
        # Making sure your output shape is correct
        assert(Z.shape == (n_H, n_W, self.n_C))
        
        # Save information in "cache" for the backprop
        #cache = (A_prev, W, b, hparameters)
        self.D=Z
        A=self.Activation(Z)

        #return Z, cache
        return A




    def backward(self,dA, A_prev):

    
       
        dZ=dA*self.derivative(self.D)
        # Retrieve information from "cache"
        #(A_prev, W, b, hparameters) = cache
        
        # Retrieve dimensions from A_prev's shape
        #(m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        (n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        # Retrieve dimensions from W's shape
        #(f, f, n_C_prev, n_C) = W.shape
        
        # Retrieve information from "hparameters"
        #stride = hparameters["stride"]
        #pad = hparameters["pad"]
        
        # Retrieve dimensions from dZ's shape
        #(m, n_H, n_W, n_C) = dZ.shape
        ( n_H, n_W, n_C) = dZ.shape
        
        # Initialize dA_prev, dW, db with the correct shapes
        dA_prev = np.zeros((n_H_prev, n_W_prev, n_C_prev))     
        #dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))                      
        dW = np.zeros((self.f, self.f, n_C_prev, n_C))
        db = np.zeros((1, 1, 1, n_C))

        # Pad A_prev and dA_prev
        A_prev_pad = self.zero_pad(A_prev)
        dA_prev_pad = self.zero_pad(dA_prev)
        
    #for i in range(m):                       # loop over the training examples
        
        # select ith training example from A_prev_pad and dA_prev_pad
        a_prev_pad = A_prev_pad
        da_prev_pad = dA_prev_pad
        #a_prev_pad = A_prev_pad[i]
        #da_prev_pad = dA_prev_pad[i]
        for h in range(n_H):                   # loop over vertical axis of the output volume
            for w in range(n_W):               # loop over horizontal axis of the output volume
                for c in range(n_C):           # loop over the channels of the output volume
                    
                    # Find the corners of the current "slice"
                    vert_start = h * self.stride

                    vert_end = vert_start + self.f
                    horiz_start = w * self.stride

                    horiz_end = horiz_start + self.f
                    
                    # Use the corners to define the slice from a_prev_pad
                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    # Update gradients for the window and the filter's parameters using the code formulas given above
                    #da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += self.W[:,:,:,c] * dZ[h, w, c]
                    #dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                    #db[:,:,:,c] += dZ[i, h, w, c]
                    dW[:,:,:,c] += a_slice * dZ[h, w, c]
                    db[:,:,:,c] += dZ[h, w, c]
                    
      
        #dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
        dA_prev=da_prev_pad[self.pad:-self.pad, self.pad:-self.pad, :]
        assert(dA_prev.shape == (n_H_prev, n_W_prev, n_C_prev))
    

        self.W=self.W-self.learning_rate*dW
        self.b=self.b-self.learning_rate*db
        return dA_prev

class Pool():
    def __init__(self,pool_size=2,n_prev=3, strides=2, padding="valid", mode = "max"):

        self.f=pool_size
        self.n_prev=n_prev
        self.stride=strides
        if padding== "valid":
            self.pad=0
        else : #find equation 
            self.pad=np.floor((kernel_size-1)/2)
        self.mode=mode



    def forward(self ,A_prev):
       
       # (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        (n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        #f = hparameters["f"]
        #stride = hparameters["stride"]
        
        # Define the dimensions of the output
        n_H = int(1 + (n_H_prev - self.f) / self.stride)
        n_W = int(1 + (n_W_prev - self.f) / self.stride)
        n_C = n_C_prev
        
        # Initialize output matrix A
        A = np.zeros((n_H, n_W,n_C))              
        #A = np.zeros((m, n_H, n_W, n_C))  
        ### START CODE HERE ###
    #for i in range(m):                           # loop over the training examples
        for h in range(n_H):                     # loop on the vertical axis of the output volume
            for w in range(n_W):                 # loop on the horizontal axis of the output volume
                for c in range (n_C):            # loop over the channels of the output volume
                    
                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = h * self.stride
                    vert_end = vert_start + self.f
                    horiz_start = w * self.stride
                    horiz_end = horiz_start + self.f
                    
                    # Use the corners to define the current slice on the ith training example of A_prev, channel c. (≈1 line)
                    a_prev_slice = A_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                    #a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]
                    # Compute the pooling operation on the slice. Use an if statment to differentiate the modes. Use np.max/np.mean.
                    #if mode == "max":
                    #    A[i, h, w, c] = np.max(a_prev_slice)
                    #elif mode == "average":
                    #    A[i, h, w, c] = np.mean(a_prev_slice)
                    if self.mode == "max":
                        A[h, w, c] = np.max(a_prev_slice)
                    elif self.mode == "average":
                        A[h, w, c] = np.mean(a_prev_slice)
        
        #cache = (A_prev, hparameters)
        
        # Making sure your output shape is correct
        assert(A.shape == (n_H, n_W, n_C))
        
        return A









    def create_mask_from_window(self ,x):

        mask = x == np.max(x)
        
        return mask




    def distribute_value(self ,dz, shape):

        (n_H, n_W) = shape
        
        # Compute the value to distribute on the matrix (≈1 line)
        average = dz / (n_H * n_W)
        
        # Create a matrix where every entry is the "average" value (≈1 line)
        a = np.ones(shape) * average
        ### END CODE HERE ###
        
        return a




    def backward(self ,dA,A_prev):

        #(A_prev, hparameters) = cache
        
        # Retrieve hyperparameters from "hparameters" (≈2 lines)
        #stride = hparameters["stride"]
        #f = hparameters["f"]
        
        # Retrieve dimensions from A_prev's shape and dA's shape (≈2 lines)
        #m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
        #m, n_H, n_W, n_C = dA.shape
        n_H_prev, n_W_prev, n_C_prev = A_prev.shape
        n_H, n_W, n_C = dA.shape
        # Initialize dA_prev with zeros (≈1 line)
        dA_prev = np.zeros(A_prev.shape)
        
    #for i in range(m):                       # loop over the training examples
        # select training example from A_prev (≈1 line)
        a_prev = A_prev
        #a_prev = A_prev[i]
        for h in range(n_H):                   # loop on the vertical axis
            for w in range(n_W):               # loop on the horizontal axis
                for c in range(n_C):           # loop over the channels (depth)
                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = h
                    vert_end = vert_start + self.f
                    horiz_start = w
                    horiz_end = horiz_start + self.f
                    
                    # Compute the backward propagation in both modes.
                    if self.mode == "max":
                        # Use the corners and "c" to define the current slice from a_prev (≈1 line)
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        # Create the mask from a_prev_slice (≈1 line)
                        mask = self.create_mask_from_window(a_prev_slice)
                        # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA) (≈1 line)
                        #dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += np.multiply(mask, dA[i, h, w, c])
                        dA_prev[ vert_start:vert_end, horiz_start:horiz_end, c] += np.multiply(mask, dA[ h, w, c])
                        
                    elif mode == "average":
                        # Get the value a from dA (≈1 line)
                        #da = dA[i, h, w, c]
                        da = dA[h, w, c]
                        # Define the shape of the filter as fxf (≈1 line)
                        shape = (f, f)
                        # Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da. (≈1 line)
                        #dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += distribute_value(da, shape)
                        dA_prev[vert_start:vert_end, horiz_start:horiz_end, c] += self.distribute_value(da, shape)
        
        assert(dA_prev.shape == A_prev.shape)
        
        return dA_prev






class model():


    def __init__(self,data,label):
        self.data=data
        self.label=label
     
    def softmax(self,x):

        x_exp = np.exp(x)
        x_sum = np.sum(x_exp, axis=0, keepdims=True)
        s = x_exp/x_sum
        return s

    def cross_entropy(self,pred,y):
        """
        X is the output from fully connected layer (num_examples x num_classes)
        y is labels (num_examples x 1)
            Note that y is not one-hot encoded vector. 
            It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
        """
        return -np.log(pred[y])  

        



    def delta_cross_entropy(self,inp,pred,label):
        """
        X is the output from fully connected layer (num_examples x num_classes)
        y is labels (num_examples x 1)
            Note that y is not one-hot encoded vector. 
            It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
        """
        delta = np.zeros(len(pred))
        for i in range(len(pred)):
          if i == label:
            delta[i] = -(1 - pred[label])
          else:
            delta[i] = pred[i]
        return delta



    def optimizer(self,batch_size=10,epochs=11):
        datatrain=data
        for x in epochs:
            
            for i in args:
                  datatrain=model.arg[i].forward(datatrain)

            print(cross_entropy(datatrain,label))  
            delta=delta_cross_entropy(datatrain,label)

            for i in args:
                    
                delta=model.arg[i].backward(delta)




C1=Conv(filters=6,n_prev=1,kernel_size=5, strides=1, padding="same",activation="tanh")

S2=Pool(pool_size=2,n_prev=6, strides=2, padding="valid", mode = "max")

FC6=Dense(1176,10, activation="softmax")










grad=0
print(train_labels.shape)
mod=model(train_images,train_images)
for i in range(0,10):
  data=train_images[i].reshape((28,28,1))
  conout=C1.forward(data)
  poolout= S2.forward(conout)
  poolflat=poolout.flatten()
  denseout =FC6.forward(poolflat)
  grad+=mod.delta_cross_entropy(poolflat,denseout,train_labels[i])
  #loss+=cross_entropy(denseout,train_labelsi)
grad=grad/10
print(grad.shape)
for i in range(0,10):
  graddense=FC6.backward(grad,poolflat)
  graddense=graddense.reshape((14,14,6))
  gradpool= S2.backward(graddense,conout)
  X=C1.backward(gradpool,data)