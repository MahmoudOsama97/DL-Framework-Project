

class Dense():

    
    def __init__(self, input_units,output_units,activation="ReLU" ,learning_rate=0.01):
        # A dense layer is a layer which performs a learned affine transformation:
        # f(x) = <W*x> + b
        
        self.learning_rate = learning_rate
        self.weights = np.random.normal(loc=0.0, 
                                        scale = np.sqrt(2/(input_units+output_units)), 
                                        size = (input_units,output_units))
        self.biases = np.zeros(output_units)
        
    def forward(self,input):
        if activation="ReLU":
            forward=np.dot(input,self.weights) + self.biases
            return np.maximum(0,forward)

        elif activation="sigmoid":
            forward=np.dot(input,self.weights) + self.biases
            return  1/(1 + np.exp(-forward))


        elif activation="softmax":
            X=np.dot(input,self.weights) + self.biases

            exps=np.exp(X)
            return exps/np.sum(exps)
         

  #  def cost(self,labels):
   #     cost = -1/m * np.sum(np.multiply(np.log(self.forward(self,input)),labels) +  np.multiply(np.log(1-self.forward(self,input)), (1-labels)))
    
    def backward(self,grad_output,input):
        # compute d f / d x = d f / d dense * d dense / d x
        # where d dense/ d x = weights transposed
        grad_input = np.dot(grad_output, self.weights.T)
        
        # compute gradient w.r.t. weights and biases
        grad_weights = np.dot(input.T, grad_output)
        grad_biases = grad_output.mean(axis=0)*input.shape[0]
        
        assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape
        
        # Here we perform a stochastic gradient descent step. 
        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases
        
        return grad_input

class Conv():
    def __init__(self,filters=256,n_prev=3,kernel_size=11, strides=1, padding="valid",activation="tanh",learning_rate=0.01):

        self.n_C=filters
        self.W=np.random(size=(kernel_size,kernel_size,n_prev,filters))
        self.b=np.zeros(size=(1,filters))
        self.stride=strides
        self.pad=padding
        self.f=kernel_size
        self.n_C_prev=n_prev
        self.activation=activation
        self.D
    def sigmoid(z):


        s = 1/(1+np.exp(-z))

        
        return s
        
    def tanh(self,z):
        

        s = (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
        
        return s
        
    def relu(self,x):
     
        s = np.maximum(0,x)
        
        return s
    def sigmoid_derivative(self,x):

        s = sigmoid(x)
        ds = s*(1-s)
     
        
        return ds

    def tanh_derivative(self,x):

        t = tanh(x)
        dt = 1-np.power(t,2)
     
        return dt

    def relu_derivative(self,x):
     
        return x>0    

    def activation(self,x):
        if activation="relu":
            relu(x)
        if activation="tanh":
            tanh(x)    
        if activation="sigmoid":
            sigmoid(x)


    def derivative(self,x):
        if activation="relu":
            relu_derivative(x)
        if activation="tanh":
            tanh_derivative(x)    
        if activation="sigmoid":
            sigmoid_derivative(x)



    def conv_single_step(a_slice_prev, W, b):
    
    s = np.multiply(a_slice_prev, W) + b
    Z = np.sum(s)

    return Z



    def zero_pad(self,X, pad):

        X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)
    
        return X_pad



    def forward(self,A_prev):

        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        
        #(f, f, n_C_prev, n_C) = W.shape

        #stride = hparameters['stride']
        #pad = hparameters['pad']
        
        n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
        n_W = int((n_W_prev - f + 2 * pad) / stride) + 1
        
        Z = np.zeros((m, n_H, n_W, n_C))
        
        A_prev_pad = zero_pad(A_prev, pad)
        
        for i in range(m):                                 # loop over the batch of training examples
            a_prev_pad = A_prev_pad[i]                     # Select ith training example's padded activation
            for h in range(n_H):                           # loop over vertical axis of the output volume
                for w in range(n_W):                       # loop over horizontal axis of the output volume
                    for c in range(n_C):                   # loop over channels (= #filters) of the output volume
                        # Find the corners of the current "slice" (≈4 lines)
                        vert_start = h * stride
                        vert_end = vert_start + f
                        horiz_start = w * stride
                        horiz_end = horiz_start + f
                        # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
                        a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                        # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈1 line)
                        Z[i, h, w, c] = conv_single_step(a_slice_prev, W[...,c], b[...,c])
                                                
        # Making sure your output shape is correct
        assert(Z.shape == (m, n_H, n_W, n_C))
        
        # Save information in "cache" for the backprop
        #cache = (A_prev, W, b, hparameters)
        D=Z
        A=activation(Z)

        #return Z, cache
        return A




    def backward(self,dA, A_prev):

    

        dZ=dA*derivative(Z)
        # Retrieve information from "cache"
        #(A_prev, W, b, hparameters) = cache
        
        # Retrieve dimensions from A_prev's shape
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        
        # Retrieve dimensions from W's shape
        #(f, f, n_C_prev, n_C) = W.shape
        
        # Retrieve information from "hparameters"
        #stride = hparameters["stride"]
        #pad = hparameters["pad"]
        
        # Retrieve dimensions from dZ's shape
        (m, n_H, n_W, n_C) = dZ.shape
        
        # Initialize dA_prev, dW, db with the correct shapes
        dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))                           
        dW = np.zeros((f, f, n_C_prev, n_C))
        db = np.zeros((1, 1, 1, n_C))

        # Pad A_prev and dA_prev
        A_prev_pad = zero_pad(A_prev, pad)
        dA_prev_pad = zero_pad(dA_prev, pad)
        
        for i in range(m):                       # loop over the training examples
            
            # select ith training example from A_prev_pad and dA_prev_pad
            a_prev_pad = A_prev_pad[i]
            da_prev_pad = dA_prev_pad[i]
            
            for h in range(n_H):                   # loop over vertical axis of the output volume
                for w in range(n_W):               # loop over horizontal axis of the output volume
                    for c in range(n_C):           # loop over the channels of the output volume
                        
                        # Find the corners of the current "slice"
                        vert_start = h * stride

                        vert_end = vert_start + f
                        horiz_start = w * stride

                        horiz_end = horiz_start + f
                        
                        # Use the corners to define the slice from a_prev_pad
                        a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                        # Update gradients for the window and the filter's parameters using the code formulas given above
                        da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                        dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                        db[:,:,:,c] += dZ[i, h, w, c]
                    
      
        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]

    assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
    

    W=W-learning_rate*dW
    b=b-learning_rate*db
    return dA_prev

class Pool():
    def __init__(self,pool_size=2,n_prev=3, strides=2, padding="valid", mode = "max"):

        self.f=pool_size
        self.n_prev=n_prev
        self.stride=strides
        self.padding=padding
        self.mode=mode



    def forward(self ,A_prev):
       
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        
        #f = hparameters["f"]
        #stride = hparameters["stride"]
        
        # Define the dimensions of the output
        n_H = int(1 + (n_H_prev - f) / stride)
        n_W = int(1 + (n_W_prev - f) / stride)
        n_C = n_C_prev
        
        # Initialize output matrix A
        A = np.zeros((m, n_H, n_W, n_C))              
        
        ### START CODE HERE ###
        for i in range(m):                           # loop over the training examples
            for h in range(n_H):                     # loop on the vertical axis of the output volume
                for w in range(n_W):                 # loop on the horizontal axis of the output volume
                    for c in range (n_C):            # loop over the channels of the output volume
                        
                        # Find the corners of the current "slice" (≈4 lines)
                        vert_start = h * stride
                        vert_end = vert_start + f
                        horiz_start = w * stride
                        horiz_end = horiz_start + f
                        
                        # Use the corners to define the current slice on the ith training example of A_prev, channel c. (≈1 line)
                        a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]
                        
                        # Compute the pooling operation on the slice. Use an if statment to differentiate the modes. Use np.max/np.mean.
                        if mode == "max":
                            A[i, h, w, c] = np.max(a_prev_slice)
                        elif mode == "average":
                            A[i, h, w, c] = np.mean(a_prev_slice)
        
        #cache = (A_prev, hparameters)
        
        # Making sure your output shape is correct
        assert(A.shape == (m, n_H, n_W, n_C))
        
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
        m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
        m, n_H, n_W, n_C = dA.shape
        
        # Initialize dA_prev with zeros (≈1 line)
        dA_prev = np.zeros(A_prev.shape)
        
        for i in range(m):                       # loop over the training examples
            # select training example from A_prev (≈1 line)
            a_prev = A_prev[i]
            for h in range(n_H):                   # loop on the vertical axis
                for w in range(n_W):               # loop on the horizontal axis
                    for c in range(n_C):           # loop over the channels (depth)
                        # Find the corners of the current "slice" (≈4 lines)
                        vert_start = h
                        vert_end = vert_start + f
                        horiz_start = w
                        horiz_end = horiz_start + f
                        
                        # Compute the backward propagation in both modes.
                        if mode == "max":
                            # Use the corners and "c" to define the current slice from a_prev (≈1 line)
                            a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                            # Create the mask from a_prev_slice (≈1 line)
                            mask = create_mask_from_window(a_prev_slice)
                            # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA) (≈1 line)
                            dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += np.multiply(mask, dA[i, h, w, c])
                            
                        elif mode == "average":
                            # Get the value a from dA (≈1 line)
                            da = dA[i, h, w, c]
                            # Define the shape of the filter as fxf (≈1 line)
                            shape = (f, f)
                            # Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da. (≈1 line)
                            dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += distribute_value(da, shape)
                            
        
        assert(dA_prev.shape == A_prev.shape)
        
        return dA_pre






class model():


    def __init__(self,*args,data):
        self.args=args
        self.data=data

     
    def softmax(self,x):

        x_exp = np.exp(x)
        x_sum = np.sum(x_exp, axis=1, keepdims=True)
        s = x_exp/x_sum
        return s

    def cross_entropy(self,X,y):
        """
        X is the output from fully connected layer (num_examples x num_classes)
        y is labels (num_examples x 1)
            Note that y is not one-hot encoded vector. 
            It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
        """
        m = y.shape[0]
        p = softmax(X)
        # We use multidimensional array indexing to extract 
        # softmax probability of the correct label for each sample.
        # Refer to https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays for understanding multidimensional array indexing.
        log_likelihood = -np.log(p[range(m),y])
        loss = np.sum(log_likelihood) / m
        return loss   
        def cost
        



    def delta_cross_entropy(X,y):
        """
        X is the output from fully connected layer (num_examples x num_classes)
        y is labels (num_examples x 1)
            Note that y is not one-hot encoded vector. 
            It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
        """
        m = y.shape[0]
        grad = softmax(X)
        grad[range(m),y] -= 1
        grad = grad/m
        return grad



    def optimizer(self,epochs=11):

        for x in epochs:
            
            for i in args:
                    model.arg[i].forward(X)
                        
                delta=delta_cross_entropy()

                for i in args 
                    
                    model.arg[i].backward(delta)

