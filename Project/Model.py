class model():

    def __init__(self):
        self.layers = []
        

    def normalization(self,Arr):
      epsilon=0.0000001
      arr=0 
      arr = Arr - Arr.mean(axis=0)
      arr = arr / (np.abs(arr).max(axis=0)+epsilon)
      return arr

    def add(self,Layer):
      self.layers.append(Layer)  

    def accuracy (self,predection,label):
        accuracy=[]
        for i in range(len(predection)):      
            accuracy.append(predection[i]==label[i])
        return sum(accuracy)/len(predection) 

    def evaluate(self, x_test , y_test, loss_type="categorical_crossentropy", batch_size=1):
          lossobj=Loss(loss_type)
          output_layer=[]
          forward_outputs=[]
          test_predection=[]
          flatten_shape=0
          samples=len(x_test)
          batches=int(samples/batchsize)
          print("Testing is running----------------->")
          new_samples=len(x_test)
          test_predection.clear()
          forward_outputs.clear()
          for j in range(batches):       
            for b in range(batchsize):    
              output_layer.clear()     
              forward_input=0    
              forward_input = x_test[j*batchsize+b]
              data=forward_input
              output_layer.append(data)
              # forward propagation       
              for layer in self.layers:
                  if (layer=="flatten"):
                    flatten_shape=forward_input.shape                      
                    forward_input=forward_input.flatten()
                    output_layer.append(forward_input)                                          
                  else:    
                    forward_input = layer.forward(forward_input)
                    output_layer.append(forward_input) 
              forward_outputs.append(np.argmax(output_layer[-1]))
              test_predection.append(output_layer[-1])     
          print("Test_Loss ={}      and      Test_accuracy={}".format(lossobj.loss (test_predection, y_test),self.accuracy(forward_outputs,y_test)))   

    def fit(self, x_train, y_train, loss_type="categorical_crossentropy" ,epochs=0, validation_split=0.1,batchsize=1,plot=1):
        lossobj=Loss(loss_type)
        datalength = len(x_train)
        splitor=int(datalength*validation_split)
        x_validation = x_train[0:splitor]  #as i get the data randamly so i always get from the beggining to the disered length
        y_validation = y_train[0:splitor]
        x_train =      x_train[splitor:]
        y_train =      y_train[splitor:]
        output_layer=[]
        output_batch_layer=[]
        batch_grad=[]
        forward_outputs=[]
        validation_forward_outputs=[]
        train_predection=[]
        validation_predection=[]
        flatten_shape=0
        samples= len(x_train)
        batches=int(samples/batchsize)
        for i in range(epochs):

            print("Training is running----------------->")
            #x_train, y_train=shuffle(x_train,y_train,random_state=0) # re-arrange data randomly   
            new_samples=len(x_train)
            #loss = 0
            validation_grad=0
            train_predection.clear()
            validation_predection.clear()
            validation_forward_outputs.clear()
            forward_outputs.clear()
  
            for j in range(batches):

              output_batch_layer.clear()
              batch_grad.clear()

              for b in range(batchsize):    

                output_layer.clear()     
                forward_input=0    
                forward_input = x_train[j*batchsize+b]
                data=forward_input
                output_layer.append(data)
                # forward propagation       

                for layer in self.layers:

                    if (layer=="flatten"):
                      flatten_shape=forward_input.shape                      
                      forward_input=forward_input.flatten()
                      output_layer.append(forward_input)                                          
                    else:    
                      #if (type(layer) == Conv or Dense ):
                        #norm=np.linalg.norm(forward_input)
                        #forward_input=forward_input/(norm+0.00000001)
                      forward_input = layer.forward(forward_input)
                      output_layer.append(forward_input)
                          
                forward_outputs.append(np.argmax(output_layer[-1]))
                train_predection.append(output_layer[-1])
                grad=lossobj.delta_loss(output_layer[-1],y_train[j*batchsize+b])
                output_batch_layer.append(output_layer)
                batch_grad.append(grad)
                #backward propagation  
              reversed_layers= self.layers[::-1]
              
              for k in range(batchsize):
                back=len(output_batch_layer[k])-2    
                for rlayer in  reversed_layers: 
                    if (rlayer=="flatten"):
                      backward_output=backward_output.reshape(flatten_shape)   
                      batch_grad[k]=backward_output   
                      back=back-1
                    else:
                      backward_output=rlayer.backward(batch_grad[k],output_batch_layer[k][back])
                      batch_grad[k]=backward_output
                      back=back-1

            print("validation is running----------------->")
            for l in range(samples):  
                validation_forward_input= x_validation[l]
                for layer in self.layers:
                  if (layer=="flatten"):            
                      validation_forward_input=validation_forward_input.flatten()           
                  else:
                      validation_forward_input= layer.forward(validation_forward_input)       
                validation_forward_outputs.append(np.argmax(validation_forward_input))
                validation_predection.append(validation_forward_input)
            #validation_loss += lossobj.loss(forward_input,y_train[l])
        
            print("Epoch {}--------------------->".format(i+1))
            print("Loss ={}      and      accuracy={}".format(lossobj.loss (train_predection, y_train[0:samples]),self.accuracy(forward_outputs,y_train[0:samples])))
            print("validation_loss ={}     and      validation_accuracy={}".format( lossobj.loss (validation_predection, y_validation[0:samples]),self.accuracy(validation_forward_outputs,y_validation[0:samples])))
                          