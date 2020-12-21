

# C1 Convolutional Layer
#model.add(layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation=’tanh’, input_shape=(28,28,1), padding=”same”))
C1=Conv(filters=6,n_prev=3,kernel_size=5, strides=1, padding="same",activation="tanh")
# S2 Pooling Layer
#model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding=’valid’))
S2=Pool(pool_size=2,n_prev=6, strides=1, padding="valid", mode = "average")

# C3 Convolutional Layer
#model.add(layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation=’tanh’, padding=’valid’))
C3=Conv(filters=16,n_prev=6,kernel_size=5, strides=1, padding="valid",activation="tanh")
# S4 Pooling Layer
#model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding=’valid’))
S4=Pool(pool_size=2,n_prev=16, strides=2, padding="valid", mode = "average")

# C5 Fully Connected Convolutional Layer
#model.add(layers.Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation=’tanh’, padding=’valid’))

C5=Conv(filters=120,n_prev=16,kernel_size=5, strides=1, padding="valid",activation="tanh")

#Flatten the CNN output so that we can connect it with fully connected layers
#model.add(layers.Flatten())
flat=C5.flatten()
# FC6 Fully Connected Layer
#model.add(layers.Dense(84, activation=’tanh’))
FC6=Dense(flat.size(),84, activation=’tanh’)
#Output Layer with softmax activation
#model.add(layers.Dense(10, activation=’softmax’))
FC6=Dense(84,10, activation=’softmax’)
# Compile the model
list m=[C1,S2,C3,S4,C5]
model.optimize()





    