model_test=model()

model_test.add(Conv(filters=6,n_prev=1,kernel_size=5, strides=1, padding="same",activation="relu"))
model_test.add(Pool(pool_size=2,n_prev=6, strides=2, padding="valid", mode = "max"))
model_test.add(Conv(filters=16,n_prev=1,kernel_size=5, strides=1, padding="valid",activation="relu"))
model_test.add(Pool(pool_size=2,n_prev=16, strides=2, padding="valid", mode = "max"))
model_test.add(Conv(filters=120,n_prev=16,kernel_size=5, strides=1, padding="valid",activation="relu"))
model_test.add("flatten")
model_test.add(Dense(120,84, activation="relu"))
model_test.add(Dense(84,10, activation="softmax"))
train_images = train_images.reshape((60000,28,28,1))


model_test.fit(train_images , train_labels,epochs=3,validation_split=0.1,batchsize=10,plot=0)