#!/usr/bin/env python
# coding: utf-8

# ## LOAD LIBRARIES

# In[17]:


import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import os


# In[18]:


Image_Size = 256
BATCH_SIZE = 32
Channels = 3
EPOCHS =20


# ## This Code Loads the Data Into Tensorflow Database

# In[19]:


Dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "Data",
    shuffle = True,
    image_size=(Image_Size,Image_Size),
    batch_size = BATCH_SIZE
)


# ## Analyse the DATA

# In[20]:


Class_names = Dataset.class_names


# In[21]:


Class_names


# In[22]:


len(Dataset)


# In[23]:


9*32


# In[24]:


plt.figure(figsize=(10,10))
for image_batch, label_batch in Dataset.take(1):
    for i in range(12):
        ax = plt.subplot(3,4,i+1)
    
        #print(image_batch.shape)
        #print(label_batch.numpy())
        #print(image_batch[0].numpy)## Changing tensor to a numpy
        #print(image_batch[0].shape)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.axis("off")
        plt.title(Class_names[label_batch[i]])


# In[25]:


len(Dataset)


# ## Splitting the DATASET

# In[26]:


#80%  ==>  training
#20%  ==>  10% validation, 10% test to measure the accuracy of the model


# ### Using Dataset.take to Split the DATA

# In[27]:


train_size = 0.8
len(Dataset)* train_size # getting the percentage of the train data size from the whole data


# In[28]:


train_ds = Dataset.take(7)
len(train_ds)## Taking the train size percentage from the Data 


# In[29]:


test_ds = Dataset.skip(7)
len(test_ds)## Skipping  the train size to get the test size


# In[30]:


val_size =0.1
len(Dataset)*val_size## splitting the test size into Validation dataset and test dataset


# In[31]:


val_ds= test_ds.take(1)
len(val_ds)


# In[32]:


test_ds = test_ds.skip(1)
len(test_ds)


# ## Splitting the DATASET

# In[33]:


def get_dataset_partitions_tf(ds, train_split = 0.8, val_split =0.1, test_split =0.1, shuffle =True, shuffle_size =10000):
    ds_size = len(ds)
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed = 12)
    train_size =int(train_split *ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)## Taking the train size from the dataset
    val_ds = ds.skip(train_size).take(val_size)# Skip the train_size and the remaining 20% take Val_size
    
    test_ds = val_ds= ds.skip(train_size).skip(val_size)## skip both train and vals_size and the remaining is Test_size
    return train_ds, val_ds, test_ds


# In[34]:


train_ds, val_ds, test_ds = get_dataset_partitions_tf(Dataset)


# In[35]:


len(train_ds)


# In[36]:


len(val_ds)


# In[37]:


len(test_ds)


# ## Caching to improve the performance of the pipeline
# #### Shuffle 1000 will shuffle the images 
# #### Prefetch  to loads the next set of batch from the disk to improve performance 

# In[38]:


train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size =tf.data.AUTOTUNE)

val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size =tf.data.AUTOTUNE)

test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size =tf.data.AUTOTUNE) ## Necessary for training performance


# ### Preprocessing Resizing and Rescaling 

# In[39]:


resize_and_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Rescaling(Image_Size,Image_Size),
    layers.experimental.preprocessing.Rescaling(1.0/255) ## Rescaling the images to 255
])


# ## Creating more samples due to the fewer images to maximize the variables for effective prediction 

# In[40]:


data_augmentaion = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2), ## Randomflip and some rotation to have a diverse forms of the data 
    layers.experimental.preprocessing.RandomZoom(0.1),## to make the model a robust one.
    layers.experimental.preprocessing.RandomContrast(0.1)
])


# ## Build First Classifier (CNN)

# In[41]:


input_shape = (BATCH_SIZE,Image_Size,Image_Size,Channels)
n_classes = 3

model = models.Sequential([
    resize_and_rescale,
    data_augmentaion,
    layers.Conv2D(32,(3,3), activation = 'relu',input_shape =input_shape),## Need to have a lot of layers in order for the prediction to be inact
    layers.MaxPooling2D((2,2)),## this helps to scans over the image to pull out the max values of the image 
    
    layers.Conv2D(64, kernel_size = (3,3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(64, kernel_size = (3,3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(64,(3,3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(64,(3,3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),## flatten
    layers.Dense(64, activation = 'relu'),# and add a densed layer
    layers.Dense(n_classes, activation ='softmax')])# softmax activation fucntion normalizes the probability of the classes

model.build(input_shape = input_shape) ## Force defining the Nero achitecture 


# In[42]:


model.summary()### Module achitecture


# ## Defining the optimizer, loss function and metrics

# In[43]:


model.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False),
    metrics = ['accuracy'])## accuracy is the metric used to track the training process

#callback = keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                       # patience = 3,
                                       # restore_best_weights = True)


# ## Training the Network

# In[44]:


history = model.fit(
    train_ds,
    epochs =EPOCHS,
    batch_size = BATCH_SIZE,
    verbose = 1,
    validation_data = val_ds
)


# ### To define how well the model is performed with a data that hasn't been seen by the model in order to avoid any bias
# 

# In[45]:


scores = model.evaluate(test_ds)## runing the model on the test_ds for the first time (avoid bias)


# In[46]:


scores


# In[47]:


history 


# In[48]:


history.params


# In[49]:


history.history.keys()


# ## Plotting History 

# In[50]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']


# In[51]:


plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.plot(range(EPOCHS), acc, label = 'Training Accuracy')
plt.plot(range(EPOCHS), val_acc, label = 'Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy') ## High accuracy was achieved

#plt.figure(figsize=(8,8))
plt.subplot(1,2,2)
plt.plot(range(EPOCHS), loss, label = 'Training Loss')
plt.plot(range(EPOCHS), val_loss, label = 'Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# In[52]:


np.argmax([1.3001359e-05, 1.9586462e-04, 9.9979120e-01])


# ## Making a Prediction

# In[58]:


plt.figure(figsize=(8,8))
for images_batch, labels_batch in test_ds.take(1):## taking one batch
    first_image = images_batch[4].numpy().astype('uint8')
    first_label = label_batch[4].numpy()
    
    print("First image to Predict")
    plt.imshow(first_image)
    print("Actual Label:",Class_names[first_label])
    
    batch_prediction = model.predict(image_batch)
    print("Predicted Label:",Class_names[np.argmax(batch_prediction[4])])
    plt.axis('off')


# ## Function Determining the Predicted_Class/Confidence_Level of the Model

# In[54]:


def predict(model,img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array = tf.expand_dims(img_array,0)# create batch
    
    predictions = model.predict(img_array)
    
    predicted_class = Class_names[np.argmax(predictions[0])]
    
    Confidence = round(100* (np.max(predictions[0])),2)
    return predicted_class, Confidence


# In[59]:


plt.figure(figsize=(15,15))
for images, labels in test_ds.take(1):## one batch
    for i in range(6):## displaying only 9 images out if the batch
        ax = plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        
        Predicted_Class, Confidence = predict(model, images[i].numpy())
        Actual_Class = Class_names[labels[i]]
        
        plt.title(f"Actual: {Actual_Class}, \n Predicted: {Predicted_Class}.\n Confidence: {Confidence}%")
        
        plt.axis('off')
        plt.savefig('Fig')


# ## Saving the Model

# In[ ]:


model_version = 1
model.save(f"../models/{model_version }")# model.. will take your from present drectory to the new model directory


# In[ ]:


#import os
#model_version = max([int(i) for i in os.listdir("../models")+[0]])+1 # Changing a String to Integer
#model.save(f"../models/{model_version }")


# In[ ]:





# In[ ]:




