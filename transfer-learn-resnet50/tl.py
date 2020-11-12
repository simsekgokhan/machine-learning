from os import listdir
from os.path import isfile, join
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ( 
  preprocess_input, decode_predictions
)
from tensorflow.keras.models import load_model

### ---- Global Params
## Configurables
train_the_model  = 1  # 0 to load model from disk
save_the_model   = 0  # save model to disk
test_the_model   = 0
plot_history     = 1
saved_model      = 'trained_resnet50.h5'

# learn_path = 'E:/gsimsek/ML/bed_sofa/learn'
# labels     = ['bed', 'sofa']
# learn_path = 'E:/gsimsek/ML/cats_dogs/learn'
# labels     = ['cat', 'dog']
learn_path = 'E:/gsimsek/ML/bat_emu/learn'
labels     = ['bat', 'emu']

## Non-Configurables
history = None
test_path = learn_path + '/../test'


### ---- 1. Create and configure new model based on ResNet50
def create_model():
  base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top = False)
  # print(base_model.summary())
  # for i, layer in enumerate(base_model.layers): print(i, layer.name)

  output = base_model.output
  ## Condense feature maps from the output
  output = tf.keras.layers.GlobalAveragePooling2D()(output)
  ## Add dense fully connected artificial neural network at the end
  output = tf.keras.layers.Dense(1024, activation='relu')(output)
  output = tf.keras.layers.Dense(1024, activation='relu')(output)
  output = tf.keras.layers.Dense(1024, activation='relu')(output)
  output = tf.keras.layers.Dense(512, activation='relu')(output)
  ## Final layer has 2 output neurons since we're classifying beds and sofas
  final_output = tf.keras.layers.Dense(2, activation ='softmax')(output)
  ## Create our own network/model
  model = tf.keras.models.Model(inputs=base_model.input, outputs=final_output)
  # print(model.summary())

  ## Freeze the layers up to layer 174 (pre-trained layers from ResNet50)
  ## For layer 175 and up we want them trainable
  for layer in model.layers[:175]: layer.trainable = False
  for layer in model.layers[175:]: layer.trainable = True
  return model

### ---- 2. Train
def train(learn_path):
  if train_the_model == 0: return
  model = create_model()
  train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input
  )

  train_generator = train_datagen.flow_from_directory(
    learn_path, 
    target_size = (224, 224),
    color_mode = 'rgb',
    batch_size = 32,
    class_mode = 'categorical',
    shuffle = True
  )

  # Optional. e.g. label_map -> {'bed': 0, 'sofa': 1}
  label_map = (train_generator.class_indices)  
  print('--- Classes found:', label_map)

  ## Train on 5 epochs
  model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
  global history
  history = model.fit_generator(generator = train_generator, epochs = 5)
  # Or, use fit() as framework suggests, but worse accuracy, why?
  # history = model.fit(x = train_generator, batch_size = 32, epochs = 5)

  ## Optionally save the trained model to disk
  if save_the_model: model.save(saved_model) 
  return model


### ---- 3. Predict
def predict(img_path):
  img = image.load_img(img_path)
  img = tf.keras.preprocessing.image.img_to_array(img)
  img = np.expand_dims(img, axis = 0)
  img = tf.keras.applications.resnet50.preprocess_input(img)
  np.set_printoptions(suppress=True) # Suppress scientific notation
  prediction = model.predict(img)
  # print('Prediction:', prediction)
  return prediction

### ---- 4.
def test_model(test_path, label1, label2):
  if test_the_model == 0: return
  label1 = label1[:3]   # Use first three chars
  label2 = label2[:3]
  print('\n--- Testing Model...')
  print('==========================================')
  print(f'Passed  {label1}        {label2}       Image')
  print('-------------------------------------------')
  files = [f for f in listdir(test_path) if isfile(join(test_path, f))]
  passed = 0

  for file in files:
    prediction = predict(test_path + '/' + file)    
    pred1 = prediction[0][0]
    pred2 = prediction[0][1]      
    passed_str = 'No '
    if(file.startswith(label1) and pred1 > pred2): 
      passed_str = 'Yes'
    elif(file.startswith(label2) and pred2 > pred1):
      passed_str = 'Yes'
    passed = passed + 1 if passed_str == 'Yes' else passed     
    print(f"{passed_str}     {pred1:.5f}    {pred2:.5f}   {file}")  
  
  accuracy = passed/len(files)
  print('-------------')
  print('Passed:', passed)
  print('Total :', len(files))
  print('-------------------------------------------')
  print(f'Test Accuracy: {accuracy:.2f}')
  print('-------------------------------------------')

  return accuracy

def plot_training_history():
  global history
  if history is None or plot_history == 0: return
  accuracy = [0, *history.history['accuracy']]
  loss = [0, *history.history['loss']]

  plt.figure()
  plt.plot(accuracy, label='Training Accuracy')
  plt.title('Training Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')

  plt.figure()
  plt.plot(loss, label='Training Loss')
  plt.title('Training Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.show()


### -------------- MAIN
model = train(learn_path) if train_the_model else load_model(saved_model)
plot_training_history()  

predict(img_path = "E:/gsimsek/ML/cats_dogs/val/cat.jpg")

test_model(test_path, labels[0], labels[1])
