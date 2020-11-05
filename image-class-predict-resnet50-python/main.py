### ---- Image class prediction with Keras ---
### Host: Intel Core i7 3.6GHZ, Win7, 16GB RAM 

import os.path, time
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50 # 2.8 sec
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ( 
  preprocess_input, decode_predictions
)

### 1. Load pre-trained model
model = ResNet50(weights='imagenet')  # 2 sec
# From start to here: 4.8 sec

### 2. Predict function (0.1 sec for 200KB image)
def predict(img_path):  
  print('\n******************** predict(): ', img_path)
  start = time.time()
  print('Load & Process image...')
  if not os.path.isfile(img_path):
    print('Err: No such file')
    return

  img = image.load_img(img_path, target_size=(224, 224))
  img_array = image.img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)
  img_array = preprocess_input(img_array)

  print('Predicting...')
  result = model.predict(img_array)
  print('=====================')
  print('Predicted Keras:', decode_predictions(result, top=2)[0])
  print(f'Time spent: {(time.time() - start):-.3} secs')
  print('=====================')
  # 

### 3. Wait & process new job
# Use timestamp or unique ID in production
last_id = '';   # id => filename for simplicity
while True: 
  f = open("file.txt", "r")
  id = f.read()
  if last_id != id: 
    last_id = id
    # print('Processing: ', id)
    predict(id)
    print(f'Waiting for new request...')
  time.sleep(0.9)   # seconds