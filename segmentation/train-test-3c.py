import numpy as np
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
  
def main():
  
  X = np.load('input/X.npy')
  Y = np.load('input/Y.npy') 
  #print("shuffle")
  #X, Y = shuffle(X, Y, random_state = 5564)
  print("train, test, val")
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.15, random_state = 5564)
  X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.15, random_state = 5564)

  print(X_train.shape)
  print(X_val.shape)
  print(X_test.shape)
  print(X.shape)
  del(X, Y)

  print("separate ids")
  v7labs_images = np.load('input/v7labs_images.npy')
  montgomery_images = np.load('input/montgomery_images.npy')
  jsrt_images = np.load('input/jsrt_images.npy')
  shenzhen_images = np.load('input/shenzhen_images.npy')

  shenzhen_test_ids = []
  jsrt_test_ids = []
  montgomery_test_ids = []
  v7labs_test_ids = []
  other_test_ids = []

  nimages = X_test.shape[0]
  for idx in range(nimages):
    test_image = X_test[idx,:,:,0]
    if any(np.array_equal(test_image, x) for x in shenzhen_images):
      shenzhen_test_ids.append(idx)
    elif any(np.array_equal(test_image, x) for x in montgomery_images):
      montgomery_test_ids.append(idx)
    elif any(np.array_equal(test_image, x) for x in jsrt_images):
      jsrt_test_ids.append(idx)
    elif any(np.array_equal(test_image, x) for x in v7labs_images):
      v7labs_test_ids.append(idx)
    else:
      other_test_ids.append(idx)
      
  del(shenzhen_images,montgomery_images,jsrt_images,v7labs_images)
      
  lines = [shenzhen_test_ids, jsrt_test_ids, montgomery_test_ids, v7labs_test_ids, other_test_ids]
  with open('input/test_ids.txt', 'w') as f:
      for line in lines:
          f.write(str(line))
          f.write('\n')

  # ## Save Dataset
  print("saving")
  np.save('input/X_test.npy', X_test)
  np.save('input/Y_test.npy', Y_test)
  np.save('input/X_val.npy', X_val)
  np.save('input/Y_val.npy', Y_val)
  np.save('input/X_train.npy', X_train)
  np.save('input/Y_train.npy', Y_train)


if __name__ == "__main__":
    main()