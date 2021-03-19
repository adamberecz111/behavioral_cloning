import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Convolution2D
from keras.layers import AveragePooling2D

lines = []
with open('./data/driving_log.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

del (lines[0])       
images = []
measurements = []
for row in lines:
    steering_center = float(row[3])

    correction = 0.2
    steering_left = steering_center + correction
    steering_right = steering_center - correction
    
    path = "./data/"
    row[1] = row[1].replace(" ","")
    row[2] = row[2].replace(" ","")
    img_center = cv2.imread(path + row[0])
    img_left = cv2.imread(path + row[1])
    img_right = cv2.imread(path + row[2])
    images.extend([img_center, img_left, img_right])
    measurements.extend([steering_center, steering_left, steering_right])

aug_img, aug_measure = [], []
for image,measurement in zip(images, measurements):
    aug_img.append(image)
    aug_measure.append(measurement)
    aug_img.append(cv2.flip(image,1))
    aug_measure.append(measurement*-1.0)
    
x_train = np.array(aug_img)
y_train = np.array(aug_measure)
model = Sequential()

model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

model.add(Convolution2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(160,320,3)))
model.add(AveragePooling2D())
model.add(Convolution2D(filters=16, kernel_size=(3, 3), activation='relu'))
model.add(AveragePooling2D())
model.add(Flatten())
model.add(Dense(units=120, activation='relu'))
model.add(Dense(units=84, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')