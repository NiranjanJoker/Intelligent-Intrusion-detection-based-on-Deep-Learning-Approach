import os
from os.path import join
from pyexpat import features
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Model, Sequential
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Input, Reshape
from keras.layers import Activation, Dropout, Flatten, Dense
import seaborn as sns
import matplotlib.pyplot as plt
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

#DatasetLocation = 
df = pd.DataFrame(pd.read_csv(r"C:\\Users\\Niranjan\Downloads\\Compressed\\merges-csv-files.csv",low_memory=False))  #this will change the csv dataset to pandas dataframe
df = df.drop(["Timestamp"], axis=1)
df = df.drop(["Dst Port"], axis=1) # drop these 2 columns 

df1 = pd.get_dummies(df["Protocol"]) #one-hot encoding
df1 = pd.concat((df,df1), axis=1)	

df1 = df1.drop(["Protocol"], axis=1)  #drop the protocol feature after one hot encoding it 

features = df1.iloc[:,0:80] #slicing the dataset into features and a label
Y = df1.iloc[:,-1] 
	
features = features.drop(["Label"], axis=1) #dropping the label from the features slice
features = features.replace([np.inf, -np.inf], 0.0) #changing the infs to nans
features = features.fillna(0.0)

subset = ['Flow Duration', 'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow IAT Mean', 'Flow IAT Std',
 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Bwd IAT Std', 'Bwd IAT Max', 'Min Packet Length', 'Max Packet Length',
 'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'Average Packet Size', 'Avg Bwd Segment Size',
 'Init_Win_bytes_forward', 'Active Mean', 'Active Min', 'Idle Mean', 'Idle Max', 'Idle Min']
X_new=features.iloc[:,subset]
inputs= Input(shape=(6,6,1))

#defining the model
x=Conv2D(100, 2, activation = 'relu', padding='same')(inputs)
x=MaxPooling2D((2,2),(2,2))(x)
x=Conv2D(50, 2, activation = 'relu', padding='same')(x)
x=MaxPooling2D()(x)
x=Flatten()(x)
x=Dense(10, activation='relu' )(x)
predictions = Dense(15, activation='softmax')(x)
model = Model(inputs=inputs, outputs=predictions)


kf=KFold(n_splits=3, random_state=1, shuffle=True)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()
result = np.zeros((X_new.shape[0], 36))
result[:, :-5] = X_new

# reshaping dataset
result = np.reshape(result, (result.shape[0], 6, 6))
result = result[..., tensorflow.newaxis]
from keras.utils.np_utils import to_categorical 
# labels are created using values of Y
label = []
for i in Y:
  if i == "BENIGN":
    label.append("0")
  if i == "DoS Hulk":
    label.append("1")
  if i == "PortScan":
    label.append("2")
  if i == "DDoS":
    label.append("3")
  if i == "DoS GoldenEye":
    label.append("4")
  if i == "FTP-Patator":
    label.append("5")
  if i == "SSH-Patator":
    label.append("6")
  if i == "DoS slowloris":
    label.append("7")
  if i == "DoS Slowhttptest":
    label.append("8")
  if i == "Bot":
    label.append("9")
  if i == "Web Attack-Brute Force":
    label.append("10")
  if i == "Web Attack-XSS":
    label.append("11")
  if i == "Infiltration":
    label.append("12")
  if i == "Web Attack-Sql Injection":
    label.append("13")
  if i == "Heartbleed":
    label.append("14")

label=np.asarray(label)
label=to_categorical(label)
X_train, X_test, y_train, y_test = train_test_split( result, label, test_size=0.2, random_state=42)
# training the CNN
from keras.callbacks import ModelCheckpoint
for train_index, test_index in kf.split(X_train):
  training_X,testing_X=X_train[train_index], X_train[test_index]
  training_Y, testing_Y=y_train[train_index], y_train[test_index]
  filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
  checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
  callbacks_list = [checkpoint]
  history=model.fit (X_train,y_train, epochs=5, batch_size=2000, verbose=1, validation_data=(testing_X, testing_Y), callbacks=callbacks_list)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylim(0.6, 1.0)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
model.save('ids-cnn1.h5')
model.load_weights('drive/MyDrive/ids-cnn1.h5')
score = model.evaluate(X_test, y_test, verbose=0)
print(f'Test accuracy: {score[1]}')
