#importing layers
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
#importing model type
from keras.models import Sequential

#building model for binary classification
#creating model by adding consecutive convolutional and maxpooling layers
model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D() )
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D() )
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#taking images for training
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

#reading images to train and test the model
training_set = train_datagen.flow_from_directory(
        'train',
        target_size=(150,150),
        batch_size=16,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'test',
        target_size=(150,150),
        batch_size=16,
        class_mode='binary')

#fitting the data to model i.e training the model on input images
model_saved=model.fit_generator(
        training_set,
        epochs=10,
        validation_data=test_set,)

#saving the trained model
model.save('mymaskdetectormodel.h5',model_saved)
