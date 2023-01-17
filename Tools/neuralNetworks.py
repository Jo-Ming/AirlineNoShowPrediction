import tensorflow as tf
import numpy as np
import time
from sklearn.model_selection import train_test_split #this function is boss
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt #for visualising 
from tensorflow.keras import regularizers
from sklearn.datasets import make_classification
from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
import shap
from keras.layers import Dropout
from keras import optimizers
#from tensorflow.keras.optimizers import SGD

def trainModel1(trainingData, testingData, trainingClasses, testingClasses):

    model = tf.keras.models.Sequential() #we are using a feed forward model.
    model.add(tf.keras.layers.Dense(6, activation= 'relu')) #this is the first dense layer using the recified linear activation function

    model.add(tf.keras.layers.Dense(64, activation= 'relu', kernel_regularizer=regularizers.l2(0.001)))

    model.add(tf.keras.layers.Dense(64, activation= 'softmax'))

    model.add(tf.keras.layers.Dense(1, activation= 'sigmoid')) # output layer using nodes because there are 2 outputs
          
    model.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])

    history = model.fit(trainingData, trainingClasses, epochs = 30, batch_size = 20) #Training our model history is an object that allows us to use callbacks 
    val_loss, val_acc = model.evaluate(testingData, testingClasses) #testing our model
    print(val_loss)
    print(val_acc)
    
    model.summary()

    print("model saved. :)")
    model.save('trainedModels/' + 'model_1') #save the model

    history_dict = history.history
    print(history_dict.keys())

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    #plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def trainModel2(trainingData, testingData, trainingClasses, testingClasses):

    model = tf.keras.models.Sequential() #we are using a feed forward model.
    model.add(tf.keras.layers.Dense(22, activation= 'relu')) #this is the first dense layer using the recified linear activation function

    model.add(tf.keras.layers.Dense(128, activation= 'relu', kernel_regularizer=regularizers.l2(0.001)))

    model.add(tf.keras.layers.Dense(32, activation= 'softmax', kernel_regularizer=regularizers.l2(0.001)))

    model.add(tf.keras.layers.Dense(1, activation= 'sigmoid')) # output layer using nodes because there are 2 outputs
          
    model.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])
    
    history = model.fit(trainingData, trainingClasses, epochs = 30, batch_size = 10) #Training our model history is an object that allows us to use callbacks 
    val_loss, val_acc = model.evaluate(testingData, testingClasses) #testing our model
    print(val_loss)
    print(val_acc)
    
    model.summary()

    print("model saved. :)")
    model.save('trainedModels/' + 'model_1') #save the model

    history_dict = history.history
    print(history_dict.keys())

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    #plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def trainModel3(trainingData, testingData, trainingClasses, testingClasses):

    model = tf.keras.models.Sequential() #we are using a feed forward model.
    model.add(tf.keras.layers.Dense(9, activation= 'relu')) #this is the first dense layer using the recified linear activation function

    model.add(tf.keras.layers.Dense(128, activation= 'relu', kernel_regularizer=regularizers.l2(0.001)))

    model.add(tf.keras.layers.Dense(32, activation= 'relu', kernel_regularizer=regularizers.l2(0.001)))

    model.add(tf.keras.layers.Dense(1, activation= 'sigmoid')) # output layer using nodes because there are 2 outputs
          
    model.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])
    
    history = model.fit(trainingData, trainingClasses, epochs = 30, batch_size = 20) #Training our model history is an object that allows us to use callbacks 
    val_loss, val_acc = model.evaluate(testingData, testingClasses) #testing our model
    print(val_loss)
    print(val_acc)
    
    model.summary()

    print("model saved. :)")
    model.save('trainedModels/' + 'model_1') #save the model

    history_dict = history.history
    print(history_dict.keys())

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    #plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def trainModel3(trainingData, testingData, trainingClasses, testingClasses, epochs, batch_size):

    model = tf.keras.models.Sequential() #we are using a feed forward model.
    model.add(tf.keras.layers.Dense(9, activation= 'relu')) #this is the first dense layer using the recified linear activation function

    model.add(tf.keras.layers.Dense(128, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001)))

    model.add(tf.keras.layers.Dense(32, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001)))

    model.add(tf.keras.layers.Dense(1, activation= 'sigmoid')) # output layer using nodes because there are 2 outputs
          
    model.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])
    
    history = model.fit(trainingData, trainingClasses, epochs = epochs, batch_size = batch_size) #Training our model history is an object that allows us to use callbacks 
    val_loss, val_acc = model.evaluate(testingData, testingClasses) #testing our model
    print(val_loss)
    print(val_acc)
    
    model.summary()

    print("model saved. :)")
    model.save('trainedModels/' + 'model_3') #save the model

    history_dict = history.history
    print(history_dict.keys())

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    #plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    classNames=['Non_MCT_Violation', 'MCT_Violation']

    confidenceValues = model.predict(testingData)
    predictedClasses = []
    i = 0
    for confidence in confidenceValues:
        predictedClasses.append(round(confidence[0]))

    cm = confusion_matrix(testingClasses, predictedClasses)
    print(cm)

    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['Non-MCT_Violation','MCT_Violation']
    plt.title('MCT_Violation Confusion Matrix - Test Data')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    s = [['TrueNeg.\n','FalsePos.\n'], ['FalseNeg.\n', 'TruePos.\n']]
    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
    plt.show()

def trainModel4(trainingData, testingData, trainingClasses, testingClasses, epochs, batch_size):

    model = tf.keras.models.Sequential() #we are using a feed forward model.
    model.add(tf.keras.layers.Dense(19, activation= 'relu')) #this is the first dense layer using the recified linear activation function

    model.add(tf.keras.layers.Dense(128, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001)))

    model.add(tf.keras.layers.Dense(32, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001)))

    model.add(tf.keras.layers.Dense(1, activation= 'sigmoid')) # output layer using nodes because there are 2 outputs
          
    model.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])
    
    history = model.fit(trainingData, trainingClasses, epochs = epochs, batch_size = batch_size) #Training our model history is an object that allows us to use callbacks 
    val_loss, val_acc = model.evaluate(testingData, testingClasses) #testing our model
    print(val_loss)
    print(val_acc)
    
    model.summary()

    print("model saved. :)")
    model.save('trainedModels/' + 'defaultTestModel') #save the model

    history_dict = history.history
    print(history_dict.keys())

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    #plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    classNames=['Non_MCT_Violation', 'MCT_Violation']

    confidenceValues = model.predict(testingData)
    predictedClasses = []
    i = 0
    for confidence in confidenceValues:
        predictedClasses.append(round(confidence[0]))

    cm = confusion_matrix(testingClasses, predictedClasses)
    print(cm)

    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['Non-MCT_Violation','MCT_Violation']
    plt.title('MCT_Violation Confusion Matrix - Test Data')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    s = [['TrueNeg.\n','FalsePos.\n'], ['FalseNeg.\n', 'TruePos.\n']]
    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
    plt.show()

def trainModel5(trainingData, testingData, trainingClasses, testingClasses, epochs, batch_size):

    model = tf.keras.models.Sequential() #we are using a feed forward model.
    model.add(tf.keras.layers.Dense(11, activation= 'relu')) #this is the first dense layer using the recified linear activation function

    model.add(tf.keras.layers.Dense(128, activation= 'relu', kernel_regularizer=regularizers.l2(0.001)))

    model.add(tf.keras.layers.Dense(32, activation= 'relu', kernel_regularizer=regularizers.l2(0.001)))

    model.add(tf.keras.layers.Dense(1, activation= 'sigmoid')) # output layer using nodes because there are 2 outputs
          
    model.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])
    
    history = model.fit(trainingData, trainingClasses, epochs = epochs, batch_size = batch_size) #Training our model history is an object that allows us to use callbacks 
    val_loss, val_acc = model.evaluate(testingData, testingClasses) #testing our model
    print(val_loss)
    print(val_acc)
    
    model.summary()

    print("model saved. :)")
    model.save('trainedModels/' + 'model_1') #save the model

    history_dict = history.history
    print(history_dict.keys())

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    #plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    classNames=['Non_MCT_Violation', 'MCT_Violation']

    confidenceValues = model.predict(testingData)
    predictedClasses = []
    i = 0
    for confidence in confidenceValues:
        predictedClasses.append(round(confidence[0]))

    cm = confusion_matrix(testingClasses, predictedClasses)
    print(cm)

    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['Non-MCT_Violation','MCT_Violation']
    plt.title('MCT_Violation Confusion Matrix - Test Data')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    s = [['TrueNeg.\n','FalsePos.\n'], ['FalseNeg.\n', 'TruePos.\n']]
    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j]) + " = " + str(cm[i][j]))
    plt.show()

def trainModel6(trainingData, testingData, trainingClasses, testingClasses, epochs, batch_size):
    
    model = tf.keras.models.Sequential() #we are using a feed forward model.
    model.add(tf.keras.layers.Dense(19, activation= 'relu')) #this is the first dense layer using the recified linear activation function

    model.add(tf.keras.layers.Dense(512, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001)))

    model.add(tf.keras.layers.Dense(128, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001)))

    model.add(tf.keras.layers.Dense(32, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001))) # output layer using nodes because there are 2 outputs

    model.add(tf.keras.layers.Dense(1, activation= 'sigmoid')) # output layer using nodes because there are 2 outputs
          
    model.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])
    
    history = model.fit(trainingData, trainingClasses, epochs = epochs, batch_size = batch_size) #Training our model history is an object that allows us to use callbacks 
    val_loss, val_acc = model.evaluate(testingData, testingClasses) #testing our model
    print(val_loss)
    print(val_acc)
    
    model.summary()

    print("model saved. :)")
    model.save('trainedModels/' + 'allConnections1') #save the model

    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    #plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    
    classNames=['Non_MCT_Violation', 'MCT_Violation']

    confidenceValues = model.predict(testingData)
    predictedClasses = []
    i = 0
    for confidence in confidenceValues:
        predictedClasses.append(round(confidence[0]))

    cm = confusion_matrix(testingClasses, predictedClasses)
    print(cm)

    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['Non-MCT_Violation','MCT_Violation']
    plt.title('MCT_Violation Confusion Matrix - Test Data')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    s = [['TrueNeg.\n','FalsePos.\n'], ['FalseNeg.\n', 'TruePos.\n']]
    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
    plt.show()

def trainHeftyModel(trainingData, testingData, trainingClasses, testingClasses, epochs, batch_size, saveName):
    
    model = tf.keras.models.Sequential() #we are using a feed forward model.
    model.add(tf.keras.layers.Dense(201, activation= 'relu')) #this is the first dense layer using the recified linear activation function

    model.add(tf.keras.layers.Dense(512, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001)))

    model.add(tf.keras.layers.Dense(1024, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001)))

    model.add(tf.keras.layers.Dense(512, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001))) # output layer using nodes because there are 2 outputs

    model.add(tf.keras.layers.Dense(256, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001))) # output layer using nodes because there are 2 outputs

    model.add(tf.keras.layers.Dense(128, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001))) # output layer using nodes because there are 2 outputs

    model.add(tf.keras.layers.Dense(32, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001))) # output layer using nodes because there are 2 outputs

    model.add(tf.keras.layers.Dense(1, activation= 'sigmoid')) # output layer using nodes because there are 2 outputs
          
    model.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])
    
    history = model.fit(trainingData, trainingClasses, epochs = epochs, batch_size = batch_size) #Training our model history is an object that allows us to use callbacks 
    val_loss, val_acc = model.evaluate(testingData, testingClasses) #testing our model
    print(val_loss)
    print(val_acc)
    
    model.summary()

    print("model saved. :)")
    model.save('trainedModels/' + saveName) #save the model

    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    #plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    
    classNames=['Non_MCT_Violation', 'MCT_Violation']

    confidenceValues = model.predict(testingData)
    predictedClasses = []
    i = 0
    for confidence in confidenceValues:
        predictedClasses.append(round(confidence[0]))

    cm = confusion_matrix(testingClasses, predictedClasses)
    print(cm)

    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['Non-MCT_Violation','MCT_Violation']
    plt.title('MCT_Violation Confusion Matrix - Test Data')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    s = [['TrueNeg.\n','FalsePos.\n'], ['FalseNeg.\n', 'TruePos.\n']]
    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
    plt.show()

#this model is similar to the heftModel but the training data is 'richer'
def trainHughHeftyModel(chord, trainingData, testingData, trainingClasses, testingClasses, epochs, batch_size, saveName):
    
    model = tf.keras.models.Sequential() #we are using a feed forward model.
    model.add(tf.keras.layers.Dense(204, activation= 'relu')) #this is the first dense layer using the recified linear activation function

    model.add(tf.keras.layers.Dense(512, activation= 'relu', kernel_regularizer=regularizers.l2(0.00001)))

    model.add(tf.keras.layers.Dense(1024, activation= 'relu', kernel_regularizer=regularizers.l2(0.00001)))

    model.add(tf.keras.layers.Dense(512, activation= 'relu', kernel_regularizer=regularizers.l2(0.00001))) # output layer using nodes because there are 2 outputs

    model.add(tf.keras.layers.Dense(256, activation= 'relu', kernel_regularizer=regularizers.l2(0.00001))) # output layer using nodes because there are 2 outputs

    model.add(tf.keras.layers.Dense(128, activation= 'relu', kernel_regularizer=regularizers.l2(0.00001))) # output layer using nodes because there are 2 outputs

    model.add(tf.keras.layers.Dense(32, activation= 'relu', kernel_regularizer=regularizers.l2(0.00001))) # output layer using nodes because there are 2 outputs

    model.add(tf.keras.layers.Dense(1, activation= 'sigmoid')) # output layer using nodes because there are 2 outputs
          
    model.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])
    
    history = model.fit(trainingData, trainingClasses, epochs = epochs, batch_size = batch_size) #Training our model history is an object that allows us to use callbacks 
    val_loss, val_acc = model.evaluate(testingData, testingClasses) #testing our model
    print(val_loss)
    print(val_acc)
    
    
    model.summary()

    print("model saved. :)")
    model.save('trainedModels/' + str(saveName)) #save the model

    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    #plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    
    classNames=['Non_MCT_Violation', 'MCT_Violation']

    confidenceValues = model.predict(testingData)
    predictedClasses = []
    i = 0
    for confidence in confidenceValues:
        predictedClasses.append(round(confidence[0]))

    cm = confusion_matrix(testingClasses, predictedClasses)
    print(cm)

    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['Non-MCT_Violation','MCT_Violation']
    plt.title('MCT_Violation Confusion Matrix - Test Data')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    s = [['TrueNeg.\n','FalsePos.\n'], ['FalseNeg.\n', 'TruePos.\n']]
    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
    plt.show()
    

    shap.initjs() #initialise visualiser
    shapData = testingData[0:30]
    explainer = shap.KernelExplainer(model.predict,shapData)#make our explainer
    shap_values = explainer.shap_values(shapData,nsamples=100)
    shap.summary_plot(shap_values,shapData, feature_names=chord)
    #shap.force_plot(explainer.expected_value, shap_values[0], shapData, feature_names=chord)

    return model

#this model is similar to the heftModel but the training data is 'richer'
def trainHughHeftyDropoutModel(chord, trainingData, testingData, trainingClasses, testingClasses, epochs, batch_size, saveName):
    
    model = tf.keras.models.Sequential() #we are using a feed forward model.
    model.add(tf.keras.layers.Dense(199, activation= 'relu')) #this is the first dense layer using the recified linear activation function

    model.add(Dropout(0.15))

    model.add(tf.keras.layers.Dense(512, activation= 'relu'))

    model.add(Dropout(0.15))

    model.add(tf.keras.layers.Dense(1024, activation= 'relu'))

    model.add(Dropout(0.15))

    model.add(tf.keras.layers.Dense(512, activation= 'relu')) # output layer using nodes because there are 2 outputs

    model.add(Dropout(0.15))

    model.add(tf.keras.layers.Dense(256, activation= 'relu')) # output layer using nodes because there are 2 outputs

    model.add(Dropout(0.15))

    model.add(tf.keras.layers.Dense(128, activation= 'relu')) # output layer using nodes because there are 2 outputs

    model.add(Dropout(0.15))

    model.add(tf.keras.layers.Dense(32, activation= 'relu')) # output layer using nodes because there are 2 outputs

    model.add(tf.keras.layers.Dense(1, activation= 'tanh')) # output layer using nodes because there are 2 outputs
          
    model.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])
    
    history = model.fit(trainingData, trainingClasses, epochs = epochs, batch_size = batch_size) #Training our model history is an object that allows us to use callbacks 
    val_loss, val_acc = model.evaluate(testingData, testingClasses) #testing our model
    print(val_loss)
    print(val_acc)
    
    model.summary()

    print("model saved. :)")
    model.save('trainedModels/' + str(saveName)) #save the model

    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    #plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    
    classNames=['Non_MCT_Violation', 'MCT_Violation']

    confidenceValues = model.predict(testingData)
    predictedClasses = []
    i = 0
    for confidence in confidenceValues:
        predictedClasses.append(round(confidence[0]))

    cm = confusion_matrix(testingClasses, predictedClasses)
    print(cm)

    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['Non-MCT_Violation','MCT_Violation']
    plt.title('MCT_Violation Confusion Matrix - Test Data')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    s = [['TrueNeg.\n','FalsePos.\n'], ['FalseNeg.\n', 'TruePos.\n']]
    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
    plt.show()

    shap.initjs() #initialise visualiser
    shapData = testingData[0:30]
    explainer = shap.KernelExplainer(model.predict,shapData)#make our explainer
    shap_values = explainer.shap_values(shapData,nsamples=100)
    shap.summary_plot(shap_values,shapData, feature_names=chord)
    shap.force_plot(explainer.expected_value, shap_values[0], shapData, feature_names=chord)

    return model

#this model is similar to the heftModel but the training data is 'richer'
def tanhModel(chord, trainingData, testingData, trainingClasses, testingClasses, epochs, batch_size, saveName):
    
    model = tf.keras.models.Sequential() #we are using a feed forward model.
    model.add(tf.keras.layers.Dense(107, activation= 'relu')) #this is the first dense layer using the recified linear activation function

    #model.add(Dropout(0.1))

    model.add(tf.keras.layers.Dense(256, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001)))

    #model.add(Dropout(0.1))

    model.add(tf.keras.layers.Dense(512, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001)))

    #model.add(Dropout(0.1))

    model.add(tf.keras.layers.Dense(256, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001))) # output layer using nodes because there are 2 outputs

    #model.add(Dropout(0.1))

    model.add(tf.keras.layers.Dense(128, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001))) # output layer using nodes because there are 2 outputs

    #model.add(Dropout(0.1))

    model.add(tf.keras.layers.Dense(32, activation= 'relu')) # output layer using nodes because there are 2 outputs

    model.add(tf.keras.layers.Dense(1, activation= 'sigmoid')) # output layer using nodes because there are 2 outputs
          
    model.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])
    
    history = model.fit(trainingData, trainingClasses, epochs = epochs, batch_size = batch_size, shuffle = True)
     #Training our model history is an object that allows us to use callbacks 
    val_loss, val_acc = model.evaluate(testingData, testingClasses) #testing our model
    print(val_loss)
    print(val_acc)
    
    model.summary()

    print("model saved. :)")
    model.save('trainedModels/' + str(saveName)) #save the model

    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    #plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    
    classNames=['Non_MCT_Violation', 'MCT_Violation']

    confidenceValues = model.predict(testingData)
    predictedClasses = []
    i = 0
    for confidence in confidenceValues:
        predictedClasses.append(round(confidence[0]))

    cm = confusion_matrix(testingClasses, predictedClasses)
    print(cm)

    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['Non-MCT_Violation','MCT_Violation']
    plt.title('MCT_Violation Confusion Matrix - Test Data')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    s = [['TrueNeg.\n','FalsePos.\n'], ['FalseNeg.\n', 'TruePos.\n']]
    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
    plt.show()

    shap.initjs() #initialise visualiser
    shapData = testingData[0:50]
    explainer = shap.KernelExplainer(model.predict,shapData)#make our explainer
    shap_values = explainer.shap_values(shapData,nsamples=50)
    shap.summary_plot(shap_values,shapData, feature_names=chord)
    shap.force_plot(explainer.expected_value, shap_values[0], shapData, feature_names=chord)

    return model

# This is the slim model but we are keeping certain features in the dark such as minimum connection and time under over
def slimModel(chord, trainingData, testingData, trainingClasses, testingClasses, epochs, batch_size, saveName):
    
    model = tf.keras.models.Sequential() #we are using a feed forward model.
    model.add(tf.keras.layers.Dense(25, activation= 'relu')) #this is the first dense layer using the recified linear activation function

    model.add(tf.keras.layers.Dense(128, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001))) # output layer using nodes because there are 2 outputs
    #model.add(Dropout(0.1))

    model.add(tf.keras.layers.Dense(256, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001)))

    #model.add(Dropout(0.1))

    model.add(tf.keras.layers.Dense(512, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001)))
    #model.add(Dropout(0.1))

    model.add(tf.keras.layers.Dense(256, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001)))

    #model.add(Dropout(0.1))

    model.add(tf.keras.layers.Dense(128, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001))) # output layer using nodes because there are 2 outputs

    #model.add(Dropout(0.1))

    model.add(tf.keras.layers.Dense(32, activation= 'relu')) # output layer using nodes because there are 2 outputs

    model.add(tf.keras.layers.Dense(1, activation= 'sigmoid')) # output layer using nodes because there are 2 outputs
          
    model.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])
    
    history = model.fit(trainingData, trainingClasses, epochs = epochs, batch_size = batch_size)
     #Training our model history is an object that allows us to use callbacks 
    val_loss, val_acc = model.evaluate(testingData, testingClasses) #testing our model
    print(val_loss)
    print(val_acc)
    
    model.summary()

    print("model saved. :)")
    model.save('trainedModels/' + str(saveName)) #save the model

    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    #plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    
    classNames=['Non_MCT_Violation', 'MCT_Violation']

    confidenceValues = model.predict(testingData)
    predictedClasses = []
    i = 0
    for confidence in confidenceValues:
        predictedClasses.append(round(confidence[0]))

    cm = confusion_matrix(testingClasses, predictedClasses)
    print(cm)

    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['Non-MCT_Violation','MCT_Violation']
    plt.title('MCT_Violation Confusion Matrix - Test Data')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    s = [['TrueNeg.\n','FalsePos.\n'], ['FalseNeg.\n', 'TruePos.\n']]
    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
    plt.show()

    shap.initjs() #initialise visualiser
    shapData = testingData[0:30] #get the shap data for the first 100 values. There are 2 mct violations in the first 100 mctExampleModelData
    explainer = shap.KernelExplainer(model.predict,shapData)#make our explainer
    expected_value = explainer.expected_value
    
    shap_values = explainer.shap_values(shapData,nsamples=30)

    shap.summary_plot(shap_values,shapData, max_display=10, feature_names=chord) #this plot shows feature significance

    #shap.decision_plot(expected_value, shap_values[1], shapData[1])

    #shap.force_plot(explainer.expected_value, shap_values[0], shapData[0], feature_names=chord)

    return model

# This is the slim model but we are keeping certain features in the dark such as minimum connection and time under over
def slimShadySGDModel(chord, trainingData, testingData, trainingClasses, testingClasses, epochs, batch_size, saveName):
    
    model = tf.keras.models.Sequential() #we are using a feed forward model.
    model.add(tf.keras.layers.Dense(30, activation= 'relu')) #this is the first dense layer using the recified linear activation function

    model.add(tf.keras.layers.Dense(128, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001))) # output layer using nodes because there are 2 outputs
    #model.add(Dropout(0.1))

    model.add(tf.keras.layers.Dense(256, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001)))

    #model.add(Dropout(0.1))

    model.add(tf.keras.layers.Dense(512, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001)))
    #model.add(Dropout(0.1))

    model.add(tf.keras.layers.Dense(256, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001)))

    #model.add(Dropout(0.1))

    model.add(tf.keras.layers.Dense(128, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001))) # output layer using nodes because there are 2 outputs

    #model.add(Dropout(0.1))

    model.add(tf.keras.layers.Dense(32, activation= 'relu')) # output layer using nodes because there are 2 outputs

    model.add(tf.keras.layers.Dense(1, activation= 'sigmoid')) # output layer using nodes because there are 2 outputs

    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
          
    model.compile(optimizer='sgd',
            loss='binary_crossentropy',
            metrics=['accuracy'])
    
    history = model.fit(trainingData, trainingClasses, epochs = epochs, batch_size = batch_size)
     #Training our model history is an object that allows us to use callbacks 
    val_loss, val_acc = model.evaluate(testingData, testingClasses) #testing our model
    print(val_loss)
    print(val_acc)
    
    model.summary()

    print("model saved. :)")
    model.save('trainedModels/' + str(saveName)) #save the model

    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    #plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    
    classNames=['Non_MCT_Violation', 'MCT_Violation']

    confidenceValues = model.predict(testingData)
    predictedClasses = []
    i = 0
    for confidence in confidenceValues:
        predictedClasses.append(round(confidence[0]))

    cm = confusion_matrix(testingClasses, predictedClasses)
    print(cm)

    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['Non-MCT_Violation','MCT_Violation']
    plt.title('MCT_Violation Confusion Matrix - Test Data')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    s = [['TrueNeg.\n','FalsePos.\n'], ['FalseNeg.\n', 'TruePos.\n']]
    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
    plt.show()

    shap.initjs() #initialise visualiser
    shapData = testingData[0:50]
    explainer = shap.KernelExplainer(model.predict,shapData)#make our explainer
    shap_values = explainer.shap_values(shapData,nsamples=50)
    shap.summary_plot(shap_values,shapData, feature_names=chord)

    #stil havent figured out he force plot
    shap.force_plot(explainer.expected_value, shap_values[0:], testingData[0,:])

    return model
    

def showConfusionMatrix(testingClasses, predictedClasses):
    cm = confusion_matrix(testingClasses, predictedClasses)
    print(cm)

    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['Non-MCT_Violation','MCT_Violation']
    plt.title('MCT_Violation Confusion Matrix - Test Data')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    s = [['TrueNeg.\n','FalsePos.\n'], ['FalseNeg.\n', 'TruePos.\n']]
    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
    plt.show()


