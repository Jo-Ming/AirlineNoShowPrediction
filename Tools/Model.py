import os
import tensorflow as tf
from tensorflow import keras
import matplotlib as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.python.ops.math_ops import argmax
from Tools.modelCompiler import *
import shap


class Model():

    """
    Things to do:
                1. have a cipher key attribute. so we can encode parsed data
                2. have a features attribute which contains the fields the model was trained on.
    """

    def __init__(self, name, location):
        print("Model object has been created.")
        self.name = name
        self.location = location
        self.displayModel()
        self.loadSavedModel()
    
    def getname(self):
        return self.name
    
    def getLocation(self):
        return self.location
        
    def displayModel(self):
        print("Name: " + str(self.name))
        print('location: ' + str(self.location))
    #this will work for a saved tensorflow model.
    def loadSavedModel(self):
        print("Loading model from " + str(self.location) + "...")
        try:
            _model = tf.keras.models.load_model(self.location)
            self.model = _model
            print("Model loaded.")
        except:
            print("Unable to load model at location " + str(self.location))
    
    def shapGraph(self, data):
        shap.initjs() #initialise visualiser
        explainer = shap.TreeExplainer(Model)
        shap_values = explainer.shap_values(data)


    def trainModel(self, trainingData, testingData, trainingClasses, testingClasses, epochs, batch_size, saveName):

        #model = self.model
        history = self.model.fit(trainingData, trainingClasses, epochs = epochs, batch_size = batch_size) #Training our model history is an object that allows us to use callbacks 
        val_loss, val_acc = Model.evaluate(testingData, testingClasses) #testing our model
        print(val_loss)
        print(val_acc)

        print("model trained.")
        
        Model.summary()

        Model.save('trainedModels/' + saveName) #save the model
        print("model saved. :)")
        return history.history

    #predict using given dataflow, returns a list of confidence values for given list.
    def getConfidenceValues(self, dataflow):
        try:
            confidenceValues = self.model.predict(dataflow)
            return confidenceValues
        except:
            print("Unable to predict parsed dataflow using " + str(self.name))
            print("Parsed data shape: " + str(dataflow.shape))
            print("Expected Shape: ToDo, problem is almost always going to be mismatching data shape or normalisation methods not matching.")
        return 0

    #will round confidence values into their classifications for final outputs
    def getBinaryPredictions(model, confidenceValues):
        print("Getting predictions")
        predictedClasses = []
        i = 0
        for confidence in confidenceValues:
            predictedClasses.append(round(confidence[0]))
        return predictedClasses

    #will work for multi class prediction. argmax returns the point with the highest value (which would represent the class catagory)
    def getClassPredictions(model, data):
        confidenceValues = model.predict(data)
        predictedClasses = []
        i = 0
        for confidence in confidenceValues:
            predictedClasses.append(argmax(confidence))
        return predictedClasses
        
    def plotConfusionMatrix(self, testingClasses, predictedClasses, classnames):

        cm = confusion_matrix(testingClasses, predictedClasses)
        print(cm)

        plt.clf()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
        classNames = classnames
        plt.title(str(self.name) + "Confusion Matrix")
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

    def plotAccValLossGraphs(history):
        history.history = history
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
