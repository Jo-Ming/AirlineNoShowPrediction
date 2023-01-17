#This is so we can import from the child directory.
import os, sys
from random import Random

p = os.path.abspath('.')
sys.path.insert(1, p)

from Tools.Feeder import Feeder
import shap
import tensorflow as tf
from tensorflow.keras import regularizers
from keras.layers import Dropout
from Tools import controlHelper
from Tools.dataSpout import dataSpout
from Tools.Model import Model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #for visualising 
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle #to randomize the data order when training

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

#identity number
pnrUnique = ['pnr_unique']

#these are all the fields is the chord excluding pnr_unique
chord = [ 'nosho', 'cancelled', 'seg_cancelled', 'pax_cancelled', 'pnr_status', 'no_in_party',
                'domestic_international', 'advance_booking_days', 'class', 'booked_connection_time', 'minimum_connection_time',
                'inbound_arrival_datetime', 'departure_datetime', 'departure_datetime_utc', 'day_of_week', 'board_point',
                'off_point', 'segment_distance', 'inbound_airport', 'inbound_segment_no', 'inbound_route',
                'mkt_carrier_code', 'mkt_flight_no', 'op_carrier_code', 'op_flight_no', 'op_booking_class',
                'equipment', 'gender_code', 'passenger_type_code', 'passenger_type', 'document_birthdate',
                'nosho_type', 'pos_violation', 'group_violation', 'fake_name_violation', 'test_booking', 
                'missing_ttl', 'ttl_incorrect', 'duplicate', 'hidden_group_flag', 'marriage_violation',
                'mct_violation', 'time_under_over', 'fake_name_violation_match', 'fake_name_violation_match_name', 'test_passenger',
                'inbound_arrival_datetime_utc_epoch', 'departure_datetime_epoch', 'inbound_arrival_datetime_utc',
                'departure_datetime_utc',  'departure_datetime_sys']

#first lets seperate the fields into groups.

#targetVariable.
targetField = ['nosho'] 

#The violations found through the current logic approaches.
violationFields = ['pos_violation', 'group_violation', 'fake_name_violation', 'test_booking', 
                'missing_ttl', 'ttl_incorrect', 'duplicate', 'hidden_group_flag', 'marriage_violation',
                'mct_violation', 'fake_name_violation_match', 'fake_name_violation_match_name']
#fields such with strings/classes which contain useful data but arn't yet in a numerical form.
enumerateFields = ['domestic_international', 'class', 'board_point', 'off_point', 'inbound_airport', 'inbound_route',
                     'op_booking_class', 'gender_code', 'passenger_type_code', 'passenger_type']
#status fields (which mostly only have 1 value)
statusFields = ['test_passenger', 'cancelled', 'seg_cancelled', 'pax_cancelled', 'pnr_status']
#These fields are fields that I probably won't use for training a model.
tempEnumFields = ['mkt_carrier_code', 'mkt_flight_no', 'op_carrier_code', 'op_flight_no']
#fields in the form of a string which we want in an integer form.
fieldsToInt = ['no_in_party', 'advance_booking_days', 'day_of_week', 'segment_distance', 'inbound_segment_no','equipment',
             'time_under_over','minimum_connection_time', 'booked_connection_time', 'inbound_arrival_datetime_utc_epoch', 'departure_datetime_epoch']
#dates + time fields
dateTimeFields = ['departure_datetime', 'inbound_arrival_datetime', 'inbound_arrival_datetime_utc', 'departure_datetime_utc',  'departure_datetime_sys']
dateFields = ['document_birthdate']

enrichmentChord = ['InboundToBoarding_Distance', 'UTCdateTimeDifferenceMinutes'] # names of new fields being added
UTCChord = ['UTC_inbound_arrival_datetime', 'UTC_departure_datetime'] # names of new times being added

def main():
    #first lets get the training data
    #dataflow = Feeder('CSVFiles\\CSVFiles2.csv' , 2500000)
    dataflow = Feeder('CSVFiles\\training_data_1k.csv' , 'max')
    dataflow.nodeSlot.entries = shuffle(dataflow.nodeSlot.entries, random_state = 42)
    fullChord = dataflow.getAllTableFields()
    print(fullChord)

    #set the datastream
    dataflow.setDataStream(targetField + enumerateFields + fieldsToInt + dateTimeFields + dateFields)
    dataflow.typeCastChordToInt(fieldsToInt)

    dataflow.distanceTimeEnrichment()

    #evaluate the dataStream
    #dataflow.evaluateDataStream()

    dataflow.encodeChord(enumerateFields + targetField)

    #set the target Label
    dataflow.setTargetDataFromField('nosho')
    #evaluate the targetdata 
    dataflow.evaluateTargetData()
    dataflow.tanhNormaliseChord(enumerateFields + fieldsToInt + dateTimeFields + dateFields + enrichmentChord + UTCChord)

    tensor, targetClasses = dataflow.getDataflowTensor()

    shapChord = dataflow.getDataflowFields()

    trainingData, testingData, trainingClasses, testingClasses = dataflow.splitDataflowAndTargetClasses(tensor, targetClasses)

    #trainingData_shuffled, trainingClasses_shuffled = shuffle(trainingData, trainingClasses, random_state = 5)
    #testingData_shuffled, testingClasses_shuffled = shuffle(testingData, testingClasses, random_state = 5)

    #noShowModel = TrainModel4(trainingData_shuffled, testingData_shuffled, trainingClasses_shuffled, testingClasses_shuffled, 30, 15, "Model4Test")
    noShowModel = TrainModel5(trainingData, testingData, trainingClasses, testingClasses, 75, 35, "exampleModel", shapChord)

    controlHelper.saveModel(noShowModel, dataflow, "exampleSave")

def TrainModel1(trainingData, testingData, trainingClasses, testingClasses, epochs, batch_size, saveName):
    
    #this section is the actual model.
    
    model = tf.keras.models.Sequential() #we are using a feed forward model.
    model.add(tf.keras.layers.Dense(56, activation= 'relu')) #this is the first dense layer using the recified linear activation function

    model.add(tf.keras.layers.Dense(128, activation= 'relu', kernel_regularizer=regularizers.l2(0.0005))) # output layer using nodes because there are 2 outputs

    model.add(tf.keras.layers.Dense(256, activation= 'relu', kernel_regularizer=regularizers.l2(0.0005)))

    model.add(tf.keras.layers.Dense(512, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001)))

    model.add(tf.keras.layers.Dense(512, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001)))

    model.add(tf.keras.layers.Dense(256, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001)))

    model.add(tf.keras.layers.Dense(128, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001))) # output layer using nodes because there are 2 outputs

    model.add(tf.keras.layers.Dense(64, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001)))

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

    classNames=['No Show', 'Show']

    confidenceValues = model.predict(testingData)
    predictedClasses = []
    i = 0
    for confidence in confidenceValues:
        predictedClasses.append(round(confidence[0]))

    cm = confusion_matrix(testingClasses, predictedClasses)
    print(cm)

    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['No Show', 'Show']
    plt.title('No Show Confusion Matrix')
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

    return model

def TrainModel2(trainingData, testingData, trainingClasses, testingClasses, epochs, batch_size, saveName, shapChord):

    #has one less dense layer compared to model 1
    
    #this section is the actual model.
    
    model = tf.keras.models.Sequential() #we are using a feed forward model.
    model.add(tf.keras.layers.Dense(56, activation= 'relu')) #this is the first dense layer using the recified linear activation function

    model.add(tf.keras.layers.Dense(128, activation= 'relu', kernel_regularizer=regularizers.l2(0.000001))) # output layer using nodes because there are 2 outputs

    model.add(tf.keras.layers.Dense(256, activation= 'relu', kernel_regularizer=regularizers.l2(0.000001)))

    model.add(tf.keras.layers.Dense(512, activation= 'relu', kernel_regularizer=regularizers.l2(0.000001)))

    #model.add(tf.keras.layers.Dense(512, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001)))

    model.add(tf.keras.layers.Dense(256, activation= 'relu', kernel_regularizer=regularizers.l2(0.000001)))

    model.add(tf.keras.layers.Dense(128, activation= 'relu', kernel_regularizer=regularizers.l2(0.000001))) # output layer using nodes because there are 2 outputs

    model.add(tf.keras.layers.Dense(64, activation= 'relu', kernel_regularizer=regularizers.l2(0.000001)))

    model.add(tf.keras.layers.Dense(32, activation= 'relu', kernel_regularizer=regularizers.l2(0.000001))) # output layer using nodes because there are 2 outputs

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

    classNames=['No Show', 'Show']

    confidenceValues = model.predict(testingData)
    predictedClasses = []
    i = 0
    for confidence in confidenceValues:
        predictedClasses.append(round(confidence[0]))

    cm = confusion_matrix(testingClasses, predictedClasses)
    print(cm)

    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['No Show', 'Show']
    plt.title('No Show Confusion Matrix')
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
    shap_values = explainer.shap_values(shapData,nsamples=30)
    shap.summary_plot(shap_values,shapData, feature_names=shapChord)

    return model

def TrainModel3(trainingData, testingData, trainingClasses, testingClasses, epochs, batch_size, saveName):
    
    #has one less dense layer compared to model 1
    
    #this section is the actual model.
    
    model = tf.keras.models.Sequential() #we are using a feed forward model.
    model.add(tf.keras.layers.Dense(56, activation= 'relu')) #this is the first dense layer using the recified linear activation function

    model.add(tf.keras.layers.Dense(128, activation= 'relu')) # output layer using nodes because there are 2 outputs

    model.add(tf.keras.layers.Dense(256, activation= 'relu'))

    model.add(tf.keras.layers.Dense(512, activation= 'relu'))

    #model.add(tf.keras.layers.Dense(512, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001)))

    model.add(tf.keras.layers.Dense(256, activation= 'relu'))

    model.add(tf.keras.layers.Dense(128, activation= 'relu')) # output layer using nodes because there are 2 outputs

    model.add(tf.keras.layers.Dense(64, activation= 'relu'))

    model.add(tf.keras.layers.Dense(32, activation= 'relu')) # output layer using nodes because there are 2 outputs

    model.add(tf.keras.layers.Dense(1, activation= 'tanh')) # output layer using nodes because there are 2 outputs
          
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
    plt.title('No Show Confusion Matrix')
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
    shap_values = explainer.shap_values(shapData,nsamples=30)
    shap.summary_plot(shap_values,shapData, feature_names=chord)
    #shap.force_plot(explainer.expected_value, shap_values[0], shapData, feature_names=chord)

    return model

def TrainModel4(trainingData, testingData, trainingClasses, testingClasses, epochs, batch_size, saveName):
        
    #has one less dense layer compared to model 1
    
    #this section is the actual model.
    
    model = tf.keras.models.Sequential() #we are using a feed forward model.
    model.add(tf.keras.layers.Dense(56, activation= 'relu')) #this is the first dense layer using the recified linear activation function

    model.add(tf.keras.layers.Dense(1024, activation= 'relu'))

    model.add(tf.keras.layers.Dense(2048, activation= 'relu')) # output layer using nodes because there are 2 outputs

    model.add(tf.keras.layers.Dense(128, activation= 'relu')) # output layer using nodes because there are 2 outputs

    model.add(tf.keras.layers.Dense(1, activation= 'tanh')) # output layer using nodes because there are 2 outputs
          
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
    plt.title('No Show Confusion Matrix')
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
    shap_values = explainer.shap_values(shapData,nsamples=30)
    shap.summary_plot(shap_values,shapData, feature_names=chord)
    #shap.force_plot(explainer.expected_value, shap_values[0], shapData, feature_names=chord)

    return model

#trying a softmax last layer
def TrainModel5(trainingData, testingData, trainingClasses, testingClasses, epochs, batch_size, saveName, shapChord):
    
    #has one less dense layer compared to model 1
    
    #this section is the actual model.
    
    model = tf.keras.models.Sequential() #we are using a feed forward model.
    model.add(tf.keras.layers.Dense(56, activation= 'relu')) #this is the first dense layer using the recified linear activation function

    model.add(tf.keras.layers.Dense(128, activation= 'relu', kernel_regularizer=regularizers.l2(0.000001))) # output layer using nodes because there are 2 outputs

    model.add(tf.keras.layers.Dense(256, activation= 'relu', kernel_regularizer=regularizers.l2(0.000001)))

    model.add(tf.keras.layers.Dense(512, activation= 'relu', kernel_regularizer=regularizers.l2(0.000001)))

    #model.add(tf.keras.layers.Dense(512, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001)))

    model.add(tf.keras.layers.Dense(256, activation= 'relu', kernel_regularizer=regularizers.l2(0.000001)))

    model.add(tf.keras.layers.Dense(128, activation= 'relu', kernel_regularizer=regularizers.l2(0.000001))) # output layer using nodes because there are 2 outputs

    model.add(tf.keras.layers.Dense(64, activation= 'relu', kernel_regularizer=regularizers.l2(0.000001)))

    model.add(tf.keras.layers.Dense(32, activation= 'relu', kernel_regularizer=regularizers.l2(0.000001))) # output layer using nodes because there are 2 outputs

    model.add(tf.keras.layers.Dense(1, activation= 'softmax')) # output layer using nodes because there are 2 outputs
          
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

    classNames=['No Show', 'Show']

    confidenceValues = model.predict(testingData)
    predictedClasses = []
    i = 0
    for confidence in confidenceValues:
        predictedClasses.append(round(confidence[0]))

    cm = confusion_matrix(testingClasses, predictedClasses)
    print(cm)

    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['No Show', 'Show']
    plt.title('No Show Confusion Matrix')
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
    shap_values = explainer.shap_values(shapData,nsamples=30)
    shap.summary_plot(shap_values,shapData, feature_names=shapChord)

    return model

main()
