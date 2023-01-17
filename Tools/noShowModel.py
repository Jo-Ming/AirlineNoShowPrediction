#This is so we can import from the child directory.
import os, sys

p = os.path.abspath('.')
sys.path.insert(1, p)
from Tools.Model import Model #this contains functionality for the model, such as 
from Tools.entryNode import entryNode
import numpy as np
import shap
#This will load a no show model so we can make requests for prediction

from Tools.Feeder import Feeder #for development to get testing data

def demo_execution():

    #model save location
    currentModelLocation = 'savedModels\\NoShow_2.5M_50Epoch_sigmoid_shuffle1\\model'
    #dataflowSaveLocation 
    dataFlowLocation = 'model\97.8\data\dataflow.csv'

    #here the model will construct itself using the given location.
    noShowModel = Model(name = 'NoShowModel', location = currentModelLocation) # Make sure to use '\\' or '/' to traverse directories.
    
    #noShowModel.shapDemo()


class noShowModel(Model):
        
    #this takes a dictionary (same for,at as the rowDictionary attribute instide an entry node)
    def predictFromDictionary(self, inputDictionary):
        #the input is given in the form of a dictionary.
        #to give it some functionality we can convert it into an entryNode.
        return 0
    
    def takeInput(self, rowDictionary):
        #take thwe dict and make it an entryNode
        entry = entryNode(rowDictionary)
        #show itsekf.
        entry.display()
    
    def shapDemo(self):
        #first, lets get some data. just using mct data for now until I get no show data.
        modelFeeder = Feeder('C:\\Users\\Jo Ming\\Documents\\AirevoWorkspace\\AirevoCode-Space\\NoShowModel\\CSVFiles\\CSVFiles2.csv', 42) #for say 30 values
        dataflow, labels, chord = modelFeeder.loadTrainingData(42) #get dataflow and targetField and applies some data enrichment

        #lets explain som of the models predictions using shap
        
        shapData = np.array(dataflow) #for shap alone we could use pandas.DataFrame() however, keras only takes numpy arrays.
        #shapData = np.array(dataflow[:42]) #for shap alone we could use pandas.DataFrame() however, keras only takes numpy arrays.
        explainer = shap.KernelExplainer(self.model,shapData)#make our explainer
        shap_values = explainer.shap_values(shapData)

        #summarize
        shap.summary_plot(shap_values, shapData, feature_names=chord)

        #try plot with individual points.
        shap_value = shap_values[0]
        inputData = dataflow[0]
        
        shap.decision_plot(explainer.expected_value, shap_value, inputData, feature_names=chord)
        shap.initjs()
        #shap.force_plot(explainer.expected_value, shap_value, inputData, feature_names=chord)
        #shap.waterfall_plot(explainer.expected_value, shap_value, inputData)
    

   