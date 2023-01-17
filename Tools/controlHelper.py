from dataSpout import dataSpout
from neuralNetworks import *
from Model import Model #object will be used to load and train models.
from Evaluator import Evaluator #will contain code for exectuing evaluations.
import shap
from Feeder import Feeder
from pathlib import Path #useful for directory management 

#test1(mctChord, chordToEncode)
def evaluate(): 
    """For this test we are going to:
    1. load a model
    2. evaluate the model """  

    testModel = Model("hughHefty", 'TrainedModels\hughHeftyModel')
    mctEvaluation = Evaluator(testModel)
    mctEvaluation.heftyEvaluation(testModel)

#takes two lists and removes the elements of the second list from the first.
def removeFromChord(chord, removeList):
    for key in removeList:
        pointer = 0
        for note in chord:
            if key == note:
                chord.pop(pointer)
                print(str(key) + " Popped from chord.")
                break
            pointer +=1
    return chord

#will save everything essential to load and use model in the future.
def saveModel(model, dataSpout, saveName):
    #perhaps not so many folders are needed. But i'm just going to keep them seperate for simplicity/clarity.
    savePath = "savedModels/" + saveName
    modelPath = savePath  + "/model"
    dataPath = savePath  + "/data"
    
    #make a new parent directory if it doesn't already exist to save everything inside.
    Path(savePath).mkdir(parents=True, exist_ok=True)

    model.save(modelPath) #save the model
    print("model saved. :)")
  
    #make a folder for the training data. This may not be needed. will hold the training data, chord, and keys
    Path(dataPath).mkdir(parents=True, exist_ok=True)

    dataSpout.saveDataflowToCSV((dataPath + "/dataflow.csv")) #save the dataflow.
    #dataSpout.saveDatastreamDictionaryToCSV((dataPath + "/dataflow.csv")) #save the chord.
    dataSpout.saveCipherKeys((dataPath + "/cipherKeys.csv")) #save the cipher keys the model was trained on.
    dataSpout.saveMeanDeviations((dataPath + "/meanDeviations.csv")) #save parameters for normalising.