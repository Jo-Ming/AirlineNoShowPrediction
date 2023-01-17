"""
contains a list of methods and functions which can be called. kept here to reduce volume of code in the noShowController file.
"""

import csv
import pandas as pd

def csvToDict(location):
    with open(location, mode='r') as infile:
        reader = csv.reader(infile)
        with open(location, mode='r') as outfile:
            #writer = csv.writer(outfile)
            mydict = {rows[0]:rows[1] for rows in reader}

    return mydict

#returns a dictionary containing the cipherWheels used for encoding.
def readCipherKeys(location):
    dict = {}
    list = []

    #read the csv lines into a list
    with open(location, mode='r') as infile:
        reader = csv.reader(infile)
        with open(location, mode='r') as outfile:
            for row in infile:
                list.append(row[0:-1]) #trim off the /n at the end of each row.
    
    #convert list into a dictionary.
    header = list[0].split(',')
    for i in range(0, len(header)):
        dict.update({header[i]: [list[i*2 + 1].split(',') , list[i*2+2].split(',')]})

    return dict

#can be used to get the model training chord.
def readHeader(location):
    header = []
    #read the csv header
    with open(location, mode='r') as infile:
        reader = csv.reader(infile)
        with open(location, mode='r') as outfile:
            header = next(outfile)
    
    header = header[0:-1] #We can trim off the '\n' that comes at the end of each .csv line string \\\                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
    header = header.split(',') #break the spring up into a list
    return header

def readMeansAndDeviations(location):
    header = []
    indexList = []

    #read the csv lines into a list
    with open(location, mode='r') as infile:
        reader = csv.reader(infile)
        with open(location, mode='r') as outfile:
            for row in infile:
                indexList.append(row[0:-1]) #trim off the /n at the end of each row.

    return indexList

def getStrandFromEntryList(entryList, field):
    strand = []
    for entry in entryList:
        strand.append(entry.getValue(field))

from Tools.dataSpout import dataSpout
from Tools.neuralNetworks import *
from Tools.Model import Model #object will be used to load and train models.
from Tools.Evaluator import Evaluator #will contain code for exectuing evaluations.
import shap
from Tools.Feeder import Feeder
from pathlib import Path #useful for directory management 
from Tools.entryNode import entryNode

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
    #dataSpout.saveDatastreamDictionaryToCSV((dataPath + "/chord.csv")) #save the chord.
    dataSpout.saveCipherKeys((dataPath + "/cipherKeys.csv")) #save the cipher keys the model was trained on.
    dataSpout.saveMeanDeviations((dataPath + "/meanDeviations.csv")) #save parameters for normalising.

#will take a list of dictionarys containing the rowdata and convert them to entryNodes
def dictionaryListToEntryList(self, dictionaryList):
    entryList = []
    for rowData in dictionaryList:
        _entryNode = entryNode(rowData)
        entryList.append(_entryNode)
    return entryList

def encodeStrand(cipherKeyLocation, field, strand):
    encodingDict = readCipherKeys(cipherKeyLocation)
    cipherWheel = encodingDict.get(field)
    encodedStrand = []

    print("Strand encoding: " + str(field) + ":  " + str(cipherWheel))

    for element in strand:
        pointer = getPointer(element, cipherWheel[0])
        encodedStrand.append(pointer)
    
    return encodedStrand

def getPointer( key, valueList):
    pointer = 0
    for value in valueList:
        if value == key:
            return pointer
        else:
            pointer += 1
    print("Error, when encoding entry: " +
              "Unable to find value in the cipherKeyDictionary." + str(key))
    print("It's possible that this value has never been seen by the model.")