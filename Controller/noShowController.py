#This is so we can import from the child directory.
import os, sys

p = os.path.abspath('.')
sys.path.insert(1, p)

#imports
from Tools.Feeder import Feeder
from Tools.noShowModel import noShowModel
from Tools.entryNode import entryNode
from Tools import controlHelper
import controllerHelperFunctions
import numpy as np
from Tools.dataPreprocessor import dataPreprocessor
import shap
import json
from Tools.Evaluator import Evaluator
import ast # used for converting string representation of list into list.
import tensorflow as tf
import operator


"""Useful Things to call"""

#modelLocation = 'savedModels\\model2_randomState42_noshowPredictor'
modelLocation = 'savedModels\\model2_randomState42_noshowPredictor'


def demoExecution():
    #initialise controller
    controller = NoShowController()

    #load a model in.
    controller.loadModel(modelLocation)

    #controller.noShowModel.shapDemo() #show shap stuff

    #get an example entryList.
    entryList = controller.loadRandomEntries(1000, 2,'CSVFiles\\training_data_1k.csv') #list size 1, random seed 1
    #dictList = controller.loadRandomDictList(10, 2,'CSVFiles\\training_data_1k.csv')

    #exampleJSON = '"[{\'pnr_no_unique\': \'JIADCE 2018.01.01\', \'segment_no\': \'1\', \'passenger_no\': \'3\', \'cancelled\': \'N\', \'seg_cancelled\': \'N\', \'pax_cancelled\': \'N\', \'pnr_status\': \'ACTIVE\', \'no_in_party\': \'3\', \'domestic_international\': \'I\', \'advance_booking_days\': \'20\', \'class\': \'R\', \'booked_connection_time\': \'\', \'minimum_connection_time\': \'\', \'inbound_arrival_datetime\': \'\', \'inbound_arrival_datetime_epoch\': \'\', \'inbound_arrival_datetime_utc\': \'\', \'inbound_arrival_datetime_utc_epoch\': \'\', \'departure_datetime\': \'2018-01-22 10:40:00\', \'departure_datetime_utc\': \'2018-01-22 07:40:00\', \'departure_datetime_sys\': \'2018-01-22 02:40:00\', \'departure_datetime_epoch\': \'1516617600\', \'departure_datetime_utc_epoch\': \'1516606800\', \'day_of_week\': \'1\', \'board_point\': \'SVO\', \'off_point\': \'TIV\', \'flight_route\': \'SVO-TIV\', \'segment_distance\': \'2015\', \'inbound_airport\': \'\', \'inbound_segment_no\': \'\', \'inbound_route\': \'\', \'inbound_equipment\': \'\', \'mkt_carrier_code\': \'SU\', \'mkt_flight_no\': \'2050\', \'op_carrier_code\': \'SU\', \'op_flight_no\': \'2050\', \'op_booking_class\': \'R\', \'equipment\': \'59\', \'gender_code\': \'M\', \'passenger_type_code\': \'CHD\', \'passenger_type\': \'C\', \'document_birthdate\': \'2013-08-05\', \'nosho\': \'t\', \'nosho_type\': \'M\', \'pos_violation\': \'\', \'group_violation\': \'N\', \'fake_name_violation\': \'N\', \'test_booking\': \'N\', \'missing_ttl\': \'N\', \'ttl_incorrect\': \'N\', \'duplicate\': \'N\', \'hidden_group_flag\': \'N\', \'marriage_violation\': \'N\', \'mct_violation\': \'N\', \'time_under_over\': \'\', \'fake_name_violation_match\': \'\', \'fake_name_violation_match_name\': \'\', \'test_passenger\': \'N\'}]"'

    #jsonString = controller.ToJsonOutput(dictList)
    #jsonOutput = controller.predictFromJSON(exampleJSON)
    #print(jsonOutput)
    #shapValues = controller.shapValuesFromJSON(exampleJSON)
    #print(shapValues)

    #controller.updateModelWeights(jsonString, 10, 10, "Test_modelUpdate", evalTargets=True, evalDatastream=True)

    #dictListFromJson = controller.jsonToDictList(jsonString)

    #entryListFromJSON = controller.entryListFromJSON(jsonString)

    #shapValues = controller.getShapValuesForEntries(entryList)
    #lets get the confidence values.
    #confidenceValues = controller.getConfidenceList(entryList)

    #print(confidenceValues)

    #get a dictionary containing confidenceValues
    predictionary = controller.getPredictionary2(entryList)

    
    print(controller.sortPredictionary(predictionary))


    #json = controller.getChordJSON()
    #controller.evaluateNoShowModel(entryList, _classnames = ['Show', 'No Show'])

    #these are still in progress...
    #visualise output most significant features for x entries.
    #controller.shapSummaryPlot(entryList)
    #controller.visualiseOutput(entryList)
    #controller.getWaterfallPlot(entryList)

    return 0



# set the controller live
class NoShowController():

    def __innit__(self, modelLocation):
        self.noShowmodel = self.loadModel(modelLocation)
        self.noShowModel = "Empty"

    # loads a set model to prepare for requests.
    def loadModel(self, modelLocation):
        currentModelLocation= modelLocation + '\\model'
        # first, the object must initialise.
        # create a model object so we can interact with model.
        # be sure to use '\\' or '/' when traversing directories.
        _noShowModel = noShowModel(
            "NoShowModel", location=currentModelLocation)
        self.noShowModel =  _noShowModel
        #return _noShowModel
    
    def getModel(self, modelLocation):
        currentModelLocation= modelLocation + '\\model'
        _noShowModel = noShowModel("NoShowModel", location=modelLocation)
        return _noShowModel.model
        

    
    # Main function to call when predicting for a json string containing Entries.
    def predictFromJSON(self, jsonString):
  
        print("Step 1. processing json string into entry List...")
        print("Converting to dictionaryList...")
        dictionaryList = self.jsonToDictList(jsonString) #will contain the rowDictionary data needed for entry nodes.
        print("Complete.")
        print("Generating EntryList for " +str(len(dictionaryList))+" entries...")
        entryNodeList = self.entryNodeListFromDictList(dictionaryList) #entryNodes are ther object types we can interact with.
        print("Complete.")
        print("Step 2 data farmatting, enrichment, and prediction.")
        print("Predicting for entries...")
        predictionary = self.getPredictionary2(entryNodeList) #getPredictionary2 uses a seat level key rather than a booking level key composed of pnr_no_unique + segment_no + passenger_no
        print("Complete.")
        print("Packaging output...")
        jsonOutput = self.ToJsonOutput(predictionary)
        print("Sussessfully predicted json input. ")

        return jsonOutput
    
    #This will return the most influencing features for the given entries. This is function cost grows exponentially, as these values are computed using a game theoretical approach.
    def shapValuesFromJSON(self, jsonString):

        print("Step 1. processing json string into entry List...")
        print("Converting to dictionaryList...")
        dictionaryList = self.jsonToDictList(jsonString) #will contain the rowDictionary data needed for entry nodes.
        print("Complete.")
        print("Generating EntryList for " +str(len(dictionaryList))+" entries...")
        entryNodeList = self.entryNodeListFromDictList(dictionaryList) #entryNodes are ther object types we can interact with.
        print("Complete.")

        print("Retrieveing shap values...")
        shapValues = self.getShapValuesForEntries(entryNodeList)
        print("Complete.")
        print("Packaging output.")
        shapDict = {}
        i = 0
        for entry in entryNodeList:
            shapDict.update({entry.seatKey : shapValues[i]})
            i+= 1
        jsonOutput = self.ToJsonOutput(shapDict)
        print("Successfully retrieved shap values for given entries.")

        return jsonOutput
    
    #here you will parse a json string with a list of entry dictionaries. It will then be processed into a tensor so it can be parsed into a ML model.
    def getDataflowFeederFromJSON(self, jsonString):
        #print("Converting JSON to tensor...")
        print("Converting to dictionaryList...")
        dictionaryList = self.jsonToDictList(jsonString) #will contain the rowDictionary data needed for entry nodes.
        print("Complete.")
        print("Generating EntryList for " +str(len(dictionaryList))+" entries...")
        entryNodeList = self.entryNodeListFromDictList(dictionaryList) #entryNodes are ther object types we can interact with.
        print("Processing and encoding data...")
        _dataPreprocessor = self.processEntryList(entryNodeList) # data is now processed and enriched.
        if _dataPreprocessor == None:
            print("Error, Failed to process data.")
            return 0
        print("Complete")
        print("Normalising the Data...")
        #now we need to normalise the dataStreamDict
        #first let get the means and meanDeviations
        meansAndDeviations = controllerHelperFunctions.readMeansAndDeviations(modelLocation + "\\data\\meanDeviations.csv")
        #and normalise
        _dataPreprocessor.tanhNormaliseDataStream(meansAndDeviations)
        print("Complete")
    
        return _dataPreprocessor
    
    def getDataflowFeederFromJSONWithTargetData(self, jsonString):
        #print("Converting JSON to tensor...")
        print("Converting to dictionaryList...")
        dictionaryList = self.jsonToDictList(jsonString) #will contain the rowDictionary data needed for entry nodes.
        print("Complete.")
        print("Generating EntryList for " +str(len(dictionaryList))+" entries...")
        entryNodeList = self.entryNodeListFromDictList(dictionaryList) #entryNodes are ther object types we can interact with.
        print("Processing and encoding data...")
        _dataPreprocessor = self.processEntryListWithTargetData(entryNodeList) # data is now processed and enriched.
        if _dataPreprocessor == None:
            print("Error, Failed to process data.")
            return 0
        print("Complete")
        print("Normalising the Data...")
        #now we need to normalise the dataStreamDict
        #first let get the means and meanDeviations
        meansAndDeviations = controllerHelperFunctions.readMeansAndDeviations(modelLocation + "\\data\\meanDeviations.csv")
        #and normalise
        _dataPreprocessor.tanhNormaliseDataStream(meansAndDeviations)
        print("Complete")
    
        return _dataPreprocessor


    
    #will take the loaded model and train on top of its loaded weights.
    #eval targets and eval datastream will evaluate the data if True
    def updateModelWeights(self, JSONString, epochs, batch_size, saveName, evalTargets = True, evalDatastream = True):
        #take the loaded model.
        modelDirectory = self.noShowModel
        loaded_model = tf.keras.models.load_model(self.noShowModel.location)

        print("Creating the models data feeder...")

        modelFeeder = self.getDataflowFeederFromJSONWithTargetData(JSONString)
        #set the target Label. Data must be labeled so be sure a field is present.
        print("Setting target data...")
        modelFeeder.setTargetDataFromField('nosho')
        print("target data set.")
        dataStreamChord = modelFeeder.getDataflowFields()
        trainingChord = self.getModelTrainingChord()
        print("Full Feeder Chord = " + str(dataStreamChord))
        print("Original model training chord = " + str(trainingChord))
        if(dataStreamChord != trainingChord):
            print("Error: feeder chord does not match the models training chord.")
            return 0

        #perform evaluation if parameters are set true.
        #although the evaluation is performed after the nullToZero function is called so if we want to see the true null percentage
        #we should call this function straingt after the datastream is set. or before the null to zero is called .
        if evalTargets == True:
            print("evaluating Target data")
            modelFeeder.evaluateTargetData()
        if evalDatastream == True:
            print("Evaluating the dataflow.")
            modelFeeder.evaluateDataStream()
        
        #now that the data should be enriched, formatted, encoded, and normalised. The tensor can be retrieved.
        tensor = modelFeeder.getDataflowTensor()
        targetClasses = modelFeeder.targetData

        print("splitting tensor into 80:20 training and testing data ")
        trainingData, testingData, trainingClasses, testingClasses = modelFeeder.splitDataflowAndTargetClasses(tensor, targetClasses)

        print("Beginning model training...")
        history = loaded_model.fit(trainingData, trainingClasses, epochs = epochs, batch_size = batch_size)
        print("Training complete.")
        #Training our model history is an object that allows us to use callbacks 
        val_loss, val_acc = loaded_model.evaluate(testingData, testingClasses) #testing our model
        print(val_loss)
        print(val_acc)
    
        loaded_model.summary()

        print("saving model...")
        loaded_model.save('trainedModels/' + str(saveName)) #save the model on its own somewhere

        #for the controlHelper.savModel to work the feeder.meansAndDeviations and feeder.fieldkeys need to be in the right format.
        #implement quickfix to format these fields
        modelFeeder = self.quickfixFormat(modelFeeder)
        controlHelper.saveModel(loaded_model, modelFeeder, saveName) #full save with directory structure 
        print("model saved. :)")
           
        
        return 0       
    
    #I set the attributes into the wrong format so the controlHelper.savemodel() wont work
    #this definetely will add a little bit of time to the saving process but probably not a significant amount of time.
    # if this is to be fixed in the future inside the updateModelWeights() function make sure the fieldKeys and meansAndDeviations are in the format stated below from when they are read from a saved model.
    def quickfixFormat(self, feeder):
        #first the self. fieldkeys are in format: {field:[[keys],[index]]} and should be in form {field:{key:index}}
        formattedFieldKeys = {}
        keys = feeder.fieldKeys.keys()
        for key in keys:
            cipher = feeder.fieldKeys.get(key)
            i = 0
            tokenDict = {}
            for i in range(len(cipher[0])):
                tokenDict.update({cipher[0][i]:cipher[1][i]})
            formattedFieldKeys.update({key:tokenDict})
        feeder.fieldKeys = formattedFieldKeys
        #next, we want to update the meansAndDeviations attribute. the current field is empty we want the format {field:[mean,standardDeviation]}
        meansAndDeviations =controllerHelperFunctions.readMeansAndDeviations(modelLocation + "\\data\\meanDeviations.csv")
        feeder.meanDeviations = {}
        i = 0
        for key in keys:
            feeder.meanDeviations.update({key:meansAndDeviations[i+1]})


        return feeder

    def loadRandomEntries(self, poolSize, seedNumber, dataLocation):
        randomData = Feeder(dataLocation, poolSize)
        entryList = randomData.loadRandomNSEntries(poolSize, seedNumber, dataLocation)
        return entryList

    #for convenience Im just going to pull the row data from the entries. Definitley not the fastest way to do this.
    def loadRandomDictList(self, poolSize, seedNumber, dataLocation):
        dictList = []
        entryList = self.loadRandomEntries(poolSize, seedNumber, dataLocation)
        for entry in entryList:
            dictList.append(entry.rowDictionary)
        return dictList
    
    #this can be called to return a json string containing the features required for the model loaded.
    def getChordJSON(self):
        chord = self.getModelTrainingChord()
        jsonString = json.dumps(chord)
        return jsonString

    def getModelTrainingChord(self):
        try:
            #we will just use the cipherkey .csv. we could also use the dataflow. Basically we just need the header.
            cipherKeyLocation = modelLocation + "\\data\\meanDeviations.csv"
            modelTrainingChord = controllerHelperFunctions.readHeader(cipherKeyLocation)
            return modelTrainingChord
        except:
            print("Error retrieving training Chord. Please check model location and save format of the model folder. modelName\\data\\meanDeviations should be present.")

    def getConfidenceList(self, entryList):
        # check to see if the model is loaded.
        if self.noShowModel == "Empty":
            print("No model detected.Auto Loading Controller No show model...")
            self.loadModel(modelLocation)
            print("Model loaded :)")
        #next we need to normalise using the same method/parameters as the model was trained on.
        #the tensor can be parsed to get confidence values.
        dataflow = self.getDataflow(entryList)
        confidenceValues = self.noShowModel.getConfidenceValues(dataflow)
        return confidenceValues
    
    #takes predictionary and returns it sorted by confidence.
    def sortPredictionary(self, predictionary):
        sortedDict = sorted(predictionary.items(), key=lambda x: x[1], reverse=True)
        return sortedDict

    #will categorise confidence values into classes: ExtremelyConfident 95%+, veryConfident, 90-95%, confident 90-95%, lessConfident 85-90%, estimate 80-85%, notConfident 75-80%, unsure 75%-
    def getConfidenceClasses(self, confidenceList):
        categoryList = []
        for value in confidenceList:
            value = float(value)
            category = 'null'
            if value <= 75:
                category = 'unsure'
            elif value <= 80:
                category = 'notConfident'
            elif value <= 85:
                category = 'estimate'
            elif value <= 90:
                category = 'confident'
            elif value <= 95:
                category = 'veryConfident'
            elif value <= 100:
                category = 'extremelyConfident'
            elif value > 100:
                print('Error, Exceeded Upper Boundary')
                category = 'Error, Exceeded Upper Boundary'
            elif value < 0:
                print('Error, Exceeded Lower Boundary')
                category = 'Error, Exceeded Lower Boundary'

            categoryList.append(category)
            return categoryList

    #this methof will contain our standard method for processing data. Change this when we want to improve data manipulation + enrichment.
    def processEntryList(self, entryList):
        try:
            #since we already have the functionality needed in the feeder (which inherits from the dataSpout), lets use the feeder dataStream for the required data enrichment.
            #we can use the dataPreprocessor
            _dataPreprocessor = dataPreprocessor("FromEntryList", 0)
            print("dataPreprocessor (our glorified dataspout) created successfully...")
            #set the channel using entryList
            _dataPreprocessor.setChannel(entryList) 
            print("Feed channel set.")

            #get the chord and remove the fields we dont need if present.
            fullChord = self.getModelTrainingChord()
            print("ModelTraining chord retrieved successfully...")
            #in testing the noshow label was also present so uncomment this line to remove the field. Other fields to be removed can be added to the parsed array.
            #fullChord = controllerHelperFunctions.removeFromChord(fullChord, ['nosho']) #remove target label

            #set the dataStream
            _dataPreprocessor.setDataStream(fullChord)

            #first we need to perform enrichment to populated the 'none' strands
            _dataPreprocessor.distanceTimeEnrichment()

            #encode
            self.encodeDataStream(_dataPreprocessor, modelLocation) #uses the model cipher keys.
            #get rid of null values.
            _dataPreprocessor.nullToZeroChord(fullChord)

            return _dataPreprocessor
        except:
            print("Error. Failed to process entryList.")
    
    def processEntryListWithTargetData(self, entryList):
        try:
            #since we already have the functionality needed in the feeder (which inherits from the dataSpout), lets use the feeder dataStream for the required data enrichment.
            #we can use the dataPreprocessor
            _dataPreprocessor = dataPreprocessor("FromEntryList", 0)
            print("dataPreprocessor (our glorified dataspout) created successfully...")
            #set the channel using entryList
            _dataPreprocessor.setChannel(entryList) 
            print("Feed channel set.")

            #get the chord and remove the fields we dont need if present.
            fullChord = self.getModelTrainingChord() + ['nosho']
            print("ModelTraining chord retrieved successfully...")
            #in testing the noshow label was also present so uncomment this line to remove the field. Other fields to be removed can be added to the parsed array.
            #fullChord = controllerHelperFunctions.removeFromChord(fullChord, ['nosho']) #remove target label

            #set the dataStream
            _dataPreprocessor.setDataStream(fullChord)

            #first we need to perform enrichment to populated the 'none' strands
            _dataPreprocessor.distanceTimeEnrichment()

            #encode
            self.encodeDataStreamAndUpdateCipher(_dataPreprocessor, modelLocation) #uses the model cipher keys.
            #get rid of null values.
            _dataPreprocessor.nullToZeroChord(fullChord)

            return _dataPreprocessor
        except:
            print("Error. Failed to process entryList.")

    def ProcessListForVisualisation(self, entryList):
        originalEntryList = entryList

        #since we already have the functionality needed in the feeder, lets use the feeder dataStream for the required data enrichment.
        #we can use the dataPreprocessor
        _dataPreprocessor = dataPreprocessor("FromEntryList", 0)
        #set the channel using entryList
        _dataPreprocessor.setChannel(entryList)

        #get the chord and remove the field we dont need
        trainingChord = self.getModelTrainingChord()
        fullChord = _dataPreprocessor.getAllTableFields()

        #set the dataStream
        _dataPreprocessor.setDataStream(fullChord)

        #first we need to perform enrichment to populated the 'none' strands
        _dataPreprocessor.distanceTimeEnrichment()

        #encode
        self.encodeDataStreamWithNosho(_dataPreprocessor, modelLocation) #uses the model cipher keys.
        #get rid of null values.
        _dataPreprocessor.nullToZeroChord(trainingChord)

        return _dataPreprocessor

    #will return in the form of a tensor/2D array only containing float/int values.
    def getDataflow(self, entryList):
        _dataPreprocessor = self.processEntryList(entryList)

        #now we need to normalise the dataStreamDict
        #first let get the means and meanDeviations
        meansAndDeviations = controllerHelperFunctions.readMeansAndDeviations(modelLocation + "\\data\\meanDeviations.csv")
        #and normalise
        _dataPreprocessor.tanhNormaliseDataStream(meansAndDeviations)

        #now convert datastream to tensor
        tensor = _dataPreprocessor.getDataflowTensor()

        return tensor

    def getDataStreamDict(self, entryList):
        _dataPreprocessor = self.processEntryList(entryList)
        return _dataPreprocessor.dataStream

    def getWholeDataStreamDict(self, entryList):
        _dataPreprocessor = self.ProcessListForVisualisation(entryList)
        return _dataPreprocessor.dataStream


    def getDataflowAndLabels(self, entryList):
        originalEntryList = entryList

        #since we already have the functionality needed in the feeder, lets use the feeder dataStream for the required data enrichment.
        #we can use the dataPreprocessor
        _dataPreprocessor = dataPreprocessor("FromEntryList", 0)
        #set the channel using entryList
        _dataPreprocessor.setChannel(entryList)

        #get the chord and remove the field we dont need
        fullChord = self.getModelTrainingChord()

        #set the dataStream
        _dataPreprocessor.setDataStream(fullChord + ['nosho']) #quickfix just add the target strand ourself

        #get our labels
        labels = _dataPreprocessor.popStrand('nosho')

        #first we need to perform enrichment to populated the 'none' strands
        _dataPreprocessor.distanceTimeEnrichment()

        #encode
        self.encodeDataStream(_dataPreprocessor, modelLocation) #uses the model cipher keys.
        #get rid of null values.
        _dataPreprocessor.nullToZeroChord(fullChord)

        #now we need to normalise
        #first let get the means and meanDeviations
        meansAndDeviations = controllerHelperFunctions.readMeansAndDeviations(modelLocation + "\\data\\meanDeviations.csv")
        #and normalise
        _dataPreprocessor.tanhNormaliseDataStream(meansAndDeviations)

        #now convert datastream to tensor
        tensor = _dataPreprocessor.getDataflowTensor()

        return tensor, labels

    #basically takes a dataspout, and uses the cipher from the saved model across the datastream (which should already be set!)
    def encodeDataStream(self, dataPreprocessor, modelLocation):
        try:
            #cipher keys can be foubnd inside the data directory of the saved model.
            cipherKeyLocation = modelLocation + "\\data\\cipherKeys.csv"

            #first we need to get the chord of fields the model needed to encode into integer values.
            encodingChord = controllerHelperFunctions.readHeader(cipherKeyLocation)
            #encodingChord = controllerHelperFunctions.removeFromChord(encodingChord, ["nosho"]) #remove target label if present.

            encodingDict = controllerHelperFunctions.readCipherKeys(cipherKeyLocation)

            dataPreprocessor.encodeDataStream(encodingChord, encodingDict)
        except: 
            print("Error. Failed to encode datastream.")
    
    #basically takes a dataspout, and uses the cipher from the saved model across the datastream (which should already be set!)
    #this method will update the cipherwheel already set by adding new values should they appear.
    def encodeDataStreamAndUpdateCipher(self, dataPreprocessor, modelLocation):
        try:
            #cipher keys can be foubnd inside the data directory of the saved model.
            cipherKeyLocation = modelLocation + "\\data\\cipherKeys.csv"

            #first we need to get the chord of fields the model needed to encode into integer values.
            encodingChord = controllerHelperFunctions.readHeader(cipherKeyLocation)
            #encodingChord = controllerHelperFunctions.removeFromChord(encodingChord, ["nosho"]) #remove target label if present.

            encodingDict = controllerHelperFunctions.readCipherKeys(cipherKeyLocation)

            
            dataPreprocessor.encodeDataStream(encodingChord, encodingDict)
        except: 
            print("Error. Failed to encode datastream.")

    #basically takes a dataspout, and uses the cipher from the saved model across the datastream (which should already be set!)
    def encodeDataStreamWithNosho(self, dataPreprocessor, modelLocation):
        #cipher keys can be foubnd inside the data directory of the saved model.
        cipherKeyLocation = modelLocation + "\\data\\cipherKeys.csv"

        #first we need to get the chord of fields the model needed to encode into integer values.
        encodingChord = controllerHelperFunctions.readHeader(cipherKeyLocation)

        encodingDict = controllerHelperFunctions.readCipherKeys(cipherKeyLocation)

        dataPreprocessor.encodeDataStream(encodingChord, encodingDict)

    def visualiseOutput(self, entryList):

        dataflow = self.getDataflow(entryList)
        chord = self.getModelTrainingChord()
        dataflow = np.array(dataflow)
        #shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
        explainer = shap.KernelExplainer(self.noShowModel.model, dataflow)#make our explainer
        shap_values = explainer.shap_values(dataflow)
        shap_values = shap_values[0] #kernel explainer adds an extra dimension

        shap.summary_plot(shap_values,dataflow, feature_names=chord)


        shap.decision_plot(explainer.expected_value[0], shap_values[0], feature_names=chord)
        shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0], shap_values[0], feature_names=chord)
        shap.force_plot(explainer.expected_value[0], shap_values[0], feature_names=chord)
        #shap.plots.force(explainer.expected_value[0], shap_values[0], feature_names=chord)

    #this will return shap values for the given entries which represent the influence of each feature towards the models outputs for those entries.
    def getShapValuesForEntries(self, entryList):
        dataflow = self.getDataflow(entryList)
        chord = self.getModelTrainingChord()
        dataflow = np.array(dataflow)
        #shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
        explainer = shap.KernelExplainer(self.noShowModel.model, dataflow)#make our explainer
        shap_values = explainer.shap_values(dataflow)
        shap_values = shap_values[0] #kernel explainer adds an extra dimension

        return shap_values

    def visualiseOutput2(self, entryList):

        dataflow = self.getDataflow(entryList)
        chord = self.getModelTrainingChord()
        dataflow = np.array(dataflow)
        #shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
        explainer = shap.KernelExplainer(self.noShowModel.model, dataflow)#make our explainer
        shap_values = explainer.shap_values(dataflow)

        class helper_object():
            """
            This wraps the shap object.
            It takes as input i, which indicates the index of the observation to be explained.
            """
            def __init__(self, i):
                self.base_values = shap_values.base_values[i][0]
                self.data = dataflow[i] # X Values
                self.feature_names = chord # The feature names
                self.values = shap_values.values[i]

        shap.summary_plot(shap_values,dataflow, feature_names=chord)


        decision_plot = shap.decision_plot(explainer.expected_value[0], shap_values[0], feature_names=chord)
        shap.initjs()
        shap.plots.waterfall(helper_object(0), len(shap_values[0]))
        #shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0], dataflow[0], feature_names=chord)
        #forcePlot_plot = shap.force_plot(np.asarray(explainer.expected_value[0]), shap_values[0], dataflow[0])

    def getWaterfallPlot(self, entryList):
        dataflow = self.getDataflow(entryList)
        chord = self.getModelTrainingChord()
        dataflow = np.array(dataflow)
        explainer = shap.KernelExplainer(self.noShowModel.model.predict, dataflow)#make our explainer

        for entry in dataflow:
            shap.initjs()
            shap_value = explainer.shap_values(np.asarray(entry))
            waterfall = shap.plot.waterfall(shap_value)
            #shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0], shap_values[0], dataflow[0], feature_names=chord)

    def shapSummaryPlot(self, entryList):
        dataflow = self.getDataflow(entryList)
        chord = self.getModelTrainingChord()

        entry = np.array(dataflow)
        explainer = shap.DeepExplainer(self.noShowModel.model.predict, entry)#make our explainer
        shap_values = explainer.shap_values(entry)
        #summarize
        shap.summary_plot(shap_values, dataflow, feature_names=chord)
        shap.initjs()
        shap_plot = shap.force_plot(explainer.expected_value, shap_values[0], dataflow[0], feature_names=chord, show=True)

        return 0

    #will be the main function to call for getting predictions.
    def getPredictionary(self, entryList):
        predictionary = {}
        #get confidence values
        confidenceList = self.getConfidenceList(entryList)
        i=0
        for entry in entryList:
            pnr_no_unique = entry.getPnr_no_unique()
            predictionary.update({pnr_no_unique:float(confidenceList[i])})
            i+=1
        return predictionary

    #will attatch a confidence value to a unique key made of pnr_no_unique, seat_no, passenger_no
    def getPredictionary2(self, entryList):
        predictionary = {}
        #get confidence values
        confidenceList = self.getConfidenceList(entryList)
        i=0
        for entry in entryList:
            uniqueKey = entry.seatKey #need to update the
            predictionary.update({str(uniqueKey) :float(confidenceList[i])})
            i+=1
        return predictionary

    #used for testing. Will convert a list of entryNodes into a json string containing their entry.rowDictionary attribute.
    def entryListToJSON(self, entryList):
        jsonList = []

        for entry in entryList:
            rowDict = entry.rowDictionary
            r = json.dumps(rowDict)
            jsonList.append(r)

        return jsonList

    #converts whatever is passed in into json.
    def ToJsonOutput(self, input):
        jsonOutput = json.dumps(str(input))
        return jsonOutput

    #converts a given json string (which contains a list of dictionaries) into a list.
    def jsonToDictList(self, jsonInput):
        output = json.loads(jsonInput)
        outputList = ast.literal_eval(output)
        return outputList

    def entryNodeListFromDictList(self, dictList):
        try:
            entryList = []
            for dict in dictList:
                    entry = entryNode(dict)
                    entryList.append(entry)
            return entryList
        except:
            print("Error failed to convert dictionary list into entryNodes. Please check dataType of the element in list converted from json are dictionarys.")

    def entryListFromJSON(self, jsonData):
        # Opening JSON file
        dictList = []

        for entry in jsonData:
            json.load(entry)
            dictList.append(json.dumps(entry))
        return dictList

    #entries need to contain target variable
    def evaluateNoShowModel(self, entryList, _classnames):
        #create evaluator
        eval = Evaluator(self.noShowModel)

        #we need dataflowand labels
        dataflow, labels = self.getDataflowAndLabels(entryList)

        cipherKeyLocation = modelLocation + "\\data\\cipherKeys.csv"
        encodedLabels = controllerHelperFunctions.encodeStrand(cipherKeyLocation, 'nosho', labels)

        eval.evaluateThis(dataflow, encodedLabels, classnames = _classnames)
    
    #entries need to contain target variable
    def compare2Models(self, entryList, _classnames, modelLocation1, modelLocation2):
        model1 = self.getModel(modelLocation1)
        model2 = self.getModel(modelLocation2) 
        #create evaluator
        eval = Evaluator(self.noShowModel)

        #we need dataflowand labels
        dataflow, targetClasses = self.getDataflowAndLabels(entryList)

        cipherKeyLocation = modelLocation + "\\data\\cipherKeys.csv"
        encodedLabels = controllerHelperFunctions.encodeStrand(cipherKeyLocation, 'nosho', targetClasses)

        #lets add it into the second slot like so.
        eval.addSecondModel(model2)

        #the evaluator can now use this data to compare the 2 models.
        eval.compareModelsAgainstThis(dataflow, encodedLabels, classnames = _classnames)


demoExecution()
