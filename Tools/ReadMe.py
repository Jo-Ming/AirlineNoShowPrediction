"""This documant will comtain example demonstrations of tools and how to set them up. The idea of these tools are to make repetative tasks
needed to be applied to most machine learning models, more accessible to allow users to play with the data more freely. 

To learn how to use it it is recommended to debug and step through explantation. 

To set up the environment look at the readme.txt document to make sure that the code will run.

"""

"""To set up the API, 
1. follow the readme to make sure the environment is running.
2. Create the controller.py intended for your application. 
3. Import the relavent packages for application. """

from Feeder import Feeder #can be used to regurgetate data, or read.csv to dataflow
from dataSpout import dataSpout #to create and manipulate a datafeed from a channel.
import controlHelper #this library has remove from chord and the save functionality.
from dataVisualiser import dataVisualiser #will use libraries like seaborn to visualise the data.
from Evaluator import Evaluator #will contain code for exectuing evaluations.
from Model import Model #can be used to load models, get predictions and confidence values.

"""We can apply the tools to an application to help us do stuff. Put a break point at each of these routines to see how the tools work. """
def demoController():
    #we can get a dataflow from a .csv
    #showingTheFeeder()
    
    #dataSpout can be used to manipulate the dataflow.
    #showDataSpoutExampleFunctions()#example functions 

    #we can visualise data
    #showDataVisualisating() #dataVisualiser object can be used to generate graphs of training/testing data
    
    #We can train a model.
    #freshlyTrainedModel = trainAModel()#also shows some evaluations on the model.

    #we can load a pre saved model.
    #model = loadModel()

    #we can evaluate the performance of a model.
    evaluatingAModel()

    #we can compare the performance of 2 models.
    comparing2Models()


"""Example 1. 
    The Feeder: Creating data feed with your own .csv which then allows to to perform pre built fucntions onto the dataflow."""

def showingTheFeeder(): #feeder allows the user to state a location outside the channel bank.
    #import the feeder like so.

    #to construct a feeder we parse feeder(fileLocation, outputSize).
    myFeeder = Feeder(channel = 'C:\\Users\\Jo Ming\\Documents\\AirevoWorkspace\\AirevoCode-Space\\NoShowModel\\CSVFiles\\CSVFiles2.csv', streamSize = 10)

    #The Feeder can do all the stuff a dataSpout can...
    #for example display x number of entries:
    myFeeder.showXEntries(1) #this function shows the info from the table not the dataStream

    # You can then set the dataStream of the feeder with a chord. The chord is made up fields connected to strands (within a dictionary)
    # which contains the names of each column in the table you wish to use as your dataflow.

    # to get every column in the table.
    myFeeder.setDataStream(myFeeder.getAllTableFields()) #parse a chord

    # The dataStream becomes the attribute myFeeder.dataStream 
    # The dictionary in the attribute looks like this:
    # myFeeder.dataStream = {col1 : [entry1, entry2, entry3, ...],
    #                        col2 : [entry1, entry2, entry3, ...],
    #                        col3 : [entry1, entry2, entry3, ...],
    #                         ...
    #                        }


    # we can also set the dataStream like so if we make sure we know exactly what fields in the table we desire:
    # chord = [my, desired, fields]
    # myFeeder.setDataStream(chord) 

    # we can do an evaluation on the dataStream 
    myFeeder.evaluateDataStream()

    # We can remove a data column from our dataStream if we'd like.
    strandToUseOrThrow = myFeeder.popStrand('pnr_no_unique') #will return the strand of data and remove the fieldname and value from the dictionary.
    print("popped = " + str(strandToUseOrThrow))

    #To get a strand of data without removing it from the dataStream we can do this.
    exampleStrand = myFeeder.getStrand('departure_datetime') #make sure the column name is in the table.
    print("ExampleStrand = " + str(exampleStrand))

    #we can add or update a field and strand in the dataStream like so:
    myFeeder.addData('new_field_name', strandToUseOrThrow)
    myFeeder.popStrand('new_field_name') #get rid of it.
    
    #we can convert the quantitative fields from strings to integers.
    toIntegers = ['no_in_party', 'segment_distance', 'percentage_ffp'] 
    myFeeder.typeCastChordToInt(toIntegers) # now these have been converted from type str -> int

    #we can also encrypt fields from strings to integers 
    myFeeder.encodeStrand('mct_violation') #this will convet N, Y -> 0, 1
    #we can also encode a statedChord to integers
    chordToEncode = ['board_point', 'off_point', 'inbound_airport', 'trip_type']
    #parse the chord into 
    myFeeder.encodeChord(chordToEncode)
    #evaluate the dataStream
    myFeeder.evaluateDataStream()

    #then we can convert the dataStream into tensor to parse into a model. but first we need to set the target dataStream
    myFeeder.setTargetDataFromField('mct_violation') #once that is set we can get our dataflow
    dataflow, targetClasses = myFeeder.getDataflowTensor()
    print("Dataflow: " + str(dataflow))
    print("labels: " + str(targetClasses))

    #this can then be split into 80:20 training testing data and training testing labels.
    trainingData, testingData, trainingClasses, testingClasses = myFeeder.splitDataflowAndTargetClasses(dataflow, targetClasses)

    print("Training Data: ", trainingData)
    print("Training Classes: ", trainingClasses)
    print("Testing Data: ", testingData)
    print("Testing Classes: ", testingClasses )

    #this data can then be parsed into any model.

"""here is a chunky example of how a dataSpout or feeder can be used to manipulate its dataflow. If we find ourselves, performing the same
transformations frequently, we can add it as a feeder method."""
def showDataSpoutExampleFunctions():

    exampleDataSpout = dataSpout('pnrSegmentPassenger', 10000) #create a dataSpout object. channel = 

    #state out desired fields catagorised into chords.

    targetField = ['mct_violation'] # this is the field containinf our labels.

    depArrChord = ['departure_datetime', 'arrival_datetime'] #these fields contain our departure and arrival datetimes.

    datetimeChord = ['inbound_arrival_datetime'] #other datetime fields to find relative to boarding date.

    dateChord = ['document_birthdate'] #dates (without times)

    quantiveChord = ['day_of_week', 'no_in_party', 'booked_connection_time', 'minimum_connection_time', 'time_under_over', 
                     'equipment', 'advance_bkg_days'] #chord that need to be converted to integers as they are read as strings.

    chordToEncode = ['board_point', 'off_point', 'trip_type','domestic_international', 'gender_code', 'current_status', 
                     'class', 'inbound_route','inbound_airport', 'segment_type'] #chord which needs to be quantised/encoded into integers to be parsed. 
    
    #we can display an example entry if wish.
    #trainingSpout.showXEntries(1)

    #the metaChord/completeChord being the sum of each stated subChord.
    completeChord = targetField + datetimeChord + dateChord + quantiveChord + chordToEncode + depArrChord

    #populate our dataStream
    exampleDataSpout.setDataStream(completeChord) #datastream is set.
    exampleDataSpout.encodeStrand('mct_violation') #convert the strand from chars of '0' or '1' to integer 0 or 1
    exampleDataSpout.setTargetDataFromField('mct_violation') #make this field our self.targetData attribute, removing it from the dataStream.
    controlHelper.removeFromChord(completeChord, targetField) #take it out the complete chord for when we call the normalise function.

    #data that will be added to enrich our data.
    enrichmentChord = ['InboundToBoarding_Distance', 'UTCdateTimeDifferenceMinutes'] # names of new fields being added
    UTCChord = ['UTC_inbound_arrival_datetime', 'UTC_departure_datetime'] # names of new times being added

    #little bit of enriching. The function could definitely be modified to do the second two functions for you.
    distanceList = exampleDataSpout.getDistanceList('inbound_airport', 'board_point') #will return a strand containing distances between airport a and b.
    exampleDataSpout.addData('InboundToBoarding_Distance', distanceList) #add the data to our dataStream.
    exampleDataSpout.nullToZero('InboundToBoarding_Distance') #will set any NULL or '' values to 0 along that strand of the dataStream.

    #requires the field for airports (to find timezone) as well as the datetime field.
    exampleDataSpout.setUTCDateTimes('inbound_airport','inbound_arrival_datetime', 'UTC_inbound_arrival_datetime') # will convert the dateTime to UTC
    exampleDataSpout.setUTCDateTimes('board_point', 'departure_datetime', 'UTC_departure_datetime') #new field with UTC datetimes

    #will set a new field with the time difference in minutes between the two given datetimes
    exampleDataSpout.setDateTimeDifferences('UTC_departure_datetime', 'UTC_inbound_arrival_datetime', 'UTCdateTimeDifferenceMinutes')
    exampleDataSpout.nullToZero('UTCdateTimeDifferenceMinutes')  #null values to 0

    #find the relative time in days until boarding  date for all dates in chord. Because we are not considering timezones there is an error margin of 1 day
    exampleDataSpout.dateTimeChordDaysUntilBoarding(datetimeChord) #uses datetimes
    exampleDataSpout.dateChordDaysUntilBoarding(dateChord) #uses dates (without time)
    #datetimeToMinuteOfDay() must be called after dateTimedaysUntilBoarding() otherwise the value of boarding datetime changes.
    exampleDataSpout.datetimeChordToMinuteOfDay(depArrChord) #converts these datetimes to minute of the day.

    #convert all the quantitative fields into type int.
    exampleDataSpout.typeCastChordToInt(quantiveChord) #ensure the datatype for each fields strand is an integer. 

    #evaluate the dataStream
    exampleDataSpout.evaluateDataStream()
    #evaluate target data
    exampleDataSpout.evaluateTargetData()

    #encord our desired chord
    exampleDataSpout.encodeChord(chordToEncode)

    completeChord = completeChord + enrichmentChord + UTCChord

    print(len(exampleDataSpout.dataStream.keys())) # how many features are in the data.

    dataflow, targetClasses = exampleDataSpout.getDataflowTensor() #get the tensor and target labels.
    #we can then split the datat into 80:20, training:testing
    trainingData, testingData, trainingClasses, testingClasses = exampleDataSpout.splitDataflowAndTargetClasses(dataflow, targetClasses)#divide the tensor into training and testingdata


"""If we have a .csv file like we just generated in the model training we can also visualise the data using the dataVisualiser object.
which uses the seaborn library. Thaere are many ways to visualise the data which have not yet been implemented."""
#there are currently 7 different types plots in this object, if you know what to plot.
#more infor on how plots work found here: https://seaborn.pydata.org/examples/index.html
def showDataVisualisating():
    #first, lets import a visualiser 
    dataVis = dataVisualiser('savedModels\example2ModelSave\data\dataflow.csv', 'csv') #smaller dataset for easier visualisation
    
    #we can then use the functions in the dataVisualiser tool to plot graphs for example a heatmap like so.
    #if the figure is hard to read click the configure subplots tab at the top and then click the tight configuration button.
    dataVis.generateHeatmap() 

    dataVis.groupBoxplot('day_of_week','booked_connection_time', 'gender_code')# datavisualiser.groupbexplot(x,y,hue)

    dataVis.groupBoxplot('day_of_week','advance_bkg_days', 'gender_code')# datavisualiser.groupbexplot(x,y,hue)

    dataVis.groupBoxplot('class', 'equipment', 'gender_code') #showing how much equipment they carry by class

"""Once a process has been established like the previous data processing function. we can add it as a feeder method. This will massively reduce the
amount of code in the place whe training a model. If set to a channel we can use that method to consistently get the same data, which can be used
to train, evaluate models and visualise the data.""" 
#I have used shap to find the feature significance, I would like to go deeper into this library and create visualisations as to how the
#model came to its conclusion. example/info can be found here : https://slundberg.github.io/shap/notebooks/plots/decision_plot.html
def trainAModel():
    """First lets get our data. By calling the dataProcess method."""
    #if we have already defined a process for data it can become a feeder method. for example we can apply the precious example 
    modelFeeder = Feeder('TheQuantileWorld\segmentXpnrXpassenger_AllConnections.csv', 'max')
    dataflow, targetClasses, chord = modelFeeder.loadreducedData() #get dataflow and targetFields

    completeChord = modelFeeder.getDataflowFields() #will return a list of all the dataflow fields.

    #split the data into training and testing data
    trainingData, testingData, trainingClasses, testingClasses = modelFeeder.splitDataflowAndTargetClasses(dataflow, targetClasses)

    """Isn't this much cleaner? Now, we can use it to train a model. Increase the number of epochs for a better model. will get 99.99 accuracy within 30 epochs"""
    #this library holds various neural network architectures.
    import neuralNetworks

    #For some reason it seem that the matplot graph don't always work inside the debugger but do when you run the program.
    exampleModel = neuralNetworks.slimModel(completeChord, trainingData, testingData, trainingClasses, testingClasses, epochs = 30, batch_size = 42, saveName = "slimShadyModel")

    #The neural networks return the model after training we can then save the model like so.
    import controlHelper #contains save function. save model(model, dataSpout, saveName)
    controlHelper.saveModel(exampleModel, modelFeeder, "SlimShadyModel") #will generate a save directory for the model.


"""To Load a model. you can create a model object which takes a name and a location parameter."""
def loadModel():
    #this object then has useful fucntionality. To load we give it a name, and the models save location.
    exampleModel = Model("model",'TrainedModels\slimModel') 


"""If we wanted to evaluate a model. We just created, we can make an evaluator"""
def evaluatingAModel():
    #make sure we have imported the Model and evaluator Object.
    #make an evaluator
    
    modelToEvaluate = Model("ourFreshModel", "C:\\Users\\Jo Ming\\Documents\\AirevoWorkspace\\AirevoCode-Space\\Project1MinimumConnectionErrorModel\\savedModels\\exampleSave")

    #next we need a dataflow, and labels 
    feederTool = Feeder('C:\\Users\\Jo Ming\\Documents\\AirevoWorkspace\\AirevoCode-Space\\Project1MinimumConnectionErrorModel\\TheQuantileWorldsegmentXpnrXpassenger_AllConnections.csv', streamSize = 'max')

    dataflow, targetClasses, chord = feederTool.loadreducedData() #returns 3 pieces of info

    #we have our evaluator which is attatched to the model. For .evaluateThis we need to parse 3 things, the dataflow, labels, classNames
    myEvaluator = Evaluator(modelToEvaluate) #the evaluator attatches itself to a model then self destructs after use.
    myEvaluator.evaluateThis(dataflow, targetClasses, classnames = ['Non-MCT_Violation','MCT_Violation'])
      

"""In the situation we have trained 2 models. We may wish to measure them against eachother"""
def comparing2Models():
    #lets load our newly trained model in again.
    modelToEvaluate = Model("ourFreshModel", "C:\\Users\\Jo Ming\\Documents\\AirevoWorkspace\\AirevoCode-Space\\Project1MinimumConnectionErrorModel\\savedModels\\modelexampleSave")

    #lets make an evaluator by parsing this model to fill model slot 1. Remember an evaluator cannot exist without a model.
    evaluatorForModelComparison = Evaluator(modelToEvaluate)
    
    #to evaluate 2 models we need a second model in the evaluators model slot 2
    #Here's one I made earlier, trained for 35 epochs.
    secondModel = Model("slimShadyModel", "C:\\Users\\Jo Ming\\Documents\\AirevoWorkspace\\AirevoCode-Space\\Project1MinimumConnectionErrorModel\\savedModels\\slimShadyModel") #The slim NN architecture trained with some features kept in the dark!
    #lets add it into the second slot like so.
    evaluatorForModelComparison.addSecondModel(secondModel)

    #now to compare their performances we need to conjure some testing data.
    feederTool = Feeder('C:\\Users\\Jo Ming\\Documents\\AirevoWorkspace\\AirevoCode-Space\\Project1MinimumConnectionErrorModel\\TheQuantileWorldsegmentXpnrXpassenger_AllConnections.csv', streamSize = 'max')
    dataflow, targetClasses, chord = feederTool.loadreducedData() #returns 3 pieces of info

    #the evaluator can now use this data to compare the 2 models.
    evaluatorForModelComparison.compareModelsAgainstThis(dataflow, targetClasses, classnames = ['Non-MCT_Violation','MCT_Violation'])

demoController()

""""""




