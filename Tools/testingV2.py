from dataSpout import dataSpout
from neuralNetworks import *
from Model import Model #object will be used to load and train models.
from Feeder import Feeder
from Evaluator import Evaluator #will contain code for exectuing evaluations.
from dataVisualiser import dataVisualiser

mctChord = ['day_of_week', 'no_in_party', 'cancelled', 'current_status_category', 'current_status', 'counted_segment',
                                    'arrival_time', 'departure_time','board_point', 'off_point', 'trip_type', 'domestic_international',
                                    'segment_no', 'segment_distance', 'current_status', 'inbound_airport',
                                    'class', 'inbound_route', 'inbound_segment_no', 'segment_type', 'booked_connection_time', 
                                    'minimum_connection_time', 'time_under_over','departure_datetime', 'inbound_arrival_datetime', 'equipment']

chordToEncode = ['cancelled', 'current_status_category', 'board_point', 'off_point', 'trip_type','domestic_international',
                             'current_status', 'class', 'inbound_route','inbound_airport', 'inbound_segment_no', 'counted_segment',
                             'segment_type', 'booked_connection_time', 'minimum_connection_time', 'time_under_over']

mctBaseChord = ['mct_violation', 'no_in_party', 'segment_distance', 'departure_datetime', 'board_point', 'off_point', 'inbound_airport', 'trip_type', 'inbound_arrival_datetime']
#This test is just for code cleaning. call a dataSpout
def test1(mctChord, chordToEncode):

    demoSpout = dataSpout('testSegment_500K', 'max')#streamSize = 100
    
    #make our data stream attribute 
    demoSpout.setDataStream(mctChord)

    #demoSpout.showXEntries(1)

    #set time difference in the time fields in format .setTimeDifferences(timeField1, timeField2, name of new field)
    demoSpout.setTimeDifferences('arrival_time', 'departure_time', 'timeDifference')

    #set the dateTime differences.
    usefulCluster = ['departure_datetime', 'inbound_arrival_datetime', 'inbound_airport',
                'board_point', 'domestic_international']
    
    distanceList = demoSpout.getDistanceList('inbound_airport', 'board_point')

    demoSpout.addData('InboundToBoarding_Distance', distanceList)
    demoSpout.nullToZero('InboundToBoarding_Distance')

    #requires the field for airports (to find timezone) as well as the dattimne field.
    demoSpout.setUTCDateTimes('inbound_airport','inbound_arrival_datetime', 'UTC_inbound_arrival_datetime')
    demoSpout.setUTCDateTimes('board_point', 'departure_datetime', 'UTC_departure_datetime')

    demoSpout.setDateTimeDifferences('UTC_departure_datetime', 'UTC_inbound_arrival_datetime', 'UTCdateTimeDifferenceMinutes')
    demoSpout.nullToZero('UTCdateTimeDifferenceMinutes')

    demoSpout.setTargetDataFromTableColumn('mct_violation')

    demoSpout.nullToZero('equipment')

    #show characteristics of data strands
    demoSpout.evaluateDataStream()

    #encode a strand to integers.
    demoSpout.encodeAndNormaliseChord(chordToEncode)

    demoSpout.evaluateTargetData()

    #get data to be parsed into model
    trainingData, testingData, trainingClasses, testingClasses = demoSpout.getDataflowTensor()
    #set target data attribute
    trainModel4(trainingData, testingData, trainingClasses, testingClasses, epochs = 30, batch_size = 25)
    #enumerate subcord.
    
def test2():
    test2Chord = ['day_of_week', 'no_in_party', 'cancelled','current_status_category', 'current_status','board_point', 'off_point', 'trip_type', 'domestic_international', 
                 'current_status', 'inbound_airport','class', 'inbound_route', 'segment_type', 'booked_connection_time', 'domestic_international',
                                    'minimum_connection_time', 'time_under_over','departure_datetime', 'inbound_arrival_datetime', 'equipment']

    chordToEncode = ['cancelled', 'current_status_category', 'board_point', 'off_point', 'trip_type','domestic_international',
                             'current_status', 'class', 'inbound_route','inbound_airport',
                             'segment_type', 'booked_connection_time', 'minimum_connection_time']
    
    demoSpout = dataSpout('joseph.segment_500K', 'max')#streamSize = 100
    
    #make our data stream attribute 
    demoSpout.setDataStream(test2Chord)

    
    distanceList = demoSpout.getDistanceList('inbound_airport', 'board_point')

    demoSpout.addData('InboundToBoarding_Distance', distanceList)
    demoSpout.nullToZero('InboundToBoarding_Distance')

    #requires the field for airports (to find timezone) as well as the dattimne field.
    demoSpout.setUTCDateTimes('inbound_airport','inbound_arrival_datetime', 'UTC_inbound_arrival_datetime')
    demoSpout.setUTCDateTimes('board_point', 'departure_datetime', 'UTC_departure_datetime')

    demoSpout.setDateTimeDifferences('UTC_departure_datetime', 'UTC_inbound_arrival_datetime', 'UTCdateTimeDifferenceMinutes')
    demoSpout.nullToZero('UTCdateTimeDifferenceMinutes')

    demoSpout.setTargetDataFromTableColumn('mct_violation')

    demoSpout.nullToZero('equipment')
    demoSpout.nullToZero('time_under_over')

    #show characteristics of data strands

    #encode a strand to integers.
    demoSpout.encodeAndNormaliseChord(chordToEncode)

    #demoSpout.evaluateDataStream()
    #demoSpout.evaluateTargetData()

    #get data to be parsed into model
    dataflow, targetClasses = demoSpout.getDataflowTensor()
    trainingData, testingData, trainingClasses, testingClasses = demoSpout.splitDataflowAndTargetClasses(dataflow, targetClasses)

    #set target data attribute
    trainModel6(trainingData, testingData, trainingClasses, testingClasses, epochs = 30, batch_size = 25)
    #enumerate subcord.
#test1(mctChord, chordToEncode)
def test3(): 
    """For this test we are going to:
    1. load a model
    2. evaluate the model """

    testModel = Model("600K_model", 'TrainedModels\TrainingSet_600K')
    mctEvaluation = Evaluator(testModel)
    mctEvaluation.mctModelEvalutation1(testModel)

def test4():
    """Comapare performance of 2 models."""
    mctEvaluation = Evaluator("Dunno")
    mctEvaluation.modelROCComparision()

def test5():
    """Write our enriched dataflow to csv"""
    test2Chord = ['mct_violation', 'day_of_week', 'no_in_party', 'cancelled','current_status_category','board_point', 'off_point', 'trip_type', 'domestic_international', 
                 'current_status', 'inbound_airport','class', 'inbound_route', 'segment_type', 'booked_connection_time', 'domestic_international',
                                    'minimum_connection_time', 'time_under_over','departure_datetime', 'inbound_arrival_datetime', 'equipment']
    
    demoSpout = dataSpout('allConnectionsPnrXsegment', 'max')#streamSize = 100
    
    #make our data stream attribute 
    demoSpout.setDataStream(test2Chord)

    
    distanceList = demoSpout.getDistanceList('inbound_airport', 'board_point')

    demoSpout.addData('InboundToBoarding_Distance', distanceList)
    demoSpout.nullToZero('InboundToBoarding_Distance')

    #requires the field for airports (to find timezone) as well as the dattimne field.
    demoSpout.setUTCDateTimes('inbound_airport','inbound_arrival_datetime', 'UTC_inbound_arrival_datetime')
    demoSpout.setUTCDateTimes('board_point', 'departure_datetime', 'UTC_departure_datetime')

    demoSpout.setDateTimeDifferences('UTC_departure_datetime', 'UTC_inbound_arrival_datetime', 'UTCdateTimeDifferenceMinutes')
    demoSpout.nullToZero('UTCdateTimeDifferenceMinutes')

    demoSpout.toIntegers('mct_violation')
    demoSpout.setTargetDataFromField('mct_violation')

    demoSpout.nullToZero('equipment')
    demoSpout.nullToZero('time_under_over')

    demoSpout.saveDataflowToCSV("MCT_Model_v2\savedDatastreams\Datastream.csv")

def test6():
    
    testModel = Model("HughHeftyModel", "TrainedModels\hughHeftyModel")
    
    heftyFeeder = Feeder()

    dataflow, targetData = heftyFeeder.loadDataHeftyModeldata(spoutChannel = 'allConnectionsPnrXsegment', streamSize = 'max', evalDataStream = False, evalTargetData=False)
    
    confidenceValues = testModel.getConfidenceValues(dataflow)
    

    import evaluatorHelperFunctions
    predictedClasses = testModel.getBinaryPredictions(confidenceValues)

    confusionMatrix = evaluatorHelperFunctions.binaryConfusionMatrix(targetData, predictedClasses, classNames = ['Non-MCT_Violation','MCT_Violation'])
    print(confusionMatrix)
    evaluatorHelperFunctions.binaryConfusionMatrixAnalysis(confusionMatrix)
    #make our data stream attribute 
    #megaSpout.setDataStream(megaChord)
    #megaSpout.showXEntries(1)

def test7():
    import controlHelper
    #state out desired fields catagorised into chords.

    targetField = ['mct_violation'] # this is the field containinf our labels.

    depArrChord = ['departure_datetime', 'arrival_datetime'] #these fields contain our departure and arrival datetimes.

    datetimeChord = ['inbound_arrival_datetime'] #other datetime fields to find relative to boarding date.

    dateChord = ['document_birthdate'] #dates (without times)

    quantiveChord = ['day_of_week', 'no_in_party', 'booked_connection_time', 'minimum_connection_time', 'time_under_over', 
                     'equipment', 'advance_bkg_days'] #chord that need to be converted to integers as they are read as strings.

    chordToEncode = ['board_point', 'off_point', 'trip_type','domestic_international', 'gender_code', 'current_status', 
                     'class', 'inbound_route','inbound_airport', 'segment_type'] #chord which needs to be quantised/encoded into integers to be parsed. 
    
    
    myFeederTool = Feeder('TheQuantileWorld\segmentXpnrXpassenger_AllConnections.csv', 'max') #create a Feeder object. channel = 

    #we can display an example entry if wish.
    #trainingSpout.showXEntries(1)

    #the metaChord/completeChord being the sum of each stated subChord.
    completeChord = targetField + datetimeChord + dateChord + quantiveChord + chordToEncode + depArrChord

    #populate our dataStream
    myFeederTool.setDataStream(completeChord) #datastream is set.
    myFeederTool.encodeStrand('mct_violation') #convert the strand from chars of '0' or '1' to integer 0 or 1
    myFeederTool.setTargetDataFromField('mct_violation') #make this field our self.targetData attribute, removing it from the dataStream.
    controlHelper.removeFromChord(completeChord, targetField) #take it out the complete chord for when we call the normalise function.

    #data that will be added to enrich our data.
    enrichmentChord = ['InboundToBoarding_Distance', 'UTCdateTimeDifferenceMinutes'] # names of new fields being added
    UTCChord = ['UTC_inbound_arrival_datetime', 'UTC_departure_datetime'] # names of new times being added

    #little bit of enriching. The function could definitely be modified to do the second two functions for you.
    distanceList = myFeederTool.getDistanceList('inbound_airport', 'board_point') #will return a strand containing distances between airport a and b.
    myFeederTool.addData('InboundToBoarding_Distance', distanceList) #add the data to our dataStream.
    myFeederTool.nullToZero('InboundToBoarding_Distance') #will set any NULL or '' values to 0 along that strand of the dataStream.

    #requires the field for airports (to find timezone) as well as the datetime field.
    myFeederTool.setUTCDateTimes('inbound_airport','inbound_arrival_datetime', 'UTC_inbound_arrival_datetime') # will convert the dateTime to UTC
    myFeederTool.setUTCDateTimes('board_point', 'departure_datetime', 'UTC_departure_datetime') #new field with UTC datetimes

    #will set a new field with the time difference in minutes between the two given datetimes
    myFeederTool.setDateTimeDifferences('UTC_departure_datetime', 'UTC_inbound_arrival_datetime', 'UTCdateTimeDifferenceMinutes')
    myFeederTool.nullToZero('UTCdateTimeDifferenceMinutes')  #null values to 0

    #find the relative time in days until boarding  date for all dates in chord. Because we are not considering timezones there is an error margin of 1 day
    myFeederTool.dateTimeChordDaysUntilBoarding(datetimeChord) #uses datetimes
    myFeederTool.dateChordDaysUntilBoarding(dateChord) #uses dates (without time)
    #datetimeToMinuteOfDay() must be called after dateTimedaysUntilBoarding() otherwise the value of boarding datetime changes.
    myFeederTool.datetimeChordToMinuteOfDay(depArrChord) #converts these datetimes to minute of the day.

    #convert all the quantitative fields into type int.
    myFeederTool.typeCastChordToInt(quantiveChord) #ensure the datatype for each fields strand is an integer. 

    #evaluate the dataStream
    myFeederTool.evaluateDataStream()
    #evaluate target data
    myFeederTool.evaluateTargetData()

    #encord our desired chord
    myFeederTool.encodeChord(chordToEncode)

    completeChord = completeChord + enrichmentChord + UTCChord

    print(len(myFeederTool.dataStream.keys())) # how many features are in the data.

    dataflow, targetClasses = myFeederTool.getDataflowTensor() #get the tensor and target labels.
    #we can then split the datat into 80:20, training:testing
    trainingData, testingData, trainingClasses, testingClasses = myFeederTool.splitDataflowAndTargetClasses(dataflow, targetClasses)#divide the tensor into training and testingdata

def test8():
    """Lets play with data visualiations"""

    #create an instance of data visualiser
    dataVis = dataVisualiser('savedModels\slimModel\data\dataflow.csv') #smaller dataset for easier visualisation

    dataVis.groupBoxplot('day_of_week','advance_bkg_days', 'gender_code')# datavisualiser.groupbexplot(x,y,hue)

    #now lets test if we can form a heatmap for a trained models datastream

    return 0

test8()