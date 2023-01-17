from dataSpout import dataSpout
from neuralNetworks import *
from Model import Model #object will be used to load and train models.
from Evaluator import Evaluator #will contain code for exectuing evaluations.
import shap
from Feeder import Feeder
from pathlib import Path #useful for directory management 

def loadSegmentXpnr():
    spout = dataSpout('segmentXpnr_500K', 'max')
    spout.showXEntries(2)
    chord = ['mct_violation']
    spout.setDataStream(chord)
    spout.evaluateDataStream()

def training():
    test2Chord = ['mct_violation', 'day_of_week', 'no_in_party', 'cancelled','current_status_category','board_point', 'off_point', 'trip_type', 'domestic_international', 
                 'current_status', 'inbound_airport','class', 'inbound_route', 'segment_type', 'booked_connection_time', 'domestic_international',
                                    'minimum_connection_time', 'time_under_over','departure_datetime', 'inbound_arrival_datetime', 'equipment']

    chordToEncode = ['day_of_week', 'no_in_party', 'cancelled','current_status_category', 'board_point', 'off_point', 'trip_type', 'domestic_international', 
                 'current_status', 'inbound_airport','class', 'inbound_route', 'segment_type', 'booked_connection_time',
                                    'minimum_connection_time', 'time_under_over', 'equipment', 'UTCdateTimeDifferenceMinutes', 'InboundToBoarding_Distance']
    
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

    #show characteristics of data strands

    #encode a strand to integers.
    demoSpout.encodeAndNormaliseChord(chordToEncode)

    #demoSpout.evaluateDataStream()
    demoSpout.evaluateTargetData()

    #get data to be parsed into model
    dataflow, targetClasses = demoSpout.getDataflowTensor()
    trainingData, testingData, trainingClasses, testingClasses = demoSpout.splitDataflowAndTargetClasses(dataflow, targetClasses)

    #set target data attribute
    trainModel6(trainingData, testingData, trainingClasses, testingClasses, epochs = 150, batch_size = 100)
    #enumerate subcord.

def everything():
    megaSpout = dataSpout('allConnectionsPnrXsegment', 'max')

    megaChord = megaSpout.getAllTableFields()

    fieldsToRemove = ['mct_seg_no', 'mct_violation_type', 'mct_datetime', 'mct_inbound_segment_no', 'violation_type']

    for field in fieldsToRemove:
        i=0
        for subField in megaChord:
            if(field == subField):
                megaChord.pop(i)
                print(str(field) + "has been popped.")
            i+=1
    
    print(len(megaChord))

    megaSpout.setDataStream(megaChord)
    megaSpout.toIntegers('mct_violation')
    megaSpout.setTargetDataFromField('mct_violation')
    megaSpout.encodeAndNormaliseChord(megaChord)

    dataflow, targetClasses = megaSpout.getDataflowTensor()
    trainingData, testingData, trainingClasses, testingClasses = megaSpout.splitDataflowAndTargetClasses(dataflow, targetClasses)

    trainHeftyModel(trainingData, testingData, trainingClasses, testingClasses, epochs = 35, batch_size = 30, saveName= "heftyModel")

def everythingPlusEnrichment():
    heftySpout = dataSpout('allConnectionsPnrXsegment', 1000)
    
    #fieldsToRemove = ['mct_seg_no', 'mct_violation_type', 'mct_datetime', 'mct_inbound_segment_no', 'pnr_no_unique', 'violation_type',
    #                   'minimum_connection_time', 'booked_connection_time', 'time_under_over' ,'trip_type', 'off_point', 'board_point', 'transfer']
    
    targetField = ['mct_violation']

    primaryDatetimesChord = ['departure_datetime', 'arrival_datetime']
   

    characterChord = []

    #timeChord = ['departure_time', 'arrival_time']

    dateChord = ['arrival_date']

    dateTimeChord = ['last_update_datetime', 'ticket_notified_date', 'create_date', 'segment_create_date', 'ttl_warning_datetime', 'inbound_arrival_datetime',
                    'inbound_departure_datetime', 'departure_datetime_utc','departure_datetime_sys','first_departure_datetime', 
                    'last_departure_datetime', 'original_create_date', 'first_host_departure_datetime',
                    'calculated_ttl', 'original_create_date_agt', 'duplicate_audit_datetime', 'ttl_departure_datetime', 'first_host_dept_datetime_sys',
                    'first_host_dept_datetime_utc', 'first_host_dept_datetime_agt', 'current_ttl', 'normal_period_ttl']

    encodeChord = [ 'current_status_category', 'current_status', 'class', 'mkt_carrier_code', 'board_point', 'off_point', 'trip_type', 
                    'inbound_trip_type', 'inbound_airport', 'inbound_carrier', 'inbound_route', 'calculated_ttl_source',
                    'host_ttl_seq', 'domestic_international', 'op_carrier_code', 'op_booking_class', 'segment_type', 
                    'marriage_grouping_code',  'inbound_op_carrier', 'previous_status',
                    'host_ttl_version', 'host_ttl_type', 'csh_flight_ind', 'first_host_agent_id', 'current_host_agent_id', 
                    'first_agent_originator_id', 'current_agent_originator_id', 'last_update_level',
                    'tty_pnr_no', 'booking_type', 'group_booking', 'first_host_board_point', 'first_host_off_point', 'booking_maturity',
                    'ffp_booking', 'country_of_sale', 'pos_information', 'full_journey_string', 'full_flight_string', 'full_status_string',
                    'transfer', 'contiguous_journey', 'first_host_booking_class', 'pos_violation', 'ticketed',
                    'duplicate', 'duplicate_match_level', 'ttl_class', 'ttl_board_point',
                    'ttl_off_point', 'corporate_booking', 'ttl_warning', 'first_sand_agent_code', 'current_sand_agent_code',
                    'previous_update_level', 'destination_airport', 'destination_country', 'country_of_fare',
                    'pta', 'host_lowest_class']

    normalizeChord = ['segment_no', 'day_of_week', 'no_in_party', 'segment_distance', 'booked_connection_time', 'minimum_connection_time', 'time_under_over', 
                        'inbound_segment_no', 'equipment', 'no_of_host_int_pax_segs', 'no_of_host_dom_pax_segs', 'itinerary_duration',
                        'no_of_confirmed_pax_segs', 'no_of_host_confirmed_pax_segs', 'no_of_pax', 'no_of_pax_segs', 'no_of_host_pax_segs', 'advance_bkg_days']
    
    enrichmentChord = ['InboundToBoarding_Distance', 'UTCdateTimeDifferenceMinutes']

    megaChord = targetField + primaryDatetimesChord + characterChord + dateChord + dateTimeChord + encodeChord + normalizeChord

    fieldsToRemove = ['destination_country', 'destination_airport']

    removeFromChord(encodeChord, fieldsToRemove)

    heftySpout.setDataStream(megaChord)
    heftySpout.toIntegers('mct_violation')
    heftySpout.setTargetDataFromField('mct_violation')
    megaChord = removeFromChord(megaChord, ['mct_violation'])

    #little bit of enriching haha
    distanceList = heftySpout.getDistanceList('inbound_airport', 'board_point')

    heftySpout.addData('InboundToBoarding_Distance', distanceList)
    heftySpout.nullToZero('InboundToBoarding_Distance')

    #requires the field for airports (to find timezone) as well as the datetime field.
    heftySpout.setUTCDateTimes('inbound_airport','inbound_arrival_datetime', 'UTC_inbound_arrival_datetime')
    heftySpout.setUTCDateTimes('board_point', 'departure_datetime', 'UTC_departure_datetime')

    heftySpout.setDateTimeDifferences('UTC_departure_datetime', 'UTC_inbound_arrival_datetime', 'UTCdateTimeDifferenceMinutes')
    heftySpout.nullToZero('UTCdateTimeDifferenceMinutes')

    heftySpout.datetimesToMinuteOfDay('UTC_departure_datetime')
    heftySpout.datetimesToMinuteOfDay('UTC_inbound_arrival_datetime')

    megaChord.append('InboundToBoarding_Distance')
    megaChord.append('UTCdateTimeDifferenceMinutes')

    heftySpout.tanhNormaliseChord(enrichmentChord)

    #find the days until boarding for chord because we are not considering timezones there is an error margin of 1 day
    heftySpout.dateTimeChordDaysUntilBoarding(dateTimeChord)
    
    #tanh normalise these fields now they have been converted into relative time
    heftySpout.tanhNormaliseChord(dateTimeChord)

    #standardize the times to boarding
    heftySpout.dateChordDaysUntilBoarding(dateChord)
    heftySpout.tanhNormaliseChord(dateChord)

    heftySpout.chordToMinutesOfDay(primaryDatetimesChord)
    heftySpout.tanhNormaliseChord(primaryDatetimesChord)

    #expensive operation to save dataflow
    #megaSpout.saveDataflowToCSV(r"C:\Users\Jo Ming\Documents\AirevoWorkspace\AirevoCode-Space\Project1MinimumConnectionErrorModel\TrainedModels\hughHeftyModel\pnrXsegmentDatastream.csv") 
    
    heftySpout.encodeChord(encodeChord)
    heftySpout.tanhNormaliseChord(encodeChord)

    #convert each strand to integers 
    heftySpout.typeCastChordToInt(normalizeChord)
    heftySpout.tanhNormaliseChord(normalizeChord)
    #heftySpout.tanhNormaliseChord(encodeChord)

    heftySpout.evaluateDataStream()
    heftySpout.evaluateTargetData()


    print(len(heftySpout.dataStream.keys()))


    dataflow, targetClasses = heftySpout.getDataflowTensor()
    trainingData, testingData, trainingClasses, testingClasses = heftySpout.splitDataflowAndTargetClasses(dataflow, targetClasses)

    model = tanhModel(megaChord,trainingData, testingData, trainingClasses, testingClasses, epochs = 100, batch_size = 50, saveName = "train500DropModel0.001")
    #saveModel(model, heftySpout, "testModelSave")

def slimModel():
    #create our dataspout first.
    trainingSpout = dataSpout('pnrSegmentPassenger', 'max')

    #state out desired fields catagorised into chords.

    targetField = ['mct_violation']

    depArrChord = ['departure_datetime', 'arrival_datetime']

    datetimeChord = ['document_birthdate', 'inbound_arrival_datetime']

    quantiveChord = ['day_of_week', 'no_in_party', 'booked_connection_time', 'minimum_connection_time', 'time_under_over', 
                     'equipment', 'advance_bkg_days']

    chordToEncode = ['board_point', 'off_point', 'trip_type','domestic_international', 'gender_code', 'current_status', 
                     'class', 'inbound_route','inbound_airport', 'segment_type']
    
    #we can display an example entry if wish.
    #trainingSpout.showXEntries(1)

    #the metaChord/completeChord being the sum of each stated subChord.
    completeChord = targetField + datetimeChord + quantiveChord + chordToEncode + depArrChord

    #populate our dataStream
    trainingSpout.setDataStream(completeChord)
    trainingSpout.encodeStrand('mct_violation') #convert the strand from chars of '0' or '1' to integer 0 or 1
    trainingSpout.setTargetDataFromField('mct_violation') #make this field our target stream, removing it from the dataStream.
    removeFromChord(completeChord, targetField)#take it out the complete chord for when we call the normalise function.

    #data that will be added to enrich our data.
    enrichmentChord = ['InboundToBoarding_Distance', 'UTCdateTimeDifferenceMinutes']
    UTCChord = ['UTC_inbound_arrival_datetime', 'UTC_departure_datetime'] #to convert to minute of day.

    #little bit of enriching. The function could definitely be modified to do the second two functions for you.
    distanceList = trainingSpout.getDistanceList('inbound_airport', 'board_point')
    trainingSpout.addData('InboundToBoarding_Distance', distanceList)
    trainingSpout.nullToZero('InboundToBoarding_Distance')

    #requires the field for airports (to find timezone) as well as the datetime field.
    trainingSpout.setUTCDateTimes('inbound_airport','inbound_arrival_datetime', 'UTC_inbound_arrival_datetime')
    trainingSpout.setUTCDateTimes('board_point', 'departure_datetime', 'UTC_departure_datetime')

    trainingSpout.setDateTimeDifferences('UTC_departure_datetime', 'UTC_inbound_arrival_datetime', 'UTCdateTimeDifferenceMinutes')
    trainingSpout.nullToZero('UTCdateTimeDifferenceMinutes') 

    #find the days until boarding for chord because we are not considering timezones there is an error margin of 1 day
    trainingSpout.dateTimeChordDaysUntilBoarding(datetimeChord) 
    #datetimeToMinuteOfDay() must be called after dateTimedaysUntilBoarding()
    trainingSpout.datetimeChordToMinuteOfDay(depArrChord) 

    #convert all the quantitative fields into type int.
    trainingSpout.typeCastChordToInt(quantiveChord)

    #evaluate each fields
    trainingSpout.evaluateDataStream()
    #evaluatetargets
    trainingSpout.evaluateTargetData()

    #encord our desired chord
    trainingSpout.encodeChord(chordToEncode)

    completeChord = completeChord + enrichmentChord + UTCChord

    print(len(trainingSpout.dataStream.keys()))

    dataflow, targetClasses = trainingSpout.getDataflowTensor()
    trainingData, testingData, trainingClasses, testingClasses = trainingSpout.splitDataflowAndTargetClasses(dataflow, targetClasses)
    model = tanhModel(completeChord,trainingData, testingData, trainingClasses, testingClasses, epochs = 30, batch_size = 50, saveName = "slimModel")

def slimModel2():
    #create our dataspout first.
    trainingSpout = dataSpout('pnrSegmentPassenger', 'max')

    #state out desired fields catagorised into chords.

    targetField = ['mct_violation']

    depArrChord = ['departure_datetime', 'arrival_datetime']

    datetimeChord = ['inbound_arrival_datetime']

    dateChord = ['document_birthdate']

    quantiveChord = ['day_of_week', 'no_in_party', 'booked_connection_time', 'equipment', 'advance_bkg_days']

    chordToEncode = ['board_point', 'off_point', 'trip_type','domestic_international', 'gender_code', 'current_status', 
                     'class', 'inbound_route','inbound_airport', 'segment_type', 'inbound_trip_type', 'inbound_op_carrier', 'op_booking_class']
    
    #we can display an example entry if wish.
    #trainingSpout.showXEntries(1)

    #the metaChord/completeChord being the sum of each stated subChord.
    completeChord = targetField + datetimeChord + dateChord + quantiveChord + chordToEncode + depArrChord

    #populate our dataStream
    trainingSpout.setDataStream(completeChord)
    trainingSpout.encodeStrand('mct_violation') #convert the strand from chars of '0' or '1' to integer 0 or 1
    trainingSpout.setTargetDataFromField('mct_violation') #make this field our target stream, removing it from the dataStream.
    removeFromChord(completeChord, targetField)#take it out the complete chord for when we call the normalise function.

    #data that will be added to enrich our data.
    enrichmentChord = ['InboundToBoarding_Distance', 'UTCdateTimeDifferenceMinutes']
    UTCChord = ['UTC_inbound_arrival_datetime', 'UTC_departure_datetime'] #to convert to minute of day.

    #little bit of enriching. The function could definitely be modified to do the second two functions for you.
    distanceList = trainingSpout.getDistanceList('inbound_airport', 'board_point')
    trainingSpout.addData('InboundToBoarding_Distance', distanceList)
    trainingSpout.nullToZero('InboundToBoarding_Distance')

    #requires the field for airports (to find timezone) as well as the datetime field.
    trainingSpout.setUTCDateTimes('inbound_airport','inbound_arrival_datetime', 'UTC_inbound_arrival_datetime')
    trainingSpout.setUTCDateTimes('board_point', 'departure_datetime', 'UTC_departure_datetime')

    trainingSpout.setDateTimeDifferences('UTC_departure_datetime', 'UTC_inbound_arrival_datetime', 'UTCdateTimeDifferenceMinutes')
    trainingSpout.nullToZero('UTCdateTimeDifferenceMinutes') 

    trainingSpout.dateChordDaysUntilBoarding(dateChord) #for D.O.B
    #find the days until boarding for chord because we are not considering timezones there is an error margin of 1 day
    trainingSpout.dateTimeChordDaysUntilBoarding(datetimeChord + UTCChord) 
    #datetimeToMinuteOfDay() must be called after dateTimedaysUntilBoarding() as this changes the values of the fields
    trainingSpout.datetimeChordToMinuteOfDay(depArrChord) 

    #convert all the quantitative fields into type int.
    trainingSpout.typeCastChordToInt(quantiveChord)

    #evaluate each fields
    #trainingSpout.evaluateDataStream()
    #evaluatetargets
    #trainingSpout.evaluateTargetData()

    #encord our desired chord
    trainingSpout.encodeChord(chordToEncode)

    completeChord = completeChord + enrichmentChord + UTCChord

    trainingSpout.nullToZeroChord(completeChord)
    trainingSpout.tanhNormaliseChord(completeChord)

    print(len(trainingSpout.dataStream.keys()))

    dataflow, targetClasses = trainingSpout.getDataflowTensor()
    trainingData, testingData, trainingClasses, testingClasses = trainingSpout.splitDataflowAndTargetClasses(dataflow, targetClasses)
    #model = tanhModel(completeChord,trainingData, testingData, trainingClasses, testingClasses, epochs = 30, batch_size = 50, saveName = "slimModel2")
    model = slimModel(completeChord,trainingData, testingData, trainingClasses, testingClasses, epochs = 30, batch_size = 50, saveName = "slimModel2_testsave")
    #saveModel(model, trainingSpout, "test2")





def shapTraining():
    #load our model
    theModel = Model("testModel", "TrainedModels\hughHeftyModel")

    bigFeeder = Feeder()
    dataflow, targetClasses, chord = bigFeeder.loadDataHeftyModeldata('allConnectionsPnrXsegment', 100, False, True)
    
    theModel.getConfidenceValues(dataflow)

    shap.initjs() #initialise visualiser
    explainer = shap.KernelExplainer(theModel.model.predict,dataflow)#make our explainer
    shap_values = explainer.shap_values(dataflow,nsamples=100)
    shap.summary_plot(shap_values,dataflow,feature_names=chord)

    return 0

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

    model.save(modelPath + str(saveName)) #save the model
    print("model saved. :)")
  
    #make a folder for the training data. This may not be needed. will hold the training data, chord, and keys
    Path(dataPath).mkdir(parents=True, exist_ok=True)

    dataSpout.saveDataflowToCSV((dataPath + "/dataflow.csv")) #save the training dataSpout.
    dataSpout.saveDataflowFields((dataPath + "/fields.csv")) #save the chord.
    dataSpout.saveCipherKeys((dataPath + "/cipherKeys.csv")) #save the cipher keys the model was trained on.



# Code of your program here
#training()
#evaluate()
#loadSegmentXpnr()
#everything()
#everythingPlusEnrichment()
#shapTraining()
#slimModel()
slimModel2()


