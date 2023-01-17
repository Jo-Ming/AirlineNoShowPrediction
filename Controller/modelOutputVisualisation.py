#This is so we can import from the parent directory.
import os, sys

p = os.path.abspath('.')
sys.path.insert(1, p)

from Tools.Feeder import Feeder
import shap
from Tools import controlHelper
from Tools.dataSpout import dataSpout
from Tools.Model import Model
import pandas as pd
import numpy as np
from sklearn.utils import shuffle #to randomize the data order when training
#
def main():
    #confidenceStrand()
    workingShapPlots()
    #predictionBreakdown()
    
def workingShapPlots():
    #first, lets get some data. just using mct data for now until I get no show data.
    modelFeeder = Feeder('CSVFiles\CSVFiles2.csv', 42) #for say 30 values
    dataflow, targetClasses, chord = modelFeeder.loadNoShowData() #get dataflow and targetField and applies some data enrichment
    mctModel = Model("noshowModel", "savedModels\\model1_100Epoch_seed5_9795")

    #lets explain som of the models predictions using shap
    
    shapData = np.array(dataflow[:10], dtype=np.int32) #for shap alone we could use pandas.DataFrame() however, keras only takes numpy arrays.
    #shapData = np.array(dataflow[:42]) #for shap alone we could use pandas.DataFrame() however, keras only takes numpy arrays.
    explainer = shap.KernelExplainer(mctModel.model,shapData)#make our explainer
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

def predictionBreakdown():
    #first, lets get some data. just using mct data for now until I get no show data.
    modelFeeder = Feeder('CSVFiles\CSVFiles2.csv', 100) #for say 30 values
    dataflow, targetClasses, chord = modelFeeder.loadNoShowData() #get dataflow and targetField and applies some data enrichment
    mctModel = Model("noshowModel", "savedModels\\NowShow_2.5M_50Epoch_sigmoid_shuffle1\\model")

    #next will be to get the predictions for each of these 100 entries.
    confidenceList = mctModel.getConfidenceValues(dataflow)
    predictionList = mctModel.getBinaryPredictions(confidenceList) #gives us classes fro the confidence values.

    mctModel.plotConfusionMatrix(targetClasses, predictionList, classnames=['Show', 'No Show'])

    #lets show a breakdown
    return 0

def confidenceStrand():
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
    
    #first lets get the training data
    dataflow = Feeder('CSVFiles\CSVFiles2.csv' , 180000)
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
    #dataflow.evaluateTargetData()
    dataflow.tanhNormaliseChord(enumerateFields + fieldsToInt + dateTimeFields + dateFields + enrichmentChord + UTCChord)

    tensor, targetClasses = dataflow.getDataflowTensor()

    noShowModel = Model("NoShowModel", "savedModels\\NowShow_2M_15Epoch_sigmoid_shuffle1\\model")

    confidenceStrand = noShowModel.getConfidenceValues(tensor)

    csvFeeder = Feeder('CSVFiles\CSVFiles2.csv' , 180000)
    csvFeeder.setDataStream(targetField + enumerateFields + fieldsToInt + dateTimeFields + dateFields)
    csvFeeder.typeCastChordToInt(fieldsToInt)

    csvFeeder.distanceTimeEnrichment()
    csvFeeder.addData("Model_Confidence", confidenceStrand)

    csvFeeder.saveDataflowToCSV("CSVFiles\\dataStreamWithConfidence180K.csv")

    
main()

