from dataSpout import dataSpout
from neuralNetworks import *

def main():
    mctFullSpout = dataSpout('bensData', 'max')
    allFields = mctFullSpout.getAllTableFields()
    chordToEncode = ['board_point','off_point','inbound_airport', 'trip_type']

    print(allFields)
    
    #set data stream
    mctFullSpout.setDataStream(allFields)
    mctFullSpout.popStrand('pnr_no_unique')

    #enumerate and set target data
    mctFullSpout.evaluateField('mct_violation')
    mctFullSpout.toIntegers('mct_violation')
    mctFullSpout.evaluateField('mct_violation')
    mctFullSpout.setTargetDataFromField('mct_violation')
    
    #requires the field for airports (to find timezone) as well as the dattimne field.
    mctFullSpout.setUTCDateTimes('inbound_airport','inbound_arrival_datetime', 'UTC_inbound_arrival_datetime')
    mctFullSpout.setUTCDateTimes('board_point', 'departure_datetime', 'UTC_departure_datetime')

    mctFullSpout.setDateTimeDifferences('UTC_departure_datetime', 'UTC_inbound_arrival_datetime', 'UTCdateTimeDifferenceMinutes')
    mctFullSpout.nullToZero('UTCdateTimeDifferenceMinutes')

    distanceList = mctFullSpout.getDistanceList('inbound_airport', 'board_point')

    mctFullSpout.addData('InboundToBoarding_Distance', distanceList)
    mctFullSpout.nullToZero('InboundToBoarding_Distance')

    #encode a strand to integers.
    #mctFullSpout.evaluateDataStream()
    mctFullSpout.encodeAndNormaliseChord(chordToEncode)
    #mctFullSpout.evaluateDataStream()

    #get data to be parsed into model
    trainingData, testingData, trainingClasses, testingClasses = mctFullSpout.getDataflow()
    #set target data attribute
    trainModel3(trainingData, testingData, trainingClasses, testingClasses, epochs = 50, batch_size = 30)
    #enumerate subcord.

    return 0

main()