from Tools.dataSpout import dataSpout  # feeder inherits from dataspout
from sklearn.utils import shuffle  # to randomize the data order when training

"""In order to feed a model we will need:
    1. the relavent .csv file
    2. chord
    3. data processing method
    """

# identity number
pnrUnique = ['pnr_unique']

# these are all the fields is the chord excluding pnr_unique
chord = ['nosho', 'cancelled', 'seg_cancelled', 'pax_cancelled', 'pnr_status', 'no_in_party',
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

# first lets seperate the fields into groups.

# targetVariable.
targetField = ['nosho']

# The violations found through the current logic approaches.
violationFields = ['pos_violation', 'group_violation', 'fake_name_violation', 'test_booking',
                'missing_ttl', 'ttl_incorrect', 'duplicate', 'hidden_group_flag', 'marriage_violation',
                'mct_violation', 'fake_name_violation_match', 'fake_name_violation_match_name']
# fields such with strings/classes which contain useful data but arn't yet in a numerical form.
enumerateFields = ['domestic_international', 'class', 'board_point', 'off_point', 'inbound_airport', 'inbound_route',
                     'op_booking_class', 'gender_code', 'passenger_type_code', 'passenger_type']
# status fields (which mostly only have 1 value)
statusFields = ['test_passenger', 'cancelled',
    'seg_cancelled', 'pax_cancelled', 'pnr_status']
# These fields are fields that I probably won't use for training a model.
tempEnumFields = ['mkt_carrier_code', 'mkt_flight_no',
    'op_carrier_code', 'op_flight_no']
# fields in the form of a string which we want in an integer form.
fieldsToInt = ['no_in_party', 'advance_booking_days', 'day_of_week', 'segment_distance', 'inbound_segment_no', 'equipment',
             'time_under_over', 'minimum_connection_time', 'booked_connection_time', 'inbound_arrival_datetime_utc_epoch', 'departure_datetime_epoch']
# dates + time fields
dateTimeFields = ['departure_datetime', 'inbound_arrival_datetime',
    'inbound_arrival_datetime_utc', 'departure_datetime_utc',  'departure_datetime_sys']
dateFields = ['document_birthdate']

# names of new fields being added
enrichmentChord = ['InboundToBoarding_Distance',
    'UTCdateTimeDifferenceMinutes']
# names of new times being added
UTCChord = ['UTC_inbound_arrival_datetime', 'UTC_departure_datetime']


class Feeder(dataSpout):

    # will enrich data using the relavent fields to convert times to UTC find the arrival to boarding time differences
    def mctTimeDistanceEnrichment(self, dataSpout):
        distanceList = dataSpout.getDistanceList('inbound_airport', 'board_point')

        dataSpout.addData('InboundToBoarding_Distance', distanceList)
        dataSpout.nullToZero('InboundToBoarding_Distance')

        # requires the field for airports (to find timezone) as well as the dattimne field.
        dataSpout.setUTCDateTimes('inbound_airport', 'inbound_arrival_datetime', 'UTC_inbound_arrival_datetime')
        dataSpout.setUTCDateTimes('board_point', 'departure_datetime', 'UTC_departure_datetime')

        dataSpout.setDateTimeDifferences('UTC_departure_datetime', 'UTC_inbound_arrival_datetime', 'UTCdateTimeDifferenceMinutes')
        dataSpout.nullToZero('UTCdateTimeDifferenceMinutes')
        return dataSpout

    # takes a list of dirty fields and removes all of those instances from the parsed chord.
    def cleanChord(chord, dirtyFields):
        for dirt in dirtyFields:
            i = 0
            for note in chord:
                if(dirt == note):
                    chord.pop(i)
                i += 1
        return chord

    def decryptStrand(self, field):
        encryptedStrand = self.getStrand(field)
        dictionary = self.fieldKeys.get(field)

        decryptedStrand = []

        for element in encryptedStrand:
            token = dictionary.get(element)
            decryptedStrand.append(token)

    def spliceStrandAtIndex(self, field, index):
        strand = self.getStrand(field)
        leftStrand = strand[0:index]
        rightStrand = strand[index:]
        return leftStrand, rightStrand

    # takes two lists and removes the elements of the second list from the first.
    def removeFromChord(self, chord, removeList):
        for key in removeList:
            pointer = 0
            for note in chord:
                if key == note:
                    chord.pop(pointer)
                    print(str(key) + " Popped from chord.")
                    break
                pointer += 1
        return chord

    def distanceTimeEnrichment(self):
        """
        This enrichment will perform the following to the dataStream of a given feeder
        1. adds a distance list based on pythagorean distence between 2 airports. (based on long, lat) as fieldname: 'InboundToBoarding_Distance'
        sets the Null values in this field to 0.
        2. sets inbound_arrival_datetime to UTC using the inbound airport timezone as fieldname : 'UTC_inbound_arrival_datetime'
        3. sets the departure datetimes to UTC using the board point airport timezones as fieldname: 'UTC_departure_datetime'
        4. creates a new strand containing the time differences in minutes between the 2 UTC time fields as fieldname: 'UTCdateTimeDifferenceMinutes'
        5. creates the following new fields from the utc date time fields:

        """
        try:
            self.getDaysUntilBoardingDate('document_birthdate')

            # here some new fields to add enrichment
            # names of new fields being added
            enrichmentChord = ['InboundToBoarding_Distance',
                            'UTCdateTimeDifferenceMinutes']
            # names of new times being added
            UTCChord = ['UTC_inbound_arrival_datetime', 'UTC_departure_datetime']

            # uses extremely basic means to find distance between coordinates.
            # will return a strand containing distances between airport a and b.
            distanceList = self.getDistanceList('inbound_airport', 'board_point')
            # add the data to our dataStream.
            self.addData('InboundToBoarding_Distance', distanceList)
            # will set any NULL or '' values to 0 along that strand of the dataStream.
            self.nullToZeroFloat('InboundToBoarding_Distance')

            # requires the field for airports (to find timezone) as well as the datetime field.
            self.setUTCDateTimes('inbound_airport', 'inbound_arrival_datetime',
                                'UTC_inbound_arrival_datetime')  # will convert the dateTime to UTC
            # new field with UTC datetimes
            self.setUTCDateTimes(
                'board_point', 'departure_datetime', 'UTC_departure_datetime')

            # will set a new field with the time difference in minutes between the two given datetimes
            self.setDateTimeDifferences(
                'UTC_departure_datetime', 'UTC_inbound_arrival_datetime', 'UTCdateTimeDifferenceMinutes')
            self.nullToZero('UTCdateTimeDifferenceMinutes')

            # converts these datetimes to minute of the day
            self.datetimeChordToMinuteOfDay(UTCChord)
            self.datetimeChordToMinuteOfDay(['departure_datetime', 'departure_datetime_utc',
                                            'departure_datetime_sys', 'inbound_arrival_datetime_utc', 'inbound_arrival_datetime'])
            self.nullToZeroChord(['departure_datetime', 'departure_datetime_utc',
                                'departure_datetime_sys', 'inbound_arrival_datetime_utc', 'inbound_arrival_datetime'])
            self.nullToZeroChord(UTCChord)  # null values to 0
        except:
            print("Error, Failed to enrich the data.")

    def loadNoShowData(self):
        # ok lets do a big heat map
        self.setDataStream(targetField + enumerateFields +
                           fieldsToInt + dateTimeFields + dateFields)
        self.typeCastChordToInt(fieldsToInt)

        self.distanceTimeEnrichment()

        # evaluate the dataStream
        # dataflow.evaluateDataStream()

        self.encodeChord(enumerateFields + targetField)

        # set the target Label
        self.setTargetDataFromField('nosho')
        # evaluate the targetdata
        self.evaluateTargetData()

        tensor, targetClasses = self.getDataflowTensor()
        return tensor, targetClasses, chord

    def loadNSSeed(poolSize, seedNumber, location):
            # first lets get the training data
        dataflow = Feeder(location, poolSize)
        fullChord = dataflow.getAllTableFields()
        print(fullChord)

        # set the datastream
        dataflow.setDataStream(
            targetField + enumerateFields + fieldsToInt + dateTimeFields + dateFields)
        dataflow.typeCastChordToInt(fieldsToInt)

        dataflow.distanceTimeEnrichment()

        # evaluate the dataStream
        # dataflow.evaluateDataStream()

        dataflow.encodeChord(enumerateFields + targetField)

        # set the target Label
        dataflow.setTargetDataFromField('nosho')
        # evaluate the targetdata
        dataflow.evaluateTargetData()
        dataflow.tanhNormaliseChord(
            enumerateFields + fieldsToInt + dateTimeFields + dateFields + enrichmentChord + UTCChord)

        tensor, targetClasses = dataflow.getDataflowTensor()

        trainingData, testingData, trainingClasses, testingClasses = dataflow.splitDataflowAndTargetClasses(
            tensor, targetClasses)

        trainingData_shuffled, trainingClasses_shuffled = shuffle(
            trainingData, trainingClasses, random_state=seedNumber)
        testingData_shuffled, testingClasses_shuffled = shuffle(
            testingData, testingClasses, random_state=seedNumber)

        return trainingData_shuffled, trainingClasses_shuffled, testingData_shuffled, testingClasses_shuffled

    def loadRandomNSEntries(self, poolSize, seedNumber, location):
        # first lets get the training data
        dataflow = Feeder(location, poolSize)
        fullChord = dataflow.getAllTableFields()

        entryList = dataflow.nodeSlot.entries

        entryList = shuffle(entryList, random_state = seedNumber)

        return entryList
    
     # temporary function to demo shap. Will load the training data

    def loadTrainingData(self, quantity):
        # first lets get the training data
        dataflow = Feeder('C:\\Users\\Jo Ming\\Documents\\AirevoWorkspace\\AirevoCode-Space\\NoShowModel\\CSVFiles\\CSVFiles2.csv' , quantity)

        # set the datastream
        dataflow.setDataStream(targetField + enumerateFields + fieldsToInt + dateTimeFields + dateFields)
        dataflow.typeCastChordToInt(fieldsToInt)

        dataflow.distanceTimeEnrichment()

        # evaluate the dataStream
        # dataflow.evaluateDataStream()

        dataflow.encodeChord(enumerateFields + targetField)

        # set the target Label
        dataflow.setTargetDataFromField('nosho')
        # evaluate the targetdata 
        dataflow.evaluateTargetData()
        dataflow.tanhNormaliseChord(enumerateFields + fieldsToInt + dateTimeFields + dateFields + enrichmentChord + UTCChord)

        tensor, targetClasses = dataflow.getDataflowTensor()

        trainingData, testingData, trainingClasses, testingClasses = dataflow.splitDataflowAndTargetClasses(tensor, targetClasses)

        trainingData_shuffled, trainingClasses_shuffled = shuffle(trainingData, trainingClasses, random_state = 1)
        testingData_shuffled, testingClasses_shuffled = shuffle(testingData, testingClasses, random_state = 1)

        fullChord = dataflow.getDataflowFields()

        return testingData_shuffled, testingClasses_shuffled, fullChord
    
    def entryListToDataStream(self, entryList):
        return 0