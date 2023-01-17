"""
This class remote will designate information channels. info sources will change in the future. plus there are soooooo many tables

The idea is for the data coming out of the spout to be parsed in as dataflow for a model. 

we should never edit the original entries themselves so any transformations/manipulations should
show on the data streams. This way we can also validate the manipulated data in the future.

Update: if we store the datsream as a dictionary contaiin
"""
from tableNode import tableNode
import numpy as np
from sklearn.model_selection import train_test_split
from spoutHelperFunctions import *
import csv
import os

class dataSpout():

        def __init__(self, channel, streamSize):
                self.channelBank = {
                                'pnrSegmentPassenger' :'TheQuantileWorld\segmentXpnrXpassenger_AllConnections.csv',
                                'allConnectionsPnrXsegment': 'TheQuantileWorld\connections.csv',
                                'segmentXpnrMoreMct'      :'TheQuantileWorld\pnrXsegmentMoreMCT.csv',
                                'segmentXpnr_more_mct'      :'TheQuantileWorld\segmentXpnr_moreMCT.csv',
                                'segmentXpnr_500K'      :'TheQuantileWorld\segmentXpnr_500K.csv',
                                'segments_600K'    :'TheQuantileWorld\segments_600K.csv',
                                'evenTestSegment'  : 'TheQuantileWorld\even_mct_test.csv',
                                'bensData'         : 'TheQuantileWorld\mct_full.csv',
                                'segment_5percMCT' : 'TheQuantileWorld\segment_5percMCT.csv',
                                'joseph.Even_segment_160K' : 'TheQuantileWorld\even_segment_160K.csv',
                                'sand.passenger' : 'TheQuantileWorld/passenger_1K.csv',
                                'sand.segment' : 'TheQuantileWorld/segment_1K.csv',
                                'sand.pnr' : 'TheQuantileWorld/pnr_1K.csv',
                                'sand.seat' : 'TheQuantileWorld/seat_1K.csv',
                                'mirage.mirage_mct' : 'TheQuantileWorld/mct_1K.csv',
                                'mirage.mirage_mct_detail' : 'TheQuantileWorld/mct_detail_1K.csv',
                                'sand.airport' : 'TheQuantileWorld/airport.csv', #The entire airport field is 7248 entries
                                'cdd_raw' : 'TheQuantileWorld/cdd_raw_1k.csv',
                                'joseph.segment' : 'TheQuantileWorld/josephPNRSxSegment.csv',
                                'joseph.segment_50K' : 'TheQuantileWorld\josephPNRSxSegment_50K.csv',
                                'joseph.segment_150K' : 'TheQuantileWorld\PNRSxSegment_150K.csv',
                                'joseph.segment_500K' : 'TheQuantileWorld\josephPNRSxSegment_500K.csv'
                                }

                self.currentChannel = channel   
                self.setChannel(channel, streamSize)
                print('Data spout stream size set to ' + str(streamSize))

        def setChannel(self, channel, streamSize):
                if channel in self.channelBank:
                        self.currentChannel = channel #this is here in case we want to change the channel in the future
                else: 
                        self.currentChannel = channel # if someone wished to use data outside of the channel bank.
                self.nodeSlot = self.getTableNode() #node slot holds object retrieve data from a table.
                self.nodeSlot.setNodeProperties(streamSize) #set its attributes.
                print("TableNode has been created in spout nodeSlot. Properties: ")
                self.nodeSlot.showProperties()

        def setDataStream(self, chordKey):
                print("\nCreating data Stream...")
                self.dataStream = {}
                self.fieldKeys = {} #this is for storing keys when we enumerte fields later.
                self.meanDeviations = {} #save these whilst normalising to save compuation later.
                for key in chordKey:
                        print("Populating data stream for '" + key + "' field")
                        column = self.nodeSlot.getColumn(key)
                        if (column[0] == None):
                                print('Warning. ' +  str(key) + ' is returning None. please check field name or reference data')
                        self.addData(key, column)

                print("data stream set.")
        
        #return all the fields for dataflow.
        def getDataflowFields(self):
                return list(self.dataStream.keys())
        
        #returns the all the fields from our reference table. 
        def getAllTableFields(self):
                fieldList = self.nodeSlot.getFields()
                return fieldList

        def getFileLocation(self, key):
                #if someone wants to use data outside of the bank
                if key in self.channelBank:             
                        return self.channelBank.get(key)
                else:
                        return key

        #will return a table node using the channel as the key for our current channelbank
        def getTableNode(self):
                location = self.getFileLocation(self.currentChannel)
                _tableNode = tableNode(location)
                return _tableNode
        
        #we will get a 'stream'(a strand) of data but for only one field across entries
        def getStrand(self, field):    
                strand = self.dataStream.get(field)  
                #some of the lists allow this operation and some don't. Will need to look at what the difference is in
                if(strand == 'NoneType'):
                        print(str(field) + "this strand is empty, check what strand we are calling.")   
             
                return strand
        
        def popStrand(self, fieldName):
                print("\n" + fieldName + ' Data has been popped from dataStream.')
                return self.dataStream.pop(fieldName)
        
        def showXEntries(self, quantity):
                #display entry values returned for first 10 entres
                print("\nFor " + str(quantity) + ":")
                print("\nShowing entries...")
                entryList = self.nodeSlot.getEntries(quantity)
    
                for entry in entryList:
                        entry.display()

        #make sure spout.setStreams() has been called first
        def getChord(self, fieldList): #chordsize by number of entries
                chord = []
                for field in fieldList:
                        chord.append(self.getStrand(field))
                return chord

        #will add or replace a key and value in the dataStream.
        def addData(self, field, dataStrand):
                print("Adding " + field + " to data to datastream...")
                self.dataStream.__setitem__(field, dataStrand)
                print("Success, strand length " + str(len(dataStrand)))

        def updateDataStrand(self, field, newStrand):
                print("Updating " + str(field) + " Strand...")
                self.dataStream.update({field:newStrand})
                print("Update Complete")
        
        def typeCastChordToInt(self, chord):
                for key in chord:
                        strand = self.getStrand(key)
                        output = []
                        for element in strand:
                                try:
                                        if (element == '' or element == 'NULL'):
                                                output.append(0)
                                        else: 
                                                output.append(int(element))
                                except:
                                        print("Unable to convert " + str(key) + " field to int.")
                        self.dataStream.update({key: np.asarray(output)})
                        
        
        
        #will find the differences between 2 time strands in the format "HH:MM"/hours:minutes
        #will also remove the parsed strands.
        def setTimeDifferences(self, timeField1, timeField2, newFieldName):
                #get time strands
                timeStrand1 = self.dataStream.pop(timeField1)
                timeStrand2 = self.dataStream.pop(timeField2)

                #find time differences.
                timeDifferences = []

                i = 0                
                for i in range(len(timeStrand1)):
                        timeDifferences.append(getTimeDifference(timeStrand1[i], timeStrand2[i]))
                
                self.dataStream.__setitem__(newFieldName, timeDifferences)
        
        def setDateTimeDifferences(self, dateTimeField1, dateTimeField2, newFieldName):
                #get strands
                #dateTimeStrand1 = self.dataStream.pop(dateTimeField1)
                #dateTimeStrand2 = self.dataStream.pop(dateTimeField2)
                dateTimeStrand1 = self.getStrand(dateTimeField1)
                dateTimeStrand2 = self.getStrand(dateTimeField2)

                #find time differences.
                dateTimeDifferences = []

                i = 0                
                for i in range(len(dateTimeStrand1)):
                        dateTimeDifferences.append(getDateTimeDifference(dateTimeStrand1[i], dateTimeStrand2[i]))
                
                self.dataStream.__setitem__(newFieldName, dateTimeDifferences)
        
        def datetimeChordToMinuteOfDay(self, chord):
                for key in chord:
                        self.datetimesToMinuteOfDay(key)
        
        def datetimesToMinuteOfDay(self, dateTimeField):
                dateTimeStrand = self.dataStream.pop(dateTimeField)
                dates, times = splitDateTime(dateTimeStrand)

                #now we conver the times to minutes of the day
                minutesOfTheDay = timesToMinutesOfDay(times)

                self.dataStream.__setitem__((dateTimeField), minutesOfTheDay)
        
        def chordToMinutesOfDay(self, chord):
                for key in chord:
                        self.datetimesToMinuteOfDay(key)
        
        def dateTimeChordDaysUntilBoarding(self, chord):
                for key in chord:
                        self.getDaysUntilBoardingDateTime(key)

        def dateChordDaysUntilBoarding(self, chord):
                for key in chord:
                        self.getDaysUntilBoardingDate(key)

        def dateChordDaysfrom(self, chord, dateFrom):
                for key in chord:
                        self.getDaysFromDate(key, dateFrom)
                        
        
        def getDaysFromDate(self, dateField, dateFrom):
                dateStrand = self.dataStream.pop(dateField)

                daysFromStrand = []
                i = 0

                for i in range(len(dateStrand)):
                        daysFromStrand.append(getDaysFrom(dateStrand[i], dateFrom))
                        i+=1
                
                self.dataStream.__setitem__(dateField, daysFromStrand)

        def getDaysUntilBoardingDate(self, dateField):
                dateStrand = self.dataStream.pop(dateField)
                boardingDateTimeStrand = self.getStrand('departure_datetime')
                boardingDates, times = splitDateTime(boardingDateTimeStrand)

                daysToBoarding = []
                i = 0

                for i in range(len(dateStrand)):
                        daysToBoarding.append(getDaysUntilFromDates(dateStrand[i], boardingDates[i]))
                        i+=1
                self.dataStream.__setitem__(dateField, daysToBoarding)

        #not quite sure what to do about timezones here for now we will asuume all timezones are same as board point.
        def getDaysUntilBoardingDateTime(self, dateTimeField):
                dateTimeStrand = self.getStrand(dateTimeField)
                boardingDateTimeStrand = self.getStrand('departure_datetime')

                daysToBoarding = []
                i = 0

                for i in range(len(dateTimeStrand)):
                        daysToBoarding.append(getDaysUntil(dateTimeStrand[i], boardingDateTimeStrand[i]))
                        i += 1
                self.dataStream.__setitem__(dateTimeField, daysToBoarding)
                #self.dataStream.update({(dateTimeField): np.asarray(daysToBoarding, dtype=int)})
        
        #will need to convert the dataStream into a tensor
        def getDataflowTensor(self):
                tensor = []
                i = 0
                for key, value in self.dataStream.items():
                        pointer = 0
                        try:
                                if (i == 0): #if it is first iteration initialise tensor.
                                        for j in range(len(value)):
                                                tensor.append([])
                                for entry in value:
                                        if isinstance(entry, str):
                                                tensor[pointer].append(float(entry))
                                        else:
                                                tensor[pointer].append(entry)
                                        pointer += 1
                                i += 1
                        except:
                                print('strand in the ' + str(key) + ' field needs to be float or int before converting to tensor.')
                return tensor, self.targetData

        
        def setTargetDataFromField(self, targetField):
                self.targetData = []
                column = self.dataStream.pop(targetField)
                for entry in column:
                        self.targetData.append(int(entry))

        def setTargetDataFromTableColumn(self, targetField):
                self.targetData = []
                column = self.nodeSlot.getColumn(targetField)
                for entry in column:
                        self.targetData.append(int(entry))

        
        #evaluate just one field.
        def evaluateField(self, field):
                strand = self.getStrand(field)
                length = len(strand)
                print('\n'+ field + " Synopsis: ")
                print("Number of entrys in field: " + str(length))
                highest, lowest, nullCounter, uniqueEntries = strandAnalysis(strand)
                print("Lowest Value: " + str(lowest))
                print('Highest Value: ' + str(highest))
                print('Unique Entries: ' + str(len(uniqueEntries)))
                print('Null count: ' + str(nullCounter))
                print('Percentage Null: ' + str(round((nullCounter/length)*100, 3)))

                uniqueEntryCounter = sorted(uniqueEntries.items(), key=lambda x: x[1], reverse=True)
                uniqueEntryPercentage = []

                for entry in uniqueEntryCounter:
                        uniqueEntryPercentage.append((entry[0], (str( "Percentage: " + str(round(((entry[1]/length)*100), 5) )))))

                print(str(uniqueEntryPercentage))
                
                return highest, lowest, nullCounter, uniqueEntries, uniqueEntryCounter, uniqueEntryPercentage

        def evaluateTargetStrand(self, strand):
                length = len(strand)
                print('\n' "Target Data Synopsis: ")
                print("Number of entrys in field: " + str(length))
                highest, lowest, nullCounter, uniqueEntries = strandAnalysis(strand)
                print("Lowest Value: " + str(lowest))
                print('Highest Value: ' + str(highest))
                print('Unique Entries: ' + str(len(uniqueEntries)))
                print('Null count: ' + str(nullCounter))
                print('Percentage Null: ' + str(round((nullCounter/length)*100, 3)))

                uniqueEntryCounter = sorted(uniqueEntries.items(), key=lambda x: x[1], reverse=True)
                uniqueEntryPercentage = []

                for entry in uniqueEntryCounter:
                        uniqueEntryPercentage.append((entry[0], round(((entry[1]/length)*100), 3)))

                print(uniqueEntryPercentage)
                
                return highest, lowest, nullCounter, uniqueEntries, uniqueEntryCounter, uniqueEntryPercentage

        #will evaluate targetData
        def evaluateTargetData(self):
                try:
                        highest, lowest, nullCounter, uniqueEntries, uniqueEntryCounter, uniqueEntryPercentage = self.evaluateTargetStrand(self.targetData)
                except:
                        print('Must set self.targetData attribute.')

        #this function will show a synopsis of all the data.
        def evaluateDataStream(self):
                keywords = ['time', 'date', 'distance'] #evaluate is expensive so if it is a time or distance field do a cheaper evaluation
                containsNull = []
                containsOneValue = []
                for key in self.dataStream.keys(): #for each dictionary key.
                        contains = False
                        for keyword in keywords:
                                if (containsSubstring(key, keyword) == True):
                                        contains = True
                        if(contains): #if it is date or time i
                                strand = self.getStrand(key)
                                nullCounter = getNullCount(strand)
                                length = len(strand)

                                print('\n'+ key + ' synopsis: ')
                                print('Number of entrys in field: ' + str(length))
                                print('Null count: ' + str(nullCounter))
                                print('Percentage NULL: ' + str(round((nullCounter/length)*100, 5)))
                        else:
                                highest, lowest, nullCounter, uniqueEntries, uniqueEntryCounter, uniqueEntryPercentage = self.evaluateField(key)
                                if(uniqueEntries == 1):
                                        containsOneValue.append(key)
                        if (nullCounter>0):
                                containsNull.append(key)
                
                print('')
                if (len(containsNull)>0):
                        print('Fields contain Null values: ' + str(containsNull))
                if(len(containsOneValue)>0):
                        print("Fields only contain one value: " + str(containsOneValue))

                print('Number of fields: ' + str(len(self.dataStream)))

        def nullToZeroChord(self, chord):
                for key in chord:
                        self.nullToZero(key)
        
        def nullToZero(self, field):
                strand = self.dataStream.pop(field)
                output = []
                i = 0
                for entry in strand:
                        if (entry == 'NULL' or entry == ''):
                                output.append(0)
                        else:
                                output.append(int(entry))
                        i+=1
                self.addData(field, output)
        
        def nullToZeroFloat(self, field):
                strand = self.dataStream.pop(field)
                output = []
                i = 0
                for entry in strand:
                        if (entry == 'NULL' or entry == ''):
                                output.append(0)
                        else:
                                output.append(float(entry))
                        i+=1
                self.addData(field, output)
                
        def encodeStrand(self, field):
                strand = self.getStrand(field)

                if(strand == None or strand == 'NoneType'):
                        print(field + ' is not present in chord.')
                        return 0

                #now encode
                cipherKey, cipher = encodeAsIntegers(strand)
                
                #update the datastream 
                self.dataStream.update({field:np.asarray(cipher)})
                self.fieldKeys.__setitem__(field, cipherKey)
                #print(str(cipherKey))
        
        def normalizeStrand(self, fieldToNormalize):
                #get strand
                strand = self.getStrand(fieldToNormalize)
                #then normalise.
                normalisedStrand = normaliseStrand(strand)
                self.dataStream.update({fieldToNormalize:np.asarray(normalisedStrand)})
        
        def standardizeStrand(self, fieldToStandardize):
                #get strand
                strand = self.getStrand(fieldToStandardize)
                #then normalise.
                standardizedStrand = standardizeStrand(strand)
                self.dataStream.update({fieldToStandardize:np.asarray(standardizedStrand)})

        
        def splitDataflowAndTargetClasses(self, dataflow, targetClasses):
                return splitDataAndClasses(dataflow, targetClasses)
        
        def encodeAndStandardise(self, field):
                strand = self.getStrand(field)

                if(strand == None):
                        print(field + 'is not present in chord.')

                #now encode
                cipherKey, cipher = encodeAsIntegers(strand)

                #then normalise.
                normalisedStrand = standardiseStrand(cipher)
                
                #update the datastream 
                self.dataStream.update({field:np.asarray(normalisedStrand)})
                self.fieldKeys.__setitem__(field, cipherKey)

        def encodeChord(self, chord):
                for key in chord:
                        print("encoding " + str(key) + "...")
                        self.encodeStrand(key)
        
        def normalizeChord(self, chord):
                for key in chord:
                        strand = self.getStrand(key)
                        print("minmax normalizing " + str(key) + "...")
                        normalisedStrand = normaliseStrand(strand)
                        #update the datastream 
                        self.dataStream.update({key:np.asarray(normalisedStrand)})

        def standardizeChord(self, chord):
                for key in chord:
                        print("standardizing " + str(key) + "...")
                        strand = self.getStrand(key)
                        standardizedStrand = standardizeStrand(strand)
                        #update our dataStream
                        self.dataStream.update({key:np.asarray(standardizedStrand)})
        
        def tanhNormaliseChord(self, chord):
                for key in chord:
                        print("tanh normalizing " + str(key) + "...")
                        strand = self.getStrand(key)
                        normalizedStrand, meanDeviation = tanhNormalization(strand)
                        #update our datastream
                        self.dataStream.update({key:np.asarray(normalizedStrand)})
                        self.meanDeviations.update({key:meanDeviation})
        
        #get the strand of distances between 2 lists/strands of airports.
        def getDistanceList(self, IATAField1, IATAField2):
                IATAList1 = self.getStrand(IATAField1)
                IATAList2 = self.getStrand(IATAField2)

                coordList1 = getAirportCoordsList(IATAList1)
                coordList2 = getAirportCoordsList(IATAList2)
                distanceList = []

                for i in range(len(coordList1)):
                        coord1 = coordList1[i]
                        coord2 = coordList2[i]
                        if (coord1 == 'NULL' or coord2 == 'NULL'):
                                distanceList.append('NULL')
                        else:
                                distanceList.append(pythagorianDistance(coord1, coord2))
                return distanceList
        
        #need to declare te field containing IATA coedes and field containing datetimes.
        def getUTCDateTimes(self, IATAField, dateTimeField):
                airports = self.getStrand(IATAField)
                dateTimes = self.getStrand(dateTimeField)

                UTCDateTimes = strandToUTC(airports, dateTimes)
                return UTCDateTimes

        #will remove old dateTimeStrand and replace it with a new strand in UTC
        def setUTCDateTimes(self, IATAField, dateTimeField, newFieldName):
                #uses the IATA to find the timezone of the current datetime and creates the new field with UTC datetimes.
                UTCDateTimes = self.getUTCDateTimes(IATAField, dateTimeField)
                self.getStrand(dateTimeField)
                self.dataStream.__setitem__(newFieldName, UTCDateTimes)

        #we will use a datetime because this field is always not null.
        def ToMinuteOfDay(self, dateTimeField):
                #minute of day will be a value between 0 -> 1440
                dateTimeStrand = self.getStrand(dateTimeField)
                dates, times = splitDateTime(dateTimeStrand)

                minuteOfDayStrand = []
                for time in times:
                        minuteOfDay = minuteOfDay(time)
                        minuteOfDayStrand.append(minuteOfDay)
                
                return minuteOfDayStrand
        
         #save the chordKey
        
        def saveDataflowFields(self, saveLocation):
                print("Saving dataflow...")
                fields = self.getDataflowFields()

                mode = 'a' if os.path.exists(saveLocation) else 'w'
                with open(saveLocation, mode, newline='') as csvFile:  
                        writer = csv.writer(csvFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        fieldnames = list(self.dataStream.keys())

                        #write our header
                        writer.writerow(fieldnames)
                
                csvFile.close()
                print("Saving dataflow complete.")
        
        #save the chordKey
        def saveCipherKeys(self, saveLocation):
                
                mode = 'a' if os.path.exists(saveLocation) else 'w'
                with open(saveLocation, mode, newline='') as csvFile:  
                        writer = csv.writer(csvFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        fieldnames = list(self.fieldKeys.keys())

                        #write our header
                        writer.writerow(fieldnames)
                        
                        for field in self.fieldKeys:

                                dictionary = self.fieldKeys[field]
                                key = dictionary.keys()
                                values = dictionary.values()
                                
                                writer.writerow(key)
                                writer.writerow(values)            
                
                csvFile.close()
                print("Saving Cipher keys complete.")
        
        #save the parameters used for normalising to save re computing later.
        def saveMeanDeviations(self, saveLocation):
                print("Saving means and standard deviations...")
                mode = 'a' if os.path.exists(saveLocation) else 'w'
                with open(saveLocation, mode, newline='') as csvFile:  
                        writer = csv.writer(csvFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        fieldnames = list(self.meanDeviations.keys())

                        #write our header
                        writer.writerow(fieldnames)
                        for key in self.meanDeviations:
                                writer.writerow(self.meanDeviations.get(key))
                print("Save complete.")

        
        #will need to convert the dataStream into a tensor
        def saveDataflowToCSV(self, saveLocation):
                """So currently the datasream holds the strands which is kind of like having the 'columns' of the 'table'. To write the dataflow as a .csv
                 that can be read by pandas. we need to write the rows of our 'table'. By converting the datastream into a 2D tensor we can then move through 
                 tensor row by row (entry by entry), which will contain the indecies on each entries features. """

                print("Preparing to save dataflow...")
                #create a matrix as our canvas
                matrix = []
                keyPointer = 0
                fieldnames = list(self.dataStream.keys())
                #create a replica of our data onto our canvas
                for key, strand in self.dataStream.items():
                        strandPointer = 0
                        
                        if (keyPointer == 0): #if it is first iteration initialise tensor.
                                for j in range(len(strand)):
                                        matrix.append([]) #create a row for each entry                       

                        #If our entry is a string, then attempt to add it as an int (as there shouldn't be any naurally occurring floats in a given table.)
                        for entry in strand:

                                if isinstance(entry, str):
                                        try:  
                                                matrix[strandPointer].append(int(entry))
                                        except: #otherwise add it as a string (This seems to usually be usually a varChar or a datetime)
                                                matrix[strandPointer].append(entry)
                                else:
                                        matrix[strandPointer].append(entry)
                                strandPointer += 1
                        keyPointer += 1
                
                print("Writing dataflow to " + saveLocation + "...")
                
                #this block will pick up a permission error.
                try:
                        assert os.path.isfile(saveLocation)
                except:
                        print(str(saveLocation) + " Either points to folder or doesn't exist.")
 
                #lets create a csv object in write mode if file exists
                mode = 'a' if os.path.exists(saveLocation) else 'w'
                #now we copy our canvas into the .csv
                with open(saveLocation, mode, newline='') as csvFile:  
                        writer = csv.writer(csvFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        fieldnames = list(self.dataStream.keys())

                        #write our header
                        writer.writerow(fieldnames)
                        #now we can use the matrix to write our rows to csv.
                        for row in matrix:
                                #writer.writerow(map(lambda x:x, row))
                                writer.writerow(row)
                
                csvFile.close()
                print("Saving dataflow complete.")

                        
                        
        
