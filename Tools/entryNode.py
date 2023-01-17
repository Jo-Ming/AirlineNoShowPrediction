import numpy as np #used for normalisation.

class entryNode:

    #constructs itself using a dictionary as an attribute {'key' : 'Value'}
    def __init__(self, rowData):
        self.rowDictionary = rowData
        self.size = len(self.rowDictionary)
        self.pnr = self.getValue("pnr_no_unique")
        segmentNo = self.getValue('segment_no')
        passengerNo = self.getValue('passenger_no')
        self.seatKey = self.pnr + " " + segmentNo + " " + passengerNo
    
    #returns attibute
    def getPnr_no_unique(self):
        return self.pnr_no_unique
        
    #dictionaries are mutable and such are not a stable architecture for connecting dictionaries across keys.
    def getValueArray(self):
        valueArray = []
        for value in self.values():
            valueArray.append(value)
        return valueArray

    #returns an array of each field
    def getFieldArray(self):
        fieldArray = []
        for key in self.keys():
            fieldArray.append(key)
        return fieldArray

    #returns the dictionary as a whole
    def getEntry(self):
        return self.rowDictionary
    
    #returns all items in the dictionary (basically does the same thing as getEntry)
    def getItems(self):
        return self.rowDictionay.items()
    
    #returns an array of all the values for each field
    def getValues(self):
        return self.rowDictionary.values()
    
    #will return individual value from a dictionary key.
    def getValue(self, key):
        return self.rowDictionary.get(key)
    
    #returns an array of all fields 
    def getFields(self):
        return self.rowDictionary.keys()
    
    #method will display the entry in a human readable format
    def display(self):
        """
        if (len(fields) != len(values)): #hopefully this never happens lol maybe I can blame Jack...
            print("Error the lengths of fields and values do not match.")
        else: """
        print('\n')
        for key in self.rowDictionary: 
            token = self.getValue(key)
            print( key + ' ---> ', token)
    
    #will print the key and value in terminal
    def showChord(self, chordKey):
        for key in chordKey:
            print(key + ' ---> ' + str(self.getValue(key)))
    
    def getChord(self, chordKey):
        chordData = []
        for key in chordKey:
            chordData.append(self.getValue(key))
        return chordData
    
    #entry will concode itself using 
    def encode(self, cipherWheelDictionary, chordToEncode):
        self.encodedRowDict = {} #the attribute where we will stroe the encoding.

        for field in chordToEncode: #iterate through our chord to encode
            cipherWheel = cipherWheelDictionary.get(field) #get the relavent cipherWheel

            value = self.getValue(field) #get the value
            pointer = self.getPointer(value, cipherWheel[0]) #find the pointer in our wheel

            self.encodedRowDict.update({field:pointer}) #update the encoded dict attribute

    #this method will return a pointer to the key in the cipher wheel. The location will be its encoding.
    def getPointer(self, key, valueList):
        pointer = 0
        for value in valueList:
            if value == key:
                return pointer
            else:
                pointer += 1  
        print("Error, when encoding entry: " + str(self.pnr_no_unique) + "Unable to find value in the cipherKeyDictionary.")
        print("It's possible that this value has never been seen by the model.")
    
    #havent tested this, but will be needed in future if normalisation method changes.
    def NullToZeroChord(self, chord):
        for field in chord:
            value = self.getValue(field)
            if (value == '' or value == 'NULL'):
                self.rowDictionary.update({field: 0})
    
    #uses the means and standard deviations of each field the used as the saved models training data to perform a hyperbolic tan normalisation. 
    def tanhNormalisation(self, meansAndStandardDeviations):
        self.tanhNormalisedRowDict = {} #new attribute.
        fullChord = meansAndStandardDeviations[0].split(",") #gets each of the field names.
        meansAndStandardDeviations = meansAndStandardDeviations[1:] #trim off the first line.

        i = 0
        for field in fullChord:

            value = self.encodedRowDict.get(field)
            if value == None: #if the field is outside the encoding chord then pull the value from the raw row dict.
                value = self.getValue(field)
                #in the current method of model training the method .typeCastChordToInt(fieldsToInt) is called at the start which sets any null fields to 0
                #for I will set null values to 0 here however there will be a entry.setChordNullsToZero() method as this could very well be useful in the future.
                if (value == 'NULL' or value == ''): #easy quickfix null to 0 catch.
                    value = 0
    
            meanAndDeviation = meansAndStandardDeviations[i].split(",")
            mean = float(meanAndDeviation[0])
            standardDeviation = float(meanAndDeviation[1])
            tanhNormalizedValue = (np.tanh(1 * ((float(value) - mean) / standardDeviation)))

            #update attribute dictionary woth new value
            self.tanhNormalisedRowDict.update({field:tanhNormalizedValue})
            i+=1
    
    def getTanhChordValues(self, chord):
        chordData = []
        for key in chord:
            chordData.append(self.tanhNormalisedRowDict.get(key))
        return chordData

