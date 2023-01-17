from Tools.Feeder import Feeder
from Tools.tableNode import tableNode
import numpy as np

# this class will inherit al method from the feeder and will be used to prePocess data with encoding, normalising, and formating before parsing into our model.


class dataPreprocessor(Feeder):

    # here we will perform an override of the setChannel method. Where this object wil take a list of entries instead.
    def setChannel(self, entryList):
        # node slot holds object retrieve data from a table.
        entryListTable = tableNode("entryList")
        entryListTable.setEntriesFromList(entryList)
        self.nodeSlot = entryListTable

    # Encodes the datastream using the same encoding the model was trained on.
    # in the future I would like to update the cipher wheel when the model comes across new values.
    def encodeDataStream(self, encodingChord, encodingDict):
        #update the fieldKeys attribute of the feeder
        self.fieldKeys = {}

        for field in encodingChord:
            encodedStrand = []
            cipherWheel = encodingDict.get(field)
            strand = self.getStrand(field)
            self.fieldKeys.update({field: cipherWheel})

            for entry in strand:

                pointer = self.getPointer(entry, cipherWheel[0])
                if pointer == 'NULL': #this means the values doesn't exist in the dictionary...
                    print("Adding " + str(entry) + " to encoding dictionary.")
                    pointer = len(cipherWheel[0]) #because the cipher is base 0 the length will represent the next index in the wheel.
                    cipherWheel[0].append(entry)
                    cipherWheel[1].append(pointer)
                    #update the encodingDict
                    self.fieldKeys.update({field:cipherWheel})
                    encodedStrand.append(pointer)
                else:    
                    encodedStrand.append(pointer)

            self.dataStream.update({field: encodedStrand})

    # this method will return a pointer to the key in the cipher wheel. The location will be its encoding.
    def getPointer(self, key, valueList):
        pointer = 0
        for value in valueList:
            if value == key:
                return pointer
            else:
                pointer += 1
        print("Error, when encoding entry: " +
              "Unable to find value in the cipherKeyDictionary." + str(key))
        print("It's possible that this value has never been seen by the model.")
        return 'NULL' #for now if not present reurn null

    def tanhNormaliseDataStream(self, meansAndDeviations):
        # gives each field name the model normalised for trainingdata
        fullChord = meansAndDeviations[0].split(',')
        i = 1
        for field in fullChord:
            meanAndDeviation = meansAndDeviations[i].split(',')
            mean = float(meanAndDeviation[0])
            standardDeviation = float(meanAndDeviation[1])

            strand = self.getStrand(field)
            tanhNormalisedStrand = []
            for entry in strand:
                if (entry == '' or entry == 'NULL'):
                    print("Warning null values found in field: " + str(field))
                normalisedValue = (
                    np.tanh(1 * ((float(entry) - mean) / standardDeviation)))
                tanhNormalisedStrand.append(normalisedValue)
            self.dataStream.update({field: tanhNormalisedStrand})
            i += 1

    #override
    # will need to convert the dataStream into a tensor
    def getDataflowTensor(self):
        tensor = []
        i = 0
        for key, value in self.dataStream.items():
            pointer = 0
            try:
                if (i == 0):  # if it is first iteration initialise tensor.
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
                print('strand in the ' + str(key) +
                        ' field needs to be float or int before converting to tensor.')
        return tensor
