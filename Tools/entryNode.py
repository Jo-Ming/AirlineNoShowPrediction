class entryNode:

    #constructs itself using a dictionary as an attribute {'key' : 'Value'}
    def __init__(self, rowData):
        self.rowDictionary = rowData
        self.size = len(self.rowDictionary)
    
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
