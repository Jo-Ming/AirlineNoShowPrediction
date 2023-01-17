import csv
import os
from entryNode import entryNode #this is to split the row data into entries.

class tableNode:

    def __init__(self, location):
        self.location = location #Data will be stored as an array of rows for now
        self.entries = [] #will store a list of entry nodes

    def showRows(self):
        for row in self.data:
            print(row)
    
    #Currently This method will retrieve the located data from a csv file 
    def setNodeProperties(self, streamSize):
        """ Step 1: populate the data attribute. Currently by fetching from a csv. now we are making entryNode objects"""
        with open(self.location, mode = 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            line_count = 0
            for row in csv_reader:
                if (streamSize != 'max'):
                    if line_count == streamSize:
                        break
                _entryNode = entryNode(row)
                self.entries.append(_entryNode)
                line_count += 1
            print(f'Processed {line_count} lines.')
        #self.data = tableRows

        """ Step 2. Add other useful properties for a tableNode. """
         #Here we split the full path into directory path + filename eg. csv/right/here/my.csv --> csv/right/here , my.csv
        path = self.location
        _, fileName = os.path.split(path)

        self.filename = fileName
        self.numberOfEntries = len(self.entries)
        if (self.numberOfEntries > 0):
            self.numberOfFields = len(self.getEntry(0).getFields())
        else:
            print("Population of 0 entries. Issue loading table.")
    
    #shows the proerties of the table.
    def showProperties(self):
        print("Table name = " + self.filename)
        print("Location of file: " + self.location)
        print("Number of Entries: " + str(self.numberOfEntries))
        print("Number of Fields: " + str(self.numberOfFields) + '\n')
    
    #returns row entry at given location as a entryNode object.
    def getEntry(self, pointer):
        return self.entries[pointer]
        
    #displays a specified number of entries.
    def showEntries(self, quantity):
        for i in range (quantity):
            entry = self.getEntry(i)
            print(entry.getEntry())
    #displays a specified number of entries.
    def getFields(self):
        fieldList = []
        #first we can retrieve the fields in a dictionary form
        fieldDict = self.getEntry(0).getFields() #use first entry as reference.
        for field in fieldDict:
            fieldList.append(field)
        return fieldList

    #shows each field/key from the table.
    def showFields(self, pointer = 1): #for now we will always reference the first entry
        entry = self.entries[pointer]
        print(entry.getFields())
    
    #will return a list containing values of x entries. 
    def getEntryValues(self,quantity):
        valueStream = [] #ad each list to stream so we can seperate individuals.
        for i in range(quantity):
            valueList = [] #fresh list iteration (will seperate entries).
            _entry = self.getEntry(i)
            values = _entry.getValues()
            for value in values:
                valueList.append(value)
            valueStream.append(valueList)

        return valueStream
    
    def getTableChord(self, chordKey):
        chordData = []
        for entry in self.entries:
            chordData.append(entry.getChord(chordKey))
        return chordData
    
    #will return a list containing X entry objects. 
    def getEntryObjects(self,quantity):
        entryList = [] #add to our empty list to (will also seperate entries)
        for i in range(quantity):
            _entry = self.getEntry(i)
            entryList.append(_entry)
        return entryList
    
    #will return a list containing values of x entry objects. 
    def getEntries(self,quantity):
        entryList = [] #add to our empty list to (will also seperate entries)
        for i in range(quantity):
            _entry = self.getEntry(i)
            entryList.append(_entry)
        return entryList

    def getValuesAtPoint(self, pointer):
        _entry = self.entries[pointer]
        return _entry.getValues()
    
    def getColumn(self, key):
        column = []
        for entry in self.entries:
            column.append(entry.getValue(key))
        return column

        