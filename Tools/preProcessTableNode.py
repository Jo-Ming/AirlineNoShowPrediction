from Tools.tableNode import tableNode

#this object will inherit from the table node but override the setTableProperties function to populate using an entryList instead of reading from a .csv.
class preProcessTableNode(tableNode):

    #here we shall perform an override, to pupulate the entries using a parsed list.
    def setTableProperties(self, entryList):
        for entry in entryList:
            self.entries.append(entry)
        
        self.numberOfEntries = len(self.entries)
        if (self.numberOfEntries > 0):
            self.numberOfFields = len(self.getEntry(0).getFields())
        else:
            print("Population of 0 entries. Issue loading table.")
    
    def return0():
        return 0