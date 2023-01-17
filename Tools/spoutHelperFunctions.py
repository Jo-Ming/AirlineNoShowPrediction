import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import datetime
import pytz
from airportDictionary import loadAirportDictionary
import math
from datetime import date


#will find difference in minutes of two times in format HH:MM
def getTimeDifference(time1, time2):
        hours1, minutes1 = time1.split(":")
        hours2, minutes2 = time2.split(":")
        return ((int(hours1)*60)+ int(minutes1)) - ((int(hours2)*60)+ int(minutes2))

def getDaysUntil(dateTimeFrom, dateTimeTill):
    #null catching
        if(dateTimeFrom == 'NULL' or dateTimeTill == 'NULL' or dateTimeFrom == '' or dateTimeTill == ''):
            return 0
        #divide the dates from times
        try:
            date1, time1 = dateTimeFrom.split(" ")
            date2, time2 = dateTimeTill.split(" ")
        except: #at least one of these values are null or only date/time
            return 0
            
        #break down the dates. try different formats depending
        try:
            year1, month1, day1 = date1.split("-")
            year2, month2, day2 = date2.split("-")
        except:
            year1, month1, day1 = date1.split("/")
            year2, month2, day2 = date2.split("/")

        #get these values in minutes
        dateFrom = date(int(year1), int(month1), int(day1))
        dateTill = date(int(year2), int(month2), int(day2))
        daysUntil = dateTill - dateFrom

        return daysUntil.days

def getDaysFrom(dateTill, dateFrom):
        #null catching
        if(dateFrom == 'NULL' or dateTill == 'NULL' or dateFrom == '' or dateTill == ''):
            return 0
    
        #break down the dates. try different formats depending
        try:
            year1, month1, day1 = dateFrom.split("-")
            year2, month2, day2 = dateTill.split("-")
        except:
            year1, month1, day1 = dateFrom.split("/")
            year2, month2, day2 = dateTill.split("/")

        #get these values in minutes
        dateFrom = date(int(year1), int(month1), int(day1))
        dateTill = date(int(year2), int(month2), int(day2))
        daysUntil = dateTill - dateFrom

        return daysUntil.days
    
def getDaysUntilFromDates(dateFrom, dateTill):
    #null catching
        if(dateFrom == 'NULL' or dateTill == 'NULL' or dateFrom == '' or dateTill == ''):
            return 0
    
        #break down the dates. try different formats depending
        try:
            year1, month1, day1 = dateFrom.split("-")
            year2, month2, day2 = dateTill.split("-")
        except:
            year1, month1, day1 = dateFrom.split("/")
            year2, month2, day2 = dateTill.split("/")

        #get these values in minutes
        dateFrom = date(int(year1), int(month1), int(day1))
        dateTill = date(int(year2), int(month2), int(day2))
        daysUntil = dateTill - dateFrom

        return daysUntil.days

        
#will take 2 dateTimes and return the difference in minutes.
def getDateTimeDifference(dateTime1, dateTime2):
        #null catching
        if(dateTime1 == 'NULL' or dateTime2 == 'NULL'):
            return 'NULL'
        #divide the dates from times
        try:
            date1, time1 = dateTime1.split(" ")
            date2, time2 = dateTime2.split(" ")
        except: #at least one of these values are null
            return 0
        #break down the dates
        try:
            year1, month1, day1 = date1.split("-")
            year2, month2, day2 = date2.split("-")
        except:
            year1, month1, day1 = date1.split("/")
            year2, month2, day2 = date2.split("/")

        #get these values in minutes
        yearDiff = abs(int(year2) - int(year1))*525600
        monthDiff = abs(int(month2) - int(month1))*43800
        dayDiff = abs(int(day2) - int(day1))*1440

        #split time into hours and minutes.
        try:
            hours1, minutes1, seconds1 = time1.split(":")
            hours2, minutes2, seconds2 = time2.split(":")
        except:
            hours1, minutes1 = time1.split(":")
            hours2, minutes2 = time2.split(":")

        #find time differences.
        hoursDiff = abs(int(hours2) - int(hours1))*60
        minutesDiff = abs(int(minutes2)-int(minutes1))

        #is commutative so just add them all together.
        return (yearDiff + monthDiff + dayDiff + hoursDiff + minutesDiff)

#a method to encode/cipher the given fields into integers. as a form of dimensionality reduction
#will bind a representing integer to each label given, returns as an dictionary. Might be faster using list comprehension to encode?
def encodeAsIntegersOld(strand): #chord = ['C','D','E','F','G','A','B','C','D','C']
#using enumerate() the returned list will not be in the order you sent it.
    mapping_to_numbers = {}
    cipher = np.zeros((len(strand),), dtype=int) # to store the original coordinaates of the chord/label List.
    for i, entry in enumerate(strand):
        if entry not in mapping_to_numbers:
            mapping_to_numbers[entry] = len(mapping_to_numbers)
        cipher[i] = int(mapping_to_numbers[entry])
        #print(cipher) # [0. 1. 2. 3. 1. 2.]
    return mapping_to_numbers, cipher # {'d': 3, 'a': 0, 'c': 2, 'b': 1}

#a method to encode/cipher the given fields into integers. as a form of dimensionality reduction
#will bind a representing integer to each label given, returns as an dictionary. Might be faster using list comprehension to encode?
def encodeAsIntegers(strand): #chord = ['C','D','E','F','G','A','B','C','D','C']
#using enumerate() the returned list will not be in the order you sent it.
    mapping_to_numbers = {}
    cipher = [] # to store the original coordinaates of the chord/label List.
    for i, entry in enumerate(strand):
        if entry not in mapping_to_numbers:
            mapping_to_numbers[entry] = len(mapping_to_numbers)   
        cipher.append(int(mapping_to_numbers[entry]))
    return mapping_to_numbers, cipher # {'d': 3, 'a': 0, 'c': 2, 'b': 1}

def splitDataAndClasses(dataFlow, targetData):
    trainingData, testingData, trainingClasses, testingClasses = train_test_split(dataFlow, targetData, test_size = 0.2, random_state = 1)
    trainingData = np.array(trainingData)
    testingData = np.array(testingData)
    trainingClasses = np.array(trainingClasses)
    testingClasses = np.array(testingClasses)
    return trainingData, testingData, trainingClasses, testingClasses

def strandAnalysis(strand):
    lowest = strand[0]
    highest = strand[0]
    nullCounter = 0
    uniqueEntries = {}

    for entry in strand:
        try:
            if entry not in uniqueEntries:
                uniqueEntries.update({entry:0})
            if (entry == 'NULL' or entry == ''):
                nullCounter += 1
            else:
                if (float(entry)<float(lowest)):
                    lowest = entry
                elif(float(entry)>float(highest)):
                    highest = entry
            counter = uniqueEntries.get(entry)
            uniqueEntries.update({entry: counter + 1})
        except:
            lowest = "N/A"
            highest = "N/A"
            if entry not in uniqueEntries:
                uniqueEntries.update({entry : 0})
            if (entry == 'NULL'):
                nullCounter += 1
            counter = uniqueEntries.get(entry)
            uniqueEntries.update({entry: counter + 1})
            
    return highest, lowest, nullCounter, uniqueEntries

def getNullCount(strand):
    nullCounter = 0
    for entry in strand:
        if (entry == 'NULL' or entry == ''):
            nullCounter += 1
    return nullCounter

def containsSubstring(string, sub_str):
    if (string.find(sub_str) == -1):
        return False
    else:
        return True

def normaliseStrand(strand):
    normalisedStrand = preprocessing.normalize([strand]) #needs to be parsed as 2d array, returns 2 dimensoinal araay
    return np.reshape(normalisedStrand, (normalisedStrand.size))#reshape back into a 1d array.

def standardizeStrand(strand):
    standardizedStrand = preprocessing.scale([strand])
    return np.reshape(standardizedStrand, (standardizedStrand.size))

def tanhNormalization(strand):

    mean = np.mean(strand, axis = 0)
    standardDeviation = np.std(strand, axis = 0)

    #tanhNormalizedStrand = 0.5 * (np.tanh(0.01 * ((strand - mean) / standardDeviation)) + 1)
    tanhNormalizedStrand = (np.tanh(0.1 * ((strand - mean) / standardDeviation)))

    return np.reshape(tanhNormalizedStrand, (tanhNormalizedStrand.size)), [mean, standardDeviation]


def standardiseStrand(strand):
    standard = preprocessing.scale(strand)
    return standard

def getDistanceList(coordList1, coordList2):
    distanceList = []

    for i in range(len(coordList1)):
        coord1 = coordList1[i]
        coord2 = coordList2[i]
        if (coord1 == 'NULL' or coord2 == 'NULL'):
            distanceList.append('NULL')
        else:
            distanceList.append(pythagorianDistance(coord1, coord2))
    return distanceList

    #coords will be in the form of a tuple (longitude, latitude)
def pythagorianDistance(coord1,coord2):
    a = coord1[0] - coord2[0]
    b = coord1[1] - coord2[1]

    c = math.sqrt(a**2 + b**2)
    return c

def convertDatetimeToTimezone(dateTime, initialTimeZone, targetTimeZone):
    if (initialTimeZone == 'NULL'):
        return 'NULL'

    timeZone1 = pytz.timezone(initialTimeZone)
    timeZone2 = pytz.timezone(targetTimeZone)

    try:
        dateTime = datetime.datetime.strptime(dateTime,"%Y-%m-%d %H:%M:%S")
        dateTime = timeZone1.localize(dateTime)
        dateTime = dateTime.astimezone(timeZone2)
        dateTime = dateTime.strftime("%Y-%m-%d %H:%M:%S")
    except:
        dateTime = datetime.datetime.strptime(dateTime,"%d/%m/%Y %H:%M")
        dateTime = timeZone1.localize(dateTime)
        dateTime = dateTime.astimezone(timeZone2)
        dateTime = dateTime.strftime("%d/%m/%Y %H:%M")

    return dateTime

#takes a list of IATA codes and returns a list of coresponding time zones.
def getAirportTimeZones(IATAList):
    #If we are getting timezone over a list it will be faster to only have to load the data once
    #rather than load the dictionary each iteration.
    airports = loadAirportDictionary()
    timeZoneList = []
    couldntResolve = [] #will contain a list of airports we were unable to resolve.
    for IATA in IATAList:
        if(IATA == 'NULL' or IATA ==''):
            timeZoneList.append('NULL')
        else:
            try:
                airport = airports.get(IATA)
                timeZoneList.append(airport.get('tz'))
            except:
                print("There is no airport in airport Dictionary with IATA code " + IATA)
                timeZoneList.append('NULL')
                couldntResolve.append(IATA)
    return timeZoneList

def getAirportCoordsList(IATAList):
    #If we are getting coords over a list it will be faster to only have to load the data once
    #rather than load the dictionary each iteration.
    airports = loadAirportDictionary()
    coordList = []
    couldntResolve = [] #will contain a list of airports we were unable to resolve.
    for IATA in IATAList:
        if(IATA == 'NULL' or IATA == ''):
            coordList.append('NULL')
        else:
            try:
                airport = airports.get(IATA)
                coordList.append((airport.get('lon'),airport.get('lat')))
            except:
                print("There is no airport in airport Dictionary with IATA code ", IATA)
                coordList.append('NULL')
                couldntResolve.append(IATA)
    return coordList

    #this function takes a list of dateTimes, a list of corresponding timeZones, and our desiredTimeZone
def shiftStrandTimeZones(dateTimeStrand, timeZoneStrand, toTimeZone):
    outputStrand = []
    i = 0
    for dateTime in dateTimeStrand:
        newTime = convertDatetimeToTimezone(dateTime, timeZoneStrand[i], toTimeZone)
        outputStrand.append(newTime)
        i+=1
    return outputStrand

def strandToUTC(airports, dateTimes):
    timeZones = getAirportTimeZones(airports)
    newTimes = shiftStrandTimeZones(dateTimes, timeZones, 'UTC')
    return newTimes

#time is given in a HH:MM:SS (hour:minute:seconds)
#to prepare it for a model we want the format from "hours:minutes:seconds" to [Hours, minutes, seconds]
def breakdownTime(timeString):
    outputList = [] #store indexes
    token = "" #temp variable
    i = 0
    for i in range(len(timeString)):
        if (timeString[i] == ':'):       
            outputList.append(int(token))
            token = ""
        else: #pop to token into out list
            token = token + timeString[i]
    outputList.append(int(token))
    return outputList[0], outputList[1] #only return minutes and hours

def toMinuteOfDay(time):
    hours, minutes = breakdownTime(time)
    try:
        minuteOfDay = (int(hours)*60) + int(minutes) 
    except:
        minuteOfDay = 'NULL'
        print("Unable to convert time string to int." + str(time))
    return minuteOfDay

def timesToMinutesOfDay(timeStrand):
    minutesOfDay = []
    for time in timeStrand:
        if time != 'NULL':
            minuteOfDay = toMinuteOfDay(time)
            minutesOfDay.append(minuteOfDay)
        else:
            minutesOfDay.append('NULL')
    return minutesOfDay

#get the strand of distances between 2 lists/strands of airports. Currently using pythagorean distance.
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

#will splite a datetime into date and time. 
def splitDateTime(dateTimestrand):
    dateStrand = []
    timeStrand = []
    strands = []
    for dateTime in dateTimestrand:
        if (dateTime == 'NULL' or dateTime == ''):
            dateStrand.append('NULL')
            timeStrand.append('NULL')
        else:
            separation = dateTime.split(" ")
            dateStrand.append(separation[0])
            timeStrand.append(separation[1])
    return dateStrand, timeStrand


