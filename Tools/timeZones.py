import datetime
import pytz
from airportDictionary import loadAirportDictionary

from dataSpout import dataSpout
from spoutHelperFunctions import *
import math

timeChord = ['departure_datetime', 'inbound_arrival_datetime', 'inbound_airport',
             'board_point', 'domestic_international']

def timeZoneTest(timeChord):
    timeSpout = dataSpout('joseph.segment_150K', 150000)#streamSize = 100
    segmentTableFields = timeSpout.getAllTableFields()

    timeSpout.setDataStream(timeChord)

    """Lets first test by shifting board point list first to UTC which will have no NULL values."""

    boardPoints = timeSpout.getStrand('board_point')
    departureTimeZones = getAirportTimeZones(boardPoints)
    departureDateTimes = timeSpout.getStrand('departure_datetime')
    #convert to UTC
    UTCBoardPointTimes = shiftStrandTimeZones(departureDateTimes, departureTimeZones, 'UTC')

    """Now lets get the inbound times in UTC"""

    inboundAirports = timeSpout.getStrand('inbound_airport')
    inboundTimeZones = getAirportTimeZones(inboundAirports)
    inboundDateTimes = timeSpout.getStrand('inbound_arrival_datetime')
    UTCInboundTimes = shiftStrandTimeZones(inboundDateTimes, inboundTimeZones, 'UTC')

    #get coords 
    inboundCoords = getAirportCoordsList(inboundAirports)
    boardingCoords = getAirportCoordsList(boardPoints)

    distanceList = getDistanceList(inboundCoords, boardingCoords)
    #timeSpout.evaluateDataStream()

    print('yeah baby')

#this function takes a list of dateTimes, a list of corresponding timeZones, and our desiredTimeZone
def shiftStrandTimeZones(dateTimeStrand, timeZoneStrand, toTimeZone):
    outputStrand = []
    i = 0
    for dateTime in dateTimeStrand:
        newTime = convertDatetimeToTimezone(dateTime, timeZoneStrand[i], toTimeZone)
        outputStrand.append(newTime)
        i+=1
    return outputStrand

#takes airportCode and returns timeDiff/Timezone
def findLongLatFromIATA(IATA):
    airports = loadAirportDictionary()
    if(IATA == 'NULL'):
        coords = 'NULL'
    else:
        try:
            airport = airports.get(IATA)
            coords = (airport.get('lon'), airport.get('lat'))
        except:
            print("There is no airport in airport Dictionary with IATA code " + IATA)
    return coords

def getAirportCoordsList(IATAList):
    #If we are getting coords over a list it will be faster to only have to load the data once
    #rather than load the dictionary each iteration.
    airports = loadAirportDictionary()
    coordList = []
    couldntResolve = [] #will contain a list of airports we were unable to resolve.
    for IATA in IATAList:
        if(IATA == 'NULL'):
            coordList.append('NULL')
        else:
            try:
                airport = airports.get(IATA)
                coordList.append((airport.get('lon'),airport.get('lat')))
            except:
                print("There is no airport in airport Dictionary with IATA code " + IATA)
                coordList.append('NULL')
                couldntResolve.append(IATA)
    return coordList


def findTimeZoneFromIATA(IATA):
    airports = loadAirportDictionary()
    if(IATA == 'NULL'):
        timeZone = 'NULL'
    else:
        try:
            airport = airports.get(IATA)
            timeZone = airport.get('tz')
        except:
            print("There is no airport in airport Dictionary with IATA code " + IATA)
    return timeZone

#takes a list of IATA codes and returns a list of coresponding time zones.
def getAirportTimeZones(IATAList):
    #If we are getting timezone over a list it will be faster to only have to load the data once
    #rather than load the dictionary each iteration.
    airports = loadAirportDictionary()
    timeZoneList = []
    couldntResolve = [] #will contain a list of airports we were unable to resolve.
    for IATA in IATAList:
        if(IATA == 'NULL'):
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

def convertDatetimeToTimezone(dateTime, initialTimeZone, targetTimeZone):
    if (initialTimeZone == 'NULL'):
        return 'NULL'

    timeZone1 = pytz.timezone(initialTimeZone)
    timeZone2 = pytz.timezone(targetTimeZone)

    dateTime = datetime.datetime.strptime(dateTime,"%Y-%m-%d %H:%M:%S")
    dateTime = timeZone1.localize(dateTime)
    dateTime = dateTime.astimezone(timeZone2)
    dateTime = dateTime.strftime("%Y-%m-%d %H:%M:%S")

    return dateTime

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

timeZoneTest(timeChord)