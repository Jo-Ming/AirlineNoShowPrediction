#This is so we can import from the parent directory.
import os, sys

p = os.path.abspath('.')
sys.path.insert(1, p)

# Handle date time conversions between pandas and matplotlib
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from Tools.entryNode import entryNode
from Tools.tableNode import tableNode
from Tools.dataVisualiser import dataVisualiser
from Tools.dataSpout import dataSpout
from Tools.Feeder import Feeder
from sklearn.utils import shuffle #to randomize the data order when training

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from noShowController import NoShowController
import numpy as np

#for more customised plots that arnt available in the data visualiser.
import seaborn as sns
import matplotlib.pyplot as plt

##############################################################################################################################
#State the location of the .csv file, and the number of entries you wish to load.
##############################################################################################################################
fileLocation = 'CSVFiles\CSVFiles2.csv'
size = 180000  #use 'max' for all of the table entries.
##############################################################################################################################

#once you know the chord we can then divide the chord up based on how they will be treated.

#here is the full chord excluding pnr_unique for the current noShow model
chord = [ 'nosho', 'cancelled', 'seg_cancelled', 'pax_cancelled', 'pnr_status', 'no_in_party',
                'domestic_international', 'advance_booking_days', 'class', 'booked_connection_time', 'minimum_connection_time',
                'inbound_arrival_datetime', 'departure_datetime', 'departure_datetime_utc', 'day_of_week', 'board_point',
                'off_point', 'segment_distance', 'inbound_airport', 'inbound_segment_no', 'inbound_route',
                'mkt_carrier_code', 'mkt_flight_no', 'op_carrier_code', 'op_flight_no', 'op_booking_class',
                'equipment', 'gender_code', 'passenger_type_code', 'passenger_type', 'document_birthdate',
                'nosho_type', 'pos_violation', 'group_violation', 'fake_name_violation', 'test_booking', 
                'duplicate', 'hidden_group_flag', 'marriage_violation',
                'mct_violation', 'time_under_over', 'fake_name_violation_match', 'fake_name_violation_match_name', 'test_passenger']

###################################################################################################################################
#lets divide the fields into subchords grouped by how they will be treated.
###################################################################################################################################

#targetVariable.
targetField = ['nosho'] 

#The violations found through the current logic approaches.
violationFields = ['pos_violation', 'group_violation', 'fake_name_violation', 'test_booking', 
                    'duplicate', 'mct_violation', 'fake_name_violation_match', 'fake_name_violation_match_name']
#fields such with strings/classes which contain useful data but arn't yet in a numerical form.
enumerateFields = ['domestic_international', 'class', 'board_point', 'off_point', 'inbound_airport', 'inbound_route',
                     'op_booking_class', 'gender_code', 'passenger_type_code', 'passenger_type']
#status fields (which mostly only have 1 value)
statusFields = ['test_passenger', 'cancelled', 'seg_cancelled', 'pax_cancelled', 'pnr_status']
#These fields are fields that I probably won't use for training a model.
tempEnumFields = ['mkt_carrier_code', 'mkt_flight_no', 'op_carrier_code', 'op_flight_no']
#fields in the form of a string which we want in an integer form.
fieldsToInt = ['no_in_party', 'advance_booking_days', 'day_of_week', 'segment_distance', 'inbound_segment_no',
            'equipment', 'time_under_over','minimum_connection_time', 'booked_connection_time']
#dates + time fields
dateTimeFields = ['departure_datetime', 'inbound_arrival_datetime']
dateFields = ['document_birthdate']

def Plots3DDemo():
    #initialise controller 
    controller = NoShowController()
    controller.loadModel(modelLocation= 'savedModels\\model2_randomState42_noshowPredictor')

    #get an example entryList.
    entryList = controller.loadRandomEntries(1000, 2, 'CSVFiles\\trainingData2.csv') #list size 1, random seed 1

    dataStreamDict = controller.getWholeDataStreamDict(entryList)
    confidences = controller.getConfidenceList(entryList)
    confidences = np.reshape(confidences, (len(confidences)))
    dataStreamDict.update({"Confidence": confidences.tolist()})

    #generateHeatmapFromDataStream(dataStreamDict)

    ages = dataStreamDict.get("document_birthdate")
    advancedBookingDays = dataStreamDict.get("advance_booking_days")
    genderCodes = dataStreamDict.get("gender_code")
    _class = dataStreamDict.get("class")
    flightNo = dataStreamDict.get("op_flight_no")
    dayOfWeek = dataStreamDict.get("day_of_week")
    mctVio = dataStreamDict.get("mct_violation")
    noInParty = dataStreamDict.get("no_in_party")

    nosho = dataStreamDict.get("nosho")

    plot3DBar1(dayOfWeek, noInParty)

    #plot3DScatty1(ages, advancedBookingDays, confidences, genderCodes)
    #plot3DScatty2(_class, advancedBookingDays, confidences, ages)
    #plot3DScatty3(dayOfWeek, flightNo, confidences, nosho)

    #plot3DMesh(ages, advancedBookingDays, confidences)
    #plot3DWireframe(ages, advancedBookingDays, confidences)

    #plot3dExample()

    return 0

def plot3DBar1(dayOfWeek, noInParty):
    # setup the figure and axes
    fig = plt.figure(figsize=(8, 3))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # fake data
   
    xMesh, yMesh = np.meshgrid(dayOfWeek, noInParty)
    x, y = xMesh.ravel(), yMesh.ravel()

    top = x + y
    bottom = np.zeros_like(top)
    width = depth = 1

    ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
    ax1.set_title('Shaded')

    plt.show()

def plot3DScatty1(ages, advancedBookingdays, confidences, genderCodes):
    
    #make a figure and subplot to plot onto
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #now plot points
    scatty_plot = ax.scatter(xs=ages, ys=advancedBookingdays, zs=confidences, c=genderCodes)
    #set a colour bar for our gender codes
    cb = plt.colorbar(scatty_plot, pad=0.2)
    cb.set_ticks([0,1,2,3])
    cb.set_ticklabels(["Male", "Female", "Null", "U?"])
    #set a title
    ax.set_title("class, advancedBookingDays against confidence of noShow")

    #set labels
    ax.set_xlabel("Age in days")
    ax.set_ylabel("Days Booked In Advanced")
    ax.set_zlabel("Confidence in No Show")

    plt.show()

def plot3DScatty2(_class, advancedBookingDays, confidences, ages):
        
    #make a figure and subplot to plot onto
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #now plot points
    scatty_plot = ax.scatter(xs=ages, ys=advancedBookingDays, zs=confidences, c=_class)
    #set a colour bar for our gender codes
    cb = plt.colorbar(scatty_plot, pad=0.2)
    cb.set_ticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25])
    cb.set_ticklabels(['N','E','O','Q','X','Z','R','T','K','S','Y','L','H','J','I','U','W','D','P','B','V','G','M','C','A','F'])
    #set a title
    ax.set_title("Age, advancedBookingDays against confidence of noShow")

    #set labels
    ax.set_xlabel("Age in days")
    ax.set_ylabel("Days Booked In Advanced")
    ax.set_zlabel("Confidence in No Show")

    plt.show()

def plot3DScatty3(dayOfWeek, flightNo, confidences, nosho):
        
    #make a figure and subplot to plot onto
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim3d(0, 7)
    ax.set_ylim3d(0, 20)
    ax.set_zlim3d(0, 1)

    #now plot points
    scatty_plot = ax.scatter(xs=dayOfWeek, ys=flightNo, zs=confidences, c=nosho)
    #set a colour bar for our gender codes
    cb = plt.colorbar(scatty_plot, pad=0.1)
    cb.set_ticks([0,1])
    cb.set_ticklabels(["Show", "No Show"])
    #set a title
    ax.set_title("Age, advancedBookingDays against confidence of noShow")

    #set labels
    ax.set_xlabel("Day Of Week")
    ax.set_ylabel("Flight No")
    ax.set_zlabel("Confidence in No Show")

    plt.show()



def plot3DMesh(x, y, z):
        X, Y = np.meshgrid(x, y)
        #Z = confidenceMesh(X,Y,z)
        Z = z* np.ones((X.shape))
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.contour3D(X, Y, Z, cmap='viridis')
        ax.set_xlabel("age")
        ax.set_ylabel("days Booked In Advanced")
        ax.set_zlabel("Confidence");

        plt.show()

def plot3DWireframe(x, y, z):
        X, Y = np.meshgrid(x, y)
        #Z = confidenceMesh(X,Y,z)
        Z = z* np.ones((X.shape))
        ax = plt.figure()
        ax.plot_wireframe(X, Y, Z, color ='green')
        ax.contour3D(X, Y, Z, cmap='viridis')
        ax.set_xlabel("age")
        ax.set_ylabel("days Booked In Advanced")
        ax.set_zlabel("Confidence");

        plt.show()

def plot3dExample():
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt

    x = np.linspace(-6, 6, 30)
    y = np.linspace(-6, 6, 30)

    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 50, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z');

    plt.show()

def confidenceMesh(x,y,z):
    #return np.array(z)*(np.array(x)+np.array(y))
    #return np.array(z)+(np.array(x)*np.array(y)*0.1)
    return z*(1/x*y)
    #return z*((1-(1/x))*(1-(1/y)))

def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

def main():
    #first lets get our feeder for the training data
    noShowData = Feeder(fileLocation, size) #takes the location and size

    #you can get the full chord within a .csv file by calling the following:
    fullChord = noShowData.getAllTableFields()
    print("The table contains the fields: " + str(fullChord))
    print("Number of Columns: " + str(len(fullChord)))

    #small summary of the data
    #getHead(fileLocation)
    #showEntries(noShowData, 1) #show an example
    noShowData.setDataStream(chord)
    noShowData.evaluateDataStream() #evaluation

    #now we can show some visualisations.
    noShowData = showHeatmap(noShowData) #returns the feeder with its dataStream set and enriched.
    featuresVisualiser = dataVisualiser('dict', noShowData.dataStream)

    #implots
    featuresVisualiser.implot('no_in_party' ,'advance_booking_days', 'nosho')
    featuresVisualiser.implot('no_in_party' ,'advance_booking_days', 'gender_code')

    #featuresVisualiser.groupedBarplot('class' ,'advance_booking_days', 'nosho')
    #featuresVisualiser.stackedHistogram('advance_booking_days', 'nosho')

    violationAnalysis(noShowData)
"""
next(item for item in dicts if item["name"] == "Pam")
{'age': 7, 'name': 'Pam'}

next((i for i, item in enumerate(dicts) if item["name"] == "Pam"), None) #gives index

"""
#perhaps this is too low level 

def isViolation(entry):
    isViolation = False
    for violation in violationFields:
        value = entry.rowDictionary.get(violation)
        if(value != '' and value != 'NULL' and value != 'N'):
            isViolation = True
    return isViolation

def isNoShow(entry):
    isNoShow = entry.rowDictionary.get('nosho')
    if isNoShow == 't':
        return True
    else:
        return False

def violationAnalysis(dataflow):
    dataflow.setDataStream(targetField + violationFields)
    dataflow.evaluateDataStream()

    print("Total number of entries: " +str(size))
    #find no shows
    noShowEntries = []
    noShowLocations = []
    location = 0

    #will populate a list of no shows
    for entry in dataflow.nodeSlot.entries:
        if(isNoShow(entry)):
            noShowEntries.append(entry)
            noShowLocations.append(location)
        location += 1
    
    print("Number of No Shows: " + str(len(noShowEntries)))

    violatingEntries = []
    violatingEntryLocations = []

    #will populate a list of violating entries
    location = 0 
    for entry in dataflow.nodeSlot.entries:
        if(isViolation(entry)):
            violatingEntries.append(entry)
            violatingEntryLocations.append(location)
        location += 1
    
    print("Number of Violating Entries: " + str(len(violatingEntries)))

    location = 0
    noShowORViolatingEntries = []
    NSorVioLocations = []

    #populates a list of violations or no shows
    for entry in dataflow.nodeSlot.entries:
        if(isViolation(entry) or isNoShow(entry)):
            noShowORViolatingEntries.append(entry)
            NSorVioLocations.append(location)
        location += 1
    
    print("Number of NoShow OR violating entries: " + str(len(noShowORViolatingEntries)))

    violatingNoShows = []
    violatingNoShowLocations = []
    locaiton = 0
    #will populate a list of noshows which also have violating flags.
    for entry in dataflow.nodeSlot.entries:
        if(isViolation(entry) and isNoShow(entry)):
            violatingNoShows.append(entry)
            violatingNoShowLocations.append(location)
        location += 1

    print("Number of No Shows with violations: " + str(len(violatingNoShows)))

    noShowsNoViolations = []
    noShowsNoViolationsLocations = []
    #populates a list of no shows without violations.
    for entry in dataflow.nodeSlot.entries:
        if(not isViolation(entry) and isNoShow(entry)):
            noShowsNoViolations.append(entry)
            noShowsNoViolationsLocations.append(location)
        location += 1
    
    print("Number of noShows Without violations: " + str(len(noShowsNoViolations)))

    showsWithViolations = []
    showsWithViolationsLocations = []
    #populates a list of shows with violaitons.
    for entry in dataflow.nodeSlot.entries:
        if(isViolation(entry) and not isNoShow(entry)):
            showsWithViolations.append(entry)
            showsWithViolationsLocations.append(location)
        location += 1
    
    print("Number of shows with violations: " + str(len(showsWithViolations)))

    showsNoViolations = []
    showsNoViolationsLoactions = []

    #populates a list of shows without violations.
    for entry in dataflow.nodeSlot.entries:
        if(not isViolation(entry) and not isNoShow(entry)):
            showsNoViolations.append(entry)
            showsNoViolationsLoactions.append(location)
        location += 1

    print("Number of shows with no violations: " + str(len(showsNoViolations)))
    return 0

def generateGroupBarPlot(feeder, x, y, hue):
    featuresVisualiser = dataVisualiser('dict', feeder.dataStream)
    featuresVisualiser.groupedBarplot(x , y, hue)

def showEntries(feeder, x):
    #show the number of entries
    feeder.showXEntries(x)

def generateHeatmap(feeder):
    #create a visualiser usingf the datastream of the feeder/dataspout
    featuresVisualiser = dataVisualiser('dict', feeder.dataStream)
    featuresVisualiser.generateHeatmap() #call the generate heatmap function   

def generateHeatmapFromDataStream(dataStream):
    #create a visualiser usingf the datastream of the feeder/dataspout
    featuresVisualiser = dataVisualiser('dict', dataStream)
    featuresVisualiser.generateHeatmap() #call the generate heatmap function   

def generateImplot(feeder, x, y, hue):
    implotVis = dataVisualiser('dict', feeder.dataStream)
    implotVis.implot(x,y,hue)

def showHeatmap(feeder):
    #first lets get our feeder for the training data
    trainingData = feeder

    #ok first lets generate a heatmap to look for correlations.

    #set the dataStream 
    trainingData.setDataStream(targetField + fieldsToInt)

    #evaluate the dataStream
    #trainingData.evaluateDataStream()

    #for the mini heatmap we will just use the Quantitative Fields.
    #lets convert the quantive fields into integers frist.
    trainingData.typeCastChordToInt(fieldsToInt) #this will also set any Null fields to 0.
    #lets also encode our targetField
    trainingData.encodeChord(targetField)
    #now we can plot
    generateHeatmap(trainingData)

    #lets plot one for the violation and target fields
    trainingData.setDataStream(targetField + violationFields)
    trainingData.encodeChord(targetField + violationFields)
    generateHeatmap(trainingData)

    #ok lets do a big heat map
    trainingData.setDataStream(targetField + enumerateFields + fieldsToInt + dateTimeFields + dateFields)
    trainingData.typeCastChordToInt(fieldsToInt)

    trainingData.distanceTimeEnrichment()

    trainingData.encodeChord(enumerateFields + targetField)

    generateHeatmap(trainingData)
    return trainingData

def getHead(fileLocation):
    #for pnr Daily
    Dvis = dataVisualiser('csv',fileLocation)
    Dvis.info()
    Dvis.head()
   
#main()
Plots3DDemo()