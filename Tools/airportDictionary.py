"""
This file is dedicated to retreveing the airport data from an IATA code, holds information 
such as the name, city, longitude, latitude, elevation, and timezone of an airport.
"""
#this library is the current source of data
import airportsdata

def loadAirportDictionary():
    
    airportDictionary = airportsdata.load('IATA') #load all airports and data
    #these are some old airport codes that is used in the training data, so we will just all the same data to the old codes to make things simple.
    airportDictionary.update({'SXF': {'icao': 'EDDB', 'iata': 'BER', 'name': 'Berlin Brandenburg Airport', 'city': 'Berlin', 'subd': 'Brandenburg', 'country': 'DE', 'elevation': 157.0, 'lat': 52.362167, 'lon': 13.500667, 'tz': 'Europe/Berlin'}})
    airportDictionary.update({'TSE': {'icao':'UACC', 'iata':'NQZ', 'name':'Astana International Airport', 'city':'Astana', 'subd':'Aqmola', 'country':'KZ', 'elevation':1165.0, 'lat':51.0222015381, 'lon':71.4669036865 , 'tz':'Asia/Almaty'}})
    airportDictionary.update({'XER': {'icao':'', 'iata':'XER', 'name':'Gare de Strasbourg', 'city':'Strasbourg', 'subd':'Alsace', 'country':'FR', 'elevation': 476, 'lat':48.585188, 'lon': 7.734294, 'tz':'Europe/Paris'}})
    #airports not in dict
    airportDictionary.update({'XYD': {'icao':'', 'iata':'XYD', 'name':'Lyon Part-Dieu Railway', 'city':'Lyon', 'subd':'Rhône-Alpes', 'country':'FR', 'elevation': 569, 'lat':45.760575, 'lon': 4.860409, 'tz':'Asia/Almaty'}})
    airportDictionary.update({'SCX': {'icao':'MMSZ', 'iata':'SCX', 'name':'Salina Cruz Airport', 'city':'Salina Cruz', 'subd':'Oaxaca', 'country':'MX', 'elevation': 77, 'lat':16.207706, 'lon': -95.202133, 'tz':'America/Mexico_City'}})
    airportDictionary.update({'XWG': {'icao':'', 'iata':'XWG', 'name':'Gare de Strasbourg', 'city':'Strasbourg', 'subd':'Alsace', 'country':'FR', 'elevation': 476, 'lat':48.585188, 'lon': 7.734294, 'tz':'Europe/Paris'}})
    airportDictionary.update({'QJZ': {'icao':'', 'iata':'QJZ', 'name':'Gare de Nantes', 'city':'Nantes', 'subd':'Pays de la Loire', 'country':'FR', 'elevation': 28, 'lat':47.217568, 'lon': -1.542028, 'tz':'Europe/Paris'}})
    airportDictionary.update({'ZYR': {'icao':'', 'iata':'ZYR', 'name':'Brussels-South Railway Station', 'city':'Brussels', 'subd':'East Flanders (Oost-Vlaanderen)', 'country':'BE', 'elevation': 180, 'lat':50.800000, 'lon': 4.400000, 'tz':'Europe/Brussels'}})
    airportDictionary.update({'NGK': {'icao':'', 'iata':'NGK', 'name':'Nogliki Airport', 'city':'Nogliki-Sakhalin', 'subd':'Sakhalin Oblast', 'country':'RU', 'elevation': 1308, 'lat':50.690985, 'lon': 142.950569, 'tz':'Asia/Sakhalin'}})
    airportDictionary.update({'ZDH': {'icao':'', 'iata':'ZDH', 'name':'Gare de Nîmes', 'city':'Basel', 'subd':'Basel-Stadt', 'country':'CH', 'elevation': 913, 'lat':47.547589, 'lon': 7.589662, 'tz':'Europe/Zurich'}})
    airportDictionary.update({'NER': {'icao':'', 'iata':'NER', 'name':'Chulman Neryungri Airport', 'city':'Chulman', 'subd':'Neryungrinsky', 'country':'RU', 'elevation': 2805, 'lat':56.904148, 'lon': 124.903823, 'tz':'Asia/Yakutsk'}})
    airportDictionary.update({'ZYN': {'icao':'', 'iata':'ZYN', 'name':'Gare de Nîmes', 'city':'Nîmes', 'subd':'Languedoc-Roussillon', 'country':'FR', 'elevation': 144, 'lat':43.832555, 'lon': 4.366111, 'tz':'Asia/Sakhalin'}})
    airportDictionary.update({'ZWS': {'icao':'', 'iata':'ZWS', 'name':'Stuttgart Hauptbahnhof', 'city':'Stuttgart', 'subd':'Berlin', 'country':'DE', 'elevation': 200, 'lat':48.783611, 'lon': 9.181667, 'tz':'Europe/Berlin'}})
    airportDictionary.update({'XDB': {'icao':'', 'iata':'XDB', 'name':'Lille Airport', 'city':'Lille', 'subd':'Nord-Pas-de-Calais', 'country':'FR', 'elevation': 151, 'lat':50.571660, 'lon': 3.106113, 'tz':'Europe/Paris'}})
    airportDictionary.update({'KLF': {'icao':'', 'iata':'KLF', 'name':'Kaluga (Grabtsevo) Airport', 'city':'Kaluga', 'subd':'Central District', 'country':'RU', 'elevation': 600, 'lat':54.550000, 'lon': 36.367000, 'tz':'Europe/Moscow'}})
    airportDictionary.update({'ZLN': {'icao':'', 'iata':'ZLN', 'name':'Gare du Mans', 'city':'Le Mans', 'subd':'Pays de la Loire', 'country':'FR', 'elevation': 170, 'lat':47.995617, 'lon': 0.192446, 'tz':'Europe/Paris'}})
    airportDictionary.update({'TLN': {'icao':'', 'iata':'TLN', 'name':'Toulon–Hyères Airport', 'city':'Toulon', 'subd':"Provence-Alpes-Côte d'Azur", 'country':'FR', 'elevation': 39, 'lat':43.119537, 'lon': 5.935626, 'tz':'Europe/Paris'}})
    airportDictionary.update({'ZWE': {'icao':'', 'iata':'ZWE', 'name':'Antwerpen-Centraal railway station', 'city':'Antwerp', 'subd':'Antwerp (Antwerpen)', 'country':'BE', 'elevation': 29, 'lat':51.217192, 'lon': 4.421253, 'tz':'Europe/Brussels'}})
    airportDictionary.update({'RIZ': {'icao':'ZSRZ', 'iata':'RIZ', 'name':'Rizhao Shanzihe Airport', 'city':'Rizhao', 'subd':'Antwerp (Antwerpen)', 'country':'CN', 'elevation': 118, 'lat':35.405033, 'lon': 119.324403, 'tz':'Asia/Shanghai'}})
    airportDictionary.update({'XNB': {'icao':'OMDB', 'iata':'RIZ', 'name':'Dubai Airport', 'city':'Dubai', 'subd':'Dubai', 'country':'AE', 'elevation': 62, 'lat':25.2644444, 'lon': 55.3116667, 'tz':'Asia/Dubai'}})
    airportDictionary.update({'QKL': {'icao':'', 'iata':'QKL', 'name':'Cologne Railway', 'city':'Cologne', 'subd':'North Rhine-Westphalia (Nordrhein-Westfalen)', 'country':'DE', 'elevation': 100, 'lat':50.942500, 'lon': 6.958056, 'tz':'Europe/Berlin'}})
    airportDictionary.update({'ZFJ': {'icao':'', 'iata':'ZFJ', 'name':'Gare de Rennes', 'city':'Rennes', 'subd':'Bretagne', 'country':'FR', 'elevation': 98, 'lat':48.103497, 'lon': -1.672278, 'tz':'Europe/Paris'}})
    airportDictionary.update({'DTZ': {'icao':'', 'iata':'DTZ', 'name':'Dortmund Railway Station', 'city':'Dortmund', 'subd':'', 'country':'FR', 'elevation': 98, 'lat':51.5175, 'lon': 7.4589, 'tz':'Europe/Berlin'}})
    airportDictionary.update({'THX': {'icao':'UOTT', 'iata':'THX', 'name':'Turukhansk Airport', 'city':'Turukhansk', 'subd':'', 'country':'RU', 'elevation': 0, 'lat':65.79722, 'lon': 87.93528, 'tz':'Asia/Krasnoyarsk'}})
    airportDictionary.update({'XSH': {'icao':'', 'iata':'XSH', 'name':'Gare de Saint-Pierre-des-Corps', 'city':'Saint-Pierre-des-Corps', 'subd':'Pays de la Loire', 'country':'FR', 'elevation': 159, 'lat': 47.386029, 'lon': 0.723580, 'tz':'Asia/Krasnoyarsk'}})
    airportDictionary.update({'EYK': {'icao':'', 'iata':'EYK', 'name':'Beloyarsk Airport', 'city':'Beloyarsky', 'subd':'Tyumen Oblast', 'country':'RU', 'elevation': 25, 'lat': 63.694070, 'lon': 66.698791, 'tz':'Asia/Yekaterinburg'}})
    airportDictionary.update({'XZI': {'icao':'', 'iata':'XZI', 'name':'Gare de Lorraine TGV', 'city':'Louvigny', 'subd':'Alsace', 'country':'FR', 'elevation': 691, 'lat': 48.947784, 'lon': 6.169876, 'tz':'Europe/Paris'}})
    airportDictionary.update({'NLI': {'icao':'UHNN', 'iata':'NLI', 'name':'Nikolayevsk-na-Amure', 'city':'', 'subd':'', 'country':'RU', 'elevation': 52.00, 'lat': 53.154998779297, 'lon': 140.64999389648, 'tz':'Asia/Vladivostok'}})

    return airportDictionary