
#recent play

#for training new models it would be fuun to be able 

#wouldn't it be cool to have a way to pull a randomly seeded data from a datapool.
class datapool():

    #in order to construct this class the environment needs to be initialised.
    def __innit__():
        try:

            #for the code to be 
            #This is so we can import from the parent directory.
            import os, sys

            p = os.path.abspath('.')
            sys.path.insert(1, p)


            from Tools.entryNode import entryNode
            from Tools.tableNode import tableNode
            from Tools.dataVisualiser import dataVisualiser
            from Tools.dataSpout import dataSpout
            from Tools.Feeder import Feeder
        except:
            print("Import error :(. ")


    location = 'CSVFiles\CSVFiles.training_data.csv'
    size = 1000000


    
        


