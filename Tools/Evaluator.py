#this class will attatch itself to a model object, execte an evaluation, then destroy itself.
from Tools.Feeder import Feeder #feeder can be used to retrieve streams of data.
from Tools.Model import Model
from Tools.evaluatorHelperFunctions import*
import time
from sklearn.metrics import roc_auc_score

class Evaluator:

    def __init__(self, model):
        self.modelSlot1 = model
    
    def __del__(self):
        print("Evaluator has self destructed!!!")
    
    def addSecondModel(self, model2):
        self.modelSlot2 = model2
    
    #evaluates performance of model in modelSlot1 on parsed data and returns a list of confidence values.
    def evaluateThis(self, dataflow, labels, classnames):
        
        #1. we time how long it takes to predict for our whole input
        print("1. Speed Test.")
        print("Predicting for " + str(len(dataflow)) + " entries...")

        start_time = time.time()
        confidenceValues = self.modelSlot1.getConfidenceValues(dataflow)
        end_time = time.time()

        print("Time taken: " + str(end_time-start_time)) #display time taken.

        #2. we can create a confusion matrix by parsing the labels, dataflow, and classnames.
        #we need class predictions instead of confidence values. So lets convert these into classifications.
        predictionList = self.modelSlot1.getBinaryPredictions(confidenceValues)
        confusionMatrix = binaryConfusionMatrix(self.modelSlot1.name, labels, predictionList, classNames = classnames)

        print("Confusion Matrix.")
        print(confusionMatrix)

        #once we have this we can perform an analysis 
        TP, FP, FN, TN = binaryConfusionMatrixAnalysis(confusionMatrix)

        print("Show: ", TP)
        print("False Show: ", FP)
        print("False Negatives: ", FN)
        print("True Negatives: ", TN)
        #plotted with x-Axis = 1-specificity , y-axis = 1-sensitivity
        showROCCurve(labels, predictionList, self.modelSlot1.name)
        return predictionList

    def compareModelsAgainstThis(self, dataflow, labels, classnames):
        print("Model 1. Evaluation...")
        predictions1 = self.evaluateThis(dataflow, labels, classnames)
        predictions2 = self.evaluateModelSlot2(dataflow, labels, classnames)
        
        #now we can plot their ROC curves on the same graph.
        showROCCurve2Models(labels, predictions1, predictions2, self.modelSlot1.name, self.modelSlot2.name)

    #evaluates performance of model in modelSlot2 on parsed data and returns a list of confidence values.
    def evaluateModelSlot2(self, dataflow, labels, classnames):
        
        #1. we time how long it takes to predict for our whole input
        print("1. Speed Test.")
        print("Predicting for " + str(len(dataflow)) + " entries...")

        start_time = time.time()
        confidenceValues = self.modelSlot2.getConfidenceValues(dataflow)
        end_time = time.time()

        print("Time taken: " + str(end_time-start_time))

        #2. we can create a confusion matrix by parsing the labels, dataflow, and classnames.
        #we need class predictions instead of confidence values. So lets convert these into classifications.
        predictionList = self.modelSlot2.getBinaryPredictions(confidenceValues)
        confusionMatrix = binaryConfusionMatrix(self.modelSlot2.name, labels, predictionList, classNames = classnames)

        print("Confusion Matrix.")
        print(confusionMatrix)

        #once we have this we can perform an analysis 
        TP, FP, FN, TN = binaryConfusionMatrixAnalysis(confusionMatrix)

        print("True Positives: ", TP)
        print("False Positives: ", FP)
        print("False Negatives: ", FN)
        print("True Negatives: ", TN)
        #plotted with x-Axis = 1-specificity , y-axis = 1-sensitivity
        showROCCurve(labels, predictionList, self.modelSlot2.name)
        return predictionList
    
    """These functions were for testing and developing"""
    def mctModelEvalutation1(self, testModel):
        """
        1. get data to run evaluation on.
        2. get metrics for evaluation.
        """

        mctFeeder = Feeder() #make ourselves a feeder
        dataflow, targetData = mctFeeder.getMCTData1(evaluateDataStream = False, evaluateTargetData = False, streamSize = 500000)

        print("Predicting for " + str(len(dataflow)) + " Entries...")

        start_time = time.time()
        confideceValues = testModel.getConfidenceValues(dataflow)
        end_time = time.time()

        print("time: " + str(end_time-start_time))

        predictedClasses = testModel.getBinaryPredictions(confideceValues)
        confusionMatrix = binaryConfusionMatrix(targetData, predictedClasses, classNames = ['Non-MCT_Violation','MCT_Violation']) #get confusion matrix
        binaryConfusionMatrixAnalysis(confusionMatrix) #analyse confucion matrix
        showROCCurve(targetData, predictedClasses)
    
    def mctModelEvaluation(self, testModel, chord):
        """
        1. get data to run evaluation on.
        2. get metrics for evaluation.
        """

        mctFeeder = Feeder() #make ourselves a feeder
        dataflow, targetData = mctFeeder.getMCTData1(evaluateDataStream = False, evaluateTargetData = False, streamSize = 500000)

        print("Predicting for " + str(len(dataflow)) + " Entries...")

        start_time = time.time()
        confideceValues = testModel.getConfidenceValues(dataflow)
        end_time = time.time()

        print("time: " + str(end_time-start_time))

        predictedClasses = testModel.getBinaryPredictions(confideceValues)
        confusionMatrix = binaryConfusionMatrix(targetData, predictedClasses, classNames = ['Non-MCT_Violation','MCT_Violation']) #get confusion matrix
        binaryConfusionMatrixAnalysis(confusionMatrix) #analyse confucion matrix
        showROCCurve(targetData, predictedClasses)
    
    def heftyEvaluation(self):
        mctFeeder = Feeder()
        dataFlow, targetData = mctFeeder.loadData
    
    def modelROCComparision(self):
        model_1 = Model("Test_MCT_Model", 'TrainedModels\defaultTestModel')
        model_2 = Model("600KModel", 'TrainedModels\TrainingSet_600')

        mctFeeder = Feeder() #make ourselves a feeder
        dataflow, targetData = mctFeeder.getMCTData1(evaluateDataStream = False, evaluateTargetData = False, streamSize = 500000)

        print(str(model_1.name) + " Predicting for " + str(len(dataflow)) + " Entries...")

        start_time = time.time()
        confideceValues1 = model_1.getConfidenceValues(dataflow)
        end_time = time.time()

        print(str(model_1.name) + " time: " + str(end_time-start_time))

        print(str(model_2.name) + " Predicting for " + str(len(dataflow)) + " Entries...")

        start_time = time.time()
        confideceValues2 = model_2.getConfidenceValues(dataflow)
        end_time = time.time()

        print(str(model_1.name) + " time: " + str(end_time-start_time))

        predictedClasses1 = model_1.getBinaryPredictions(confideceValues1)
        predictedClasses2 = model_2.getBinaryPredictions(confideceValues2)
        showROCCurve2Models(targetData, predictedClasses1)
    
    
        
    
        

