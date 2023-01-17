from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt #for visualising 
import numpy as np

def binaryConfusionMatrix(title, targetData, predictions, classNames):
        cm = confusion_matrix(targetData, predictions)

        plt.clf()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
        plt.title(title)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        tick_marks = np.arange(len(classNames))
        plt.xticks(tick_marks, classNames, rotation=45)
        plt.yticks(tick_marks, classNames)
        #s = [['TrueNeg.\n','FalsePos.\n'], ['FalseNeg.\n', 'TruePos.\n']]
        for i in range(2):
            for j in range(2):
                plt.text(j,i, str(cm[i][j]))
        plt.show()
        return cm

def binaryConfusionMatrixAnalysis(confusionMatrix):
    """1. find accuracy. defined as correctPredictions/TotalPredictions
       2. Sensitivity (True Positive rate). defined as TP/(FN+TP)
       3. Specificity (True Negative rate). defiend as TN/(TN+FP)
       4. False positive rate. defined as FP/(TN+FP)
    """
    TP = confusionMatrix[0][0]
    FP = confusionMatrix[0][1]
    FN = confusionMatrix[1][0]
    TN = confusionMatrix[1][1]

    accuracy = (TP + TN)/(TP+FP+FN+TN)
    sensitivity = TP/(FN+TP)
    specificity = TN/(TN+FP)
    FalsePosRate = FP/(FP+TN)

    print("Accuracy: " + str(accuracy))
    print("Sensitivity (True Positive Rate) TP/(FN+TP) = " + str(sensitivity))
    print("Specificity (True Negative Rate) TN/(TN+FP) = " + str(specificity))
    print("False Positive Rate. FP/(FP+TN) = " + str(FalsePosRate))

    return TP, FP, FN, TN

def showROCCurve(targetLabels, PredictedValues, modelName):
    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(targetLabels, PredictedValues)
    # calculate AUC
    auc = roc_auc_score(targetLabels, PredictedValues)
    print('AUC: %.3f' % auc)

    # roc curve for tpr = fpr 
    random_probs = [0 for i in range(len(targetLabels))]
    p_fpr, p_tpr, _ = roc_curve(targetLabels, random_probs, pos_label=1)

    # calculate roc curves
    fpr1, tpr1, _ = roc_curve(targetLabels, PredictedValues)
    # plot the roc curve for the model
    plt.style.use('seaborn')

    # plot roc curves
    plt.plot(fpr1, tpr1, linestyle='--',color='orange', label=modelName)
    plt.plot(p_fpr, p_tpr, linestyle='--', color='blue', label ='Equilibrium')
    # title
    plt.title('ROC curve')
    # x label
    plt.xlabel('False Positive Rate')
    # y label
    plt.ylabel('True Positive rate')

    plt.legend(loc='best')
    plt.savefig('ROC',dpi=300)
    plt.show();

def showROCCurve2Models(targetLabels, predictedValues1, predictedvalues2, modelName1, modelName2):
    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(targetLabels, predictedValues1)
    # calculate AUC
    auc = roc_auc_score(targetLabels, predictedValues1)
    print('AUC: %.3f' % auc)

    # roc curve for tpr = fpr 
    random_probs = [0 for i in range(len(targetLabels))]
    p_fpr, p_tpr, _ = roc_curve(targetLabels, random_probs, pos_label=1)

    # calculate roc curves
    fpr1, tpr1, _ = roc_curve(targetLabels, predictedValues1)
    fpr2, tpr2, _ = roc_curve(targetLabels, predictedvalues2)
    # plot the roc curve for the model
    plt.style.use('seaborn')

    # plot roc curves
    plt.plot(fpr1, tpr1, linestyle='--',color='orange', label=modelName1)
    plt.plot(fpr2, tpr2, linestyle='--',color='green', label=modelName2)
    plt.plot(p_fpr, p_tpr, linestyle='--', color='blue', label='Equilibrium')
    # title
    plt.title('ROC curve')
    # x label
    plt.xlabel('False Positive Rate')
    # y label
    plt.ylabel('True Positive rate')

    plt.legend(loc='best')
    plt.savefig('ROC',dpi=300)
    plt.show();