import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from Tools.dataSpout import dataSpout

class dataVisualiser(): #inherits a Feeder
    #if we can load a pandas df from a dataStream dict, that would be nice.
    def __init__(self, mode, parameter):
        self.fileLocation = parameter
        #if we are reading from a csv the parametwer needs to point to a valid location to a csv.
        try:
            if mode == 'csv':
                self.dataflow = self.loadDataToAnalyse(parameter)
            #if the mode is dict mode then the parameter parsed needs to be a dictionary.
            elif mode == 'dict':
                self.dataflow = pd.DataFrame.from_dict(parameter)
        except:
            print("Error. Make sure that the mode is set to 'csv' or 'dict'. The parameter has to be a valid file location or a dictionary from a dataspout datastream")
       
    def loadDataToAnalyse(self, pathOfCSV):
        dataflow = pd.read_csv(pathOfCSV) #thank you pandas :)
        #print(dataflow.head())
        return dataflow
    
    def info(self):
        self.dataflow.info()

    def head(self):
        print(self.dataflow.head())

    def distributionPlot(self, field): 
        sns.displot(self.dataflow[field], kde=True)
        plt.show()

    def histogramPlot(self, field): #histplot
        sns.histplot(self.dataflow[field])
        plt.show()

    def jointPlot(self, xAxis, yAxis): #jointplot
        sns.jointplot(x=xAxis, y=yAxis, data = self.dataflow, kind = 'reg') #kind = 'reg', 'kde', 'hex'
        plt.show()

    #kernel distribution plot is commonly used.
    def kdePlot(self, x_, y_, hue_):
        sns.kdeplot(self.dataflow, x=x_, y=y_, hue= hue_, kind = "kde")
        plt.show()

    #tonnes of information
    def pairPlot(self):
        sns.pairplot(self.dataflow)
        plt.show()
    
    def linePlot(self, x_, y_):
        sns.lineplot(x = x_, y = y_,data = self.dataflow)
        plt.show()

    def generateHeatmap(self):

        plt.figure(figsize=(12,10))
        sns.set_context('paper', font_scale=1.0)

        dataMatrix = self.dataflow.corr()
        sns.heatmap(dataMatrix, annot=True, cmap="coolwarm")
        plt.show()
    
    def groupBoxplot(self, x_, y_, hue_):
        sns.set_theme(style="ticks", palette="pastel")

        # Load the example tips dataset
        data_ = self.dataflow

        # Draw a nested boxplot to show bills by day and time
        sns.boxplot(x=x_, y=y_,
                    hue=hue_, palette=["m", "g"],
                    data=data_)
        sns.despine(offset=10, trim=True)

        plt.show()

    def implot(self, x_, y_, hue_):
        #get data
        data_ = self.dataflow

        # Plot sepal width as a function of sepal_length across days
        g = sns.lmplot(
            data= data_,
            x= x_, y=y_, hue=hue_,
            height=5
        )
            
        # Use more informative axis labels than are provided by default
        #g.set_axis_labels("Snoot length (mm)", "Snoot depth (mm)")
        plt.show()
    
    def groupedBarplot(self, x_, y_, hue_):
        
        sns.set_theme(style="darkgrid")


        # Draw a nested barplot by species and sex
        g = sns.catplot(
            data=self.dataflow, kind="bar",
            x=x_, y=y_, hue=hue_,
            ci="sd", palette="dark", alpha=.6, height=6
        )
        g.despine(left=True)
        g.set_axis_labels(str(x_), str(y_))
        g.legend.set_title("")
        plt.show()
    
    def stackedHistogram(self, x_, hue_):
        sns.set_theme(style="ticks")

        f, ax = plt.subplots(figsize=(7, 5))
        sns.despine(f)

        sns.histplot(
            data = self.dataflow,
            x=x_, hue=hue_,
            multiple="stack",
            palette="light:m_r",
            edgecolor=".3",
            linewidth=.5,
            log_scale=True,
        )
        ax.xaxis.set_major_formatter(plt.ticker.ScalarFormatter())
        ax.set_xticks([500, 1000, 2000, 5000, 10000])
    

    
    