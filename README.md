# AirlineNoShowPrediction
Using Machine learning to model predictions of people not showing up for flights.
A tool kit I designed/made a while back for shaping/saving and manipulating data for any kind of machine learning model. It implements many python packages, Tensorflow, pandas, numpy, seaborn, SHAP and many others.
Inside of the tools there is a Readme.py showing how to use these tools for machine learning models.
There are also some visualisation tools for visualizing data used for machine learning models as well as graphs visualizing the output of the models.
This will work for any kind of data in the format of a .csv or .json file and can save manipulated training as either.

Here are some random pictures I found from this project.

there is functionality to show a confusion matrix for results of model
![98 537](https://user-images.githubusercontent.com/60296036/212785887-6a081085-1fd6-49f2-a7f1-4d2b0a7a15b8.JPG)
![97 95](https://user-images.githubusercontent.com/60296036/212786614-0e2d066b-0e78-45f1-a51f-cba46d0bed6f.JPG)

Training visualisation

![97 9](https://user-images.githubusercontent.com/60296036/212787430-f2819a1e-3e9c-4b64-87ec-a17390845212.JPG)
![loss](https://user-images.githubusercontent.com/60296036/212787460-208a6d3f-c0ec-4e50-8480-3d4aabbb89fe.JPG)


As well as all kinds of visualisations for the training data. You can plot any feature agains any other by just passing the column names into the data visualisation functionality.

![randomGraphsOfStuffs](https://user-images.githubusercontent.com/60296036/212785967-71d1ac3d-77ea-4294-954d-27818c8bf1dc.JPG)

Even 3D graph models.
![plot1](https://user-images.githubusercontent.com/60296036/212787126-0fd70ab5-46b9-4b22-899d-00538cf3d1eb.JPG)
![2](https://user-images.githubusercontent.com/60296036/212787151-b49f2956-59fc-4f98-bb70-06d508a7bd06.JPG)

For output visualisation

significance of each feature for the output. This was derived by calculating shap (weighted) values which uses a recursive game theoretical approach. May explain later but its just a python library.
![significantFeatures](https://user-images.githubusercontent.com/60296036/212786273-9d969f15-f904-431f-aae4-d1d4dbf9774a.JPG)
![vis1](https://user-images.githubusercontent.com/60296036/212787562-433db769-f1d7-43a1-af5a-83ba25c07e46.PNG)
![vis2](https://user-images.githubusercontent.com/60296036/212787577-3bb37922-e22a-428a-8580-547b490054c4.PNG)
![vis3](https://user-images.githubusercontent.com/60296036/212787590-2ccfa8f8-229e-4167-8a16-983afc919196.PNG)
![vis4](https://user-images.githubusercontent.com/60296036/212787599-28ba1ff7-4fb7-4b07-a82d-fa88aab306c3.PNG)
![vis5](https://user-images.githubusercontent.com/60296036/212787612-0d9d672d-d843-4b6a-86f6-58e51bfc671c.PNG)


I forgot what these are but they look cool
![pnrdaily](https://user-images.githubusercontent.com/60296036/212786484-09be1f61-7a9d-4128-8c17-e4351145d394.JPG)

![pnrDailyNS](https://user-images.githubusercontent.com/60296036/212786495-ae9b7a3a-1385-4748-88b3-eb624f9e353b.JPG)

But ultimately yeah. I think this is pretty implementable for most ML models and datasets.
