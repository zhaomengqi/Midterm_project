# Zillow-Data-Analysis


## Working with large datasets and machine learning: working on a real-world data set of Zillow data and expand the housing problem in the class.   https://www.continuum.io/blog/developer-blog/productionizing-and-deploying-data-science- 

You could review the following and the links posted to understand the problem:
• https://www.kaggle.com/c/zillow-prize-1
• https://www.kaggle.com/philippsp/exploratory-analysis-zillow
• https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-zillow-prize 
• https://www.kaggle.com/captcalculator/a-very-extensive-zillow-exploratory-analysis

### 1.Data ingestion, EDA, Wrangling:
• Download the data from Zillow. (https://www.kaggle.com/c/zillow-prize-1)

• Create an IPYB notebook and Conduct an in-depth EDA (See below for ideas; 

• Put together a note on what data cleansing is required for automation

• Clean up the data and take care of missing data values using a Python script

• Combine the 2016 and 2017 properties by adding an additional column for year

• Programmatically write the data to a S3 bucket named “ZillowData”. This should be
downloadable by anyone who has the links.

• Write a report documenting your data ingestion, wrangling steps.

### 2.Build a prediction model
• try out different prediction models to predict the log errors.

• Use RMS and MAPE as measures and try:
o Multiple linear regression o Randomforests
o Neural networks

• Which model works best? Write a report discussing the different models you considered and which one works best. You should consider interpretability, computational overhead, accuracy measures, etc. in your discussion.

###  3.Model deployment: choose an enterprise platform to deploy your model.
https://docs.google.com/spreadsheets/d/17NqDJHdJtqfvgVHAl2_YplG9O8t3PvWB233YaByI2w8/edit?ts=59c65fd3#gid=558508381

• Trained model from step 2 or redo the “best” model in the assigned platform. 
• Advertise the JSON API to use to invoke the model
• Deploy the model and provide examples on how to invoke the api and how to interpret the
results. Create a Jupyter notebook to illustrate how to use your REST API

###  4.Enhancing your REST API: Geospatial search: each record has a Latitude and Longitude. The goal is to create a REST API that given a Lat and Long, should return the top 10 closest homes.

Write a Jupyter notebook and illustrate using this REST API and review these articles for the algorithm and how to use SQL to get these results:

✓ http://www.arubin.org/files/geo_search.pdf

✓ https://www.percona.com/blog/2014/06/19/using-udfs-for-geo-distance-search-in-mysql/ 

✓ https://www.percona.com/blog/2013/10/21/using-the-new-mysql-spatial-functions-5-6-for-geo-enabled-applications/

    
