#####
#CAMELS Linear Regression Models: REGIONAL
#Written by: Logan Qualls, email: lmqualls@crimson.ua.edu
#Finalized on: November 29, 2020
#Loops through precipitation and streamflow data for CAMELS basins (gauges) in a SPECIFIED hydrological region
#and creates linear models for those gauges; stores all r-squared values representing the comparison between predicted data
#and observed streamflow data; creates attribute field for r-squared value and fills that field with corresponding
#r-squared values of the linear model run for that gauge
#Parameters include: (0) workspace, (1) hydrological region (01 - 18), (2) path to "maurer" folder, (3) path to "usgs_streamflow" folder,
#(4) path to "HCDN_nhru_final_671" shapefile
#####

#Import libraries
import pandas as pd
import numpy as np
import arcpy
from sklearn.linear_model import LinearRegression
import os

#Set workspace (Output Folder)
arcpy.env.workspace = arcpy.GetParameterAsText(0)

#Set paths to maurer precipitation data and USGS streamflow data
region = arcpy.GetParameterAsText(1)
maurer = arcpy.GetParameterAsText(2)
usgs = arcpy.GetParameterAsText(3)
maurer_reg = os.path.join(maurer,region)
usgs_reg = os.path.join(usgs,region)
pfiles = os.listdir(maurer_reg)
qfiles = os.listdir(usgs_reg)

#Extract gauge from filename
gauges = []
for gauge in qfiles:
    substring = gauge[0:8]
    gauges.append(substring)

#Create dictionary for basin areas
areadf = {}
for gauge in gauges:
    pfile = maurer_reg + "\\" + gauge + "_lump_maurer_forcing_leap.txt"
    file = open(pfile,'r')
    lines = file.readlines()
    area = lines[2]
    area = eval(area)
    areadf.update({gauge: area})
    file.close()

#Read in precipitation data
colnames = ('Year','Mnth','Day')
pfile = maurer_reg +  '\\' + pfiles[0]
pdf = pd.read_csv(pfile, sep = "\s+", header = 3, usecols = [0,1,2], names = colnames)
maurer_datetime = pd.to_datetime(pdf[['Year', 'Mnth', 'Day']].rename(columns={'Year': 'year', 'Mnth': 'month', 'Day': 'day'}))
pdf["Datetime"] = maurer_datetime
pdf = pd.DataFrame(pdf["Datetime"])
pdf = pdf.set_index('Datetime')

colnames = ('Year','Mnth','Day','P0')
for gauge in gauges:
    pfile = maurer_reg + "\\" + gauge + "_lump_maurer_forcing_leap.txt"
    p = pd.read_csv(pfile, sep = "\s+", header = 3, usecols = [0,1,2,5], names = colnames)
    maurer_datetime = pd.to_datetime(p[['Year', 'Mnth', 'Day']].rename(columns={'Year': 'year', 'Mnth': 'month', 'Day': 'day'}))
    p["Datetime"] = maurer_datetime
    p = p[["Datetime","P0"]]
    p = p.set_index("Datetime")
    pdf = pdf.join(p,how = 'inner',rsuffix= f'_{gauge}')
pdf.columns = gauges

#Read in streamflow data
colnames =  ("Year","Month","Day")
qfile = os.path.join(usgs_reg,qfiles[0])
qdf = pd.read_csv(qfile, sep = "\s+", usecols = [1,2,3], names = colnames)   #Following should work, issue with access
usgs_datetime = pd.to_datetime(qdf[['Year', 'Month', 'Day']].rename(columns={'Year': 'year', 'Month': 'month', 'Day': 'day'}))
qdf["Datetime"] = usgs_datetime
qdf = pd.DataFrame(qdf["Datetime"])
qdf = qdf.set_index("Datetime")

colnames =  ("Year","Month","Day","Q0")
for gauge in gauges:
    p = pd.DataFrame()
    qfile = usgs_reg + "\\" + gauge + "_streamflow_qc.txt"
    q = pd.read_csv(qfile, sep = "\s+", usecols = [1,2,3,4], names = colnames)
    usgs_datetime = pd.to_datetime(q[['Year', 'Month', 'Day']].rename(columns={'Year': 'year', 'Month': 'month', 'Day': 'day'}))
    q["Datetime"] = usgs_datetime
    q = q[["Datetime","Q0"]]
    q = q.set_index("Datetime")
    qdf = qdf.join(q,how = 'right',rsuffix = f'_{gauge}')
qdf.columns = gauges

# Replace missing data with NaNs
for gauge in gauges:
    qdf[gauge] = qdf[gauge].replace(-999.0, np.nan)

# Divide Raw Streamflow Data by Basin Area to Normalize
for gauge in gauges:
    normalizer = lambda x: x / float(areadf[gauge])
    qdf[gauge] = qdf[gauge].apply(normalizer)

# Standardize Streamflow Data
for gauge in gauges:
    mean = qdf[gauge].mean()
    stdev = qdf.std()[gauge]
    standardizer = lambda x: (x - mean) / stdev
    qdf[gauge] = qdf[gauge].apply(standardizer)

#Define Train/Test Periods
train = pd.date_range('10/01/1999','09/30/2008')
test = pd.date_range('10/01/1989','09/30/1999')

# Create lagged dataset, run model, fill dictionary with gauge: r-sqaured info for each model
r2_dict = {}

for gauge in gauges:
    pqdf = pd.DataFrame()
    pqdf["P0"] = pdf[gauge]
    pqdf["Q0"] = qdf[gauge]
    pqdf['P1'] = pqdf['P0'].shift(periods=1, fill_value=np.nan)
    pqdf['Q1'] = pqdf['Q0'].shift(periods=1, fill_value=np.nan)
    pqdf['P2'] = pqdf['P0'].shift(periods=2, fill_value=np.nan)
    pqdf['Q2'] = pqdf['Q0'].shift(periods=2, fill_value=np.nan)
    pqdf['P3'] = pqdf['P0'].shift(periods=3, fill_value=np.nan)
    pqdf['Q3'] = pqdf['Q0'].shift(periods=3, fill_value=np.nan)
    pqdf['P4'] = pqdf['P0'].shift(periods=4, fill_value=np.nan)
    pqdf['Q4'] = pqdf['Q0'].shift(periods=4, fill_value=np.nan)
    pqdf['P5'] = pqdf['P0'].shift(periods=5, fill_value=np.nan)
    pqdf['Q5'] = pqdf['Q0'].shift(periods=5, fill_value=np.nan)
    pqdf = pqdf.dropna()

    ytrain = pqdf['Q0'].loc[train[0]:train[-1]]
    ytest = pqdf['Q0'].loc[test[0]:test[-1]]

    xcols = [col for col in pqdf.columns if col != 'Q0']
    xtrain = pqdf[xcols].loc[train[0]:train[-1]]
    xtest = pqdf[xcols].loc[test[0]:test[-1]]

    reg = LinearRegression().fit(xtrain, ytrain)
    r2 = reg.score(xtest, ytest)
    r2_dict[gauge[1:9]] = r2

#Add column to attribute table ("r_squared")
shapefile = arcpy.GetParameterAsText(4)
rsquared = arcpy.AddField_management(shapefile, "r_squared", "FLOAT")

#In row where hr_ID == gauge, change r-squared value to r-squared of corresponding linear model prediction
for gauge,r in r2_dict.items():
    with arcpy.da.UpdateCursor(shapefile,["hru_id","r_squared"]) as cursor:
        for row in cursor:
            if row[0] != int(gauge): continue
            if row[0] == int(gauge):
                row[1] = r
                cursor.updateRow(row)
                del cursor