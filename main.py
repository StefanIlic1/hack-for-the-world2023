import pandas as pd
import numpy as np
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics


general_data = pd.read_csv("data.csv")
counties = pd.read_csv("Counties.csv", sep = ";")
states = pd.read_csv("states.csv")

def identify_state_of_county(county_id):
    """This function returns the city that the given county id is in"""
    """If the county is not in the specific list of states, then it returns an empty df"""
    num = str(county_id)[0:2]
    num = int(num)
    return states[states["estados_id"] == num]

def transform(df):
    #It creates two lists with the name of the county and its state
    munic = []
    esta = []
    for i in range(len(df["municipal_id"])):
        ind = counties[counties["municipal_code"] == df["municipal_id"][i]]["name_of_county"].index[0]
        nome_mun = counties[counties["municipal_code"] == df["municipal_id"][i]]["name_of_county"][ind]

        ind_es = counties[counties["municipal_code"] == df["municipal_id"][i]]["municipal_code"][ind]
        m = identify_state_of_county(ind_es)["Estados"].index[0]
        nome_est = identify_state_of_county(ind_es)["Estados"][m]

        munic.append(nome_mun)
        esta.append(nome_est)
    return munic, esta

def stats_per_year(df,stat,Mean):
    """ returns the sum of a given sum of a specific stat grouped by years"""
    sum = df[["year",stat]].groupby(['year']).sum()
    media = sum[stat].mean()
    vals = []
    if Mean == False:
        for k in sum[stat]:
            vals.append(k)
    else:
        for k in sum[stat]:
            vals.append(k/media)
        return np.array(vals),sum.index



# Plotting statistical correlations 

new = general_data[["year","municipal_id","area","deforested","increment","forest","clouds","not_observed","deforestation","hydrography"]]
corr = round(new.corr(),2)
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(9, 5))

sns.heatmap(corr, mask=mask, vmin=-1,vmax=1,annot = True)



lista_mun , lista_est = transform(general_data) #lets add the new columns to our dataset
general_data["municipal_name"] = lista_mun
general_data["state_name"] = lista_est


lista1, anos1 = stats_per_year(general_data,"deforested",True)
lista2, anos2 = stats_per_year(general_data,"forest",True)
print(lista2)
plt.figure(figsize=(9, 5))
plt.plot(list(anos1),lista1,"r", label="Deforestation Rate of Change")
plt.plot(list(anos2),lista2,"g", label="Forest Area Rate of Change")
plt.ylabel("Percentage %")
plt.legend()
plt.show()

indexedList = general_data[["year","municipal_id","area","deforested","increment","forest","clouds","not_observed","deforestation","hydrography"
]]

years = list(set(indexedList["year"]))
yearsList = [] 
deforestationLevel = [] 
municipalsList = [] 
statesList = [] 

for year in years: 
    new = general_data[general_data["year"]==year]
    new = new.sort_values(by = ["deforestation"], ascending = False)

    yearsList.append(np.array(list(new.copy().iloc[0:10]["year"])[:]))
    deforestationLevel.append(np.array(list(new.copy().iloc[0:10]["deforestation"])[:]))
    municipalsList.append(np.array(list(new.copy().iloc[0:10]["municipal_name"])[:]))
    statesList.append(np.array(list(new.copy().iloc[0:10]["state_name"])[:]))

newDictionary =  dict() 
newDictionary["yearsList"] = np.array(yearsList).reshape(1,-1)[0]
newDictionary["deforestationLevel"] = np.array(deforestationLevel).reshape(1,-1)[0]
newDictionary["municipalsList"] = np.array(municipalsList).reshape(1,-1)[0]
newDictionary["statesList"] = np.array(statesList).reshape(1,-1)[0]
statsDataFrame = pd.DataFrame(newDictionary)

municipals = list(Counter(statsDataFrame["municipalsList"]))

mat = []
for i in range(len(municipals)):  
  data = general_data[general_data["municipal_name"] == municipals[i]]
  mat.append(stats_per_year(data,"deforested",True)[0])

plt.figure(figsize=(15, 8))
for i in range(len(municipals)):
  if municipals[i] == "São Félix do Xingu":
    plt.plot(list(set(statsDataFrame.yearsList)),mat[i])  
    plt.text(list(set(statsDataFrame.yearsList))[-1]-4, mat[i][-1]-1000, str(municipals[i]), fontsize = 10)
  elif municipals[i] == "Arinos":
    plt.plot(list(set(statsDataFrame.yearsList)),mat[i])  
    plt.text(list(set(statsDataFrame.yearsList))[-1]-4, mat[i][-1], str(municipals[i]), fontsize = 10)
  else:
    plt.plot(list(set(statsDataFrame.yearsList)),mat[i])
    plt.text(list(set(statsDataFrame.yearsList))[-1], mat[i][-1], str(municipals[i]), fontsize = 10)
plt.ylabel("Deforestation Rate",fontsize = 20)
plt.show()







"""
# Setting up machine learning 
listOfStates = list(Counter(general_data["state_name"]))
mat = []



for i in range(len(listOfStates)):
    data =  general_data[general_data["state_name"]==listOfStates[i]]
    mat.append(stats_per_year(data,"deforestation",False)[0])



print("The length of the states are ", (len(listOfStates)))
print("The legnth of the mat is", len(mat))


trackedYears = []
trackedStates = []

for st in listOfStates:
    for i in range(2000, 2022):
        trackedYears.append(i)
        trackedStates.append(st)

d = {}

d["year"] = trackedYears
d["state"] = trackedStates
d["deforestation"] = np.array(mat).reshape(1,-1)[0]

X = pd.DataFrame(d)
labels = list(Counter(X["state"]))

X["state"] = LabelEncoder().fit_transform(X["state"])
Y = X.pop("deforestation")
encoded_labels = list(Counter(X["state"]))

# Machine Learning Model 

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

KN = KNeighborsRegressor()
bag = BaggingRegressor()

mod = GridSearchCV(estimator=KN, param_grid= {'n_neighbors':[1,2,3,4,5,6,7,8,9,10]}, cv=2)
mod2 = GridSearchCV(estimator=bag,param_grid= {'n_estimators':[100,120,130,150,180]},cv=2)
mod3 = GridSearchCV(estimator=SGDRegressor(max_iter=1200,early_stopping=True),param_grid={'penalty':["l1","l2"]} ,cv=2)

vot = VotingRegressor(estimators=[('kn',mod),('bag',mod2),('est',mod3)])

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.1,random_state=True)

vot.fit(X_train,y_train)
metrics.r2_score(y_test,vot.predict(X_test))
# training score

m = []
for i in encoded_labels:
  for year in range(2022,2024): # to change to 2030 
      m.append([year,i])
pred = scaler.transform(m)

prediction = vot.predict(pred)

Df = pd.DataFrame(d)
ano = [i for i in range(2000,2024)]

plt.figure(figsize=(15,8))
c = 0
for i in labels:
  dat = Df[Df["state"] == i]
  es = list(dat["deforestation"])
  es.append(prediction[c])
  es.append(prediction[c+1])
  plt.plot(ano,es,label = i)
  c+=2

plt.axvline(2021, color='k', linestyle='--')
plt.legend()
plt.xticks(ano, rotation=45)
plt.show()
"""