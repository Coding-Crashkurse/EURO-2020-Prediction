import pandas as pd


teams = ['Austria', 'Belgium', 'Croatia', 'Czech Republic', 'Denmark', 'England', 'Finland', 'France', 'Germany', 'Hungary', 'Italy', 'Netherlands',
 'North Macedonia', 'Poland', 'Portugal', 'Russia', 'Scotland', 'Slovakia', 'Spain', 'Sweden', 'Switzerland', 'Turkey', 'Ukraine', 'Wales']

data = pd.read_csv("C:/Users/User/Desktop/PythonEuro/results.csv")
# Drop columns
df = data.drop(["city", "country", "neutral"], axis=1)
df = df[df.date > '2015-01-01']
df = df.drop(["tournament", "date"], axis=1)

df.columns

df = df[(df["home_team"].isin(teams)) | (df["away_team"].isin(teams))]

def relabel(df):
    if df["home_score"] == df["away_score"]:
        return 0
    elif df["home_score"] > df["away_score"]:
        return 1
    else: 
        return -1

df["Heimsieg"] = df.apply(relabel, axis=1)
df.drop(["home_score", "away_score"], axis=1, inplace=True)
df.dtypes

df

obj_cols = df.columns[df.dtypes == "object"].to_list()

# Encoding
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
encoded_columns = pd.DataFrame(enc.fit_transform(df[obj_cols]))
numerical_data = df.drop(obj_cols, axis=1)

numerical_data

numerical_data = numerical_data.reset_index().drop("index", axis=1)
encoded_columns = encoded_columns.reset_index().drop("index", axis=1)

prediction_df = pd.concat([numerical_data, encoded_columns], axis=1)
prediction_df.head()
      
y_data = prediction_df["Heimsieg"]
X_data = prediction_df.drop("Heimsieg", axis=1)      
     
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, train_size=0.8, test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=250, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_pred, y_test)


### Neue kommende Fälle
home_teams = ['Turkey', 'Wales', 'Denmark', 'Belgium', 'England', 'Austria', 'Netherlands', 'Scotland', 'Poland', 'Spain', 'Hungary', 'France']
away_teams = ['Italy', 'Switzerland', 'Finland', 'Russia', 'Croatia', 'North Macedonia', 'Ukraine', 'Czech Republic', 'Slovakia', 'Sweden', 'Portugal', 'Germany']

kommende_spiele_dict = {'home_team' : home_teams,
                       'away_team' : away_teams,
                      }

kommende_spiele = pd.DataFrame(data=kommende_spiele_dict)

gesamt_df = df.append(kommende_spiele).reset_index()
gesamt_df.drop("index", axis=1, inplace=True)

encoded_colums = pd.DataFrame(enc.fit_transform(gesamt_df[obj_cols]))
numerical_data = gesamt_df.drop(obj_cols, axis=1)
numerical_data = numerical_data.reset_index().drop("index", axis=1)
encoded_columns = encoded_columns.reset_index().drop("index", axis=1)

# One-hot encoding removed index; put it back
prediction_df = pd.merge(numerical_data, encoded_colums, left_index=True, right_index=True)
prediction_df
      
y_data = prediction_df["Heimsieg"]
X_data = prediction_df.drop("Heimsieg", axis=1)  

X_train = X_data.head(1247)
X_test = X_data.tail(12)
y_train = y_data.head(1247)

clf = RandomForestClassifier(n_estimators=250, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
result = pd.merge(kommende_spiele, pd.Series(y_pred, name="pred"), left_index=True, right_index=True)

mapper = {1: "HEIMSIEG", -1: "HEIMNIEDERLAGE", 0: "UNENTSCHIEDEN"}

for index, row in kommende_spiele.iterrows():
    print(f"Vorhersage für {row['home_team']} gegen {row['away_team']}: {mapper.get(y_pred[index])}")


def lists_to_df(home_teams, away_teams):
    kommende_spiele_dict = {'home_team' : home_teams, 'away_team' : away_teams }
    kommende_spiele = pd.DataFrame(data=kommende_spiele_dict)
    return kommende_spiele
    

gruppe_a_home = ["Turkey", "Wales", "Turkey", "Italty", "Italy", "Switzerland"]    
gruppe_a_away = ["Italy", "Switzerland", "Wales", "Switzerland", "Wales", "Turkey"]

lists_to_df(gruppe_a_home, gruppe_a_away)


teams = {"Turkey": 0, "Wales": 0, "Italy": 0, "Switzerland": 0}

for index, row in result.iterrows():
    if row["pred"] == 1:
        teams[row["home_team"]] += 3
    elif row["pred"] == -1:
        teams[row["away_team"]] += 3
    else:
        teams[row["away_team"]] += 1
        teams[row["away_team"]] += 1

