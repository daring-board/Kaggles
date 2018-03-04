import pandas as pd
import csv as csv
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

def addFamily(df):
    df['Family'] = df['SibSp'] + df['Parch']
    df['Family'].loc[df['Family'] > 0] = 1
    df['Family'].loc[df['Family'] == 0] = 0

def modifyAge(df):
    '''
    中央値を取得する
    '''
    median_age = df['Age'].dropna().median()
    '''
    欠損値ならば、中央値を与える
    '''
    for idx in df.index:
        if not isinstance(df.loc[idx, 'Age'], int):
            df.loc[idx, 'Age'] = median_age

def modifyEmbarked(df):
    '''
    船に乗った港
    '''
    df["Embarked"] = df["Embarked"].fillna("S")
    df.loc[df["Embarked"] == "S", "Embarked"] = 0
    df.loc[df["Embarked"] == "C", "Embarked"] = 1
    df.loc[df["Embarked"] == "Q", "Embarked"] = 2


train_df = pd.read_csv("train.csv", header=0)
print(train_df.head(5))

'''
データフレームにGenderを追加。
Sexカラムがfemaleならば0、maleならば1
'''
train_df['Gender'] = train_df['Sex'].map({'female': 0, 'male': 1}).astype(int)
print(train_df.head(5))

'''
年齢データを加工する
'''
modifyAge(train_df)

'''
客室番号のデータを加工する
'''
rooms = ['A', 'B', 'C', 'D', 'E', 'F']
for idx in train_df.index:
    if not pd.isnull(train_df.iloc[idx]['Cabin']):
        for r_idx in range(len(rooms)):
            if isinstance(train_df.iloc[idx]['Cabin'], int):
                continue
            if rooms[r_idx] in train_df.iloc[idx]['Cabin']:
                train_df.loc[idx, 'Cabin'] = r_idx
        if not isinstance(train_df.iloc[idx]['Cabin'], int):
            train_df.loc[idx, 'Cabin'] = 99
    else:
        train_df.loc[idx, 'Cabin'] = 99


# '''
# 兄弟または配偶者の数のデータを加工する
# '''
# modifySibSp(train_df)
#
# '''
# 子供または親の数のデータを加工する
# '''
# modifyParch(train_df)
addFamily(train_df)

'''
乗船港のデータを加工する
'''
modifyEmbarked(train_df)

train_df = train_df.drop(["Name", "Ticket", "Sex", "Parch", "SibSp", "Fare", "PassengerId"], axis=1)
print(train_df.head(5))

'''
テストデータを読み込む
'''
test_df = pd.read_csv('test.csv', header=0)

test_df['Gender'] = test_df['Sex'].map({'female': 0, 'male': 1}).astype(int)
print(test_df.head(5))

'''
年齢データを加工する
'''
modifyAge(test_df)

'''
客室番号のデータを加工する
'''
rooms = ['A', 'B', 'C', 'D', 'E', 'F']
for idx in test_df.index:
    if not pd.isnull(test_df.iloc[idx]['Cabin']):
        for r_idx in range(len(rooms)):
            if isinstance(test_df.iloc[idx]['Cabin'], int):
                continue
            if rooms[r_idx] in test_df.iloc[idx]['Cabin']:
                test_df.loc[idx, 'Cabin'] = r_idx
        if not isinstance(test_df.iloc[idx]['Cabin'], int):
            test_df.loc[idx, 'Cabin'] = 99
    else:
        test_df.loc[idx, 'Cabin'] = 99

# '''
# 兄弟または配偶者の数
# '''
# modifySibSp(test_df)
# '''
# 子供または親の数のデータを加工する
# '''
# modifyParch(test_df)
addFamily(test_df)

'''
乗船港のデータを加工する
'''
modifyEmbarked(test_df)

'''
PassengerIdを取り出し、解析に使用しないカラムをドロップ
'''
ids = test_df['PassengerId'].values
test_df = test_df.drop(["Name", "Ticket", "Sex", "Parch", "SibSp", "Fare", "PassengerId"], axis=1)

train_data = train_df.values
test_data = test_df.values

#model = RandomForestClassifier(n_estimators=50)
model = GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3)
fitted = model.fit(train_data[0::, 1::], train_data[0::, 0])
output = fitted.predict(test_data).astype(int)

submit_fit = open('titanic_submit.csv', 'w')
file_object = csv.writer(submit_fit, lineterminator='\n')
file_object.writerow(['PassengerId', 'Survived'])
for idx in range(len(ids)):
    file_object.writerow([ids[idx], output[idx]])
submit_fit.close()
