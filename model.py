import pandas as pd
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from sklearn.preprocessing import StandardScaler
from keras.wrappers.scikit_learn import KerasClassifier

def prep_data(df):
    # Drop unwanted features
    df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

    # Fill missing data: Age and Fare with the mean, Embarked with most frequent value
    df[['Age']] = df[['Age']].fillna(value=df[['Age']].mean())
    df[['Fare']] = df[['Fare']].fillna(value=df[['Fare']].mean())
    df[['Embarked']] = df[['Embarked']].fillna(value=df['Embarked'].value_counts().idxmax())

    # Convert categorical  features into numeric
    df['Sex'] = df['Sex'].map({'female': 1, 'male': 0}).astype(int)

    # Convert Embarked to one-hot
    enbarked_one_hot = pd.get_dummies(df['Embarked'], prefix='Embarked')
    df = df.drop('Embarked', axis=1)
    df = df.join(enbarked_one_hot)


    return df

def create_model(optimizer='adam'):
    model = Sequential()
    model.add(Dense(16, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# pre-selected paramters
best_epochs = 200
best_batch_size = 5
best_init = 'glorotuniform'
best_optimizer = 'rmsprop'
verbose = 0

# Load data set
train = pd.read_csv("data/train.csv", index_col='PassengerId')
test = pd.read_csv("data/test.csv", index_col='PassengerId')

X = prep_data(train)
X = X.drop(['Survived'], axis=1).values.astype(float)

X_test = prep_data(test)
X_test = X_test.values.astype(float)

y = train[['Survived']]

scale = StandardScaler()
X = scale.fit_transform(X)
X_test = scale.fit_transform(X_test)

model = KerasClassifier(build_fn=create_model,
                             optimizer=best_optimizer,
                             # init=best_init,
                             epochs=best_epochs,
                             batch_size=best_batch_size,
                             verbose=verbose)

model.fit(X, y)

# Check predictions
pred = model.predict(X)
diff = pred - y


v = (diff == 0).astype(int).sum()/len(diff)
print(v.values)

# Predict with test data
predictions = model.predict(X_test)

# Format for submission
submission = pd.DataFrame({
    'PassengerId': test.index,
    'Survived': predictions[:,0],
})


submission.sort_values('PassengerId', inplace=True)
submission.to_csv("run1.csv", index=False)

