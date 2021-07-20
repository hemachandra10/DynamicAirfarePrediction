import numpy as np
import pandas as pd
from sklearn import datasets
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import csv

'''
Assign a number to each city. 'Mumbai' : 1; 'Lucknow':2; 'Kolkata':3; 'Chennai':4; 'Delhi':5; 'Patna':6 'Hyderabad':7
Class = 1 implies "business"
gender = 1 implies male and gender = 0 implies female 
is_doctor = 1 implies the person is a doctorate
'''

def pre_processing(train_df):
    train_df['Name'] = train_df['Name'].astype('|S')  # converts "object" to "string"
    train_df['From'] = train_df['From'].astype('|S')
    train_df['To'] = train_df['To'].astype('|S')
    train_df_rows = train_df.shape[0]
    train_df_cols = train_df.shape[1]
    # parse each row of the dataset and populate the pre-processed dataset accordingly
    ages = []
    prior_booking_days = []  # no of days between booking date and flight date
    gender = []
    is_doctor = []
    Class = []
    From = []
    To = []
    Flight_date_months = []
    Flight_time = []
    for i in range(train_df_rows):
        # first column is name
        ages.append(pd.to_datetime('2016-12-12').year - pd.to_datetime(train_df['Date of Birth'][i]).year)
        prior_booking_days.append(
            (pd.to_datetime(train_df['Flight Date'][i]) - pd.to_datetime(train_df['Booking Date'][i])).days)
        var = train_df['Name'][i].split()
        if (var[1][0:1].decode("utf-8")) == 'M':
            gender.append(1)
        else:
            gender.append(0)
        if (var[0][0:2].decode("utf-8") == 'Dr'):
            is_doctor.append(1)
        else:
            is_doctor.append(0)
        if (train_df['Class'][i] == "Business"):
            Class.append(1)
        else:
            Class.append(0)
        From.append(cities_dict[train_df['From'][i].decode("utf-8")])
        To.append(cities_dict[train_df['To'][i].decode("utf-8")])
        Flight_date_months.append(pd.to_datetime(train_df['Flight Date'][i]).month)
        Flight_time.append(pd.to_datetime(train_df['Flight Time'][i]).hour)

    # add "age" column to the data drame and drop the column "Date of Birth"
    ages_val = pd.Series(ages)
    train_df['age'] = ages_val.values
    train_df = train_df.drop('Date of Birth', axis=1)

    # add the column, "prior_booking_days"
    prior_booking_days_val = pd.Series(prior_booking_days)
    train_df['prior_booking_days'] = prior_booking_days_val.values

    # add the column, "gender" and "is_doctor"
    gender_val = pd.Series(gender)
    train_df['gender'] = gender_val.values
    is_doctor_val = pd.Series(is_doctor)
    train_df['is_doctor'] = is_doctor_val.values

    # drop the column "Name"
    train_df = train_df.drop('Name', axis=1)

    # drop the column "Class" and the list, "Class", we have populated
    train_df = train_df.drop('Class', axis=1)
    Class_val = pd.Series(Class)
    train_df['Class'] = Class_val.values

    # drop the columns, "From" and "To" and add the lists "From" and "To"
    train_df = train_df.drop('From', axis=1)
    train_df = train_df.drop('To', axis=1)
    From_val = pd.Series(From)
    train_df['From'] = From_val.values
    To_val = pd.Series(To)
    train_df['To'] = To_val.values

    # drop the column, "Booking Date"
    train_df = train_df.drop('Booking Date', axis=1)

    # drop the column, "Flight Date" and add the list, "Flight_date_months"
    train_df = train_df.drop('Flight Date', axis=1)
    Flight_date_months_val = pd.Series(Flight_date_months)
    train_df['Flight Date Month'] = Flight_date_months_val.values

    # drop the column, "Flight Time" and add the list, "Flight_time"
    train_df = train_df.drop('Flight Time', axis=1)
    Flight_time_val = pd.Series(Flight_time)
    train_df['Flight Time'] = Flight_time_val.values

    return train_df

    # print(train_df.columns.values)
    # print(train_df['is_doctor'])

def find_correlation():
    # After the data is preprocessed, find the columns that are highly correlated to the target column, "fare"
    # We have a total of 11 columns and we will find out the 7 columns that are highly correlated with the column "fare"
    correlation = train_df.corr(method='pearson')
    columns = correlation.nlargest(10, 'Fare').index
    print(correlation)
    print(columns)

def normalize_columns(train_df):
    # Normalization: y = (x - min) / (max - min) -----> transforms the values to 0-1
    # normalixe column, "age"
    train_df_rows = train_df.shape[0]
    train_df_cols = train_df.shape[1]
    cols = ['age', 'prior_booking_days', 'Flight Date Month', 'Flight Time']
    for name in cols:
        min_ele = train_df[name].min()
        max_ele = train_df[name].max()
        new_values = []
        for i in range(train_df_rows):
            new_values.append(((train_df[name][i] - min_ele) / (max_ele - min_ele)))
        train_df = train_df.drop(name, axis=1)
        new_values_val = pd.Series(new_values)
        train_df[name] = new_values_val.values
    return train_df

def find_best_regression_model():
    global x_train, y_train, x_test, y_test
    x = train_df
    y = x['Fare'].values
    x = x.drop('Fare', axis=1).values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

    pipelines = []
    pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()), ('LR', LinearRegression())])))
    pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()), ('LASSO', Lasso())])))
    pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()), ('EN', ElasticNet())])))
    pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()), ('KNN', KNeighborsRegressor())])))
    pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()), ('CART', DecisionTreeRegressor())])))
    pipelines.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()), ('GBM', GradientBoostingRegressor())])))

    results = []
    names = []
    for name, model in pipelines:
        kfold = KFold(n_splits=10, random_state=21)
        cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

def find_estimators_count():
    global x_train,y_train
    scaler = StandardScaler().fit(x_train)
    rescaledX = scaler.transform(x_train)
    param_grid = dict(n_estimators=np.array([50, 100, 200, 300, 400]))
    model = GradientBoostingRegressor(random_state=21)
    kfold = KFold(n_splits=10, random_state=21)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=kfold)
    grid_result = grid.fit(rescaledX, y_train)

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

def GBM():
    global x_train, y_train, x_test, y_test, predictions, predictions_test_df
    scaler = StandardScaler().fit(x_train)
    rescaled_X_train = scaler.transform(x_train)
    model = GradientBoostingRegressor(random_state=21, n_estimators=400)
    model.fit(rescaled_X_train, y_train)

    # transform the validation dataset
    rescaled_X_test = scaler.transform(x_test)
    predictions = model.predict(rescaled_X_test)
    print(mean_squared_error(y_test, predictions))

    # transform the validation dataset
    rescaled_X_test = scaler.transform(test_df)
    predictions_test_df = model.predict(rescaled_X_test)


if __name__ == "__main__":
    cities_dict = {'Mumbai': 1, 'Lucknow': 2, 'Kolkata': 3, 'Chennai': 4, 'Delhi': 5, 'Patna': 6, 'Hyderabad': 7}
    train_df = pd.read_csv("train.csv")
    # create a pre-processed dataset
    train_df['Fare'] = np.log(train_df['Fare'])
    train_df = pre_processing(train_df)
    train_df = normalize_columns(train_df)
    train_df.to_csv('preprocessed_train_data.csv')
    train_df = train_df.drop('Flight Date Month', axis=1)
    #train_df = train_df.drop('To', axis=1)

    test_df = pd.read_csv("test.csv")
    test_df = pre_processing(test_df)
    test_df = normalize_columns(test_df)
    test_df.to_csv('preprocessed_test_data.csv')
    test_df = test_df.drop('Flight Date Month', axis=1)

    find_correlation()
    find_best_regression_model()
    # GBM is found to outperform other regression models
    find_estimators_count()
    GBM()
    compare = pd.DataFrame({'Prediction': predictions, 'Test Data': y_test})
    print(compare.head(10))

    actual_y_test = np.exp(y_test)
    actual_predicted = np.exp(predictions)
    diff = abs(actual_y_test - actual_predicted)

    compare_actual = pd.DataFrame({'Test Data': actual_y_test, 'Predicted Price': actual_predicted, 'Difference': diff})
    compare_actual = compare_actual.astype(int)
    print(compare_actual.head(5))

    actual_predicted = np.exp(predictions_test_df)
    print(actual_predicted)
    print(type(actual_predicted))
    #np.savetxt('final_test.out', actual_predicted, delimiter=',')
    df = pd.DataFrame(actual_predicted)
    df.to_csv("final_result.csv")







