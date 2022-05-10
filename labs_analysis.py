'''
Script to prepare the dataset and conduct the analysis.
Three analysis conducted:
    - Reconstruction error in test set using learned variables of train set
    - Masked imputation error in full dataset following masking out 20% outcome variable
    - Regression error using test set latent variables after latent model and regression model fitting in train set
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import datetime
from time import perf_counter as pc
import numpy as np
from sklearn.preprocessing import normalize
from nmf import nmf
from nmf.nmf import NMF, non_negative_factorization
get_ipython().run_line_magic('run', 'Supporting_functions.ipynb')
from sklearn import metrics
import random
from math import floor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_rows', 20)
import warnings
warnings.simplefilter(action='ignore')

# name the columns in the dataset
columns = ['MRN', 'measurement_id', 'concept_id','date', 'datetime',
           'concept_type_id', 'operator_concept_id', 'value', 'value_concept', 'unit_concept', 'range_low',
           'range_high', 'provider_id', 'visit_occurrence_id', 'source_concept_value', 'source_concept', 'units', 'source_value']

labs = pd.read_csv('HIDDEN_DIRECTORY', sep = '|', header = None, names = columns, parse_dates = ['date'])


# idenitfy how many samples exist
labs.MRN.nunique()


# extract the columns of interest
labs = labs.loc[:,['MRN', 'concept_id','date',
           'concept_type_id', 'value',  'unit_concept', 'range_low',
           'range_high', 'source_concept_value', 'units']]

labs.head()

#Unique lab tests
print(labs['concept_id'].nunique())

#Mean tests per patients
print(labs.groupby('MRN').count().mean()[0])

#Median number of tests per patients
print(labs.groupby('MRN').count().median()[0])

#unique lab tests per patient
print(labs.groupby(['MRN', 'concept_id']).size().reset_index(level = 'concept_id', drop = True).groupby('MRN').count().mean())

#Lab observation per lab tests per patient
print(labs.groupby(['MRN', 'concept_id']).size().reset_index(level = 'concept_id', drop = True).groupby('MRN').mean().mean())


#Cumalitve sum, sorted from most filled lab values
cumulative = pd.DataFrame((100*labs.groupby('concept_id').count().sort_values(by = 'MRN', ascending = False).cumsum()/ labs.groupby('concept_id').count().sum()).iloc[:,0])
cumulative['source_concept'] = [labs['source_concept_value'][labs['concept_id'] == concept].iloc[0] for concept in cumulative.index]

#most common lab values
cumulative.iloc[0:20]


#########################################################################################################################
# Create dataframe for factorization

def data_processing(labs, outcome_column):
    '''
    process data to deal with outliers in the target and only take data prior to the target
    Inputs: 
        labs - dataset with all lab values
        outcome_column - the target lab value e.g. glucose
    '''
    #Deal with outliers in glucose column i.e. those with >500 which is hihgly unlikely
    high = labs.loc[labs[(labs['concept_id'] == 3004501) & (labs['value'] > 500)].groupby('MRN').date.idxmax(), ['MRN', 'value']]
    #replace with score of 200 + random noise (this is higher end of normal values)
    for index in high.index:
        value = 200 + np.ceil(np.random.normal(0, 6))
        labs.at[index, 'value'] = value

    #extract the latest date for the outcome of interest so can filter for lab values at that level
    latest_date = labs.loc[labs[labs['concept_id'] == outcome_column].groupby('MRN').date.idxmax(), ['MRN', 'date']]
    latest_date_dict = latest_date.set_index('MRN')['date'].to_dict()

    #todays date for all MRNs with no outcome measure (these can still be used in factorization)
    for MRN in set(labs.MRN.unique()) - set(np.array(list(latest_date_dict.keys()))):
        latest_date_dict[MRN] = datetime.date.today()

    #filter data recorded before the outcome of interest
    labs = labs[labs.apply(lambda x: latest_date_dict[x.MRN] >= x.date, axis = 1)]
    
    return labs




def df_for_factorization(df):
    '''
    create a matrix for factorization using the lab values
    This is done by moving from narrow data format to wide data format
    Inputs: 
        df: lab values dataset
    '''
    #Take the latest recording only
    df = df.loc[df.groupby(['MRN', 'concept_id']).date.idxmax(), ['MRN', 'concept_id', 'value']]
    #Make each row a unique MRN
    df = df.pivot_table(index = 'MRN', columns = 'concept_id', values = 'value')
    return df

# OPTIONAL AND CURRENTLY SET TO SELECT ALL COLUMNS
def filter_sparsity(df, sparse_limit):
    '''
    select only columns with at least X% filled
    Inputs:
        df: matrix of lab values
        sparse_limit: limit of how sparse a column can be before being removed
    '''
    return df.iloc[:,((df.isna().sum(axis = 0) / df.shape[0]) < sparse_limit).values]



def train_test_split(df, train_size):
    rows = np.arange(df.shape[0])
    rows = np.random.RandomState(seed=42).permutation(rows)
    train_rows = rows[:round(len(rows)*train_size)]
    test_rows = rows[round(len(rows)*train_size):]
    df_train = df.iloc[train_rows]
    df_test = df.iloc[test_rows]
    return abs(df_train), abs(df_test)





def data_scaler(df_filter, df_train, df_test):
    '''
    Scale the data using min-max (must be 0-1 normalized for NMF)
    Return also the scaler to reverse the transformation after prediction is made
    Inputs:
        datasets for training and testing
    '''
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df_filter)
    df_scaled = pd.DataFrame(df_scaled, columns = df_filter.columns, index = df_filter.index)
    
    scaler = MinMaxScaler()
    df_train_norm = scaler.fit_transform(df_train)
    df_train_norm = pd.DataFrame(df_train_norm, columns = df_train.columns, index = df_train.index)

    df_test_norm = scaler.transform(df_test)
    return scaler, df_scaled, df_train_norm, df_test_norm


#prepare data for analysis
def data_for_analysis(labs, outcome_column, sparse_limit, train_size):
    '''
    Aggregate function caller
    Inputs:
        Labs - labs dataset in narrow format
        outcome_column - column of interest
        sparse_limit - what level of sparsity to filter column with
        train_size - how large the train dataset should be
    '''
    labs = data_processing(labs, outcome_column)
    df = df_for_factorization(labs)
    df_filter = filter_sparsity(df, 1)
    df_filter = df_filter.drop(labels = [0], axis = 1)
    df_train, df_test = train_test_split(df_filter,0.8)
    scaler, df_scaled, df_train_norm, df_test_norm = data_scaler(df_filter, df_train, df_test)
    return scaler, df_filter, df_scaled, df_train, df_train_norm, df_test, df_test_norm 

#################################################################################################################################
## Running NMF

#data prep using glucose column
scaler, df_filter, df_scaled, df_train, df_train_norm, df_test, df_test_norm = data_for_analysis(labs, 3004501, 1, 0.8)


## calculate the reconstruction error

#reconstruction error using the full scaled dataset
reconstruction_error = {}
ranks = [1, 5, 10, 20, 40, 60, 80, 100, 200, 300, 400, 600]
for j in ranks:
    model = NMF(n_components = j, init = 'random', solver = 'mu', random_state = 0, verbose = True)
    W = model.fit_transform(df_scaled)
    reconstruction_error[j]  = model.reconstruction_err_
    print('{}: {}'.format(j, reconstruction_error[j]))

#save the errors
reconstruction_error = pd.DataFrame.from_dict(reconstruction_error, orient = 'index')
reconstruction_error.sort_index(inplace=True)
reconstruction_error.to_csv("../Output/reconstruction_error.csv", sep='|',index=True, header = False)

#load the saved results
reconstruction_error = pd.read_csv('../Output/reconstruction_error.csv', sep='|', names= ['rank', 'error'], index_col = 'rank')

#plot the errors test
plt.style.use('seaborn-whitegrid')
np.set_printoptions(suppress=True)
plt.plot(reconstruction_error.index, reconstruction_error.error)
plt.xlabel('Rank')
plt.ylabel('Reconstruction error (RMSE)')
plt.title('NMF reconstruction error')
plt.axhspan(10, 15, facecolor='0.1', alpha=0.2, label = 'Flattening of reconstruction error improvement')
plt.legend()
plt.show()


###### HELPER FUNCTIONS FOR THE ESTIMATION TASKS ###############
def extract_mrns(df, col_num):
    '''
       Extracts MRNs for patients with outcome data to be used to calculate the error in the test set. This is needed as some individuals in the dataset dont have the outcome
    '''
    df[df == 0] = np.nan
    glucose_col = df.columns.get_loc(col_num)
    gluc_mrns = list(df.loc[:,col_num].index[df.loc[:,col_num].notna()])
    mrns = []
    for mrn in gluc_mrns:
        mrns.append(df.loc[:,col_num].index.get_loc(mrn))
    return mrns, glucose_col

def prediction_error(df_train_norm, df_test_norm, df_test, ranks, mrns, target_col, error, scaler):
    '''
    predicting test set error in the outcome of interest
    This works by fitting a factor model on the train set and then using the learned factors to reconstruct the test dataset
    '''
    for j in ranks:
        ##fit the factor model
        model = NMF(n_components = j, init = 'random', solver = 'mu', random_state = 0, verbose = True).fit(df_train_norm)
        ##use learned variables on the test set
        W_test= model.transform(df_test_norm)
        H= model.components_
        reconstructed = np.dot(W_test,H)
        ##reverse the normalisation to get actual lab value of interest estimate (done to get clinically meaningful interpretation of the estimate)
        reconstructed_reversed = scaler.inverse_transform(reconstructed)
        ##calculate the RMSE from the true dataset
        error[j]  = np.sqrt(np.mean((df_test.iloc[mrns,target_col] - reconstructed_reversed[mrns, target_col])**2))
        print('{}: {}'.format(j, error[j]))
    return error

def prediction_error_masked(df, df_norm, ranks, mrns, target_col, error, masked_percent, scaler):
    '''
    predicting the value in the masked dataset
    This works by masking certain data points in the outcome and calculating the error at the reconstructed value to the true value
    '''
    random.seed(1)
    #mask % of the data
    nums = np.sort(random.sample(mrns, floor(len(mrns)*masked_percent)))
    df_norm.iloc[nums,target_col] = np.nan
    
    #predicting measurements for masked data
    for j in ranks:
        model = NMF(n_components = j, init = 'random', solver = 'mu', random_state = 0, verbose = True)
        W= model.fit_transform(df_norm)
        H= model.components_
        reconstructed = np.dot(W,H)
        ##reverse the normalisation to get actual lab value of interest estimate (done to get clinically meaningful interpretation of the estimate)
        reconstructed_reversed = scaler.inverse_transform(reconstructed)
        ##predict error in the masked dataset
        error[j]  = np.sum(abs(df.iloc[nums,target_col] - reconstructed_reversed[nums, target_col]))/df.shape[0]
        print('{}: {}'.format(j, error[j]))
    return error

def regression_prediction(df_train, df_train_norm, df_test, df_test_norm, ranks, mrns_train, mrns_test, target_col, error):
    '''
    Use the latent variables as features in a regression task to predict the outcome
    '''
    #Drop glucose and fit NMF to patients with glucose measurement
    df_train_norm_2 = df_train_norm.drop(df_train_norm.columns[target_col], axis=1)
    df_train_norm_2 = df_train_norm[mrns_train]
    df_test_norm_2 = df_test_norm[mrns_test]
    outcome_train = df_train.iloc[mrns_train, target_col]
    outcome_test = df_test.iloc[mrns_test,target_col]
    
    #predicting measurements
    for j in ranks:
        #fit latent variables in the train set
        model = NMF(n_components = j, init = 'random', solver = 'mu', random_state = 0, verbose = True).fit(df_train_norm_2)
        #use this to fit latent variables in the test set
        W_train = model.transform(df_train_norm_2)
        W_test= model.transform(df_test_norm_2)
        H= model.components_
        #train a LR model o the train latent variables
        LR = LinearRegression(fit_intercept=True, normalize=False, copy_X=True).fit(W_train, outcome_train)
        #predict on the test latent variables
        prediction = np.array(LR.predict(W_test))
        error[j]  = np.sqrt(np.mean((prediction - np.array(outcome_test))**2))
        print(error)
    return error

######################### ESTIMATION TASKS USING GLUCOSE AS OUTCOME ######################### 
#Predict test set glucose
glucose_error_test = {}
ranks = [1, 5, 10, 20, 40, 60, 80, 100, 200, 300, 400, 600]
glucose_col_num = 3004501
mrns, target_col = extract_mrns(df_test, glucose_col_num)
glucose_error_test =  prediction_error(df_train_norm, df_test_norm, df_test, ranks, mrns, target_col, glucose_error_test, scaler)


#Predict glucose masked
glucose_error_masked = {}
ranks = [1, 5, 10, 20, 40, 60, 80, 100, 200, 300, 400, 600]
glucose_col_num = 3004501
mrns, target_col = extract_mrns(df_filter, glucose_col_num)
glucose_error_masked = prediction_error_masked(df_filter, df_scaled, ranks, mrns, target_col, glucose_error_masked, 0.2, scaler)


#Predict glucose regression
glucose_error_regression = {}
ranks = [1, 5, 10, 20, 40, 60, 80, 100, 200, 300, 400, 600]
glucose_col_num = 3004501
mrns_train, target_col = extract_mrns(df_train, glucose_col_num)
mrns_test, target_col = extract_mrns(df_test, glucose_col_num)
regression_prediction(df_train, df_train_norm, df_test, df_test_norm, ranks, mrns_train, mrns_test, target_col, glucose_error_regression)


#Save the errors into file
glucose_errors_test = pd.DataFrame.from_dict(glucose_error_test, orient = 'index')
glucose_errors_test.sort_index(inplace=True)
glucose_errors_test.to_csv("../Output/glucose_error_test1.csv", sep='|',index=True, header = False)

glucose_errors_masked = pd.DataFrame.from_dict(glucose_error_masked, orient = 'index')
glucose_errors_masked.sort_index(inplace=True)
glucose_errors_masked.to_csv("../Output/glucose_error_mask1.csv", sep='|',index=True, header = False)

glucose_errors_regression = pd.DataFrame.from_dict(glucose_error_regression, orient = 'index')
glucose_errors_regression.sort_index(inplace=True)
glucose_errors_regression.to_csv("../Output/glucose_error_regression1.csv", sep='|',index=True, header = False)


#load the saved results
glucose_errors_test = pd.read_csv('../Output/glucose_error_test1.csv', sep='|', names= ['rank', 'error'], index_col = 'rank')
glucose_errors_masked = pd.read_csv('../Output/glucose_error_mask1.csv', sep='|', names= ['rank', 'error'], index_col = 'rank')
glucose_errors_regression = pd.read_csv('../Output/glucose_error_regression1.csv', sep='|', names= ['rank', 'error'], index_col = 'rank')


#plot the errors test
np.set_printoptions(suppress=True)
plt.plot(glucose_errors_test.index, np.log(glucose_errors_test.error))
plt.xlabel('Rank')
plt.ylabel('Average log reconstruction error (mg/dL)')
plt.title('NMF blood glucose reconstruction error test')
plt.axhline(y=np.log(10), color='slategrey', linestyle='--', label = 'Error threshold: log(10)mg/dL')
plt.ylim((0.8,4))
plt.legend()
plt.show()


#plot the errors imputation
np.set_printoptions(suppress=True)
plt.plot(glucose_errors_masked.index, np.log(glucose_errors_masked.error))
plt.xlabel('Rank')
plt.ylabel('Average log imputation error (mg/dL)')
plt.title('NMF blood glucose imputation error masked')
plt.axhline(y=np.log(10), color='slategrey', linestyle='--', label = 'Error threshold: log(10)mg/dL')
plt.ylim((0.8,4))
plt.legend()
plt.show()


#plot the errors regression
np.set_printoptions(suppress=True)
plt.plot(glucose_errors_regression.index, np.log(glucose_errors_regression.error))
plt.xlabel('Rank')
plt.ylabel('Average log prediction error (mg/dL)')
plt.title('NMF blood glucose RMSE regression')
plt.axhline(y=np.log(10), color='slategrey', linestyle='--', label = 'Prediction error threshold: log(10)mg/dL')
plt.ylim((0.8,4))
plt.legend()
plt.show()


######################### ESTIMATION TASKS USING HBA1C AS OUTCOME ######################### 
#Predict test set hba1c
hba1c_error_test = {}
ranks = [1, 5, 10, 20, 40, 60, 80, 100, 200, 300, 400, 600]
hba1c_col_num = 3004410
mrns, target_col = extract_mrns(df_test, hba1c_col_num)
hba1c_error_test = prediction_error(df_train_norm, df_test_norm, df_test, ranks, mrns, target_col, hba1c_error_test, scaler)

#Predict hba1c masked
hba1c_error_masked = {}
ranks = [1, 5, 10, 20, 40, 60, 80, 100, 200, 300, 400, 600]
hba1c_col_num = 3004410
mrns, target_col = extract_mrns(df_filter, hba1c_col_num)
hba1c_error_masked = prediction_error_masked(df_filter, df_scaled, ranks, mrns, target_col, hba1c_error_masked, 0.2, scaler)

#Predict hba1c regression
hba1c_error_regression = {}
ranks = [1, 5, 10, 20, 40, 60, 80, 100, 200, 300, 400, 600]
hba1c_col_num = 3004410
mrns_train, target_col = extract_mrns(df_train, hba1c_col_num)
mrns_test, target_col = extract_mrns(df_test, hba1c_col_num)
regression_prediction(df_train, df_train_norm, df_test, df_test_norm, ranks, mrns_train, mrns_test, target_col, hba1c_error_regression)

#Save the errors into file
hba1c_errors_test = pd.DataFrame.from_dict(hba1c_error_test, orient = 'index', )
hba1c_errors_test.sort_index(inplace=True)
hba1c_errors_test.to_csv("../Output/hba1c_error_test1.csv", sep='|',index=True, header = False)

hba1c_errors_masked = pd.DataFrame.from_dict(hba1c_error_masked, orient = 'index')
hba1c_errors_masked.sort_index(inplace=True)
hba1c_errors_masked.to_csv("../Output/hba1c_error_mask1.csv", sep='|',index=True, header = False)

hba1c_errors_regression = pd.DataFrame.from_dict(hba1c_error_regression, orient = 'index', )
hba1c_errors_regression.sort_index(inplace=True)
hba1c_errors_regression.to_csv("../Output/hba1c_error_regression1.csv", sep='|',index=True, header = False)

#load the saved results
hba1c_errors_test = pd.read_csv('../Output/hba1c_error_test1.csv', sep='|', names= ['rank', 'error'], index_col = 'rank')
hba1c_errors_masked = pd.read_csv('../Output/hba1c_error_mask1.csv', sep='|', names= ['rank', 'error'], index_col = 'rank')
hba1c_errors_regression = pd.read_csv('../Output/hba1c_error_regression1.csv', sep='|', names= ['rank', 'error'], index_col = 'rank')

#plot the errors
np.set_printoptions(suppress=True)
plt.plot(hba1c_errors_test.index, np.log(hba1c_errors_test.error))
plt.xlabel('Rank')
plt.ylabel('Average log reconstruction error (%)')
plt.title('NMF HbA1c reconstruction error by rank test Hba1c')
plt.axhline(y=np.log(0.4), color='slategrey', linestyle='--', label = 'Error threshold: log(0.4)%')
plt.ylim((-2.5,0.5))
plt.legend()
plt.show()

#plot the errors
np.set_printoptions(suppress=True)
plt.plot(hba1c_errors_masked.index, np.log(hba1c_errors_masked.error))
plt.xlabel('Rank')
plt.ylabel('Average log imputation error (%)')
plt.title('NMF HbA1c imputation error by rank, masked Hba1c')
plt.axhline(y=np.log(0.4), color='slategrey', linestyle='--', label = 'Error threshold: log(0.4)%')
plt.ylim((-2.5,0.5))
plt.legend()
plt.show()

#plot the errors
np.set_printoptions(suppress=True)
plt.plot(hba1c_errors_regression.index, np.log(hba1c_errors_regression.error))
plt.xlabel('Rank')
plt.ylabel('Average log prediction error (%)')
plt.title('NMF HbA1c RMSE error')
plt.axhline(y=np.log(0.4), color='slategrey', linestyle='--', label = 'Prediction error threshold: log(0.4)%')
plt.ylim((-2.5,0.5))
plt.legend()
plt.show()

