# Created by: yeqing wang
# Date: 2025-03-04 11:43:30 
# Description: Reg_sample2, simpler version of Reg_sample
import numpy as np

##### for each mouse make a reward matrix and regressor set
output = {}

for mice in data.mouse.unique().tolist():
    mouse = {}
    df = data[data.mouse == mice]
    for sessions in df.session.unique().tolist():

        R = []  #reward
        N = []  #not reward
        C = []  #choice
        B = []  #?what is that bias??
# yw this is bandit task , five trials, look for 20 ????

        # print(R)
        y = pd.Series()
        df1 = df[df.session == sessions]
        df1 = df1.reset_index(drop=True)
        for index, row in df1.iloc[5:].iterrows():

            # crerate matrices to fill
            reward_matrix = np.zeros(5)
            no_reward_matrix = np.zeros(5)
            choice_matrix = np.zeros(5)

            # take each of five trials backwards
            for x in np.arange(1, 6):

                # take the trial that is x steps backwards
                temp_trial = df1.iloc[index - x]
                if temp_trial['right_or_left'] == 1 and temp_trial['reward'] == 1:
                    reward_matrix[x - 1] = 1

                elif temp_trial['right_or_left'] == 0 and temp_trial['reward'] == 1:
                    reward_matrix[x - 1] = -1 # if they press the left reward, -1
            # this is a orthogal one from previous: y is what is the prob of choosing pardicular lever
                elif temp_trial['right_or_left'] == 1 and temp_trial['reward'] == 0:
                    no_reward_matrix[x - 1] = 1
                elif temp_trial['right_or_left'] == 0 and temp_trial['reward'] == 0:
                    no_reward_matrix[x - 1] = -1

                if temp_trial['right_or_left'] == 1:
                    choice_matrix[x - 1] = 1
                elif temp_trial['right_or_left'] == 0:
                    choice_matrix[x - 1] = -1

            R.append(reward_matrix)
            N.append(no_reward_matrix)
            C.append(choice_matrix)
            B.append([1])
            y_temp = df1['right_or_left'].iloc[5:]

        y = y.append(y_temp) # y for all the
        X = np.concatenate((R, N, C, B), axis=1)
        mouse[sessions] = [y, X]
    output[mice] = mouse

# yw: how you run regression from python
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

coefs_all_df = pd.DataFrame()

for mice in output:
    for sessions in output[mice]:
        if len(output[mice][sessions][0] > 100):
            param_range = np.arange(0, 1.1, 0.1)
            param_grid = [{'C': param_range, 'l1_ratio': param_range}] # c is penalty strength.
            # gs is here to find all the c and l1_ratior, is to find what is the best for r square. try every combination.
            gs = GridSearchCV(
                estimator=linear_model.LogisticRegression(penalty='elasticnet', solver='saga', fit_intercept=False),
                param_grid=param_grid, scoring=None, cv=5) # cv = 5, is to the cross validation.
            gs = gs.fit(output[mice][sessions][1].astype('int'), output[mice][sessions][0].astype('int'))

            reg = linear_model.LogisticRegression(C=gs.best_params_['C'], penalty='elasticnet', solver='saga',
                                                  fit_intercept=False, l1_ratio=gs.best_params_['l1_ratio'])
            reg.fit(output[mice][sessions][1].astype('int'), output[mice][sessions][0].astype('int')) # to what extend can i predict mouse bhv from past.

            score = gs.best_score_  # 5cv score from grid search with the best parameter

            temp_df = pd.DataFrame(
                {'mouse': mice, 'session': sessions, 'trials_back': [1, 2, 3, 4, 5], 'R': reg.coef_[0][:5],
                 'N': reg.coef_[0][5:10], 'C': reg.coef_[0][10:15],
                 'B': reg.coef_[0][15],
                 'best_C': gs.best_params_['C'], 'best_l1_ratio': gs.best_params_['l1_ratio'], 'CV_score': score})

            coefs_all_df = coefs_all_df.append(temp_df)
