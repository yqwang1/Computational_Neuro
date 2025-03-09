# Created by: yeqing wang
# Date: 2025-03-04 11:40:54 
# Description: Reg_sample provided by Andrew M as part of theory club.
import numpy as np
def full_regression_CPD(data, cv, downsample, perm, mouse, session):
    # predictors:

    # reward
    # previous reward
    # left or right
    # BRL value
    # BRL pss
    # RW value

    ### partition data
    data_e = data[(data.mouse == mouse) & (data.session == session)]

    ### now make a predictor matrix to run a regression on a traces
    rew = data_e.reward.values - 0.5

    # choice
    choice = data_e.right_or_left.values - 0.5

    # past choice
    pc = data_e.right_or_left.shift(1).values - 0.5
    # remove NaN from first location due to shift
    pc[0] = 0
    prev_choice = pc

    # previous_reward
    pr = data_e.reward.shift(1).values - 0.5
    # remove NaN from first location due to shift
    pr[0] = 0
    prev_rew = pr

    # HMM prediction
    HMM_p = data_e.HMM_predict_RPE.values - 0.5

    # RW prediction
    RW_p = data_e.RW_RPE.values - 0.5

    # transition
    trans = data_e.transition.values / 2

    # rew * choice
    inter = (data_e.right_or_left.values - 0.5) * data_e.correct

    X_w_rew = np.array([rew, choice, pr, pc, RW_p, HMM_p]).T  # trans, HMM_p, RW_p,
    X = X_w_rew

    # Activity matrix [n_trials, n_timepoints] as a float (not an object!)
    yT = np.array([x.astype('float') for x in data_e.warped_m])

    # make correct dimesnsions

    yT1 = yT.reshape(len(data_e), -1)

    yT1 = scipy.signal.resample(yT1, int(yT1.shape[0] / downsample), axis=1)

    # run regression
    rew_reg = linear_model.RidgeCV(cv=cv, fit_intercept=True)
    rew_reg.fit(X, yT)

    score = rew_reg.score(X, yT)

    # do Coefficients of partial determination (remove one point at a time)
    cpd = CPD(X, yT, cv)

    # now do permutations for stats

    betas_perm = [[] for i in range(perm)]  # To store permuted predictor loadings for each session.
    cpd_perm = [[] for i in range(perm)]  # To store permuted cpd for each session.

    for i in range(perm):
        X = np.roll(X, np.random.randint(len(data)), axis=0)
        shuff_reg = linear_model.RidgeCV(cv=cv, fit_intercept=True)
        shuff_reg.fit(X, yT1)
        betas_perm[i].append(shuff_reg.coef_)  # Predictor loadings
        cpd_perm[i].append(CPD(X, yT1, cv))

    return rew_reg.coef_, rew_reg.intercept_, cpd, cpd_perm, X.shape[1], score


def CPD(X, y, cv):
    '''Evaluate coefficient of partial determination for each predictor in X'''
    ols = linear_model.RidgeCV(cv=cv, fit_intercept=True)
    ols.fit(X, y)
    sse = np.sum((ols.predict(X) - y) ** 2, axis=0)
    cpd = np.zeros([y.shape[1], X.shape[1]])
    for i in range(X.shape[1]):
        X_i = np.delete(X, i, axis=1)
        ols.fit(X_i, y)
        sse_X_i = np.sum((ols.predict(X_i) - y) ** 2, axis=0)
        cpd[:, i] = (sse_X_i - sse) / sse_X_i
    return cpd

