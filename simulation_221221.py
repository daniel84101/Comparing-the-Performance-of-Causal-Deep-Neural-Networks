import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from causal_nets import causal_net_estimate
import pandas as pd

### First Exercise, random assignment

EDU_KEN = pd.read_csv('/Users/danielchiang/Dropbox/Rochester/ECON 544/kenya.csv', sep="\t")
# I am targeting the results from table 2 column 2, thus take out the variables in the regression model accordingly
# to be included in the analysis
KEN_working = EDU_KEN[['stdR_litscore', 'girl', 'percentile', 'agetest', 'etpteacher', 'tracking']]
KEN_working = KEN_working.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
# these lists are for storing the results
st_ate = []
st_CI = []
lt_ate = []
lt_CI = []

Y = KEN_working[['stdR_litscore']].to_numpy()
X = KEN_working[['girl', 'percentile', 'agetest', 'etpteacher']].to_numpy()
T = KEN_working[['tracking']].to_numpy()

# splitting training and validation dataset for the short-term effect
X_train, X_valid, T_train, T_valid, Y_train, Y_valid = train_test_split(X, T, Y, test_size=0.2, random_state=42)

spec = [[60],[100],[30,20],[30,10],[30,30],[100,30,20],[80,30,20]]

for i in spec:
    tau_pred, mu0_pred, prob_t_pred, psi_0, psi_1, history, history_ps = causal_net_estimate(
        [X_train, T_train, Y_train], [X_valid, T_valid, Y_valid], [X, T, Y], [60],
        dropout_rates=None, batch_size=None, alpha=0., r_par=0., optimizer='Adam', learning_rate=0.0009,
        max_epochs_without_change=30, max_nepochs=10000, seed=None, estimate_ps=False, verbose=True)
    
    # Calculate the average treatment effect
    ate = np.mean(psi_1-psi_0)

    # Calculate the 95% confidence interval for average treatment effect
    CI_lowerbound = ate - norm.ppf(0.975)*np.std(psi_1-psi_0)/np.sqrt(len(psi_0))
    CI_upperbound = ate + norm.ppf(0.975)*np.std(psi_1-psi_0)/np.sqrt(len(psi_0))
    CI = [CI_lowerbound, CI_upperbound]
    
    # Store the result into the sets
    st_ate.append(ate)
    st_CI.append(CI)

## now, I am targeting the penal B of table 2 column 2
KEN_working = EDU_KEN[['stdR_r2_totalscore', 'girl', 'percentile', 'agetest', 'etpteacher', 'tracking']]
KEN_working = KEN_working.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

Y = KEN_working[['stdR_r2_totalscore']].to_numpy()
X = KEN_working[['girl', 'percentile', 'agetest', 'etpteacher']].to_numpy()
T = KEN_working[['tracking']].to_numpy()

# splitting training and validation dataset for the short-term effect
X_train, X_valid, T_train, T_valid, Y_train, Y_valid = train_test_split(X, T, Y, test_size=0.2, random_state=42)

for i in spec:
    tau_pred, mu0_pred, prob_t_pred, psi_0, psi_1, history, history_ps = causal_net_estimate(
        [X_train, T_train, Y_train], [X_valid, T_valid, Y_valid], [X, T, Y], [60],
        dropout_rates=None, batch_size=None, alpha=0., r_par=0., optimizer='Adam', learning_rate=0.0009,
        max_epochs_without_change=30, max_nepochs=10000, seed=None, estimate_ps=False, verbose=True)
    
    # Calculate the average treatment effect
    ate = np.mean(psi_1-psi_0)

    # Calculate the 95% confidence interval for average treatment effect
    CI_lowerbound = ate - norm.ppf(0.975)*np.std(psi_1-psi_0)/np.sqrt(len(psi_0))
    CI_upperbound = ate + norm.ppf(0.975)*np.std(psi_1-psi_0)/np.sqrt(len(psi_0))
    CI = [CI_lowerbound, CI_upperbound]
    
    # Store the result into the sets
    lt_ate.append(ate)
    lt_CI.append(CI)

# Export the result to corresponding csv files
np.savetxt("state.csv", st_ate, delimiter =", ", fmt ='% s')
np.savetxt("stCI.csv", st_CI, delimiter =", ", fmt ='% s')
np.savetxt("ltate.csv", lt_ate, delimiter =", ", fmt ='% s')
np.savetxt("ltCI.csv", lt_CI, delimiter =", ", fmt ='% s')


### in the second exercise, I am going to move to the non-random treatment
DEM = pd.read_csv('/Users/danielchiang/Dropbox/Rochester/ECON 544/direct.csv', sep="\t", encoding= 'unicode_escape')

# first exercise focus on the first stage of a fuzzy RD
DEM_working = DEM[['Direct_democracy', 'min_1d', 'zeropop', 'zeropopsmall', 'logcap_poor1918', 'logcap_poor_external1918', 'logcap_poor_internal1918', 'no_org_citizen1917_share', 'nr_poor_relief1917', 'nr_old_on_poor_relief1917', 'nr_dir_child_poor_r1917', 'nr_child_on_poor_r1917', 'nr_fullpoor_1917', 'nr_poor_external1917', 'nr_poor_internal1917',  'poorhouse1917', 'poorslots1917', 'area1918', 'land_area1918', 'arable_land1918', 'realtaxbase1918', 'shareagri1917', 'population1918', 'elig_state1917', 'turnout_state1917', 'leftvote1917', 'dum1', 'dum2', 'dum3', 'dum4', 'dum5', 'dum6', 'dum7', 'dum8', 'dum9', 'dum10', 'dum11', 'dum12', 'dum13', 'dum14', 'dum15', 'dum16', 'dum17', 'dum18', 'dum19', 'dum20']]
DEM_working = DEM_working.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
fs_atm = []
fs_CI = []
ss_atm = []
ss_CI = []

Y = DEM_working[['Direct_democracy']].to_numpy()
T = DEM_working[['min_1d']].to_numpy()
X = DEM_working[['zeropop', 'zeropopsmall', 'logcap_poor1918', 'logcap_poor_external1918', 'logcap_poor_internal1918', 'no_org_citizen1917_share', 'nr_poor_relief1917', 'nr_old_on_poor_relief1917', 'nr_dir_child_poor_r1917', 'nr_child_on_poor_r1917', 'nr_fullpoor_1917', 'nr_poor_external1917', 'nr_poor_internal1917',  'poorhouse1917', 'poorslots1917', 'area1918', 'land_area1918', 'arable_land1918', 'realtaxbase1918', 'shareagri1917', 'population1918', 'elig_state1917', 'turnout_state1917', 'leftvote1917', 'dum1', 'dum2', 'dum3', 'dum4', 'dum5', 'dum6', 'dum7', 'dum8', 'dum9', 'dum10', 'dum11', 'dum12', 'dum13', 'dum14', 'dum15', 'dum16', 'dum17', 'dum18', 'dum19', 'dum20']].to_numpy()


# splitting training and validation dataset for the short-term effect
X_train, X_valid, T_train, T_valid, Y_train, Y_valid = train_test_split(X, T, Y, test_size=0.2, random_state=42, stratify=T)

for i in spec:
    tau_pred, mu0_pred, prob_t_pred, psi_0, psi_1, history, history_ps = causal_net_estimate(
        [X_train, T_train, Y_train], [X_valid, T_valid, Y_valid], [X, T, Y], [60],
        dropout_rates=None, batch_size=None, alpha=0., r_par=0., optimizer='Adam', learning_rate=0.0009,
        max_epochs_without_change=30, max_nepochs=10000, seed=None, estimate_ps=False, verbose=True)
    
    # Calculate the average treatment effect
    ate = np.mean(psi_1-psi_0)

    # Calculate the 95% confidence interval for average treatment effect
    CI_lowerbound = ate - norm.ppf(0.975)*np.std(psi_1-psi_0)/np.sqrt(len(psi_0))
    CI_upperbound = ate + norm.ppf(0.975)*np.std(psi_1-psi_0)/np.sqrt(len(psi_0))
    CI = [CI_lowerbound, CI_upperbound]
    
    # Store the result into the sets
    fs_atm.append(ate)
    fs_CI.append(CI)

## then move to the full result
DEM_working = DEM[['Direct_democracy', 'min_1d', 'zeropop', 'zeropopsmall', 'logcap_poor1918', 'logcap_poor_external1918', 'logcap_poor_internal1918', 'no_org_citizen1917_share', 'nr_poor_relief1917', 'nr_old_on_poor_relief1917', 'nr_dir_child_poor_r1917', 'nr_child_on_poor_r1917', 'nr_fullpoor_1917', 'nr_poor_external1917', 'nr_poor_internal1917', 'poorhouse1917', 'poorslots1917', 'area1918', 'land_area1918', 'arable_land1918', 'realtaxbase1918', 'shareagri1917', 'population1918', 'elig_state1917', 'turnout_state1917', 'leftvote1917', 'dum1', 'dum2', 'dum3', 'dum4', 'dum5', 'dum6', 'dum7', 'dum8', 'dum9', 'dum10', 'dum11', 'dum12', 'dum13', 'dum14', 'dum15', 'dum16', 'dum17', 'dum18', 'dum19', 'dum20', 'logcap_poor']]
DEM_working = DEM_working.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)


Y = DEM_working[['logcap_poor']].to_numpy()
T = DEM_working[['Direct_democracy']].to_numpy()
X = DEM_working[['min_1d', 'zeropop', 'zeropopsmall', 'logcap_poor1918', 'logcap_poor_external1918', 'logcap_poor_internal1918', 'no_org_citizen1917_share', 'nr_poor_relief1917', 'nr_old_on_poor_relief1917', 'nr_dir_child_poor_r1917', 'nr_child_on_poor_r1917', 'nr_fullpoor_1917', 'nr_poor_external1917', 'nr_poor_internal1917',  'poorhouse1917', 'poorslots1917', 'area1918', 'land_area1918', 'arable_land1918', 'realtaxbase1918', 'shareagri1917', 'population1918', 'elig_state1917', 'turnout_state1917', 'leftvote1917', 'dum1', 'dum2', 'dum3', 'dum4', 'dum5', 'dum6', 'dum7', 'dum8', 'dum9', 'dum10', 'dum11', 'dum12', 'dum13', 'dum14', 'dum15', 'dum16', 'dum17', 'dum18', 'dum19', 'dum20']].to_numpy()


# splitting training and validation dataset for the short-term effect
X_train, X_valid, T_train, T_valid, Y_train, Y_valid = train_test_split(X, T, Y, test_size=0.2, random_state=42, stratify=T)
for i in spec:
    tau_pred, mu0_pred, prob_t_pred, psi_0, psi_1, history, history_ps = causal_net_estimate(
        [X_train, T_train, Y_train], [X_valid, T_valid, Y_valid], [X, T, Y], [60],
        dropout_rates=None, batch_size=None, alpha=0., r_par=0., optimizer='Adam', learning_rate=0.0009,
        max_epochs_without_change=30, max_nepochs=10000, seed=None, estimate_ps=False, verbose=True)
    
    # Calculate the average treatment effect
    ate = np.mean(psi_1-psi_0)

    # Calculate the 95% confidence interval for average treatment effect
    CI_lowerbound = ate - norm.ppf(0.975)*np.std(psi_1-psi_0)/np.sqrt(len(psi_0))
    CI_upperbound = ate + norm.ppf(0.975)*np.std(psi_1-psi_0)/np.sqrt(len(psi_0))
    CI = [CI_lowerbound, CI_upperbound]
    
    # Store the result into the sets
    ss_atm.append(ate)
    ss_CI.append(CI)

# Export the result to corresponding csv files
np.savetxt("fsate.csv", fs_atm, delimiter =", ", fmt ='% s')
np.savetxt("fsCI.csv", fs_CI, delimiter =", ", fmt ='% s')
np.savetxt("ssate.csv", ss_atm, delimiter =", ", fmt ='% s')
np.savetxt("ssCI.csv", ss_CI, delimiter =", ", fmt ='% s')

## follow on a smaller set of controls
DEM_working = DEM[['Direct_democracy', 'min_1d', 'zeropop', 'zeropopsmall', 'logcap_poor1918', 'logcap_poor_external1918', 'logcap_poor_internal1918', 'no_org_citizen1917_share', 'nr_poor_relief1917', 'nr_old_on_poor_relief1917', 'nr_dir_child_poor_r1917', 'nr_child_on_poor_r1917', 'nr_fullpoor_1917', 'nr_poor_external1917', 'nr_poor_internal1917', 'poorhouse1917', 'poorslots1917', 'area1918', 'land_area1918', 'arable_land1918', 'realtaxbase1918', 'shareagri1917', 'population1918', 'elig_state1917', 'turnout_state1917', 'leftvote1917', 'dum1', 'dum2', 'dum3', 'dum4', 'dum5', 'dum6', 'dum7', 'dum8', 'dum9', 'dum10', 'dum11', 'dum12', 'dum13', 'dum14', 'dum15', 'dum16', 'dum17', 'dum18', 'dum19', 'dum20', 'logcap_poor']]
DEM_working = DEM_working.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)


Y = DEM_working[['logcap_poor']].to_numpy()
T = DEM_working[['Direct_democracy']].to_numpy()
X = DEM_working[['min_1d', 'zeropop', 'zeropopsmall']].to_numpy()

X_train, X_valid, T_train, T_valid, Y_train, Y_valid = train_test_split(X, T, Y, test_size=0.2, random_state=0, stratify=T)
tau_pred, mu0_pred, prob_t_pred, psi_0, psi_1, history, history_ps = causal_net_estimate(
    [X_train, T_train, Y_train], [X_valid, T_valid, Y_valid], [X, T, Y], [30,30],
    dropout_rates=None, batch_size=None, alpha=0., r_par=0., optimizer='Adam', learning_rate=0.0009,
    max_epochs_without_change=30, max_nepochs=10000, seed=None, estimate_ps=False, verbose=True)

ate = np.mean(psi_1-psi_0)

CI_lowerbound = ate - norm.ppf(0.975)*np.std(psi_1-psi_0)/np.sqrt(len(psi_0))
CI_upperbound = ate + norm.ppf(0.975)*np.std(psi_1-psi_0)/np.sqrt(len(psi_0))

print(ate)
print(CI_lowerbound)
print(CI_upperbound)

## follow on a more local sample
DEM_working = DEM_working[DEM_working['zeropop']<=20]
DEM_working = DEM_working[DEM_working['zeropop']>=-20]

Y = DEM_working[['logcap_poor']].to_numpy()
T = DEM_working[['Direct_democracy']].to_numpy()
X = DEM_working[['min_1d', 'zeropop', 'zeropopsmall']].to_numpy()

X_train, X_valid, T_train, T_valid, Y_train, Y_valid = train_test_split(X, T, Y, test_size=0.2, random_state=0,                                                                       stratify=T)
tau_pred, mu0_pred, prob_t_pred, psi_0, psi_1, history, history_ps = causal_net_estimate(
    [X_train, T_train, Y_train], [X_valid, T_valid, Y_valid], [X, T, Y], [30,30],
    dropout_rates=None, batch_size=None, alpha=0., r_par=0., optimizer='Adam', learning_rate=0.0009,
    max_epochs_without_change=30, max_nepochs=10000, seed=None, estimate_ps=False, verbose=True)

ate = np.mean(psi_1-psi_0)

CI_lowerbound = ate - norm.ppf(0.975)*np.std(psi_1-psi_0)/np.sqrt(len(psi_0))
CI_upperbound = ate + norm.ppf(0.975)*np.std(psi_1-psi_0)/np.sqrt(len(psi_0))

print(ate)
print(CI_lowerbound)
print(CI_upperbound)

