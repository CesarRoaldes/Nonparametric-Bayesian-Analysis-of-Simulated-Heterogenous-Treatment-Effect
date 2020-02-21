import os
import time
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

dir_path = os.path.dirname(os.path.realpath(__file__))

sns.set_style("ticks")
font = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 22}
plt.rc('font', **font)


##############
# Require the part1.R script to be run in order to generate data
df = pd.read_csv(dir_path+'/../data/simulated_data.csv', sep=',')
df.drop(['inter'], axis=1, inplace=True)
df.rename({"X_9": "X_sex", "X_10": "X_new"}, axis=1, inplace=True)
pos_y = df.columns.get_loc("y_hete")


# Splitting treatment/control samples.
dft = df[df["t"] == 1]
dfc = df[df["t"] == 0]


Xt, yt = dft.iloc[:, 1:pos_y].values, dft.iloc[:, pos_y].values
Xc, yc = dfc.iloc[:, 1:pos_y].values, dfc.iloc[:, pos_y].values

##########
# Bayesian Forests of 1000 trees each were fit to the treatment
# and control group samples. No random variable subsetting :
# the variability in the resulting prediction surface is due
# entirely to posterior uncertainty about the DGP
N = 1000
bf_t = RandomForestRegressor(N,
                             min_samples_leaf=1000,
                             max_features=None,
                             bayesian_bootstrap=True,
                             bootstrap=False,
                             max_depth=10,
                             n_jobs=-1,
                             random_state=11)
bf_c = RandomForestRegressor(N,
                             min_samples_leaf=1000,
                             max_features=None,
                             bayesian_bootstrap=True,
                             bootstrap=False,
                             max_depth=10,
                             n_jobs=-1,
                             random_state=22)
bf_t.fit(Xt, yt)
bf_c.fit(Xc, yc)


pickle.dump(bf_t, open(dir_path+'/../pickle/bf_t.pkl', 'wb'))
bf_t = pickle.load(open(dir_path+'/../pickle/bf_t.pkl', 'rb'))

pickle.dump(bf_c, open(dir_path+'/../pickle/bf_c.pkl', 'wb'))
bf_c = pickle.load(open(dir_path+'/../pickle/bf_c.pkl', 'rb'))


##########
# Posterior sample over prediction rules for the conditional average
# treatment effects (18).
obs1 = df.sample(random_state=1)
obs2 = df.sample(random_state=2)
obs3 = df.sample(random_state=3)
obs4 = df.sample(random_state=4)

f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4,
                                       sharex=False, sharey=False,
                                       figsize=(20, 5))

# Bayesian Forest treatment effects : Each plot shows the posterior
# distribution for population CART treatment effect prediction at
# features x_i from a samped user

postDistrib_treatmentEffect = np.zeros(N)
for i, (cart_t, cart_c) in enumerate(zip(bf_t.estimators_, bf_c.estimators_)):
    y_t = cart_t.predict(obs1.drop(["t", "y_hete"], axis=1).values)
    y_c = cart_c.predict(obs1.drop(["t", "y_hete"], axis=1).values)
    postDistrib_treatmentEffect[i] = y_t - y_c
sns.kdeplot(postDistrib_treatmentEffect, ax=ax1, shade=True, color='dimgrey')
ax1.set_xlabel('predicted treatment effect')
ax1.set_ylabel('density')

postDistrib_treatmentEffect = np.zeros(N)
for i, (cart_t, cart_c) in enumerate(zip(bf_t.estimators_, bf_c.estimators_)):
    y_t = cart_t.predict(obs2.drop(["t", "y_hete"], axis=1).values)
    y_c = cart_c.predict(obs2.drop(["t", "y_hete"], axis=1).values)
    postDistrib_treatmentEffect[i] = y_t - y_c
sns.kdeplot(postDistrib_treatmentEffect, ax=ax2, shade=True, color='dimgrey')
ax2.set_xlabel('predicted treatment effect')
ax2.set_ylabel('density')

postDistrib_treatmentEffect = np.zeros(N)
for i, (cart_t, cart_c) in enumerate(zip(bf_t.estimators_, bf_c.estimators_)):
    y_t = cart_t.predict(obs3.drop(["t", "y_hete"], axis=1).values)
    y_c = cart_c.predict(obs3.drop(["t", "y_hete"], axis=1).values)
    postDistrib_treatmentEffect[i] = y_t - y_c
sns.kdeplot(postDistrib_treatmentEffect, ax=ax3, shade=True, color='dimgrey')
ax3.set_xlabel('predicted treatment effect')
ax3.set_ylabel('density')

postDistrib_treatmentEffect = np.zeros(N)
for i, (cart_t, cart_c) in enumerate(zip(bf_t.estimators_, bf_c.estimators_)):
    y_t = cart_t.predict(obs4.drop(["t", "y_hete"], axis=1).values)
    y_c = cart_c.predict(obs4.drop(["t", "y_hete"], axis=1).values)
    postDistrib_treatmentEffect[i] = y_t - y_c
sns.kdeplot(postDistrib_treatmentEffect, ax=ax4, shade=True, color='dimgrey')
ax4.set_xlabel('predicted treatment effect')
ax4.set_ylabel('density')

f.tight_layout(pad=2)
plt.suptitle('Posterior distribution for population CART treatment effect prediction')
sns.despine()
plt.savefig(dir_path+'/../graphs/postDistrib_treatmentEffect_nl.png')


############
# Average Treatment Effect conditional on variable j being in set X (19).
# j = ['X_new', 'X_sex', X_1]
np.random.seed(777)
theta = np.random.exponential(scale=1, size=df.shape[0])


def conditionalAverageTreatmentEffect(df_condi, theta_condi):
    """ Compute the Average Treatment Effect """
    conditional_AverageTreatmentEffect = np.zeros(N)
    for i, (cart_t, cart_c) in enumerate(zip(bf_t.estimators_, bf_c.estimators_)):
        y_t_j = cart_t.predict(df_condi.drop(["t", "y_hete"], axis=1).values)
        y_c_j = cart_c.predict(df_condi.drop(["t", "y_hete"], axis=1).values)
        ate_j = np.dot(theta_condi, y_t_j - y_c_j) / theta_condi.sum()
        conditional_AverageTreatmentEffect[i] = ate_j
    return conditional_AverageTreatmentEffect


df_X_sex_0 = df[df['X_sex'] == 0]
df_X_sex_1 = df[df['X_sex'] == 1]
theta_X_sex_0 = theta[df['X_sex'] == 0]
theta_X_sex_1 = theta[df['X_sex'] == 1]
te_X_sex_0 = conditionalAverageTreatmentEffect(df_X_sex_0, theta_X_sex_0)
te_X_sex_1 = conditionalAverageTreatmentEffect(df_X_sex_1, theta_X_sex_1)
ate_X_sex_0 = te_X_sex_0.mean()
ate_X_sex_1 = te_X_sex_1.mean()

df_X_new_0 = df[df['X_new'] == 0]
df_X_new_1 = df[df['X_new'] == 1]
theta_X_new_0 = theta[df['X_new'] == 0]
theta_X_new_1 = theta[df['X_new'] == 1]
te_X_new_0 = conditionalAverageTreatmentEffect(df_X_new_0, theta_X_new_0)
te_X_new_1 = conditionalAverageTreatmentEffect(df_X_new_1, theta_X_new_1)
ate_X_new_0 = te_X_new_0.mean()  # 1.14
ate_X_new_1 = te_X_new_1.mean()  # 1.12

np.quantile(df['X_1'], 0.2)
np.quantile(df['X_1'], 0.4)
np.quantile(df['X_1'], 0.6)
np.quantile(df['X_1'], 0.8)
np.quantile(df['X_1'], 1)

df_X_1_20 = df[df['X_1'] < np.quantile(df['X_1'], 0.2)]
df_X_1_40 = df[(df['X_1'] < np.quantile(df['X_1'], 0.4))
               & (df['X_1'] > np.quantile(df['X_1'], 0.2))]
df_X_1_60 = df[(df['X_1'] < np.quantile(df['X_1'], 0.6))
               & (df['X_1'] > np.quantile(df['X_1'], 0.4))]
df_X_1_80 = df[(df['X_1'] < np.quantile(df['X_1'], 0.8))
               & (df['X_1'] > np.quantile(df['X_1'], 0.6))]
df_X_1_100 = df[df['X_1'] > np.quantile(df['X_1'], 0.8)]

theta_X_1_20 = theta[df['X_1'] < np.quantile(df['X_1'], 0.2)]
theta_X_1_40 = theta[(df['X_1'] < np.quantile(df['X_1'], 0.4))
                     & (df['X_1'] > np.quantile(df['X_1'], 0.2))]
theta_X_1_60 = theta[(df['X_1'] < np.quantile(df['X_1'], 0.6))
                     & (df['X_1'] > np.quantile(df['X_1'], 0.4))]
theta_X_1_80 = theta[(df['X_1'] < np.quantile(df['X_1'], 0.8))
                     & (df['X_1'] > np.quantile(df['X_1'], 0.6))]
theta_X_1_100 = theta[df['X_1'] > np.quantile(df['X_1'], 0.8)]


te_X_1_20 = conditionalAverageTreatmentEffect(df_X_1_20, theta_X_1_20)
te_X_1_40 = conditionalAverageTreatmentEffect(df_X_1_40, theta_X_1_40)
te_X_1_60 = conditionalAverageTreatmentEffect(df_X_1_60, theta_X_1_60)
te_X_1_80 = conditionalAverageTreatmentEffect(df_X_1_80, theta_X_1_80)
te_X_1_100 = conditionalAverageTreatmentEffect(df_X_1_100, theta_X_1_100)

# Importation of OLS estimations from R script
OLS_est = pd.read_csv(dir_path+'/../data/OLS_est.csv', sep=',')


# Boxplots
f, (ax1, ax2, ax3) = plt.subplots(1, 3,
                                  sharex=False, sharey=False,
                                  figsize=(20, 10),
                                  gridspec_kw={'width_ratios': [1, 1, 2]})

sns.boxplot(data=np.hstack((te_X_sex_0.reshape(1000, 1),
                            te_X_sex_1.reshape(1000, 1))),
            ax=ax1, color='lightcoral', boxprops=dict(alpha=.8))
sns.boxplot(data=np.hstack((OLS_est['sex2'].values.reshape(1000, 1),
                            OLS_est['sex1'].values.reshape(1000, 1))),
            ax=ax1, color='lightsteelblue', boxprops=dict(alpha=.8))
ax1.set_xlabel('X_sex')
ax1.set_ylabel('Conditional Average Treatment Effect')
for i, artist in enumerate(ax1.artists):
    col = artist.get_facecolor()
    artist.set_edgecolor(col)
    for j in range(i*6, i*6+6):
        line = ax1.lines[j]
        line.set_color(col)
        line.set_mfc(col)
        line.set_mec(col)

sns.boxplot(data=np.hstack((te_X_new_0.reshape(1000, 1),
                            te_X_new_1.reshape(1000, 1))),
            ax=ax2, color='lightcoral', boxprops=dict(alpha=.8))
sns.boxplot(data=np.hstack((OLS_est['old'].values.reshape(1000, 1),
                            OLS_est['new'].values.reshape(1000, 1))),
            ax=ax2, color='lightsteelblue', boxprops=dict(alpha=.8))
ax2.set_xlabel('X_new')
ax2.set_ylabel('Conditional Average Treatment Effect')
for i, artist in enumerate(ax2.artists):
    col = artist.get_facecolor()
    artist.set_edgecolor(col)
    for j in range(i*6, i*6+6):
        line = ax2.lines[j]
        line.set_color(col)
        line.set_mfc(col)
        line.set_mec(col)

sns.boxplot(data=np.hstack((te_X_1_20.reshape(1000, 1),
                            te_X_1_40.reshape(1000, 1),
                            te_X_1_60.reshape(1000, 1),
                            te_X_1_80.reshape(1000, 1),
                            te_X_1_100.reshape(1000, 1))),
            ax=ax3, color='lightcoral', boxprops=dict(alpha=.8))
sns.boxplot(data=np.hstack((OLS_est['quant1'].values.reshape(1000, 1),
                            OLS_est['quant2'].values.reshape(1000, 1),
                            OLS_est['quant3'].values.reshape(1000, 1),
                            OLS_est['quant4'].values.reshape(1000, 1),
                            OLS_est['quant5'].values.reshape(1000, 1))),
            ax=ax3, color='lightsteelblue', boxprops=dict(alpha=.8))
ax3.set_xlabel('X_spend')
ax3.set_ylabel('Conditional Average Treatment Effect')
ax3.set_xticklabels(['[0-20]', '[20-40]', '[40-60]', '[60-80]', '[80-100]'])
for i, artist in enumerate(ax3.artists):
    col = artist.get_facecolor()
    artist.set_edgecolor(col)
    for j in range(i*6, i*6+6):
        line = ax3.lines[j]
        line.set_color(col)
        line.set_mfc(col)
        line.set_mec(col)

f.tight_layout(pad=2)
sns.despine()
plt.savefig(dir_path+'/../graphs/conditionalATE_nl.png')


#############
# Finally, each DGP realization provides a prediction for the
# average treatment effect.
# Posterior mean and standard deviation for y_t − y_c on every obs.
np.random.seed(778)
averageTreatmentEffect = np.zeros(N)
for i, (cart_t, cart_c) in enumerate(zip(bf_t.estimators_, bf_c.estimators_)):
    theta = np.random.exponential(scale=1, size=df.shape[0])
    y_t = cart_t.predict(df.drop(["t", "y_hete"], axis=1).values)
    y_c = cart_c.predict(df.drop(["t", "y_hete"], axis=1).values)
    ate = np.dot(theta, y_t - y_c) / theta.sum()
    averageTreatmentEffect[i] = ate

averageTreatmentEffect.mean()
averageTreatmentEffect.std()
