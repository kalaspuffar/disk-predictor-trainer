import os
import gc
import datetime
import sys
import json

import numpy as np
import pandas as pd
import pickle

import dask.dataframe as dd
from dask.dataframe import to_datetime
from dask.diagnostics import ProgressBar

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
#from dask_ml.wrappers import Incremental
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    pairwise_distances_argmin,
)

from src import utils

pbar = ProgressBar()
pbar.register()

# inferred int64 types cause a type mismatch (int vs float) error when dask sees a null value
# null values cannot be interpreted as ints
custom_dtypes = {
    "seagate": {
        "date": "str",
        "serial_number": "object",
        "capacity_bytes": "float32",
        "failure": "float32",
        "smart_1_normalized": "float32",
        "smart_1_raw": "float32",
        "smart_5_normalized": "float32",
        "smart_5_raw": "float32",
        "smart_7_normalized": "float32",
        "smart_7_raw": "float32",
        "smart_9_normalized": "float32",
        "smart_9_raw": "float32",
        "smart_10_normalized": "float32",
        "smart_10_raw": "float32",
        "smart_184_normalized": "float32",
        "smart_184_raw": "float32",
        "smart_187_normalized": "float32",
        "smart_187_raw": "float32",
        "smart_188_normalized": "float32",
        "smart_188_raw": "float32",
        "smart_189_normalized": "float32",
        "smart_189_raw": "float32",
        "smart_190_normalized": "float32",
        "smart_190_raw": "float32",
        "smart_193_normalized": "float32",
        "smart_193_raw": "float32",
        "smart_194_normalized": "float32",
        "smart_194_raw": "float32",
        "smart_197_normalized": "float32",
        "smart_197_raw": "float32",
        "smart_198_normalized": "float32",
        "smart_198_raw": "float32",
        "smart_240_normalized": "float32",
        "smart_240_raw": "float32",
        "smart_241_normalized": "float32",
        "smart_241_raw": "float32",
        "smart_242_normalized": "float32",
        "smart_242_raw": "float32",
    },
    "hgst": {
        "date": "str",
        "serial_number": "object",
        "capacity_bytes": "float32",
        "failure": "float32",
        "smart_1_normalized": "float32",
        "smart_1_raw": "float32",
        "smart_2_normalized": "float32",
        "smart_2_raw": "float32",
        "smart_3_normalized": "float32",
        "smart_3_raw": "float32",
        "smart_4_normalized": "float32",
        "smart_4_raw": "float32",
        "smart_5_normalized": "float32",
        "smart_5_raw": "float32",
        "smart_7_normalized": "float32",
        "smart_7_raw": "float32",
        "smart_8_normalized": "float32",
        "smart_8_raw": "float32",
        "smart_9_normalized": "float32",
        "smart_9_raw": "float32",
        "smart_10_normalized": "float32",
        "smart_10_raw": "float32",
        "smart_12_normalized": "float32",
        "smart_12_raw": "float32",
        "smart_22_normalized": "float32",
        "smart_22_raw": "float32",
        "smart_192_normalized": "float32",
        "smart_192_raw": "float32",
        "smart_193_normalized": "float32",
        "smart_193_raw": "float32",
        "smart_194_normalized": "float32",
        "smart_194_raw": "float32",
        "smart_196_normalized": "float32",
        "smart_196_raw": "float32",
        "smart_197_normalized": "float32",
        "smart_197_raw": "float32",
        "smart_198_normalized": "float32",
        "smart_198_raw": "float32",
        "smart_199_normalized": "float32",
        "smart_199_raw": "float32",
    },
    "hitachi": {
        "date": "str",
        "serial_number": "object",
        "capacity_bytes": "float32",
        "failure": "float32",
        "smart_1_normalized": "float32",
        "smart_1_raw": "float32",
        "smart_2_normalized": "float32",
        "smart_2_raw": "float32",
        "smart_3_normalized": "float32",
        "smart_3_raw": "float32",
        "smart_4_normalized": "float32",
        "smart_4_raw": "float32",
        "smart_5_normalized": "float32",
        "smart_5_raw": "float32",
        "smart_7_normalized": "float32",
        "smart_7_raw": "float32",
        "smart_8_normalized": "float32",
        "smart_8_raw": "float32",
        "smart_9_normalized": "float32",
        "smart_9_raw": "float32",
        "smart_10_normalized": "float32",
        "smart_10_raw": "float32",
        "smart_12_normalized": "float32",
        "smart_12_raw": "float32",
        "smart_192_normalized": "float32",
        "smart_192_raw": "float32",
        "smart_193_normalized": "float32",
        "smart_193_raw": "float32",
        "smart_194_normalized": "float32",
        "smart_194_raw": "float32",
        "smart_196_normalized": "float32",
        "smart_196_raw": "float32",
        "smart_197_normalized": "float32",
        "smart_197_raw": "float32",
        "smart_198_normalized": "float32",
        "smart_198_raw": "float32",
        "smart_199_normalized": "float32",
        "smart_199_raw": "float32",
    },
    "toshiba": {
        "date": "str",
        "serial_number": "object",
        "capacity_bytes": "float32",
        "failure": "float32",
        "smart_1_normalized": "float32",
        "smart_1_raw": "float32",
        "smart_2_normalized": "float32",
        "smart_2_raw": "float32",
        "smart_3_normalized": "float32",
        "smart_3_raw": "float32",
        "smart_4_normalized": "float32",
        "smart_4_raw": "float32",
        "smart_5_normalized": "float32",
        "smart_5_raw": "float32",
        "smart_7_normalized": "float32",
        "smart_7_raw": "float32",
        "smart_8_normalized": "float32",
        "smart_8_raw": "float32",
        "smart_9_normalized": "float32",
        "smart_9_raw": "float32",
        "smart_10_normalized": "float32",
        "smart_10_raw": "float32",
        "smart_12_normalized": "float32",
        "smart_12_raw": "float32",
        "smart_192_normalized": "float32",
        "smart_192_raw": "float32",
        "smart_193_normalized": "float32",
        "smart_193_raw": "float32",
        "smart_194_normalized": "float32",
        "smart_194_raw": "float32",
        "smart_196_normalized": "float32",
        "smart_196_raw": "float32",
        "smart_197_normalized": "float32",
        "smart_197_raw": "float32",
        "smart_198_normalized": "float32",
        "smart_198_raw": "float32",
        "smart_199_normalized": "float32",
        "smart_199_raw": "float32",
    },
    "wdc": {
        "date": "str",
        "serial_number": "object",
        "capacity_bytes": "float32",
        "failure": "float32",
        "smart_1_normalized": "float32",
        "smart_1_raw": "float32",
        "smart_3_normalized": "float32",
        "smart_3_raw": "float32",
        "smart_4_normalized": "float32",
        "smart_4_raw": "float32",
        "smart_5_normalized": "float32",
        "smart_5_raw": "float32",
        "smart_7_normalized": "float32",
        "smart_7_raw": "float32",
        "smart_9_normalized": "float32",
        "smart_9_raw": "float32",
        "smart_10_normalized": "float32",
        "smart_10_raw": "float32",
        "smart_12_normalized": "float32",
        "smart_12_raw": "float32",
        "smart_192_normalized": "float32",
        "smart_192_raw": "float32",
        "smart_193_normalized": "float32",
        "smart_193_raw": "float32",
        "smart_194_normalized": "float32",
        "smart_194_raw": "float32",
        "smart_196_normalized": "float32",
        "smart_196_raw": "float32",
        "smart_197_normalized": "float32",
        "smart_197_raw": "float32",
        "smart_198_normalized": "float32",
        "smart_198_raw": "float32",
        "smart_199_normalized": "float32",
        "smart_199_raw": "float32",
    }    
}

MANUFACTURER = sys.argv[1]
DATA_DIR = "/home/woden/predict"

pattern = os.path.join(DATA_DIR, f"data_Q*_????_{MANUFACTURER}_clean", "*.csv")

# Read all matching CSVs
df = dd.read_csv(
    pattern,
    dtype=custom_dtypes[MANUFACTURER],
    usecols=custom_dtypes[MANUFACTURER].keys()
)

df = utils.optimal_repartition_df(df)

# convert from str to datetime
df = df[df["date"] != "0.0"]
df["date"] = to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])
df["date"] = df["date"].astype("datetime64[ns]")

# define thresholds as timedelta
BAD_THRESHOLD_NDAYS = np.timedelta64(14, "D")
WARNING_THRESHOLD_NDAYS = np.timedelta64(42, "D")

# get the serial numbers for all the failed hard drives
failed_serials = df[df["failure"] == 1]["serial_number"].compute()

# failed drives data
failed_df = df[df["serial_number"].isin(failed_serials)]
failed_df.head()

num_working_serials = 5000
working_sers = df[~df["serial_number"].isin(failed_serials)]["serial_number"].unique()
working_repr_sers = working_sers.sample(
    frac=(num_working_serials / len(working_sers))
).compute()

# downsample the dataset
working_df = df[df["serial_number"].isin(working_repr_sers)]
working_df.head()

# concatenate rows
df = dd.concat([failed_df, working_df], interleave_partitions=True)
df.head()

del failed_df
del working_df
gc.collect()

# =============================== FOR DASK =============================== #
# create meta of the resulting failed_df otherwise dask complains
rul_meta = df._meta
rul_meta = rul_meta.assign(rul_days=rul_meta["date"].max() - rul_meta["date"])
# ======================================================================== #

# get remaining useful life as diff(today, maxday)
# reset index coz result is multiindexed. drop=True coz serial_number already exists as a col
df = (
    df.groupby("serial_number")
    .apply(utils.append_rul_days_column, meta=rul_meta)
    .reset_index(drop=True)
)
df.head()

# remove working drive data that is recorded after [quarter end minus 6 weeks]
# because we dont know (as of quarter end) if those drives survived more than 6 weeks or not
df = df[
    (df["serial_number"].isin(failed_serials))
    | (df["rul_days"] >= WARNING_THRESHOLD_NDAYS)
]
print(dd.compute(df.shape))

# NOTE: assignment must be done in th
# df.head()is order otherwise it wont be correct. FIXME
# assign all as good initially
df["status"] = 0

# overwrite those which have rul less than 6 weeks as warning
df["status"] = df["status"].mask(df["rul_days"] < WARNING_THRESHOLD_NDAYS, 1)

# overwrite those which have rul less than 2 weeks as bad
df["status"] = df["status"].mask(df["rul_days"] < BAD_THRESHOLD_NDAYS, 2)

df.head()

# columns for which diffs must be calculated
diff_cols = [col for col in df.columns if col.lower().startswith("smart")]

def add_deltas(group):
    # FIXME: workaround for passing diff_cols as parameter
    global diff_cols
    for colname in diff_cols:
        # add rate of change
        roc_colname = "d_" + colname
        group[roc_colname] = (group[colname] - group[colname].shift(1)).fillna(
            method="bfill"
        )
        # add rate of rate of change
        roroc_colname = "d2_" + colname
        group[roroc_colname] = (
            group[roc_colname] - group[roc_colname].shift(1)
        ).fillna(method="bfill")
    return group

# # =============================== FOR DASK =============================== #
# # create meta of the resulting failed_df otherwise dask complains
#rul_meta = df._meta
#rul_meta = rul_meta.assign(rul_days=rul_meta['date'].max()-rul_meta['date'])
# # ======================================================================== #

# get remaining useful life as diff(today, maxday)
# reset index coz result is multiindexed. drop=True coz serial_number already exists as a col
df = df.groupby("serial_number").apply(add_deltas).reset_index(drop=True)

# dcols = [col for col in df.columns if ((col.lower().startswith('d_')) or (col.lower().startswith('d2_')))]
# X_train.drop(dcols, axis=1).isna().any().any()    # False. this means nans arise from diffs
df = df.fillna(pd.Timedelta(seconds=0))
df.isna().any().compute()

print("Before shape compute")

dd.compute(df.shape)

print("After shape compute")

def pandas_rolling_feats(
    df,
    window=6,
    drop_cols=("date", "failure", "capacity_bytes", "rul"),
    group_cols=("serial_number"),
    cap=True,
):
    # save the status labels
    statuses = df["status"]

    # group by serials, drop cols which are not to be aggregated
    if drop_cols is not None:
        grouped_df = df.drop(drop_cols, axis=1).groupby(group_cols)
    else:
        grouped_df = df.groupby(group_cols)

    # feature columns
    featcols = grouped_df.first().columns

    # get mean value in last 6 days
    means = grouped_df.rolling(window)[featcols].mean()

    # get std in last 6 days
    stds = grouped_df.rolling(window)[featcols].std()

    # coefficient of variation
    cvs = stds.divide(means, fill_value=0)

    # rename before mergeing
    means = means.rename(columns={col: "mean_" + col for col in means.columns})
    stds = stds.rename(columns={col: "std_" + col for col in stds.columns})
    cvs = cvs.rename(columns={col: "cv_" + col for col in cvs.columns})

    # combine features into one df
    res = means.merge(stds, left_index=True, right_index=True)
    res = res.merge(cvs, left_index=True, right_index=True)

    # drop rows where all columns are nans
    res = res.dropna(how="all")

    # fill nans created by cv calculation
    res = res.fillna(0)

    # capacity of hard drive
    if cap:
        capacities = (
            df[["serial_number", "capacity_bytes"]].groupby("serial_number").max()
        )
        res = res.merge(capacities, left_index=True, right_index=True)

    # bring serial number back as a col instead of index, preserve the corresponding indices
    res = res.reset_index(level=[0])

    # add status labels back.
    res = res.merge(statuses, left_index=True, right_index=True)

    return res

# must convert dask to pandas
df = df.compute()

print("After convert to pandas")

# MUST make sure indices are unique before processing
df = df.reset_index(drop=True)
feats_df = pandas_rolling_feats(
    df,
    window=6,
    drop_cols=["date", "capacity_bytes", "failure", "rul_days", "status"],
    group_cols=["serial_number"],
)

del df
gc.collect()


# infinities get added to df - remove these
feats_df = feats_df.replace([np.inf, -np.inf], np.nan).dropna()
feats_df.head()

print("Before series preparation")

# ['date', 'serial_number', 'model', 'failure', 'rul_days', 'status']
# ['serial_number', 'status']
X_arr = feats_df.drop(["serial_number", "status"], axis=1)
Y_arr = feats_df[["serial_number", "status"]]

del feats_df
gc.collect()

# failed serials left after reduction
failed_sers_red = pd.Series(Y_arr["serial_number"].unique())
failed_sers_red = failed_sers_red[failed_sers_red.isin(failed_serials)]

# working serials left after reduction
working_sers_red = pd.Series(Y_arr["serial_number"].unique())
working_sers_red = working_sers_red[~working_sers_red.isin(failed_serials)]

print("Before split")

# split working and failed
working_train, working_test = train_test_split(
    working_sers_red, test_size=0.2, random_state=42
)
failed_train, failed_test = train_test_split(
    failed_sers_red, test_size=0.2, random_state=42
)

print("After split")

# use serial numbers to generate train/test set
# CHECKED OK - train/test ratio 0.8, fail/work and overall both
X_train_work = X_arr[Y_arr["serial_number"].isin(working_train)]
X_train_fail = X_arr[Y_arr["serial_number"].isin(failed_train)]
X_train = pd.concat([X_train_work, X_train_fail])

Y_train_work = Y_arr[Y_arr["serial_number"].isin(working_train)]["status"]
Y_train_fail = Y_arr[Y_arr["serial_number"].isin(failed_train)]["status"]
Y_train = pd.concat([Y_train_work, Y_train_fail])

print("After half of the concatination")

X_test_work = X_arr[Y_arr["serial_number"].isin(working_test)]
X_test_fail = X_arr[Y_arr["serial_number"].isin(failed_test)]
X_test = pd.concat([X_test_work, X_test_fail])

Y_test_work = Y_arr[Y_arr["serial_number"].isin(working_test)]["status"]
Y_test_fail = Y_arr[Y_arr["serial_number"].isin(failed_test)]["status"]
Y_test = pd.concat([Y_test_work, Y_test_fail])

# verufy proportions
print(X_train_fail.shape)
print(X_test_fail.shape)
print(X_train_work.shape)
print(X_test_work.shape)

print(X_train.columns)

col_list = X_train.columns.tolist()
fconfig_name = "models/{}.config".format(MANUFACTURER)

with open(fconfig_name, "w") as f:
    json.dump(col_list, f, indent=2)

print("Create Scaler")

# robust scaling to not be outlier sensitive
#scaler = RobustScaler()
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

print("Scaler FIT")

# use mean values as threshold
class0_mean = X_train[(Y_train == 0).values, :].mean(axis=0, keepdims=True)
class1_mean = X_train[(Y_train == 1).values, :].mean(axis=0, keepdims=True)
class2_mean = X_train[(Y_train == 2).values, :].mean(axis=0, keepdims=True)
class_means = np.vstack((class0_mean, class1_mean, class2_mean))

# predict based on a distance metric from mean values
# NOTE: canberra, cosine generally work better than l2,l1,etc
print("Scaler predict mean")

preds = pairwise_distances_argmin(
    scaler.transform(X_test), class_means, metric="canberra"
)

print("Scaler END predict mean")


# how does the baseline look
cm = confusion_matrix(Y_test, preds)
cm
cm / cm.sum(axis=1, keepdims=True)

# use median values as threshold
class0_median = np.median(X_train[(Y_train == 0).values, :], axis=0, keepdims=True)
class1_median = np.median(X_train[(Y_train == 1).values, :], axis=0, keepdims=True)
class2_median = np.median(X_train[(Y_train == 2).values, :], axis=0, keepdims=True)
class_medians = np.vstack((class0_median, class1_median, class2_median))

# predict based on a distance metric from median values
# NOTE: canberra, cosine generally work better than l2,l1,etc
print("Scaler predict median")
preds = pairwise_distances_argmin(
    scaler.transform(X_test), class_medians, metric="canberra"
)
print("Scaler END predict median")

# how does the baseline look
cm = confusion_matrix(Y_test, preds)
cm
cm / cm.sum(axis=1, keepdims=True)

# use min values as threshold
class0_min = X_train[(Y_train == 0).values, :].min(axis=0, keepdims=True)
class1_min = X_train[(Y_train == 1).values, :].min(axis=0, keepdims=True)
class2_min = X_train[(Y_train == 2).values, :].min(axis=0, keepdims=True)
class_mins = np.vstack((class0_min, class1_min, class2_min))

# predict based on a distance metric from min values
# NOTE: canberra, cosine generally work better than l2,l1,etc
print("Scaler predict min")
preds = pairwise_distances_argmin(scaler.transform(X_test), class_mins, metric="cosine")
print("Scaler END predict min")

# how does the baseline look
cm = confusion_matrix(Y_test, preds)
cm
cm / cm.sum(axis=1, keepdims=True)

# use max values as threshold
class0_max = X_train[(Y_train == 0).values, :].max(axis=0, keepdims=True)
class1_max = X_train[(Y_train == 1).values, :].max(axis=0, keepdims=True)
class2_max = X_train[(Y_train == 2).values, :].max(axis=0, keepdims=True)
class_maxs = np.vstack((class0_max, class1_max, class2_max))

# predict based on a distance metric from max values
# NOTE: canberra, cosine generally work better than l2,l1,etc
# preds = pairwise_distances_argmin(X_test, class_maxs, metric='cosine')
print("Scaler predict max")
class0_confs = (scaler.transform(X_test) > class0_max).sum(axis=1, keepdims=True)
class1_confs = (scaler.transform(X_test) > class1_max).sum(axis=1, keepdims=True)
class2_confs = (scaler.transform(X_test) > class2_max).sum(axis=1, keepdims=True)
class_confs = np.hstack((class0_confs, class1_confs, class2_confs))
preds = np.argmax(class_confs, axis=1)
print("Scaler END predict max")

# how does the baseline look
cm = confusion_matrix(Y_test, preds)
cm
cm / cm.sum(axis=1, keepdims=True)

# training is going to require a lot of memory. free as much as possible
del X_train_work
del X_train_fail
del Y_train_work
del Y_train_fail
del X_test_work
del X_test_fail
del Y_test_work
del Y_test_fail
gc.collect()

fscaler_name = "models/{}_scaler.pkl".format(MANUFACTURER)
# Save .pkl files
with open(fscaler_name, "wb") as f_scaler:
    pickle.dump(scaler, f_scaler)

print("Saved scaler")

dt_clf = DecisionTreeClassifier(random_state=24)
dt_clf.fit(X_train, Y_train)

print("Decision tree Trained")

# get preds
dt_preds = dt_clf.predict(scaler.transform(X_test))
dt_confmat = confusion_matrix(Y_test, dt_preds)

print(dt_confmat)
print(classification_report(Y_test, dt_preds, target_names=['good', 'warning', 'bad']))

dt_predictor_name = "dt_models/{}_predictor.pkl".format(MANUFACTURER)

with open(dt_predictor_name, "wb") as f_model:
    pickle.dump(dt_clf, f_model)

print("Saved decision tree classifier")

rf_clf = RandomForestClassifier(n_estimators=12, class_weight='balanced', n_jobs=-1, random_state=24)
rf_clf.fit(X_train, Y_train)

print("Random forest Trained")

# get preds
rf_preds = rf_clf.predict(scaler.transform(X_test))
rf_confmat = confusion_matrix(Y_test, rf_preds)

print(rf_confmat)
print(classification_report(Y_test, rf_preds, target_names=['good', 'warning', 'bad']))

rf_predictor_name = "rf_models/{}_predictor.pkl".format(MANUFACTURER)

with open(rf_predictor_name, "wb") as f_model:
    pickle.dump(rf_clf, f_model)

print("Saved random forest classifier")

svc = SVC(class_weight="balanced")
svc.fit(X_train, Y_train)

print("SVC Trained")

# get preds
svc_preds = svc.predict(scaler.transform(X_test))
svc_confmat = confusion_matrix(Y_test, svc_preds)

print(svc_confmat)
print(classification_report(Y_test, svc_preds, target_names=["good", "warning", "bad"]))

svc_predictor_name = "svc_models/{}_predictor.pkl".format(MANUFACTURER)

with open(svc_predictor_name, "wb") as f_model:
    pickle.dump(svc, f_model)

print("Saved SVM model.")

mlp = MLPClassifier(hidden_layer_sizes=(128, 512, 512, 128),
                    activation='relu',
                    batch_size=256,
                    warm_start=True,
                    random_state=24)
# clf = Incremental(mlp)
mlp.fit(X_train, Y_train)

print("MLP Trained")

# with parallel_backend('dask'):
# get preds
mlp_preds = mlp.predict(scaler.transform(X_test))
mlp_confmat = confusion_matrix(Y_test, mlp_preds)

print(np.around(mlp_confmat / mlp_confmat.sum(axis=1, keepdims=True), decimals=2))
print(classification_report(Y_test, mlp_preds, target_names=['good', 'warning', 'bad']))

mlp_predictor_name = "mlp_models/{}_predictor.pkl".format(MANUFACTURER)

with open(mlp_predictor_name, "wb") as f_model:
    pickle.dump(mlp, f_model)


print("Saved MLP classifier")
