import gc
import os
import sys

import numpy as np
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

import seaborn as sns

from src import utils

sns.set()
BASE_DIR = "/home/woden/predict"
DATA_DIR = os.path.join(BASE_DIR, sys.argv[1])

# register progress bar for compute calls in dask so we have an estimate of how long task will take
pbar = ProgressBar()
pbar.register()

# inferred int32 types cause a type mismatch (int vs float) error when dask sees a null value
# null values cannot be interpreted as ints
custom_dtypes = {
    "date": "object",
    "serial_number": "object",
    "model": "object",
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
    "smart_11_normalized": "float32",
    "smart_11_raw": "float32",
    "smart_12_normalized": "float32",
    "smart_12_raw": "float32",
    "smart_13_normalized": "float32",
    "smart_13_raw": "float32",
    "smart_15_normalized": "float32",
    "smart_15_raw": "float32",
    "smart_16_normalized": "float32",
    "smart_16_raw": "float32",
    "smart_17_normalized": "float32",
    "smart_17_raw": "float32",
    "smart_22_normalized": "float32",
    "smart_22_raw": "float32",
    "smart_23_normalized": "float32",
    "smart_23_raw": "float32",
    "smart_24_normalized": "float32",
    "smart_24_raw": "float32",
    "smart_168_normalized": "float32",
    "smart_168_raw": "float32",
    "smart_170_normalized": "float32",
    "smart_170_raw": "float32",
    "smart_173_normalized": "float32",
    "smart_173_raw": "float32",
    "smart_174_normalized": "float32",
    "smart_174_raw": "float32",
    "smart_177_normalized": "float32",
    "smart_177_raw": "float32",
    "smart_179_normalized": "float32",
    "smart_179_raw": "float32",
    "smart_181_normalized": "float32",
    "smart_181_raw": "float32",
    "smart_182_normalized": "float32",
    "smart_182_raw": "float32",
    "smart_183_normalized": "float32",
    "smart_183_raw": "float32",
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
    "smart_191_normalized": "float32",
    "smart_191_raw": "float32",
    "smart_192_normalized": "float32",
    "smart_192_raw": "float32",
    "smart_193_normalized": "float32",
    "smart_193_raw": "float32",
    "smart_194_normalized": "float32",
    "smart_194_raw": "float32",
    "smart_195_normalized": "float32",
    "smart_195_raw": "float32",
    "smart_196_normalized": "float32",
    "smart_196_raw": "float32",
    "smart_197_normalized": "float32",
    "smart_197_raw": "float32",
    "smart_198_normalized": "float32",
    "smart_198_raw": "float32",
    "smart_199_normalized": "float32",
    "smart_199_raw": "float32",
    "smart_200_normalized": "float32",
    "smart_200_raw": "float32",
    "smart_201_normalized": "float32",
    "smart_201_raw": "float32",
    "smart_218_normalized": "float32",
    "smart_218_raw": "float32",
    "smart_220_normalized": "float32",
    "smart_220_raw": "float32",
    "smart_222_normalized": "float32",
    "smart_222_raw": "float32",
    "smart_223_normalized": "float32",
    "smart_223_raw": "float32",
    "smart_224_normalized": "float32",
    "smart_224_raw": "float32",
    "smart_225_normalized": "float32",
    "smart_225_raw": "float32",
    "smart_226_normalized": "float32",
    "smart_226_raw": "float32",
    "smart_231_normalized": "float32",
    "smart_231_raw": "float32",
    "smart_232_normalized": "float32",
    "smart_232_raw": "float32",
    "smart_233_normalized": "float32",
    "smart_233_raw": "float32",
    "smart_235_normalized": "float32",
    "smart_235_raw": "float32",
    "smart_240_normalized": "float32",
    "smart_240_raw": "float32",
    "smart_241_normalized": "float32",
    "smart_241_raw": "float32",
    "smart_242_normalized": "float32",
    "smart_242_raw": "float32",
    "smart_250_normalized": "float32",
    "smart_250_raw": "float32",
    "smart_251_normalized": "float32",
    "smart_251_raw": "float32",
    "smart_252_normalized": "float32",
    "smart_252_raw": "float32",
    "smart_254_normalized": "float32",
    "smart_254_raw": "float32",
    "smart_255_normalized": "float32",
    "smart_255_raw": "float32",
}

# read all the data into one dataframe
df = dd.read_csv(os.path.join(DATA_DIR, "*.csv"), dtype=custom_dtypes)

# get the seagate data
seagate_df = df[df["model"].str.startswith(("S", "ZA"))]

# get the serial numbers for all the failed hard drives
failed_serials = seagate_df[seagate_df["failure"] == 1]["serial_number"].compute()

# get the serial numbers for all the failed hard drives, date of failure, and its model
# multiple entries will exist per serial number since it will be a time series. get only the last one
working_serials = (
    seagate_df[~seagate_df["serial_number"].isin(failed_serials)]["serial_number"]
    .drop_duplicates(keep="last")
    .compute()
)

# proof of duplicate indices
seagate_df.loc[0, :].compute().head()

# work with critical columns for now
CRITICAL_STATS = [
    1,
    5,
    7,
    9,
    10,
    184,
    187,
    188,
    189,
    190,
    193,
    194,
    196,
    197,
    198,
    201,
    240,
    241,
    242,
]  # NOTE: 201 is all nans
crit_cols_raw = ["smart_{}_raw".format(i) for i in CRITICAL_STATS]
crit_cols_normalized = ["smart_{}_normalized".format(i) for i in CRITICAL_STATS]

done_stats = [
    2,
    8,
    11,
    196,
    223,
    225,
    250,
    251,
    252,
    16,
    17,
    168,
    170,
    173,
    174,
    177,
    218,
    231,
    232,
    233,
    235,
    254,
    183,
    200,
    195,
    191,
]
done_cols = ["smart_{}_raw".format(i) for i in done_stats] + [
    "smart_{}_normalized".format(i) for i in done_stats
]

# the columns to keep for analysis
keep_cols = (
    ["date", "serial_number", "capacity_bytes", "failure"]
    + crit_cols_raw
    + crit_cols_normalized
)

# dummy value to replace nans
DUMMY_VALUE = -100

# copy of df on which to perform cleaning tasks
clean_df = seagate_df[keep_cols]

# meta data for later use
initial_shape = dd.compute(clean_df.shape)[0]
num_total_datapts = initial_shape[0]
print("Initial shape =", initial_shape)

# how bad is the nan situation for critical columns? get counts as a percent of total
seagate_nans = utils.get_nan_count_percent(clean_df, num_total_datapts)

# show only values which at least some nans
seagate_nans[
    (seagate_nans["percent"] != 0) & (seagate_nans["percent"] != 1)
].compute().sort_values(by="percent", ascending=False)

# drop columns that are all nans. NOTE: dropna on axis=1 is not supported in dask yet
MAJORITY_THRESHOLD = 0.99
#clean_df = clean_df.drop(
#    seagate_nans[seagate_nans["percent"] > MAJORITY_THRESHOLD].index.compute(), axis=1
#)
to_drop = (
    seagate_nans[seagate_nans["percent"] > MAJORITY_THRESHOLD]
    .index
    .compute()
    .tolist()
)
clean_df = clean_df.drop(to_drop, axis=1)

dd.compute(clean_df.shape)

# get the data points where 193 is null. inspect it for patterns
nan193_df = clean_df[clean_df["smart_193_raw"].isna()].compute()

# number of nans in this subset as a percentage of nans in the overall data
utils.get_nan_count_percent(nan193_df, num_total_datapts)

# number of wokring vs failed in overall df
seagate_df["failure"].value_counts().compute()

# number of working vs failed drives in subset
nan193_df["failure"].value_counts()

# do the failed drives report nans only on the last day?
# to figure this, get num of rows we have for failed sers
# if >1 that means failed drives reported nan on other days as well
sers = nan193_df[nan193_df["failure"] == 1]["serial_number"]
nan193_df[nan193_df["serial_number"].isin(sers)]["serial_number"].value_counts()

# retain columns iff they belong to a working drive or have non null values for 193 (and 1, 5, 7 .. by extension)
clean_df = clean_df[
    ~clean_df["smart_193_raw"].isna() | clean_df["serial_number"].isin(sers)
]

print(dd.compute(clean_df.shape)[0])
utils.get_nan_count_percent(clean_df).compute()

# collect garbage
del nan193_df
del sers
gc.collect()

# get the data points where 240 is null. inspect it for patterns
nan240_df = clean_df[clean_df["smart_240_raw"].isna()].compute()

# number of nans in this subset as a percentage of nans in the original overall data
utils.get_nan_count_percent(nan240_df, num_total_datapts).sort_values(
    by="percent", ascending=False
)

# fill in dummy values
cols_to_fill = [
    "smart_240_raw",
    "smart_240_normalized",
    "smart_241_raw",
    "smart_241_normalized",
    "smart_242_raw",
    "smart_242_normalized",
]

# must do it in for loop, dask does not like indexing with list
# plus, its not straightforward to mask isna of specific columns
for col in cols_to_fill:
    clean_df[col] = clean_df[col].mask(clean_df[col].isna(), DUMMY_VALUE)

# how are things after this part
print(dd.compute(clean_df.shape)[0])
utils.get_nan_count_percent(clean_df).compute()

del nan240_df
gc.collect()

# get a subset of df to investigate the behavior of cols with too many nans - eg why is smart 184 mostly nans?
sub = clean_df[
    [
        "date",
        "serial_number",
        "smart_184_raw",
        "smart_184_normalized",
        "smart_189_raw",
        "smart_189_normalized",
    ]
]

# general description of data in these rows
sub[
    ["smart_184_raw", "smart_184_normalized", "smart_189_raw", "smart_189_normalized"]
].compute().describe()

# histograms to visualize anything unusual
sub[
    ["smart_184_raw", "smart_184_normalized", "smart_189_raw", "smart_189_normalized"]
].compute().hist(bins=50)

# get the data of the failed drives from the subset and see if there is any pattern in failed vs non failed
sub_failed = sub[["date", "serial_number", "smart_189_raw", "smart_184_raw"]][
    sub["serial_number"].isin(failed_serials)
]
sub_working = sub[["date", "serial_number", "smart_189_raw", "smart_184_raw"]][
    ~sub["serial_number"].isin(failed_serials)
]

# are 184 and 189 simulataneously nan for all drives
print(
    (sub_failed["smart_184_raw"].isna() == sub_failed["smart_189_raw"].isna())
    .all()
    .compute()
)
print(
    (sub_working["smart_184_raw"].isna() == sub_working["smart_189_raw"].isna())
    .all()
    .compute()
)

# from the subset, get serial numbers where 184 is nan, for failed drives
sub_failed_nanserials = (
    sub_failed[sub_failed["smart_184_raw"].isna()]["serial_number"].unique().compute()
)
print(sub_failed_nanserials.shape)

# from the subset, get serial numbers where 184 is nan, for working drives
sub_working_nanserials = (
    sub_working[sub_working["smart_184_raw"].isna()]["serial_number"].unique().compute()
)
print(sub_working_nanserials.shape)

# how many drives in cleaned version of dataset
# this should be same as in whole dataset - i.e. we shouldn't have removed ALL the data for a drive yet
print(clean_df["serial_number"].drop_duplicates().isin(failed_serials).sum().compute())
print(
    seagate_df["serial_number"].drop_duplicates().isin(failed_serials).sum().compute()
)

# for each serial number, what percentage of values in the time series are nans?
# this is to confirm a hunch - drives have either all values as nans or no values as nans
failed_nanpercent = (
    sub_failed[["serial_number", "smart_184_raw"]]
    .groupby("serial_number")
    .apply(lambda group: group["smart_184_raw"].isna().sum() / group.shape[0])
)
working_nanpercent = (
    sub_working[["serial_number", "smart_184_raw"]]
    .groupby("serial_number")
    .apply(lambda group: group["smart_184_raw"].isna().sum() / group.shape[0])
)

# how many failed drives are there which have neither all-nan nor no-nan
failed_nanpercent[
    (failed_nanpercent != 1) & (failed_nanpercent != 0)
].compute()  # S3010MAK, S3010LHR

# how many working drives are there which have neither all-nan nor no-nan
working_nanpercent[
    (working_nanpercent != 1) & (working_nanpercent != 0)
].compute().sort_values(
    ascending=False
)  # 87 values, max percent 0.011494/1.0

# get all the serial numbers where 184 is seen to be nan at least once
nan184_serials = (
    clean_df[clean_df["smart_184_raw"].isna()]["serial_number"].unique().compute()
)  # len = 32005

isallnan184_serials = clean_df[clean_df["serial_number"].isin(nan184_serials)][
    ["serial_number", "smart_184_raw"]
]
isallnan184_serials = isallnan184_serials.groupby("serial_number").apply(
    lambda g: g["smart_184_raw"].isna().all()
)

# check which class is minority - this will be used for lookup using isin, for efficiency
isallnan184_serials.value_counts().compute()

# serial numbers which have all-nans for 184 and 189 and are therefore to be filled with dummy value
dummyfill_sers = isallnan184_serials[isallnan184_serials].index.compute()
dummyfill_sers

# fill columns with dummy value
cols_to_fill = [
    "smart_184_raw",
    "smart_184_normalized",
    "smart_189_raw",
    "smart_189_normalized",
]
for col in cols_to_fill:
    clean_df[col] = clean_df[col].mask(
        clean_df["serial_number"].isin(dummyfill_sers), DUMMY_VALUE
    )

# how are things after this part
print(dd.compute(clean_df.shape)[0])
utils.get_nan_count_percent(clean_df).compute()

# collect garbage
del sub
del sub_failed
del sub_working
del dummyfill_sers
del nan184_serials
del isallnan184_serials
del sub_failed_nanserials
del sub_working_nanserials
del failed_nanpercent
del working_nanpercent
gc.collect()

# fill nans with mean values
clean_df = clean_df.fillna(value=clean_df.mean())
print("Final shape =", dd.compute(clean_df.shape)[0])

# verify that there are no more nans
utils.get_nan_count_percent(clean_df).compute()

# get ideal number of partitions
PARTITION_SIZE_BYTES = 100 * 10 ** 6
DF_SIZE_BYTES = clean_df.memory_usage(deep=True).sum().compute()
NUM_PARTITIONS = int(np.ceil(DF_SIZE_BYTES / PARTITION_SIZE_BYTES))

# repartition and save cleaned version of data
clean_df = clean_df.repartition(npartitions=NUM_PARTITIONS)

# create save dir as same name as data dir, but suffixed with "_clean"
head, tail = os.path.split(os.path.normpath(DATA_DIR))
save_dir = os.path.join(head, tail + "_seagate_clean")

# if path doesnt exist, mkdir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# save partitions
clean_df.to_csv(os.path.join(save_dir, "partition_*.csv"), index=False)

# unregister dask compute call progress bar
pbar.unregister()
