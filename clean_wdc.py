import gc
import os
import sys

import numpy as np

import dask.dataframe as dd
from dask.diagnostics import ProgressBar

from src import utils

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

# get the hgst data
hgst_df = df[df["model"].str.startswith("WDC")]

# get the serial numbers for all the failed hard drives
failed_serials = hgst_df[hgst_df["failure"] == 1]["serial_number"].compute()

dd.compute(hgst_df.shape)

# proof of duplicate indices
hgst_df.loc[0, :].compute().head()

# work with critical columns for now
CRITICAL_STATS = [
    1,
    5,
    7,
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

# these are the columns analyzed and nans are accounted for these (see https://trello.com/c/tjFl6RHf)
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
clean_df = hgst_df

# get ideal number of partitions
PARTITION_SIZE_BYTES = 100 * 10 ** 6
DF_SIZE_BYTES = clean_df.memory_usage(deep=True).sum().compute()
NUM_PARTITIONS = int(np.ceil(DF_SIZE_BYTES / PARTITION_SIZE_BYTES))

# repartition and save cleaned version of data
clean_df = clean_df.repartition(npartitions=NUM_PARTITIONS)

# meta data for later use
initial_shape = dd.compute(clean_df.shape)[0]
num_total_datapts = initial_shape[0]
print("Initial shape =", initial_shape)

# how bad is the nan situation for critical columns? get counts as a percent of total
hgst_nans = utils.get_nan_count_percent(clean_df, num_total_datapts)

# show only values which at least some nans
#  & (hgst_nans['percent'] != 1)
# (hgst_nans['percent'] != 0) &
hgst_nans[(hgst_nans["percent"] != 1)].compute().sort_values(
    by="percent", ascending=False
)

# get columns for which all the values are nans
all_nan_cols = hgst_nans[hgst_nans["percent"] == 1].index.compute()

# sanity check -- make sure the columns identified as all-nans are actually so
clean_df[all_nan_cols[:6]].head()

# drop it like it's hot
clean_df = clean_df.drop(all_nan_cols, axis=1)

# how do things look after this
utils.get_nan_count_percent(clean_df, num_total_datapts).compute().sort_values(
    by="percent", ascending=False
)

# get the data points where 193 is null. inspect it for patterns
nan193_df = clean_df[clean_df["smart_193_raw"].isna()].compute()

# number of nans in this subset as a percentage of nans in the overall data
utils.get_nan_count_percent(nan193_df, num_total_datapts)

# number of failed drives in subset
len(nan193_df[nan193_df["failure"] == 1])

# the serial numbers which are producing nans
nan193_sers = nan193_df["serial_number"].unique()
nan193_sers

# inspect samples one by one to see if there is anything we need to consider before filling nans
clean_df[clean_df["serial_number"] == "PL2331LAHDW5VJ"].compute()

# get the number of rows that contain nan values, for each group
ser_nanpercent = (
    nan193_df[["date", "serial_number", "smart_193_raw"]]
    .groupby("serial_number")
    .apply(lambda group: group["smart_193_raw"].isna().sum() / group.shape[0])
)

# are there any groups (serial_numbers) that have more than 1 nan row?
(ser_nanpercent != 1).any()

# get the days on which nan values occured
nan193_df["date"].value_counts()

# check if there were ANY non-nan values on this doomsday
badday_df = clean_df[clean_df["date"] == "2018-11-17"]

# for each col, true means ALL drives reported nans for this col on this day
is_allnan_col = badday_df.isna().all().compute()

# percentage of cols for whom ALL drives reported nans
is_allnan_col.sum() / is_allnan_col.shape[0]

# did we have any failure cases on this day
(badday_df["failure"] == 1).any().compute()

# are there any entries on this day where smart 22 is not null?
# this is checked because smart 22 is almost always null, so if we find something non-null, it's worth keeping
badday_df[["smart_22_raw", "smart_22_normalized"]].isna().all().compute()

# see what non nan values existed
badday_df[~badday_df["smart_22_raw"].isna()][
    ["date", "serial_number", "model", "capacity_bytes", "failure", "smart_22_raw"]
].compute()

# fill in dummy values
cols_to_fill = list(nan193_df.columns)

# dont want to fill non-smart atributes like date, serial numbers
cols_to_fill = [col for col in cols_to_fill if col.startswith("smart")]

# dont want to fill smart 22
cols_to_fill.remove("smart_22_raw")
cols_to_fill.remove("smart_22_normalized")

# must do it in for loop, dask does not like indexing with list
# plus, its not straightforward to mask isna of specific columns
for col in cols_to_fill:
    if col.startswith("smart"):
        #         # TODO: replace value_counts+max with median, when it is implemented in dask
        #         clean_df[col] = clean_df[col].fillna(value=clean_df[col].value_counts().idxmax())
        clean_df[col] = clean_df[col].ffill()

# how do things look after this
utils.get_nan_count_percent(clean_df, num_total_datapts).compute()

# clean up unused memory
del nan193_df
del nan193_sers
del ser_nanpercent
del badday_df
del is_allnan_col
del cols_to_fill
gc.collect()

# serial numbers of all drives where 22 is reported as non nan at least once
nonnan22_serials = (
    clean_df[~clean_df["smart_22_raw"].isna()]["serial_number"].unique().compute()
)  # len = 2334

# of these serial numbers, which ones report at least one nan as well
isanynan22_serials = clean_df[clean_df["serial_number"].isin(nonnan22_serials)][
    ["date", "serial_number", "smart_22_raw"]
]
isanynan22_serials = (
    isanynan22_serials.groupby("serial_number")
    .apply(lambda g: g["smart_22_raw"].isna().any())
    .compute()
)

# these are the drives that report at least one nan, and are known to be helium drives
# because they have reported non-nan value for smart 22 at least once
helium_nans = isanynan22_serials[isanynan22_serials]
helium_nans

# get detailed data for these drives
cols = ["date", "serial_number", "failure", "smart_22_raw", "smart_22_normalized"]
tmp = clean_df[clean_df["serial_number"].isin(helium_nans.index)][cols].compute()

# on what dates do nan values occur for smart 22 for helium drives
tmp[tmp["smart_22_raw"].isna()]["date"].unique()
print(tmp[tmp["smart_22_raw"].isna()]["date"].unique())


tmp[["serial_number", "smart_22_raw"]].groupby("serial_number").agg(["mean", "std"])

tmp[["serial_number", "smart_22_normalized"]].groupby("serial_number").agg(
    ["mean", "std"]
)


clean_df["smart_22_normalized"].value_counts().compute()

clean_df["smart_22_raw"].value_counts().compute()

clean_df[clean_df["serial_number"].isin(helium_nans.index)][
    "smart_22_raw"
].mean().compute()

(clean_df["date"] == "2018-11-17").sum().compute()

# rows which were recorded on the doomsday, and belong to drives that report non null smart 22
is_helium = clean_df["serial_number"].isin(helium_nans.index)
is_doomsday = clean_df["date"] == "2018-11-17"

# replace with mean values within that group
cols_to_fill = ["smart_22_raw", "smart_22_normalized"]
for col in cols_to_fill:
    clean_df[col] = clean_df[col].mask(
        is_helium & is_doomsday, clean_df[is_helium][col].mean()
    )

# make sure that as of this cell, drives report either all nans or no nans for smart 22
tmp = clean_df[clean_df["serial_number"].isin(nonnan22_serials)][
    ["date", "serial_number", "smart_22_raw"]
]
tmp = tmp.groupby("serial_number").apply(lambda g: g["smart_22_raw"].isna().any())
tmp.any().compute()

# fill the rest of the nan values with dummy values
cols_to_fill = ["smart_22_raw", "smart_22_normalized"]
for col in cols_to_fill:
    clean_df[col] = clean_df[col].fillna(DUMMY_VALUE)

# how do things look after this
utils.get_nan_count_percent(clean_df, num_total_datapts).compute().sort_values(
    by="percent", ascending=False
)

# clean up garbage
del tmp
del nonnan22_serials
del isanynan22_serials
del helium_nans
gc.collect()

# get ideal number of partitions
PARTITION_SIZE_BYTES = 100 * 10 ** 6
DF_SIZE_BYTES = clean_df.memory_usage(deep=True).sum().compute()
NUM_PARTITIONS = int(np.ceil(DF_SIZE_BYTES / PARTITION_SIZE_BYTES))

# repartition and save cleaned version of data
clean_df = clean_df.repartition(npartitions=NUM_PARTITIONS)

# create save dir as same name as data dir, but suffixed with "_clean" and vendor name
head, tail = os.path.split(os.path.normpath(DATA_DIR))
save_dir = os.path.join(head, tail + "_wdc_clean")

# if path doesnt exist, mkdir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# save partitions
clean_df.to_csv(os.path.join(save_dir, "partition_*.csv"), index=False)

# unregister dask compute call progress bar
pbar.unregister()
