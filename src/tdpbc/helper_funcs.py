import math

import numpy as np
import pandas as pd

from src.preprocess.helper_funcs import extract_df_datetime_range


def convert_df_resampled_means(df, resampling_rule, offset, origin='start'):
    tmp = df.set_index(['datetime']).copy()
    #TODO: do something with offset
    tmp = tmp.resample(resampling_rule, origin=origin).mean()
    return tmp.reset_index()

def set_heads_of_dt_range_to_nan(df, start_datetime, end_datetime, columns):
    df_copy = df.copy()
    mask = (df_copy['datetime'] >= start_datetime) & (df_copy['datetime'] < end_datetime)
    for col in columns:
        df_copy.loc[mask, col] = np.NaN
    return df_copy

def calc_tdpbc_for_daytime_range(head_meas, head_res_leakfree, pressure_logger,
                                 weekdays_to_include, start_date, end_date,
                                 start_daytime, end_daytime, lin_weighting):
    
    def _weighted_avg_and_std(values, weights):
        """
        Return the weighted average and standard deviation.

        values, weights -- Numpy ndarrays with the same shape.
        """
        average = np.average(values, weights=weights)
        # Fast and numerically precise:
        variance = np.average((values-average)**2, weights=weights)
        return (average, math.sqrt(variance))

    # Calculate head diff from start_date to end_date
    head_diff = head_res_leakfree.reset_index(drop=True)[(0, pressure_logger)] - head_meas.reset_index(drop=True)[pressure_logger]
    head_diff = pd.DataFrame(head_diff)
    head_diff['datetime'] = head_meas.reset_index(drop=True)['datetime']
    head_diff_dt_range = extract_df_datetime_range(head_diff, start_date, end_date)

    # Select the daytime range only (e.g. 20 mins from 22:33 to 22:52)
    head_diff_dt_range.set_index('datetime', inplace=True)
    head_diff_dt_range = head_diff_dt_range.between_time(start_daytime, end_daytime, include_end=True)

    # Filter on which weekdays to be included in the TDPBC calculation
    head_diff_dt_range.insert(1, 'weekday', head_diff_dt_range.index.day_name())
    head_diff_dt_range = head_diff_dt_range[head_diff_dt_range['weekday'].isin(weekdays_to_include)]
    head_diff_dt_range.drop('weekday', axis='columns', inplace=True)

    # Calculate mean per daytime range
    # (e.g. mean of values from 22:33 to 22:52, for every day)
    bias_corr_per_day = head_diff_dt_range.groupby([head_diff_dt_range.index.dayofyear]).mean()
    bias_corr_per_day = bias_corr_per_day.values.flatten()
    
    if lin_weighting:
        weights_unnormalized = np.arange(1, len(bias_corr_per_day)+1)
        weights = weights_unnormalized/np.sum(weights_unnormalized)
        bias_corr_mean, bias_corr_std = _weighted_avg_and_std(bias_corr_per_day, weights)    
        return bias_corr_mean, bias_corr_std
    else:
        # ignore nans if they occur
        return np.nanmean(bias_corr_per_day), np.nanstd(bias_corr_per_day)