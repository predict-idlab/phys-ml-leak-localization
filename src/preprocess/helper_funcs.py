from src.exceptions import PressureToHeadConversionError
from src.preprocess.config import BAR_TO_MWC_UNITS_CONVERSION_FACTOR

import numpy as np
import pandas as pd


def extract_df_datetime_range(df, start_date, end_date):
    mask = (df['datetime'] >= start_date) & (df['datetime'] < end_date)
    df_datetime_range = df.loc[mask]
    df_datetime_range.reset_index(drop=True, inplace=True)
    return df_datetime_range

def repeat_rows(df, n):
    return df.loc[df.index.repeat(n)]

def insert_timestamps_in_df(df, timestamps, column_name='datetime'):
    if column_name not in df.columns:
        df.insert(0, column_name, timestamps)
    return df

def replace_heads_with_nan_over_dt_range(df, replace_start, replace_end, pressure_loggers):
    
    datetimes_to_replace = pd.date_range(start=replace_start, end=replace_end, freq='min')

    for datetime in datetimes_to_replace:
        for logger in pressure_loggers:
            df.loc[df['datetime'] == datetime, (0,logger)] = np.nan    
    return df

def convert_df_to_hourly_averages(df):
    tmp = df.set_index(['datetime']).copy()
    tmp = tmp.resample('H').mean()
    return tmp.reset_index()

def convert_pressures_to_heads(pressures_df, pressure_loggers, wn, conv_factor=BAR_TO_MWC_UNITS_CONVERSION_FACTOR):
    """
    Convert dataframe containing pressure measurements (bar units) to heads (mwc units).
    
    Parameters
    ---------
    pressures_df: pd.DataFrame
        Pandas dataframe containing the pressures for each pressure logger per column.
        The asset id of the pressure logger is indicated in each column name.
    pressure_loggers: list
        List of the asset id's of the pressure loggers to be converted.
    wn: wntr.Model
        wn model describing the network in which the pressure measurements were taken.
    conv_factor: float
        Conversion factor used to convert from bar to mwc units.
    
    Returns
    --------
    heads_df: pd.DataFrame
        Converted heads in mwc units.
    """
    heads_df = pressures_df.copy()
    
    head_meas_cols = list(heads_df.columns)
    if 'datetime' in head_meas_cols:
        head_meas_cols.remove('datetime')
    if set(head_meas_cols) != set(pressure_loggers):
        raise PressureToHeadConversionError("All pressure loggers present in the dataframe have to be converted.")
    
    for asset_id in pressure_loggers:
        pressure_sensor_node_elevation = wn.get_node(asset_id).elevation
        heads_df[asset_id] = pressures_df[asset_id] * conv_factor + pressure_sensor_node_elevation   
    return heads_df

def preprocess_exp_leaks_info(exp_leaks_info, round_leak_datetimes_5min):
    exp_leaks_info['start_leak_datetime'] = pd.to_datetime(exp_leaks_info['Datum'] +' ' + exp_leaks_info['Beginuur'],
                                                       format="%d/%m/%Y %H:%M") 
    exp_leaks_info['end_leak_datetime'] = pd.to_datetime(exp_leaks_info['Datum'] +' ' + exp_leaks_info['Einduur'],
                                                       format="%d/%m/%Y %H:%M")
    exp_leaks_info.drop(['Datum','Beginuur','Einduur'], axis='columns', inplace=True)
    
    if round_leak_datetimes_5min:
        exp_leaks_info['start_leak_datetime'] = exp_leaks_info['start_leak_datetime'].dt.ceil(freq='5Min')
        exp_leaks_info['end_leak_datetime'] = exp_leaks_info['end_leak_datetime'].dt.floor(freq='5Min')
    
    return exp_leaks_info

def create_temporal_bins(start_datetime, end_datetime, freq_mins):
    
    bins_df = pd.DataFrame()
    freq = str(freq_mins) + "Min"
    total_nr_of_bins = len(pd.date_range(start=start_datetime, end=end_datetime, freq=freq))
    bins_df['bin_nr'] = np.arange(total_nr_of_bins)
    bins_df['start_leak_datetime'] = pd.date_range(start=start_datetime, end=end_datetime, freq=freq)
    bins_df['end_leak_datetime'] = bins_df['start_leak_datetime'] + pd.DateOffset(minutes=freq_mins)
    
    return bins_df

def preprocess_pressure_data_json(pressure_data_json):
    pressure_data_df = pd.DataFrame(pressure_data_json)
    pressure_data_df['Pressure'] = pd.to_numeric(pressure_data_df['Pressure'])
    pressure_data_df = pressure_data_df.pivot_table(index='Timestamp', columns='brandkraan_meetpunt', values='Pressure')
    pressure_data_df.insert(0, 'datetime', pressure_data_df.index)
    pressure_data_df = pressure_data_df.reset_index(drop=True)
    
    return pressure_data_df