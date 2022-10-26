import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.preprocess.helper_funcs import (extract_df_datetime_range,
                                         convert_pressures_to_heads,
                                         repeat_rows,
                                         insert_timestamps_in_df,
                                         replace_heads_with_nan_over_dt_range,
                                         convert_df_to_hourly_averages)

from src.preprocess.config import P_LOGGER_ALWAYS_NAN_FOR_MC_2, SIMS_TIMESTEP_IN_MINUTES


class HeadSimsLeakyPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, sims_start, sims_end, sims_extract_start, sims_extract_end,
                 delete_last_row=True):
            self.sims_start = sims_start
            self.sims_end = sims_end
            self.sims_extract_start = sims_extract_start
            self.sims_extract_end = sims_extract_end
            self._delete_last_row = delete_last_row

    def fit(self, X, y=None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        
        # Delete last row
        if self._delete_last_row :
            X = X[:-1]
        
        # Reset index
        X = X.reset_index(drop=True)
        
        # Set timestamps
        sims_period_timestamps = pd.date_range(start=self.sims_start, end=self.sims_end,
                                               closed='left', freq=str(SIMS_TIMESTEP_IN_MINUTES)+'min')
        X = insert_timestamps_in_df(X, sims_period_timestamps)
        
        # Extract a datetime range
        X = extract_df_datetime_range(X,
                                      start_date=self.sims_extract_start,
                                      end_date=self.sims_extract_end)
        return X

    
class HeadSimsLeakFreePreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, sims_start, sims_end, extreme_high_p_sim_start=None, extreme_high_p_sim_end=None,
                 extreme_high_p_loggers=None):
            self.sims_start = sims_start
            self.sims_end = sims_end
            self.extreme_high_p_sim_start = extreme_high_p_sim_start
            self.extreme_high_p_sim_end = extreme_high_p_sim_end
            self.extreme_high_p_loggers = extreme_high_p_loggers

    def fit(self, X, y=None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        
        # Delete last row
        #X = X[:-1]
        
        # Reset index
        X = X.reset_index(drop=True)
        
        # Set timestamps
        sims_period_timestamps = pd.date_range(start=self.sims_start, end=self.sims_end,
                                               closed='left', freq=str(SIMS_TIMESTEP_IN_MINUTES)+'min')
        X = insert_timestamps_in_df(X, sims_period_timestamps)
        
        # Replace extremely high pressure head values with NaN
        if self.extreme_high_p_sim_start:
            X = replace_heads_with_nan_over_dt_range(X,
                                                     self.extreme_high_p_sim_start,
                                                     self.extreme_high_p_sim_end,
                                                     self.extreme_high_p_loggers)
        return X

    
class PressureMeasurementsPreprocessor(BaseEstimator, TransformerMixin):
    """Converts pressures to heads."""
    
    def __init__(self, sims_start, sims_end, pressure_loggers_to_convert, wn,
                 shift_one_hour=True, shift_n_hours=None, drop_nan_p_logger=True):
            self.sims_start = sims_start
            self.sims_end = sims_end
            self.pressure_loggers_to_convert = pressure_loggers_to_convert
            self.wn = wn
            self._shift_one_hour = shift_one_hour
            self._shift_n_hours = shift_n_hours
            self._drop_nan_p_logger = drop_nan_p_logger
            

    def fit(self, X, y=None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        
        X['datetime'] = pd.to_datetime(X['datetime'])
        
        # Shift datetime with 1 hour to correct for timezone
        if self._shift_one_hour:
            X['datetime'] = pd.Index(X['datetime']).shift(1, freq='H')
            
        # Shift datetime with n hours to correct for timezone
        if self._shift_n_hours:
            X['datetime'] = pd.Index(X['datetime']).shift(self._shift_n_hours, freq='H')
        
        # Set type of pressure columns to float
        for column in X.columns[1:]:
            X[column] = X[column].astype(float)
        
        # Drop pressure sensor that gives nan values only
        if self._drop_nan_p_logger:
            if P_LOGGER_ALWAYS_NAN_FOR_MC_2 in X.columns:
                X = X.drop(columns=[P_LOGGER_ALWAYS_NAN_FOR_MC_2])
            
        X = extract_df_datetime_range(df=X, start_date=self.sims_start, end_date=self.sims_end)
        
        # Convert from pressure to heads (pressure head + elevation head in mwc units)
        X = convert_pressures_to_heads(X, self.pressure_loggers_to_convert, self.wn)
        
        # Sample every n minutes
        X = X.iloc[::SIMS_TIMESTEP_IN_MINUTES, :]
        X = X.reset_index(drop=True)
        
        return X