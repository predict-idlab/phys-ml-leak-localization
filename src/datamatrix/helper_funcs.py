import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from src.preprocess.helper_funcs import extract_df_datetime_range
from src.preprocess.residuals import noisy_resids_from_one_resid
from src.tdpbc.helper_funcs import convert_df_resampled_means
from src.exceptions import DataMatrixConstructionError


def compute_leak_duration_in_mins(leak_start_datetime, leak_end_datetime):
    leak_duration = pd.to_datetime(leak_end_datetime) - pd.to_datetime(leak_start_datetime)
    leak_duration_in_mins = int(leak_duration.total_seconds() / 60)
    return leak_duration_in_mins

def calc_head_residuals(heads, heads_leakfree, nr_of_reps, datetimes, columns):
    head_residuals = pd.DataFrame(np.tile(heads_leakfree.iloc[:,1:].values, nr_of_reps) - heads.iloc[:,1:].values)
    head_residuals.insert(0, 'datetime', datetimes)
    head_residuals.columns = columns
    return head_residuals

def construct_train_data_matrix_of_time_step(scenarios, head_residuals, p_logs_as_features,
                                            leak_locs, datetime, time, tdpbc_by_sensor, nr_of_samples_per_leak=40):
    """Construct data matrix X for 1 datetime. """
    
    nr_of_features = len(p_logs_as_features)
        
    X = np.zeros(((len(leak_locs), nr_of_samples_per_leak, nr_of_features)))

    for i in range(len(leak_locs)):  
        leak_loc = leak_locs[i]
        
        #extract residuals and leak sizes for one leak location
        leak_loc_scenarios_indices = scenarios.index[scenarios['leaks','loc'] == leak_loc]
        residuals_for_one_leak_loc_tmp = head_residuals[head_residuals['datetime'] == datetime].loc[:,leak_loc_scenarios_indices]
        residuals_for_one_leak_loc = residuals_for_one_leak_loc_tmp.iloc[0, :]
        
        # Construct data matrix for the leak loc, every pressure logger
        # in p_logs_as_features is a feature
        X_of_leak_loc = np.zeros(((nr_of_features, nr_of_samples_per_leak)))
        
        for j, p_log in enumerate(p_logs_as_features):
            residual_p_log = residuals_for_one_leak_loc.xs(p_log, level=1, drop_level=False).values[0]
            
            # Make nr_samples_per_leak copies of the residuals, add Gaussian noise
            tdbpc_residual_std = tdpbc_by_sensor[p_log]['std']
            
            residuals_p_log_noisy = noisy_resids_from_one_resid(residual_p_log, tdbpc_residual_std,
                                                      nr_of_generated_residuals=nr_of_samples_per_leak,
                                                      random_seed=int(leak_loc)+int(p_log))
            X_of_leak_loc[j] = residuals_p_log_noisy
        
        X[i] = X_of_leak_loc.T
        
    X = np.vstack(X)
    return X

def construct_test_data_vector_of_time_step(head_meas_residuals, p_logs_as_features,
                                            datetime, time, tdpbc_by_sensor):
    
    nr_of_features = len(p_logs_as_features)
        
    X = np.zeros(((1, 1, nr_of_features)))
        
    #extract residuals and leak sizes for one leak location
    residuals_for_one_leak_loc_tmp = head_meas_residuals[head_meas_residuals['datetime'] == datetime]
    residuals_for_one_leak_loc = residuals_for_one_leak_loc_tmp.iloc[0, :]

    # Construct data matrix for the leak loc, every pressure logger
    # in p_logs_as_features is a feature
    X_of_leak_loc = np.zeros(((nr_of_features, 1)))

    for j, p_log in enumerate(p_logs_as_features):
        residual_p_log = residuals_for_one_leak_loc[p_log]
        tdbpc_residual_mean = tdpbc_by_sensor[p_log]['mean']
        residual_p_log_unbiased = residual_p_log - tdbpc_residual_mean

        X_of_leak_loc[j] = residual_p_log_unbiased

    X[0] = X_of_leak_loc.T
        
    X = np.vstack(X)
    return X

def construct_train_and_test_matrix_for_leak_exp(leak_location,
                                                 leak_start_datetime,
                                                 leak_end_datetime,
                                                 head_meas,
                                                 head_res,
                                                 head_res_leakfree,
                                                 tdpbc_per_exp_leak_location,
                                                 scenarios,
                                                 p_logs_as_features,
                                                 all_leak_locs,
                                                 nr_of_train_samples_per_leak):
    
    if (head_meas['datetime'] == head_res_leakfree['datetime']).all() == False:
        error_message = "Datetime column of head measurements and leakfree head simulations must match exactly."
        raise DataMatrixConstructionError(error_message)
    
    # Extract head simulations and measurements during the leak experiment
    head_res_leakfree_rs = extract_df_datetime_range(head_res_leakfree, start_date=leak_start_datetime,end_date=leak_end_datetime)
    head_res_rs = extract_df_datetime_range(head_res, start_date=leak_start_datetime, end_date=leak_end_datetime)
    head_meas_rs = extract_df_datetime_range(head_meas, start_date=leak_start_datetime, end_date=leak_end_datetime)
    
    # Calculate duration of the leak
    leak_duration_in_mins = compute_leak_duration_in_mins(leak_start_datetime, leak_end_datetime)
    
    # Calculate leak start time
    leak_start_time = str(pd.to_datetime(leak_start_datetime).time())

    # Resample heads: calculate mean head value over the leak duration
    resampling_rule = str(leak_duration_in_mins) + 'Min'
    leak_start_time_minutes = int(pd.DatetimeIndex([leak_start_datetime])[0].strftime("%M"))
    leak_offset_in_mins = leak_start_time_minutes % leak_duration_in_mins
    offset = str(leak_offset_in_mins) + "Min"
    head_res_leakfree_rs = convert_df_resampled_means(head_res_leakfree_rs, resampling_rule=resampling_rule, offset=offset)
    head_res_rs = convert_df_resampled_means(head_res_rs, resampling_rule=resampling_rule, offset=offset)
    head_meas_rs = convert_df_resampled_means(head_meas_rs, resampling_rule=resampling_rule, offset=offset)
    
    # Calculate simulated and real residuals
    head_residuals = calc_head_residuals(head_res_rs, head_res_leakfree_rs, len(all_leak_locs),
                                         head_meas_rs['datetime'], head_res_rs.columns)
    head_meas_residuals = calc_head_residuals(head_meas_rs, head_res_leakfree_rs, 1,
                                              head_meas_rs['datetime'], head_meas_rs.columns)
    
    # Calculate data matrix for training
    X = construct_train_data_matrix_of_time_step(scenarios=scenarios,
                                                 head_residuals=head_residuals,
                                                 p_logs_as_features=p_logs_as_features,
                                                 leak_locs=all_leak_locs,
                                                 datetime=leak_start_datetime,
                                                 time=leak_start_time,
                                                 tdpbc_by_sensor=tdpbc_per_exp_leak_location[leak_location],
                                                 nr_of_samples_per_leak=nr_of_train_samples_per_leak)
    
    # Calculate test vector of leak experiment
    X_real_test = construct_test_data_vector_of_time_step(head_meas_residuals=head_meas_residuals,
                                                          p_logs_as_features=p_logs_as_features,
                                                          datetime=leak_start_datetime,
                                                          time=leak_start_time,
                                                          tdpbc_by_sensor=tdpbc_per_exp_leak_location[leak_location])
    
    # Apply feature standardisation
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    X_real_test_scaled = scaler.transform(X_real_test)
    
    return X_scaled, X_real_test_scaled


def construct_train_matrix_for_bin(leak_start_datetime,
                                   leak_end_datetime,
                                   head_meas,
                                   head_res,
                                   head_res_leakfree,
                                   tdpbc_per_bin,
                                   bin_nr,
                                   scenarios,
                                   p_logs_as_features,
                                   all_leak_locs,
                                   nr_of_train_samples_per_leak):
    
    if (head_meas['datetime'] == head_res_leakfree['datetime']).all() == False:
        error_message = "Datetime column of head measurements and leakfree head simulations must match exactly."
        raise DataMatrixConstructionError(error_message)
    
    # Extract head simulations and measurements during the leak experiment
    head_res_leakfree_rs = extract_df_datetime_range(head_res_leakfree, start_date=leak_start_datetime,end_date=leak_end_datetime)
    head_res_rs = extract_df_datetime_range(head_res, start_date=leak_start_datetime, end_date=leak_end_datetime)
    head_meas_rs = extract_df_datetime_range(head_meas, start_date=leak_start_datetime, end_date=leak_end_datetime)
    
    # Calculate duration of the leak
    leak_duration_in_mins = compute_leak_duration_in_mins(leak_start_datetime, leak_end_datetime)
    
    # Calculate leak start time
    leak_start_time = str(pd.to_datetime(leak_start_datetime).time())

    # Resample heads: calculate mean head value over the leak duration
    resampling_rule = str(leak_duration_in_mins) + 'Min'
    leak_start_time_minutes = int(pd.DatetimeIndex([leak_start_datetime])[0].strftime("%M"))
    leak_offset_in_mins = leak_start_time_minutes % leak_duration_in_mins
    offset = str(leak_offset_in_mins) + "Min"
    head_res_leakfree_rs = convert_df_resampled_means(head_res_leakfree_rs, resampling_rule=resampling_rule, offset=offset)
    head_res_rs = convert_df_resampled_means(head_res_rs, resampling_rule=resampling_rule, offset=offset)
    head_meas_rs = convert_df_resampled_means(head_meas_rs, resampling_rule=resampling_rule, offset=offset)
    
    # Calculate simulated head residuals
    head_residuals = calc_head_residuals(head_res_rs, head_res_leakfree_rs, len(all_leak_locs),
                                         head_meas_rs['datetime'], head_res_rs.columns)
    
    # Calculate data matrix for training
    X = construct_train_data_matrix_of_time_step(scenarios=scenarios,
                                                 head_residuals=head_residuals,
                                                 p_logs_as_features=p_logs_as_features,
                                                 leak_locs=all_leak_locs,
                                                 datetime=leak_start_datetime,
                                                 time=leak_start_time,
                                                 tdpbc_by_sensor=tdpbc_per_bin[bin_nr],
                                                 nr_of_samples_per_leak=nr_of_train_samples_per_leak)
    
    # Apply feature standardisation
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    
    return X_scaled, scaler


def construct_test_matrix_for_bin(leak_start_datetime,
                                  leak_end_datetime,
                                  head_meas,
                                  head_res_leakfree,
                                  tdpbc_per_bin,
                                  bin_nr,
                                  p_logs_as_features,
                                  scaler):
    
    if (head_meas['datetime'] == head_res_leakfree['datetime']).all() == False:
        error_message = "Datetime column of head measurements and leakfree head simulations must match exactly."
        raise DataMatrixConstructionError(error_message)
    
    # Extract head simulations and measurements during the leak experiment
    head_res_leakfree_rs = extract_df_datetime_range(head_res_leakfree, start_date=leak_start_datetime,end_date=leak_end_datetime)
    head_meas_rs = extract_df_datetime_range(head_meas, start_date=leak_start_datetime, end_date=leak_end_datetime)
    
    # Calculate duration of the leak
    leak_duration_in_mins = compute_leak_duration_in_mins(leak_start_datetime, leak_end_datetime)
    
    # Calculate leak start time
    leak_start_time = str(pd.to_datetime(leak_start_datetime).time())

    # Resample heads: calculate mean head value over the leak duration
    resampling_rule = str(leak_duration_in_mins) + 'Min'
    leak_start_time_minutes = int(pd.DatetimeIndex([leak_start_datetime])[0].strftime("%M"))
    leak_offset_in_mins = leak_start_time_minutes % leak_duration_in_mins
    offset = str(leak_offset_in_mins) + "Min"
    head_res_leakfree_rs = convert_df_resampled_means(head_res_leakfree_rs, resampling_rule=resampling_rule, offset=offset)
    head_meas_rs = convert_df_resampled_means(head_meas_rs, resampling_rule=resampling_rule, offset=offset)
    
    # Calculate simulated and real residuals
    head_meas_residuals = calc_head_residuals(head_meas_rs, head_res_leakfree_rs, 1,
                                              head_meas_rs['datetime'], head_meas_rs.columns)
    
    # Calculate test vector of leak experiment
    X_real_test = construct_test_data_vector_of_time_step(head_meas_residuals=head_meas_residuals,
                                                          p_logs_as_features=p_logs_as_features,
                                                          datetime=leak_start_datetime,
                                                          time=leak_start_time,
                                                          tdpbc_by_sensor=tdpbc_per_bin[bin_nr])
    
    # Apply feature standardisation using scaler (obtained during training)
    X_real_test_scaled = scaler.transform(X_real_test)
    
    return X_real_test_scaled


def construct_ungrouped_leak_labels(leak_locs, nr_of_samples_per_leak=40):
    """Construct the target vector. Similar leak locs are not grouped. """
    
    y_ungrouped = [] #TODO: purely in numpy
    for i in range(len(leak_locs)):
        leak_loc = leak_locs[i]
        y_of_leak_loc = [leak_loc]*nr_of_samples_per_leak
        y_ungrouped.extend(y_of_leak_loc)
        
    return np.asarray(y_ungrouped)