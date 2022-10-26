import numpy as np
import pandas as pd

from src.datamatrix.helper_funcs import construct_train_and_test_matrix_for_leak_exp


def test_construct_train_and_test_matrix_for_leak_exp():
    # Given
    leak_location = '775125'
    leak_start_datetime = '2020-08-05 09:35:00'
    leak_end_datetime = '2020-08-05 09:55:00'
    pressure_logger_id_1 = '42'
    pressure_logger_id_2 = '43'
    leak_loc_simulated_1 = '9000'
    leak_loc_simulated_2 = '9001'

    head_meas_dummy = pd.DataFrame()
    head_meas_dummy['datetime'] = pd.date_range(start="2020-08-04 00:00:00",
                                                end="2020-08-07 00:00:00",
                                                freq='5Min')
    head_meas_dummy[pressure_logger_id_1] = 90.0
    head_meas_dummy[pressure_logger_id_2] = 140.0

    head_res_leakfree_dummy = pd.DataFrame()
    head_res_leakfree_dummy['datetime'] = pd.date_range(start="2020-08-04 00:00:00",
                                                end="2020-08-07 00:00:00",
                                                freq='5Min')
    head_res_leakfree_dummy[pressure_logger_id_1] = 100.0
    head_res_leakfree_dummy.columns = pd.MultiIndex.from_tuples([('datetime',''),(0, pressure_logger_id_1)])
    head_res_leakfree_dummy[(0, pressure_logger_id_2)] = 150.0

    head_res_dummy = pd.DataFrame()
    head_res_dummy['datetime'] = pd.date_range(start="2020-08-05 00:00:00",
                                                end="2020-08-06 00:00:00",
                                                freq='5Min')
    head_res_dummy[pressure_logger_id_1] = 80.0
    head_res_dummy.columns = pd.MultiIndex.from_tuples([('datetime',''),(0, pressure_logger_id_1)])
    head_res_dummy[(0, pressure_logger_id_1)] = 80.0
    head_res_dummy[(0, pressure_logger_id_2)] = 130.0
    head_res_dummy[(1, pressure_logger_id_1)] = 60.0
    head_res_dummy[(1, pressure_logger_id_2)] = 110.0

    tdpbc_per_exp_leak_location_dummy = {}
    tdpbc_per_exp_leak_location_dummy[leak_location] = {}
    tdpbc_per_exp_leak_location_dummy[leak_location][pressure_logger_id_1] = {}
    tdpbc_per_exp_leak_location_dummy[leak_location][pressure_logger_id_2] = {}

    tdpbc_per_exp_leak_location_dummy[leak_location][pressure_logger_id_1]['mean'] = 5.0
    tdpbc_per_exp_leak_location_dummy[leak_location][pressure_logger_id_1]['std'] = 0.0 # no TDPBC uncertainty
    tdpbc_per_exp_leak_location_dummy[leak_location][pressure_logger_id_2]['mean'] = 5.0
    tdpbc_per_exp_leak_location_dummy[leak_location][pressure_logger_id_2]['std'] = 0.0 # no TDPBC uncertainty

    scenarios_dummy = pd.DataFrame()

    scenarios_dummy['leaks_loc'] = [leak_loc_simulated_1, leak_loc_simulated_2]
    scenarios_dummy.columns = pd.MultiIndex.from_tuples([('leaks', 'loc')])

    p_logs_as_features_dummy = [pressure_logger_id_1, pressure_logger_id_2]
    all_leak_locs_dummy = [leak_loc_simulated_1, leak_loc_simulated_2]
    
    # When
    X, X_real_test = construct_train_and_test_matrix_for_leak_exp(leak_location=leak_location,
                                                              leak_start_datetime=leak_start_datetime,
                                                              leak_end_datetime=leak_end_datetime,
                                                              head_meas=head_meas_dummy,
                                                              head_res=head_res_dummy,
                                                              head_res_leakfree=head_res_leakfree_dummy,
                                                              tdpbc_per_exp_leak_location=tdpbc_per_exp_leak_location_dummy,
                                                              scenarios=scenarios_dummy,
                                                              p_logs_as_features=p_logs_as_features_dummy,
                                                              all_leak_locs=all_leak_locs_dummy,
                                                              nr_of_train_samples_per_leak=3)
    # Then
    assert np.array_equal(X, np.array([[-1., -1.],
                                       [-1., -1.],
                                       [-1., -1.],
                                       [ 1.,  1.],
                                       [ 1.,  1.],
                                       [ 1.,  1.]]))
    assert np.array_equal(X_real_test, np.array([[-2.5, -2.5]]))