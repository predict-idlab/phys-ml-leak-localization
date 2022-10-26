import pandas as pd
from src.tdpbc.helper_funcs import calc_tdpbc_for_daytime_range


def test_tdpbc_calculation_no_weighting(head_meas_dummy, head_res_leakfree_dummy):
    # Given
    start_time = "19:00:00"
    end_time = "19:20:00"
    weekdays_to_include = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    
    # When
    (tdpbc_mean, tdpbc_std)= calc_tdpbc_for_daytime_range(head_meas_dummy, head_res_leakfree_dummy,
                                                          '42',
                                                          weekdays_to_include,
                                                          "2020-08-03 00:00:00", "2020-08-05 00:00:00",
                                                          start_time, end_time,
                                                          lin_weighting=False)
    
    # Then
    assert int(tdpbc_mean) == 150
    assert int(tdpbc_std) == 50
    
def test_tdpbc_calculation_linear_weighting(head_meas_dummy, head_res_leakfree_dummy):
    # Given
    start_time = "19:00:00"
    end_time = "19:20:00"
    weekdays_to_include = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    
    # When
    (tdpbc_mean, tdpbc_std)= calc_tdpbc_for_daytime_range(head_meas_dummy, head_res_leakfree_dummy,
                                                          '42',
                                                          weekdays_to_include,
                                                          "2020-08-03 00:00:00", "2020-08-05 00:00:00",
                                                          start_time, end_time,
                                                          lin_weighting=True)
    
    # Then
    assert int(tdpbc_mean) == 133
    assert int(tdpbc_std) == 47