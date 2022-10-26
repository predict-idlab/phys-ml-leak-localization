import pytest
import pandas as pd


# Represents head measurements with sample rate of 5 min. over 2 days
@pytest.fixture
def head_meas_dummy():
    head_meas_dummy = pd.DataFrame()
    head_meas_dummy['datetime'] = pd.date_range(start="2020-08-03 00:00:00",
                                                end="2020-08-05 00:00:00",
                                                freq='5Min')
    head_meas_dummy['42'] = 100
    head_meas_dummy.loc[(head_meas_dummy['datetime'] < "2020-08-05 00:00:00") &
                        (head_meas_dummy['datetime'] >= "2020-08-04 00:00:00"), '42'] = 200
    return head_meas_dummy


# Represents leakfree head simulation with sample rate of 5 min. over 2 days
@pytest.fixture
def head_res_leakfree_dummy():
    head_res_leakfree_dummy = pd.DataFrame()
    head_res_leakfree_dummy['datetime'] = pd.date_range(start="2020-08-03 00:00:00",
                                                end="2020-08-05 00:00:00",
                                                freq='5Min')
    head_res_leakfree_dummy['42'] = 300
    head_res_leakfree_dummy.columns = pd.MultiIndex.from_tuples([('datetime',''),(0, '42')])
    return head_res_leakfree_dummy