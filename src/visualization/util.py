import numpy as np

def normalized_abs_log_value(value, min_value, max_value, roundoff=1e-3):
    value_log_abs = np.log(np.abs(value)+roundoff)

    min_value_log_abs = np.log(np.abs(min_value)+roundoff)
    max_value_log_abs = np.log(np.abs(max_value)+roundoff)

    return (value_log_abs - min_value_log_abs)/(max_value_log_abs - min_value_log_abs)

def normalized_value(value, max_value):

    return value/max_value