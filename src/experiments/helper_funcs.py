import numpy as np
from sklearn.metrics import log_loss


def compute_mean_log_loss_over_leak_experiments(leak_locations, X_test_per_leak_exp, trained_models_per_leak_exp,
                                                hydrants_nb_to_label_dict, nr_of_leak_candidates):
    log_loss_values = []
    
    for leak_loc in leak_locations:
        leak_loc_true = leak_loc
        X_real_test = X_test_per_leak_exp[leak_loc_true]
        log_reg_clf = trained_models_per_leak_exp[leak_loc_true]
        y_probs_log_reg_real = log_reg_clf.predict_proba(X_real_test).reshape((1, 1, nr_of_leak_candidates))
        true_leak_loc_idx = hydrants_nb_to_label_dict[leak_loc_true]
        leak_probs = y_probs_log_reg_real.flatten()
        log_loss_value = log_loss([true_leak_loc_idx], [leak_probs], labels=np.arange(nr_of_leak_candidates))
        log_loss_values.append(log_loss_value)
        
    return np.mean(log_loss_values)