import plotly.graph_objects as go
from src.preprocess.residuals import noisy_resids_from_one_resid


def plot_2D_scatter_head_residuals_sim_vs_real(scenarios, head_residuals, head_meas_residuals, leak_locs,
                                               meas_loc_x, meas_loc_y, start_dt_leak, start_time_leak, tdpbc_by_sensor, 
                                               nr_samples_per_leak = 40, show_legend=False):
    
    fig = go.Figure()

    for leak_loc in leak_locs:

        # Extract residuals for measurement locations x and y for leak scenario leak_loc, at datetime
        leak_loc_scenarios_indices = scenarios.index[scenarios['leaks','loc'] == leak_loc]
        residuals_for_one_leak_loc = head_residuals[head_residuals['datetime'] == start_dt_leak].loc[:,leak_loc_scenarios_indices]
        residual_pairs = residuals_for_one_leak_loc.iloc[0, :]
        residual_x = residual_pairs.xs(meas_loc_x, level=1, drop_level=False).values[0]
        residual_y = residual_pairs.xs(meas_loc_y, level=1, drop_level=False).values[0]
        
        meas_residual_x = head_meas_residuals[head_meas_residuals['datetime'] == start_dt_leak][meas_loc_x]
        meas_residual_y = head_meas_residuals[head_meas_residuals['datetime'] == start_dt_leak][meas_loc_y]
        
        tdbpc_residual_x_mean = tdpbc_by_sensor[meas_loc_x]['means'][start_time_leak]
        tdbpc_residual_x_std = tdpbc_by_sensor[meas_loc_x]['stds'][start_time_leak]
        
        tdbpc_residual_y_mean = tdpbc_by_sensor[meas_loc_y]['means'][start_time_leak]
        tdbpc_residual_y_std = tdpbc_by_sensor[meas_loc_y]['stds'][start_time_leak]
        
        meas_residual_x_debiased = meas_residual_x + tdbpc_residual_x_mean
        meas_residual_y_debiased = meas_residual_y + tdbpc_residual_y_mean

        # Make nr_samples_per_leak copies of the residuals, add Gaussian noise with std = noise_lvl
        residuals_x = noisy_resids_from_one_resid(residual_x, tdbpc_residual_x_std,
                                                  nr_of_generated_residuals=nr_samples_per_leak,
                                                  random_seed=int(leak_loc)) # Use leak loc as random seed
        
        residuals_y = noisy_resids_from_one_resid(residual_y, tdbpc_residual_y_std,
                                                  nr_of_generated_residuals=nr_samples_per_leak,
                                                  random_seed=int(leak_loc)+1) # Make sure that random seed differs
        
        # Plot the simulated residual vectors (x,y) as a scatter plot
        fig.add_trace(go.Scatter(x=residuals_x, y=residuals_y,
                                 mode='markers',
                                 name=leak_loc,
                                 opacity=0.8
                                 #text=residuals_x.index.get_level_values(0) #show index of residual on hover
                                 ))
        
    # Plot the real residuals in the scatter plot
    fig.add_trace(go.Scatter(x=meas_residual_x_debiased, y=meas_residual_y_debiased,
                             mode='markers',
                             name="Measured residuals, debiased",
                             marker=dict(size=20),
                             marker_symbol='x',
                             marker_color='black',
                             opacity=0.8
                             #text=residuals_x.index.get_level_values(0) #show index of residual on hover
                             ))

    fig.update_layout(
        title="Datetime: "+str(start_dt_leak),
        width=800,
        height=800,
        xaxis_title="Head residual: " + str(meas_loc_x),
        yaxis_title="Head residual: " + str(meas_loc_y),
        showlegend=show_legend,
        #xaxis_range=[-1.1,0.6],
        #yaxis_range=[-1.8,0.7]
    )
    
    return fig