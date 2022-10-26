import numpy as np
import pandas as pd

import plotly.graph_objects as go


def scenario_indices_of_leak_loc(leak_loc, scenarios):
    leak_loc_scenarios_indices = scenarios.index[scenarios['leaks','loc'] == leak_loc]
    return list(leak_loc_scenarios_indices)

def plot_heads_comparison_with_leak(leak_loc, scenarios, head_results, head_results_leakfree,
                                    pressure_loggers_to_plot, 
                                    leak_loc_in_experiment=False, exp_leaks_info=None, head_measurements=None, override=False):
    """Plot the timeseries of the heads of the leakfree scenario vs. a leaky scenario at leak_loc.
       The heads of a leak scenario that was measured in an in-field experiment,
       can also plotted be next to the simulations."""

    scenario_index = scenario_indices_of_leak_loc(leak_loc, scenarios)[0]

    fig = go.Figure()

    # Time series to plot
    for pressure_logger in pressure_loggers_to_plot:
        # Plot head measurement for this pressure logger
        if leak_loc_in_experiment or override:
            fig.add_trace(go.Scatter(x=head_measurements['datetime'],
                                     y=head_measurements[pressure_logger],
                                     name='measurement ' + str(pressure_logger),
                                     marker_color='grey'))
        # Plot leakfree head simulation for this pressure logger
        fig.add_trace(go.Scatter(x=head_results_leakfree['datetime'],
                                 y=head_results_leakfree[(0, pressure_logger)],
                                 name='leak free sim. ' + str(pressure_logger),
                                 marker_color='blue'))
        # Plot head simulation with leak simulated at leak_loc for this pressure logger
        fig.add_trace(go.Scatter(x=head_results['datetime'],
                         y=head_results[(scenario_index, pressure_logger)],
                         name='leak sim. ' + str(pressure_logger),
                         marker_color='red')) # hide trace by default

    # Indicate time period of experimental leak
    if leak_loc_in_experiment:
        start_datetime_of_leak = exp_leaks_info[exp_leaks_info['3GE FID'] == leak_loc]['start_leak_datetime'].values[0]
        end_datetime_of_leak = exp_leaks_info[exp_leaks_info['3GE FID'] == leak_loc]['end_leak_datetime'].values[0]

        fig.update_layout(
            shapes=[
                dict(
                    type="rect",
                    xref="x", # x-reference is assigned to the x-values
                    yref="paper", # y-reference is assigned to the plot paper [0,1]
                    x0=str(start_datetime_of_leak), y0=0,
                    x1=str(end_datetime_of_leak), y1=1,
                    fillcolor="LightSalmon", opacity=0.3, layer="below", line_width=0,
                ),
            ]
        )

    fig.update_layout(
        title="Leak at " + str(leak_loc), autosize=False, width=1100, height=700,
    )

    # Add rangeslider
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1 day",
                         step="day",
                         stepmode="backward"),
                    dict(count=5,
                         label="5 hours",
                         step="hour",
                         stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    fig.layout.yaxis.fixedrange = False
    fig.update_yaxes(title_text="Head (m)")

    return fig


def plot_heads_comparison_with_leak_compare_old(leak_loc, scenarios, head_results, head_results_leakfree,
                                    pressure_loggers_to_plot, 
                                    leak_loc_in_experiment=False, exp_leaks_info=None, head_measurements=None, head_measurements_old=None,override=False):
    """Plot the timeseries of the heads of the leakfree scenario vs. a leaky scenario at leak_loc.
       The heads of a leak scenario that was measured in an in-field experiment,
       can also plotted be next to the simulations."""

    scenario_index = scenario_indices_of_leak_loc(leak_loc, scenarios)[0]

    fig = go.Figure()

    # Time series to plot
    for pressure_logger in pressure_loggers_to_plot:
        # Plot head measurement for this pressure logger
        if leak_loc_in_experiment or override:
            fig.add_trace(go.Scatter(x=head_measurements['datetime'],
                                     y=head_measurements[pressure_logger],
                                     name='measurement ' + str(pressure_logger),
                                     marker_color='grey'))
        if leak_loc_in_experiment or override:
            if pressure_logger in list(head_measurements_old.columns):
                fig.add_trace(go.Scatter(x=head_measurements_old['datetime'],
                                         y=head_measurements_old[pressure_logger],
                                         name='measurement old' + str(pressure_logger),
                                         marker_color='black'))
        # Plot leakfree head simulation for this pressure logger
        fig.add_trace(go.Scatter(x=head_results_leakfree['datetime'],
                                 y=head_results_leakfree[(0, pressure_logger)],
                                 name='leak free sim. ' + str(pressure_logger),
                                 marker_color='blue'))
        # Plot head simulation with leak simulated at leak_loc for this pressure logger
        fig.add_trace(go.Scatter(x=head_results['datetime'],
                         y=head_results[(scenario_index, pressure_logger)],
                         name='leak sim. ' + str(pressure_logger),
                         marker_color='red')) # hide trace by default

    # Indicate time period of experimental leak
    if leak_loc_in_experiment:
        start_datetime_of_leak = exp_leaks_info[exp_leaks_info['3GE FID'] == leak_loc]['start_leak_datetime'].values[0]
        end_datetime_of_leak = exp_leaks_info[exp_leaks_info['3GE FID'] == leak_loc]['end_leak_datetime'].values[0]

        fig.update_layout(
            shapes=[
                dict(
                    type="rect",
                    xref="x", # x-reference is assigned to the x-values
                    yref="paper", # y-reference is assigned to the plot paper [0,1]
                    x0=str(start_datetime_of_leak), y0=0,
                    x1=str(end_datetime_of_leak), y1=1,
                    fillcolor="LightSalmon", opacity=0.3, layer="below", line_width=0,
                ),
            ]
        )

    fig.update_layout(
        title="Leak at " + str(leak_loc), autosize=False, width=1100, height=700,
    )

    # Add rangeslider
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1 day",
                         step="day",
                         stepmode="backward"),
                    dict(count=5,
                         label="5 hours",
                         step="hour",
                         stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    fig.layout.yaxis.fixedrange = False
    fig.update_yaxes(title_text="Head (m)")

    return fig
    
def plot_resampled_means_head_results_and_meas(pressure_loggers_to_plot,
                                               exp_leaks_info,
                                               head_results_leakfree,
                                               head_measurements):

    fig = go.Figure()
    
    for pressure_logger in pressure_loggers_to_plot:
        
        # Plot leakfree head simulation for this pressure logger
        fig.add_trace(go.Scatter(x=head_results_leakfree['datetime'],
                                 y=head_results_leakfree[(0, pressure_logger)],
                                 name='leak free sim. ' + str(pressure_logger),
                                 marker_color='blue',
                                 line_shape='hv'))
        
        fig.add_trace(go.Scatter(x=head_measurements['datetime'],
                                 y=head_measurements[pressure_logger],
                                 name='measurements ' + str(pressure_logger),
                                 marker_color='grey',
                                 line_shape='hv'))
        
        fig.add_trace(go.Scatter(x=head_measurements['datetime'],
                                 y=head_results_leakfree[(0, pressure_logger)] - head_measurements[pressure_logger],
                                 name='difference ' + str(pressure_logger),
                                 marker_color='green',
                                 line_shape='hv'))

    fig.update_layout(
        title="Title", autosize=False, width=1100, height=700,
    )

    fig.layout.yaxis.fixedrange = False
    fig.update_yaxes(title_text="Head (m)")

    fig.show()
    return fig

def plot_head_meas_and_tdpbc_vs_head_sim(pressure_loggers_to_plot,
                                         head_results_leakfree_hourly_averages,
                                         head_measurements_hourly_averages,
                                         tdpbc_by_sensor):

    fig = go.Figure()
    
    for pressure_logger in pressure_loggers_to_plot:
        
        new_length = len(head_results_leakfree_hourly_averages[(0, pressure_logger)])/24
        tdpbc_means_repeated = np.tile(tdpbc_by_sensor[pressure_logger]['means'], int(new_length)) #TODO: assertion
        tdpbc_stds_repeated = np.tile(tdpbc_by_sensor[pressure_logger]['stds'], int(new_length))
        head_results_corrected = head_results_leakfree_hourly_averages[(0, pressure_logger)] - tdpbc_means_repeated
        
        # Plot original head results
        fig.add_trace(go.Scatter(x=head_results_leakfree_hourly_averages['datetime'],
                         y=head_results_leakfree_hourly_averages[(0, pressure_logger)],
                         name='simulation original ' + str(pressure_logger),
                         marker_color='blue',
                         line_shape='hv'))
        
        # Plot corrected head result
        head_results_corrected_lower = head_results_corrected - tdpbc_stds_repeated
        head_results_corrected_upper = head_results_corrected + tdpbc_stds_repeated
        
        fig.add_trace(go.Scatter(x=head_measurements_hourly_averages['datetime'],
                                 y=head_results_corrected_lower,
                                 fill=None,
                                 line_shape='hv',
                                 marker_color='rgba(26,150,65,0.0)',
                                 showlegend=False))
        
        fig.add_trace(go.Scatter(x=head_measurements_hourly_averages['datetime'],
                                 y=head_results_corrected_upper,
                                 fill='tonexty',
                                 fillcolor='rgba(0,0,255,0.5)',
                                 marker_color='rgba(0,0,255,0.0)',
                                 line_shape='hv',
                                 name="simulation \u00B1 1\u03C3 " + str(pressure_logger))) # utf-8 for +- and sigma  
                     
        fig.add_trace(go.Scatter(x=head_measurements_hourly_averages['datetime'],
                                 y=head_measurements_hourly_averages[pressure_logger],
                                 name='measurements ' + str(pressure_logger),
                                 marker_color='grey',
                                 line_shape='hv'))
    
    fig.update_layout(
        title="Time-dependent pressure bias correction by sensor", autosize=False, width=1100, height=700,
    )

    fig.layout.yaxis.fixedrange = False
    fig.update_yaxes(title_text="Head (m)")

    return fig


def plot_tdpbc_heads_leakfree_vs_leaky(pressure_loggers_to_plot,
                                       tdpbc_by_sensor,
                                       scenarios,
                                       head_meas_rs,
                                       head_res_leakfree_rs,
                                       head_res_rs,
                                       hour_index_of_leak,
                                       leak_loc,
                                       nr_of_line_segments):
    
    def _extend_x_dt(dt_series):
        timedelta = list(dt_series)[-1] - list(dt_series)[-2]
        dt_series_tail = pd.Series(list(dt_series)[-1] + timedelta)
        dt_series = dt_series.append(dt_series_tail, ignore_index=True)
        return dt_series

    def _extend_y(y):
        y = list(y)
        y.append(y[-1])
        return y
    
    scenario_index = scenario_indices_of_leak_loc(leak_loc, scenarios)[0]
    
    fig = go.Figure()
    
    for pressure_logger in pressure_loggers_to_plot:

        new_length = len(head_res_leakfree_rs[(0, pressure_logger)])/nr_of_line_segments
        tdpbc_means = list(tdpbc_by_sensor[pressure_logger]['means'].values())
        tdpbc_means_repeated = np.tile(tdpbc_means, int(new_length)) #TODO: assertion
        tdpbc_stds = list(tdpbc_by_sensor[pressure_logger]['stds'].values())
        tdpbc_stds_repeated = np.tile(tdpbc_stds, int(new_length))
        head_res_leakfree_corrected = head_res_leakfree_rs[(0, pressure_logger)] - tdpbc_means_repeated
        head_res_corrected = head_res_rs[(scenario_index, pressure_logger)] - tdpbc_means_repeated
        
        # Upper lower and lower values of uncertainty range corrected head results
        head_res_leakfree_corrected_lower = head_res_leakfree_corrected - tdpbc_stds_repeated
        head_res_leakfree_corrected_upper = head_res_leakfree_corrected + tdpbc_stds_repeated
        head_res_corrected_lower = head_res_corrected - tdpbc_stds_repeated
        head_res_corrected_upper = head_res_corrected + tdpbc_stds_repeated
        
        # Plot original head results
        fig.add_trace(go.Scatter(x=_extend_x_dt(head_res_leakfree_rs['datetime']),
                         y=_extend_y(head_res_leakfree_rs[(0, pressure_logger)]),
                         name='leak free simulation ' + str(pressure_logger),
                         marker_color='blue',
                         line_shape='hv',
                         visible='legendonly'))
        
        fig.add_trace(go.Scatter(x=head_res_rs['datetime'][hour_index_of_leak:hour_index_of_leak+2],
                         y=head_res_rs[(scenario_index, pressure_logger)][hour_index_of_leak:hour_index_of_leak+2],
                         name='leaky simulation ' + str(pressure_logger),
                         marker_color='red',
                         mode='lines',
                         line_shape='hv',
                         visible='legendonly'))
        
        # Plot unbiased head results, with uncertainties
        fig.add_trace(go.Scatter(x= _extend_x_dt(head_meas_rs['datetime']),
                                 y= _extend_y(head_res_leakfree_corrected_lower),
                                 fill=None,
                                 line_shape='hv',
                                 marker_color='rgba(0,0,255,0.0)',
                                 showlegend=False))
        
        fig.add_trace(go.Scatter(x= _extend_x_dt(head_meas_rs['datetime']),
                                 y= _extend_y(head_res_leakfree_corrected_upper),
                                 fill='tonexty',
                                 fillcolor='rgba(0,0,255,0.35)',
                                 marker_color='rgba(0,0,255,0.0)',
                                 line_shape='hv',
                                 name="leak free simulation debiased \u00B1 1\u03C3 " + str(pressure_logger))) # utf-8 for +- and sigma 
        
        fig.add_trace(go.Scatter(x=head_meas_rs['datetime'][hour_index_of_leak:hour_index_of_leak+2],
                                 y=head_res_corrected_lower[hour_index_of_leak:hour_index_of_leak+2],
                                 fill=None,
                                 line_shape='hv',
                                 marker_color='rgba(255,0,0,0.0)',
                                 showlegend=False))
        
        fig.add_trace(go.Scatter(x=head_meas_rs['datetime'][hour_index_of_leak:hour_index_of_leak+2],
                                 y=head_res_corrected_upper[hour_index_of_leak:hour_index_of_leak+2],
                                 fill='tonexty',
                                 fillcolor='rgba(255,0,0,0.35)',
                                 marker_color='rgba(255,0,0,0.0)',
                                 line_shape='hv',
                                 name="leaky simulation debiased \u00B1 1\u03C3 " + str(pressure_logger))) # utf-8 for +- and sigma 
        
        # Plot actual head measurements
        fig.add_trace(go.Scatter(x= _extend_x_dt(head_meas_rs['datetime']),
                                 y= _extend_y(head_meas_rs[pressure_logger]),
                                 name='measurements ' + str(pressure_logger),
                                 marker_color='grey',
                                 line_shape='hv'))
    
    fig.update_layout(
        title="Time-dependent pressure bias correction by sensor", autosize=False, width=1100, height=700,
    )

    fig.layout.yaxis.fixedrange = False
    fig.update_yaxes(title_text="Head (m)")

    return fig