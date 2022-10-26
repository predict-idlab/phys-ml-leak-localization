from typing import List, Dict, Tuple
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from matplotlib import cm
import plotly
import plotly.graph_objects as go

import wntr

from src.visualization.util import normalized_abs_log_value, normalized_value


class WDNFigBuilder(ABC):
    """
    Builds a Plotly graph object figure, visualizing a WDN at particular datetime,
    from node traces, edge traces, and annotations.
    """
    
    def __init__(self, wn: wntr.network.model.WaterNetworkModel,
                       datetime: str,
                       fig_title_font_size: int,
                       fig_width: int,
                       fig_height: int,
                       fig_hoverlabel_font_size: int,
                       fig_legend_font_size: int,
                       fig_initial_zoom_xaxes: Tuple[int, int],
                       fig_initial_zoom_yaxes: Tuple[int, int]) -> None:
        self._wn = wn
        self._datetime = datetime
        self._fig_title_font_size = fig_title_font_size
        self._fig_width = fig_width
        self._fig_height = fig_height
        self._fig_hoverlabel_font_size = fig_hoverlabel_font_size
        self._fig_legend_font_size = fig_legend_font_size
        self._fig_initial_zoom_xaxes = fig_initial_zoom_xaxes
        self._fig_initial_zoom_yaxes = fig_initial_zoom_yaxes
        self._wdn_fig = None
        
    @property
    def wdn_fig(self):
        wdn_fig = self._wdn_fig
        return wdn_fig
    
    @abstractmethod
    def _build_node_traces(self) -> List[plotly.graph_objs._scatter.Scatter]:
        pass
    
    @abstractmethod
    def _build_edge_traces(self) -> List[plotly.graph_objs._scatter.Scatter]:
        pass
    
    @abstractmethod
    def _build_annotations(self) -> List:
        pass
    
    def build_wdn_fig(self) -> None:
        node_traces = self._build_node_traces()
        edges_traces = self._build_edge_traces()
        all_traces = edges_traces + node_traces
        annotations = self._build_annotations()
        
        self._wdn_fig = go.Figure(data=all_traces,
                                  layout=go.Layout(
                                      titlefont_size=self._fig_title_font_size,
                                      showlegend=True,
                                      width=self._fig_width,
                                      height=self._fig_height,
                                      hovermode='closest',
                                      margin=dict(b=20,l=5,r=5,t=40),
                                      annotations = annotations,
                                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                      legend=dict(x=0,y=1),
                                      plot_bgcolor='white'))

        self._wdn_fig.update_layout(hoverlabel=dict(font_size=self._fig_hoverlabel_font_size))
        self._wdn_fig.update_layout(legend=dict(font_size=self._fig_legend_font_size))
        
        if self._fig_initial_zoom_xaxes:
            start = self._fig_initial_zoom_xaxes[0]
            end = self._fig_initial_zoom_xaxes[1]
            self._wdn_fig.update_xaxes(type="date", range=[start, end])
        if self._fig_initial_zoom_yaxes:
            start = self._fig_initial_zoom_yaxes[0]
            end = self._fig_initial_zoom_yaxes[1]
            self._wdn_fig.update_yaxes(type="date", range=[start, end])
        
        
class WDNOverviewFigBuilder(WDNFigBuilder):
    """
    Build a Plotly graph object figure, visualizing a WDN, from lists of node and edge traces, and annotations.
    All nodes, hydrants and pressure sensors are shown as nodes.
    Pipes are plotted as edges in an undirected graph.
    """

    def __init__(self, wn: wntr.network.model.WaterNetworkModel,
                       datetime: str,
                       pressure_log_asset_ids: List[str],
                       hydrants_asset_ids: List[str],
                       leaks_asset_ids=None,
                       fig_title_font_size=16,
                       fig_width=900,
                       fig_height=900,
                       fig_hoverlabel_font_size=20,
                       fig_legend_font_size=20,
                       fig_initial_zoom_xaxes=None,
                       fig_initial_zoom_yaxes=None) -> None:
        
        super().__init__(wn, datetime, fig_title_font_size, fig_width, fig_height,
                         fig_hoverlabel_font_size, fig_legend_font_size,
                         fig_initial_zoom_xaxes,fig_initial_zoom_yaxes)
        self._pressure_log_asset_ids = pressure_log_asset_ids
        self._hydrants_asset_ids = hydrants_asset_ids
        self._leaks_asset_ids = leaks_asset_ids

    def _build_node_traces(self) -> List[plotly.graph_objs._scatter.Scatter]:
        # Build trace of all nodes 
        #all_nodes_asset_ids = list(self._wn.get_graph().nodes())
        #all_nodes = WDNNodes(self._wn, all_nodes_asset_ids,
        #                     name="other", color='lightgrey', show_hover_text=False)
        
        # Build trace of hydrants
        hydrant_nodes = WDNNodes(self._wn, self._hydrants_asset_ids, name="hydrant", color='lightgrey', size=7)
        
        # Build trace of pressure loggers
        pressure_log_nodes = WDNNodes(self._wn, self._pressure_log_asset_ids, name="pressure log", color='red')
        
        #node_traces = [all_nodes.trace,
        #               hydrant_nodes.trace,
        #               pressure_log_nodes.trace]
        
        node_traces = [hydrant_nodes.trace,
                       pressure_log_nodes.trace]
        
        # Also build trace of leak experiment locations, if they are given
        if self._leaks_asset_ids:
            leaks_nodes = WDNNodes(self._wn, self._leaks_asset_ids, name="leak experiments", color='cyan')
            node_traces.append(leaks_nodes.trace)
        
        return node_traces

    def _build_edge_traces(self) -> List[plotly.graph_objs._scatter.Scatter]:
        # Build trace that shows the layout of the pipes only,
        # without additional information such as flow direction or magnitude.
        pipes = WDNEdgesUndirectional(self._wn, name='pipe', color='lightgrey')
        
        return [pipes.trace]
    
    def _build_annotations(self) -> List:
        return []
    
    
class WDNFlowBalancesFigBuilder(WDNFigBuilder):
    """
    Build a Plotly graph object figure, visualizing a WDN for a leak scenario at a particular datetime,
    from lists of node and edge traces, and annotations.
    Node demands, flow magnitudes and directions are visualized.
    """

    def __init__(self, wn: wntr.network.model.WaterNetworkModel,
                       datetime: str,
                       pressure_log_asset_ids: List[str],
                       hydrants_asset_ids: List[str],
                       exp_leak_locs_asset_ids: List[str],
                       demands_results: pd.DataFrame,
                       flowrate_results: pd.DataFrame,
                       scenario_idx,
                       flow_color_map = cm.get_cmap('inferno_r'),
                       fig_title_font_size=16,
                       fig_width=1600,
                       fig_height=1600,
                       fig_hoverlabel_font_size=50,
                       fig_legend_font_size=30) -> None:
        
        super().__init__(wn, datetime, fig_title_font_size, fig_width, fig_height,
                         fig_hoverlabel_font_size, fig_legend_font_size)
        self._pressure_log_asset_ids = pressure_log_asset_ids
        self._hydrants_asset_ids = hydrants_asset_ids
        self._exp_leak_locs_asset_ids = exp_leak_locs_asset_ids
        self._demands_results = demands_results
        self._flowrate_results = flowrate_results
        self._scenario_idx = scenario_idx
        self._flow_color_map = flow_color_map
        self._flowrates_of_edges = self._construct_flowrates_of_edges()
        self._demands_of_nodes = self._construct_demands_of_nodes()
    
    def _construct_flowrates_of_edges(self) -> Dict:
        """Flowrate value per edge for the specified scenario and datetime."""
        
        tmp = self._flowrate_results
        flowrate_results_scen_dt = tmp[tmp['datetime'] == self._datetime][self._scenario_idx]
        
        flowrates_of_edges = {}
        for edge in list(self._wn.get_graph().edges):
            edge_name = edge[2]
            flowrate = flowrate_results_scen_dt[edge_name].values[0]
            flowrates_of_edges[edge_name] = flowrate
        
        return flowrates_of_edges
    
    def _construct_demands_of_nodes(self) -> Dict:
        """Demand value (converted to L/s) per node for the specified scenario and datetime."""
        
        tmp = self._demands_results
        demand_results_for_scen_at_dt = tmp[tmp['datetime'] == self._datetime][self._scenario_idx]
        
        demands_of_nodes = {}
        for node_name in list(self._wn.get_graph().nodes):
            demand_of_node = demand_results_for_scen_at_dt[node_name].values[0]
            # Convert raw value to string, and L/s units
            demand_of_node = str(round(demand_of_node*1000,4)) + str(" L/s")
            demands_of_nodes[node_name] = demand_of_node
            
        return demands_of_nodes

    def _build_node_traces(self) -> List[plotly.graph_objs._scatter.Scatter]:
        
        print("Building nodes...")
        
        # Build all nodes
        all_nodes_asset_ids = list(self._wn.get_graph().nodes())
        all_nodes = WDNNodes(self._wn, all_nodes_asset_ids,
                             name="other", color='lightgrey', show_hover_text=True)  
        all_nodes.add_property_to_hovertext_of_nodes(self._demands_of_nodes)
        
        # Build hydrants
        hydrant_nodes = WDNNodes(self._wn, self._hydrants_asset_ids, name="hydrant", color='grey')
            
        # Build pressure loggers
        pressure_log_nodes = WDNNodes(self._wn, self._pressure_log_asset_ids, name="pressure log",
                                      color='red', size=18)
        # Build experimental leaks
        exp_leak_locs_nodes = WDNNodes(self._wn, self._exp_leak_locs_asset_ids, name="exp. leak",
                                       color='cyan', size=20)
        
        return [all_nodes.trace,
                hydrant_nodes.trace,
                pressure_log_nodes.trace,
                exp_leak_locs_nodes.trace]

    def _build_edge_traces(self) -> List[plotly.graph_objs._scatter.Scatter]:
        
        print("Building edges...")
        
        # Build traces that show the layout of the pipes,
        # with flow magnitudes visualized as colors.
        all_edges = WDNEdgesColorized(self._wn, self._flowrates_of_edges, self._flow_color_map)
        
        return all_edges.traces
    
    def _build_annotations(self) -> List:
        
        print("Building annotations...")
        
        #Coordinates "from-nodes"
        x0 = []
        y0 = []
        #Coordinates "to-nodes"
        x1 = []
        y1 = []
        
        edge_traces_flowrate_list = []
        edge_traces_color_list = []

        # Get min and max of all flowrate values
        all_flowrate_values = list(self._flowrates_of_edges.values())
        min_flowrate_value = np.min(np.abs(all_flowrate_values))
        max_flowrate_value = np.max(np.abs(all_flowrate_values))
        
        for node_from, node_to, edge_name in list(self._wn.get_graph().edges):
            flowrate_of_edge = self._flowrates_of_edges[edge_name]
            
            # Switch direction of flow if negative
            if flowrate_of_edge < 0.0:
                tmp = node_from
                node_from = node_to
                node_to = tmp

            node_x0, node_y0 = self._wn.get_graph().nodes[node_from]['pos']
            node_x1, node_y1 = self._wn.get_graph().nodes[node_to]['pos']
            
            x0.append(node_x0)
            x1.append(node_x1)
            y0.append(node_y0)
            y1.append(node_y1)
            
            edge_traces_flowrate_list.append(np.abs(flowrate_of_edge))
            
            normalized_flowrate_abs_log = normalized_abs_log_value(flowrate_of_edge, min_flowrate_value, max_flowrate_value,
                                                                   roundoff=1e-6)
            rgba_value = str(self._flow_color_map(normalized_flowrate_abs_log))
            edge_traces_color_list.append(rgba_value)
            
            flow_direction_annotations = [dict(ax=((9/20)*x1[i]+(11/20)*x0[i]),
                                      ay=((9/20)*y1[i]+(11/20)*y0[i]),
                                      axref='x', ayref='y',
                                      x=((11/20)*x1[i]+(9/20)*x0[i]), # Trick to get hovertext in middle of the pipe
                                      y=((11/20)*y1[i]+(9/20)*y0[i]),
                                      xref='x', yref='y',
                                      showarrow=True,
                                      arrowhead=1,
                                      arrowsize=4.0,
                                      hovertext=str(round(edge_traces_flowrate_list[i]*1000, 4)) + " L/s", # Flow magn. in readable units
                                      arrowcolor="rgba"+edge_traces_color_list[i]) for i in range(0, len(x0)) # Same color as edge traces
                                     ]
        
        return flow_direction_annotations
    
    
class WDNMaxHeadResidualFigBuilder(WDNFigBuilder):
    """
    Build a Plotly graph object figure, visualizing a WDN, from lists of node and edge traces, and annotations.
    Nodes can be assigned a colorized property.
    Pipes are plotted as edges in an undirected graph.
    """

    def __init__(self, wn: wntr.network.model.WaterNetworkModel,
                       datetime: str,
                       pressure_log_asset_ids: List[str],
                       hydrants_asset_ids: List[str],
                       reservoirs_asset_ids: List[str],
                       head_residuals: pd.DataFrame,
                       scenarios: pd.DataFrame,
                       max_head_resid_color_map = cm.get_cmap('inferno_r'),
                       max_head_color_map_scaling = 'log',
                       max_head_color_map_max_bound = None,
                       fig_title_font_size=16,
                       fig_width=900,
                       fig_height=900,
                       fig_hoverlabel_font_size=20,
                       fig_legend_font_size=20,
                       fig_initial_zoom_xaxes=None,
                       fig_initial_zoom_yaxes=None) -> None:
        
        super().__init__(wn, datetime, fig_title_font_size, fig_width, fig_height,
                         fig_hoverlabel_font_size, fig_legend_font_size,
                         fig_initial_zoom_xaxes, fig_initial_zoom_yaxes)
        self._pressure_log_asset_ids = pressure_log_asset_ids
        self._hydrants_asset_ids = hydrants_asset_ids
        self._reservoirs_asset_ids = reservoirs_asset_ids
        self._head_residuals = head_residuals
        self._scenarios = scenarios
        self._max_head_residuals_of_hydrants_float, self._max_head_residuals_of_hydrants_str = self._construct_max_head_residuals_of_hydrants()
        self._max_head_resid_color_map = max_head_resid_color_map
        self._max_head_color_map_scaling = max_head_color_map_scaling
        self._max_head_color_map_max_bound = max_head_color_map_max_bound
        
    def _construct_max_head_residuals_of_hydrants(self) -> Dict:
        """Maximal head residuals occuring in pressure sensors per hydrant node
          for a leak occuring at the hydrant."""        

        max_head_residuals_of_nodes_float = {}
        max_head_residuals_of_nodes_str = {}
        for node_name in self._hydrants_asset_ids:
            # Get scenario index for leak in this hydrant
            leak_loc_scenarios_indices = self._scenarios.index[self._scenarios['leaks','loc'] == node_name]
            scen_idx = list(leak_loc_scenarios_indices)[0]
            # Get head residuals for this leak scenario
            tmp = self._head_residuals
            head_residuals_for_scen_at_dt = tmp[tmp['datetime'] == self._datetime][scen_idx].values[0]        
            # Calculate maximal head residual occuring in P sensors for this leak scenario
            max_head_residual_of_node_float = np.min(head_residuals_for_scen_at_dt)
            max_head_residuals_of_nodes_float[node_name] = max_head_residual_of_node_float
            max_head_residual_of_node_str = str(round(max_head_residual_of_node_float,3)) + str(" m")
            max_head_residuals_of_nodes_str[node_name] = max_head_residual_of_node_str
            
        return max_head_residuals_of_nodes_float, max_head_residuals_of_nodes_str
    
    def _build_node_traces(self) -> List[plotly.graph_objs._scatter.Scatter]:
        # Build trace of all nodes 
        all_nodes_asset_ids = list(self._wn.get_graph().nodes())
        all_nodes = WDNNodes(self._wn, all_nodes_asset_ids,
                             name="other", color='lightgrey', show_hover_text=False)
        
        # Build trace of hydrants
        hydrant_nodes = WDNNodes(self._wn, self._hydrants_asset_ids, name="hydrant")
        hydrant_nodes.add_property_to_hovertext_of_nodes(self._max_head_residuals_of_hydrants_str)
        hydrant_nodes.colorize_nodes_to_property(self._max_head_residuals_of_hydrants_float,
                                                 self._max_head_resid_color_map,
                                                 self._max_head_color_map_scaling,
                                                 self._max_head_color_map_max_bound)
        
        # Build trace of reservoirs
        reservoir_nodes = WDNNodes(self._wn, self._reservoirs_asset_ids, name="reservoir", color='green', size=18)
        
        # Build trace of pressure loggers
        pressure_log_nodes = WDNNodes(self._wn, self._pressure_log_asset_ids, name="pressure log", color='red', size=14)
        
        #return [all_nodes.trace,
        #        hydrant_nodes.trace,
        #        pressure_log_nodes.trace,
        #        reservoir_nodes.trace]
        return [hydrant_nodes.trace,
                pressure_log_nodes.trace,
                reservoir_nodes.trace]

    def _build_edge_traces(self) -> List[plotly.graph_objs._scatter.Scatter]:
        # Build trace that shows the layout of the pipes only,
        # without additional information such as flow direction or magnitude.
        pipes = WDNEdgesUndirectional(self._wn, name='pipe', color='lightgrey')
        
        return [pipes.trace]
    
    def _build_annotations(self) -> List:
        return []
    

class WDNLeakProbsFigBuilder(WDNFigBuilder):
    """
    Build a Plotly graph object figure, visualizing a WDN, from lists of node and edge traces, and annotations.
    Nodes can be assigned a colorized property.
    Pipes are plotted as edges in an undirected graph.
    """

    def __init__(self, wn: wntr.network.model.WaterNetworkModel,
                       datetime: str,
                       pressure_log_asset_ids: List[str],
                       hydrants_asset_ids: List[str],
                       true_leak_loc: str,
                       labels_of_hydrants: Dict,
                       leak_probs: List,
                       leak_probs_color_map = cm.get_cmap('inferno_r'),
                       leak_probs_color_map_scaling = 'log',
                       leak_probs_color_map_max_bound = None,
                       fig_title_font_size=16,
                       fig_width=900,
                       fig_height=900,
                       fig_hoverlabel_font_size=20,
                       fig_legend_font_size=20,
                       fig_initial_zoom_xaxes=None,
                       fig_initial_zoom_yaxes=None,
                       show_other_nodes_than_hydrant=True) -> None:
        
        super().__init__(wn, datetime, fig_title_font_size, fig_width, fig_height,
                         fig_hoverlabel_font_size, fig_legend_font_size,
                         fig_initial_zoom_xaxes, fig_initial_zoom_yaxes)
        self._pressure_log_asset_ids = pressure_log_asset_ids
        self._hydrants_asset_ids = hydrants_asset_ids
        self._true_leak_loc = true_leak_loc
        self._labels_of_hydrants = labels_of_hydrants
        self._leak_probs = leak_probs
        self._leak_probs_at_hydrants_float, self._leak_probs_at_hydrants_str = self._construct_leak_probs_at_hydrants()
        self._leak_probs_color_map = leak_probs_color_map
        self._leak_probs_color_map_scaling = leak_probs_color_map_scaling
        self._leak_probs_color_map_max_bound = leak_probs_color_map_max_bound
        self._show_other_nodes_than_hydrant = show_other_nodes_than_hydrant
        
    def _construct_leak_probs_at_hydrants(self) -> Dict:
        """Probabilities of a leak occuring at the hydrant."""        

        leak_probs_at_hydrants_float = {}
        leak_probs_at_hydrants_str = {}
        
        for node_name in self._hydrants_asset_ids:
            y_label = self._labels_of_hydrants[node_name]
            node_leak_prob = self._leak_probs[y_label]
            
            leak_prob_at_hydrant_float = node_leak_prob
            leak_probs_at_hydrants_float[node_name] = node_leak_prob
            leak_prob_at_hydrant_str = "p(leak) = " + str(round(node_leak_prob, 4))
            leak_probs_at_hydrants_str[node_name] = leak_prob_at_hydrant_str
            
        return leak_probs_at_hydrants_float, leak_probs_at_hydrants_str
    
    def _build_node_traces(self) -> List[plotly.graph_objs._scatter.Scatter]:
        # Build trace of all nodes 
        if self._show_other_nodes_than_hydrant:
            all_nodes_asset_ids = list(self._wn.get_graph().nodes())
            all_nodes = WDNNodes(self._wn, all_nodes_asset_ids,
                                 name="other", color='lightgrey', show_hover_text=False)
        # Build trace of hydrants
        hydrant_nodes = WDNNodes(self._wn, self._hydrants_asset_ids, name="hydrant")
        hydrant_nodes.add_property_to_hovertext_of_nodes(self._leak_probs_at_hydrants_str)
        hydrant_nodes.colorize_nodes_to_property(self._leak_probs_at_hydrants_float,
                                                 self._leak_probs_color_map,
                                                 self._leak_probs_color_map_scaling,
                                                 self._leak_probs_color_map_max_bound)
        # Build trace of pressure loggers
        pressure_log_nodes = WDNNodes(self._wn, self._pressure_log_asset_ids, name="pressure log", color='red', size=14)
        
        # Build trace of true leak loc
        true_leak_loc_nodes = WDNNodes(self._wn, [self._true_leak_loc], name="true leak loc.", color='cyan', size=16)
        
        all_traces = []
        if self._show_other_nodes_than_hydrant:
            all_traces.append(all_nodes.trace)
        all_traces.append(hydrant_nodes.trace)
        all_traces.append(pressure_log_nodes.trace)
        all_traces.append(true_leak_loc_nodes.trace)
        
        return all_traces

    def _build_edge_traces(self) -> List[plotly.graph_objs._scatter.Scatter]:
        # Build trace that shows the layout of the pipes only,
        # without additional information such as flow direction or magnitude.
        pipes = WDNEdgesUndirectional(self._wn, name='pipe', color='lightgrey')
        
        return [pipes.trace]
    
    def _build_annotations(self) -> List:
        return []

    
class WDNEdgesLeakProbsFigBuilder(WDNFigBuilder):
    """
    Build a Plotly graph object figure, visualizing a WDN, from lists of node and edge traces, and annotations.
    Pipes can be assigned a colorized property.
    """

    def __init__(self, wn: wntr.network.model.WaterNetworkModel,
                       datetime: str,
                       pressure_log_asset_ids: List[str],
                       hydrants_asset_ids: List[str],
                       exp_leak_locs_asset_ids: List[str],
                       edges_to_colorize: List[str],
                       edges_leak_probs: List[float],
                       scenario_idx,
                       flow_color_map = cm.get_cmap('inferno_r'),
                       fig_title_font_size=16,
                       fig_width=1600,
                       fig_height=1600,
                       fig_hoverlabel_font_size=50,
                       fig_legend_font_size=30) -> None:
        
        super().__init__(wn, datetime, fig_title_font_size, fig_width, fig_height,
                         fig_hoverlabel_font_size, fig_legend_font_size)
        self._pressure_log_asset_ids = pressure_log_asset_ids
        self._hydrants_asset_ids = hydrants_asset_ids
        self._exp_leak_locs_asset_ids = exp_leak_locs_asset_ids
        self._edges_leak_probs = edges_leak_probs
        self._scenario_idx = scenario_idx
        self._flow_color_map = flow_color_map
        self._edges_to_colorize = edges_to_colorize

    def _build_node_traces(self) -> List[plotly.graph_objs._scatter.Scatter]:
        
        print("Building nodes...")
        
        # Build all nodes
        all_nodes_asset_ids = list(self._wn.get_graph().nodes())
        all_nodes = WDNNodes(self._wn, all_nodes_asset_ids,
                             name="other", color='lightgrey', show_hover_text=True)  
        
        # Build hydrants
        hydrant_nodes = WDNNodes(self._wn, self._hydrants_asset_ids, name="hydrant", color='grey')
            
        # Build pressure loggers
        pressure_log_nodes = WDNNodes(self._wn, self._pressure_log_asset_ids, name="pressure log",
                                      color='red', size=18)
        # Build experimental leaks
        exp_leak_locs_nodes = WDNNodes(self._wn, self._exp_leak_locs_asset_ids, name="exp. leak",
                                       color='cyan', size=20)
        
        return [pressure_log_nodes.trace]

    def _build_edge_traces(self) -> List[plotly.graph_objs._scatter.Scatter]:
        
        print("Building edges...")
        
        # Build traces that show the layout of the pipes,
        # with flow magnitudes visualized as colors.
        all_edges = WDNEdgesColorized(self._wn, self._edges_leak_probs, self._flow_color_map, self._edges_to_colorize)
        
        return all_edges.traces
    
    def _build_annotations(self) -> List:
        return []
    

class WDNNodes(object):
    """
    A representation of a list of nodes, given by their asset id's, of a WDN.
    The final representation for plotting purposes is a Plotly Figure trace.
    """

    def __init__(self, wn, asset_ids: List[str], name: str, color='grey', size=10,
                       show_hover_text=True) -> None:
        G = wn.get_graph()
        
        #self._names = list(G.nodes())
        self._name = name
        self._x_coords = [G.nodes[node]['pos'][0] for node in asset_ids]
        self._y_coords = [G.nodes[node]['pos'][1] for node in asset_ids]
        self._node_names = asset_ids
        self._node_hover_text = asset_ids
        self._show_hover_text = show_hover_text
        self._colors = [color]*len(asset_ids)
        self._size = size
        self._trace = None
        
    @property
    def trace(self):
        self.construct_trace()
        return self._trace
    
    def construct_trace(self) -> None:
        self._trace = go.Scatter(x=self._x_coords,
                                 y=self._y_coords,
                                 mode='markers',
                                 hoverinfo='text',
                                 name=self._name,
                                 marker=dict(
                                     color=self._colors,
                                     size=self._size,
                                     line_width=0))
        if self._show_hover_text:
            self._trace.text = self._node_hover_text
            
    def add_property_to_hovertext_of_nodes(self, property_per_node: Dict[str,str]) -> None:
        
        self._node_hover_text = []
        for node_name in self._node_names:
            try:
                property_of_node = property_per_node[node_name]
            except Keyerror:
                print(f"No property given for node {node_name}")
                raise
            node_hover_text_combined = node_name + "<br>" + property_of_node
            self._node_hover_text.append(node_hover_text_combined)
            
    def colorize_nodes_to_property(self, property_per_node: Dict, node_color_map,
                                   node_color_map_scaling, color_map_max_bound=None) -> None:
        
        all_property_values = list(property_per_node.values())
        min_property_value = np.min(np.abs(all_property_values))
        max_property_value = np.max(np.abs(all_property_values))
        
        self._colors = []
        
        for node_name in self._node_names:
            try:
                property_value_of_node = property_per_node[node_name]
            except Keyerror:
                print(f"No property given for node {node_name}")
                raise
            
            if node_color_map_scaling == 'log':
                color_map_value = normalized_abs_log_value(property_value_of_node,
                                                                       min_property_value,
                                                                       max_property_value,
                                                                       roundoff=1e-1)
            if node_color_map_scaling == 'log_set_max':
                color_map_value = normalized_abs_log_value(property_value_of_node,
                                                                       min_property_value,
                                                                       color_map_max_bound,
                                                                       roundoff=1e-1)
            elif node_color_map_scaling == 'linear':
                color_map_value = normalized_value(property_value_of_node,
                                                   max_property_value)
            elif node_color_map_scaling == 'linear_set_max':
                color_map_value = normalized_value(property_value_of_node,
                                                   color_map_max_bound)
                
            rgba_value = str(node_color_map(color_map_value))
            self._colors.append("rgba" + rgba_value)
            

class WDNEdgesUndirectional(object):
    """
    A representation of WDN pipes as undirectional edges in a Plotly Go.Scatter() trace.
    The final representation for plotting purposes can be accessed as a Plotly Figure trace.
    """

    def __init__(self, wn, name='pipe', color='lightgrey') -> None:
        G = wn.get_graph()
        
        self._name = name
        self._edges_x = []
        self._edges_y = []
        for edge in G.edges():
            x0, y0 = G.nodes[edge[0]]['pos'] # "From" node coordinates of edge
            x1, y1 = G.nodes[edge[1]]['pos'] # "To" node coordinates of edge
            self._edges_x.append(x0)
            self._edges_x.append(x1)
            self._edges_x.append(None)
            self._edges_y.append(y0)
            self._edges_y.append(y1)
            self._edges_y.append(None)
          
        # Edges 
        self._color = color
        self._trace = None

    @property
    def trace(self):
        self.construct_trace()
        return self._trace
    
    def construct_trace(self) -> None:
        self._trace = go.Scatter(
                          x=self._edges_x, y=self._edges_y,
                          line=dict(width=0.5, color=self._color),
                          hoverinfo='none',
                          mode='lines',
                          name=self._name)
        

class WDNEdgesColorized(object):
    """
    A representation of WDN pipes as colorized edges as a list of Plotly Go.Scatter() traces.
    Pipes are colored according to their flow magnitudes.
    """

    def __init__(self, wn, flowrates_of_edges: Dict, flow_color_map, edges_to_colorize=None) -> None:
        self._wn = wn
        self._flowrates_of_edges = flowrates_of_edges
        self._flow_color_map = flow_color_map
        self._traces = []
        self._edges_to_colorize = edges_to_colorize

    @property
    def traces(self):
        self.construct_traces()
        return self._traces
    
    def construct_traces(self) -> None:
        G = self._wn.get_graph()
        
        # Get min and max of all flowrate values
        all_flowrate_values = list(self._flowrates_of_edges.values())
        min_flowrate_value = np.min(np.abs(all_flowrate_values))
        max_flowrate_value = np.max(np.abs(all_flowrate_values))
        
        for edge in G.edges:
            
            # only color edges in self._edges_to_colorize, if it is actually given
            # otherwise all edges in G.edges are colorized
            if self._edges_to_colorize:
                if edge[2] not in self._edges_to_colorize:
                    continue
                    
            from_node = edge[0]
            to_node = edge[1]
            edge_name = edge[2]
            
            x = [G.nodes[from_node]['pos'][0], G.nodes[to_node]['pos'][0]]
            y = [G.nodes[from_node]['pos'][1], G.nodes[to_node]['pos'][1]]
            
            try:
                flowrate_of_edge = self._flowrates_of_edges[edge_name]
            except Keyerror:
                print(f"Flowrate not given for {edge_name}")
                raise
            
            # Calculate color corresponding to colormap for this flowrate
            #normalized_flowrate_abs_log = normalized_abs_log_value(flowrate_of_edge,
            #                                                       min_flowrate_value,
            #                                                       max_flowrate_value,
            #                                                       roundoff=1e-1) #change to 1e-6 for flows!
            normalized_flowrate_abs_log = normalized_value(flowrate_of_edge,
                                                           0.002)
            rgba_value = str(self._flow_color_map(normalized_flowrate_abs_log))
            
            # Construct edge as colored trace
            edge_trace = go.Scatter(x=x,
                                    y=y,
                                    mode='lines',
                                    hovertext=str(flowrate_of_edge),
                                    line=dict(width=8.0, color="rgba" + rgba_value),
                                    showlegend=False,
                                    hoverinfo='skip')
            self._traces.append(edge_trace)