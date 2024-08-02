# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)

import numpy as np
import pandas as pd
import copy 
import warnings

from chronix2grid.generation.dispatch.utils import RampMode


def filter_ramps(net, mode):
    """
    Apply filters on ramp constraints, corresponding to different levels of
    complexity of the economic dispatch.
    Parameters
    ----------
    net: pypsa.Network
        The instance to modify
    mode: RampMode
        The level of difficulty to use for ramps
    Returns
    -------
    The modified pypsa.Network instance
    """
    hydro_names = net.generators[net.generators.carrier == 'hydro'].index.tolist()
    thermal_names = net.generators[net.generators.carrier == 'thermal'].index.tolist()
    nuclear_names = net.generators[net.generators.carrier == 'nuclear'].index.tolist()

    if mode == RampMode.medium:
        net = remove_ramps(net, thermal_names)
    if mode == RampMode.easy:
        net = remove_ramps(net, hydro_names + thermal_names)
    if mode == RampMode.none:
        net = remove_ramps(net, nuclear_names + hydro_names + thermal_names)
    return net


def remove_ramps(net, gen_names):
    """
    Remove de ramp constraints for the gen_names generators.
    Parameters
    ----------
    net: pypsa.Network
        The instance to modify
    gen_names: list
        List of generator names for which ramp constraints are removed
    Returns
    -------
    The modified pypsa.Network instance
    """
    net.generators.loc[gen_names, ['ramp_limit_up', 'ramp_limit_down']] = np.nan
    return net

def update_gen_constrains(gen_constraints_user):
    """Generator constraint passed by user
    
    Parameters
    ----------
    gen_constraints_user : dict
        Valid keys p_max_pu, p_min_pu
    
    Returns
    -------
    dict
        Updated gen dict constratins
    """    
    gen_constraints = {'p_max_pu': None, 'p_min_pu': None}
    gen_constraints.update(gen_constraints_user)
    return gen_constraints

def update_params(num, start_date, params_user):
    """ Function to update parameters dictionary to run
    Linear OPF
    
    Parameters
    ----------
    num : int
        Total lenght of input data
    start_date : datetime
        Starting date of the input data
    params_user : dict
        The dictionary contains the following params:
            snapshots     : temporary date range {YEAR 2007 - non leap} to be 
                            set it to load and gen constraints data
            step_opf_min  : Assuming real-time data comes every 5 minutes, the 
                            ste_opf_min is the skipped minutes opf will run
            mode_opf      : Mode OPF formulates as single optimization problem
            reactive_comp : Factor applied to consumption to compensate reactive
                            part not modelled by linear opf
    Returns
    -------
    dict
        Updated parameters dict
    """  
    # Initialize possible param keys if they are
    # not passed by user
    params={'snapshots': [],
            'step_opf_min': 5,
            'mode_opf': 'day',
            'reactive_comp': 1.025,
    }
    params.update(params_user)
    # Get user params
    snaps = params['snapshots']
    step_opf = int(params['step_opf_min'])
    mode_opf = params['mode_opf']
    # Check some inputs
    if not 60 % step_opf == 0:
        raise RuntimeError("\"step_opf_min\" argument might be (5, 10, 15, 20, 30, 60)")
    if mode_opf is not None :
        print('mode_opf is not None')
        if not mode_opf.lower() in ['day', 'week', 'month']:
            raise RuntimeError("Please provide a valid opf mode (day, week, month")
    # Create temporary date range to be load to input data
    if snaps == []:
        snapshots = pd.date_range(start=start_date, periods=num, freq='5min')
        params.update({'snapshots': snapshots})
    return params

def preprocess_input_data(load, gen_constraints, params):
    """ Functions to modify temporarily input data as following:
           - It checks whether load and gen constraints have same lengt.
           - It sets up index in all dataframes with temp data_range format.
           - It resamples data according to param specified -step_opf_min-
           - It modify load df aggregating all loads in unique col named as agg_laod. 
    
    Parameters
    ----------
    load : dataframe
        Consumption to be filled in opf
    gen_constraints : dict
        Dictionary holding gen constraints:
           - p_max_pu: df with all max gen constraints in pu
           - p_min_pu: df with all min gen constraints in pu
    params : dict
        OPF parameters
    
    Returns
    -------
    df, dict
        Updated input data 
    """    
    # Check lenght between load and gen constraints
    for k in gen_constraints:
        if gen_constraints[k] is not None and gen_constraints[k].shape[0] != load.shape[0]:
            raise RuntimeError(f'Gen constraint in [{k}] does not have same lenght as input data')
    # Get load after adapting to step_opf_min  
    new_load = reformat_load(load, params)
    # Get gen contraints after adapting to step_opf_min
    new_gen_const = reformat_gen_constraints(gen_constraints, params, new_load.index)
    return new_load, new_gen_const
    
def reformat_load(load, params):
    """ Reformat Load input data:
          - Set date_range as index to load df
          - Resamples according to param -step_opf_min-
    
    Parameters
    ----------
    load : dataframe
        Consumption to be filled in opf
    params : dict
        OPF parameters
    
    Returns
    -------
    df
    """  
    if not isinstance(load, pd.DataFrame): 
        raise RuntimeError(f'Load must be a dataframe - > Error!! {type(load)}')
    snapshots, step_opf, q_compensation = params['snapshots'], params['step_opf_min'], params['reactive_comp']
    # Set temp index to load
    load.index = snapshots
    # Resample data and scale to compensate reactive part
    load_resampled = load.resample(f'{str(step_opf)}min').apply(lambda x: x[0])
    load_resampled *= q_compensation
    # -- Agregate load in case it is not
    if load_resampled.columns.tolist() == ['agg_load']:
        return load_resampled
    else:
        agg_load = load_resampled.sum(axis=1).to_frame()
        agg_load.columns = ['agg_load']
        return agg_load

def reformat_gen_constraints(gen_constraints, params, new_snaps):
    """
          - Set date_range as index to all dfs in gen constraints dictionary
          - Resamples according to param -step_opf_min-
    
    Parameters
    ----------
    gen_constraints : dict
        Dictionary holding gen constraints:
           - p_max_pu: df with all max gen constraints
           - p_min_pu: df with all min gen constraints 
    params : dict
        OPF parameters
    new_snaps : Datetime
        Resampled index extracted from input df
    
    Returns
    -------
    dict
        Updated gen dict constraints
    """    
    snapshots = params['snapshots']
    resampled_snapshots = new_snaps
    # Set index in gen contraints 
    for k in gen_constraints:
        if gen_constraints[k] is None:  # If no data is passed. An empty df is created
            gen_constraints.update({k: pd.DataFrame(index=snapshots)})
        else:  # Otherwise it will set up new indexes
            if not isinstance(gen_constraints[k], pd.DataFrame): 
                raise RuntimeError(f'Gen constraints {k} must be a dataframe - > Error!! {type(gen_constraints[k])}')
            gen_constraints[k].index = snapshots
            gen_constraints.update({k: gen_constraints[k]})
        # Finally a resampled df is updated in dict
        gen_constraints.update({k: gen_constraints[k].loc[resampled_snapshots]})
    return gen_constraints

def preprocess_net(net, every_min, input_data_resolution=5):
    """ Function that mainly performs the following tasks:
            - Remove all previous loads modelled in net
              and create a new single load named as agg_load.
            - Adapts the ramps up/down according to the configured
              time to run OPF.
            - Change set up commitable for all gen as False
    
    Parameters
    ----------
    net : PyPSA instance
    every_min : int 
        [description]
    input_data_resolution : int, optional
        [description], by default 5
    
    Returns
    -------
    PyPSA instance
        Modified PyPSA grid
    """      
    # Remove all loads modelled in PyPSA
    # and create one single agg_load.
    net.mremove('Load', names=net.loads.index)  
    net.add('Load', name='agg_load', bus=net.buses.index.tolist()[0])
    # Adapt ramps according to the skipping time
    # configured to run the OPF. 
    steps = every_min / input_data_resolution
    net.generators.loc[:, ['ramp_limit_up', 'ramp_limit_down']] *= steps
    # Change commitable as false
    net.generators.commitable = False
    return net

def get_grouped_snapshots(snapshot, mode): 
    """ Get grouped snapshots per opf mode
    
    Parameters
    ----------
    snapshot : datetime
    mode : str
        [day, week, month]
    """    
    # Define all posibilities mode
    periods = {'day': snapshot.groupby(snapshot.day).values(),
               'week': snapshot.groupby(snapshot.isocalendar().week).values(),
               'month': snapshot.groupby(snapshot.month).values()
    }
    return periods[mode]

def run_opf(net,
            demand,
            gen_max,
            gen_min,
            params,
            total_solar=None,
            total_wind=None,
            slack_name=None,
            slack_pmin=None,
            slack_pmax=None,
            gen_min_pu_t=None,  # used when splitting the losses, to remember, for each generators / steps the setpoint
            gen_max_pu_t=None,  # used when splitting the losses, to remember, for each generators / steps the setpoint
            **kwargs):
    """ Run linear OPF problem in PyPSA considering
    only marginal costs and ramps as LP problem.
    
    Parameters
    ----------
    net : PyPSA instance
    demand : dataframe
        Load to be filled
    gen_max : dataframe
        Generator max constraints in pu
    gen_min : dataframe
        Generator min constraints in pu
    params : dict
        OPF set up parameters
    
    Returns
    -------
    dataframe
        Results of OPF dispatch
    """    
    to_disp = {'day': demand.index.day.unique().values[0],
               'week': demand.index.isocalendar().week.unique()[0],
               'month': demand.index.month.unique().values[0],
    }
    mode = params['mode_opf']
    if mode is None:
        print(f'\n--> OPF formulation by => full chronix - Analyzing ')
    else:
        print(f'\n--> OPF formulation by => {mode} - Analyzing {mode} # {to_disp[mode]}')
    net = copy.deepcopy(net)
    
    if "PmaxErrorCorrRatio" in params:
        if "agg_solar" in net.generators.p_nom:
            init_solar = net.generators.p_nom["agg_solar"]
        if "agg_wind" in net.generators.p_nom:
            init_wind = net.generators.p_nom["agg_wind"]
            
        net.generators.p_nom *= float(params["PmaxErrorCorrRatio"])
        
        if "agg_solar" in net.generators.p_nom:
            net.generators.p_nom["agg_solar"] = init_solar
        if "agg_wind" in net.generators.p_nom:
            net.generators.p_nom["agg_wind"] = init_wind
    
    if "RampErrorCorrRatio" in params:
        if "agg_solar" in net.generators.ramp_limit_up:
            init_solar = net.generators.ramp_limit_up["agg_solar"]
        if "agg_wind" in net.generators.ramp_limit_up:
            init_wind = net.generators.ramp_limit_up["agg_wind"]
            
        net.generators.ramp_limit_up *= float(params["RampErrorCorrRatio"])
        net.generators.ramp_limit_down *= float(params["RampErrorCorrRatio"])
        
        if "agg_solar" in net.generators.ramp_limit_up:
            net.generators.ramp_limit_up["agg_solar"] = init_solar
        if "agg_wind" in net.generators.ramp_limit_up:
            net.generators.ramp_limit_up["agg_wind"] = init_wind
        
    # Reset information previously 
    # saved it in PyPSA instance
    net.loads_t.p_set = net.loads_t.p_set.iloc[0:0, 0:0]
    net.generators_t.p_max_pu = net.generators_t.p_max_pu.iloc[0:0, 0:0]
    net.generators_t.p_min_pu = net.generators_t.p_min_pu.iloc[0:0, 0:0]
    
    # Set snapshots
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        # pypsa and pandas version
        net.set_snapshots(demand.index)
    
    
    # ++  ++  ++  ++  ++  ++  ++  ++  ++  ++  ++ 
    # Fill load and gen constraints to PyPSA grid
    if total_solar is not None or total_wind is not None:
        # allow to curtail the solar and wind (to avoid infeasibility)
        gen_max = copy.deepcopy(gen_max)
        if total_solar is not None:
            gen_max["agg_solar"] = total_solar
        if total_wind is not None:
            gen_max["agg_wind"] = total_wind
    
    if total_solar is None:
        # I did not specify any solar time series, i should tell the network they are all 0.
        gen_max = copy.deepcopy(gen_max)
        gen_max["agg_solar"] = 0.
    if total_wind is None:
        # I did not specify any wind time series, i should tell the network they are all 0.
        gen_max = copy.deepcopy(gen_max)
        gen_max["agg_wind"] = 0.
        
    if slack_name is not None and slack_pmin is not None:
        # add pmin to the slack bus, to avoid negative production when losses
        # are added
        gen_min = copy.deepcopy(gen_min)
        gen_min[slack_name] = slack_pmin
            
    if slack_name is not None and slack_pmax is not None:
        # add pmin to the slack bus, to avoid negative production when losses
        # are added
        gen_max = copy.deepcopy(gen_max)
        gen_max[slack_name] = slack_pmax
    
    if slack_name is not None and "slack_ramp_limit_ratio" in params:
        if slack_name in net.generators.ramp_limit_up:
            net.generators.ramp_limit_up[slack_name] *= float(params["slack_ramp_limit_ratio"])
        else:
            if not net._pypsa_debug_flags["slack_maybe_ignored"]:
                # slack should be there !
                raise RuntimeError("Slack is missing from the grid generator but should be there")
        if slack_name in net.generators.ramp_limit_down:
            net.generators.ramp_limit_down[slack_name] *= float(params["slack_ramp_limit_ratio"])
        else:
            if not net._pypsa_debug_flags["slack_maybe_ignored"]:
                # slack should be there !
                raise RuntimeError("Slack is missing from the grid generator but should be there")
    
    if gen_max_pu_t is not None:
        # addition contraint on the max_pu, used for example when splitting the loss
        for gen_nm, max_val in gen_max_pu_t.items():
            if gen_nm in gen_max:
                gen_max[str(gen_nm)] = np.minimum(gen_max[str(gen_nm)], max_val)
            else:
                gen_max[str(gen_nm)] = max_val
    
    if gen_min_pu_t is not None:
        # addition contraint on the min_pu, used for example when splitting the loss
        for gen_nm, min_val in gen_min_pu_t.items():
            if gen_nm in gen_min:
                gen_min[str(gen_nm)] = np.maximum(gen_min[str(gen_nm)], min_val)
            else:
                gen_min[str(gen_nm)] = min_val
    
    net.loads_t.p_set = pd.concat([demand])
    net.generators_t.p_max_pu = pd.concat([gen_max], axis=1)
    net.generators_t.p_min_pu = pd.concat([gen_min], axis=1)
    
    # raise error for non feasible configurations
    if (net.generators["p_min_pu"].values < 0.).any():
        raise RuntimeError(f"Wrong input data (prods_charac.csv) there are some negative "
                           f"p_min for {net.generators.loc[net.generators['p_min_pu'].values < 0.].index}")
    if (net.generators["p_max_pu"].values < 0.).any():
        raise RuntimeError(f"Wrong input data (prods_charac.csv) there are some negative "
                           f"p_max for {net.generators.loc[net.generators['p_max_pu'].values < 0.].index}")
    if (net.generators["ramp_limit_up"].values < 0.).any():
        raise RuntimeError(f"Wrong input data (prods_charac.csv) there are some negative "
                           f"ramps (up) for {net.generators.loc[net.generators['ramp_limit_up'].values < 0.].index}")
    if (net.generators["ramp_limit_down"].values < 0.).any():
        raise RuntimeError(f"Wrong input data (prods_charac.csv) there are some negative ramps "
                           f"(down) for {net.generators.loc[net.generators['ramp_limit_down'].values < 0.].index}")
    if (net.generators_t.p_min_pu.min() > 1.).any():
        debug_info = net.generators_t.p_min_pu.min()
        raise RuntimeError(f"A generator would have a p_min above its pmax ! most likely "
                           f"the slack... Check {debug_info.loc[debug_info > 1.].index.values}")
    if ((net.generators_t["p_max_pu"] - net.generators_t["p_min_pu"]) < 0.).any().any():
        tmp_debug = ((net.generators_t["p_max_pu"] - net.generators_t["p_min_pu"]) < 0.).any()
        raise RuntimeError(f"Invalid configuration: infeasibility detected for some "
                           f"generators (most likely the slack), check generator "
                           f"{tmp_debug[tmp_debug].index.values}")
    
    # try to detect some non feasible states
    if "agg_wind" in net.generators_t["p_max_pu"]:
        total_wind = (net.generators.loc["agg_wind", "p_nom"] * net.generators_t["p_max_pu"]["agg_wind"]).values
    else:
        total_wind = 0.
    if "agg_solar" in net.generators_t["p_max_pu"]:
        total_solar = (net.generators.loc["agg_solar", "p_nom"] * net.generators_t["p_max_pu"]["agg_solar"]).values
    else:
        total_solar = 0.
    total_pmax = net.generators.loc[(net.generators.index != "agg_wind") & (net.generators.index != "agg_solar"), "p_nom"].sum()
    tmp_ = net.generators.loc[(net.generators.index != "agg_wind") & (net.generators.index != "agg_solar"), ["p_nom", "p_min_pu"]].values
    total_pmin = (tmp_[:, 0] * tmp_[:, 1]).sum()
    min_net_demand = demand["agg_load"].values - total_solar - total_solar 
    if (min_net_demand > total_pmax).any():
        return None, RuntimeError("Some non feasible time steps are found (net demand above pmax of controlable generators)")
    if (demand["agg_load"].values < total_pmin).any():
        return None, RuntimeError("Some non feasible time steps are found (demand [with full "
                                  "curtailment of solar and wind] is below pmin of "
                                  "controlable generators)")
    
    # TODO check also feasible ramps
    # tmp_ = net.generators.loc[(net.generators.index != "agg_wind") & (net.generators.index != "agg_solar"), ["p_nom", "ramp_limit_up", "ramp_limit_down"]].values

    # ++  ++  ++  ++
    # Run Linear OPF
    status, termination_condition = net.lopf(net.snapshots, **kwargs)
    if status != 'ok':
        print('** OPF failed to find an optimal solution **')
        return None, termination_condition
    else:
        print('-- opf succeeded  >Objective value (should be greater than zero!')
        return net.generators_t.p.copy(), termination_condition


def interpolate_dispatch(dispatch, method='quadratic'):
    """Function to interpolate in case opf in running for 
    steps greater than 5 min.
    
    Parameters
    ----------
    dispatch : dataframe
        OPF dispatch result
    method : str, optional
        Pandas interpolation approach, by default 'quadratic'
    
    Returns
    -------
    dataframe
        Interpolated dispatch results
    """    
    # Create dataframe with full index
    dff = dispatch.copy().asfreq('5min')
    # Convert them to numeric columns
    for col in dff:
        dff[col] = pd.to_numeric(dff[col], errors='coerce')
    interpolated_df = dff.interpolate(method=method, axis=0)
    # Force to put zero for very samell values
    criteria_small_value = 1e-4
    interpolated_df[interpolated_df < criteria_small_value] = 0
    return interpolated_df.round(2)