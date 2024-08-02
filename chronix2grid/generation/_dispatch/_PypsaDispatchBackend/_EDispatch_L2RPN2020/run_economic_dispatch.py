# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)

import argparse
import os
import time
from typing import Dict, Literal

import numpy as np
import pandas as pd
import pypsa

from .utils import get_grouped_snapshots
from .utils import interpolate_dispatch
from .utils import preprocess_input_data
from .utils import preprocess_net, filter_ramps
from .utils import run_opf
from .utils import update_gen_constrains, update_params

## Dépendances Chronix2Grid !!
from chronix2grid.generation.dispatch.utils import RampMode
import chronix2grid.constants as cst


def main_run_disptach(pypsa_net, 
                      load,
                      total_solar,
                      total_wind,
                      params={},
                      gen_constraints=None,
                      ramp_mode=RampMode.hard,
                      **kwargs):
    # Update gen constrains dict with 
    # values passed by the users and params
    if gen_constraints is None:
        gen_constraints = {}

    gen_constraints : Dict[Literal["p_min_pu", "p_max_pu"], pd.DataFrame]= update_gen_constrains(gen_constraints)  
    for el, df in gen_constraints.items():
        df.bfill(inplace=True)  # propagate possible NaNs "backward", good for first Na values
        df.ffill(inplace=True)  # propagate possible NaNs "forward", good if last Na values  
    params = update_params(load.shape[0], load.index[0], params)

    print ('Preprocessing input data..')    
    # Preprocess input data:
    #   - Add date range as index 
    #   - Check whether gen constraints has same lenght as load
    load_, gen_constraints_ = preprocess_input_data(load, gen_constraints, params)
    if total_solar is not None:
        solar_ = pd.DataFrame({"agg_solar": 1.0 * total_solar.values}, index=load_.index)
    else:
        solar_ = None
    if total_wind is not None:
        wind_ = pd.DataFrame({"agg_wind": 1.0 * total_wind.values}, index=load_.index)
    else:
        wind_ = None
    tot_snap = load_.index

    print('Filter generators ramps up/down')
    # Preprocess pypsa net ramps according to
    # the level specified
    pypsa_net = filter_ramps(pypsa_net, ramp_mode)

    print('Adapting PyPSA grid with parameters..')
    # Preprocess net parameters:
    #   - Change ramps according to params step_opf_min (assuming original 
    #     values are normalizing for every 5 minutes)
    #   - It checks for all gen units if commitable variables is False
    #     (commitable as False helps to create a LP problem for PyPSA)
    pypsa_net = preprocess_net(pypsa_net, params['step_opf_min'])

    months = tot_snap.month.unique()
    slack_name = None
    slack_pmin = None
    slack_pmax = None
    slack_ignored_debug_flag = False  # is the slack ignored due to a debug flag
    if 'slack_name' in params and pypsa_net._pypsa_debug_add_slack:
        slack_name = str(params["slack_name"])
        if slack_name not in pypsa_net.generators.index:
            if pypsa_net._pypsa_debug_flags["slack_maybe_ignored"]:
                if slack_name not in pypsa_net._all_gen_names:
                    raise RuntimeError("Wrong configuration detected: make sure that the slack is a generator "
                                    "but also a controlable one !")
                slack_ignored_debug_flag = True  # in this case it is !
            else:
                print(pypsa_net.generators.index)
                raise RuntimeError("Wrong configuration detected: make sure that the slack is a generator "
                                "but also a controlable one !")
                
        if 'slack_pmin' in params and not slack_ignored_debug_flag:
            # user specified a pmin for the slack bus
            slack_pmin = float(params["slack_pmin"]) / float(pypsa_net.generators.loc[slack_name].p_nom)
            if slack_name not in gen_constraints["p_min_pu"]:
                gen_constraints["p_min_pu"][slack_name] = slack_pmin
            else:
                gen_constraints["p_min_pu"][slack_name] = np.maximum(slack_pmin, gen_constraints["p_min_pu"][slack_name].values)
                
            if slack_pmin < gen_constraints["p_min_pu"][slack_name].min():
                raise RuntimeError("Wrong configuration detected: you put in 'params_opf' "
                                   "a minimum value for the slack ('slack_pmin') below its pmin.")
            if slack_pmin > 1.:
                raise RuntimeError("Unfeasible parameters: slack pmin would be higher than its pmax.")
        if 'slack_pmax' in params and not slack_ignored_debug_flag:
            # user specified a pmin for the slack bus
            slack_pmax = float(params["slack_pmax"]) / float(pypsa_net.generators.loc[slack_name].p_nom)
            if slack_name not in gen_constraints["p_max_pu"]:
                gen_constraints["p_max_pu"][slack_name] = slack_pmax
            else:
                gen_constraints["p_max_pu"][slack_name] = np.minimum(slack_pmin, gen_constraints["p_max_pu"][slack_name].values)
            if slack_pmax > gen_constraints["p_max_pu"][slack_name].max():
                raise RuntimeError("Wrong configuration detected: you put in 'params_opf' "
                                   "a maximum value for the slack ('slack_pmax') above its pmax.")
            if slack_pmax > 1.:
                print("Doubtful parameters: slack pmax would be higher than its pmax, basically "
                      "the parameters `slack_pmax` of `params_opf.json` is ignored.")
        
    error_ = False
    start = time.time()
    results, termination_conditions = [], []
    if (params['mode_opf'] is not None):
        print(f'mode_opf is not None: {params["mode_opf"]}')
        for month in months:
            # Get snapshots per month
            snap_per_month = tot_snap[tot_snap.month == month]
            # Filter input data per month
            load_per_month = load_.loc[snap_per_month]
            if solar_ is not None:
                total_solar_per_month = solar_.loc[snap_per_month]
            else:
                total_solar_per_month = None
            if wind_ is not None:
                total_wind_per_month = wind_.loc[snap_per_month]
            else:
                total_wind_per_month = None
            # Get gen constraints separated and filter by month
            g_max_pu, g_min_pu = gen_constraints_['p_max_pu'], gen_constraints_['p_min_pu']
            g_max_pu_per_month = g_max_pu.loc[snap_per_month]
            g_min_pu_per_month = g_min_pu.loc[snap_per_month]
            # Get grouped snapsshots given monthly snapshots
            if (params['mode_opf'] is not None):
                snap_per_mode = get_grouped_snapshots(snap_per_month, params['mode_opf'])
                for snap_id, snaps in enumerate(snap_per_mode):
                    # Truncate input data per mode (day, week, month)
                    load_per_mode = load_per_month.loc[snaps]
                    if total_solar_per_month is not None:
                        total_solar_per_mode = total_solar_per_month.loc[snaps]
                    else:
                        total_solar_per_mode = None
                    if total_wind_per_month is not None:
                        total_wind_per_mode = total_wind_per_month.loc[snaps]
                    else:
                        total_wind_per_mode = None
                    gen_max_pu_per_mode = g_max_pu_per_month.loc[snaps]
                    gen_min_pu_per_mode = g_min_pu_per_month.loc[snaps]
                    # Run opf given in specified mode
                    dispatch, termination_condition = run_opf(
                        pypsa_net,
                        load_per_mode,
                        gen_max_pu_per_mode,
                        gen_min_pu_per_mode, params, 
                        total_solar=total_solar_per_mode,
                        total_wind=total_wind_per_mode,
                        slack_name=slack_name,
                        slack_pmin=slack_pmin,
                        slack_pmax=slack_pmax,
                        **kwargs)
                    if dispatch is None:
                        print(f"ERROR: dispatch failed for 'month' {month} (snap {snap_id})")
                        error_ = True
                        break
                    results.append(dispatch)
                    termination_conditions.append(termination_condition)
    else:
        g_max_pu, g_min_pu = gen_constraints_['p_max_pu'], gen_constraints_['p_min_pu']
        dispatch, termination_condition = run_opf(
               pypsa_net, load_, g_max_pu,
               g_min_pu, params,
               total_solar=solar_,
               total_wind=wind_,
               slack_name=slack_name,
               slack_pmin=slack_pmin,
               slack_pmax=slack_pmax,
               **kwargs)

        if dispatch is None:
            error_ = True
            print(f"ERROR: dispatch failed.")
        results.append(dispatch)
        termination_conditions.append(termination_condition)

    if error_:
        return None, termination_condition, None
    
    # Unpack individual dispatchs and prices
    opf_prod = pd.DataFrame()
    for df in results:
        opf_prod = pd.concat([opf_prod, df], axis=0)

    # Sort by datetime
    opf_prod.sort_index(inplace=True)
    # Create complete prod_p dataframe and interpolate missing rows
    prod_p = opf_prod.copy()
    # Apply interpolation in case of step_opf_min greater than 5 min
    if params['step_opf_min'] > 5:
        print ('\n => Interpolating dispatch into 5 minutes resolution..')
        prod_p = interpolate_dispatch(prod_p)

    # Get the prices of the marginal generator at each timestep
    marginal_costs = pypsa_net.generators.marginal_cost
    marginal_prices = prod_p.apply(
        lambda row: marginal_costs[row[row > 0].index].max(),
        axis=1)

    # Add noise to results
    # gen_cap = pypsa_net.generators.p_nom
    # prod_p_with_noise = add_noise_gen(prod_p, gen_cap, noise_factor=0.0007)

    end = time.time()
    print('Total time {} min'.format(round((end - start)/60, 2)))
    print('OPF Done......')
    # at this stage prod_p contains the renewable agg_solar and agg_wind
    return prod_p, termination_conditions, marginal_prices

# In case to launch by the terminal
# ++  ++  ++  ++  ++  ++  ++  ++  +
# Vars to set up...
# PYPSA_CASE = './L2RPN_2020_ieee_118_pypsa_simplified'
REF_DATA_DIR = './reference_data'
DESTINATION_PATH = './OPF_rel/'
POSSIBLE_MODE_OPF = ['day', 'week', 'month']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Launch the evaluation of the Grid2Op ("Grid To Operate") code.')
    parser.add_argument('--grid_path', default='', type=str,
                        help='PyPSA grid dir')
    parser.add_argument('--conso_full_path', default=os.path.join(os.path.abspath(REF_DATA_DIR), 'load_2007.csv.bz2'), type=str,
                       help='Specify the consumption or demanda full name')
    parser.add_argument('--gen_const', type=str,
                        help='Specify full path of gen max and min constraints')
    parser.add_argument('--dest_path', default=os.path.abspath(DESTINATION_PATH), type=str,
                         help='Specify the destination dir')
    parser.add_argument('--mode_opf', default='day', type=str,
                        help='Optimization mode, for the opf (possible values are {})'.format(POSSIBLE_MODE_OPF))
    parser.add_argument('--rescaled_min', default=5, type=int,
                        help='Run the optimizer every "rescaled_min" minutes (default 5)'),

    args = parser.parse_args()
    if not args.mode_opf.lower() in POSSIBLE_MODE_OPF:
        raise RuntimeError("Please provide an argument \"mode_opf\" among {}".format(POSSIBLE_MODE_OPF))
    rescaled_min = args.rescaled_min
    try:
        rescaled_min = int(rescaled_min)
    except:
        RuntimeError("\"rescaled_min\" argument should be convertible to an integer.")

    if not  60 % rescaled_min == 0:
        raise RuntimeError("\"rescaled_min\" argument should be multiple of 5 (so 5, or 15 is ok, but not 17 nor 3)")

    # **  **  **  **  ** 
    # Load the PyPSA grid
    net = pypsa.Network(import_name=args.grid_path)

    # Load consumption data without index
    print ('Reading input data -> load...')
    demand = pd.read_csv(args.conso_full_path)
#     demand.drop('datetime', axis=1, inplace=True)

    # Check if gen contraints are given
    gen_const = {'p_max_pu': None, 'p_min_pu': None}
    if args.gen_const:
        gen_max_constraints = pd.read_csv(os.path.join(args.gen_const), 'gen_max_pu.csv.bz2')
        gen_min_constraints = pd.read_csv(os.path.join(args.gen_const), 'gen_min_pu.csv.bz2')
        gen_const.update({'p_max_pu': gen_max_constraints, 'p_min_pu': gen_min_constraints})

    # Define parameters
    params = {'step_opf_min': args.rescaled_min, 
               'mode_opf': args.mode_opf.lower(),
             }
    # Run Economic Dispatch
    prod_p_dispatch = main_run_disptach(net, 
                                        demand, 
                                        gen_constraints=gen_const, 
                                        params=params)

    destination_path = os.path.abspath(args.dest_path)
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    # Save as csv
    prod_p_dispatch.to_csv(
        os.path.join(destination_path, 'prod_p.csv.bz2'),
        sep=';',
        float_format=cst.FLOATING_POINT_PRECISION_FORMAT,
        index=False)
    
    print('OPF Done......')
