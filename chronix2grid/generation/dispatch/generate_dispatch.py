# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)

from .PypsaDispatchBackend.EDispatch_L2RPN2020 import RampMode # TODO: Supprimer cette dépendance car pas utile (utiliser utils dans chronix2grid)
from .dispatch_loss_utils import run_grid2op_simulation_donothing, correct_scenario_loss, move_chronics_temporarily, \
    remove_temporary_chronics, remove_simulation_data, move_env_temporarily
import shutil
import os
import pathlib


def main(dispatcher, input_folder, output_folder, grid_folder, seed, params, params_opf,renewable_in_OPF=False):
    """

    Parameters
    ----------
    dispatcher : Dispatcher
        The Dispatcher instance used for running the OPF
    input_folder : str
        The path to the directory containing the inputs for the dispatch
    output_folder
        The path of the directory that will receive the outputs of the dispatch
    grid_folder
        The path of the directory that contains grid information (used only if loss simulation are activated)
    seed : int
        Random seed for parallel execution
    params_opf : dict
        Options for the OPF

    Returns
    -------
    DispatchResults
        The namedtuple return by Dispatcher.run method
    """

    #np.random.seed(seed) # already done before
    if ("renewable_in_opf" in params.keys()):
        renewable_in_OPF=params["renewable_in_opf"]
    else:
        renewable_in_OPF=False
    hydro_constraints = dispatcher.make_hydro_constraints_from_res_load_scenario()

    if renewable_in_OPF:
        load_with_losses = dispatcher.net_load(params_opf['losses_pct'],
                                               name=dispatcher.loads.index[0],include_renewable=False)
        ##########
        #Bypass solar and wind for now
        dispatch_results = dispatcher.run(
            load=load_with_losses,
            total_solar=dispatcher.solar_p.sum(axis=1),
            total_wind=dispatcher.wind_p.sum(axis=1),
            params=params_opf,
            gen_constraints=hydro_constraints,
            ramp_mode=parse_ramp_mode(params_opf['ramp_mode']),
            by_carrier=params_opf['dispatch_by_carrier'],
            pyomo=params_opf['pyomo'],
            solver_name=params_opf['solver_name']
        )
        #dispatch_results.chronix.prods_dispatch=dispatch_results.chronix.prods_dispatch.drop(["agg_wind", "agg_solar"], axis=1)
    else:
        load_with_losses = dispatcher.net_load(params_opf['losses_pct'],
                                               name=dispatcher.loads.index[0], include_renewable=True)
        ##########
        # Bypass solar and wind for now
        dispatch_results = dispatcher.run(
            load=load_with_losses,
            total_solar=None,
            total_wind=None,
            params=params_opf,
            gen_constraints=hydro_constraints,
            ramp_mode=parse_ramp_mode(params_opf['ramp_mode']),
            by_carrier=params_opf['dispatch_by_carrier'],
            pyomo=params_opf['pyomo'],
            solver_name=params_opf['solver_name']
        )
        #####
        # These column should be removed as wind and solar were not considered in the opf here
        if(dispatch_results):
            dispatch_results.chronix.prods_dispatch = dispatch_results.chronix.prods_dispatch.drop(
                ["agg_wind", "agg_solar"], axis=1)

    dispatcher.save_results(params, output_folder)

    is_dispatch_successful=(dispatcher.chronix_scenario.prods_dispatch is not None) and (len(dispatcher.chronix_scenario.prods_dispatch.columns)>=1)
    if params_opf["loss_grid2op_simulation"] and is_dispatch_successful:
        new_prod_p, new_prod_forecasted_p = simulate_loss(grid_folder, output_folder, params_opf, write_results = True)
        dispatch_results = update_results_loss(dispatch_results, new_prod_p, params_opf)

    return dispatch_results

def update_results_loss(dispatch_results, new_prod_p, params_opf):
    dispatch_results[0].prods_dispatch[params_opf['nameSlack']] = new_prod_p[params_opf['nameSlack']]
    return dispatch_results

def simulate_loss(input_folder, output_folder, params_opf, write_results = True):
    scenario_folder_path = output_folder
    grid_folder_g2op = input_folder

    #because of possible multiprocessing, me need to create a new grid2op grid folder for each scenario
    #to avoid interferences
    grid_temporary_path=move_env_temporarily(scenario_folder_path, grid_folder_g2op)

    move_chronics_temporarily(scenario_folder_path, grid_temporary_path)
    agent_results_path = str(pathlib.Path(scenario_folder_path).parent.parent)
    # try:

    episode_data = run_grid2op_simulation_donothing(grid_temporary_path, scenario_folder_path,write_results=write_results,agent_results_path=agent_results_path)
    # except RuntimeError:
    #     remove_temporary_chronics(grid_folder_g2op)
    #     raise RuntimeError("Error in Grid2op simulation, temporary folder deleted")
    dispatch_results_corrected = correct_scenario_loss(scenario_folder_path, params_opf, grid_folder_g2op, episode_data)

    #remove temporary env folder
    shutil.rmtree(grid_temporary_path)

    # remove_simulation_data(scenario_folder_path)
    return dispatch_results_corrected


def parse_ramp_mode(mode):
    """
    Parse a string representing the difficulty of the ramps in the OPF into
    a RampMode Enum value
    Parameters
    ----------
    mode : str
        The difficulty mode for the ramps in the OPF

    Returns
    -------
    RampMode
        The encoded RampMode value

    """
    if mode == 'hard':
        return RampMode.hard
    if mode == 'medium':
        return RampMode.medium
    if mode == 'easy':
        return RampMode.easy
    if mode == '':
        return RampMode.none
    raise ValueError(f'mode only takes values from (hard, medium, easy, none), '
                     '{mode} was passed')
