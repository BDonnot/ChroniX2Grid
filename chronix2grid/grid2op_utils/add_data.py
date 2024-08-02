# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)
import os
import json
from multiprocessing import Pool

import grid2op
from chronix2grid.grid2op_utils.utils import generate_a_scenario, get_last_scenario_id
from numpy.random import default_rng

  
# wrapper function for generate_a_scenario
def generate_a_scenario_wrapper(args):
    (path_env, name_gen, gen_type, output_dir,
        start_date, dt, scen_id, load_seed, renew_seed,
        gen_p_forecast_seed, handle_loss, files_to_copy,
        save_ref_curve, day_lag, tol_zero, debug, load_weekly_pattern, 
        kwargs) = args
    res_gen = generate_a_scenario(path_env=path_env,
                                  name_gen=name_gen, 
                                  gen_type=gen_type,
                                  output_dir=output_dir,
                                  start_date=start_date,
                                  dt=dt,
                                  scen_id=scen_id,
                                  load_seed=load_seed,
                                  renew_seed=renew_seed, 
                                  gen_p_forecast_seed=gen_p_forecast_seed,
                                  handle_loss=handle_loss,
                                  files_to_copy=files_to_copy,
                                  save_ref_curve=save_ref_curve,
                                  day_lag=day_lag,
                                  tol_zero=tol_zero,
                                  debug=debug,
                                  load_weekly_pattern=load_weekly_pattern,
                                  **kwargs)
    return res_gen


def add_data(env: grid2op.Environment.Environment,
             seed=None,
             nb_scenario=1,
             nb_core=1,  # TODO
             with_loss=True,
             files_to_copy=("maintenance_meta.json",
                            "params_load.json",
                            "params_forecasts.json"),
             save_ref_curve=False,
             day_lag=6,  # TODO 6 because it's 2050
             debug=False,
             tol_zero=1e-3,
             load_weekly_pattern=None,
             **kwargs
             ):
    """This function adds some data to already existing scenarios.
    
    .. warning::
        You should not start this function twice. Before starting a new run, make sure the previous one has terminated (otherwise you might
        erase some previously generated scenario)

    Parameters
    ----------
    env : _type_
        The grid2op environment
    seed:
        The seed to use (the same seed is guaranteed to generate the same scenarios)
    nb_scenario: ``int``
        The number of scenarios to generate
    nb_core: ``int``
        The number of core you want to use (to speed up the generation process)
    with_loss: ``bool``
        Do you make sure that the generated data will not be modified too much when running with grid2op (default = True).
        Setting it to False will speed up (by quite a lot) the generation process, but will degrade the data quality.
    load_weekly_pattern: pd.DataFrame
        The pattern used as a reference to generate the loads.
        
        It sould be a dataframe with 2 columns: `datetime` and `test`.
        
        In the first column (`datetime`) you should have time stamp (format "%Y-%m-%d %H:%M:%S" 
        *eg* `2017-01-07 23:55:00`). The second oui should have a number "approximately one" which
        gives the relative ratio of demand for the whole grid at this time stamp.
        
        We only tested this when the data was given at 5 minutes resolution (two consecutive rows are 
        distant from 5 minutes) and with the equivalent of 2 years of data. It might work
        (or not) in other cases...
        
    """
    # required parameters
    env_name = type(env).env_name
    output_dir = os.path.join(env.get_path_env(), "chronics")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    last_scen = get_last_scenario_id(output_dir)
    scen_ids = [f"{el}" for el in range(last_scen +1, last_scen + 1 + nb_scenario)]
    with open(os.path.join(env.get_path_env(), "scenario_params.json"), "r", encoding="utf-8") as f:
        dict_ref = json.load(f)
        
    dt = dict_ref["dt"]
    li_months = dict_ref["all_dates"]

    # generate the seeds
    if seed is not None:
        prng = default_rng(seed)
    else:
        prng = default_rng()
        
    load_seeds = []
    renew_seeds = []
    gen_p_forecast_seeds = []
    for scen_id in scen_ids:
        for start_date in li_months:
            load_seed, renew_seed, gen_p_forecast_seed = prng.integers(2**32 - 1, size=3)
            load_seeds.append(load_seed)
            renew_seeds.append(renew_seed)
            gen_p_forecast_seeds.append(gen_p_forecast_seed)
    
    # generate the data
    path_env = env.get_path_env()
    name_gen = env.name_gen
    gen_type = env.gen_type
    argss = []
    for j, scen_id in enumerate(scen_ids):
        for i, start_date in enumerate(li_months):
            # if start_date != "2035-03-12":
            #     continue
            seed_num = i + j * len(li_months)
            argss.append((path_env,
                          name_gen,
                          gen_type,
                          output_dir,
                          start_date,
                          dt,
                          scen_id,
                          load_seeds[seed_num],
                          renew_seeds[seed_num],
                          gen_p_forecast_seeds[seed_num],
                          with_loss,
                          files_to_copy,
                          save_ref_curve,
                          day_lag,
                          tol_zero,
                          debug,
                          load_weekly_pattern,
                          kwargs,
                          ))
    if nb_core == 1:
        errors = {}
        for args in argss:
            res_gen = generate_a_scenario_wrapper(args)
            error_, *_ = res_gen
            if error_ is not None:
                error_date = args[4]
                print("=============================")
                print(f"     Error for {error_date} {scen_id}        ")
                print(f"{error_}")
                print("=============================")
                errors[f'{error_date}_{scen_id}'] = f"{error_}"
                
                # load previous data
                path_json_error = os.path.join(output_dir, "errors.json")
                if os.path.exists(path_json_error):
                    with open(path_json_error, "r", encoding="utf-8") as f:
                        err_tmp = json.load(f)
                    for k in err_tmp:
                        errors[k] = err_tmp[k]
                    
                # write the log
                with open(path_json_error, "w", encoding="utf-8") as f:
                    json.dump(errors, fp=f)
    else:
        with Pool(nb_core) as p:
            p.map(generate_a_scenario_wrapper, argss)
