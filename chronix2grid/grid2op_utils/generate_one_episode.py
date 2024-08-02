# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)
import os
import numpy as np
import json

import grid2op
from chronix2grid.grid2op_utils.utils import generate_a_scenario
from numpy.random import default_rng


def generate_one_episode(env: grid2op.Environment.Environment,
                         dict_ref,
                         dt,
                         start_date,
                         nb_steps=None,
                         seed=None,
                         with_loss=True,
                         files_to_copy=("maintenance_meta.json", "params_load.json", "params_forecasts.json")):
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
        
    """
    # generate the seeds
    if seed is not None:
        prng = default_rng(seed)
    else:
        prng = default_rng()
    
    load_seed, renew_seed, gen_p_forecast_seed = prng.integers(2**32 - 1, size=3)
    path_env = env.get_path_env()
    name_gen = env.name_gen
    gen_type = env.gen_type
    scen_id = "0"
    error_ = "first"
    output_dir = None
    while error_ is not None:
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
                                      handle_loss=with_loss,
                                      nb_steps=nb_steps,
                                      files_to_copy=files_to_copy)
        error_, quality_, load_p, load_p_forecasted, load_q, load_q_forecasted, res_gen_p_df, res_gen_p_forecasted_df = res_gen
    return load_p, load_p_forecasted, load_q, load_q_forecasted, res_gen_p_df, res_gen_p_forecasted_df
