# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)

import os
import json
from numpy.random import default_rng

# Other Python libraries
import pandas as pd
import numpy as np

# Libraries developed for this module
from . import consumption_utils as conso
from .. import generation_utils as utils


def get_add_dim(params, loads_charac):
    add_dim = 0
    dx_corr = int(params['dx_corr'])
    dy_corr = int(params['dy_corr'])
    for x,y  in zip(loads_charac["x"], loads_charac["y"]):
        x_plus = int(x // dx_corr + 1)
        y_plus = int(y // dy_corr + 1)
        add_dim = max(y_plus, add_dim)
        add_dim = max(x_plus, add_dim)
    return add_dim


def main(scenario_destination_path,
         seed,
         params,
         loads_charac,
         load_weekly_pattern,
         write_results=True,
         day_lag=0,
         return_ref_curve=False,
         use_legacy=True):
    """
    This is the load generation function, it allows you to generate consumption chronics based on demand nodes characteristics and on weekly demand patterns.

    Parameters
    ----------
    scenario_destination_path (string): where results are written
    seed (int): random seed of the scenario
    params (dict): system params such as timestep or mesh characteristics
    loads_charac (pandas.DataFrame): characteristics of loads node such as Pmax and type of demand
    load_weekly_pattern (pandas.DataFrame): 5 minutes weekly load chronic that represent specificity of the demand context
    write_results (boolean): whether to write_results or not. Default is True

    Returns
    -------
    pandas.DataFrame: loads chronics generated at every node with additional gaussian noise
    pandas.DataFrame: loads chronics forecasted for the scenario without additional gaussian noise
    """

    # Set random seed of scenario
    prng = default_rng(seed)
    #np.random.seed(seed) #older version - to be removed

    # Define reference datetime indices
    datetime_index = pd.date_range(
        start=params['start_date'],
        end=params['end_date'],
        freq=str(params['dt']) + 'min')

    add_dim = get_add_dim(params, loads_charac)
    
    # Generate GLOBAL temperature noise
    print('Computing global auto-correlated spatio-temporal noise for thermosensible demand...') ## temperature is simply to reflect the fact that loads is correlated spatially, and so is the real "temperature". It is not the real temperature.
    temperature_noise = utils.generate_coarse_noise(prng, params, 'temperature', add_dim=add_dim)

    print('Computing loads ...')
    start_day = datetime_index[0]
    loads_series = conso.compute_loads(loads_charac,
                                       temperature_noise,
                                       params,
                                       load_weekly_pattern,
                                       start_day=start_day,
                                       add_dim=add_dim,
                                       day_lag=day_lag,
                                       return_ref_curve=return_ref_curve,
                                       use_legacy=use_legacy)
    if return_ref_curve:
        loads_series, ref_curve = loads_series
    loads_series['datetime'] = datetime_index

    # Save files
    if scenario_destination_path is not None:
        print('Saving files in zipped csv in "{}"'.format(scenario_destination_path))
        if not os.path.exists(scenario_destination_path):
            os.makedirs(scenario_destination_path)
            
    load_p_forecasted = conso.create_csv(prng, loads_series, scenario_destination_path,
                                        forecasted=True, reordering=True,
                                        shift=True, write_results=write_results, index=False)
    load_p = conso.create_csv(
        prng,
        loads_series, scenario_destination_path,
        reordering=True,
        noise=params['planned_std'],
        write_results=write_results,
        index=False
    )
    if not return_ref_curve:
        return load_p, load_p_forecasted
    else:
        return load_p, load_p_forecasted, ref_curve
