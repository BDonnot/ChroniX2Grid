# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)

import os
import json
import pandas as pd
import numpy as np
import pathlib

from chronix2grid.getting_started.example.input.generation.patterns import ref_pattern_path
from chronix2grid.generation.consumption import ConsumptionGeneratorBackend
from chronix2grid.generation.consumption.consumption_utils import (get_seasonal_pattern,
                                                                   compute_load_pattern)

from chronix2grid.generation.consumption.generate_load import get_add_dim as get_add_dim_load

from chronix2grid.grid2op_utils.noise_generation_utils import (get_forecast,
                                                               generate_noise)
       
       
def get_data_frames(load_p, load_p_for, loads_charac, datetime_index, nb_h):
    load_p_for = np.transpose(load_p_for, (1, 0))
    load_p = np.transpose(load_p, (1, 0))
    load_p_df = pd.DataFrame(load_p, columns=loads_charac["name"])
    load_p_for_df = pd.DataFrame(load_p_for, columns=loads_charac["name"])
    
    load_p_df["datetime"] = datetime_index
    load_p_df.set_index('datetime', inplace=True)
    load_p_df = load_p_df.sort_index(ascending=True)
    size_after = len(load_p_df) - 1
    load_p_df = load_p_df.head(size_after)
    
    load_p_for_df = load_p_for_df.head(size_after * nb_h)
    return load_p_df, load_p_for_df
    
    
def get_load_ref(loads_charac, load_params, load_weekly_pattern, isoweekday_lwp=None, hour_minutes_lwp=None):
    weekly_pattern = load_weekly_pattern['test'].values
    pmax_weekly_pattern = []
    for index, name in enumerate(loads_charac['name']):
        mask = (loads_charac['name'] == name)
        Pmax = loads_charac[mask]['Pmax'].values[0]
        tmp_ = Pmax * compute_load_pattern(load_params, weekly_pattern, index,
                                           isoweekday_lwp=isoweekday_lwp,
                                           hour_minutes_lwp=hour_minutes_lwp)
        pmax_weekly_pattern.append(tmp_.reshape(1, -1))
    load_ref = np.concatenate(pmax_weekly_pattern)
    return load_ref


def generate_new_loads(load_seed,
                       load_params,
                       forecasts_params,
                       loads_charac,
                       gen_charac,
                       load_weekly_pattern,
                       data_type='temperature',
                       day_lag=6,
                       return_ref_curve=True):    
    # read the parameters from the inputs
    nb_load = len(loads_charac['name'])
    nb_h = len(forecasts_params["h"])
    datetime_index = pd.date_range(start=load_params['start_date'],
                                   end=load_params['end_date'],
                                   freq=str(load_params['dt']) + 'min')
    nb_t = datetime_index.shape[0] 
    std_temperature_noise = float(load_params['std_temperature_noise'])
    
    # retrieve the reference curve "bar"
    if isinstance(load_weekly_pattern, pd.DataFrame):
        datetime_lwp = pd.to_datetime(load_weekly_pattern["datetime"], format="%Y-%m-%d %H:%M:%S")
    else:
        raise NotImplementedError("TODO load weekly pattern !")
    isoweekday = np.array([el.isoweekday() for el in datetime_lwp])
    hour_minutes = np.array([el.hour * 60 + el.minute for el in datetime_lwp])
    load_ref = get_load_ref(loads_charac, load_params, load_weekly_pattern,
                            isoweekday_lwp=isoweekday, hour_minutes_lwp=hour_minutes)
    # (nb_load, nb_t)
    
    # Compute seasonal pattern
    seasonal_pattern_unit = get_seasonal_pattern(load_params)
    seasonal_pattern_load = np.tile(seasonal_pattern_unit, (nb_load, 1))
    # (nb_load, nb_t)
    if return_ref_curve:
        load_hat = load_ref * seasonal_pattern_load
        load_hat = 1.0 * load_hat.T[:-1,:]
    
    # compute the noise for the loads
    loads_noise, hs, std_hs = generate_noise(loads_charac,
                                             gen_charac,
                                             forecasts_params,
                                             load_params,
                                             load_seed,
                                             data_type,
                                             get_add_dim_load,
                                             nb_t,
                                             load_params,
                                             loads_charac,
                                             nb_load,
                                             add_h0=True)
        
    # generate the "real" loads
    load_p = load_ref * (std_temperature_noise * loads_noise[:,:,0] + seasonal_pattern_load)
    # shape (n_load, nb_t)
    
    # now generate all the forecasts
    load_p_for = get_forecast(load_p, loads_noise, hs, std_hs, loads_charac)
    # shape (n_load, nb_t, nb_h)
    
    # create the data frames
    load_p_df, load_p_for_df = get_data_frames(load_p,
                                               load_p_for,
                                               loads_charac,
                                               datetime_index,
                                               nb_h)
    if not return_ref_curve:
        return load_p_df, load_p_for_df
    return load_p_df, load_p_for_df, load_hat


def generate_loads(path_env,
                   load_seed,
                   forecast_prng,
                   start_date_dt,
                   end_date_dt,
                   dt,
                   number_of_minutes,
                   generic_params,
                   load_q_from_p_coeff_default=0.7,
                   day_lag=6,
                   load_weekly_pattern=None):
    """
    This function generates the load for each consumption on a grid

    Parameters
    ----------
    path_env : _type_
        _description_
    load_seed : _type_
        _description_
    start_date_dt : _type_
        _description_
    end_date_dt : _type_
        _description_
    dt : _type_
        _description_
    number_of_minutes : _type_
        _description_
    generic_params : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    with open(os.path.join(path_env, "params_load.json"), "r") as f:
        load_params = json.load(f)
    load_params["start_date"] = start_date_dt
    load_params["end_date"] = end_date_dt
    load_params["dt"] = int(dt)
    load_params["T"] = number_of_minutes
    load_params["planned_std"] = float(generic_params["planned_std"])
    
    if "load_q_from_p_coeff" in load_params:
        load_q_from_p_coeff = float(load_params["load_q_from_p_coeff"])
    else:
        load_q_from_p_coeff = load_q_from_p_coeff_default
        
    forecasts_params = {}
    new_forecasts = False
    path_for_ = os.path.join(path_env, "params_forecasts.json")
    if os.path.exists(path_for_):
        with open(path_for_, "r") as f:
            forecasts_params = json.load(f)
        new_forecasts = True
    
    loads_charac = pd.read_csv(os.path.join(path_env, "loads_charac.csv"), sep=",")
    gen_charac = pd.read_csv(os.path.join(path_env, "prods_charac.csv"), sep=",")
    if load_weekly_pattern is None:
        load_weekly_pattern = pd.read_csv(os.path.join(ref_pattern_path, "load_weekly_pattern.csv"), sep=",")
    else:
        if isinstance(load_weekly_pattern, (str, os.PathLike)):
            load_weekly_pattern = pathlib.Path(load_weekly_pattern)
        elif isinstance(load_weekly_pattern, (np.ndarray, pd.DataFrame)):
            load_weekly_pattern = pd.DataFrame(load_weekly_pattern)
        else:
            raise TypeError("load_weekly_pattern kwargs should be either a str "
                            "(or pathlike) or a pandas Dataframe (or numpy array) and not "
                            f"{type(load_weekly_pattern)}")
    
    if new_forecasts:
        # TODO load weekly pattern !
        load_p, load_p_forecasted, load_ref_curve = generate_new_loads(load_seed,
                                                                       load_params,
                                                                       forecasts_params,
                                                                       loads_charac,
                                                                       gen_charac,
                                                                       load_weekly_pattern,
                                                                       day_lag=day_lag,
                                                                       return_ref_curve=True)
    else:
        load_generator = ConsumptionGeneratorBackend(out_path=None,
                                                     seed=load_seed, 
                                                     params=load_params,
                                                     loads_charac=loads_charac,
                                                     write_results=False,
                                                     load_config_manager=None,
                                                     day_lag=day_lag)
        
        load_p, load_p_forecasted, load_ref_curve = load_generator.run(load_weekly_pattern=load_weekly_pattern,  # TODO load weekly pattern
                                                                       return_ref_curve=True,
                                                                       use_legacy=False)
    
    load_q = load_p * load_q_from_p_coeff
    load_q_forecasted = load_p_forecasted * load_q_from_p_coeff
    return (new_forecasts, forecasts_params, load_params, loads_charac,
            load_p, load_q, load_p_forecasted, load_q_forecasted, load_ref_curve)
