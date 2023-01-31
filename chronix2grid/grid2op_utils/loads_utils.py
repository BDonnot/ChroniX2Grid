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
import warnings
import itertools
    
from numpy.random import default_rng


from chronix2grid.getting_started.example.input.generation.patterns import ref_pattern_path
from chronix2grid.generation.consumption import ConsumptionGeneratorBackend
from chronix2grid.generation.consumption.generate_load import get_add_dim
from chronix2grid.generation.consumption.consumption_utils import (get_seasonal_pattern,
                                                                    compute_load_pattern)
from chronix2grid.generation.generation_utils import get_nx_ny_nt


def generate_coords_mesh(Nx_comp, Ny_comp, Nt_comp, Nh_comp, size,
                         ratio_adjust=1.0):
    # distort the distance for the mesh (because we use kNN later on)
    max_mesh = max(Nx_comp, Ny_comp, Nt_comp, Nh_comp)
    rho_mesh_x = max_mesh / Nx_comp * ratio_adjust
    rho_mesh_y = max_mesh / Ny_comp * ratio_adjust
    rho_mesh_t = max_mesh / Nt_comp * ratio_adjust
    rho_mesh_h = max_mesh / Nh_comp * ratio_adjust
    
    # generate coordinates of the mesh
    coords_mesh = np.empty((size, 4))
    for i, row in enumerate(itertools.product(range(Nx_comp), range(Ny_comp), range(Nt_comp), range(Nh_comp))):
        coords_mesh[i,:] = row
    coords_mesh[:,0] /= (Nx_comp - 1) * rho_mesh_x
    coords_mesh[:,1] /= (Ny_comp - 1) * rho_mesh_y
    coords_mesh[:,2] /= (Nt_comp - 1) * rho_mesh_t
    coords_mesh[:,3] /= (Nh_comp - 1) * rho_mesh_h
    return coords_mesh, rho_mesh_x, rho_mesh_y, rho_mesh_t, rho_mesh_h
    

def get_load_mesh_tmp(nb_t, hs_, hs_mins, rho_mesh_t, rho_mesh_h):
    nb_h = len(hs_)
    # now find the noise for all the loads
    load_mesh_tmp = np.empty((nb_t * nb_h, 4))
    for i, row in enumerate(itertools.product(range(nb_t), hs_)):
        load_mesh_tmp[i,:] = (0, 0, *row)
    # compute where the "point" is on the mesh
    load_mesh_tmp[:,2] += 1
    load_mesh_tmp[:,2] /= (nb_t + 1)
    load_mesh_tmp[:,3] /= (np.max(hs_mins) + 5)
    
    # don't forget to re normalize it
    load_mesh_tmp[:,2] /= rho_mesh_t
    load_mesh_tmp[:,3] /= rho_mesh_h
    return load_mesh_tmp


def compute_noise(load_mesh_tmp, loads_charac, model,
                  range_x, range_y,
                  delta_x, delta_y,
                  rho_mesh_x, rho_mesh_y,
                  nb_t, nb_h, nb_load):
    
    loads_noise = np.zeros((nb_load, nb_t, 1 + nb_h))
    for load_id, (load_x, load_y) in loads_charac[["x", 'y']].iterrows():
        # compute where the "point" is on the mesh
        load_mesh = 1.0 * load_mesh_tmp
        load_mesh[:,0] = (load_x - range_x[0])/delta_x
        load_mesh[:,1] = (load_y - range_y[0])/delta_y
        
        # don't forget to re normalize it
        load_mesh[:,0] /= rho_mesh_x
        load_mesh[:,1] /= rho_mesh_y
        
        noise_load = model.predict(load_mesh)
        loads_noise[load_id, :, :] = noise_load.reshape(nb_t, nb_h + 1)
        
    # renormalize the noise
    loads_noise -= loads_noise.mean()
    loads_noise /= loads_noise.std()  
    return loads_noise
    
    
def get_forecast(load_p, loads_noise, hs, std_hs, loads_charac):
    nb_load = load_p.shape[0]
    nb_t = load_p.shape[1]
    nb_h = len(hs)
    
    load_p_for = np.stack([np.roll(load_p, -h) for h in hs], axis=2)
    
    loads_noise_forecast = 1.0 * loads_noise[:,:,1:]
    loads_noise_forecast *= np.array(std_hs).reshape(1,12)
    loads_noise_forecast *= loads_charac["Pmax"].values.reshape(nb_load, 1, 1)
    load_p_for += loads_noise_forecast
    # keep the last value for the "forecasts outside the episode"
    for col, ts in enumerate(hs):
        load_p_for[:, (nb_t - ts):, col] = load_p_for[:, (nb_t - ts) - 1, col].reshape(nb_load, 1)
    
    # put everything in the right shape
    load_p_for = load_p_for.reshape(nb_load, nb_t * nb_h)
    return load_p_for
       
       
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
    
    
def get_knn_fitted(forecasts_params, coords_mesh, noise_mesh):
    from sklearn.neighbors import KNeighborsRegressor
    if "n_neighbors" in forecasts_params:
        n_neighbors = int(forecasts_params["n_neighbors"])
        assert n_neighbors > 0, f"n_neighbors parameters should be > 0 and is currently {n_neighbors}"
    else:
        n_neighbors = 16 * 15 - 32  # 16 = les 16 arretes du cube 4d dans lequel il est
    
    if "leaf_size_knn" in forecasts_params:
        leaf_size_knn = int(forecasts_params["leaf_size_knn"])
    else:
        leaf_size_knn = 100  # found to be the best compromise
    
    if "algorithm_knn" in forecasts_params:
        algorithm_knn = str(forecasts_params["algorithm_knn"])
    else:
        algorithm_knn = "auto"  # should be good
        
    model = KNeighborsRegressor(n_neighbors=n_neighbors,  
                                weights="distance",
                                leaf_size=leaf_size_knn,
                                algorithm=algorithm_knn)
    model.fit(coords_mesh, noise_mesh.reshape(-1))
    return model


def get_forecast_parameters(forecasts_params, load_params):    
    hs_mins = [int(el) for el in forecasts_params["h"]]
    # convert the h in "number of steps" (and not in minutes)
    hs = [h // load_params['dt'] for h in hs_mins]
    # check consistency
    for h, h_min in zip(hs, hs_mins):
        if h * load_params['dt'] != h_min:
            # TODO logger
            warnings.warn("Some forecast horizons are not muliple of the duration of a steps. They will be rounded")
    
    # load the parameters "h" for the forecasts
    std_hs = [float(el) for el in forecasts_params["h_std_load"]]
    for std in std_hs:
        assert std > 0, f"all parameters of 'h_std_load' should be >0 we found {std}"
    assert len(std_hs) == len(hs), "you should provide as much 'error' as there are forecasts horizon"
    return hs_mins, hs, std_hs
    
    
def get_load_ref(loads_charac, load_params, load_weekly_pattern, day_lag):
    weekly_pattern = load_weekly_pattern['test'].values
    pmax_weekly_pattern = []
    for index, name in enumerate(loads_charac['name']):
        mask = (loads_charac['name'] == name)
        Pmax = loads_charac[mask]['Pmax'].values[0]
        tmp_ = Pmax * compute_load_pattern(load_params, weekly_pattern, index, day_lag)
        pmax_weekly_pattern.append(tmp_.reshape(1, -1))
    load_ref = np.concatenate(pmax_weekly_pattern)
    return load_ref


def resize_mesh_factor(loads_charac, gen_charac, ratio_border=20.):
    # in the generation we will map this mesh to [0., 1.]
    range_x = min(loads_charac["x"].min(), gen_charac["x"].min()), max(loads_charac["x"].max(), gen_charac["x"].max())
    range_y = min(loads_charac["y"].min(), gen_charac["y"].min()), max(loads_charac["y"].max(), gen_charac["y"].max())
    delta_x_ = range_x[1] - range_x[0]
    range_x = (range_x[0] - delta_x_ / ratio_border, range_x[1] + delta_x_ / ratio_border)
    delta_y_ = range_y[1] - range_y[0]
    range_y = (range_y[0] - delta_y_/ ratio_border, range_y[1] + delta_y_ / ratio_border)
    # the (+/- delta_x_ / 20.) is to handle side effect at the border of the mesh
    
    delta_x = range_x[1] - range_x[0]
    delta_y = range_y[1] - range_y[0]
    return delta_x, delta_y, range_x, range_y
   
    
def get_iid_noise(load_seed, load_params, forecasts_params,
                  loads_charac, data_type):
    prng = default_rng(load_seed)
    add_dim = get_add_dim(load_params, loads_charac)
    Nx_comp, Ny_comp, Nt_comp = get_nx_ny_nt(data_type, load_params, add_dim)
    Nh_comp = forecasts_params["nb_h_iid"]
    noise_mesh = prng.normal(0, 1, (Nx_comp, Ny_comp, Nt_comp, Nh_comp))
    return noise_mesh, (Nx_comp, Ny_comp, Nt_comp, Nh_comp)


def generate_new_loads(load_seed,
                       load_params,
                       forecasts_params,
                       loads_charac,
                       gen_charac,
                       load_weekly_pattern,
                       data_type='temperature',
                       day_lag=6):    
    # read the parameters from the inputs
    nb_load = len(loads_charac['name'])
    nb_h = len(forecasts_params["h"])
    datetime_index = pd.date_range(start=load_params['start_date'],
                                   end=load_params['end_date'],
                                   freq=str(load_params['dt']) + 'min')
    nb_t = datetime_index.shape[0]
    hs_mins, hs, std_hs = get_forecast_parameters(forecasts_params, load_params)    
    std_temperature_noise = float(load_params['std_temperature_noise'])
    
    # compute the "real" size of the mesh 
    delta_x, delta_y, range_x, range_y = resize_mesh_factor(loads_charac, gen_charac)
    
    # generate the independant data on the mesh
    tmp_ = get_iid_noise(load_seed, load_params, forecasts_params,
                         loads_charac, data_type)
    noise_mesh, (Nx_comp, Ny_comp, Nt_comp, Nh_comp) = tmp_
    # retrieve the reference curve "bar"
    load_ref = get_load_ref(loads_charac, load_params, load_weekly_pattern, day_lag)
    # (nb_load, nb_t)
    
    # Compute seasonal pattern
    seasonal_pattern_unit = get_seasonal_pattern(load_params)
    seasonal_pattern_load = np.tile(seasonal_pattern_unit, (nb_load, 1))
    # (nb_load, nb_t)
    
    # get the inteporlation on the mesh
    res_mesh = generate_coords_mesh(Nx_comp,
                                    Ny_comp,
                                    Nt_comp,
                                    Nh_comp,
                                    noise_mesh.size) 
    coords_mesh, rho_mesh_x, rho_mesh_y, rho_mesh_t, rho_mesh_h = res_mesh
    
    # "fit" the kNN    
    model = get_knn_fitted(forecasts_params, coords_mesh, noise_mesh)
    
    # get the "temporary" for the load coordinates
    hs_ = [0] + hs
    load_mesh_tmp = get_load_mesh_tmp(nb_t, hs_, hs_mins,
                                      rho_mesh_t, rho_mesh_h)
    
    loads_noise = compute_noise(load_mesh_tmp, loads_charac, model,
                                range_x, range_y,
                                delta_x, delta_y,
                                rho_mesh_x, rho_mesh_y,
                                nb_t, nb_h, nb_load)
        
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
    
    return load_p_df, load_p_for_df


def generate_loads(path_env,
                   load_seed,
                   forecast_prng,
                   start_date_dt,
                   end_date_dt,
                   dt,
                   number_of_minutes,
                   generic_params,
                   load_q_from_p_coeff=0.7,
                   day_lag=6):
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
    
    forecasts_params = {}
    new_forecasts = False
    path_for_ = os.path.join(path_env, "params_forecasts.json")
    if os.path.exists(path_for_):
        with open(path_for_, "r") as f:
            forecasts_params = json.load(f)
        new_forecasts = True
    
    loads_charac = pd.read_csv(os.path.join(path_env, "loads_charac.csv"), sep=",")
    gen_charac = pd.read_csv(os.path.join(path_env, "prods_charac.csv"), sep=",")
    load_weekly_pattern = pd.read_csv(os.path.join(ref_pattern_path, "load_weekly_pattern.csv"), sep=",")
    
    if new_forecasts:
        load_p, load_p_forecasted = generate_new_loads(load_seed,
                                                       load_params,
                                                       forecasts_params,
                                                       loads_charac,
                                                       gen_charac,
                                                       load_weekly_pattern,
                                                       day_lag=day_lag)
    else:
        load_generator = ConsumptionGeneratorBackend(out_path=None,
                                                     seed=load_seed, 
                                                     params=load_params,
                                                     loads_charac=loads_charac,
                                                     write_results=False,
                                                     load_config_manager=None,
                                                     day_lag=day_lag)
        
        load_p, load_p_forecasted = load_generator.run(load_weekly_pattern=load_weekly_pattern)
    
    load_q = load_p * load_q_from_p_coeff
    load_q_forecasted = load_p_forecasted * load_q_from_p_coeff
    return new_forecasts, load_p, load_q, load_p_forecasted, load_q_forecasted
