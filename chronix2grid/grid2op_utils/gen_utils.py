# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)

import json
import pandas as pd
import os
import cvxpy as cp
import numpy as np

from chronix2grid.generation.renewable.generate_solar_wind import get_add_dim as get_add_dim_renew
from chronix2grid.grid2op_utils.noise_generation_utils import (get_forecast,
                                                               generate_noise)


def fix_nan_hydro_i_dont_know_why(arr):
    # economic_dispatch.make_hydro_constraints_from_res_load_scenario() sometimes return all nans for a given row...
    # don't ask me why
    if np.all(np.isfinite(arr)):
        # nothing to do, data are correct
        return 1.0 * arr
    if np.all(~np.isfinite(arr)):
        # cannot do anything: nothing is finite
        return arr
    if np.any(np.isfinite(arr).sum(axis=0) == arr.shape[0]):
        # there is at least a column full of Nan I cannot do anything either
        return np.NaN * arr
    # to fix the nan values, I assign the last known finite value per column
    res = 1.0 * arr
    for col_id in range(arr.shape[1]):
        finite_ind = np.where(np.isfinite(res[:,col_id]))[0]
        infinite_ind = np.where(~np.isfinite(res[:,col_id]))[0]
        idx = np.searchsorted(finite_ind, infinite_ind)
        idx[idx == finite_ind.shape[0]] = finite_ind.shape[0] - 1
        res[infinite_ind, col_id] = 1.0 * arr[finite_ind[idx], col_id]
    return res


def get_gen_ids_hydro(env):
    gen_id = 0
    ids_hyrdo = []
    for i in range(env.n_gen):
        if env.gen_redispatchable[i]:
            if env.gen_type[i] == "hydro":
                ids_hyrdo.append(gen_id)
            gen_id += 1
    ids_hyrdo = np.array(ids_hyrdo)
    return ids_hyrdo
    

def fix_forecast_ramps(nb_h,
                       load_p,
                       load_p_forecasted,
                       res_gen_p_df,
                       res_gen_p_forecasted_df,
                       env_for_loss,
                       hydro_constraints,
                       params):
    
    #### cvxpy
    total_step = 1 # for now
    total_gen = np.sum(env_for_loss.gen_redispatchable)
    load_f = load_p_forecasted.sum(axis=1)
    scaling_factor = env_for_loss.gen_pmax[env_for_loss.gen_redispatchable]
    scale_for_loads =  np.repeat(scaling_factor.reshape(1,-1), total_step, axis=0)
    ids_hyrdo = get_gen_ids_hydro(env_for_loss)
    
    p_min = np.repeat(env_for_loss.gen_pmin[env_for_loss.gen_redispatchable].reshape(1,-1)  / scaling_factor,
                      total_step + nb_h,
                      axis=0)
    p_max = np.repeat(env_for_loss.gen_pmax[env_for_loss.gen_redispatchable].reshape(1,-1) * params["PmaxErrorCorrRatio"] / scaling_factor,
                      total_step + nb_h,
                      axis=0)
    
    ramp_min = np.repeat(-env_for_loss.gen_max_ramp_down[env_for_loss.gen_redispatchable].reshape(1,-1) * params["RampErrorCorrRatio"] / scaling_factor,
                         total_step + (nb_h-1),
                         axis=0)
    ramp_max = np.repeat(env_for_loss.gen_max_ramp_up[env_for_loss.gen_redispatchable].reshape(1,-1) * params["RampErrorCorrRatio"] / scaling_factor,
                         total_step + (nb_h-1),
                         axis=0)
    
    if "hydro_ramp_reduction_factor" in params:
        ramp_max[:, ids_hyrdo] /= float(params["hydro_ramp_reduction_factor"])
        ramp_min[:, ids_hyrdo] /= float(params["hydro_ramp_reduction_factor"])
        
    res_gen_p = np.zeros((res_gen_p_forecasted_df.shape[0], total_gen))
    amount_curtailed_for = np.zeros((res_gen_p_forecasted_df.shape[0], ))
    has_error = np.zeros(res_gen_p_forecasted_df.shape[0], dtype=bool)
    t0_errors = []
    errors = []
    indx_forecasts_for_hydro_only = np.arange(0, nb_h + 1)
    for t0 in range(res_gen_p_df.shape[0] - 1):
        # forecast are "consistent from a power system point of view" batch by batch
        # losses are not handled here !
        loss_scale = res_gen_p_df.iloc[t0].sum() / load_p.iloc[t0].sum()
        indx_forecasts = np.arange(t0*nb_h, (t0+1)*nb_h)
        
        prod_renew_for = res_gen_p_forecasted_df.iloc[indx_forecasts, env_for_loss.gen_renewable].sum(axis=1)
        loss_for = (load_f.iloc[indx_forecasts] * loss_scale - prod_renew_for)
        net_load = np.array(([res_gen_p_df.iloc[t0, env_for_loss.gen_redispatchable].sum()] +  # value in the env
                             loss_for.tolist())
                           )
        # curtailment
        curt_t = cp.Variable(shape=(total_step + nb_h, ), nonneg=True)
        # curt_t = cp.multiply(curt_t_scaled, sum_pmax_renew)
        renew = np.array(([res_gen_p_df.iloc[t0, env_for_loss.gen_renewable].sum()] +  # value in the env
                          prod_renew_for.tolist())
                        )
        scale_curt_factor = np.maximum(renew, 1.)
        curt_t_scaled = cp.multiply(curt_t, 1. / scale_curt_factor)
        
        # generation
        p_t = cp.Variable(shape=(total_step + nb_h, total_gen), pos=True)
        real_p = cp.multiply(p_t, scale_for_loads)
        target_vector = 1.0 * np.concatenate((res_gen_p_df.iloc[t0, env_for_loss.gen_redispatchable].values.reshape(1, total_gen),
                                              res_gen_p_forecasted_df.iloc[indx_forecasts, env_for_loss.gen_redispatchable].values),
                                              axis=0).reshape(1+nb_h, total_gen)
        
        p_max_here = 1.0 * p_max
        if hydro_constraints is not None:
            p_max_here[:, ids_hyrdo] = fix_nan_hydro_i_dont_know_why(hydro_constraints["p_max_pu"].values[indx_forecasts_for_hydro_only, :])
            indx_forecasts_for_hydro_only += 1
            indx_forecasts_for_hydro_only[indx_forecasts_for_hydro_only > (res_gen_p_df.shape[0] - 1)] = res_gen_p_df.shape[0] - 1
        
        turned_off_orig = 1.0 * (target_vector == 0.)
        target_vector /= scale_for_loads
        constraints = [real_p[0,:] == res_gen_p_df.iloc[t0, env_for_loss.gen_redispatchable].values,
                       p_t >= p_min,
                       p_t <= p_max_here,
                       p_t[1:,:] - p_t[:-1,:] >= ramp_min,
                       p_t[1:,:] - p_t[:-1,:] <= ramp_max,
                       cp.sum(real_p, axis=1) >= (net_load.reshape(-1) + curt_t),
                       cp.sum(real_p, axis=1) <= 1.01 * (net_load.reshape(-1) + curt_t),
                       curt_t <= renew,
                       curt_t >= 0.,
                       curt_t[0] == 0.
                       ]
        
        cost = cp.sum_squares(p_t - target_vector) + cp.norm1(cp.multiply(p_t, turned_off_orig)) + 10. * cp.sum_squares(curt_t_scaled)  # TODO normalize last stuff
        prob = cp.Problem(cp.Minimize(cost), constraints)
        try:
            res_opt = prob.solve()
        except cp.error.SolverError as exc_:
            t0_errors.append(t0)
            errors.append(RuntimeError(f"cvxpy failed to find a solution for t0 {t0}, error {exc_}"))
            has_error[indx_forecasts] = True
            continue
        
        if not np.isfinite(res_opt):
            t0_errors.append(t0)
            errors.append(RuntimeError(f"cvxpy failed to find a solution for t0 {t0} and returned an infinite cost"))
            has_error[indx_forecasts] = True
            continue
        
        # assign the generators
        gen_p_after_optim = real_p.value
        if gen_p_after_optim is None:
            t0_errors.append(t0)
            errors.append(f"cvxpy failed to find a solution for t0 {t0}, and returned None")
            has_error[indx_forecasts] = True
            continue
        
        if not has_error[indx_forecasts[0]]:
            res_gen_p[indx_forecasts] = 1.0 * gen_p_after_optim[1:,:]
            amount_curtailed_for[indx_forecasts] = curt_t.value[1:]      
    # last value is not used anyway
    # res_gen_p[-1, :] = 1.0 * gen_p_after_optim[1,:]
    has_error[-nb_h:] = True
        
    res_gen_p_forecasted_df_res = 1.0 * res_gen_p_forecasted_df
    # assign controlable generators
    res_gen_p_forecasted_df_res.iloc[~has_error, env_for_loss.gen_redispatchable] = res_gen_p[~has_error,:]
    
    # fix for curtailment
    total_renew = res_gen_p_forecasted_df_res.iloc[:, env_for_loss.gen_renewable].sum(axis=1)
    total_renew[total_renew <= 1.] = 1.  # normalize for value close to 0.
    ratio = (total_renew - amount_curtailed_for) / total_renew
    res_gen_p_forecasted_df_res.iloc[:, env_for_loss.gen_renewable] *= np.stack([ratio for _ in 
                                                                                 range(np.sum(env_for_loss.gen_renewable))],
                                                                                 axis=1)
    
    # fix for tiny negative value that, with the magic of rounding
    # can lead to -0.1 in the data
    res_gen_p_forecasted_df_res[res_gen_p_forecasted_df_res < 0.] = 0.
    
    # make sure the forecasts are always above the demands, even the the opf failed
    return res_gen_p_forecasted_df_res, t0_errors, errors, amount_curtailed_for


def fix_negative(forecast):
    # inplace
    forecast[forecast < 0.] = 0.
    

def fix_pmax(forecast, pmax):
    # forecasts : (nb_gen, nb_t, nb_h)
    # pmax: (nb_gen)
    # not inplace
    pmax_ = np.stack([pmax for _ in range(forecast.shape[1])], axis=1)
    # pmax_: (nb_gen, nb_t)
    pmax_ = np.stack([pmax_ for _ in range(forecast.shape[-1])], axis=2)
    # pmax_: (nb_gen, nb_t, nb_h)
    
    init_shape = forecast.shape
    forecast_ = forecast.ravel()
    pmax_ = pmax_.ravel()
    mask_ = forecast_ > pmax_
    forecast_[mask_] = pmax_[mask_]
    forecast_ = forecast_.reshape(init_shape)
    return forecast_
    
    
def fix_solar(forecast, real_value, pmax, tol=1e-2):
    # inplace !
    
    # if value < 0. then it's 0.
    fix_negative(forecast)
    
    # if above pmax then pmax
    forecast = fix_pmax(forecast, pmax)
    
    # no solar at night
    real_value_ = np.stack([np.roll(real_value, -h) for h in range(forecast.shape[-1])], axis=2)
    forecast[real_value_ <= tol] = 0.


def fix_wind(forecast, real_value, pmax):
    # inplace !
    
    # if value < 0. then it's 0.
    fix_negative(forecast)
    
    # if above pmax then pmax
    forecast = fix_pmax(forecast, pmax)


def generate_new_gen_forecasts(prng,
                               forecasts_params,
                               load_params,
                               loads_charac,
                               gens_charac,
                               path_env,
                               res_gen_p_df):
    
    # read the parameters from the inputs
    nb_gen = len(gens_charac['name'])
    nb_h = len(forecasts_params["h"])
    datetime_index = pd.date_range(start=load_params['start_date'],
                                   end=load_params['end_date'],
                                   freq=str(load_params['dt']) + 'min')
    nb_t = datetime_index.shape[0]    
    
    # load the parameters controling the RES
    with open(os.path.join(path_env, "params_res.json"), "r") as f:
        res_params = json.load(f)
    
    res_gen_p_forecasted = None
    
    forecasts_params["T"] = load_params["T"]
    res_params["T"] = load_params["T"]
    
    # generate the independant data on the mesh
    for data_type, fun_fix in zip(["solar", "wind"],
                                  [fix_solar, fix_wind]):
        
        # get the parameters for this data type
        mask_this_type = gens_charac["type"].values == data_type
        gen_carac_this_type = gens_charac.loc[mask_this_type]
        nb_gen_this_type = gen_carac_this_type.shape[0]
        
        # generrate the noise for all the generators
        this_noise, hs, std_hs = generate_noise(loads_charac,
                                                gens_charac,
                                                forecasts_params,
                                                load_params,
                                                None,
                                                data_type,
                                                get_add_dim_renew,
                                                nb_t,
                                                res_params,
                                                gen_carac_this_type,
                                                nb_gen_this_type,
                                                add_h0=False,
                                                prng_noise=prng)
        # shape: (nb_elem, nb_t, nb_h)
        
        # generate all the forecasts
        gen_p_this_type = 1.0 * res_gen_p_df.loc[:, mask_this_type].values.T
        gen_p_for_this_type = get_forecast(gen_p_this_type,
                                           this_noise[:, :-1, :],
                                           hs,
                                           std_hs,
                                           gen_carac_this_type,
                                           reshape=False,
                                           keep_first_dim=True)
        
        # fix the value of the forecast (if above pmax or bellow pmin for example)
        fun_fix(gen_p_for_this_type,
                gen_p_this_type,
                gen_carac_this_type["Pmax"].values)
        
        # now put everything in the right shape
        if res_gen_p_forecasted is None:
            res_gen_p_forecasted = np.stack([res_gen_p_df.values.T for _ in hs], axis=2)
        res_gen_p_forecasted[mask_this_type, :, :] = gen_p_for_this_type
        
    res_gen_p_forecasted = res_gen_p_forecasted.reshape(nb_gen, (nb_t - 1) * nb_h)
    res_gen_p_forecasted = np.transpose(res_gen_p_forecasted, (1, 0))
    res_gen_p_forecasted_df = pd.DataFrame(res_gen_p_forecasted, columns=gens_charac["name"])
    return res_gen_p_forecasted_df, nb_h


def generate_forecasts_gen(new_forecasts,
                           prng,
                           load_p,
                           load_p_forecasted, 
                           res_gen_p_df,
                           sigma,
                           env_for_loss,
                           hydro_constraints,
                           forecasts_params,
                           load_params,
                           loads_charac,
                           gens_charac,
                           path_env,
                           opf_params):
    if new_forecasts:
        res_gen_p_forecasted_df, nb_h = generate_new_gen_forecasts(prng,
                                                                   forecasts_params,
                                                                   load_params,
                                                                   loads_charac,
                                                                   gens_charac,
                                                                   path_env,
                                                                   res_gen_p_df)
    else:
        res_gen_p_forecasted_df = res_gen_p_df * prng.lognormal(mean=0.0,
                                                                sigma=sigma,
                                                                size=res_gen_p_df.shape)
        res_gen_p_forecasted_df = res_gen_p_forecasted_df.shift(-1)
        res_gen_p_forecasted_df.iloc[-1] = 1.0 * res_gen_p_forecasted_df.iloc[-2]
        nb_h = 1
            
    # "fix" cases where forecasts are bellow the loads => in that case scale the
    # controlable generation to be at least 1% above total demand
    loss_before_ramps_for = 1.01
    total_gen = res_gen_p_forecasted_df.sum(axis=1)
    total_demand = load_p_forecasted.sum(axis=1)
    mask_not_enough_gen = total_gen <= loss_before_ramps_for * total_demand
    nb_concerned = (mask_not_enough_gen).sum()
    tmp = type(env_for_loss).gen_pmax[env_for_loss.gen_redispatchable]
    tmp = tmp / tmp.sum()
    rep_factor = np.tile(tmp.reshape(-1,1), nb_concerned).T  # how to split 1MW on the controlable generators
    est_losses_mw = (loss_before_ramps_for * total_demand - total_gen)  # opposite of the loss per step
    # we increase the controlable generation when there is not enough generation (loss positive)
    # res_gen_p_forecasted_df.loc[mask_not_enough_gen, type(env_for_loss).gen_redispatchable] *= est_losses_mw[mask_not_enough_gen].values.reshape(-1,1) * rep_factor
    res_gen_p_forecasted_df.loc[mask_not_enough_gen, type(env_for_loss).gen_redispatchable] += est_losses_mw[mask_not_enough_gen].values.reshape(-1,1) * rep_factor
    # the above increase can lead to generator above pmax, when this is the case, 
    # I cut it
    gen_pmax = type(env_for_loss).gen_pmax
    for gen_id, is_disp in enumerate(type(env_for_loss).gen_redispatchable):
        if not is_disp:
            continue
        res_gen_p_forecasted_df.iloc[:, gen_id] = np.minimum(res_gen_p_forecasted_df.iloc[:, gen_id].values,
                                                             gen_pmax[gen_id])
    # and fix the ramps (for all h) (an optimizer is run t by t)
    tmp_ = fix_forecast_ramps(nb_h,
                              load_p,
                              load_p_forecasted,
                              res_gen_p_df,
                              res_gen_p_forecasted_df,
                              env_for_loss,
                              hydro_constraints,
                              opf_params)
    res_gen_p_forecasted_df_res, t0_errors, errors, amount_curtailed_for = tmp_
    return res_gen_p_forecasted_df_res, amount_curtailed_for, t0_errors, errors


def apply_maintenance_wind_farm(extra_winds_params, prod_wind_init,
                                start_date_dt, end_date_dt, dt_min,
                                renew_prng):
    res = 1.0 * prod_wind_init
    # perfom some checks to be sure
    assert "proba_daily_outage_per_wind_mill" in extra_winds_params  # proba for the time
    assert "nb_wind_turbine" in extra_winds_params
    assert "p_geom_failure" in extra_winds_params # proba for the number
    assert len(extra_winds_params["nb_wind_turbine"]) == prod_wind_init.shape[1]
    for el in res.columns:
        assert el in extra_winds_params["nb_wind_turbine"], f"error {el} is not in extra_winds_params.json `nb_wind_turbine`"
    nb_windturb_per_mill = [extra_winds_params["nb_wind_turbine"][el] for el in res.columns]
    proba_wind_turb_fails = float(extra_winds_params["p_geom_failure"])
    
    # now simulate the "failure" process
    proba_failure_per_time = float(extra_winds_params["proba_daily_outage_per_wind_mill"]) / (60. * 24. / float(dt_min))
    do_fail = renew_prng.uniform(size=res.shape)
    do_fail = do_fail <= proba_failure_per_time  # whether or not an external event caused a failure
    nb_failure = do_fail.sum(axis=0)  # number of failure in the scenario
    mult_factor = np.ones(res.shape)
    for wind_turb_id in range(res.shape[1]):
        nb_turb_in_service = np.full(shape=res.shape[0], fill_value=nb_windturb_per_mill[wind_turb_id])
        time_start_failure = np.where(do_fail[:,wind_turb_id])[0]
        
        # simulate intensity and duration of failures
        fail_id = 0
        while fail_id < nb_failure[wind_turb_id]:
            ts_start = time_start_failure[fail_id]
            # intensity of the external event : number of wind_mill affected
            do_turbine_fail = renew_prng.uniform(size=nb_turb_in_service[ts_start]) <=  proba_wind_turb_fails
            # duration of the outage
            duration = int(60 * 24 / float(dt_min))  # fixed at 1 day for now  # TODO !
            # and now update everything
            duration_ids = np.arange(duration) + ts_start
            duration_ids = duration_ids[duration_ids < res.shape[0]]
            nb_turb_in_service[duration_ids] -= do_turbine_fail.sum()
            fail_id += 1
        mult_factor[:, wind_turb_id] = nb_turb_in_service / nb_windturb_per_mill[wind_turb_id]
    return res * mult_factor
