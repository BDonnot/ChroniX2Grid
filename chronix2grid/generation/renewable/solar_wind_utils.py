# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)

import copy

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from .. import generation_utils as utils
import chronix2grid.constants as cst

def compute_wind_series(prng, locations, Pmax, long_noise, medium_noise,
                        short_noise, params, smoothdist, add_dim,
                        return_ref_curve=False,
                        tol=0.):
    # NB tol is set to 0.0 for legacy behaviour, otherwise tests do not pass, but this is a TERRIBLE idea.
    
    # Compute refined signals
    long_scale_signal = utils.interpolate_noise(
        long_noise,
        params,
        locations,
        time_scale=params['long_wind_corr'],
        add_dim=add_dim)
    medium_scale_signal = utils.interpolate_noise(
        medium_noise,
        params,
        locations,
        time_scale=params['medium_wind_corr'],
        add_dim=add_dim)
    short_scale_signal = utils.interpolate_noise(
        short_noise,
        params,
        locations,
        time_scale=params['short_wind_corr'],
        add_dim=add_dim)

    # Compute seasonal pattern
    Nt_inter = int(params['T'] // params['dt'] + 1)
    t = np.linspace(0, params['T'], Nt_inter, endpoint=True)
    start_min = int(
        pd.Timedelta(params['start_date'] - pd.to_datetime('2018-01-01', format='%Y-%m-%d')).total_seconds() // 60)
    seasonal_pattern = np.cos((2 * np.pi / (365 * 24 * 60)) * (t - 30 * 24 * 60 - start_min))

    # Combine signals
    std_short_wind_noise = float(params['std_short_wind_noise'])
    std_medium_wind_noise = float(params['std_medium_wind_noise'])
    std_long_wind_noise = float(params['std_long_wind_noise'])
    signal = (0.7 + 0.3 * seasonal_pattern) * (0.3 + std_medium_wind_noise * medium_scale_signal + std_long_wind_noise * long_scale_signal)
    signal += std_short_wind_noise * short_scale_signal
    signal = 1e-1 * np.exp(4 * signal)
    #signal += prng.uniform(0, SMOOTHDIST/Pmax, signal.shape)
    signal += prng.uniform(0, smoothdist, signal.shape)
    #signal += np.random.uniform(0, smoothdist, signal.shape) #older version - to be removed

    # signal *= 0.95
    signal[signal < tol] = 0.
    signal = smooth(signal)
    wind_series = Pmax * signal
    wind_series[wind_series > 0.95 * Pmax] = 0.95 * Pmax
    if not return_ref_curve:
        return wind_series
    return wind_series, 1e-1 * np.exp(4 * (0.7 + 0.3 * seasonal_pattern) * 0.3)


def compute_solar_series(prng, locations, Pmax, solar_noise, params,
                         solar_pattern, smoothdist, time_scale,
                         add_dim, scale_solar_coord_for_correlation=None,
                         return_ref_curve=False,
                         tol=0.):
    # NB tol is set to 0.0 for legacy behaviour, otherwise tests do not pass, but this is a TERRIBLE idea.

    # Compute noise at desired locations
    if scale_solar_coord_for_correlation is not None:
        locations = [float(scale_solar_coord_for_correlation) * float(locations[0]), float(scale_solar_coord_for_correlation) * float(locations[1])]
    final_noise = utils.interpolate_noise(solar_noise, params, locations, time_scale, add_dim=add_dim)

    # Compute solar pattern
    solar_pattern = compute_solar_pattern(params, solar_pattern, tol=tol)

    # Compute solar time series
    std_solar_noise = float(params['std_solar_noise'])
    if "mean_solar_pattern" in params:
        mean_solar_pattern = float(params["mean_solar_pattern"])
    else:
        # legacy behaviour
        mean_solar_pattern = 0.75
        
    signal = solar_pattern * (mean_solar_pattern + std_solar_noise * final_noise)
    #signal += prng.uniform(0, smoothdist/Pmax, signal.shape) #to be revised: since smmothdist/PMax is very small, the added noise compared to the previous sinal was unsignificant
    #signal += np.random.uniform(0, smoothdist / Pmax, signal.shape) #older version - to be removed
    # signal[signal > 1] = 1
    signal[signal < tol] = 0.
    signal = smooth(signal)
    solar_series = Pmax * signal
    # solar_series[np.isclose(solar_series, 0.)] = 0
    solar_series[solar_series > 0.95 * Pmax] = 0.95 * Pmax
    if not return_ref_curve:
        return solar_series
    else:
        return solar_series, solar_pattern * mean_solar_pattern

def compute_solar_pattern(params, solar_pattern, tol=0.0):
    """
    Loads a typical hourly pattern, and interpolates it to generate
    a smooth solar generation pattern between 0 and 1

    # NB tol is set to 0.0 for legacy behaviour, otherwise tests do not pass, but this is a TERRIBLE idea.
    
    Input:
        computation_params: (dict) Defines the mesh dimensions and
            precision. Also define the correlation scales
        interpolation_params: (dict) params of the interpolation

    Output:
        (np.array) A smooth solar pattern
    """

    start_year = pd.to_datetime(str(params['start_date'].year) + '-01-01', format='%Y-%m-%d')
    end_min = int(pd.Timedelta(params['end_date'] - start_year).total_seconds() // 60)

    Nt_inter_hr = int(end_min // 60 + 1)
    N_repet = int((Nt_inter_hr - 1) // len(solar_pattern) + 1)
    stacked_solar_pattern = solar_pattern
    for i in range(N_repet - 1):
        stacked_solar_pattern = np.append(stacked_solar_pattern, solar_pattern)
    # solar_pattern : over the all year, hourly based
    
    # The time is in minutes
    t_pattern = 60 * np.linspace(0, 8760 * N_repet, 8760 * N_repet, endpoint=False)
    f2 = interp1d(t_pattern, stacked_solar_pattern, kind='cubic')
    # f2 = interp1d(t_pattern, stacked_solar_pattern, kind='linear')

    Nt_inter = int(params['T'] // params['dt'] + 1)
    start_year = pd.to_datetime(str(params['start_date'].year) + '-01-01', format='%Y-%m-%d')
    start_min = int(pd.Timedelta(params['start_date'] - start_year).total_seconds() // 60)
    end_min = int(pd.Timedelta(params['end_date'] - start_year).total_seconds() // 60)

    t_inter = np.linspace(start_min, end_min, Nt_inter, endpoint=True)
    output = f2(t_inter)
    output = output * (output > 0)
    output[output < tol] = 0.
    if "solar_night_hour" in params:
        # addition: force the solar at 0 at night
        dts = [params['start_date'] + i * pd.Timedelta(minutes=params['dt']) for i in range(t_inter.shape[0])]
        dts_hours = np.array([el.hour for el in dts])
        # i will multiply by 1 almost all hours
        to_mult = np.ones(len(dts_hours))
        # except hours provided in solar_night_hour
        to_mult[np.isin(dts_hours, params["solar_night_hour"])] = 0.
        
        # now I want to "smooth" the "to_mult" for the "border"
        
        # left smoothing, for 1 day
        left_ = 1. / (1. + np.exp(1 - ((1+ np.arange(60 // params['dt']) )- (30. /  params['dt']))))
        # right smoothing (for 1 day too)
        right_ = 1. / (1. + np.exp(1 + ((1+ np.arange(60 // params['dt']) )- (30. /  params['dt']))))
        
        # define some constants
        nb_day = int((params['end_date'] - params['start_date']).total_seconds() // 86400)
        solar_night_hour = np.array(params["solar_night_hour"])
        
        # handles the "left" border (eg 5 am)
        first_non_zero_h = np.max(solar_night_hour[solar_night_hour <= 12]) + 1 # 12 for midday
        all_left = np.tile(left_, nb_day)
        tmp_ = np.zeros((dts_hours == first_non_zero_h).sum())
        tmp_[:all_left.shape[0]] = all_left
        to_mult[dts_hours == first_non_zero_h] = tmp_
        
        # handles "right" border (eg 22 / 10 pm)
        last_non_zero_h = np.min(solar_night_hour[solar_night_hour >= 12]) - 1
        all_right = np.tile(right_, nb_day)
        tmp_ = np.zeros((dts_hours == last_non_zero_h).sum())
        to_rem = tmp_.shape[0] - all_right.shape[0]
        tmp_[to_rem:] = all_right
        to_mult[dts_hours == last_non_zero_h] = tmp_
        
        # force the output to be 0 at night
        output *= to_mult
        
    if "force_solar_zero" in params:
        # another heuristic to cap the solar at 0.0 at "night"
        # when the solar is at 5% of its max value, then 1h after it it should be 0.
        threshold_zero = output.max() * 0.05
        max_hour_seen_non_zero = np.max(dts_hours[output >= threshold_zero])
        max_hour_seen_non_zero += int(params["force_solar_zero"])
        output[dts_hours >= max_hour_seen_non_zero] = 0.
        # same in the other direction: 1h before it's always bellow threshold, it should be 0.
        min_hour_seen_non_zero = np.min(dts_hours[output >= threshold_zero])
        min_hour_seen_non_zero -= int(params["force_solar_zero"])
        output[dts_hours <= min_hour_seen_non_zero] = 0.
    return output

def smooth(x, alpha=0.5, beta=None):
    """
    smoothing function to avoid weird statistical effect
    we want below alpha: f(x) = x
    f(alpha) = alpha, and f'(alpha) = 1
    f(beta) = 1 and f'(beta) = 0
    f cubic polynomial

    if beta is not provided, beta = 1/alpha

    :param alpha: value (x) where smoothing starts
    :param beta: y when x=1
    """
    x = copy.deepcopy(x)
    if beta is None:
        beta = 1 / alpha

    # def pol(x, alpha, beta):
    #     a = (1 - 2 * (alpha - 1) / (alpha - beta)) / (alpha - beta) ** 2
    #     b = (alpha - 1) / (alpha - beta) ** 2 - a * alpha
    #     return (x - beta) ** 2 * (a * x + b) + 1
    #
    # def pol(x, alpha, beta):
    #     a = ( 1 - 2*(alpha-1)/(alpha-beta)) / (alpha - beta)**2
    #     b = (alpha-1)/(alpha - beta)**2 - a*alpha
    #     return (x-beta)**2*(a*x+b)+1
    # alpha = 0.
    def pol(x, alpha, beta):
        return 1-np.exp(-x)
    # x[x > beta] = beta
    x = pol(x, alpha=alpha, beta=beta)
    return x


def create_csv(prng, dict_, path, reordering=True, noise=None, shift=False,
               write_results=True, index=False):
    if type(dict_) is dict:
        df = pd.DataFrame.from_dict(dict_)
    else:
        df = dict_.copy()
    df.set_index('datetime', inplace=True)
    df = df.sort_index(ascending=True)
    df = df.head(len(df ) -1)
    if reordering:
        value = []
        grid2op_default_name = True
        try:
            for name in list(df):
                value.append(utils.natural_keys(name))
        except IndexError:
            grid2op_default_name = False
            # cannot reorder in this case
        if grid2op_default_name:
            new_ordering = [x for _ ,x in sorted(zip(value ,list(df)))]
            df = df[new_ordering]
    if noise is not None:
        df *= ( 1 +noise * prng.normal(0, 1, df.shape))
        #df *= (1 + noise * np.random.normal(0, 1, df.shape)) #older version - to be removed
    if shift:
        df = df.shift(-1)
        df = df.fillna(0)
    if write_results:
        df.to_csv(path, index=index, sep=';',
                  float_format=cst.FLOATING_POINT_PRECISION_FORMAT)

    return df

