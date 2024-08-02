# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)

import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from .. import generation_utils as utils
import chronix2grid.constants as cst

__CACHED__DATA = {}  # DO NOT MODIFY THIS UNDER ANY CIRCUMSTANCES !
# TODO optim: cache the data instead of re reading everything every time


def compute_loads(loads_charac,
                  temperature_noise,
                  params,
                  load_weekly_pattern,
                  start_day,
                  add_dim,
                  day_lag=0,
                  return_ref_curve=False,
                  use_legacy=True):
    #6  # this is only TRUE if you simulate 2050 !!! formula does not really work
    
    # Compute active part of loads
    if isinstance(load_weekly_pattern, pd.DataFrame):
        weekly_pattern_vals = load_weekly_pattern['test'].values
        weekly_pattern_datetimes = load_weekly_pattern["datetime"]
    elif isinstance(load_weekly_pattern, os.PathLike):
        weekly_pattern_vals = load_weekly_pattern
        if not (load_weekly_pattern / "datetimes.csv").exists():
            raise RuntimeError(f"Impossible to load the associated dates and times in {load_weekly_pattern}. "
                               f"Make sure that the files `datetimes.csv` exists there (and is a pandas "
                               f"dataframe with at least a column name 'datetime')")
        weekly_pattern_datetimes = pd.read_csv(load_weekly_pattern / "datetimes.csv", sep=";")["datetime"]
    else:
        raise RuntimeError("at this stage `load_weekly_pattern` args should be either a path or a dataframe")
    
    if use_legacy:
        isoweekday = None
        hour_minutes = None
    else:
        datetime_lwp = pd.to_datetime(weekly_pattern_datetimes, format="%Y-%m-%d %H:%M:%S")
        isoweekday = np.array([el.isoweekday() for el in datetime_lwp])
        hour_minutes = np.array([el.hour * 60 + el.minute for el in datetime_lwp])
    
    # start_day_of_week = start_day.weekday()
    # first_dow_chronics = datetime.strptime(load_weekly_pattern["datetime"].iloc[1], "%Y-%m-%d %H:%M:%S").weekday()
    # + (calendar.isleap(start_day.year) if start_day.month >= 3 else 0)
    # day_lag = (first_dow_chronics - start_day_of_week) % 7
    # day_lag = 0
    loads_series = {}
    ref_curves = None
    for i, load_name in enumerate(loads_charac['name']):
        mask = (loads_charac['name'] == load_name)
        if loads_charac[mask]['type'].values == 'residential':
            locations = [loads_charac[mask]['x'].values[0], loads_charac[mask]['y'].values[0]]
            Pmax = loads_charac[mask]['Pmax'].values[0]
            tmp_ = compute_residential(locations,
                                       Pmax,
                                       temperature_noise,
                                       params,
                                       weekly_pattern_vals,
                                       index=i,
                                       day_lag=day_lag,
                                       add_dim=add_dim,
                                       return_ref_curve=return_ref_curve,
                                       isoweekday_lwp=isoweekday,
                                       hour_minutes_lwp=hour_minutes,
                                       load_name=load_name,
                                       weekly_pattern_datetimes=weekly_pattern_datetimes)
            if return_ref_curve:
                if ref_curves is None:
                    ref_curves = np.zeros((tmp_[0].shape[0], loads_charac.shape[0]))
                loads_series[load_name], ref_curves[:,i] = tmp_
            else:
                loads_series[load_name] = tmp_
        else:
            raise NotImplementedError(f"Load generation is not implemented for type {loads_charac[mask]['type'].values}")

        if loads_charac[mask]['type'].values == 'industrial':
            raise NotImplementedError("Impossible to generate industrial loads for now.")
            Pmax = loads_charac[mask]['Pmax'].values[0]
            loads_series[name] = compute_industrial(Pmax, params)
    if not return_ref_curve:
        return loads_series
    return loads_series, ref_curves


def get_seasonal_pattern(params):
    Nt_inter = int(params['T'] // params['dt'] + 1)
    t = np.linspace(0., (params['end_date'] - params["start_date"]).total_seconds(), Nt_inter, endpoint=True, dtype=float)
    start_year = pd.to_datetime(str(params['start_date'].year) + '-01-01', format='%Y-%m-%d')
    start_min = float(pd.Timedelta(params['start_date'] - start_year).total_seconds())
    nb_sec_per_day =  24. * 60. * 60.
    nb_sec_per_year = (365. * nb_sec_per_day)
    year_pattern = 2. * np.pi / nb_sec_per_year
    seasonal_pattern = 1.5 / 7. * np.cos(year_pattern * (t + start_min - 45 * nb_sec_per_day))  # min of the load is 15 of February so 45 days after beginning of year
    seasonal_pattern += 5.5 / 7.
    return seasonal_pattern


def compute_residential(locations,
                        Pmax,
                        temperature_noise,
                        params,
                        weekly_pattern,
                        index,
                        day_lag=None,
                        add_dim=0,
                        return_ref_curve=False,
                        isoweekday_lwp=None,
                        hour_minutes_lwp=None,
                        load_name=None,
                        weekly_pattern_datetimes=None):


    # Compute refined signals
    temperature_signal = utils.interpolate_noise(
        temperature_noise,
        params,
        locations,
        time_scale=params['temperature_corr'],
        add_dim=add_dim)
    temperature_signal = temperature_signal.astype(float)
    
    # Compute seasonal pattern
    seasonal_pattern = get_seasonal_pattern(params)

    # Get weekly pattern
    weekly_pattern = compute_load_pattern(params,
                                          weekly_pattern,
                                          index, day_lag,
                                          isoweekday_lwp=isoweekday_lwp,
                                          hour_minutes_lwp=hour_minutes_lwp,
                                          load_name=load_name,
                                          weekly_pattern_datetimes=weekly_pattern_datetimes)
    
    std_temperature_noise = params['std_temperature_noise']
    residential_series = Pmax * weekly_pattern * (std_temperature_noise * temperature_signal + seasonal_pattern)
    if return_ref_curve:
        return residential_series, Pmax * weekly_pattern * seasonal_pattern
    return residential_series

def compute_load_pattern(params,
                         weekly_pattern,
                         index,
                         day_lag=None,
                         isoweekday_lwp=None,
                         hour_minutes_lwp=None,
                         load_name=None,
                         weekly_pattern_datetimes=None,
                         use_cache=True,  # TODO propagate to function calling this one
                         fill_na=True,  # TODO propagate to function calling this one
                         ):
    """
    Loads a typical hourly pattern, and interpolates it to generate
    a smooth solar generation pattern between 0 and 1

    Input:
        computation_params: (dict) Defines the mesh dimensions and
            precision. Also define the correlation scales
            interpolation_params: (dict) params of the interpolation

    Output:
        (np.array) A smooth solar pattern
    """
    index_weekly_perweek = 12 * 24 * 7
    generate_multiple_pattern = None
    # solar_pattern resolution : 1H, 8761
    if isinstance(weekly_pattern, os.PathLike):
        # user provided a pattern in the shape of a folder
        if load_name is None:
            raise RuntimeError("When you want to use a different load pattern for each load, "
                               "you should provide the load name, ie the args `load_name`.")
        if weekly_pattern_datetimes is None:
            raise RuntimeError("When you want to use a different load pattern for each load, "
                               "you should provide the dates and times for this reference data "
                               "ie the args `weekly_pattern_datetimes`.")
        
        pattern_path = weekly_pattern / f"{load_name}.csv"
        if not pattern_path.exists():
            pattern_path = weekly_pattern / f"{load_name}.csv.gz"
            if not pattern_path.exists():
                raise RuntimeError(f"Unable to locate the weekly pattern for load {load_name} at {weekly_pattern}")
        if not use_cache:
            tmp_weekly_pattern = pd.read_csv(pattern_path, sep=";")
            if tmp_weekly_pattern.isna().any().any():
                print(f"WARNING: NA detected for load pattern {pattern_path}")
        else:
            if pattern_path in __CACHED__DATA:
                tmp_weekly_pattern = __CACHED__DATA[pattern_path]
                print(f"Using cached data for {pattern_path}")
            else:
                tmp_weekly_pattern = pd.read_csv(pattern_path, sep=";")
                if tmp_weekly_pattern.isna().any().any():
                    print(f"WARNING: NA detected for load pattern {pattern_path}")
                print(f"Loading cached data for {pattern_path}")
                __CACHED__DATA[pattern_path] = tmp_weekly_pattern
        used_weekly_pattern = tmp_weekly_pattern["values"].values
        if isoweekday_lwp is None or hour_minutes_lwp is None:
            raise RuntimeError(f"When provided a different pattern for each loads, you should also provide "
                               f"the isoweekday to simulate and also the hours / minute")
        generate_multiple_pattern = False
    elif isinstance(weekly_pattern, np.ndarray):
        # standard legacy behaviour
        used_weekly_pattern = 1. * weekly_pattern
        generate_multiple_pattern = True
    else:
        raise RuntimeError(f"weekly_pattern should be either a path or a numpy array, you provided {type(weekly_pattern)}")

    # Keep only one week of pattern
    if isoweekday_lwp is None or hour_minutes_lwp is None:
        # try to guess where to start from input data
        if day_lag is None:
            nb_step_lag_for_starting_day = 0
        else:
            nb_step_lag_for_starting_day = 12 * 24 * day_lag
        index %= int((used_weekly_pattern.shape[0] - nb_step_lag_for_starting_day) / index_weekly_perweek - 1)
        first_index = (nb_step_lag_for_starting_day + index * index_weekly_perweek)
    else:
        # be smarter and take a week starting the same weekday at the same hour than the params["start_date"]
        iso_hm_start = params["start_date"].hour * 60 + params["start_date"].minute
        isoweekday_start = params["start_date"].isoweekday()
        if generate_multiple_pattern:
            # in this case only one pattern is given for all the loads, so I come up with a solution to generate 
            # different patterns from this one
            possible_first_index = np.where((isoweekday_lwp == isoweekday_start) & (iso_hm_start == hour_minutes_lwp))[0]
            index_modulo = index % possible_first_index.shape[0]
            first_index = possible_first_index[index_modulo]
        else:
            # one pattern per load is already provided so I only here to extract the correct "week"
            week_of_year_params = params["start_date"].isocalendar()[1]  # the week number of the generated data
            week_of_year_data = np.array([el.week for el in pd.to_datetime(weekly_pattern_datetimes, format="%Y-%m-%d %H:%M:%S")])
            possible_first_index = np.where((week_of_year_data == week_of_year_params) & 
                                            (isoweekday_lwp == isoweekday_start) & 
                                            (iso_hm_start == hour_minutes_lwp))[0]
            first_index = possible_first_index[0]
    
    # now extract right week of data
    last_index = first_index + index_weekly_perweek
    used_weekly_pattern = 1.0 * used_weekly_pattern[first_index:last_index]
    not_finite = ~np.isfinite(used_weekly_pattern)
    if not_finite.any():
        print(f"WARNING: NA detected for load pattern for load index {index} (name {load_name})")
        if fill_na:
            if not not_finite.all():
                used_weekly_pattern[not_finite] = np.nanmean(used_weekly_pattern)
            else:
                print(f"WARNING: pattern full of nan for for load index {index} (name {load_name}), replaced by 1.")
                used_weekly_pattern[:] = 1.
    avg_ = np.mean(used_weekly_pattern)
    if np.abs(avg_) > 1e-5:
        used_weekly_pattern /= avg_
    else:
        print(f"WARNING: for load index {index} (name {load_name}): average value too small, pattern replaced by all one")
        used_weekly_pattern[:] = 1.

    # now generate an ouput of the right length
    if isoweekday_lwp is None or hour_minutes_lwp is None:
        # legacy usage... does not work at all I don't know why
        start_year = pd.to_datetime(str(params['start_date'].year) + '-01-01', format='%Y-%m-%d')
        T_bis = int(pd.Timedelta(params['end_date'] - start_year).total_seconds() // (60))

        Nt_inter_hr = int(T_bis // 5 + 1)
        N_repet = int((Nt_inter_hr - 1) // len(used_weekly_pattern) + 1)
        stacked_weekly_pattern = np.tile(used_weekly_pattern, N_repet)
        
        # The time is in minutes
        t_pattern = np.linspace(0, 60 * 7 * 24 * N_repet, 12 * 7 * 24 * N_repet, endpoint=False)
        f2 = interp1d(t_pattern, stacked_weekly_pattern, kind='cubic')

        Nt_inter = int(params['T'] // params['dt'] + 1)
        start_year = pd.to_datetime(str(params['start_date'].year) + '-01-01', format='%Y-%m-%d')
        start_min = int(pd.Timedelta(params['start_date'] - start_year).total_seconds() // 60)
        end_min = int(pd.Timedelta(params['end_date'] - start_year).total_seconds() // 60)
        t_inter = np.linspace(start_min, end_min, Nt_inter, endpoint=True)
        output = f2(t_inter)
        output = output * (output > 0)
    else:
        # new usage
        nb_ts = int((params['end_date'] - params['start_date']).total_seconds() / 60 / params["dt"] + 1)  # +1 is because of the buggy stuff above...
        N_repet = np.ceil(nb_ts / used_weekly_pattern.shape[0]).astype(int)
        stacked_weekly_pattern = np.tile(used_weekly_pattern, N_repet)
        output = stacked_weekly_pattern[:nb_ts]
    return output


def create_csv(prng, dict_, path, forecasted=False, reordering=True, noise=None,
               shift=False, write_results=True, index=False):
    df = pd.DataFrame.from_dict(dict_)
    df.set_index('datetime', inplace=True)
    df = df.sort_index(ascending=True)
    df = df.head(len(df)-1)  # Last value is lonely for another day
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
    if shift:
        df = df.shift(-1)
        df = df.fillna(0)

    df_reactive_power = 0.7 * df
    if noise is not None:
        df *= prng.lognormal(mean=0.0,sigma=noise, size=df.shape)
        #df *= np.random.lognormal(mean=0.0, sigma=noise, size=df.shape) #older version to be removed
        df_reactive_power *= prng.lognormal(mean=0.0, sigma=noise, size=df.shape)
        #df_reactive_power *= np.random.lognormal(mean=0.0, sigma=noise,
        #                                         size=df.shape) #older version to be removed

    if write_results:
        file_extension = '_forecasted' if forecasted else ''
        df.to_csv(
            os.path.join(path, f'load_p{file_extension}.csv.bz2'),
            index=index, sep=';', float_format=cst.FLOATING_POINT_PRECISION_FORMAT)
        df_reactive_power.to_csv(
            os.path.join(path, f'load_q{file_extension}.csv.bz2'),
            index=False, sep=';', float_format=cst.FLOATING_POINT_PRECISION_FORMAT)

    return df

