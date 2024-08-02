# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)

"""Class for the economic dispatch framework. Allows to parametrize and run
an economic dispatch based on RES and consumption time series"""

import numpy as np
import pandas as pd
from collections import namedtuple
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # TODO logger !
    # some warnings on some pypsa / numpy versions
    import pypsa

from ._EDispatch_L2RPN2020.run_economic_dispatch import main_run_disptach

## Dépendances à Chronix2Grid
from chronix2grid.generation.dispatch.EconomicDispatch import Dispatcher
from chronix2grid.generation.dispatch.utils import RampMode

DispatchResults = namedtuple('DispatchResults', ['chronix', 'terminal_conditions'])


class PypsaDispatcher(Dispatcher, pypsa.Network):
    """
    Inheriting from Dispatcher to implement abstract methods thanks to Pypsa API
    Wrapper around a pypsa.Network to add higher level methods
    """

    # PATCH
    # to avoid problems for respecting pmax and ramps when rounding production values in chronics at the end, we apply a correcting factor
    PmaxCorrectingFactor = 1.
    RampCorrectingFactor = 0.1
    DebugPmax = 1e9  # in debug mode, pmax for the "infinite" generator
    DebugCost = 1e7  # in debug mode, cost for the "infinite" generator
    
    def __init__(self,
                 *args,
                 pypsa_debug_add_inf_gen=False,
                 pypsa_debug_add_thermal=True,
                 pypsa_debug_add_hydro=True,
                 pypsa_debug_add_nuc=True,
                 pypsa_debug_add_solar=True,
                 pypsa_debug_add_wind=True,
                 pypsa_debug_add_slack=True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.add('Bus', 'only_bus')
        self.add('Load', name='agg_load', bus='only_bus')
        self._env = None  # The grid2op environment when instanciated with from_gri2dop_env
        self._df = None
        
        self._all_gen_names = set()
        self._chronix_scenario = None
        self._simplified_chronix_scenario = None
        self._has_results = False
        self._has_simplified_results = False
        
        self._pmax_solar = None
        self._pmax_wind = None
        self._pypsa_debug_add_inf_gen = bool(pypsa_debug_add_inf_gen)
        if self._pypsa_debug_add_inf_gen:
            self.add(class_name='Generator',
                     name="debug_infinite_gen",
                     bus='only_bus',
                     marginal_cost=type(self).DebugCost,
                     p_nom=type(self).DebugPmax,
                     ramp_limit_up=1.,
                     ramp_limit_down=1.,
                     )
        self._pypsa_debug_add_thermal = bool(pypsa_debug_add_thermal)
        self._pypsa_debug_add_hydro = bool(pypsa_debug_add_hydro)
        self._pypsa_debug_add_nuc = bool(pypsa_debug_add_nuc)
        self._pypsa_debug_add_solar = bool(pypsa_debug_add_solar)
        self._pypsa_debug_add_wind = bool(pypsa_debug_add_wind)
        self._pypsa_debug_add_slack = bool(pypsa_debug_add_slack)
        self._pypsa_debug_flags = {"wind": self._pypsa_debug_add_wind,
                                   "solar": self._pypsa_debug_add_solar,
                                   "hydro": self._pypsa_debug_add_hydro,
                                   "thermal": self._pypsa_debug_add_thermal,
                                   "nuclear":  self._pypsa_debug_add_nuc,
                                   "slack": self._pypsa_debug_add_slack,
                                   "infinite_gen": self._pypsa_debug_add_inf_gen,
                                   "slack_maybe_ignored": ((not self._pypsa_debug_add_nuc) or 
                                                           (not self._pypsa_debug_add_thermal) or 
                                                           (not self._pypsa_debug_add_hydro) or 
                                                           (not self._pypsa_debug_add_slack))}

    @classmethod
    def _aux_add_solar(cls, net):
        if net._pypsa_debug_add_solar and net._pmax_solar > 0.:
            net.add('Generator',
                    name='agg_solar',
                    bus='only_bus',
                    carrier="solar",
                    marginal_cost=0.,
                    p_nom=net._pmax_solar
                    )
        elif not net._pypsa_debug_add_solar and net._pmax_solar > 0.:
            print(f"Solar generators are ignored in the OPF due to a pysa_debug flag")
            
    @classmethod
    def _aux_add_wind(cls, net):
        if net._pypsa_debug_add_wind and net._pmax_wind > 0.:
            net.add('Generator',
                    name='agg_wind',
                    bus='only_bus',
                    carrier="wind",
                    marginal_cost=0.1,  # we prefer to curtail the wind if we have the choice
                    # that's because solar should be distributed on the grid
                    p_nom=net._pmax_wind
            )
        elif not net._pypsa_debug_add_wind and net._pmax_wind > 0.:
            print(f"Solar generators are ignored in the OPF due to a pysa_debug flag")
    
    @classmethod
    def _aux_add_controlable_gen(cls, net, gen_type, carrier_types_to_exclude, p_max, name, ramp_up, ramp_down, gen_cost_per_MW): 
        net._all_gen_names.add(name)
        if gen_type not in carrier_types_to_exclude:
            pnom = p_max - cls.PmaxCorrectingFactor
            if pnom <= 0.:
                print(f"\t Issue for {name}: a negative pmax would be applied. We put {cls.PmaxCorrectingFactor} instead")
                pnom = max(pnom, cls.PmaxCorrectingFactor)
                
            rampUp = (ramp_up - cls.RampCorrectingFactor) / p_max
            rampDown = (ramp_down - cls.RampCorrectingFactor) / p_max
            if rampUp <= 0.:
                print(f"\t Issue for {name}: a negative ramp_up would be applied. We put {cls.RampCorrectingFactor / p_max} instead")
                rampUp = cls.RampCorrectingFactor / p_max
            if rampDown <= 0.:
                print(f"\t Issue for {name}: a negative ramp_down would be applied. We put {cls.RampCorrectingFactor / p_max} instead")
                rampDown = cls.RampCorrectingFactor / p_max
            if net._pypsa_debug_flags[gen_type]:
                net.add(
                    class_name='Generator',
                    name=name,
                    bus='only_bus',
                    p_nom=pnom,
                    carrier=gen_type,
                    marginal_cost=gen_cost_per_MW,
                    ramp_limit_up=rampUp,
                    ramp_limit_down=rampDown,
                )
            else:
                print(f"\t generator {name} (type {gen_type}) is ignored because of a pypsa_debug flag is set for its type")
                    
    @classmethod
    def from_gri2op_env(cls,
                        grid2op_env,
                        *,
                        pypsa_debug_add_inf_gen=False,
                        pypsa_debug_add_thermal=True,
                        pypsa_debug_add_hydro=True,
                        pypsa_debug_add_nuc=True,
                        pypsa_debug_add_solar=True,
                        pypsa_debug_add_wind=True,
                        pypsa_debug_add_slack=True,):
        """
        Implements the abstract method of *Dispatcher*

        Parameters
        ----------
        grid2op_env

        Returns
        -------
        net: :class:`pypsa.Network`
        """
        net = cls(pypsa_debug_add_inf_gen=pypsa_debug_add_inf_gen,
                  pypsa_debug_add_thermal=pypsa_debug_add_thermal,
                  pypsa_debug_add_hydro=pypsa_debug_add_hydro,
                  pypsa_debug_add_nuc=pypsa_debug_add_nuc,
                  pypsa_debug_add_solar=pypsa_debug_add_solar,
                  pypsa_debug_add_wind=pypsa_debug_add_wind,
                  pypsa_debug_add_slack=pypsa_debug_add_slack,
                  )
        net._env = grid2op_env

        # add total wind and solar (for curtailment)
        net._pmax_solar = np.sum([env_cls.gen_pmax[i] 
                                  for i in range(env_cls.n_gen)
                                  if env_cls.gen_type[i] == "solar"])
        cls._aux_add_solar(net)
        
        net._pmax_wind = np.sum([grid2op_env.gen_pmax[i] 
                                 for i in range(grid2op_env.n_gen)
                                 if grid2op_env.gen_type[i] == "wind"])
        cls._aux_add_wind(net)
        
        carrier_types_to_exclude = ['wind', 'solar']
        env_cls = type(grid2op_env)
        for i, generator in enumerate(env_cls.name_gen):
            gen_type = env_cls.gen_type[i]
            p_max = env_cls.gen_pmax[i]
            ramp_up = env_cls.gen_max_ramp_up[i]
            ramp_down = env_cls.gen_max_ramp_down[i]     
            gen_cost_per_MW = env_cls.gen_cost_per_MW[i]
            cls._aux_add_controlable_gen(net, gen_type, carrier_types_to_exclude, p_max, generator, ramp_up, ramp_down, gen_cost_per_MW)
        return net

    @classmethod
    def from_dataframe(cls,
                       env_df,
                       *,
                       pypsa_debug_add_inf_gen=False,
                       pypsa_debug_add_thermal=True,
                       pypsa_debug_add_hydro=True,
                       pypsa_debug_add_nuc=True,
                       pypsa_debug_add_solar=True,
                       pypsa_debug_add_wind=True,
                       pypsa_debug_add_slack=True):
        """
        Implements the abstract method of *Dispatcher*

        Parameters
        ----------
        grid2op_env

        Returns
        -------
        net: :class:`pypsa.Network`
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            # issue with pypsa on some version of numpy 
            net = cls(pypsa_debug_add_inf_gen=pypsa_debug_add_inf_gen,
                      pypsa_debug_add_thermal=pypsa_debug_add_thermal,
                      pypsa_debug_add_hydro=pypsa_debug_add_hydro,
                      pypsa_debug_add_nuc=pypsa_debug_add_nuc,
                      pypsa_debug_add_solar=pypsa_debug_add_solar,
                      pypsa_debug_add_wind=pypsa_debug_add_wind,
                      pypsa_debug_add_slack=pypsa_debug_add_slack,
                      )
        net._df = env_df
        
        # add total wind and solar (for curtailment)
        net._pmax_solar = np.sum([p_max
                              for i, (gen_type, p_max) in enumerate(zip(env_df['type'],
                                                                        env_df['pmax']))
                              if gen_type == "solar"])
        cls._aux_add_solar(net)
        
        net._pmax_wind = np.sum([p_max
                              for i, (gen_type, p_max) in enumerate(zip(env_df['type'],
                                                                        env_df['pmax']))
                              if gen_type == "wind"])
        cls._aux_add_wind(net)
        
        carrier_types_to_exclude = ['wind', 'solar']

        for i, (generator, gen_type, p_max, ramp_up, ramp_down, gen_cost_per_MW) in enumerate(zip(env_df['name'],
                                            env_df['type'],
                                            env_df['pmax'],
                                            env_df['max_ramp_up'],
                                            env_df['max_ramp_down'],
                                            env_df['cost_per_mw'])):
            cls._aux_add_controlable_gen(net, gen_type, carrier_types_to_exclude, p_max, generator, ramp_up, ramp_down, gen_cost_per_MW)
        return net

    def run(self,
            load, 
            total_solar,
            total_wind,
            params,
            gen_constraints=None,
            ramp_mode=RampMode.hard,
            by_carrier=False,
            gen_min_pu_t=None,
            gen_max_pu_t=None,
            **kwargs):
        """
        Implements the abstract method of *Dispatcher*

        Returns
        -------
        simplified_net: :class:`pypsa.Network`
        """
        if total_solar is not None:
            total_solar = total_solar / self._pmax_solar
        if total_wind is not None:
            total_wind = total_wind / self._pmax_wind
            
        prods_dispatch, terminal_conditions, marginal_prices = \
            main_run_disptach(
                self if not by_carrier else self.simplify_net(),
                load, total_solar, total_wind, 
                params, gen_constraints, ramp_mode,
                gen_min_pu_t=gen_min_pu_t, gen_max_pu_t=gen_max_pu_t,
                **kwargs)
        if prods_dispatch is None or marginal_prices is None:
            return None
        
        if by_carrier:
            self._simplified_chronix_scenario = self._chronix_scenario.simplify_chronix()
            self._simplified_chronix_scenario.prods_dispatch = prods_dispatch
            self._simplified_chronix_scenario.marginal_prices = marginal_prices
            results = self._simplified_chronix_scenario
            self._has_simplified_results = True
            self._has_results = False
        else:
            self._chronix_scenario.prods_dispatch = prods_dispatch
            self._chronix_scenario.marginal_prices = marginal_prices
            results = self._chronix_scenario
            self._has_results = True
            self._has_simplified_results = False
        if self._env is None:
            self.reset_ramps_from_dataframe()
        else:
            self.reset_ramps_from_grid2op_env()
        
        return DispatchResults(chronix=results, terminal_conditions=terminal_conditions)

    def simplify_net(self):
        """
        Implements the abstract method of *Dispatcher*
        """
        carriers = self.generators.carrier.unique()
        simplified_net = PypsaDispatcher()
        for carrier in carriers:
            names = self.generators[self.generators.carrier == carrier].index.tolist()

            gens = self.generators.loc[
                names,
                ['p_nom', 'ramp_limit_up', 'ramp_limit_down', 'marginal_cost']
            ]
            gens['ramp_up_mw'] = gens['p_nom'] * gens['ramp_limit_up']
            gens['ramp_down_mw'] = gens['p_nom'] * gens['ramp_limit_down']
            params = gens.agg(['sum', 'mean'])
            simplified_net.add(
                class_name='Generator', name=carrier, bus='node',
                p_nom=params.loc['sum', 'p_nom'], carrier=carrier,
                marginal_cost=params.loc['mean', 'marginal_cost'],
                ramp_limit_up=params.loc['sum', 'ramp_up_mw'] / params.loc['sum', 'p_nom'],
                ramp_limit_down=params.loc['sum', 'ramp_down_mw'] / params.loc['sum', 'p_nom'],
            )
        simplified_net._hydro_file_path = self._hydro_file_path
        simplified_net._min_hydro_pu = self._min_hydro_pu.iloc[:, 0]
        simplified_net._max_hydro_pu = self._max_hydro_pu.iloc[:, 0]

        print('simplified dispatch by carrier')
        full_ramp = simplified_net.generators['p_nom'] * simplified_net.generators['ramp_limit_up']
        df_full_ramp = pd.DataFrame({'full_ramp': full_ramp})

        print(pd.concat(
            [
                simplified_net.generators[['p_nom', 'ramp_limit_up',
                                           'ramp_limit_down', 'marginal_cost']],
                df_full_ramp
            ], axis=1))
        return simplified_net
