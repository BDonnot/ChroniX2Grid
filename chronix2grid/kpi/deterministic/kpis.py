# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)

# Python built-in modules
import os

# Python packages for data processing
from scipy.fft import fft, fftfreq
import numpy as np
import pandas as pd
from itertools import product

# Plot libraries
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

plt.ioff() #to not launch windows with plot in os


# Class definition
class EconomicDispatchValidator:
    """
    Computes thematic KPIs from reference and synthetic power chronics, and write resulting charts in an image repository

    Initialize the attributes of EconomicDispatchValidator with the following arguments of constructor

    Parameters
    ----------
    ref_consumption: :class:`pandas.DataFrame`
        reference power consumption chronics per load node of the grid
    syn_consumption: :class:`pandas.DataFrame`
        synthetic power consumption chronics per load node of the grid
    ref_dispatch: :class:`pandas.DataFrame`
        reference power generation chronics per generator of the grid
    synthetic_dispatch: :class:`pandas.DataFrame`
        synthetic power generation chronics per generator of the grid
    year: ``int``
        generation year
    images_repo: ``str``
        folder in which kpi images will be stored
    prods_charac: :class:`pandas.DataFrame`
        dataframe with infos on generators (region, carrier, Pmax...)
    loads_charac: :class:`pandas.DataFrame`
        dataframe with infos on load nodes (region, carrier, Pmax...)
    ref_prices: :class:`pandas.DataFrame` or None
        price scenario for reference chronics (used for thermal and hydro kpi). If not provided, those KPI will use quantiles on load instead of price
    syn_prices: :class:`pandas.DataFrame` or None
        price scenario for synthetic chronics (used for thermal and hydro kpi). If not provided, those KPI will use quantiles on load instead of price

    """
    def __init__(self, ref_consumption, syn_consumption, ref_dispatch,
                 synthetic_dispatch, year, images_repo,
                 prods_charac=None, loads_charac=None, ref_prices=None,
                 syn_prices=None,kpi_on_syn_data_only=False):

        ## Constructor

        # Chronics
        self.syn_consumption = syn_consumption
        self.ref_consumption = ref_consumption
        self.ref_dispatch = ref_dispatch
        self.syn_dispatch = synthetic_dispatch
        self.ref_prices = ref_prices
        self.syn_prices = syn_prices
        self.kpi_on_syn_data_only=kpi_on_syn_data_only 

        # Constants
        #self.dt = (ref_dispatch.index.values[1] - ref_dispatch.index.values[0])/pd.Timedelta(minutes = 1) #does not work if ref_dispatch is none
        self.dt = (synthetic_dispatch.index.values[1] - synthetic_dispatch.index.values[0]) / pd.Timedelta(minutes=1)

        # Create repos if necessary for plot saving
        self.image_repo = images_repo
        if not os.path.exists(os.path.join(self.image_repo, 'dispatch_view')):
            os.mkdir(os.path.join(self.image_repo, 'dispatch_view'))
            os.mkdir(os.path.join(self.image_repo, 'wind_kpi'))
            os.mkdir(os.path.join(self.image_repo, 'wind_load_kpi'))
            os.mkdir(os.path.join(self.image_repo, 'solar_kpi'))
            os.mkdir(os.path.join(self.image_repo, 'nuclear_kpi'))
            os.mkdir(os.path.join(self.image_repo, 'hydro_kpi'))
            os.mkdir(os.path.join(self.image_repo, 'thermal_kpi'))
            os.mkdir(os.path.join(self.image_repo, 'thermal_load_kpi'))
            os.mkdir(os.path.join(self.image_repo, 'load_kpi'))


        # Aggregate chronics variables
        if not self.kpi_on_syn_data_only:
            self.ref_agg_conso = ref_consumption.sum(axis=1)
        else:
            self.ref_agg_conso=None
        self.syn_agg_conso = syn_consumption.sum(axis=1)
        if not self.kpi_on_syn_data_only:
            self.agg_ref_dispatch = ref_dispatch.sum(axis=1)
        else:
            self.agg_ref_dispatch=None
        self.agg_syn_dispatch = synthetic_dispatch.sum(axis=1)
        
        # Read grid characteristics
        if prods_charac is not None:
            self.regions = prods_charac['zone'].unique()
        self.prod_charac = prods_charac
        self.load_charac = loads_charac
        if syn_prices is not None and ref_prices is not None:
            self.quantile_mode = 'prices'
        else:
            self.quantile_mode = 'load'
            print("Warning: prices data have not been given for both synthetic and reference dispatch. "
                  "Quantiles will be computed on demand instead. "
                  "Next time, you can provide .../France/eco2mix/prices_"+str(year)+".csv.bz2 \n")


        # Months are used in multiple KPI's
        #self.months = self.ref_dispatch.index.month.to_frame()
        #self.months.index = self.ref_dispatch.index
        self.months = self.syn_dispatch.index.month.to_frame()
        self.months.index = self.syn_dispatch.index
        self.months.columns = ['month']
        
        # Outputs features
        self.precision = 1
        self.output = {}


    def _plot_heatmap(self, corr, title, path_png=None, save_png=True):
        
        ax = sns.heatmap(corr, 
                        fmt='.1f',
                        annot = False,
                        vmin=-1, vmax=1, center=0,
                        cmap=sns.diverging_palette(20, 220, n=200),
                        cbar_kws={"orientation": "horizontal", 
                                  "shrink": 0.35,
                                  },
                        square=True,
                        linewidths = 0.5,
        )
        ax.set_xticklabels(ax.get_xticklabels(),
                          rotation=45,
                          horizontalalignment='right'
        )

        ax.set_title(title, fontsize=15)

        if save_png:
            figure = ax.get_figure()    
            figure.savefig(path_png)

    def plot_barcharts(self, df_ref, df_syn, save_plots = True, path_name = '', title_component = '', normalized = True, every_nth = None):
        # Plot results
        if normalized:
            if not self.kpi_on_syn_data_only:
                df_ref = df_ref/df_ref.max()
            df_syn = df_syn / df_syn.max()

        plt.ioff()  # to not launch windows with plot in os
        if self.kpi_on_syn_data_only:
            fig, axes = plt.subplots(1, 1, figsize=(17, 5))
            sns.barplot(df_syn.index, df_syn)
            axes.set_title('Synthetic '+title_component, size = 9)
            if every_nth is not None:
                for n, label in enumerate(axes.xaxis.get_ticklabels()):
                    if n % every_nth != 0:
                        label.set_visible(False)
        else:
            fig, axes = plt.subplots(1, 2, figsize=(17, 5))
            sns.barplot(df_ref.index, df_ref, ax=axes[0])
            sns.barplot(df_syn.index, df_syn, ax=axes[1])
            axes[0].set_title('Reference '+title_component, size = 9)
            axes[1].set_title('Synthetic '+title_component, size = 9)

            if every_nth is not None:
                for i in range(2):
                    for n, label in enumerate(axes[i].xaxis.get_ticklabels()):
                        if n % every_nth != 0:
                            label.set_visible(False)

        if save_plots:
            fig.savefig(path_name)

    def energy_mix(self, save_plots = True):
        """
        Compute piecharts for total energy mix
        """

        # Sum of production per generator type
        if not self.kpi_on_syn_data_only:
            ref_prod_per_gen = self.ref_dispatch.sum(axis = 0)
            ref_prod_per_gen = pd.DataFrame({"Prod": ref_prod_per_gen.values, "name":ref_prod_per_gen.index})
            ref_prod_per_gen = ref_prod_per_gen.merge(self.prod_charac[["name","type"]], how = 'left',
                                                      on = 'name')
            ref_prod_per_gen = ref_prod_per_gen.groupby('type').sum()
            ref_prod_per_gen = ref_prod_per_gen.sort_index()

        syn_prod_per_gen = self.syn_dispatch.sum(axis=0)
        syn_prod_per_gen = pd.DataFrame({"Prod": syn_prod_per_gen.values, "name": syn_prod_per_gen.index})
        syn_prod_per_gen = syn_prod_per_gen.merge(self.prod_charac[["name", "type"]], how='left',
                                                  on='name')
        syn_prod_per_gen = syn_prod_per_gen.groupby('type').sum()
        syn_prod_per_gen = syn_prod_per_gen.sort_index()

        # Carrier values for label
        #labels = ref_prod_per_gen.index.unique()
        labels = syn_prod_per_gen.index.unique()

        # Distribution of prod
        plt.ioff()  # to not launch windows with plot in os
        if self.kpi_on_syn_data_only:
            fig, axes = plt.subplots(1, 1, figsize=(17, 5))
            axes.pie(syn_prod_per_gen["Prod"], labels=labels, autopct='%1.1f%%')
            axes.set_title('Synthetic Energy Mix')
        else:
            fig, axes = plt.subplots(1, 2, figsize=(17, 5))
            axes[0].pie(ref_prod_per_gen["Prod"], labels=labels, autopct='%1.1f%%')
            axes[1].pie(syn_prod_per_gen["Prod"], labels=labels, autopct='%1.1f%%')
            axes[0].set_title('Reference Energy Mix')
            axes[1].set_title('Synthetic Energy Mix')
        if save_plots:
            # Save plot as png
            fig.savefig(os.path.join(self.image_repo, 'dispatch_view', 'energy_mix.png'))

    def _pairwise_corr_different_dfs(self, df1, df2):
        """
        Compute pairwise correlation of 2 dataframes
        """        
        n_col_df1 = df1.shape[1]
        n_col_df2 = df2.shape[1]

        tmp_corr = np.zeros((n_col_df1, n_col_df2))
        for i, j in product(range(n_col_df1), range(n_col_df2)):
            tmp_corr[i, j] = df1.iloc[:, i].astype('float64').corr(df2.iloc[:, j].astype('float64'))

        corr_wind = pd.DataFrame(tmp_corr, index=df1.columns, columns=df2.columns)

        return corr_wind.round(self.precision)

    def add_trace_in_subplot(self, fig, x=None, y=None,
                             in_row=1, in_col=1, stacked=None, name = None):
        """
        Add invididual time series to the subplot
        """
        fig.add_trace(go.Scatter(x=x, y=y, stackgroup=stacked, name = name),
                      row=in_row, col=in_col)

    def plot_carriers_pw(self, curve = 'synthetic' ,stacked=True, max_col_splot=2, save_html = True, wind_solar_only = False):
        """
        Generate a temporal view of production by generators, one graph per carrier.
        The generated plot is an html interactive file
        """

        # Set chronics to consider
        if curve == 'synthetic':
            prod_p = self.syn_dispatch.copy()
        elif curve == 'reference':
            prod_p = self.ref_dispatch.copy()

        # Initialize full gen dataframe
        df_mw = pd.DataFrame()

        # Num unique carriers
        if wind_solar_only:
            unique_carriers = ['solar','wind']
        else:
            unique_carriers = self.prod_charac['type'].unique().tolist()

        # Initialize the plot
        rows = int(np.ceil(len(unique_carriers) / max_col_splot))
        fig = make_subplots(rows=rows,
                            cols=max_col_splot,
                            subplot_titles=unique_carriers)

        # Visualize stacked plots?
        if stacked:
            stacked_method = 'one'
        else:
            stacked_method = None

        x = prod_p.index
        row = col = 1
        for carrier in unique_carriers:
            # Get the gen names per carrier
            carrier_filter = self.prod_charac['type'].isin([carrier])
            gen_names = self.prod_charac.loc[carrier_filter]['name'].tolist()

            # Agregate me per carrier in dt
            if len(gen_names) != len(list(set(gen_names) & set(prod_p.columns))):
                #raise Exception
                print("For "+carrier+" type, there are some inconsitencies for the generator names between the reference data and generated data. The reference data names should match the generate data names. "
                                               "Generated data names: "+str(gen_names)+" . Reference data names "+str(prod_p.columns))
                return fig,df_mw
            else:
                tmp_df_mw = prod_p[gen_names].sum(axis=1)
                df_mw = pd.concat([df_mw, tmp_df_mw], axis=1)

                for gen in gen_names:
                    # Add trace per carrier in same axes
                    self.add_trace_in_subplot(fig, x=x, y=prod_p[gen],
                                         in_row=row, in_col=col, stacked=stacked_method, name = gen)

                    # Once all ts have been added, create a new subplot
                col += 1
                if col > max_col_splot:
                    col = 1
                    row += 1

        # Rename df_mw columns
        df_mw.columns = unique_carriers

        if save_html:
            if(stacked):
                fig.write_html(os.path.join(self.image_repo,'dispatch_view',str(curve)+'_stacked_prod_per_carrier.html'))
            else:
                fig.write_html(os.path.join(self.image_repo, 'dispatch_view', str(curve) + '_timeseries_prod_per_carrier.html'))
        return fig, df_mw

    def plot_load_pw(self ,stacked=True, max_col_splot=1, save_html = True):
        """
        Generate a temporal view of production by generators, one graph per carrier
        The generated plot is an html interactive file
        """

        # Set chronics to consider
        prod_p = self.syn_consumption.copy()

        # Initialize full gen dataframe
        df_mw = pd.DataFrame()

        # Num unique carriers
        unique_carriers = self.load_charac['type'].unique().tolist()

        # Initialize the plot
        rows = int(np.ceil(len(unique_carriers) / max_col_splot))
        fig = make_subplots(rows=rows,
                            cols=max_col_splot,
                            subplot_titles=unique_carriers)

        # Visualize stacked plots?
        if stacked:
            stacked_method = 'one'
        else:
            stacked_method = None

        x = prod_p.index
        row = col = 1
        for carrier in unique_carriers:
            # Get the gen names per carrier
            carrier_filter = self.load_charac['type'].isin([carrier])
            gen_names = self.load_charac.loc[carrier_filter]['name'].tolist()

            # Agregate me per carrier in dt
            tmp_df_mw = prod_p[gen_names].sum(axis=1)
            df_mw = pd.concat([df_mw, tmp_df_mw], axis=1)

            for gen in gen_names:
                # Add trace per carrier in same axes
                self.add_trace_in_subplot(fig, x=x, y=prod_p[gen],
                                     in_row=row, in_col=col, stacked=stacked_method, name = gen)

                # Once all ts have been added, create a new subplot
            col += 1
            if col > max_col_splot:
                col = 1
                row += 1

        # Rename df_mw columns
        df_mw.columns = unique_carriers

        if save_html:
            fig.write_html(os.path.join(self.image_repo,'dispatch_view','load_per_carrier.html'))
        return fig, df_mw


    def __hydro_in_prices(self,
                          norm_mw,
                          prices_or_demand,
                          upper_quantile, 
                          lower_quantile,
                          above_norm_cap,
                          below_norm_cap):

        
        # Get number of gen units
        n_units = norm_mw.shape[1]
        # Get the price at upper/lower quantile
        eu_upper_quantile = prices_or_demand.quantile(upper_quantile).values[0]
        eu_lower_quantile = prices_or_demand.quantile(lower_quantile).values[0]


        # Test if units are above/below normalized capacity
        is_gens_above_cap =  norm_mw.ge(above_norm_cap, axis=1)
        is_gens_below_cap =  norm_mw.le(below_norm_cap, axis=1)

        # Test if prices are greater/lower than defined quantiles
        is_price_above_q = prices_or_demand >= eu_upper_quantile
        is_price_below_q = prices_or_demand <= eu_lower_quantile

        # Stacking price bool condition for all hydro units along
        # the columns and rename them as units names
        colName = prices_or_demand.columns.values[0]
        is_price_above_q_gens = is_price_above_q[[colName] * n_units]
        is_price_above_q_gens.columns = norm_mw.columns

        is_price_below_q_gens = is_price_below_q[[colName] * n_units]
        is_price_below_q_gens.columns = norm_mw.columns
        
        # Match occurence of price high and full gen disaptch
        high_price_kpi = 100 * (is_price_above_q_gens & is_gens_above_cap).sum(axis=0) \
                        / is_price_above_q_gens.sum(axis=0)


        # Match occurence when price is lower and almost no dispatch
        low_price_kpi = 100 * (is_price_below_q_gens & is_gens_below_cap).sum(axis=0) \
                        / is_price_below_q_gens.sum(axis=0)

        return high_price_kpi.round(self.precision), low_price_kpi.round(self.precision)
               
    def __hydro_seasonal(self, hydro_mw):

        hydro_mw_month = hydro_mw.copy()
        hydro_mw_month['month'] = self.months
        mw_per_month = hydro_mw_month.groupby('month').mean().round(self.precision)
        
        return mw_per_month

        
    def hydro_kpi(self,
                  upper_quantile = 0.9,
                  lower_quantile = 0.1,
                  above_norm_cap = 0.8,
                  below_norm_cap = 0.2):

        '''
        Compute Hydro KPIs for synthetic and reference chronics
        '''  

        # Get Hydro names
        hydro_filter = self.prod_charac.type.isin(['hydro'])
        hydro_names = self.prod_charac.name.loc[hydro_filter].values

        # Normalize MW according to the max value for the reference data and synthetic one
        stat_ref_high_price=None
        stat_ref_low_price =None

        if not self.kpi_on_syn_data_only:
            hydro_ref = self.ref_dispatch[hydro_names]
            max_mw_ref = hydro_ref.max(axis=0)
            norm_mw_ref = hydro_ref / max_mw_ref

            # Stats for reference data
            if self.quantile_mode == 'prices':
                for_quantiles = self.ref_prices
            else:
                for_quantiles = pd.DataFrame({'conso': self.ref_agg_conso})
            stat_ref_high_price, stat_ref_low_price = self.__hydro_in_prices(norm_mw_ref,
                                                                             for_quantiles,
                                                                             upper_quantile,
                                                                             lower_quantile,
                                                                             above_norm_cap,
                                                                             below_norm_cap
                                                                             )


        hydro_syn = self.syn_dispatch[hydro_names]
        max_mw_syn = hydro_syn.max(axis=0)
        norm_mw_syn = hydro_syn / max_mw_syn
        

        
        # Stats for synthetic data
        if self.quantile_mode == 'prices':
            for_quantiles = self.syn_prices
        else:
            for_quantiles = pd.DataFrame({'conso':self.syn_agg_conso})
        stat_syn_high_price, stat_syn_low_price = self.__hydro_in_prices(norm_mw_syn,
                                                                         for_quantiles,
                                                                         upper_quantile, 
                                                                         lower_quantile,
                                                                         above_norm_cap,
                                                                         below_norm_cap
                                                                         )


        self.plot_barcharts(stat_ref_high_price, stat_syn_high_price, save_plots=True,
                        path_name=os.path.join(self.image_repo, 'hydro_kpi', 'high_price.png'),
                        title_component='% of time production exceed ' + str(above_norm_cap) +
                                        '*Pmax when prices are high (above quantile ' + str(
                            upper_quantile * 100) + ')',
                        normalized=False)


        self.plot_barcharts(stat_ref_low_price, stat_syn_low_price, save_plots=True,
                path_name=os.path.join(self.image_repo, 'hydro_kpi', 'low_price.png'),
                title_component='% of time production is below ' + str(below_norm_cap) +
                                '*Pmax when prices are low (under quantile ' + str(
                    lower_quantile * 100) + ')', normalized=False)

        
        # Seasonal for reference data
        ref_agg_mw_per_month = None
        ref_agg_mw_per_month_sum = 0
        if not self.kpi_on_syn_data_only:
            ref_agg_mw_per_month=self.__hydro_seasonal(hydro_ref)
            ref_agg_mw_per_month_sum=ref_agg_mw_per_month.sum(axis = 1)

        # Seasonal for synthetic data
        syn_agg_mw_per_month = self.__hydro_seasonal(hydro_syn)


        self.plot_barcharts( ref_agg_mw_per_month_sum,syn_agg_mw_per_month.sum(axis = 1), save_plots=True,
                            path_name=os.path.join(self.image_repo,'hydro_kpi','hydro_per_month.png'),
                            title_component='hydro mean production per month (% of max) for all units')

        # Write results
        # -- + -- + --
        if self.kpi_on_syn_data_only:
            self.output['Hydro'] = {
                                    'seasonal_month_for_syn': syn_agg_mw_per_month.to_dict(),
                                    'high_price_for_syn': stat_syn_high_price.to_dict(),
                                    'low_price_for_syn': stat_syn_low_price.to_dict()
                                   }
        else:
            self.output['Hydro'] = {'seasonal_month_for_ref': ref_agg_mw_per_month.to_dict(),
                                    'seasonal_month_for_syn': syn_agg_mw_per_month.to_dict(),
                                    'high_price_for_ref': stat_ref_high_price.to_dict(),
                                     'low_price_for_ref': stat_ref_low_price.to_dict(),
                                     'high_price_for_syn': stat_syn_high_price.to_dict(),
                                     'low_price_for_syn': stat_syn_low_price.to_dict()

                                   }

        return stat_ref_high_price, stat_ref_low_price, ref_agg_mw_per_month, \
               stat_syn_high_price, stat_syn_low_price, syn_agg_mw_per_month

        
    def __wind_metric_distrib(self, wind_df):

        
        # Add month as column to compute stats
        copied_wind_df = wind_df.copy()
        # wind_df_month['month'] = self.months
        
        # Get Skewness agg per month
        skewness_per_month = copied_wind_df.groupby(pd.Grouper(freq='M')).apply(lambda x: x.skew()).round(2)      
        # Set index as month value
        skewness_per_month.index = skewness_per_month.index.month
        skewness_per_month.index.rename('month', inplace=True)
        
        # Get Kurtosis agg per month                      
        kurtosis_per_month = copied_wind_df.groupby(pd.Grouper(freq='M')).apply(lambda x: x.kurtosis()).round(2)
        # Set index as month value
        kurtosis_per_month.index = kurtosis_per_month.index.month
        kurtosis_per_month.index.rename('month', inplace=True)

                                         
        return skewness_per_month, kurtosis_per_month
        
    def wind_kpi(self, save_plots=True):

        '''
        Computes Wind production KPIs
        '''

        # First KPI
        # Correlation between wind time series
        
        # Get the wind gen names
        wind_filter = self.prod_charac.type.isin(['wind'])
        wind_names = self.prod_charac.name[wind_filter]

        # From dispatch, get only wind units
        if not self.kpi_on_syn_data_only:
            wind_ref = self.ref_dispatch[wind_names]
        wind_syn = self.syn_dispatch[wind_names]
    
        # Compute correlation for all elements between both dataframes
        if not self.kpi_on_syn_data_only:
            ref_corr_wind = self._pairwise_corr_different_dfs(wind_ref, wind_ref).fillna(0)
        syn_corr_wind = self._pairwise_corr_different_dfs(wind_syn, wind_syn).fillna(0)
        
        # Write results json output
        # -- + -- + -- + -- + -- + 
        self.output['wind_kpi'] = {'corr_wind': syn_corr_wind.to_dict()}
        
        # Second KPI
        # Measure non linearity of time series
        # chaoticness_ref = self.__wind_entropy(wind_ref)
        # chaoticness_syn = self.__wind_entropy(wind_syn)
        #
        # # Write results
        # # -- + -- + --
        # self.output['wind_kpi'] = {'non_linearity_reference': chaoticness_ref.to_dict(),
        #                            'non_linearity_synthetic': chaoticness_syn.to_dict(),
        #                            }
        
        # Third KPI
        # Meaure the simmetry of wind distribution
        if not self.kpi_on_syn_data_only:
            skewness_ref, kurtosis_ref = self.__wind_metric_distrib(wind_ref)
            skewness_ref_sum=skewness_ref.sum(axis=1)
            kurtosis_ref_sum = skewness_ref.sum(axis=1)
        else:
            skewness_ref=None
            kurtosis_ref=None
            skewness_ref_sum=None
            kurtosis_ref_sum=None
        skewness_syn, kurtosis_syn = self.__wind_metric_distrib(wind_syn)

        self.plot_barcharts(skewness_ref_sum, skewness_syn.sum(axis=1), save_plots=True,
                            path_name=os.path.join(self.image_repo,'wind_kpi','skewness.png'),
                            title_component='skewness per month', normalized=False)
        self.plot_barcharts(kurtosis_ref_sum, kurtosis_syn.sum(axis=1), save_plots=True,
                            path_name=os.path.join(self.image_repo, 'wind_kpi', 'kurtosis.png'),
                            title_component='kurtosis per month', normalized=False)

        # Write results 
        # -- + -- + --
        if self.kpi_on_syn_data_only:
            self.output['wind_kpi'] = {'skewness_synthetic': skewness_syn.to_dict(),
                                       'kurtosis_synthetic': kurtosis_syn.to_dict(),
                                       }
        else:
            self.output['wind_kpi'] = {'skewness_reference': skewness_ref.to_dict(),
                                       'skewness_synthetic': skewness_syn.to_dict(),
                                       'kurtosis_reference': kurtosis_ref.to_dict(),
                                       'kurtosis_synthetic': kurtosis_syn.to_dict(),
                                       }
        
        # Four KPI
        # Plot distributions

        # Aggregate time series
        if not self.kpi_on_syn_data_only:
            agg_ref_wind = wind_ref.sum(axis=1)
        agg_syn_wind = wind_syn.sum(axis=1)

        # Plot results
        # Correlation heatmaps
        plt.ioff()  # to not launch windows with plot in os
        if self.kpi_on_syn_data_only:
            fig, axes = plt.subplots(1, 1, figsize=(17,5))
            sns.heatmap(syn_corr_wind, annot = True, linewidths=.5)
        else:
            fig, axes = plt.subplots(1, 2, figsize=(17,5))
            sns.heatmap(ref_corr_wind, annot=True, linewidths=.5, ax=axes[0])
            sns.heatmap(syn_corr_wind, annot = True, linewidths=.5, ax=axes[1])
        if save_plots:
            fig.savefig(os.path.join(self.image_repo, 'wind_kpi', 'wind_corr_heatmap.png'))

        # Distribution of prod
        if self.kpi_on_syn_data_only:
            fig, axes = plt.subplots(1, 1, figsize=(17, 5))
            sns.distplot(agg_syn_wind)
            axes.set_title('Synthetic Distribution of agregate wind production')
        else:
            fig, axes = plt.subplots(1, 2, figsize=(17, 5))
            sns.distplot(agg_ref_wind, ax=axes[0])
            sns.distplot(agg_syn_wind, ax=axes[1])
            axes[0].set_title('Reference Distribution of agregate wind production')
            axes[1].set_title('Synthetic Distribution of agregate wind production')
        if save_plots:
            # Save plot as png
            fig.savefig(os.path.join(self.image_repo,'wind_kpi','histogram.png'))

        ## Power spectral density
        #Ref
        if not self.kpi_on_syn_data_only:
            ref_transform = fft(agg_ref_wind.values)
            ref_density = [(z.real ** 2 + z.imag ** 2) for z in ref_transform]
            ref_freq = fftfreq(len(ref_density), d=self.dt * 60)

        # Syn
        syn_transform = fft(agg_syn_wind.values)
        syn_density = [(z.real ** 2 + z.imag ** 2) for z in syn_transform]
        syn_freq = fftfreq(len(syn_density), d=self.dt * 60)

        # Plot in log scale
        if self.kpi_on_syn_data_only:
            fig, axes = plt.subplots(1, 1, figsize=(17, 5))
            axes.plot(np.sort(syn_freq), syn_density, color = 'grey')
            axes.set(xscale='log', yscale='log', xlabel='Frequency (Hz)', ylabel='Power spectral density',
                        title = 'Synthetic power spectral density')
        else:
            fig, axes = plt.subplots(1, 2, figsize=(17, 5))
            axes[0].plot(np.sort(ref_freq), ref_density, color = 'grey')
            axes[0].set(xscale = 'log', yscale = 'log', xlabel = 'Frequency (Hz)', ylabel = 'Power spectral density',
                        title = 'Reference power spectral density')
            axes[1].plot(np.sort(syn_freq), syn_density, color = 'grey')
            axes[1].set(xscale='log', yscale='log', xlabel='Frequency (Hz)', ylabel='Power spectral density',
                        title = 'Synthetic power spectral density')
        if save_plots:
            fig.savefig(os.path.join(self.image_repo, 'wind_kpi', 'power_spectral_density.png'))

        ## Auto-correlation
        maxlags = 15
        if self.kpi_on_syn_data_only:
            n_syn = len(wind_syn.columns)
            n = n_syn

            fig, axes = plt.subplots(1, n, figsize=(30, 10))
            for i, gen in enumerate(wind_syn.columns):
                ts = wind_syn[gen].values
                if n==1:
                    ax = axes#[0]
                else:
                    ax = axes[i]
                ax.acorr(ts, maxlags=maxlags)
                ax.set_title('Synthetic '+gen + ' ACF', fontsize = 8)
                ax.set_xlim(1, maxlags)
        else:
            n_ref = len(wind_ref.columns)
            n_syn = len(wind_syn.columns)
            n = max(n_ref, n_syn)

            fig, axes = plt.subplots(2, n, figsize=(30, 10))
            for i, gen in enumerate(wind_ref.columns):
                ts = wind_ref[gen].values
                if n==1:
                    ax = axes[0]
                else:
                    ax = axes[0,i]
                ax.acorr(ts, maxlags=maxlags)
                ax.set_title('Reference '+gen + ' ACF', fontsize = 8)
                ax.set_xlim(1,maxlags)
            for i, gen in enumerate(wind_syn.columns):
                ts = wind_syn[gen].values
                if n==1:
                    ax = axes[1]
                else:
                    ax = axes[1,i]
                ax.acorr(ts, maxlags=maxlags)
                ax.set_title('Synthetic '+gen + ' ACF', fontsize = 8)
                ax.set_xlim(1, maxlags)
        if save_plots:
            fig.savefig(os.path.join(self.image_repo, 'wind_kpi', 'generators_autocorrelation.png'))

        ## Cross-correlation
        maxlags = 10

        if not self.kpi_on_syn_data_only:
            n_ref = len(wind_ref.columns)
            fig, axes = plt.subplots(sharex=True, sharey=True, ncols=n_ref, nrows=n_ref-1, figsize=(30, 13))
            fig.suptitle('Reference cross-correlation between wind generators', fontsize=16)
            for i, geni in enumerate(wind_ref.columns):
                for j, genj in enumerate(wind_ref.columns.values[:i]):
                    tsi = wind_ref[geni].values
                    tsj = wind_ref[genj].values
                    axes[i-1, j].xcorr(tsi, tsj, maxlags=maxlags)
                    #axes[i,j].set()
            if save_plots:
                fig.savefig(os.path.join(self.image_repo, 'wind_kpi', 'reference_generators_cross-correlation.png'))

        n_syn = len(wind_syn.columns)
        fig, axes = plt.subplots(sharex=True, sharey=True, ncols=n_syn, nrows=n_syn - 1, figsize=(30, 13))
        fig.suptitle('Synthetic cross-correlation between wind generators', fontsize=16)
        for i, geni in enumerate(wind_syn.columns):
            for j, genj in enumerate(wind_syn.columns.values[:i]):
                tsi = wind_syn[geni].values
                tsj = wind_syn[genj].values
                axes[i - 1, j].xcorr(tsi, tsj, maxlags=maxlags)
                # axes[i,j].set()
        if save_plots:
            fig.savefig(os.path.join(self.image_repo, 'wind_kpi', 'synthetic_generators_cross-correlation.png'))


        return syn_corr_wind, \
               skewness_ref, skewness_syn, \
               kurtosis_ref, kurtosis_syn



    def __solar_at_night(self,
                         solar_df,
                         params,
                         aggregated=False
                         ):
        
        if aggregated:
            solar_df = solar_df.sum(axis=1)
            
        # Extract parameters
        monthly_pattern = params['monthly_pattern']
        hours = params['hours']
        
        # Add month variable solar df
        solar_df_month = pd.concat([solar_df, self.months], axis=1)
        
        # Iterate over all season per month to check if solar is 
        # present during night hours defined by a deterministic 
        # criteria per season.
        season_at_night = {}
        for season, month in monthly_pattern.items():

            # Filter by season (including all possible months avaible in season)
            month_filter = solar_df_month.month.isin(month)
            season_solar = solar_df_month.loc[month_filter, solar_df_month.columns != 'month']
                    
            if not season_solar.empty:
                # Filter by season and hours
                sum_over_season = season_solar.drop(season_solar.between_time(hours[season][0], hours[season][1]).index).sum(axis=0)
                percen_over_season = 100 * sum_over_season / season_solar.sum(axis=0)
                percen_over_season = percen_over_season.round(self.precision)

                # Save results in a dictionary
                season_at_night.update({season : percen_over_season})
                
        return season_at_night

    def __solar_by_day(self,solar_df,params):
        # Extract parameters
        monthly_pattern = params['monthly_pattern']
        hours = params['hours']

        # Add month variable solar df
        solar_df_month = pd.concat([solar_df, self.months], axis=1)
        solar_df_by_day = pd.DataFrame(columns = solar_df.columns)

        for season, month in monthly_pattern.items():

            # Filter by season (including all possible months avaible in season)
            month_filter = solar_df_month.month.isin(month)
            season_solar = solar_df_month.loc[month_filter, solar_df_month.columns != 'month']

            if not season_solar.empty:
                # Filter by season and hours
                day_index = season_solar.between_time(hours[season][0], hours[season][1]).index
                season_solar_by_day = season_solar.loc[day_index,:]
                solar_df_by_day = pd.concat([solar_df_by_day,season_solar_by_day], axis = 0)
        return solar_df_by_day

        
    def __solar_cloudiness(self,
                           solar_df,
                           cloud_quantile,
                           factor_cloud):

        
        # Per day, we are interested to get x quantile only when
        # generators are producing energy
        solar_q_perday = solar_df.replace(0, np.nan).resample('D').quantile(cloud_quantile)

        # Compute quantile per month and diminuate with factor (thresholds)
        solar_q_permonth = solar_q_perday.copy()
        for month in self.months['month'].unique():
            for col in solar_df.columns:
                threshold = solar_q_perday.loc[solar_q_perday.index.month == month, col].quantile(cloud_quantile)
                solar_q_permonth.loc[solar_q_permonth.index.month == month, col] = (threshold * factor_cloud)

        # Measure cloudiness: we compare the quantile values per
        # day with a monthly x quantile truncated with a factor
        cloudiness = pd.DataFrame(solar_q_perday.values <= solar_q_permonth.values, index=solar_q_perday.index)
        
        percen_cloud = 100 * cloudiness.groupby(pd.Grouper(freq='M')).sum() \
                             / cloudiness.groupby(pd.Grouper(freq='M')).count()
                             
        percen_cloud.index = percen_cloud.index.month
        percen_cloud.index.rename('month', inplace=True)
        
        return percen_cloud.round(self.precision)
        
    def solar_kpi(self,
                  cloud_quantile=0.95,
                  cond_below_cloud=0.85
                  , save_plots = True,
                  **kwargs):

        '''

        Compute Solar production KPIs
        '''

        # Get the solar gen names
        solar_filter = self.prod_charac.type.isin(['solar'])
        solar_names = self.prod_charac.name.loc[solar_filter].values

        # From data, extract only solar time series
        if not self.kpi_on_syn_data_only:
            solar_ref = self.ref_dispatch[solar_names]
            agg_ref_solar = solar_ref.sum(axis=1)

        solar_syn = self.syn_dispatch[solar_names]
        agg_syn_solar = solar_syn.sum(axis = 1)

        # First KPI
        # -- + -- +

        # Distribution of prod
        plt.ioff()  # to not launch windows with plot in os
        if self.kpi_on_syn_data_only:
            fig, axes = plt.subplots(1, 1, figsize=(17, 5))
            sns.distplot(agg_syn_solar)
            axes.set_title('Synthetic Distribution of agregate solar production')
        else:
            fig, axes = plt.subplots(1, 2, figsize=(17, 5))
            sns.distplot(agg_ref_solar, ax=axes[0])
            sns.distplot(agg_syn_solar, ax=axes[1])
            axes[0].set_title('Reference Distribution of agregate solar production')
            axes[1].set_title('Synthetic Distribution of agregate solar production')
        if save_plots:
            # Save plot as png
            fig.savefig(os.path.join(self.image_repo, 'solar_kpi', 'histogram.png'))


        
        # Second KPI
        # -- + -- +
        # Get percentage solar at night
        
        if not kwargs:

            monthly_pattern = {'summer': [6,7,8], 'fall': [9,10,11],
                               'winter': [12,1,2], 'spring': [2,3,4,5]}

            hours = {'summer': ('07:00', '20:00'),
                     'fall': ('08:00', '18:00'),
                     'winter': ('09:30', '16:30'),
                     'spring': ('08:00', '18:00')}
            params = {'monthly_pattern': monthly_pattern, 'hours': hours}

        else:
            params = kwargs

        # Get percentage solar productions for reference data
        if not self.kpi_on_syn_data_only:
            solar_night_ref = self.__solar_at_night(solar_ref, params=params)
            solar_night_ref_mean = pd.DataFrame({key: [solar_night_ref[key].mean()] for key in solar_night_ref.keys()})
            solar_night_ref_mean = solar_night_ref_mean.sum(axis=0)
        else:
            solar_night_ref=None
            solar_night_ref_mean=None

        # Get percentage solar productions for synthetic data
        solar_night_syn = self.__solar_at_night(solar_syn, params=params)

        # Compute mean of generators
        solar_night_syn_mean = pd.DataFrame({key: [solar_night_syn[key].mean()] for key in solar_night_syn.keys()})
        solar_night_syn_mean = solar_night_syn_mean.sum(axis = 0)



        # Plot and save it average per season
        self.plot_barcharts(solar_night_ref_mean, solar_night_syn_mean, save_plots=True,
                            path_name=os.path.join(self.image_repo,'solar_kpi','solar_at_night.png'),
                            title_component='Mean % of production at night per season', normalized = False)

        ## Correlation Matrix by day
        # Get correlation matrix (10 x 10)
        if not self.kpi_on_syn_data_only:
            solar_ref_by_day = self.__solar_by_day(solar_ref, params)
            ref_corr_solar = self._pairwise_corr_different_dfs(solar_ref_by_day, solar_ref_by_day).fillna(0)

        solar_syn_by_day = self.__solar_by_day(solar_syn, params)

        syn_corr_solar = self._pairwise_corr_different_dfs(solar_syn_by_day, solar_syn_by_day).fillna(0)

        # Plot results
        # Correlation heatmaps
        if self.kpi_on_syn_data_only:
            fig, axes = plt.subplots(1, 1, figsize=(17, 5))
            sns.heatmap(syn_corr_solar, annot=True, linewidths=.5)
        else:
            fig, axes = plt.subplots(1, 2, figsize=(17, 5))
            sns.heatmap(ref_corr_solar, annot=True, linewidths=.5, ax=axes[0])
            sns.heatmap(syn_corr_solar, annot=True, linewidths=.5, ax=axes[1])
        if save_plots:
            fig.savefig(os.path.join(self.image_repo, 'solar_kpi', 'solar_corr_heatmap.png'))

        # Third KPI
        # -- + -- +
        # Measure Cloudiness as counting the number of days 
        # per month when the x quantile is below a factor
        # specified by the method.
        if not self.kpi_on_syn_data_only:
            cloudiness_ref = self.__solar_cloudiness(solar_ref,
                                                     cloud_quantile=cloud_quantile,
                                                     factor_cloud=cond_below_cloud)
            cloudiness_ref_average=cloudiness_ref.mean(axis=1)
        else:
            cloudiness_ref=None
            cloudiness_ref_average=None
        
        cloudiness_syn = self.__solar_cloudiness(solar_syn,
                                                 cloud_quantile=cloud_quantile,
                                                 factor_cloud=cond_below_cloud)

        self.plot_barcharts(cloudiness_ref_average, cloudiness_syn.mean(axis=1), save_plots=True,
                            path_name=os.path.join(self.image_repo,'solar_kpi','cloudiness.png'),
                            title_component='Cloudiness per month (number of daily quantile '+str(cloud_quantile)+' below '+str(round(cond_below_cloud*100))+
                                            ' % of monthly quantile '+str(cloud_quantile)+')', normalized = False)


        ## Fourth KPI: Correlation between ref and syn (agregates)
        if not self.kpi_on_syn_data_only:
            correl = round(agg_ref_solar.corr(agg_syn_solar),self.precision)

        # Write output
        # -- + -- + --
        if self.kpi_on_syn_data_only:
            self.output['solar_kpi'] = {'solar_corr': syn_corr_solar.to_dict()}
            self.output['solar_kpi']['cloudiness_synthetic'] = cloudiness_syn.to_dict()
            self.output['solar_kpi']['season_solar_at_night_synthetic'] = solar_night_syn_mean.to_dict()
        else:
            self.output['solar_kpi'] = {'solar_corr': syn_corr_solar.to_dict()}
            self.output['solar_kpi']['cloudiness_reference'] = cloudiness_ref.to_dict()
            self.output['solar_kpi']['cloudiness_synthetic'] = cloudiness_syn.to_dict()
            self.output['solar_kpi']['season_solar_at_night_reference'] = solar_night_ref_mean.to_dict()
            self.output['solar_kpi']['season_solar_at_night_synthetic'] = solar_night_syn_mean.to_dict()
            self.output['solar_kpi']['correlation_ref_syn'] = correl

        return syn_corr_solar, solar_night_ref, solar_night_syn, cloudiness_ref, cloudiness_syn


    def wind_load_kpi(self, save_plots = True):

        '''
        Compute KPIs about wind production and load correlation
        '''

        # Get the solar gen names
        wind_filter = self.prod_charac.type.isin(['wind'])

        corr_rel = []
        corr_rel_ref = []

        for region in self.regions:

            # Create zone filter for wind and loads
            wind_region_filter = self.prod_charac.zone.isin([region])
            wind_names = self.prod_charac.loc[wind_filter & wind_region_filter]['name']
            loads_names_filter = self.load_charac[self.load_charac.zone == region]['name']

            ## Dispatch
            # Extract only wind units per region
            wind_gens_in_region = self.syn_dispatch[wind_names]
            loads_in_region = self.syn_consumption[loads_names_filter]

            # Compute correlation matrix
            tmp_corr = self._pairwise_corr_different_dfs(wind_gens_in_region, loads_in_region)
            corr_rel.append(tmp_corr)

            ## Reference
            # Extract only wind units per region
            if not self.kpi_on_syn_data_only:
                wind_gens_in_region = self.ref_dispatch[wind_names]
                loads_in_region = self.ref_consumption[loads_names_filter]

                # Compute correlation matrix
                tmp_corr = self._pairwise_corr_different_dfs(wind_gens_in_region, loads_in_region)
                corr_rel_ref.append(tmp_corr)

        # Plot results
        # Correlation heatmaps
        plt.ioff()  # to not launch windows with plot in os
        for i, region in enumerate(self.regions):
            if self.kpi_on_syn_data_only:
                fig, axes = plt.subplots(1, 1, figsize=(18, 8))
                sns.heatmap(corr_rel[i], annot=True, linewidths=.5, fmt='.1g',
                            cmap=sns.diverging_palette(20, 220, n=200),vmin=-1, vmax=1, center=0)
                axes.set_xticks([])
            else:
                fig, axes = plt.subplots(2, 1, figsize=(18, 8))
                sns.heatmap(corr_rel_ref[i], annot=True, linewidths=.5, ax=axes[0], fmt='.1g',
                            cmap=sns.diverging_palette(20, 220, n=200),vmin=-1, vmax=1, center=0)
                axes[0].set_xticks([])
                sns.heatmap(corr_rel[i], annot=True, linewidths=.5, ax=axes[1], fmt='.1g',
                            cmap=sns.diverging_palette(20, 220, n=200),vmin=-1, vmax=1, center=0)
            if save_plots:
                fig.savefig(os.path.join(self.image_repo, 'wind_load_kpi', 'corr_wind_load_'+region+'.png'))


        return corr_rel

    def nuclear_kpi(self, save_plots=True):

        """
        Compute nuclear production KPIs
        """

        # Get the nuclear gen names
        nuclear_filter = self.prod_charac.type.isin(['nuclear'])
        nuclear_names = self.prod_charac.name.loc[nuclear_filter].values

        # Extract only nuclear power plants
        if not self.kpi_on_syn_data_only:
            nuclear_ref = self.ref_dispatch[nuclear_names]
            agg_nuclear_ref = nuclear_ref.sum(axis=1)

        nuclear_syn = self.syn_dispatch[nuclear_names]
        agg_nuclear_syn = nuclear_syn.sum(axis = 1)


        # Distribution of prod
        plt.ioff()  # to not launch windows with plot in os
        if self.kpi_on_syn_data_only:
            fig, axes = plt.subplots(1, 1, figsize=(17, 5))
            axes.hist(agg_nuclear_syn, bins=100, alpha=1)
            axes.set_title('Synthetic Distribution of agregate nuclear production')
        else:
            fig, axes = plt.subplots(1, 2, figsize=(17, 5))
            axes[0].hist(agg_nuclear_ref, bins = 100, alpha = 1)
            axes[1].hist(agg_nuclear_syn, bins = 100, alpha = 1)
            axes[0].set_title('Reference Distribution of agregate nuclear production')
            axes[1].set_title('Synthetic Distribution of agregate nuclear production')
        if save_plots:
            # Save plot as png
            fig.savefig(os.path.join(self.image_repo, 'nuclear_kpi', 'production_distribution.png'))

        ## Nuclear lag distribution
        if not self.kpi_on_syn_data_only:
            nuclear_lag_ref = agg_nuclear_ref.diff()
        nuclear_lag_syn = agg_nuclear_syn.diff()

        if self.kpi_on_syn_data_only:
            fig, axes = plt.subplots(1, 1, figsize=(17, 5))
            axes.hist(nuclear_lag_syn.values, bins=100, alpha=1)
            axes.set_title('Synthetic Distribution of agregate nuclear ramps in production')
        else:
            fig, axes = plt.subplots(1, 2, figsize=(17, 5))
            axes[0].hist(nuclear_lag_ref.values, bins=100, alpha=1)
            axes[1].hist(nuclear_lag_syn.values, bins=100, alpha=1)
            axes[0].set_title('Reference Distribution of agregate nuclear ramps in production')
            axes[1].set_title('Synthetic Distribution of agregate nuclear ramps in production')
        if save_plots:
            # Save plot as png
            fig.savefig(os.path.join(self.image_repo, 'nuclear_kpi', 'lag_distribution.png'))


        ## Monthly maintenance percentage time
        if not self.kpi_on_syn_data_only:
            maintenance_ref = agg_nuclear_ref.resample('1M').agg(lambda x: x[x==0.].count()/x.count())
            maintenance_ref.index = maintenance_ref.index.month.set_names('Month')
        else:
            maintenance_ref=None

        maintenance_syn = agg_nuclear_syn.resample('1M').agg(lambda x: x[x==0.].count()/x.count())
        maintenance_syn.index = maintenance_syn.index.month.set_names('Month')


        self.plot_barcharts(maintenance_ref, maintenance_syn, save_plots=save_plots,
                            path_name=os.path.join(self.image_repo, 'nuclear_kpi', 'maintenance_percentage_of_time_per_month.png'),
                            title_component='% of time in maintenance per month', normalized = False)

        return None

    def thermal_kpi(self,
                  upper_quantile=0.9,
                  lower_quantile=0.1,
                  above_norm_cap=0.8,
                  below_norm_cap=0.2):

        '''
        Compute thermal production KPIs
        '''

        # Get Hydro names
        thermal_filter = self.prod_charac.type.isin(['thermal'])
        thermal_names = self.prod_charac.name.loc[thermal_filter].values

        # Normalize MW according to the max value for
        # the reference data and synthetic one
        if not self.kpi_on_syn_data_only:
            thermal_ref = self.ref_dispatch[thermal_names]
        thermal_syn = self.syn_dispatch[thermal_names]

        if not self.kpi_on_syn_data_only:
            max_mw_ref = thermal_ref.max(axis=0)
        max_mw_syn = thermal_syn.max(axis=0)

        if not self.kpi_on_syn_data_only:
            norm_mw_ref = thermal_ref / max_mw_ref
        norm_mw_syn = thermal_syn / max_mw_syn

        # Stats for reference data
        if not self.kpi_on_syn_data_only:
            if self.quantile_mode == 'prices':
                for_quantiles = self.ref_prices
            else:
                for_quantiles = pd.DataFrame({'conso':self.ref_agg_conso})
            stat_ref_high_price, stat_ref_low_price = self.__hydro_in_prices(norm_mw_ref,
                                                                             for_quantiles,
                                                                             upper_quantile,
                                                                             lower_quantile,
                                                                             above_norm_cap,
                                                                             below_norm_cap
                                                                             )
        else:
            stat_ref_high_price=None
            stat_ref_low_price=None

        # Stats for synthetic data
        if self.quantile_mode == 'prices':
            for_quantiles = self.syn_prices
        else:
            for_quantiles = pd.DataFrame({'conso':self.syn_agg_conso})
        stat_syn_high_price, stat_syn_low_price = self.__hydro_in_prices(norm_mw_syn,
                                                                         for_quantiles,
                                                                         upper_quantile,
                                                                         lower_quantile,
                                                                         above_norm_cap,
                                                                         below_norm_cap
                                                                         )

        self.plot_barcharts(stat_ref_high_price, stat_syn_high_price, save_plots=True,
                            path_name=os.path.join(self.image_repo, 'thermal_kpi', 'high_price.png'),
                            title_component='% of time production exceed ' + str(above_norm_cap) +
                                            '*Pmax when prices are high (above quantile ' + str(
                                upper_quantile * 100) + ')', normalized = False, every_nth = 5)

        self.plot_barcharts(stat_ref_low_price, stat_syn_low_price, save_plots=True,
                                path_name=os.path.join(self.image_repo, 'thermal_kpi', 'low_price.png'),
                                title_component='% of time production is below ' + str(below_norm_cap) +
                                                '*Pmax when prices are low (under quantile ' + str(
                                    lower_quantile * 100) + ')', normalized = False, every_nth = 5)


        # Seasonal for reference data
        if not self.kpi_on_syn_data_only:
            ref_agg_mw_per_month = self.__hydro_seasonal(thermal_ref)
            ref_agg_mw_per_month_sum=ref_agg_mw_per_month.sum(axis=1)
        else:
            ref_agg_mw_per_month=None
            ref_agg_mw_per_month_sum=None

        # Seasonal for synthetic data
        syn_agg_mw_per_month = self.__hydro_seasonal(thermal_syn)

        self.plot_barcharts(ref_agg_mw_per_month_sum, syn_agg_mw_per_month.sum(axis=1), save_plots=True,
                            path_name=os.path.join(self.image_repo, 'thermal_kpi', 'thermal_per_month.png'),
                            title_component='Thermal mean production (% of max) per month for all units')

        ## Load Correlation of reference dispatch
        if not self.kpi_on_syn_data_only:
            agg_thermal_ref = thermal_ref.sum(axis=1)
            correl_ref = round(agg_thermal_ref.corr(self.ref_agg_conso), self.precision)

        ## Load Correlation of synthetic dispatch
        agg_thermal_syn = thermal_syn.sum(axis = 1)
        correl_syn = round(agg_thermal_syn.corr(self.syn_agg_conso), self.precision)


        # Write results
        # -- + -- + --

        if self.kpi_on_syn_data_only:
            self.output['thermal_kpi'] = {'high_price_for_syn': stat_syn_high_price.to_dict(),
                                          'low_price_for_syn': stat_syn_low_price.to_dict(),
                                          'syn_load_correlation': correl_syn,
                                          'seasonal_month_for_syn': syn_agg_mw_per_month.to_dict()
                                          }
        else:
            self.output['thermal_kpi'] = {'high_price_for_ref': stat_ref_high_price.to_dict(),
                                          'low_price_for_ref': stat_ref_low_price.to_dict(),
                                          'high_price_for_syn': stat_syn_high_price.to_dict(),
                                          'low_price_for_syn': stat_syn_low_price.to_dict(),
                                          'ref_load_correlation': correl_ref,
                                          'syn_load_correlation': correl_syn,
                                          'seasonal_month_for_ref': ref_agg_mw_per_month.to_dict(),
                                          'seasonal_month_for_syn': syn_agg_mw_per_month.to_dict()
                                          }

        return stat_ref_high_price, stat_ref_low_price, ref_agg_mw_per_month, \
               stat_syn_high_price, stat_syn_low_price, syn_agg_mw_per_month

    def thermal_load_kpi(self, save_plots = True):

        '''
        Compute KPIs about thermal production and load correlation
        '''

        # Get the solar gen names
        thermal_filter = self.prod_charac.type.isin(['thermal'])

        corr_rel = []
        corr_rel_ref = []
        for region in self.regions:

            # Create zone filter for wind and loads
            thermal_region_filter = self.prod_charac.zone.isin([region])
            thermal_names = self.prod_charac.loc[thermal_filter & thermal_region_filter]['name']
            loads_names_filter = self.load_charac[self.load_charac.zone == region]['name']

            ## Dispatch
            # Extract only wind units per region
            thermal_gens_in_region = self.syn_dispatch[thermal_names]
            loads_in_region = self.syn_consumption[loads_names_filter]

            # Compute correlation matrix
            tmp_corr = self._pairwise_corr_different_dfs(thermal_gens_in_region, loads_in_region)
            corr_rel.append(tmp_corr)

            ## Reference
            # Extract only wind units per region
            if not self.kpi_on_syn_data_only:
                thermal_gens_in_region = self.ref_dispatch[thermal_names]
                loads_in_region = self.ref_consumption[loads_names_filter]

                # Compute correlation matrix
                tmp_corr = self._pairwise_corr_different_dfs(thermal_gens_in_region, loads_in_region)
                corr_rel_ref.append(tmp_corr)

        # Plot results
        # Correlation heatmaps
        plt.ioff()  # to not launch windows with plot in os
        for i, region in enumerate(self.regions):
            if self.kpi_on_syn_data_only:
                fig, axes = plt.subplots(1, 1, figsize=(18, 8))
                sns.heatmap(corr_rel[i], annot=True, linewidths=.5, fmt='.1g',
                            cmap=sns.diverging_palette(20, 220, n=200),vmin=-1, vmax=1, center=0)
                axes.set_xticks([])
            else:
                fig, axes = plt.subplots(2, 1, figsize=(18, 8))
                sns.heatmap(corr_rel_ref[i], annot=True, linewidths=.5, ax=axes[0], fmt='.1g',
                            cmap=sns.diverging_palette(20, 220, n=200),vmin=-1, vmax=1, center=0)
                axes[0].set_xticks([])
                sns.heatmap(corr_rel[i], annot=True, linewidths=.5, ax=axes[1], fmt='.1g',
                            cmap=sns.diverging_palette(20, 220, n=200),vmin=-1, vmax=1, center=0)
            if save_plots:
                fig.savefig(os.path.join(self.image_repo, 'thermal_load_kpi', 'corr_thermal_load_'+region+'.png'))

        return corr_rel


    def load_kpi(self, save_plots = True):
        """
        Compute KPIs about load chronics
        """

        ## Synthetic
        # Normalized conso by day of week
        conso_day = self.syn_agg_conso.copy()
        conso_day.index = [date.weekday() for date in conso_day.index]
        conso_day = conso_day.groupby(conso_day.index).sum()
        conso_day = conso_day/conso_day.max()
        conso_day.index = conso_day.index.set_names('Day_of_week')

        # Normalized conso by week of year
        conso_week = self.syn_agg_conso.copy()
        conso_week.index = [date.week for date in conso_week.index]
        conso_week = conso_week.groupby(conso_week.index).sum()
        conso_week = conso_week/conso_week.max()
        conso_week.index = conso_week.index.set_names('Week_of_year')

        ## Reference
        # Normalized conso by day of week
        if not self.kpi_on_syn_data_only:
            conso_day_ref = self.ref_agg_conso.copy()
            conso_day_ref.index = [date.weekday() for date in conso_day_ref.index]
            conso_day_ref = conso_day_ref.groupby(conso_day_ref.index).sum()
            conso_day_ref = conso_day_ref/conso_day_ref.max()
            conso_day_ref.index = conso_day_ref.index.set_names('Day_of_week')

            # Normalized conso by week of year
            conso_week_ref = self.ref_agg_conso.copy()
            conso_week_ref.index = [date.week for date in conso_week_ref.index]
            conso_week_ref = conso_week_ref.groupby(conso_week_ref.index).sum()
            conso_week_ref = conso_week_ref/conso_week_ref.max()
            conso_week_ref.index = conso_week_ref.index.set_names('Week_of_year')

        ## Plot results
        # By day
        plt.ioff()  # to not launch windows with plot in os
        if self.kpi_on_syn_data_only:
            fig, axes = plt.subplots(1, 1, figsize=(17, 5))
            sns.barplot(conso_day.index, conso_day)
            axes.set_title('Synthetic - Normalized load per day of week', size=9)
        else:
            fig, axes = plt.subplots(1, 2, figsize=(17, 5))
            sns.barplot(conso_day_ref.index, conso_day_ref, ax=axes[0])
            sns.barplot(conso_day.index, conso_day, ax=axes[1])
            axes[0].set_title('Reference - Normalized load per day of week', size=9)
            axes[1].set_title('Synthetic - Normalized load per day of week', size=9)

        if save_plots:
            fig.savefig(os.path.join(self.image_repo,'load_kpi','load_by_day_of_week.png'))

        # By week of year
        every_nth = 3
        if self.kpi_on_syn_data_only:
            fig, axes = plt.subplots(1, 1, figsize=(17, 5))
            sns.barplot(conso_week.index, conso_week)
            for n, label in enumerate(axes.xaxis.get_ticklabels()):
                if n % every_nth != 0:
                    label.set_visible(False)
            axes.set_title('Normalized load per week of year', size=9)
            for n, label in enumerate(axes.xaxis.get_ticklabels()):
                if n % every_nth != 0:
                    label.set_visible(False)
        else:
            fig, axes = plt.subplots(1, 2, figsize=(17, 5))
            sns.barplot(conso_week_ref.index, conso_week_ref, ax=axes[0])
            sns.barplot(conso_week.index, conso_week, ax=axes[1])
            axes[0].set_title('Normalized load per week of year', size=9)
            for n, label in enumerate(axes[0].xaxis.get_ticklabels()):
                if n % every_nth != 0:
                    label.set_visible(False)
            axes[1].set_title('Normalized load per week of year', size=9)
            for n, label in enumerate(axes[1].xaxis.get_ticklabels()):
                if n % every_nth != 0:
                    label.set_visible(False)
        if save_plots:
            fig.savefig(os.path.join(self.image_repo, 'load_kpi', 'load_by_week_of_year.png'))


        return