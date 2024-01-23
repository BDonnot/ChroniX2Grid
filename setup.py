# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)

import os
from setuptools import setup, find_packages

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
    
    
setup(name='Chronix2Grid',
      version='1.2.1',
      description='A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)',
      long_description=long_description,
      long_description_content_type='text/markdown',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Programming Language :: Python :: 3.11',
          "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
          "Intended Audience :: Developers",
          "Intended Audience :: Education",
          "Intended Audience :: Science/Research",
          "Natural Language :: English"
      ],
      keywords='ML powergrid optmization RL power-systems chronics generation production load network',
      author='Mario Jothy, Nicolas Megel, Vincent Renault, Benjamin Donnot',
      author_email='mario.jothy@artelys.com',
      url="https://github.com/BDonnot/ChroniX2Grid",
      license='Mozilla Public License 2.0 (MPL 2.0)',
      packages=find_packages(),
      include_package_data=True,
      platforms=["Windows", "Linux", "Mac OS-X", "Unix"],
      install_requires=[
                        "click>=7.1.1",
                        "cufflinks>=0.17.3",
                        "decorator>=4.4.2",
                        "folium>=0.10.1",
                        "Grid2Op>=1.5.2",
                        "ipython-genutils>=0.2.0",
                        "ipywidgets>=7.5.1",
                        "json5>=0.9.3",
                        "jupyter>=1.0.0",
                        "matplotlib>=3.1.3",
                        "numpy>=1.18.3",
                        "pandas>=1.0.5",
                        "pandocfilters>=1.4.2",
                        "plotly>=4.5.2",
                        "Pyomo>=5.6.8",
                        "pytest>=6.2.2",
                        "pytest-tornasync>=0.6.0.post2",
                        "python-dateutil>=2.8.1",
                        "pytz>=2019.3",
                        "PyUtilib>=5.7.3",
                        "requests>=2.23.0",
                        "seaborn>=0.10.0",
                        "scipy>=1.4.1",
                        "widgetsnbextension>=3.5.1",
                        "lightsim2grid",
                        "pypsa",
                        "cvxpy",
                        "scikit-learn"
                        ],
      extras_require = {
                        "optional": [
                            "ligthsim2grid"
                        ],
                        "docs": [
                            "numpydoc>=0.9.2",
                            "sphinx>=2.4.4",
                            "sphinx-rtd-theme>=0.4.3",
                            "sphinxcontrib-trio>=1.1.0",
                            "autodocsumm>=0.1.13",
                            "pypsa>=0.17.0"
                        ]
                    },
      zip_safe=False,
      package_data={'chronix2grid':['getting_started/example/input/generation/case118_l2rpn_neurips_1x/*',
                                    'getting_started/example/input/generation/patterns/*',
                                    'getting_started/example/input/kpi/case118_l2rpn_neurips_1x/paramsKPI.json',
                                    'getting_started/example/input/kpi/case118_l2rpn_neurips_1x/France/eco2mix/*.csv',
                                    'getting_started/example/input/kpi/case118_l2rpn_neurips_1x/France/renewable_ninja/*.csv']},
      entry_points={'console_scripts': ['chronix2grid=chronix2grid.main:generate_mp']}
)
