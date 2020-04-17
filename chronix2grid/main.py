# Native python packages
import os

import click

# Chronix2grid modules
import generation.generate_chronics as gen
import kpi.main as kpis

# ==============================================================
## CONSTANT VARIABLES
INPUT_FOLDER = os.path.abspath('generation/input')
OUTPUT_FOLDER = os.path.abspath('generation/output')

KPI_INPUT_FOLDER = 'kpi/input'
IMAGES_FOLDER = "kpi/images'"


@click.command()
@click.option('--case', default='case118_l2rpn', help='case folder to base generation on')
@click.option('--start-date', default='2012-01-01', help='Start date to generate chronics')
@click.option('--weeks', default=4, help='Number of weeks to generate')
@click.option('--n_scenarios', default=1, help='Number of scenarios to generate')
@click.option('--mode', default='LRTK', help='Steps to execute : L for loads only (and KPI); R(K) for renewables (and KPI) only; LRTK for all generation')
def generate(case, start_date, weeks, n_scenarios, mode):
    # Folders are specific to studied case
    output_folder = os.path.join(OUTPUT_FOLDER, case)
    images_folder = os.path.join(IMAGES_FOLDER, case)

    # Reading parameters
    year, params, loads_charac, prods_charac, load_weekly_pattern, solar_pattern, lines, params_opf = gen.read_configuration(
        INPUT_FOLDER, case, start_date, weeks)

    # Chronic generation
    if 'L' in mode or 'R' in mode:
        gen.main(case, year, n_scenarios, params, INPUT_FOLDER, output_folder, prods_charac, loads_charac, lines, solar_pattern, load_weekly_pattern, params_opf, mode)

    # KPI formatting and computing
    if 'R' in mode and 'K' in mode and 'T' not in mode:
        # Get and format solar and wind on all timescale, then compute KPI and save plots
        wind_solar_only = True
        os.makedirs(IMAGES_FOLDER, exist_ok=True)
        kpis.main(KPI_INPUT_FOLDER, INPUT_FOLDER, output_folder, images_folder, year, case, n_scenarios, wind_solar_only, params, loads_charac, prods_charac)

    elif 'T' in mode and 'K' in mode:
        # Get and format dispatched chronics, then compute KPI and save plots
        wind_solar_only = False
        os.makedirs(IMAGES_FOLDER, exist_ok=True)
        kpis.main(KPI_INPUT_FOLDER, INPUT_FOLDER, output_folder, images_folder, year, case, n_scenarios, wind_solar_only, params, loads_charac, prods_charac)


if __name__ == "main":
    generate()
