PROJECT_ROOT = "../."
RESULTS_ROOT = f'{PROJECT_ROOT}/'


import matplotlib.pyplot as plt
import os
from pathlib import Path


PLOTS_DIRECTORY = f'{PROJECT_ROOT}/paper_plots'
if not os.path.exists(PLOTS_DIRECTORY):
    os.makedirs(PLOTS_DIRECTORY)

APPENDIX_NAME = 'appendix'

if not os.path.exists(f'{PLOTS_DIRECTORY}/{APPENDIX_NAME}'):
    os.makedirs(f'{PLOTS_DIRECTORY}/{APPENDIX_NAME}')

def save_plt_figure_result(path, paper = False, appendix = False):
    dir = os.path.dirname(path)
    if not os.path.exists(PLOTS_DIRECTORY):
        os.makedirs(dir)
    # plt.savefig(f'{path}.jpg')
    if paper:
        plt.savefig(f'{PLOTS_DIRECTORY}/{os.path.basename(path)}.pdf', dpi=600)
    if appendix:
        plt.savefig(f'{PLOTS_DIRECTORY}/{APPENDIX_NAME}/{os.path.basename(path)}.pdf', dpi=600)
