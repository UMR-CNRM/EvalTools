# Copyright (c) Météo France (2017-)
# This software is governed by the CeCILL-C license under French law.
# http://www.cecill.info
"""This module gathers plotting annotations in several languages."""

mean_time_scores = {
    'xlabel': {
        'ENG': 'Forecast time (hour)',
        'FR': "Heure d'échéance",
    },
}

station_scores = {
    'annotation': {
        'ENG': ('{av} processed stations over {total}\n' +
                'min: {mini}, avg: {avg}, max: {maxi}'),
        'FR': ('{av} stations traitées sur {total}\n' +
               'min: {mini}, moy: {avg}, max: {maxi}'),
    },
}

bar_exceedances = {
    'ylabel': {
        'ENG': "Number of incidences",
        'FR': "Nombre de dépassements",
    },
}

line_exceedances = {
    'ylabel': {
        'ENG': "Number of incidences",
        'FR': "Nombre de dépassements",
    },
}

bar_contingency_table = {
    'index': {
        'ENG': ['good detections', 'false alarms', 'missed alarms'],
        'FR': ['Bonnes détections', 'Fausses détections',
               'Détections manquées'],
    },
}

performance_diagram = {
    'xlabel': {
        'ENG': 'Success Ratio',
        'FR': 'Taux de succès',
    },
    'ylabel': {
        'ENG': 'Probability of Detection',
        'FR': 'Probabilité de détection',
    },
    'csi_label': {
        'ENG': 'Critical Success Index',
        'FR': 'CSI',
    },
    'freq_label': {
        'ENG': 'Frequency Bias',
        'FR': 'Biais',
    },
}

diurnal_cycle = {
    'xlabel': {
        'ENG': 'Forecast time (hour)',
        'FR': "Heure d'échéance",
    },
}

taylor_diagram = {
    'sdr': {
        'ENG': r"Standard deviation ratio ($\sigma_{mod} / \sigma_{obs}$)",
        'FR': r"quotient des écarts types ($\sigma_{mod} / \sigma_{obs}$)",
    },
    'sd': {
        'ENG': "Standard deviation",
        'FR': "Ecart type",
    },
    'corr': {
        'ENG': "correlation",
        'FR': "corrélation",
    },
}

station_score_density = {
    'ylabel': {
        'ENG': "Density",
        'FR': "Densité",
    },
}

summary_bar_chart = {
    'mean_label': {
        'ENG': "Mean obs",
        'FR': "Moyenne des obs",
    },
    'ylabel_obs': {
        'ENG': r"RMSE/bias/mean obs ($\mu$g/m3)",
        'FR': r"RMSE/biais/obs ($\mu$g/m3)",
    },
    'ylabel': {
        'ENG': r"RMSE/bias ($\mu$g/m3)",
        'FR': r"RMSE/biais ($\mu$g/m3)",
    },
    'corr': {
        'ENG': "Correlation",
        'FR': "Corrélation",
    },
    'bias': {
        'ENG': "Bias",
        'FR': "Biais",
    },
}

bar_scores_conc = {
    'annotation': {
        'ENG': 'Proportion of values used for each class (over {}):',
        'FR': 'Proportion de valeurs utilisées pour chaque classe (sur {}) :',
    },
}
