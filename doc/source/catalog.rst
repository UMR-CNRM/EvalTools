Chart catalog
=============

Time series
-----------

Observation and simulation time series can be drawn with
:func:`plot_time_series<evaltools.plotting.plotting.plot_time_series>`
function.

.. image:: charts/time_series.png

Average of a score as a function of the forecast time
-----------------------------------------------------

:func:`plot_average_ft_scores<evaltools.plotting.plotting.plot_average_ft_scores>`

The score can be computed spatially (for each time) or temporally (for each
station) and then average (mean or median) to get a value for each forecast
time.

.. image:: charts/average_ft_scores.png

Median scores for each year quarter
-----------------------------------

Median of scores computed every quarters can be drawn combining
:func:`quarterlyMedianScore<evaltools.evaluator.Evaluator.quarterly_median_score>`
method and
:func:`plot_quarterly_score<evaltools.plotting.plotting.plot_quarterly_score>`
function.

.. image:: charts/quarterlyMedianScore.png

Time scores
-----------

Scores at a given term can be drawn with
:func:`plot_time_scores<evaltools.plotting.plotting.plot_time_scores>` function.
For hourly series, the term is a interger corresponding to a forecast hour,
whereas for daily series, it corresponds to a forecast day.

.. image:: charts/time_scores.png

Station scores
--------------

Scores per station for a choosen forecast day can be displayed on a map with
:func:`plot_station_scores<evaltools.plotting.plotting.plot_station_scores>`
function.

.. image:: charts/station_scores.png

Taylor diagram
--------------

A taylor diagram for the first forecast day can be displayed with
:func:`plot_taylor_diagram<evaltools.plotting.plotting.plot_taylor_diagram>`
function.

.. image:: charts/taylor_diagram.png

Scatter plot of score quartiles
-------------------------------

The function
:func:`plot_score_quartiles<evaltools.plotting.plotting.plot_score_quartiles>`
draws a scatter plot with axis corresponding to the median of two choosen
scores. Each point on the scatter plot represents one model and the rectangle
around the point is drawn from the quartiles of the two scores.

.. image:: charts/score_quartiles.png

Scatter plot of station scores for two models
---------------------------------------------

The function
:func:`plot_comparison_scatter_plot<evaltools.plotting.plotting.plot_comparison_scatter_plot>`
displays the score calculated at every stations for two different Evaluator
objects.

.. image:: charts/comparison_scatter_plot.png

Significant differences
-----------------------

The function
:func:`plot_significant_differences<evaltools.plotting.plotting.plot_significant_differences>`
tests the significativity of differences between the distribution of station
scores of two Evaluator objects. Several scores an several couple of
Evaluator objects can be tested.

.. image:: charts/significant_differences.png

Diurnal cycle per station
-------------------------

The function
:func:`plot_diurnal_cycle<evaltools.plotting.plotting.plot_diurnal_cycle>`
displays the median of observed and simulated concentration values
at each station as a function of the forecast time.

.. image:: charts/diurnal_cycle.png

Probability density of raw data
-------------------------------

The function
:func:`plot_data_density<evaltools.plotting.plotting.plot_data_density>`
displays the probability density of observation and simulation values.

.. image:: charts/data_density.png

Probability density of a score
------------------------------

The function
:func:`plot_score_density<evaltools.plotting.plotting.plot_score_density>`
displays the probability density of score values.

.. image:: charts/score_density.png

Boxplot of data values
----------------------

The function :func:`plot_data_box<evaltools.plotting.plotting.plot_data_box>`
displays the distribution of observation and simulation values.

.. image:: charts/data_box.png

Boxplot of score values
-----------------------

The function
:func:`plot_score_box<evaltools.plotting.plotting.plot_score_box>`
displays the distribution of score values.

.. image:: charts/score_box.png

FAIRMODE diagrams
-----------------

The functions
:func:`plot_target_diagram<evaltools.fairmode.plot_target_diagram>`,
:func:`plot_fairmode_summary<evaltools.fairmode.plot_fairmode_summary>`
:func:`plot_forecast_target_diagram<evaltools.fairmode.plot_forecast_target_diagram>`
:func:`plot_forecast_target_diagram<evaltools.fairmode.plot_yearly_fairmode_summary>`
and
:func:`plot_forecast_target_diagram<evaltools.fairmode.plot_scatter_diagram>`
displays five of the Fairmode diagrams: the assessment target plot,
the summary report, the forecast target plot, the summary report for
yearly average data and the scatter diagram for yearly average data.
A description of these diagrams can be found
on `FAIRMODE website <https://fairmode.jrc.ec.europa.eu/Guidance/Assessment/>`_.

.. image:: charts/target.png

.. image:: charts/summary.png

.. image:: charts/forecast_target.png

.. image:: charts/yearly_fairmode_summary.png

.. image:: charts/scatter_diagram.png

Scores as a barplot
-------------------

The function
:func:`plot_bar_scores<evaltools.plotting.plotting.plot_bar_scores>`
displays the barplots of the average of scores computed per station. It is
possible to compute different bars for different country sets.

.. image:: charts/bar_scores.png

Threshold exceedances
---------------------

The function
:func:`plot_bar_exceedances<evaltools.plotting.plotting.plot_bar_exceedances>`
displays the daily number of exceedances.
It is possible to select different country sets.

.. image:: charts/bar_exceedances.png

The function
:func:`plot_bar_contingency_table<evaltools.plotting.plotting.plot_bar_contingency_table>`
displays, as a barplot, the contingency table regarding the number of
exceedances of a threshold.

.. image:: charts/bar_contingency_table.png

The function
:func:`plot_line_exceedances<evaltools.plotting.plotting.plot_line_exceedances>`
displays the daily number of exceedances.

.. image:: charts/line_exceedances.png

Observations vs simulations scatter plot
----------------------------------------

Observation and simulation values can be compared with
:func:`plot_values_scatter_plot<evaltools.plotting.plotting.plot_values_scatter_plot>`
function.

.. image:: charts/values_scatter_plot.png

Performance diagram
-------------------

This function
:func:`plot_performance_diagram<evaltools.plotting.plotting.plot_performance_diagram>`
draws the performance diagram by P.J. Roebber ("Visualizing multiple measures
of forecast quality", 2009).

.. image:: charts/performance_diagram.png

ROC curves
----------

This function
:func:`plot_roc_curve<evaltools.plotting.plotting.plot_roc_curve>`
draws the ROC curve relative to the choosen thresholds for each object.

.. image:: charts/roc_curve.png

Scores computed from a concentration limit value
------------------------------------------------

The function
:func:`plot_exceedances_scores<evaltools.plotting.plotting.plot_exceedances_scores>`
draws a barplot displaying:

- Accuracy
- Bias score
- Success ratio
- probability of detection (Hit rate)
- false alarm ratio
- probability of false detection = false alarm rate
- Threat Score
- Equitable Threat Score
- Peirce Skill Score (Hanssen and Kuipers discriminant)
- Heidke Skill Score
- Rousseau Skill Score
- Odds Ratio
- Odds Ratio Skill Score

More information on these scores can be found here
https://www.cawcr.gov.au/projects/verification/.

.. image:: charts/exceedances_scores.png


Summary bar chart
-----------------

The function
:func:`plot_summary_bar_chart<evaltools.plotting.plotting.plot_summary_bar_chart>`
plots a bar for RMSE and lollipops for bias and correlation.

.. image:: charts/summary_bar_chart.png


Bar chart of scores computed for different concentration classes
----------------------------------------------------------------

The function
:func:`plot_bar_scores_conc<evaltools.plotting.plotting.plot_bar_scores_conc>`
plots scores computed for each required concentration range (scores are
computed for each site and averaged).

.. image:: charts/bar_scores_conc.png
