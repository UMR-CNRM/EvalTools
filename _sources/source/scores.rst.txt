.. scores:

Available scores
================

Let :math:`y = (y_i)_{0 \leqslant i \leqslant n}` be a vector of modelled values
and :math:`x = (x_i)_{0 \leqslant i \leqslant n}` the corresponding vector of
observed values (where couple :math:`(x_j, y_j)` have been removed if :math:`y_j`
and/or :math:`x_j` is missing).

In `evaltools`, many functions take one or several scores as argument.
The names of the scores to provide to these functions are not case
sensitive and can be chosen among:

* ``'bias'`` or ``'meanbias'``: Mean Bias
    :math:`\bar{y} - \bar{x}`
* ``'BiasPct'`` or ``relative_mean_bias``: Relative mean bias
    :math:`100 \times \frac{\bar{y} - \bar{x}}{\bar{x}}`
* ``bias_std``:
    Standard deviation of bias.
* ``'correlation'`` or ``'PearsonR'``: Pearson correlation
    :math:`\frac{cov(x,y)}{\sigma_x \sigma_y} =
    \frac{\sum_i (x_i-\bar{x})(y_i-\bar{y})}
    {\sqrt{\sum_i(x_i-\bar{x})^2}\sqrt{\sum_i(y_i-\bar{y})^2}}`
* ``'crmse'``: Centered root mean square error
    :math:`\sqrt{\frac{1}{n}\sum_i((y_i-\bar{y})-(x_i-\bar{x}))^2}`
* ``'FactOf2'``: Factor of two
    Number of cases where :math:`x_i/2 \leqslant y_i \leqslant x_i*2` devided
    by :math:`n`.
* ``'FGE'``: Fractional gross error
    :math:`\frac{2}{n}\sum_i|{\frac{y_i-x_i}{y_i+x_i}}|`
* ``'FracBias'``: Fractional bias
    :math:`100 \times \frac{\bar{y} - \bar{x}}{\frac{1}{2}( \bar{x}+\bar{y})}`
* ``kendalltau`` or ``kendall_correlation``: Kendall rank correlation coefficient
    From Scipy's documentation:

        Kendall’s tau is a measure of the
        correspondence between two rankings.
        Values close to 1 indicate strong agreement, and values close to -1
        indicate strong disagreement.
        ``tau = (P - Q) / sqrt((P + Q + T) * (P + Q + U))``
        where P is the number of concordant pairs, Q the number of
        discordant pairs, T the number of ties only in x, and U the number
        of ties only in y. If a tie occurs for the same pair in both x and
        y, it is not added to either T or U. 

* ``'MAE'``: Mean absolute error
    :math:`\frac{1}{n}\sum_i|y_i-x_i|`
* ``'MMB'`` or ``'MNMB'``: Modified mean bias
    :math:`\frac{2}{n}\sum_i\frac{y_i-x_i}{y_i+x_i}`
* ``'NbObs'``:
    Number of paired modelled and observed values available.
* ``'NMSE'``: Normalised mean square error
    :math:`\frac{\frac{1}{n}\sum_i(y_i-x_i)^2}{\bar{y} \times \bar{x}}`
* ``'NRMSE'``: Normalised root mean square error
    :math:`\frac{RMSE}{\bar{x}}`
* ``'obs_mean'``: Mean of observed values
    :math:`\bar{x} = \frac{1}{n} \sum_i x_i`
* ``'obs_percentile'``:
    Pth percentile of modelled values
* ``'obs_median'``:
    Median of modelled values
* ``'obs_std'``:
    Standard deviation of observed values
* ``'RMSE'``: Root mean square error
    :math:`\sqrt{\frac{1}{n}\sum_i(y_i-x_i)^2}`
* ``'sim_mean'``: Mean of modelled values
    :math:`\bar{y} = \frac{1}{n} \sum_i y_i`
* ``'sim_median'``:
    Median of observed values
* ``'sim_percentile'``:
    Pth percentile of observed values
* ``'sim_std'``:
    Standard deviation of modelled values
* ``'SpearmanR'``:
    Spearman correlation
* ``'std_ratio'``: standard deviation ratio
    :math:`\sqrt{\frac{\sum_i(y_i-\bar{y})^2}{\sum_i(x_i-\bar{x})^2}}`
* ``'success_rate_t_u'``:
    Success rate with tolerance where t is the tolerance threshold and u is the
    utility threshold (ex: success_rate_1_2). It is the rate of values where
    the absolute difference between the observed value and the simulated
    one is lower than the tolerance threshold. If the absolute difference
    between the observed value and the simulated one lays between the tolerance
    threshold and the utility threshold, the success score of the simulated
    value is defined with a linear function equal to 1
    if abs(obs - sim) = tolerance_thr and equal to 0 if abs(obs - sim) =
    utility_thr.
* ``'variances_ratio'``: Variances ratio
    :math:`\frac{\sum_i(y_i-\bar{y})^2}{\sum_i(x_i-\bar{x})^2}`
* ``'Bias+t'``:
    Mean bias set to 0
    when the Mann–Whitney U test for the null hypothesis that observations and
    simulations have identical average values is not significant
    (pvalue > 0.05).
* ``'Ratio+t'``:
    Variances ratio set to 1
    when Levene's test for the null hypothesis that observations and
    simulations have identical variance is not significant
    (pvalue > 0.05).
* ``'PearsonR+t'``:
    Pearson correlation set to 0
    when the probability of an uncorrelated system producing datasets
    that have a Pearson correlation at least as extreme as the one
    computed from these datasets is greater than 0.05.
* ``'SpearmanR+t'``:
    Spearman correlation set to 0
    when the probability of an uncorrelated system producing datasets
    that have a Spearman correlation at least as extreme as the one
    computed from these datasets is greater than 0.05.
