# Preparation and forecasting methodology

Concatenate the authorized historical order files. Validate required fields and unique delivery IDs. Reject conflicting duplicates; remove only identical repeated order records. Count every order regardless of dispatch status. Join requested order dates to the supplied ISO calendar, then sum total order cubic metres by depot, brand and week. Sum chilled Fresh orders separately; non-Fresh chilled demand is zero.

Reject missing target values, unknown dates, negative source volume, calendar gaps, absent operating dates and incomplete depot-brand combinations. Retain only complete operating weeks. Dispatch fields do not become forecasting inputs. Blank festival names mean no festival. Do not silently fabricate targets or delete demand spikes.

Historical forecast-origin examples use only observations available at that origin for lags, rolling means, volatility, trend, outlet coverage and normalization. Known future calendar features include payday, operating days, monsoon and festival ramp. Direct models train horizon 1â€“10 examples with labels no later than each fold's cutoff. Recursive models update history using their predictions instead of future actuals.

Use expanding chronological validation to compare models and choose each target's predictor. Evaluate a bounded equal-weight ensemble alternative. Freeze model selection before evaluating the final ten-week holdout. Chilled selection metrics use Fresh rows only. WAPE is an internal selection objective; MAE, RMSE, sMAPE and supplementary RÂ² provide additional perspectives.

Retrain the frozen selection on all eligible history. Save fitted estimators, preprocessing schema, historical context and known calendar. Reload trusted artifacts to forecast the supplied future inputs. Validate row identifiers, order, schema, finite nonnegative volumes, chilledâ‰¤total, and structural chilled zeros. Future-demand perturbation checks verify that later actuals cannot alter training features or direct/recursive forecasts.

No actual data, fitted weights, observed results or competition-booklet contents are included in this public document.
