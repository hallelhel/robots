from tsfresh import extract_features


def tsfresh_features(data, time_column, label, record, index):
    """
    get features based on the time series data for each experiment
    :param data: time series data set
    :param time_column: column in the data set that indicates the relative time
    :param label: patient_id
    :param record: experiment (task) full id
    :param index: name of index column
    :return: df with tsfresh features
    """

    # this is a specific set chosen by me based on the paper.  maybe you should start with a bigger set.
    fc_parameters = {
                     "change_quantiles": [{"ql": 0.2, "qh": 1.0, "isabs": True, "f_agg": "var"}],
                     "energy_ratio_by_chunks": [{"num_segments": 10, "segment_focus": 1},
                                                {"segment_focus": 6, "num_segments": 10}],
                     "abs_energy": None,
                     "ratio_beyond_r_sigma": [{"r": 0.5}]}

    features_tsfresh = extract_features(
        data.drop([label, record], axis=1),
        column_id=index, column_sort=time_column, default_fc_parameters=fc_parameters, n_jobs=0)

    features_tsfresh[label] = data.groupby(index)[label].max()
    features_tsfresh[record] = data.groupby(index)[record].max()

    features_tsfresh.index.rename(index, inplace=True)

    return features_tsfresh
