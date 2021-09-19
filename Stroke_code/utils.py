import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, auc


def cross_validation(n_splits, data_len):
    """
    split data into folds
    :param n_splits:
    :param data_len:
    :return:
    """
    s = list(range(0, data_len))
    random.shuffle(s)
    s = [s[i::n_splits] for i in range(n_splits)]
    return s


def precision_recall_graph(y_test, y_score, classes, method):
    """
    print precision recall graph with AUC and precision for each label
    :param y_test: test true values
    :param y_score: predicted probabilities
    :param classes: classes names
    :param method: name of the mthod performed / data kind (will be printed in the headline)
    """
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i, _ in enumerate(classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                            y_score[:, i])
        average_precision[i] = precision_score(y_test[:, i], np.round(y_score[:, i]))

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
                                                                    y_score.ravel())
    average_precision["micro"] = precision_score(y_test, np.round(y_score), average="micro")

    auc_score = auc(recall["micro"], precision["micro"])
    print('PR AUC: %.3f' % auc_score)

    # A "micro-average": quantifying score on all classes jointly
    # precision["macro"], recall["macro"], _ = precision_recall_curve(Y_test.ravel(),
    #                                                                 y_score.ravel())
    # average_precision["macro"] = average_precision_score(Y_test, y_score,
    #                                                      average="macro")

    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))

    plt.figure()
    plt.step(recall['micro'], precision['micro'], where='post', label=f'micro-averaged precision score'
                                                                      f'(area= {auc_score:0.2f})')
    # plt.step(recall['macro'], precision['macro'], where='post',
    #          label=f'Average precision score, macro-averaged over all classes '
    #                f'(area = {average_precision["macro"]:0.2f})')

    for i, target in enumerate(classes):
        plt.step(recall[i], precision[i], where='post', label=f'class {target} precision '
                                                              f'{average_precision[i]:0.2f}')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(fontsize='x-small')
    plt.title(f'Precision Recall graph {method}')
    plt.show()


def scale_data(data):
    """
    normalize data by mean and std
    :param data: data set
    :return: normalize data
    """
    for column in data.columns:
        # scale
        mean_s = np.mean(data.loc[:, column])
        sd_s = np.std(data.loc[:, column])
        data.loc[:, column] = (data.loc[:, column] - mean_s) / (sd_s + 1 * np.exp(-6))

    data.fillna(0, inplace=True)
