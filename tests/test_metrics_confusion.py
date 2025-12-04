import numpy as np

from fif_mvp.train import metrics


def test_confusion_matrix_vectorized_matches_loop():
    labels = np.array([0, 1, 2, 1, 0, 2])
    preds = np.array([0, 2, 1, 1, 0, 2])
    num_labels = 3

    # Reference loop implementation
    ref = np.zeros((num_labels, num_labels), dtype=int)
    for y_true, y_pred in zip(labels, preds):
        ref[y_true, y_pred] += 1

    fast = metrics.confusion_matrix(labels, preds, num_labels)
    np.testing.assert_array_equal(fast, ref)
