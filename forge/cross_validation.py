import numpy as np

def k_fold_cross_validate(model, x_train, y_train, fold=10, **kwargs):
    sample_num = len(x_train)
    split = np.round(np.linspace(0, sample_num-1, num=fold+1))
    scores = np.empty(fold)
    for i in range(fold):
        model.fit(
            x_train = x_train[list(range(0, split[i]))+list(range(split[i+1], sample_num)), :],
            y_train = y_train[list(range(0, split[i]))+list(range(split[i+1], sample_num))],
            **kwargs
        )
        scores[i] = model.evaluate(
            x_test = x_train[split[i]:split[i+1], :],
            y_test = y_train[split[i]:split[i+1]]
        )
    return scores.mean()


def leave_one_out_cross_validate(model, x_train, y_train, **kwargs):
    sample_num = len(x_train)
    scores = np.empty(sample_num)
    for i in range(sample_num):
        model.fit(
            x_train = x_train[list(range(0, i))+list(range(i+1, sample_num)), :],
            y_train = y_train[list(range(0, i))+list(range(i+1, sample_num))],
            **kwargs
        )
        scores[i] = model.evaluate(
            x_test = x_train[i, :],
            y_test = y_train[i]
        )
    return scores.mean()
