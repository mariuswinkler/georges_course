import numpy as np

def stepwise_moving_averaged(x, w):
    n = len(x)
    m = np.zeros(len(x) - w)
    m[0] = sum(x[:w]) / w
    for i in range(1, n - w):
        m[i] = m[i - 1] + (x[i + w] - x[i - 1]) / w
    return m

def trend_function(y):
    x = np.arange(y.size) + 1
    mx = np.mean(x)
    my = np.mean(y)
    b = np.cov(y, x, bias=y.mean())[0, 1] / np.var(x)
    a = my - b * mx
    trend = a + b * x
    return trend, mx, my

def nrmse_function(y, trend, my):
    n = y.size
    mse = np.sum(np.abs(trend - y)) / n
    msemean = np.sum(np.abs(y - my)) / n
    nrmse = np.sqrt(mse / msemean)
    return nrmse