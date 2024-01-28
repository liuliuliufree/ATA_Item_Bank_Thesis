import pandas as pd
import numpy as np


def icc_gpcm(b, a, theta):
    """
    计算广义部分信用模型(GPCM)的项目特征曲线概率。

    Parameters:
    - b (array-like): 项目难度参数的数组。
    - a (float): 项目的辨别力参数。
    - theta (array-like): 个体能力参数的数组。

    Returns:
    - numpy.ndarray: 一个包含广义部分信用模型项目特征曲线概率矩阵的二维NumPy数组。

    Examples:
    >>> b = [1, 2, 3]
    >>> a = 1.5
    >>> theta = [0, 1, 2]
    >>> icc_gpcm(b, a, theta)
    array([[9.27163916e-01, 7.23945032e-02, 4.41370939e-04, 2.10112044e-07],
          [4.81102832e-01, 4.81102832e-01, 3.75653106e-02, 2.29026178e-04],
          [3.62132427e-02, 4.63786757e-01, 4.63786757e-01, 3.62132427e-02]])
    """
    D = 1.7
    K = len(b) + 1
    P = np.zeros((len(theta), K))
    z = D * a * (np.tile(theta, len(b)).reshape(len(theta), len(b),order='F') - np.tile(b, len(theta)).reshape(len(theta), len(b)))
    total = np.ones(len(theta))
    total += np.exp(z[:, 0])
    for k in range(1, z.shape[1]):
        total += np.exp(np.sum(z[:, :k+1], axis=1))
    P[:, 0] = 1 / total
    P[:, 1] = np.exp(z[:, 0]) / total
    for r in range(2, K):
        P[:, r] = np.exp(np.sum(z[:, :r], axis=1)) / total
    return P

def item_info_gpc(b, a, theta):
    """
    计算广义部分信用模型(GPCM)的项目信息。

    Parameters:
    - b(array-like):项目难度参数的数组。
    - a(float):区分度参数。
    - theta(array-like):个体能力参数的数组。

    Returns:
    - array-like: 包含每个个体能力的项目信息的数组。
    
    Examples:
    >>> b = [1, 2, 3]
    >>> a = 1.5
    >>> theta = [0, 1, 2]
    >>> item_info_gpc(b, a, theta)
    array([0.44732148, 2.10202957, 2.56753144])
    """
    D = 1.7
    P = icc_gpcm(b, a, theta)
    K = np.arange(1, len(b) + 2)
    info = np.zeros(len(theta))
    
    for i in range(len(theta)):
        info[i] = (D * a)**2 * (np.sum(K**2 * P[i, :]) - (np.sum(K * P[i, :]))**2)
    
    return info
    
    
