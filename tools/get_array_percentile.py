from typing import Iterable, Union
from numbers import Number
import numpy as np


def get_array_percentile(
        arr: np.array, percentile: Union[Union[int, float], Iterable[Union[int, float]]],
        axes: Union[int, Iterable[int]],
        tf: Iterable = (1, 0)) -> np.array:
    """Gets np.array of numbers of the same shape as an input depending on whether or not each 
        element in the input array satisfies axiswise percentile

    Args:
        arr (np.array): The input array
        percentile (number or iterable of number (length 2)): The percentile threshold. 
                    *  If float, inserts elements from `tf` into result array where 
                        elements of `arr` are < `percentile` along the provided axe(s).
                    *  If float, inserts elements from `tf` into result array where 
                        elements of `arr` are < `percentile[1]` and >= than `percentile[0]` along the provided axe(s).
        axes (int or iterable of ints): The axe(s) along which to compute percentiles.
        tf (iterable): [Default = (1, 0)] `tf[0]` is object to insert 
            if the percentile condition is met, and `tf[1]` is the object to insert otherwise.

    Returns: 
        (np.array): np.array based on percentile and arr
    """

    assert len(tf) == 2, "Length of `tf` must be 2."
    if isinstance(axes, int):
        axes = [axes]
    assert len(axes) <= arr.ndim, "The number of axes provided must not exceed the dimension of the input array."

    if isinstance(percentile, Number):
        percentile = (0, percentile)
    assert len(percentile) == 2, "Percentile tuple must be of length two, of the form (lower %, upper %)."
    lower = percentile[0]
    upper = percentile[1]
    assert lower != 100, "Lower % cannot be 100."

    axes_to_expand_into = []
    if len(axes) != 1 or arr.ndim != 1:
        axes_to_expand_into = [i for i in range(arr.ndim) if i in axes]

    if upper == 100:
        pctls_upper = np.array(float('inf')).reshape([1 for _ in range(arr.ndim - 1)])
    else:
        pctls_upper = np.percentile(arr, upper, axis=axes)

    pctls_lower = np.percentile(arr, lower, axis=axes)

    for ax in axes_to_expand_into:
        pctls_upper = np.expand_dims(pctls_upper, ax)
        pctls_lower = np.expand_dims(pctls_lower, ax)

    a = np.where((pctls_upper > arr) & (arr >= pctls_lower), *tuple(tf))

    return a
