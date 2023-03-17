from typing import Union
from collections.abc import Collection
import numpy as np


def get_array_percentile(
    arr: np.ndarray,
    percentile: Union[Union[int, float], Collection[Union[int, float]]],
    axes: Union[int, Collection[int]],
    tf: Collection = (1, 0),
) -> np.ndarray:
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
    if isinstance(percentile, Union[int, float]):
        new_percentile = (0, percentile)
    else:
        new_percentile = np.asarray(percentile).tolist()

    assert len(new_percentile) == 2, "Percentile tuple must be of length two, of the form (lower %, upper %)."
    lower = new_percentile[0]
    upper = new_percentile[1]
    assert lower != 100, "Lower % cannot be 100."

    axes_to_expand_into = []
    if isinstance(axes, Collection) and len(axes) != 1 or arr.ndim != 1:
        axes_to_expand_into = [i for i in range(arr.ndim) if i in axes]

    if upper == 100:
        pctls_upper = np.array(float("inf")).reshape([1 for _ in range(arr.ndim - 1)])
    else:
        pctls_upper = np.percentile(arr, upper, axis=[x for x in axes])

    pctls_lower = np.percentile(arr, lower, axis=[x for x in axes] if isinstance(axes, Collection) else axes)

    for ax in axes_to_expand_into:
        pctls_upper = np.expand_dims(pctls_upper, ax)
        pctls_lower = np.expand_dims(pctls_lower, ax)

    a = np.asarray(np.where((pctls_upper > arr) & (arr >= pctls_lower), *tuple(tf)))

    return a
