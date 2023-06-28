#!/usr/bin/env python3
"""Early Stopping"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Implements early stopping to monitor the cost function during training.

    Args:
        cost (float): Current value of the cost function.
        opt_cost (float): Optimal value of the cost function.
        threshold (float): Threshold value to determine if the cost
        improvement is significant.
        patience (int): Number of consecutive iterations with no
        significant improvement to trigger early stopping.
        count (int): Number of iterations with no significant improvement.

    Returns:
        Tuple[bool, int]: A tuple containing a boolean flag
        indicating whether to stop training (True) or not (False),
        and the updated count value.
    """
    cost_val = opt_cost - cost

    if cost_val > threshold:
        count = 0
    else:
        count += 1

    if count < patience:
        return (False, count)
    else:
        return (True, count)
