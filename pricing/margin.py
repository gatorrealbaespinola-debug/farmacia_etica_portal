import numpy as np

MIN_MARGIN = 0.20

def compute_margin(r, a=0.30, b=0.28, gamma=0.75):
    if r <= 1:
        return MIN_MARGIN

    return (
        MIN_MARGIN
        + a * np.log(r)
        + b * (r - 1) ** gamma
    )
