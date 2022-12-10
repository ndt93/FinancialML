from scipy.interpolate import CubicSpline


def interpolate_yield_curve(term_structure: [(float, float)]):
    """
    Interpolate a continuous interest rate yield curve from discrete term structure data
    """
    x = [i[0] for i in term_structure]
    y = [i[1] for i in term_structure]
    curve = CubicSpline(x, y)
    return curve
