import numpy as np

def naca4_coordinates(code, n_points=100):
    """Creates a set of x and y coordinates for a 4-digit NACA airfoil"""
    
    beta = np.linspace(0, np.pi, n_points)
    x = (1 - np.cos(beta)) / 2

    m = int(code[0]) / 100.0    # max camber as fraction of chord
    p = int(code[1]) / 10.0     # location of max camber as fraction of chord
    t = int(code[2:4]) / 100.0  # max thickness as fraction of chord

    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)

    if m == 0 or p == 0:
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)
    else:
        yc = np.where(
            x < p,
            m / p**2 * (2 * p * x - x**2),
            m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x - x**2)
        )
        dyc_dx = np.where(
            x < p,
            2 * m / p**2 * (p - x),
            2 * m / (1 - p)**2 * (p-x)
        )

    theta = np.arctan(dyc_dx)
    x_upper = x - yt * np.sin(theta)
    y_upper = yc + yt * np.cos(theta)
    x_lower = x + yt * np.sin(theta)
    y_lower = yc - yt * np.cos(theta)
    return x_upper, y_upper, x_lower, y_lower

def read_selig_dat(filepath):
    """Opens and parses a UIUC coordinate file"""
    with open(filepath, "r") as f:
        lines = f.readlines()

    coords = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) != 2:
            continue
        coords.append((float(parts[0]), float(parts[1])))
    coords = np.array(coords)
    x, y = coords[:, 0], coords[:, 1]

    le_idx = np.argmin(x)

    x_upper = x[le_idx::-1]
    y_upper = y[le_idx::-1]
    x_lower = x[le_idx:]
    y_lower = y[le_idx:]

    return x_upper, y_upper, x_lower, y_lower

def compute_shape_features(x_upper, y_upper, x_lower, y_lower, n_grid=200):
    """Computes max thickness and camber with their locations from interpolated upper and lower surface coordinates."""

    x_grid = np.linspace(0.001, 0.999, n_grid)
    yu = np.interp(x_grid, x_upper, y_upper)
    yl = np.interp(x_grid, x_lower, y_lower)

    thickness = yu - yl
    camber = (yu + yl) / 2

    max_thickness = thickness.max()
    thickness_loc = x_grid[np.argmax(thickness)]
    camber_loc = x_grid[np.argmax(np.abs(camber))]
    max_camber = camber[np.argmax(np.abs(camber))]

    return{
        "max_thickness": max_thickness,
        "thickness_loc": thickness_loc,
        "max_camber": max_camber,
        "camber_loc": camber_loc,
    }

if __name__ == "__main__":
    xu, yu, xl, yl = naca4_coordinates("2412")
    result = compute_shape_features(xu, yu, xl, yl)
    print(result)