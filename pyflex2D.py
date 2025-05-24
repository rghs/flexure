# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import pandas as pd
import argparse


def get_intersect(a1, a2, b1, b2):
    # Find the intersection of two straight lines by specifying two points on each
    # Method from Norbu Tsering on Stack Overflow
    # https://stackoverflow.com/questions/3252194/numpy-and-line-intersections/42727584#42727584
    """
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1, a2, b1, b2])  # s for stacked
    h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
    l1 = np.cross(h[0], h[1])  # get first line
    l2 = np.cross(h[2], h[3])  # get second line
    x, y, z = np.cross(l1, l2)  # point of intersection
    if z == 0:  # lines are parallel
        raise ValueError("Lines are parallel!")
        # return (float('inf'), float('inf'))

    int_x = x / z
    int_y = y / z
    return int_x, int_y


def get_y_val(x_series, y_series, x_unknown):
    """
    Uses get_intersect to get the Y value of a desired x point.
    """
    x_series = np.asarray(x_series)
    y_series = np.asarray(y_series)

    lower = np.max(np.where(x_series <= x_unknown))
    upper = np.max(np.where(x_series >= x_unknown))

    a1 = [x_unknown, np.min(y_series) - 1]
    a2 = [x_unknown, np.max(y_series) + 1]
    b1 = [x_series[lower], y_series[lower]]
    b2 = [x_series[upper], y_series[upper]]

    _, intersect_y = get_intersect(a1, a2, b1, b2)

    return intersect_y


def sparse(i, j, v, m, n):
    # Conversion function found here: https://stackoverflow.com/questions/40890960/numpy-scipy-equivalent-of-matlabs-sparse-function
    """
    Create and compressing a matrix that have many zeros
    Parameters:
        i: 1-D array representing the index 1 values
            Size n1
        j: 1-D array representing the index 2 values
            Size n1
        v: 1-D array representing the values
            Size n1
        m: integer representing x size of the matrix >= n1
        n: integer representing y size of the matrix >= n1
    Returns:
        s: 2-D array
            Matrix full of zeros excepting values v at indexes i, j
    """
    return csr_matrix((v, (i, j)), shape=(m, n))


def load_crustal_parameters(filepath, node_spacing):
    data = pd.read_csv(filepath, dtype=float)
    profile_extent = np.max(data["distance"])
    profile_length = int(
        (profile_extent - (profile_extent % node_spacing)) + node_spacing
    )
    profile = np.arange(0, profile_length + node_spacing, node_spacing)

    load_height = np.zeros(len(profile))
    load_density = np.zeros(len(profile))
    elastic_thicknesses = np.ones(len(profile))

    for i, distance in enumerate(profile):
        if (i == 0) or (i == len(profile) - 1):
            continue
        y = get_y_val(data["distance"], data["elevation"], distance)
        if y < 0:
            load_height[i] = 0
        else:
            load_height[i] = y

        load_density[i] = get_y_val(data["distance"], data["density"], distance)

        elastic_thicknesses[i] = get_y_val(
            data["distance"], data["elastic_thickness"], distance
        )

    load_density[0] = load_density[1]
    load_density[len(profile) - 1] = load_density[len(profile) - 2]

    elastic_thicknesses[0] = elastic_thicknesses[1]
    elastic_thicknesses[len(profile) - 1] = elastic_thicknesses[len(profile) - 2]

    return profile, load_height, load_density, elastic_thicknesses


def draw_graph(profile, deflections, loads, elastic_thicknesses, filename):
    check = bool(
        np.all(np.round(elastic_thicknesses, 0) == np.round(elastic_thicknesses[0], 0))
    )
    depression = deflections * -1
    profile_km = profile / 1000

    if check is True:
        fig, ax = plt.subplots()

        ax.plot(profile_km, loads, label="Topographic load")
        ax.plot(profile_km, depression, label="Flexural deflection")
        ax.set_ylabel("Elevation (km)")
        ax.set_xlabel("x (km)")
        ax.set_xlim(np.min(profile_km), np.max(profile_km))

    else:
        elastic_thicknesses /= 1000
        fig = plt.figure()
        gs = fig.add_gridspec(2, hspace=0)
        ax = gs.subplots(sharex=True)

        ax[0].plot(profile_km, loads, label="Topographic load")
        ax[0].plot(profile_km, depression, label="Flexural deflection")
        ax[0].set_ylabel("Elevation (m)")
        ax[0].set_xlim(np.min(profile_km), np.max(profile_km))

        ax[1].plot(profile_km, elastic_thicknesses)
        ax[1].set_ylabel("Elastic thickness (km)")
        ax[1].set_xlabel("x (km)")
        ax[1].set_ylim(np.min(elastic_thicknesses) - 2, np.max(elastic_thicknesses) + 2)

        for graph in ax:
            graph.label_outer()

    fig.savefig(filename)
    plt.close()


def calculate_flexural_rigidity(elastic_thicknesses, E, v):
    flexural_rigidity = elastic_thicknesses**3.0 * E / (12.0 * (1.0 - v**2.0))
    return flexural_rigidity


def build_matrix(profile, flexural_rigidity, rho_m, rho_fill, g):
    size_index = (len(profile) - 4) * 5 + 4
    row_i = np.zeros(size_index)
    column_i = np.zeros(size_index)
    values = np.zeros(size_index)

    count = 0
    for i in [0, 1, len(profile) - 2, len(profile) - 1]:
        row_i[count] = i
        column_i[count] = i
        values[count] = 1
        count += 1

    # Nodes
    #   a-b-c-d-e
    # nodec is center node
    B = (rho_m - rho_fill) * g
    delta = profile[2] - profile[1]
    for i in range(2, len(profile) - 2):
        node_c = i

        node_a = node_c - 2
        node_b = node_c - 1
        node_d = node_c + 1
        node_e = node_c + 2

        A = flexural_rigidity[node_c]

        for j in range(1, 6):
            row_i[count] = node_c
            if j == 1:
                column_i[count] = node_a
                values[count] = A / (delta**4)
            if j == 2:
                column_i[count] = node_b
                values[count] = -4 * A / (delta**4)
            if j == 3:
                column_i[count] = node_c
                values[count] = 6 * A / (delta**4) + B
            if j == 4:
                column_i[count] = node_d
                values[count] = -4 * A / (delta**4)
            if j == 5:
                column_i[count] = node_e
                values[count] = A / (delta**4)
            count += 1

    design_matrix = sparse(row_i, column_i, values, len(profile), len(profile))
    return design_matrix


def main():
    parser = argparse.ArgumentParser(
        prog="pyflex2D",
        usage="%(prog)s input_file [OPTIONS]",
        description="Command line program for generating flexural curves from variable loads using centred finite difference.\nPython adaptation of Flex2D script by Jay Champan (written 11/2015).\nBased on flexural equations from Turcotte and Schubert (1982).",
        epilog="Licensed under the GNU General Public License v3.0. See repository for full text.",
    )

    parser.add_argument(
        "input_file",
        help="Path to input file containing topographic profile and flexural rigidity information. All variables MUST be in m.",
    )
    parser.add_argument(
        "--output-destination",
        "-o",
        help="Destination to write output files to. Defaults to execution directory.",
        default=".",
    )
    parser.add_argument(
        "--node-spacing",
        "-D",
        help="How closely spaced the grid nodes are. Defualts to 5000 m",
        type=float,
        default=5000,
    )
    parser.add_argument(
        "--fill-density",
        "-f",
        help="Density of the basin fill. Defaults to 2700 kg m^-3",
        type=float,
        default=2700,
    )
    parser.add_argument(
        "--mantle-density",
        "-m",
        help="Density of the mantle. Defaults to 3300 kg m^-3",
        type=float,
        default=3300,
    )
    parser.add_argument(
        "--youngs-modulus",
        "-E",
        help="Youngs Modulus. Defaults to 70e9 Pa",
        type=float,
        default=70.0e9,
    )
    parser.add_argument(
        "--gravity",
        "-g",
        help="Gravitational acceleration. Defaults to 9.81 m s^-2",
        type=float,
        default=9.81,
    )
    parser.add_argument(
        "--poisson",
        "-v",
        help="Poissons Ratio. Defaults to 0.25",
        type=float,
        default=0.25,
    )

    args = parser.parse_args()

    # Physical constants
    E = args.youngs_modulus
    v = args.poisson
    g = args.gravity
    rho_m = args.mantle_density
    rho_fill = args.fill_density

    # Model parameters
    print("Reading input...")
    x, load_height, load_density, elastic_thicknesses = load_crustal_parameters(
        args.input_file, args.node_spacing
    )

    flexural_rigidity = calculate_flexural_rigidity(elastic_thicknesses, E, v)
    load_energy = load_height * load_density * g

    # Create sparse matrix components
    print("Calculating deflections...")
    design_matrix = build_matrix(x, flexural_rigidity, rho_m, rho_fill, g)
    deflections = spsolve(design_matrix, load_energy)

    print("Writing output...")
    draw_graph(
        x,
        deflections,
        load_height,
        elastic_thicknesses,
        f"{args.output_destination}/flexure_graph.svg",
    )
    deflection_data = pd.DataFrame(
        {
            "distance_m": x,
            "deflection_m": deflections,
            "load_height_m": load_height,
            "load_density_kgm3": load_density,
            "elastic_thickness_m": elastic_thicknesses,
        }
    )
    deflection_data.to_csv(f"{args.output_destination}/flexure_data.csv")

    print("Done!")
    exit(0)


if __name__ == "__main__":
    main()
