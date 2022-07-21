---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3 (based on the module python3/2022.01)
  language: python
  name: python3_2022_01
---

# Part 1: Deriving Derivatives on Triangular Grid

+++

_To run this script you need output from [Selecting and Reindexing of Area of Interest](../cutting_grid_window.ipynb)._

+++

## Theoretical Introduction

+++

Since we are working on a grid whitout any right angles, we have to come up with something new to calculate gradients or derivatives. <br>
Basically, the idea is this: We do a local two-dimensional Taylor evolution for a cell $i$, giving us a plane equation containing 3 unknowns. If we apply this plane equation to three points (here the three neighbors) in the neighborhood of the cell $i$, the equation can be solved. This procedure can be applied in matrix form to all cells simultaneously and thus calculate the derivative for a certain area in one go.

To look at this from a mathematical point of view, we make use of linear algebra: if you have the coordinates of three points, you can use them to create a plane and thus a linear system of equations. <br>
This has the form:

$$\underline{\underline{A}}~ \underline{\phi} = \underline{p}.$$

If the matrix $\underline{\underline{A}}$ is square and has full rank, then the system has a unique solution given by:

$$\underline{\phi} = \underline{\underline{A}}^{-1}~\underline{p},$$
where $\underline{\underline{A}}^{-1}$ is the inverse of $\underline{\underline{A}}$. (For matrices which are not square or of  full rank the Moore-Penrose pseudoinverse can help here, and possibly be used to generalize this code.)

We reach the respective centers of the neighboring cells via their coordinates and write down the equations in parametric form of a plane:

$$
\begin{matrix}
\alpha(x_0, y_0) + \beta (x_1-x_0) + \gamma (y_1-y_0) = p_1(x_1, y_1)\\
\alpha(x_0, y_0) + \beta (x_2-x_0) + \gamma (y_2-y_0) = p_2(x_2, y_2)\\
\alpha(x_0, y_0) + \beta (x_3-x_0) + \gamma (y_3-y_0) = p_3(x_3, y_3)
\end{matrix}
$$

Perhaps you can already see the practical use of this approach; the derivative with respect to $x_i$ or $y_i$, with $i \in \{1,2,3\}$ results in the coefficients $\beta$ or $\gamma$:

$$\frac{\partial p_i}{\partial x_i} = \beta ~~\text{and}~~ \frac{\partial p_j}{\partial y_j} = \gamma$$
This means that we have to calculate these coefficients to get the derivatives of the corresponding variable.

We write down our equations in matrix notation:

$$
\underbrace{
\begin{bmatrix}
1 & (x_1-x_0) & (y_1-y_0)\\
1 & (x_2-x_0) & (y_2-y_0)\\
1 & (x_3-x_0) & (y_3-y_0)
\end{bmatrix}}_{\underline{\underline{A}}}
\underbrace{
\begin{bmatrix}
\alpha\\
\beta\\
\gamma
\end{bmatrix}}_{\underline{\phi}}
=
\underbrace{
\begin{bmatrix}
p_1(x_1, y_1)\\
p_2(x_2, y_2)\\
p_3(x_3, y_3)
\end{bmatrix}}_{\underline{p}}
$$

As a final step, we calculate the inverse of $\underline{\underline{A}}$ and automatically obtain the coefficients $\alpha, \beta, \gamma$ by calculating the matrix product $$\underline{\phi} = \underline{\underline{A}}^{-1}~\underline{p}.$$

Let's get our hands dirty now!

```{code-cell} ipython3
import xarray as xr
import numpy as np
```

## Deriving Derivatives

+++

The individual output variables of the ICON simulations do not contain any coordinates apart from the height (only in 3D output) and the time. Therefore it is necessary to get the coordinates from the grid-file. <br>
We have created a `new_grid` via the [Selecting and Reindexing of Area of Interest](../cutting_grid_window.ipynb) script, which contains the same information as the global grid-file focused on our area of interest with the advantage of being smaller and easier to handle.

+++

Range of area of interest:

```{code-cell} ipython3
# Max Plancks birthplace, Kiel, Schleswig-Holstein, Germany
left_bound = 9.88
right_bound = 10.38
top_bound = 54.57
bottom_bound = 54.07
```

We import the `selected_indices`, hence the indices for `cells`, `vertices` and `edges`, just like the `new_grid` which contains the grid information of the area of interest.

```{code-cell} ipython3
selected_indices = xr.open_dataset(
    f"selected_indices_region_{bottom_bound}-{top_bound}_{left_bound}-{right_bound}.nc"
)
new_grid = xr.open_dataset(
    f"new_grid_region_{bottom_bound}-{top_bound}_{left_bound}-{right_bound}.nc"
)

new_grid
```

Since this is an example of Max Planck's birthplace, let's take a look at how the pressure on his birthday in 2020 behaves within the region where he spent his first hours of life, so the dark green area from the script [Selecting and Reindexing of Area of Interest](../cutting_grid_window.ipynb). For this purpose we access the pressure `pfull` of the Dyamond dpp0029 simulation. The output will be the whole globe at this point.

```{code-cell} ipython3
max_birthday_data = xr.open_dataset(
    "/work/mh0287/k203123/Dyamond++/icon-aes-dyw2/experiments/dpp0029/dpp0029_atm_3d_1_ml_20200423T000000Z.nc"
)
pfull = max_birthday_data.pfull
pfull
```

But as we only want the area around his birthplace in Kiel, we select via `.isel(ncells = selected_indices.cell)` only the cells in the area of interest. Furthermore we average in time and choose the first 10 heightlevels from the atmosphere - land/ocean surface upwards (the z-vector in the ICON simulations points to the interior of the earth, i.e. to the surface of the earth, so that the level closest to the earth has the index `90` and we thus run from `81-90`).
<br>

The next step requires a relatively large amount of memory for various reasons. It is therefore advisable to use about 32GB at this point and save the result to use in later calculations.

```{code-cell} ipython3
pfull = (
    pfull.mean(dim="time")
    .isel(ncells=selected_indices.cell)
    .sel({"height": slice(81, 90)})
)
pfull
```

Before we start with our calculations, let's first look at the surface field of the pressure. For this we use [matplotlib.pyplot.tripcolor()](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.tripcolor.html), which we have used [here](../tripcolor.ipynb) before, because it can be used to represent triangular grids very nicely.

```{code-cell} ipython3
%matplotlib inline
import matplotlib.pylab as plt
```

To make this plot we need information from the `new_grid` file, namely the vertices of each cell `.vertex_of_cell` which again lives in the 1-based Fortran world, so we compensate the index offset by subtracting `1` in `voc`.

```{code-cell} ipython3
heightlevel = 90

vlon = np.rad2deg(new_grid.vlon)
vlat = np.rad2deg(new_grid.vlat)
voc = new_grid.vertex_of_cell.T.values - 1
p_data = pfull.sel(height=heightlevel).values / 100  ### in [hPa]

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.tripcolor(
    vlon, vlat, voc, p_data, cmap=plt.cm.get_cmap("turbo"), vmin=1022, vmax=1028
)
cbar = fig.colorbar(im)

cbar.set_label("$P\mathrm{sfc}$ / hPa")
ax.set_title(r"Surface Pressure $p$")
ax.set_xlabel("Longitude / deg")
ax.set_ylabel("Latitude / deg")

plt.show()
```

### Derivatives

+++

Now to the core of this script and to the interpretation of what we described in the [Theoretical Introduction](#Theoretical-Introduction) into code. Since we are interested in derivatives on the triangular grid, we want to calculate the pressure gradient along both the x and y directions as an example at this point.
<br>

Within the function `derivative()` we find 5 paragraphs:

In the **first paragraph**, we consider the neighbors of each cell, because the idea is to span a plane over the central initial cell with their help and to determine the derivative for the central cell in x- and y-direction by means of the slope of this plane.<br>
However, this reduces the initial set of cells to those that have 3 neighbors. The cells that are at the edge of our cut out area do not necessarily have 3 neighbors and are neglected here; the area shrinks a little. Only ```valid```cells within 3 neighbors are choosen.

The **second paragraph**, is used to identify the respective coordinates.

The **third paragraph**, outputs the pressure values for the valid cells.

In the **fourth** and **fifth paragraphs** happens what we described mathematically at the [beginning](#Theoretical-Introduction). <br>
We set up a system of equations in matrix form and solve it via matrix inversion. Finally, we output the coefficients and a boolean array of the valid cells and we are done!

```{code-cell} ipython3
def derivative(grid, data):
    neighbors = (grid.neighbor_cell_index.values - 1).T
    valid = np.all(neighbors >= 0, axis=-1)

    cell_lon = np.rad2deg(new_grid.clon.values[valid])
    cell_lat = np.rad2deg(new_grid.clat.values[valid])
    neighbors_lon = np.rad2deg(new_grid.clon.values[neighbors[valid]])
    neighbors_lat = np.rad2deg(new_grid.clat.values[neighbors[valid]])

    p = data.values[..., neighbors[valid]]
    ones = np.ones_like(neighbors_lon)
    A = np.stack(
        (
            ones,
            neighbors_lon - cell_lon[:, np.newaxis],
            neighbors_lat - cell_lat[:, np.newaxis],
        ),
        axis=2,
    )
    A_inv = np.linalg.inv(A)

    alpha, beta, gamma = np.einsum("...ij,...j->i...", A_inv, p)

    return alpha, beta, gamma, valid
```

We apply our defined function `derivative()` to our `new_grid` and the pressure `pfull` and we get $\alpha, \beta, \gamma$ which we identified as $p_0, \frac{\partial p}{\partial lon}, \frac{\partial p}{\partial lat}$. These are defined at the centroid of the points $p_1, p_2, p_3$, which we used to construct the linear equation system. The centroid of the triangle spanned by the centers of the neighboring cells lies approximately (but to a good approximation) on the center of the cell for which we wanted to calculate the derivative. <br>

By substracting the coordinates of each center cell from the coordinates of the corresponding neighbor cells within our matrix $A$, we obtain the relative difference in lat/lon direction of the central cell to its neighbors. By doing this $p_0$ corresponds to the averaged pressure value based on the neighbors cell pressure values.

```{code-cell} ipython3
p0, dpdlon, dpdlat, valid = derivative(new_grid, pfull)
```

Let's look at what the pressure gradient field looks like on the surface.<br>
We notice that we are looking at a small area, since the cells that do not have exactly 3 neighbors were not included in the calculation.

```{code-cell} ipython3
dvlon = np.rad2deg(new_grid.vlon)
dvlat = np.rad2deg(new_grid.vlat)
dvoc = voc[valid]  ### choosing only the cells with 3 neighbors
dpdlon_sfc = dpdlon[9, :]  ### choosing the surface layer, index 9

fig, ax = plt.subplots(figsize=(10, 6))
im = plt.tripcolor(dvlon, dvlat, dvoc, dpdlon_sfc, cmap=plt.cm.get_cmap("turbo"))
cbar = fig.colorbar(im)

cbar.set_label("dp/dlon / hPa deg$^{-1}$")
ax.set_title(r"Longitudinal Pressure Gradient: $\frac{\partial p}{\partial lon}$")
ax.set_xlabel("Longitude / deg")
ax.set_ylabel("Latitude / deg")

plt.show()
```

```{code-cell} ipython3
dpdlat_sfc = dpdlat[9, :]

fig, ax = plt.subplots(figsize=(10, 6))
im = plt.tripcolor(dvlon, dvlat, dvoc, dpdlat_sfc, cmap=plt.cm.get_cmap("turbo"))
cbar = fig.colorbar(im)

cbar.set_label("dp/dlat / hPa deg$^{-1}$")
ax.set_title(r"Latitudinal Pressure Gradient: $\frac{\partial p}{\partial lat}$")
ax.set_xlabel("Longitude / deg")
ax.set_ylabel("Latitude / deg")

plt.show()
```

In fact, it is difficult to tell with the naked eye whether our `derivative()` function did what we wanted it to do. For this reason you can see in [Part 2: Verification of Derivative Function](test_derivative.ipynb) a test example with the same function. Have fun !

Interesting side note: we compare below the initial pressure field on which our calculations are based with the calculated one. For this purpose, we plot the surface pressure field and see the initial pressure field from the simulation output on the left, the averaged pressure field based on the pressure values of the respective neighboring cells in the center and their difference to the right. <br>
Two things are remarkable:

1) The pressure fields are almost but not exactly equal. This is because the averaged values do not necessarily have to match the initial values for physical reasons.

2) Since we only calculate the pressure gradient for cells that have exactly three neighbors, the initial pressure field shrinks. The border cells of our window are truncated. This is especially important when averaging over the window and all cells afterwards. If we then compare averaged gradient fields with averaged non-gradient fields, we run the risk of comparing areas of different size. If, for example, the sea surface temperature is to be compared with the overlying pressure gradient field on average, the SST field may only contain as many cells as the gradient field; i.e. only those cells that have exactly three neighbors.

```{code-cell} ipython3
import matplotlib.gridspec as gridspec
```

```{code-cell} ipython3
p0_sfc = p0[9, :] / 100  # selecting surface layer and converting into [hPa].
```

```{code-cell} ipython3
fig = plt.figure(figsize=(16, 6))
fig.suptitle(
    "Initial Surface Pressure $p$ vs. Averaged Surface Pressure $p_0$", fontsize=15
)
ax = fig.subplots(1, 3)

ax[0].tripcolor(
    vlon, vlat, voc, p_data, cmap=plt.cm.get_cmap("turbo"), vmin=1022, vmax=1028
)
ax[0].set_title(f"Initial Surface Pressure $p$")
ax[0].set_xlabel("Longitude / deg")
ax[0].set_ylabel("Latitude / deg")

im = ax[1].tripcolor(
    dvlon, dvlat, dvoc, p0_sfc, cmap=plt.cm.get_cmap("turbo"), vmin=1022, vmax=1028
)
ax[1].set_title("Averaged Surface Pressure $p_0$")
ax[1].set_xlabel("Longitude / deg")

im1 = ax[2].tripcolor(
    dvlon,
    dvlat,
    dvoc,
    p_data[valid] - p0_sfc,
    cmap=plt.cm.get_cmap("seismic"),
    vmin=-1.5,
    vmax=1.5,
)
ax[2].set_title("Difference: $p-p_0$")
ax[2].set_xlabel("Longitude / deg")

fig.subplots_adjust(right=0.8)
cbar = fig.add_axes([0.82, 0.12, 0.018, 0.76])
cbar1 = fig.add_axes([0.88, 0.12, 0.018, 0.76])
fig.colorbar(im, cax=cbar)
fig.colorbar(im1, cax=cbar1, label="$P\mathrm{sfc}$ / hPa")

plt.show()
```
