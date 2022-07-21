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

# Selecting and Reindexing of Area of Interest

+++

### _The elegant way of dealing with large output_

+++

_Before working through this script, it is helpful to have had a look into [Triangular Meshes and Basic Plotting](tripcolor.ipynb) to get a basic understanding of plotting on a triangular basis._

Three-dimensional global ICON output for example requires many times more memory than two-dimensional output. The handling of such large amounts of data can very quickly lead to the exhaustion of the given memory. If you are sitting right next to a supercomputer, you are tempted to just request more RAM and go for it. However, there are elegant solutions besides the powerful one and since a lot of RAM also means a lot of electricity and a lot of coolant, the motivation for an elegant way is also to save limited resources.

In many cases, we rarely look at the complete global output, but rather at a specific, selected area. It is therefore advisable to cut out only this area. For this purpose it is very helpful to reduce the grid information from the global grid file to the area of interest but in such a way that the indexing makes sense starting at 0 and counting up continuously. The advantage is that we generate a new local grid-file, which looks like the global grid-file but is much smaller in terms of storage capacity and therefore easier and faster to handle.

So let's do something for the climate 🌍 and first of all load the necessary libraries and the global grid-file:

```{code-cell} ipython3
import xarray as xr
import numpy as np
```

## Importing the Grid-File

```{code-cell} ipython3
grid = xr.open_dataset(
    "/work/mh0287/k203123/Dyamond++/icon-aes-dyw2/experiments/dpp0029/icon_grid_0015_R02B09_G.nc"
)
grid
```

```{code-cell} ipython3
grid.cell_circumcenter_cartesian_x
```

```{code-cell} ipython3
max_plancks_birthplace_x, max_plancks_birthplace_y = np.array([10.13, 54.32])

left_bound   = 10.13 - 0.25
right_bound  = 10.13 + 0.25
top_bound    = 54.32 + 0.25
bottom_bound = 54.32 - 0.25
```

### Cells

```{code-cell} ipython3
window_cell = (
    (grid.clat >= np.deg2rad(bottom_bound))
    & (grid.clat <= np.deg2rad(top_bound))
    & (grid.clon >= np.deg2rad(left_bound))
    & (grid.clon <= np.deg2rad(right_bound))
).values

(window_cell_indices,) = np.where(window_cell)
window_cell_indices
```

What we have shown above are all the indices of the cells within the green window of the native grid. Since we selected them via `np.where()` we obtained the indices in 0-based python thinking. We now do the same for the vertices and edges that appear in our selected window:

+++

### Vertices

+++

We select with the function `.isel()` and the `vertex_of_cell` information from the grid all the vertices of the cells we have cut out one step before. We also sort the indices of the vertices and delete all duplicates via`np.unique()`. Since the ICON code is written in Fortran, i.e. the integer-values are 1-based, we subtract `1` to get back to python thinking, hence 0-based.

```{code-cell} ipython3
window_vertex_indices = (
    np.unique(grid.vertex_of_cell.isel(cell=window_cell_indices).values) - 1
)
window_vertex_indices
```

### Edges

+++

Same as for the vertices, we select with the function `.isel()` and the `edge_of_cell` information from the grid all the edges of the cells we have cut out two steps before. We also sort the indices of the edges and delete all duplicates via `np.unique()`. We subtract `1` to get back to python thinking.

```{code-cell} ipython3
window_edge_indices = (
    np.unique(grid.edge_of_cell.isel(cell=window_cell_indices).values) - 1
)
window_edge_indices
```

## Constructing New Grid with Selected Cells, Vertices and Edges

+++

Wow, that's already great ! We have received a lot of information in the form of indices in individual arrays about our green window. We merge them into one dataset so that everything is compact:

```{code-cell} ipython3
selected_indices = xr.Dataset(
    {
        "cell": ("cell", window_cell_indices),
        "vertex": ("vertex", window_vertex_indices),
        "edge": ("edge", window_edge_indices),
    }
)

selected_indices
```

It could be that we need more variables for future calculations. Therefore we create a dictionary with further interesting variables, which we reindex as a precaution.

```{code-cell} ipython3
vars_to_renumber = {
    "cell": [
        "adjacent_cell_of_edge",
        "cells_of_vertex",
        "neighbor_cell_index",
        "cell_index",
        "cell_circumcenter_cartesian_x",
    ],
    "vertex": [
        "vertex_of_cell", 
        "edge_vertices", 
        "vertices_of_vertex",
    ],
    "edge": [
        "edge_of_cell", 
        "edges_of_vertex",
    ],
}
```

**We now come to the heart of this script: the reindexing.**<br>
Several things happen here, which is why it is best to define a function. This function `reindex_grid()` needs 3 inputs and returns 1 output. The inputs are the original, complete grid and the parts of the grid that should be reindexed, hence `indices` and `vars_to_renumber`. The `indices` define the cells, vertices and edges. The `vars_to_renumber` are all variables that we are still interested in and can be composed of cells, vertices and edges. Output of our function will be a `new_grid` containing all indices and variables for our green window around the birthplace of Max Planck in such a way that everything starts counting at `0`.

+++

Let's go through it step by step: <br>

**Line 1**: We define a function wiht 3 input variables.

**Line 2**: We define as `new_grid` the area in the old grid that contains the indices we selected at the beginning of this script for the cells, vertices and edges. For this we use the `.load()` function, which loads the 17GB file into memory and processes it there: this is a little faster.

**Line 3**: We open a for-loop that accesses the coordinates and the entries of the array `selected_indices`.

**Line 4**: We open an array, which is only filled with `-2` (exceptional value like `nan` but as an integer) in the original, old grid length and call it `renumbering`.

**Line 5**: We start counting at `0` at the index positions of the long renumbering array, which belong to the indices of the selected dark green area, until we have reached the length of the short, previously selected array. So what we get is an array with the length of the original grid dimension (20971520 cells, 10485762 vertices, 31457280 edges), which contains a value other than `-2` only at that position within the array which is inside the dark green selected window.

**Line 6**: we open another for loop over the remaining variables `vars_to_renumber` to be reindexed.

**Line 7**: For the variables stored in the dictionary of a particular dimension (`cell`, `vertex`, `edge`), we take one item and access it in new_grid (line 2) and subtract `1` to work in python 0-based system; this is done in the square brackets on the right side of the equal sign. We use this to select the valid position in the `renumbering` array but in total we add `1` to output the `new_grid` in the same 1-based thinking as the original `grid`.

**Line 8**: We output the `new_grid`.

```{code-cell} ipython3
def reindex_grid(grid, indices, vars_to_renumber):
    new_grid = grid.load().isel(cell=indices.cell, vertex=indices.vertex, edge=indices.edge)
    for dim, idx in indices.coords.items():
        renumbering = np.full(grid.dims[dim], -2, dtype="int")
        renumbering[idx] = np.arange(len(idx))
        for name in vars_to_renumber[dim]:
            print(name)
            print(type(name))
            #new_grid[name].data = renumbering[new_grid[name].data - 1] + 1
    return new_grid
```

After long theory we want to use our function and create the actual `new_grid`:

```{code-cell} ipython3
selected_indices.coords.items()
```

```{code-cell} ipython3
vars_to_renumber['cell'][-1]
```

```{code-cell} ipython3
grid[vars_to_renumber['cell'][-1]].data
```

```{code-cell} ipython3
new_grid = reindex_grid(grid, selected_indices, vars_to_renumber)
new_grid
```

```{code-cell} ipython3
new_grid
```

Let's see if everything worked as we wanted it and choose reindexed variable like `.vertex_of_cell` and sort it:

```{code-cell} ipython3
np.unique(new_grid.vertex_of_cell)
```

Voilà ! That worked and we are done and have now built a new grid-file tailored to the area of our interest, which was provided with a new indexing.

+++

For further processing, the two datasets `selected_indices` and `new_grid` can be saved to have them quickly accessible for further calculations.

```{code-cell} ipython3
selected_indices.to_netcdf(
    f"selected_indices_region_{bottom_bound}-{top_bound}_{left_bound}-{right_bound}.nc",
    mode="w",
)
new_grid.to_netcdf(
    f"new_grid_region_{bottom_bound}-{top_bound}_{left_bound}-{right_bound}.nc",
    mode="w",
)
```

```{code-cell} ipython3

```
