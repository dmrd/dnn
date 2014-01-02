import mayavi.mlab as ml


def plot_points(points):
    """ Plots points as cubes in space """
    if len(points):
        x, y, z, s = zip(*points)
        ml.points3d(x, y, z, s, mode='cube', scale_factor=0.2, colormap="binary", transparent=True)


def visualize3d(vis, size):
    """ Size is (x, y, z) tuple """
    points = []
    vis = vis.flatten()
    normalized = (vis - vis.min())
    normalized /= (vis.max() or 1.0)
    # Go through and record
    for x in range(size[0]):
        for y in range(size[1]):
            for z in range(size[2]):
                i = x * size[1] * size[2] + y * size[2] + z
                if normalized[i]:
                    points.append((x, y, z, normalized[i]))
                else:
                    points.append((x, y, z, 0.01))

    plot_points(points)  # Plot set points
    ml.axes(extent=[0, size[0], 0, size[1], 0, size[2]], nb_labels=1)
    ml.show()
