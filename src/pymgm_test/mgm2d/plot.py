import matplotlib.pyplot as plt


def plot(mgmobj):
    """
    Visualize the coarse levels of an MGM2D object

    Parameters:
        mgmobj : object
            MGM2D object

    Returns:
        h : list
            List of figure handles
    """
    if mgmobj is None:
        return

    num_levels = len(mgmobj[4]['levelsData'])
    h = []

    for j in range(num_levels):
        plt.figure()
        nodes = mgmobj[4]['levelsData'][j]['nodes']
        plt.plot(nodes[:, 0], nodes[:, 1], '.')
        plt.title('Nodes on level {}, total={}'.format(j, len(nodes)))
        plt.xlabel('x')
        plt.ylabel('y')
        h.append(plt.gca())
    plt.show()
    return h

# Example usage:
# Assuming mgmobj is an instance of your MGM2D object
# h = plot(mgmobj)
# plt.show()  # This will display all the plots
