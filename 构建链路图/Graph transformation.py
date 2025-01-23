def graph_transformation(graph):
    # Step 1: Create a new vertex set V' containing all road segments ei
    V_prime = graph['E'].keys()

    # Step 2: Create an empty dictionary D to store road segments with the same starting point
    D = {}

    # Step 3: Iterate through all road segments ei, if the starting point of road segment ei is the same as vertex vj, add ei to the list of vj
    for ei, edge in graph['E'].items():
        vinit = edge['vinit']
        if vinit not in D:
            D[vinit] = []
        D[vinit].append(ei)

    # Step 4: Create a chained list C to store road segments with the same ending point
    C = {}
    for ei in V_prime:
        vterm = graph['E'][ei]['vterm']
        if vterm not in C:
            C[vterm] = []
        C[vterm].append(ei)

    # Step 5: Iterate through all road segments v'i, if the ending point of road segment v'i is the same as vertex vj, assign D(vj) to C(v'i)
    A = []  # Initialize the adjacency matrix A
    for v_prime in V_prime:
        row = []  # Initialize a row for the adjacency matrix
        for other_v_prime in V_prime:
            if other_v_prime in C.get(graph['E'][v_prime]['vterm'], []):
                row.append(1)  # There is a linkage between v_prime and other_v_prime
            else:
                row.append(0)  # No linkage between v_prime and other_v_prime
        A.append(row)  # Add the row to the adjacency matrix

    return A  # Return the adjacency matrix A

if __name__ == '__main__':
    # Example usage:
    # Define a graph with intersections and road segments
    graph = {
        'E': {
            '1': {'vinit': 'A', 'vterm': 'B'},
            '2': {'vinit': 'B', 'vterm': 'C'},
            '3': {'vinit': 'D', 'vterm': 'E'},
            # '4': {'vinit': 'F', 'vterm': 'G'},
            # '5': {'vinit': 'G', 'vterm': 'A'},
            # Add more edges as needed
        }
    }
    # Transform the graph
    adjacency_matrix = graph_transformation(graph)
    print(adjacency_matrix)