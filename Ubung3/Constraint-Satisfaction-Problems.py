class CSP:
    def __init__(self, _domains, _constraints):
        # Copying domains and constraints to ensure they're not modified outside the class
        self.initial_domains = _domains.copy()
        self.domains = _domains.copy()
        self.constraints = _constraints
        # Initializing all variables to None to indicate they're unassigned
        self.assignments = {k: None for k in _domains}
        # Keeping a list of unassigned variables
        self.unassigned = list(_domains.keys())

    # Checks if assigning a given value to a variable are consistent with the CSP's constraints
    def isConsistent(self, var, value):
        for neighbor in self.constraints.get(var, []):
            if self.assignments[neighbor] == value:
                return False
        return True

    # Assigns a value to a variable and updates the unassigned variables list
    def assign(self, var, value):
        self.assignments[var] = value
        if var in self.unassigned:
            self.unassigned.remove(var)

    # Reverts a variable to being unassigned and restores its initial domain
    def unassign(self, var):
        self.assignments[var] = None
        self.unassigned.append(var)
        self.domains[var] = self.initial_domains[var].copy()

    # Selects the next variable to assign, using the Minimum Remaining Values (MRV) heuristic
    def getNextAssignableVar(self):
        return min((v for v in self.unassigned if self.domains[v]), key=lambda x: len(self.domains[x]), default=None)

    # Checks if the CSP is solved, i.e., all variables are assigned and the assignments are consistent
    def isSolved(self):
        return all(self.assignments.values()) and all(
            self.isConsistent(var, self.assignments[var]) for var in self.assignments)

    # Removes a specific value from a variable's domain
    def removeFromVariableDomain(self, variable, value):
        if value in self.domains[variable]:
            self.domains[variable].remove(value)

    # Returns a list of all variables that are neighbors to a given variable, according to the constraints
    def getNeighbours(self, variable):
        return self.constraints.get(variable, []) + [k for k, v in self.constraints.items() if
                                                     variable in v and k != variable]

    # Performs forward checking to ensure no domain becomes empty after an assignment
    def forward_checking(self, variable, value):
        for neighbour in self.getNeighbours(variable):
            if value in self.domains[neighbour] and len(self.domains[neighbour]) == 1:
                return False  # Failure if removing value leaves a neighbor with an empty domain
        return True


def backtrack_csp(csp):
    if all(csp.assignments.values()):
        return csp  # A solution has been found

    var = csp.getNextAssignableVar()

    # Try each value in the current domain of the variable
    for value in csp.domains[var]:
        # Check if assigning this value to the variable is consistent with the constraints
        if csp.isConsistent(var, value):
            # Temporarily assign this value to the variable
            original_domain = csp.domains[var].copy()
            csp.assign(var, value)

            # Apply forward checking to prune the domain of neighboring variables
            if csp.forward_checking(var, value):
                # Recursively attempt to complete the solution
                result = backtrack_csp(csp)
                # If a complete solution is found, return it
                if result:
                    return result

            # Undo the assignment and restore the original domain
            csp.unassign(var)
            csp.domains[var] = original_domain

    # If no value leads to a solution for this variable
    return None


"""
Australien? Das ist kein reales Land. Wenn Australien real wäre, dann würde ich einen Australier kennen.
"""
regions = ["WA", "NT", "SA", "Q", "NSW", "V", "T"]
domains = {region: ['red', 'green', 'blue'] for region in regions}
constraints = {
    "WA": ["NT", "SA"],
    "NT": ["WA", "SA", "Q"],
    "SA": ["WA", "NT", "Q", "NSW", "V"],
    "Q": ["NT", "SA", "NSW"],
    "NSW": ["SA", "Q", "V"],
    "V": ["SA", "NSW"],
    "T": []
}

map_csp_problem = CSP(domains, constraints)
solution = backtrack_csp(map_csp_problem)

if solution:
    print("Solution found:", solution.assignments)
    print()
else:
    print("No solution could be found.")
    raise RuntimeError

""""""""""""""""""""""""""""""""""""
"""Mecklenburg-Vorpommern"""
constraints = {
    'NM': ['S', 'LP', 'LR'],
    'S': ['LP', 'NM'],
    'LP': ['LR', 'S', 'NM', 'MS'],
    'LR': ['R', 'NM', 'LP', 'MS', 'VR'],
    'R': ['LR'],
    'VR': ['LR', 'VG', 'MS'],
    'MS': ['VG', 'LP', 'LR', 'VR'],
    'VG': ['MS', 'VR']
}
# Define the domains. Each area can have one of three colors.
domains = {region: ['red', 'green', 'blue'] for region in constraints.keys()}

# Create the CSP instance with the defined domains and constraints.
mecklenburg_vorpommern_csp = CSP(domains, constraints)

# Solve the CSP.
solution = backtrack_csp(mecklenburg_vorpommern_csp)

# Check if a solution was found and print the assignments.
if solution:
    print("Solution found for Mecklenburg-Vorpommern regions:", solution.assignments)
else:
    print("No solution could be found for the specified constraints.")
    raise RuntimeError

""""""""""""""""""""""""""""""""""""
"""
Cool way to visualise the graph
"""
import matplotlib.pyplot as plt
import networkx as nx

solution_colors = solution.assignments

positions = {
    'NM': (1, 2),
    'S': (2, 2),
    'LP': (3, 2),
    'LR': (4, 2),
    'R': (2, 1),
    'VR': (3, 1),
    'MS': (4, 1),
    'VG': (5, 1)
}

G = nx.Graph()

for region, color in solution_colors.items():
    G.add_node(region, color=color)

edges = [
    ('NM', 'S'), ('NM', 'LP'), ('NM', 'LR'),
    ('S', 'LP'),
    ('LP', 'LR'), ('LP', 'MS'),
    ('LR', 'R'), ('LR', 'MS'), ('LR', 'VR'),
    ('R', 'VR'),
    ('VR', 'MS'), ('VR', 'VG'),
    ('MS', 'VG')
]

G.add_edges_from(edges)

plt.figure(figsize=(10, 8))
node_colors = [G.nodes[region]['color'] for region in G.nodes()]  # Extract colors directly
nx.draw(G, positions, with_labels=True, node_color=node_colors, node_size=2000, edge_color='gray', linewidths=2,
        font_size=12, font_weight='bold')

plt.show()
