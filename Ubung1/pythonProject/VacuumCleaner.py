import random

loc_A, loc_B = (0, 0), (1, 0)


class Thing:
    """This represents any physical object that can appear in an Environment.
    You subclass Thing to get the things you want. Each thing can have a
    .__name__ slot (used for output only)."""

    def __repr__(self):
        return '<{}>'.format(getattr(self, '__name__', self.__class__.__name__))

    def is_alive(self):
        """Things that are 'alive' should return true."""
        return hasattr(self, 'alive') and self.alive

    def show_state(self):
        """Display the agent's internal state. Subclasses should override."""
        print("I don't know how to show_state.")

    def display(self, canvas, x, y, width, height):
        """Display an image of this Thing on the canvas."""
        # Do we need this?
        pass


class Agent(Thing):
    """An Agent is a subclass of Thing with one required slot,
    .program, which should hold a function that takes one argument, the
    percept, and returns an action. (What counts as a percept or action
    will depend on the specific environment in which the agent exists.)
    Note that 'program' is a slot, not a method. If it were a method,
    then the program could 'cheat' and look at aspects of the agent.
    It's not supposed to do that: the program can only look at the
    percepts. An agent program that needs a model of the world (and of
    the agent itself) will have to build and maintain its own model.
    There is an optional slot, .performance, which is a number giving
    the performance measure of the agent in its environment."""

    def __init__(self, program):
        self.program = program


def TraceAgent(agent):
    """Wrap the agent's program to print its input and output. This will let
    you see what the agent is doing in the environment."""
    old_program = agent.program

    def new_program(percept):
        action = old_program(percept)
        print('{} perceives {} and does {}'.format(agent, percept, action))
        return action

    agent.program = new_program
    return agent


def ReflexVacuumAgent():
    """A reflex agent for the two-state vacuum environment. [Figure 2.8]"""

    def program(percept):
        location, status = percept
        if status == 'Dirty':
            return 'Suck'
        else:  # Choose to move randomly if the current location is clean
            return 'Right' if random.choice([True, False]) else 'Left'

    return Agent(program)


def ModelBasedVacuumAgent():
    """An agent that keeps track of what locations are clean or dirty."""
    model = {loc_A: None, loc_B: None}  # None means unknown

    def program(percept):
        location, status = percept
        model[location] = status
        if model[loc_A] == model[loc_B] == 'Clean':
            return 'NoOp'
        elif status == 'Dirty':
            return 'Suck'
        else:
            return 'Right' if location == loc_A else 'Left'

    return Agent(program)


class Environment:
    """Abstract class representing an Environment. 'Real' Environment classes
    inherit from this. Your Environment will typically need to implement:
        percept: Define the percept that an agent sees.
        execute_action: Define the effects of executing an action.
                           Also update the agent. Performance slot.
    The environment keeps a list of .things and .agents (which is a subset
    of .things). Each agent has a .performance slot, initialized to 0.
    Each thing has a .location slot, even though some environments may not
    need this."""

    def __init__(self):
        self.things = []
        self.agents = []

    def thing_classes(self):
        return []  # List of classes that can go into environment

    def percept(self, agent):
        """Return the percept that the agent sees at this point. (Implement this.)"""
        raise NotImplementedError

    def execute_action(self, agent, action):
        """Change the world to reflect this action. (Implement this.)"""
        raise NotImplementedError

    def default_location(self, thing):
        """Default location to place a new thing with unspecified location."""
        return None

    def exogenous_change(self):
        """If there is spontaneous change in the world, override this."""
        pass

    def step(self):
        """Run the environment for one time step. If the
        actions and exogenous changes are independent, this method will
        do. If there are interactions between them, you'll need to
        override this method."""
        actions = []
        for agent in self.agents:
            actions.append(agent.program(self.percept(agent)))
        for (agent, action) in zip(self.agents, actions):
            self.execute_action(agent, action)
        self.exogenous_change()

    def run(self, steps=1000):
        """Run the Environment for given number of time steps."""
        for step in range(steps):
            self.step()

    def list_things_at(self, location, tclass=Thing):
        """Return all things exactly at a given location."""
        return [thing for thing in self.things
                if thing.location == location and isinstance(thing, tclass)]

    def some_things_at(self, location, tclass=Thing):
        """Return true if at least one of the things at location
        is an instance of class tclass (or a subclass)."""
        return self.list_things_at(location, tclass) != []

    def add_thing(self, thing, location=None):
        if not isinstance(thing, Thing):
            thing = Agent(thing)
        if thing in self.things:
            print("Can't add the same thing twice")
        else:
            # Überprüfe, ob die Umgebung Attribute für width und height besitzt
            if location is None and hasattr(self, 'width') and hasattr(self, 'height'):
                location = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
            thing.location = location if location is not None else self.default_location(thing)
            self.things.append(thing)
            if isinstance(thing, Agent):
                thing.performance = 0
                self.agents.append(thing)

    def delete_thing(self, thing):
        """Remove a thing from the environment."""
        try:
            self.things.remove(thing)
        except ValueError as e:
            print(e)
            print("  in Environment delete_thing")
            print("  Thing to be removed: {} at {}".format(thing, thing.location))
            print("  from list: {}".format([(thing, thing.location) for thing in self.things]))
        if thing in self.agents:
            self.agents.remove(thing)


class TrivialVacuumEnvironment(Environment):
    """This environment has two locations, A and B. Each can be Dirty
    or Clean. The agent perceives its location and the location's
    status. This serves as an example of how to implement a simple
    Environment."""

    def __init__(self):
        super().__init__()
        self.status = {loc_A: random.choice(['Clean', 'Dirty']),
                       loc_B: random.choice(['Clean', 'Dirty'])}

    def thing_classes(self):
        return [Wall, Dirt, ReflexVacuumAgent, RandomVacuumAgent,
                TableDrivenVacuumAgent, ModelBasedVacuumAgent]

    def percept(self, agent):
        """Returns the agent's location, and the location status (Dirty/Clean)."""
        return (agent.location, self.status[agent.location])

    def execute_action(self, agent, action):
        """Change agent's location and/or location's status; track performance.
        Score 10 for each dirt cleaned; -1 for each move."""
        if action == 'Right':
            agent.location = loc_B
            agent.performance -= 1
        elif action == 'Left':
            agent.location = loc_A
            agent.performance -= 1
        elif action == 'Suck':
            if self.status[agent.location] == 'Dirty':
                agent.performance += 10
            self.status[agent.location] = 'Clean'

    def default_location(self, thing):
        """Agents start in either location at random."""
        return random.choice([loc_A, loc_B])


class VacuumEnvironment(Environment):
    def __init__(self, width=3, height=3):
        super().__init__()
        self.width = width
        self.height = height
        # Initialisiere das Gitter mit zufälligen Zuständen
        self.status = {(x, y): random.choice(['Clean', 'Dirty']) for x in range(width) for y in range(height)}

    def percept(self, agent):
        """Gibt die aktuelle Position des Agenten und den Zustand dieser Position zurück."""
        return agent.location, self.status[agent.location]

    def execute_action(self, agent, action):
        """Führt eine Aktion aus und ändert die Umgebung und/oder den Zustand des Agenten entsprechend."""
        if action == 'Suck':
            self.status[agent.location] = 'Clean'
        elif action in ('MoveRight', 'MoveLeft', 'MoveUp', 'MoveDown'):
            x, y = agent.location
            if action == 'MoveRight' and x < self.width - 1:
                agent.location = (x + 1, y)
            elif action == 'MoveLeft' and x > 0:
                agent.location = (x - 1, y)
            elif action == 'MoveUp' and y > 0:
                agent.location = (x, y - 1)
            elif action == 'MoveDown' and y < self.height - 1:
                agent.location = (x, y + 1)


class GridVacuumAgent(Agent):
    def __init__(self):
        super().__init__(self.grid_program)  # Pass the program function
        self.directions = ['MoveRight', 'MoveDown', 'MoveLeft', 'MoveUp']  # Possible directions
        self.directionIndex = 0  # Start by moving right

    def grid_program(self, percept):
        """Defines the program that controls the agent's actions."""
        location, status = percept
        # Clean if dirty
        if status == 'Dirty':
            return 'Suck'
        # Move according to the current direction
        action = self.directions[self.directionIndex]
        # Prepare to change direction in the next step, if needed
        self.directionIndex = (self.directionIndex + 1) % len(self.directions)
        return action


# 1
print("1")
a = ReflexVacuumAgent()
print(a.program((loc_A, 'Clean')))
print(a.program((loc_B, 'Clean')))
print(a.program((loc_A, 'Dirty')))
print(a.program((loc_A, 'Dirty')))

# 2
print("\n" + "2")
e = TrivialVacuumEnvironment()
e.add_thing(TraceAgent(ModelBasedVacuumAgent()))
e.run(5)

# 3
print("\n" + "3")
e3 = VacuumEnvironment(width=5, height=5)  # Create a larger grid for more interesting behavior
grid_vacuum_agent = GridVacuumAgent()
e3.add_thing(grid_vacuum_agent, location=(0, 0))  # Start at the top-left corner
e3.run(25)  # Run for a number of steps to see the agent in action
