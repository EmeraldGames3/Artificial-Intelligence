import numpy as np


def reward(s, a, snew):
    if a == "upC" and snew == 2:
        return 10
    elif a == "right" and snew == (s + 1) % 6:
        return 5
    elif a == "left" and snew == (s - 1) % 6:
        return 5
    else:
        return -1


def performAction(s, a1):
    actions = ["upC", "left", "right"]
    if a1 == "up":
        probabilities = [0.5, 0.25, 0.25]
    else:
        probabilities = [1 / 3, 1 / 3, 1 / 3]
    actual_action = np.random.choice(actions, p=probabilities)

    if actual_action == "upC":
        snew = 2
    elif actual_action == "right":
        snew = (s + 1) % 6
    elif actual_action == "left":
        snew = (s - 1) % 6
    else:
        snew = s

    return actual_action, snew


def qLearn(actions, states, gamma=0.5):
    Q = np.zeros((6, 4))
    action_indices = {"up": 0, "upC": 1, "left": 2, "right": 3}

    for i in range(len(actions) - 1):
        s = states[i]
        a = actions[i][1]
        snew = states[i + 1]
        r = reward(s, a, snew)

        a_index = action_indices[a]
        Q[s, a_index] = r + gamma * np.max(Q[snew, :])

    return Q


a1s = np.random.choice(np.array(["up", "upC", "left", "right"]), size=100, replace=True)
states = [0]
actions = []

for i in range(len(a1s)):
    sanew = performAction(states[i], a1s[i])
    actions.append((a1s[i], sanew[0]))
    states.append(sanew[1])

Q = qLearn(actions, states)
print(Q)
