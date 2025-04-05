import numpy as np
import random
from queue import PriorityQueue

#Goal state
goal_state = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])

#Heuristic function
def heuristic(state, goal=goal_state):
    distance = 0
    for i in range(3):
        for j in range(3):
            if state[i, j] != 0:
                goal_pos = np.argwhere(goal == state[i, j])[0]
                distance += abs(i - goal_pos[0]) + abs(j - goal_pos[1])
    return distance

#A* search for the sliding puzzle
def a_star_search(start):
    start_tuple = tuple(start.flatten())  #Flatten the matrix
    goal_tuple = tuple(goal_state.flatten())

    frontier = PriorityQueue()
    frontier.put((0 + heuristic(start), 0, start_tuple))

    came_from = {}
    cost_so_far = {}

    came_from[start_tuple] = None
    cost_so_far[start_tuple] = 0

    while not frontier.empty():
        _, current_cost, current = frontier.get()
        current_state = np.array(current).reshape(3, 3)  #convert back into 3x3

        if current == goal_tuple:
            break

        #Get the empty space (0) position
        empty_pos = tuple(np.argwhere(current_state == 0)[0])
        x, y = empty_pos

        #Generate neighbors
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  #Up, Down, Left, Right
        for dx, dy in moves:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < 3 and 0 <= new_y < 3:
                #Swap the empty space with the adjacent tile
                new_state = current_state.copy()
                new_state[x, y], new_state[new_x, new_y] = new_state[new_x, new_y], new_state[x, y]
                new_tuple = tuple(new_state.flatten())

                new_cost = current_cost + 1
                if new_tuple not in cost_so_far or new_cost < cost_so_far[new_tuple]:
                    cost_so_far[new_tuple] = new_cost
                    priority = new_cost + heuristic(new_state)
                    frontier.put((priority, new_cost, new_tuple))
                    came_from[new_tuple] = current

    #Reconstruct the path
    current = goal_tuple
    path = []
    while current != start_tuple:
        path.append(np.array(current).reshape(3, 3))
        current = came_from[current]
    path.append(start)
    path.reverse()

    return path, cost_so_far[goal_tuple]

#Check if the puzzle is solvable
def is_solvable(puzzle):
    one_d_puzzle = puzzle.flatten()
    inversions = 0
    for i in range(len(one_d_puzzle)):
        for j in range(i + 1, len(one_d_puzzle)):
            if one_d_puzzle[i] > one_d_puzzle[j] and one_d_puzzle[i] != 0 and one_d_puzzle[j] != 0:
                inversions += 1
    return inversions % 2 == 0

#Generate a solvable puzzle
def generate_solvable_puzzle():
    while True:
        puzzle = generate_puzzle()
        if is_solvable(puzzle):
            return puzzle

#Generate random puzzle
def generate_puzzle():
    list1 = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    puzzle = np.zeros((3, 3), dtype=int)
    for i in range(3):
        for j in range(3):
            value = random.choice(list1)
            puzzle[i, j] = value
            list1.remove(value)
    return puzzle

#Test A* Search
start = generate_solvable_puzzle()
print("Initial Puzzle:")
print(start)

path, cost = a_star_search(start)

print("\nSolution Path:")
for step in path:
    print(step)
    print()

print(f"Total moves: {cost}")
