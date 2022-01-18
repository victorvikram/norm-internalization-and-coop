import numpy as np

class SpatialGrid:
    def __init__(self, size, model):
        self.grid = np.zeros((size, size, 5))
        self.size = size
        self.model = model
    
    # SpatialGrid ->
    # Adds a forager at the given location. If direction is None, this is the
    # forager's home square; otherwise, the agent traveled in that direction to
    # get to this square

    # direction guide:
    # 1 - location is above agent's home square (agent coming from below)
    # 2 - location is below agent's home square (agent coming from above)
    # 3 - location is to the left of agent's home square (agent coming from the right)
    # 4 - location is to the right of agent's home square (agent coming from the left)
    def add_agent(self, location, direction=None):
        if direction is None:
            self.grid[location[0], location[1], 0] += 1
        else:
            # if location is
            # direction = 1 - above agent's home square, the agent is from below
            # direction = 2 - below agent's home square, the agent is from above,
            # etc. Need to reverse up/down and left/right because of direction gives the 
            # direction the agent traveled, which is the opposite of the direction
            # the agent came from
            self.grid[location[0], location[1], [0, 2, 1, 4, 3][direction]] += 1 
    
    # SpatialGrid ->
    # remove agent from this square, similar to add_agent
    def delete_agent(self, location, direction=None):
        if direction is None:
            self.grid[location[0], location[1], 0] -= 1
        else:
            self.grid[location[0], location[1], [0, 2, 1, 4, 3][direction]] -= 1
        
    # SpatialGrid ->
    # Adds a group of agents to location as their home square.
    # **tested**
    def add_group(self, location, n, index=None):
        self.grid[location[0], location[1], 0] = n
    
    # SpatialGrid -> Number
    # Returns the numner of foragers at this location from all groups.
    # **tested** in testInvariants
    def num_foragers(self, location):
        return self.grid[location[0], location[1], :].sum()
    
    # SpatialGrid Int Int -> NdArray NdArray
    # row_indices: the indices for the row of the cells around cell (row, col)
    # col_indices: the indices for the col of the cells around cell (row, col)
    # 0 - up, 1 - down, 2 - left, 3 - right
    # **played**
    # **tested**
    def generate_indices(self, row, col):
        col_indices = np.array([col, col, self.modular_add(col, -1), self.modular_add(col, 1)])
        row_indices = np.array([self.modular_add(row, -1), self.modular_add(row, 1), row, row])
        return row_indices, col_indices
    
    # SpatialGrid Int Int -> NdArray
    # Calculates the average number of agents present on an adjacent square
    # Output: n
    # **tested**
    def calculate_n_outside(self, row, col):
        row_indices, col_indices = self.generate_indices(row, col)
        relevant_squares = self.grid[row_indices, col_indices,:]
        all_counts = relevant_squares.sum(axis=1)
        n_outside = all_counts.sum()/4

        return n_outside
    
    # Decides whether a new group should form on this square. (Must have at 
    # least n agents from some group foraging on this square.) If so, returns
    # the coordinates of the original group that these foragers are splitting 
    # off from and the index indicating the direction of the original group 
    # relative to the new group.
    # **tested**
    def group_to_bud(self, square, n):
        row = square[0]
        col = square[1]

        # which index has a plurality of the agents
        index = np.argmax(self.grid[row, col, :])
        
        # if that group has more than n agents on the square
        if self.grid[row, col, index] >= n:
            coord_of_origin = self.direction_to_coord(row, col, index) # find the square where those agents are from
            return coord_of_origin
        
        else:
            return None
    
    # SpatialGrid Int Int -> Int 
    # performs modular addition on a and b
    # **played**
    # **tested**
    def modular_add(self, a, b):
        return (a + b) % self.size
    
    # SpatialGrid Int Int Int -> (int, int)
    # returns the (row, col) of the square that is obtained if you travel 
    # in the given direction 
    # 0 - stay, 1 - up, 2 - down, 3 - left, 4 - right
    # **tested**
    def direction_to_coord(self, row, col, direction):
        if direction == 0:
            return (row, col)
        elif direction == 1:
            return (self.modular_add(row, -1), col)
        elif direction == 2:
            return (self.modular_add(row, 1), col)
        elif direction == 3:
            return (row, self.modular_add(col, -1))
        elif direction == 4:
            return (row, self.modular_add(col, 1))
        else:
            return   