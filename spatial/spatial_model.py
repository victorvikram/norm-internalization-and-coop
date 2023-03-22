import random 
import numpy as np
import csv
from collections import defaultdict

from spatial_agent import SpatialAgent
from spatial_group import SpatialGroup
from spatial_grid import SpatialGrid
from spatial_logging import Logger

class SpatialModel:
    def __init__(
        self,
        n=20, # minimum number of agents in a group
        g=10, # number of groups to start
        size=10, # grid is a size by size square
        benefit=60, # public benefit gained by cooperating
        resources=20, # resources available on a given square
        cost_coop=20, # cost of cooperating
        cost_distant=5, # cost of going to an adjacent square to forage
        cost_stayin_alive=2, # cost of staying alive every round
        cost_repro=2, # cost of reproducing
        p_mutation=0, # probability of mutating
        threshold=0.5, # level at which civic learners increase their propensity to cooperate
        present_weight=0.3, # weight of the present data point in agent's moving average
        learning_rate = 0.05, # maximum change of pi every round
        epsilon=0.05, # probability of going against strategy 
        p_swap=0, # probability of migration if an agent is on a square of another group
        distrib=[1/3, 1/3, 1/3, 0], # probability of each agent type: [static, selfish, civic, coop]
        mut_distrib=None, # probability of mutating into each agent type: [static, selfish, civic, coop] note that mut_distrib has four slots!!
        years=1, # number of rounds
        memory=10, # memory for rolling window of average cooperation level, testing purposes
        rand=True, # turn on and off randomness (for testing purposes)
        write_log=True, # turns on and off logging
        p_obs=None, # can set p_obs to a constant value
        log_groups=False, # logs detailed info about groups
        mean_lifespan=50,
        similarity_threshold=1
        ): 

        param_dict = {
            "n": n, 
            "g": g,
            "size": size,
            "benefit": benefit,
            "resources": resources,
            "cost_coop": cost_coop,
            "cost_distant": cost_distant,
            "cost_stayin_alive": cost_stayin_alive,
            "cost_repro": cost_repro,
            "p_mutation": p_mutation,
            "threshold": threshold,
            "present_weight": present_weight,
            "learning_rate": learning_rate,
            "epsilon": epsilon,
            "p_swap": p_swap,
            "rand": rand,
            "distrib": distrib,
            "years": years,
            "p_obs": p_obs,
            "mean_lifespan": mean_lifespan,
            "similarity_threshold": similarity_threshold
        }

        # DEMOGRAPHICS AND GEOGRAPHY
        self.n = n
        self.g = g
        self.size = size
        self.mean_lifespan = mean_lifespan

        # COSTS AND BENEFITS
        self.benefit = benefit
        self.resources = resources
        self.cost_distant = cost_distant
        self.cost_stayin_alive = cost_stayin_alive
        self.cost_coop = cost_coop
        self.cost_repro = cost_repro
        self.death_age = {"sum": 0, "count": 0}

        if cost_distant > resources:
            raise Exception("cost_distant must be less than resources")

        # STRATEGIES AND ACTIONS
        self.p_mutation = p_mutation
        self.threshold = threshold
        self.present_weight = present_weight
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.p_swap = p_swap
        self.p_obs = p_obs
        self.similarity_threshold = similarity_threshold

        # RANDOMNESS ADJUSTMENT 
        self.rand = rand

        # INITIALIZING GROUPS AND AGENTS

        # forager_grid
        # count of foragers from all adjacent squares in the last round, to be used for decision-making in the current round. 
        # First index: [Row] 0 - topmost, n - bottom-most
        # Second index: [Col] 0 - leftmost, n - right-most
        # Third index: 0 - from this square, 1 - from above, 2 - from below, 3 - from left, 4 - from right 
        self.forager_grid = SpatialGrid(self.size, model=self)
        
        # forager_grid_next: count of foragers from all adjacent square on this round, is updated as agents make choices
        # necessary so that all agents decide based on last round's numbers
        self.forager_grid_next = SpatialGrid(self.size, self) 

        # grid_group_indices: size by size grid. grid_group_indices[i, j] is the index of the group
        # located at square (i, j). if there is no group at (i, j), then grid_group_indices[i, j] = -1.
        self.grid_group_indices = -np.ones((self.size, self.size)) 

        # dictionary with group IDs
        # only contains live groups (non-empty ones)
        self.groups = {}
        
        self.distrib = distrib
        if mut_distrib is None:
            self.mut_distrib = distrib
        else:
            self.mut_distrib = mut_distrib
        SpatialGroup.next_id = 0
        SpatialAgent.next_id = 0

        self.initialize_groups()
        self.year = 0
        self.years = years

        # LOGGING
        self.memory = memory
        self.write_log = write_log
        if self.write_log:
            config = f'y{years}_n{n}_g{g}_c{cost_coop}_b{benefit}_r{resources}_t{threshold}_pm{p_mutation}_ps{p_swap}_distrib{round(self.distrib[2], 2)}_cd{cost_distant}' 
            self.log_groups = log_groups
            self.logger = Logger(self, config, param_dict)
        
        self.can_terminate = False

    # SpatialModel -> None
    # initializes groups by finding a location for each group, and setting the count of foragers 
    # on each groups squares to one
    # *tested*
    def initialize_groups(self):
        # sample the points at which groups will be located
        group_points = self.sample_points(self.g)
        
        for i, point in enumerate(group_points):
            # initialize a group and fill it with agents
            group = SpatialGroup(model=self, location=tuple(point), agents=[])
            agents = [SpatialAgent(model=self, mean_lifespan=self.mean_lifespan, group=group) for j in range(self.n)] 
            group.set_agents(agents)

            # add the group to the model by changing
            # - adding population to the forager_grid
            # - adding the group index to grid_group_indices
            # - adding the group to the self.groups dictionary
            self.add_group(group, mod_forager_grid=True)

    # SpatialModel -> 
    # adds a group to the model, may or may not modify the forager_grid
    # **tested**
    def add_group(self, group, mod_forager_grid=False):
        # add the group to the grid as long as no group already exists there
        if self.grid_group_indices[group.location] == -1:
            self.grid_group_indices[group.location] = group.id 
            self.groups[group.id] = group # add the group to the list of groups

            # modify the forager_grid if desired
            # this is usually not needed, since forager_grid should reflect the foraging decisions of the previous round
            if mod_forager_grid:
                self.forager_grid.add_group(group.location, len(group.agents)) # initialize count of foragers on group square

    # SpatialModel -> List 
    # gives a random list of number distinct points on the map
    # **tested**
    def sample_points(self,number):
        points = [(i, j) for i in range(self.size) for j in range(self.size)]

        if self.rand:
            group_points = random.sample(points, number)
        else:
            group_points = points[:number]
        
        return group_points

    # SpatialModel -> 
    # some quantities need to be updated at the beginning of the loop, this function
    # consolidates all those updates 
    def increment_entities(self):
        self.death_age["sum"] = 0
        self.death_age["count"] = 0
        for group in self.groups.values():
            group.n_rounds += 1
            group.first_round = False
            group.just_budded = False
            for agent in group.agents:
                agent.first_round = False
                agent.age += 1

    def main(self):
        for i in range(self.years):
            self.loop()
            print(self.year)

            if self.can_terminate:
                break
    
    # SpatialModel -> 
    # does the main loop of the model 
    # - increment quantities
    # - death & birth
    # - bud groups
    # - square decisions
    # - cooperation decisions
    # - distribution of goods
    # - agent learning
    def loop(self, rand=True, rand_square=True):

        # CHANGED 
        # Only do this stuff after the first round, since the initialization occurs
        # before the first round
        if self.year != 0:
            self.increment_entities() # increments flags and counters 
            self.cycle_of_life() # kills and reproduces agents
            self.bud_groups() # split groups when needed

        # make decisions 
        self.square_decisions() # where to forage
        self.coop_decisions() # whether to cooperate
            
        # aggregates the payoffs for each group, and then distributes them to the agents
        for group in self.groups.values():
            group.group_distribution()
        
        # update agent pi-values based on what happened to them
        for group in self.groups.values():
            for agent in group.agents:
                agent.learn()

        # before agents die off, write stats
        # keep stats on number of agents and number of cooperators for each learning style:
        if self.write_log:
            self.logger.log_stats()
        
        self.year += 1

        # return n_agents
    
    # SpatialModel -> 
    # calls reproduction on all groups
    # **tested**
    def cycle_of_life(self):
        # first copy groups into a new list, since self.groups will be modified in the loop
        initial_groups = list(self.groups.values()) 

        # call death_and_birth on each group
        for group in initial_groups:
            group.death_and_birth()

    # FUNCTIONS FOR FORAGING/COOPERATING DECISIONS

    # SpatialModel ->
    # calls every agent from every group to make a decision on where to go
    # default_probs -- for testing purposes, allows us to feed in custom probabilities 
    # **tested**
    def square_decisions(self, default_probs=None):

        # make a list of all the agents, since they move around groups
        all_agents = []
        for group in self.groups.values():
            all_agents += group.agents
        
        self.calculations = 0 # testing purposes, checks how many times foraging probs was calculated
        
        foraging_probs_dict = defaultdict(list) # memoize the foraging probs for each group
        
        # for each agent choose a square
        for agent in all_agents: 
            # CHANGED we were using the same probabilities for each group!
            if default_probs is None:
                # when probs have been memoized for this group, used them
                if foraging_probs_dict[agent.group]:
                    probs = foraging_probs_dict[agent.group]
                # when they've not been calculated, calculate them
                else:
                    self.calculations += 1
                    probs = self.calc_foraging_probs(agent.group.location[0], agent.group.location[1])
                    foraging_probs_dict[agent.group] = probs
                agent.choose_square(probs)
            else:
                agent.choose_square(default_probs)
                
        # we've been updating forager_grid_next so as not to interfere with agent decision-making
        # so replace forager_grid with forager_grid_next
        self.forager_grid = self.forager_grid_next
        self.forager_grid_next = SpatialGrid(self.size, self)

    # SpatialModel -> 
    # calls every agent to decide whether to cooperate
    # **tested**
    def coop_decisions(self, rand=True):
        min_pi_sum = 0
        n_agents = 0
        for group in self.groups.values():
            cooperator_count = 0

            for agent in group.agents:
                # returns true if agent cooperates
                cooperator_count += agent.choose_coop(rand=rand, p_obs=self.p_obs)

                if self.year != 0:
                    n_here = self.forager_grid.num_foragers(agent.square)
                    min_pi_needed = 1 - agent.p_obs * group.avg_benefit/(self.cost_coop/n_here)
                    min_pi_sum += min_pi_needed
                    n_agents += 1
            
            # track the percentage of cooperators
            group.pct_cooperators = cooperator_count / len(group.agents)

            # keep a list of the last few pct_cooperators (self.memory determines the length of the list)
            group.pct_cooperators_memory.append(group.pct_cooperators)
            if len(group.pct_cooperators_memory) > self.memory:
                group.pct_cooperators_memory.pop(0)

            group.avg_pct_cooperators = sum(group.pct_cooperators_memory)/len(group.pct_cooperators_memory)

    # SpatialModel Int Int -> [Float, Float, Float, Float, Float]
    # Gives the probability of foraging on 
    # 0 - the current square, 1- up square, 2- down square, 3- left square, 4- right square
    # **tested**
    def calc_foraging_probs(self, group_row, group_col):
        payoff_here, payoff_outside = self.calc_expected_payoffs(group_row, group_col)

        # the prob of here should be proportional to the payoff here, 
        # prob outside should be proportional to payoff outside. For simplicity, don't discriminate based on direction
        prob_here = payoff_here / (payoff_here + payoff_outside)
        prob_outside = (payoff_outside / (payoff_here + payoff_outside)) / 4

        return [prob_here, prob_outside, prob_outside, prob_outside, prob_outside]

    # SpatialModel Int Int -> Int Int 
    # Takes the location of a group and calculates expected payoff for foraging on the square
    # versus foraging on a neighboring square 
    # **tested**
    def calc_expected_payoffs(self, group_row, group_col):
        n_outside = self.forager_grid.calculate_n_outside(group_row, group_col)
        n_here = self.forager_grid.num_foragers((group_row, group_col))

        # add one just in case these are 0, can justify it by arguing the +1 represents the possible presence of this agent
        payoff_here = self.resources / (n_here + 1) 
        payoff_outside = (self.resources - self.cost_distant) / (n_outside + 1)
        return payoff_here, payoff_outside
    
    # FUNCTIONS FOR SPLITTING GROUPS

    # SpatialModel -> 
    # Checks over all squares and where criteria are met for a new group, creates a bud group
    # **tested**
    def bud_groups(self):
        # check all squares
        for row in range(self.size):
            for col in range(self.size):
                # if the square doesn't have a group on it
                if self.grid_group_indices[row, col] == -1:
                    # determine which group (if any) should split and form a
                    # new group on this square
                    coord_of_origin = self.forager_grid.group_to_bud((row, col), int(self.n))
                    if coord_of_origin is not None:
                        self.split_group(coord_of_origin, (row, col))

    # SpatialModel Tuple Tuple ->
    # Takes the group on curr_square and splits it into two groups, one on curr_square
    # and the other on new_square. All agents foraging on new_square go into the new group
    # index is the direction of curr_square relative to new_square (e.g curr_square is to the 
    # LEFT (3) of new_square)
    # **tested**
    def split_group(self, curr_square, new_square):
        group_index = self.grid_group_indices[curr_square]
        
        # if the group_index is -1, that means the group met the criteria to split
        # but all agents died. the group therefore doesn't split
        if group_index == -1:
            return
        else:
            old_group = self.groups[group_index]

        # split up the agents according to what square they're on
        new_group_agents = []
        old_group_agents = []
        for agent in old_group.agents:
            if agent.square == new_square:
                new_group_agents.append(agent)
            else:
                old_group_agents.append(agent)

        # create the new group   
        new_group = SpatialGroup(self, new_square, [], avg_benefit=old_group.avg_benefit, is_bud=True)
        self.add_group(new_group, mod_forager_grid=False) 
        new_group.set_agents(new_group_agents)  # order is important, because there is a chance that new_group_agents will be empty, in which case we want the new group deleted
       
        old_group.first_round = True # overwrites old average benefit stats since lots of agents leaving
        old_group.set_agents(old_group_agents)
        old_group.just_budded = True
        old_group.budded_to = new_square


if __name__ == "__main__":
    cm = SpatialModel(n=20, g=10, size=10, resources=20, cost_coop=20, benefit=65, 
                cost_distant=5, cost_stayin_alive=2, cost_repro=2, threshold=0.5, p_mutation=0.01, p_swap=0.3, distrib=[0.49, 0.49, 0.02, 0], mut_distrib=[1/3, 1/3, 1/3, 0], years=11000, mean_lifespan=20, log_groups=False)
    cm.main()
        
