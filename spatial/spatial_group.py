import random

class SpatialGroup:
    next_id = 0

    def __init__(self, model, location, agents, avg_benefit=None, is_bud=False):

        self.location = location

        # identifying id, superstructures
        self.id = SpatialGroup.next_id
        SpatialGroup.next_id += 1
        self.model = model

        # cooperation
        self.pct_cooperators = None
        self.avg_pct_cooperators = 0
        self.pct_cooperators_memory = []
        self.avg_benefit = avg_benefit

        # strategy makeup
        self.all_civic = False
        self.mostly_civic = False
        self.majority_civic = False
        self.previously_all_civic = False
        self.n_agents = {}
        
        # budding and group history
        self.first_round = True
        self.n_rounds = 0
        self.is_bud = is_bud
        self.just_budded = False
        self.budded_to = None

        # initialize agents
        self.agents = agents 
        self.recount_agents()

        

    # SpatialGroup -> void
    # calculates how much individual and collective benefit each agent generates, and then 
    # increments each agent's fitness according to the rules
    # **tested**
    def group_distribution(self, rand=True):
        public_benefit = 0
        caught_count = 0
        
        # add up the public benefit and the number of agents that got caught
        for agent in self.agents:
            public_benefit += agent.public_benefit

            if agent.cooperate:
                agent.caught = False 
            else: 
                if rand:
                    i = random.random()
                    agent.caught = (i < agent.p_obs)
                else:  
                    agent.caught = (agent.p_obs > 0.5)
                    
            caught_count += agent.caught
        
        # split the public benefit between the uncaught agents
        share = public_benefit / (len(self.agents) - caught_count) if public_benefit > 0 else 0

        # avg_benefit is a rolling average of the shares, for coop decisions in the future
        if self.first_round:
            self.avg_benefit = share
        else:
            self.avg_benefit = (1 - self.model.present_weight)*self.avg_benefit + self.model.present_weight*share
            
        for agent in self.agents:
            # give agents their fitness
            if not agent.caught:
                agent.fitness_diff = agent.private_benefit + share
            else:
                agent.fitness_diff = agent.private_benefit
            
            # CHANGED
            # modify avg_fitness_diff, which is used for selfish learners in their learn stage
            if not agent.first_round:
                agent.avg_fitness_diff = (1 - self.model.present_weight) * agent.avg_fitness_diff + self.model.present_weight * agent.fitness_diff
            else:
                agent.avg_fitness_diff = agent.fitness_diff

            agent.fitness += agent.fitness_diff

        return caught_count, share, public_benefit
    
    # SpatialGroup -> Int
    # all agents in the group who cannot pay the cost of staying alive die off.
    # agents who can pay the cost of reproducing reproduce, and the child is
    # added to the group.
    # returns the count of agents that were tested
    def death_and_birth(self, rand=True):
        counter = 0 # testing purposes, check if we've seen each agent
        initial_agents = self.agents.copy()
        new_agents = []
        for agent in initial_agents:
            # if the agent survives, check if it has a child
            if agent.survives():
                new_agents.append(agent)
                child = agent.child(rand=rand)
                if child is not None:
                    new_agents.append(child)
            
            counter += 1

        # do it like this because removing agents has a high time complexity
        self.set_agents(new_agents) 
        
        return counter
    
    # SpatialGroup ->
    # updates the n_agents dictionary with the current strategy counts
    # also updates the flags
    # **tested**
    def recount_agents(self):
        self.n_agents = {"static": 0, "selfish": 0, "civic": 0}
        for agent in self.agents:
            self.n_agents[agent.learning] += 1

        self.previously_all_civic = self.all_civic
        self.majority_civic = self.n_agents["civic"] > (0.5 * len(self.agents))
        self.mostly_civic = self.n_agents["civic"] > (0.9 * len(self.agents))
        self.all_civic = self.n_agents["civic"] == len(self.agents)
    
    # SpatialGroup ->
    # sets the agents to a new list, updating group fields ccordingly
    # **tested**
    def set_agents(self, agents):
        for agent in agents:
            agent.group = self

        self.agents = agents

        if not agents:
            self.die()
        else:
            self.recount_agents()

    # SpatialGroup -> 
    # Adds an agent to the group, and updates fields accordingly
    # **tested**
    def add_agent(self, agent):
        self.n_agents[agent.learning] += 1
        self.agents.append(agent)
        agent.group = self

        self.majority_civic = self.n_agents["civic"] > (0.5 * len(self.agents))
        self.mostly_civic = self.n_agents["civic"] > (0.9 * len(self.agents))
        self.all_civic = self.n_agents["civic"] == len(self.agents)
   
    # SpatialGroup ->
    # Removes an agent from the group, and updates fields accordingly
    # **tested**
    def remove_agent(self, agent):
        self.n_agents[agent.learning] -= 1
        self.agents.remove(agent)

        if len(self.agents) == 0:
            self.die()
        else:     
            self.majority_civic = self.n_agents["civic"] > (0.5 * len(self.agents))
            self.mostly_civic = self.n_agents["civic"] > (0.9 * len(self.agents))
            self.all_civic = self.n_agents["civic"] == len(self.agents)

    # SpatialGroup ->
    # kills off a group, removing it from the dictionary of groups and the grid of group indices
    # these are the only ways we keep track of which groups are where
    # **tested**
    def die(self):
        self.model.groups.pop(self.id)
        self.model.grid_group_indices[self.location] = -1
        """
        if self.model.write_log:
            if self.all_civic:
                # self.model.civic_writer2.write(f'\n{self.model.year}: civic-only group {self.id} at' + 
                # f'{self.location} died off')
                self.model.logger.all_civic_deltas['civic groups died'] += 1
            else:
                self.model.logger.all_civic_deltas['noncivic groups died'] += 1
        """
