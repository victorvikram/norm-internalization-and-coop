import random
import numpy as np

class SpatialAgent: 
    next_id = 0

    def __init__(self, model, group, mean_lifespan=50, pi=None, learning=None, rand=True):
        # assign identifying id, and superstructures
        self.id = SpatialAgent.next_id
        SpatialAgent.next_id += 1
        self.model = model
        self.birth_group = group
        self.group = group

        # initialize propensity to cooperate
        if pi is None:
            self.pi = np.random.uniform(low=-1, high=1)
        else:
            self.pi = pi
        
        self.avg_pi = self.pi
        
        # initialize learning style
        if learning is None:
            self.learning = random.choices(["static", "selfish", "civic", "coop"], weights=self.model.distrib)[0] # only happens if this agent is not offspring
        else:
            self.learning = learning

        if self.learning == "coop":
            self.pi = 1

        # variables about life and death
        if rand:
            self.lifespan = np.random.normal(mean_lifespan, mean_lifespan/3)
            self.lifespan = self.lifespan if self.lifespan >= 0 else 0 
        else:
            self.lifespan = mean_lifespan 
        self.age = 0
        self.first_round = True

        # variables for fitness, not to be used until initially set in distribution
        # EXCEPT self.fitness, which becomes active immediately
        self.avg_fitness_diff = None
        self.fitness_diff = None
        self.fitness = self.model.cost_stayin_alive

        # variables for location, not to be used until initially set in model.square_decisions
        self.square = None
        self.foraging_direction = None
        self.just_migrated = False
        
        # variables for cooperation, not to be used until initially set in model.coop_decisions
        self.cooperate = None
        self.public_benefit = None
        self.private_benefit = None

    
    # SpatialAgent List Int -> Tuple 
    # assigns a foraging square to an agent
    # returns where agent forages
    # **tested**
    def choose_square(self, probs=None):
        # choose direction to forage in, get which square that corresponds to
        # 0 - stay, 1 - up, 2 - down, 3 - left, 4 - right
        self.foraging_direction = random.choices(range(5), probs)[0] 
        self.square = self.model.forager_grid.direction_to_coord(self.group.location[0], self.group.location[1], self.foraging_direction) # 
        potential_new_group_index = self.model.grid_group_indices[self.square]
        
        # if the agent moved, and the agent decides to migrate
        if self.foraging_direction != 0 and self.migration_choice(potential_new_group_index):
            self.just_migrated = True
            self.model.forager_grid_next.add_agent(self.square) # add agent to forager_grid_next as a home agent
            self.switch_group(potential_new_group_index) # put agent into the new group
        else:
            self.just_migrated = False
            self.model.forager_grid_next.add_agent(self.square, self.foraging_direction) # add agent to forager_grid as a visiting agent
        
        return self.square
    
    # if there is a group on the agent's square, and that group fulfills the similarity criterion, then the agent
    # may migrate with probability p_swap
    def migration_choice(self, potential_group_index):
        # there is no group on the square the agent is on
        if potential_group_index == -1:
            return False 

        potential_group = self.model.groups[potential_group_index]

        # if the difference in group cooperation levels is allowable
        if abs(potential_group.avg_pct_cooperators - self.group.avg_pct_cooperators) <= self.model.similarity_threshold:
            return random.choices([True, False], weights = [self.model.p_swap, 1-self.model.p_swap])[0]
        else:
            return False
        
    # SpatialAgent, Float List -> Boolean 
    # agent decides whether to cooperate
    # returns whether agent cooperates
    # **tested**
    def choose_coop(self, rand=True, p_obs=None):
        # randomly select p_obs
        self.p_obs = random.uniform(0, 1) if p_obs is None else p_obs

        # if there are enough resources to pay the full coop cost, pay that, otherwise, pay what's left over
        foraging_cost = self.model.cost_distant if self.foraging_direction != 0 else 0
        coop_contrib = min(self.model.resources - foraging_cost, self.model.cost_coop)
        if coop_contrib < 0:
            coop_contrib = 0


        n_here = self.model.forager_grid.num_foragers(self.square)
        coop_cost = coop_contrib/n_here # perform adjustment according to agents present
        
        # choose whether to cooperate based on cost
        if self.model.year == 0:
            self.coop_strategy = False # (self.model.year == 0 and self.pi > 0.5) # POSSIBLY add a condition if p_obs = 0
        else:
            self.coop_strategy = (self.p_obs * self.group.avg_benefit) >= (coop_cost * (1 - self.pi))

        if rand:
            self.cooperate = random.choices([self.coop_strategy, not self.coop_strategy], weights=[1 - self.model.epsilon, self.model.epsilon], k=1)[0]
        else:
            self.cooperate = self.coop_strategy
        
        # calculate payoffs generated by the agent
        if self.cooperate:
            self.private_benefit = (self.model.resources - foraging_cost - coop_contrib)/n_here
            self.public_benefit= (coop_contrib / self.model.cost_coop) * self.model.benefit/n_here
        else: 
            self.private_benefit = (self.model.resources - foraging_cost)/n_here
            self.public_benefit = 0

        return self.cooperate
    
    # SpatialAgent -> boolean
    # returns true if the agent can pay the cost for survival and hasn't reached its lifespan, otherwise false
    # **tested**
    def survives(self):
        self.fitness -= self.model.cost_stayin_alive
        if self.fitness >= 0 and self.age <= self.lifespan:
            return True
        else:
            """
            if self.model.write_log:
                self.model.logger.population_change_stats['deaths'] += 1
                if self.age == 1:
                    self.model.logger.population_change_stats['first round deaths'] += 1
                elif self.age <= 5:
                    self.model.logger.population_change_stats['early deaths'] += 1
            """
            if self.model.write_log:
                self.model.logger.datadict["demographics"]["total"] += 1
                self.model.logger.datadict["demographics"]["migrated"] += (self.birth_group != self.group)
                self.model.logger.datadict["demographics"]["age"] += self.age
            
            return False

    # SpatialAgent, Int -> SpatialAgent
    # if the agent can pay the cost of reproducing, creates a child of this
    # agent and returns the child, otherwise returns None
    # **tested**
    def child(self, rand=True):
        if self.fitness - self.model.cost_repro >= self.model.cost_stayin_alive:
            self.fitness -= self.model.cost_repro
            
            # mutate learning style with model.p_mutation probability
            # perturb pi by some little amount
            if rand:
                learning_style = random.choices([self.learning, None], weights = [1-self.model.p_mutation, self.model.p_mutation])[0]
                pi = np.random.normal(self.pi, self.model.learning_rate * 0.2) if learning_style != "coop" else 1

                # there was a mutation
                if learning_style is None:
                    learning_style = random.choices(["static", "selfish", "civic", "coop"], weights=self.model.mut_distrib)[0]

            else:
                learning_style = self.learning
                pi = self.pi
            
            new_agent = SpatialAgent(self.model, self.group, pi=pi, mean_lifespan=self.model.mean_lifespan, learning=learning_style, rand=rand)
            
            # not tested
            """
            if self.model.write_log:
                self.model.logger.population_change_stats['agents born'] += 1
            """
            return new_agent
        else:
            return None

    # SpatialAgent ->
    # Puts agent in the group with index new_group_num; removes them from the current group
    # **tested**
    def switch_group(self, new_group_num):
        old_group = self.group
        new_group = self.model.groups[new_group_num]

        old_group.remove_agent(self)
        new_group.add_agent(self)


    # SpatialAgent -> void
    # the agent's pi value is updated based on its learning style and the
    # outcome of the previous round.
    # **tested**
    def learn(self, rand=True):
        inc = self.model.learning_rate # scale factor for changes
        
        
        if self.learning == "selfish":
            # if a selfish learner had a good outcome in the previous round, it will lean towards
            # following the same strategy (whether it's cooperating or defecting); otherwise, it will
            # lean towards the opposite strategy.

            # when avg_pi > pi and avg_fitness_diff > fitness_diff, pi should go up
            # when avg_pi > pi and avg_fitness_diff < fitness_diff, pi should go down
            # when avg_pi < pi and avg_fitness_diff > fitness_diff, pi should go down
            # when avg_pi < pi and avg_fitness_diff < fitness_diff, pi should go up
            
            if self.avg_fitness_diff + self.fitness_diff != 0 :
                vector = (self.avg_pi - self.pi)*(self.avg_fitness_diff - self.fitness_diff)/ self.avg_fitness_diff # CHANGED
            else:
                vector = 0
                    
            if rand:
                if vector == 0:
                    # if the avg_fitness_diff + fitness_diff are 0, they are both 0 since fitness_diff is always pos
                    # but we don't want to get stuck in this situation, so we add noise
                    vector = np.random.normal(loc=vector, scale=inc**2) # vector is roughly proportional to inc**2
                else:
                    vector = np.random.normal(loc=vector, scale=abs(vector)) # random noise, otherwise there is no way for avg_pi to differ from pi    

            self.pi += vector

        # a civic learner will move closer to an always-cooperator if others cooperated in the last round,
        # and will move closer to an always-defector if others defected in the last round.
        elif self.learning == "civic":
            if self.group.pct_cooperators > self.model.threshold:
                self.pi = (1 - inc)*self.pi + inc # weighted average of pi and 1
                
            else:
                self.pi = (1 - inc)*self.pi # weighted average of pi and 0
        
        # this is a discrete version of the civic learner
        # simply switches back and forth from an always-cooperator to an EV-maximizer
        elif self.learning == "switch":
            if self.group.pct_cooperators > self.model.threshold:
                self.pi = 1
            else:
                self.pi = 0
        
        # average the old avg pi with the new pi level
        self.avg_pi = self.model.present_weight*self.pi + (1 - self.model.present_weight)*self.avg_pi