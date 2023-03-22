from mesa import Agent

import random
import numpy as np

from enum import Enum 

class Strategy(Enum):
    MISCREANT = 1
    DECEIVER = 2
    CITIZEN = 3
    SAINT = 4
    CIVIC = 5
    SELFISH = 6
    STATIC = 7


class PinheadAgent(Agent):
    def __init__(self, unique_id, model, strategy, group, starting_fitness=0, pi=0):
        super().__init__(unique_id, model)

        self.strategy = strategy
        self.fitness = starting_fitness
        self.avg_fitness = starting_fitness
        self.observed = False
        self.cooperates = False
        self.is_new_agent = True
        self.p_obs = 0
        self.base_fitness = self.model.fitness
        self.erred = False
        self.default_choice = False
        self.pi = pi
        self.avg_pi = pi

        self.model.curr_indiv_id = int(self.unique_id[1:]) + 1
        self.add_to_group(group)
        
        self.p_coop = 0 # testing purposes only
        self.migrated = False
        self.migration_partner = None
        self.s_prob = 0 # testing purposes only
        self.r_prob = 0 # testing purposes only
    
    """
    PinheadAgent -> 
    modifies variables so that the agent gets added to the group it's been assigned to
    **tested**
    """
    def add_to_group(self, group):
        self.model.indiv_table[self] = group
        self.model.group_table[group].append(self)
        group.inc_strategy_count(self.strategy)
        # self.model.schedule.add(self)

        self.model.agent_counts[self.strategy] += 1
    """
    PinheadAgent ->
    asks every agent to cooperate or defect, and decides if the agent gets caught
    **tested**
    """
    def step(self): 
        p = self.random.uniform(0,1)
        self.p_obs = p
        self.cooperates = self.make_choice()

        # need the fitness from the last round to be preserved for the learning step
        self.fitness = self.base_fitness

        self.observed = self.random.choices([True, False], weights = [p, 1-p])[0]
    
    """
    PinheadAgent -> Boolean
    returns whether the miscreant cooperates
    **tested**
    """
    def miscreant_choice(self, rand=True):
        return self.final_choice(self.model.epsilon, rand)

    """
    PinheadAgent -> Boolean
    Returns the choice of a deceiver based on the EV of cooperating and defecting
    **tested**
    """
    def deceiver_choice(self, rand=True):
        ev_coop, ev_def = self.calc_evs()
        p_coop = 1 - self.model.epsilon if ev_coop >= ev_def else self.model.epsilon
        return self.final_choice(p_coop, rand)

    """
    PinheadAgent -> Float Float
    calculates the EV of cooperating vs defecting based on the pubic benefit from the last round
    **tested**
    """
    def calc_evs(self):
        group = self.model.indiv_table[self]
        ev_coop = self.model.fitness + group.average_benefit - self.model.cost
        ev_def = self.p_obs*(self.model.fitness) + (1-self.p_obs)*(self.model.fitness + group.average_benefit)

        return ev_coop, ev_def

    """
    PinheadAgent -> Boolean
    returns the choice of a citizen, which is to cooperate 
    **tested**
    """
    def citizen_choice(self, rand=True):
        group = self.model.indiv_table[self]
        prop_cooperators = group.num_cooperated/self.model.n # FLAG
        # changed this because a threshold of 0.3 would be more equivalent to 0.6, since only half of agents are seen
        if prop_cooperators >= self.model.threshold:
            return self.saint_choice(rand)
        else:
            return self.deceiver_choice(rand)
    
    def learner_choice(self, rand=True):

        if self.strategy == Strategy.CIVIC:
            self.civic_learn()
        elif self.strategy == Strategy.SELFISH:
            self.selfish_learn(rand)
    
        
        group = self.model.indiv_table[self]
        if self.p_obs * group.average_benefit >= self.model.cost * (1 - self.pi):
            return self.saint_choice(rand)
        else:
            return self.miscreant_choice(rand)
    

    def selfish_learn(self, rand=True):
        vec = (self.pi - self.avg_pi)*(self.fitness - self.avg_fitness) / self.fitness 
        
        if rand:    
            vec = np.random.normal(loc=vec, scale=abs(vec)/2)

        self.pi += vec
        self.avg_pi = (1 - self.model.present_weight) * self.avg_pi + self.model.present_weight * self.pi
    
    def civic_learn(self):
        group = self.model.indiv_table[self]
        if group.num_cooperated / self.model.n >= self.model.threshold:
            self.pi = (1 - self.model.learning_rate)*self.pi + self.model.learning_rate # weighted average of pi and 1    
        else:
            self.pi = (1 - self.model.learning_rate)*self.pi

        self.avg_pi = (1 - self.model.present_weight) * self.avg_pi + self.model.present_weight * self.pi


    """
    PinheadAgent -> Boolean
    Feeds 1 - epsilon probability of cooperating to final 
    **tested**
    """
    def saint_choice(self, rand=True):
        return self.final_choice(1 - self.model.epsilon, rand)

    """
    PinheadAgent -> Boolean
    make a choice of whether to cooperate given a probability of cooperating
    **tested**
    """
    def final_choice(self, p_coop, rand=True):
        self.p_coop = p_coop
        self.default_choice = (p_coop > 0.5)
        if rand:
            return self.random.choices([True, False], weights = [p_coop, 1-p_coop])[0]
        else:
            return self.default_choice
    """
    PinheadAgent -> 
    removes a given agent from its group and the indiv table. Which means it died
    **tested**
    """
    def kill_indiv(self, remove_from_group=True):
        if remove_from_group:
            group = self.model.indiv_table[self]
            group.dec_strategy_count(self.strategy)
            self.model.group_table[group].remove(self)
        
        self.model.indiv_table.pop(self)

    """
    Agent decides whether or not to cooperate. Returns True if agent
    cooperates, False otherwise.
    **tested**
    """
    def make_choice(self, rand=True):
        if(self.strategy == Strategy.MISCREANT):
            return self.miscreant_choice(rand)
        elif(self.strategy == Strategy.DECEIVER):
            return self.deceiver_choice(rand)
        elif(self.strategy == Strategy.CITIZEN):
            return self.citizen_choice(rand)
        elif(self.strategy == Strategy.SAINT):
            return self.saint_choice(rand)
        elif(self.strategy in [Strategy.CIVIC, Strategy.SELFISH, Strategy.STATIC]):
            return self.learner_choice(rand)
        else:
            return self.new_choice(rand)