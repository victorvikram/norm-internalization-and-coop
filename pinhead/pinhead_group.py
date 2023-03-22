from mesa import Agent
import copy

from pinhead_agent import PinheadAgent, Strategy
import random
import numpy as np

import math

class PinheadGroup(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

        self.num_cooperated = 0
        self.num_seen_cooperating = 0
        self.average_benefit = 0
        self.average_fitness = 0

        self.model.curr_group_id = int(unique_id[1:]) + 1
        self.model.group_table[self] = []
        # self.model.schedule.add(self)

        self.agent_counts = {strat: 0 for strat in Strategy}
    
        self.fought = False
        self.enemy = None
    
    def step_distrib(self):
        n_indivs = self.model.n
        self.cooperation()
        self.average_fitness = self.model.fitness + self.num_cooperated*(self.model.benefit-self.model.cost)/n_indivs
    
    def step_reproduce(self, rand=True):
        n_deaths, _ = self.death()
        self.birth(n_deaths)

    """
    PinheadGroup ->
    Agents decide whether to cooperate 
    The benefits of cooperation are shared among all agents except those who
    were caught defecting.
    **tested**
    """
    def cooperation(self):
        self.num_cooperated = 0
        indivs = self.model.group_table[self]
        indivs_rewarded = []
        for indiv in indivs:
            if indiv.cooperates:
                indiv.fitness -= self.model.cost
                self.num_cooperated += 1
                indivs_rewarded.append(indiv)
            elif not indiv.observed:
                indivs_rewarded.append(indiv)

        if len(indivs_rewarded) > 0:
            self.average_benefit = self.num_cooperated*self.model.benefit/len(indivs_rewarded)

            for indiv in indivs_rewarded:
                indiv.fitness += self.average_benefit
        else:
            self.average_benefit = 0
        
        for indiv in indivs:
            indiv.avg_fitness = (0.8 * indiv.avg_fitness) + 0.2 * indiv.fitness
        

    """
    PinheadGroup ->
    Some agents are chosen to survive, and the rest die off. Agents survive according to 
    the survival rate, and are chosen to survive with a probability proportional to their fitness
    **tested**
    """
    def death(self):
        indivs = self.model.group_table[self].copy()
        n = len(indivs)
        new_agents = []
        total_fitness = sum([agent.fitness for agent in indivs])

        s_probs = []
        # calculate mortality probabilities 
        for indiv in indivs:
            indiv.s_prob = indiv.fitness/total_fitness if total_fitness > 0 else 1/n
            s_probs.append(indiv.s_prob)

        # decide whether each agent dies or survives
        surviving_indices = set(np.random.choice(range(n), size=math.floor(n*self.model.p_survive), replace=False, p=s_probs))

        n_deaths = 0
        for i, indiv in enumerate(indivs):
            if i in surviving_indices:
                new_agents.append(indiv)
            else:
                n_deaths += 1
                indiv.kill_indiv(remove_from_group=False)
        
        # reset agent list and agent counts
        self.model.group_table[self] = new_agents
        self.initialize_strategy_counts()
        
        return n_deaths, s_probs
    
    """
    PinheadGroup ->
    Agents are chosen to reproduce according to their fitnesses. These offspring replace
    the dead agents
    **tested**
    """
    def birth(self, n_deaths):
        indivs = self.model.group_table[self]
        deaths = n_deaths
        survivals = len(indivs)

        total_fitness = sum([agent.fitness for agent in indivs])
        r_probs = []

        for indiv in indivs:
            indiv.r_prob = (indiv.fitness/total_fitness)
            r_probs.append(indiv.r_prob)
 
        reproducer_indices = self.random.choices(range(survivals), weights =r_probs, k=deaths)
        
        for ind in reproducer_indices:
            reproducing_indiv = indivs[ind]
            mutate = np.random.choice([True, False], p=[self.model.p_mutation, 1 - self.model.p_mutation])
            
            if mutate:
                strategy = np.random.choice([Strategy.MISCREANT, Strategy.DECEIVER, Strategy.CITIZEN, Strategy.SAINT, Strategy.CIVIC, Strategy.SELFISH, Strategy.STATIC],
                    p=[self.model.mut_distrib["miscreant"], 
                        self.model.mut_distrib["deceiver"], 
                        self.model.mut_distrib["citizen"], 
                        self.model.mut_distrib["saint"],
                        self.model.mut_distrib["civic"], 
                        self.model.mut_distrib["selfish"],
                        self.model.mut_distrib["static"]])
            else:
                strategy = reproducing_indiv.strategy

            new_pi = np.random.normal(loc=reproducing_indiv.pi, scale=0.05)
            
            new_indiv = PinheadAgent("i" + str(self.model.curr_indiv_id), self.model, strategy, self, reproducing_indiv.fitness, pi=new_pi)
    
    def initialize_strategy_counts(self):
        self.agent_counts = {strat: 0 for strat in Strategy}

        for indiv in self.model.group_table[self]:
            self.agent_counts[indiv.strategy] += 1

    def inc_strategy_count(self, strategy):
        self.agent_counts[strategy] += 1

    def dec_strategy_count(self, strategy):
        self.agent_counts[strategy] -= 1
    
    def print_agents(self):
        indivs = self.model.group_table[self]
        print([indiv.unique_id for indiv in indivs])

    def kill_group(self):
        dead_indivs = copy.copy(self.model.group_table[self])

        for dead_indiv in dead_indivs:
            dead_indiv.kill_indiv(remove_from_group=False) # group is dying anyway

        self.model.group_table.pop(self)