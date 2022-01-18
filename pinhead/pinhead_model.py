from mesa import Model
from pinhead_agent import PinheadAgent, Strategy
from pinhead_group import PinheadGroup
from pinhead_scheduler import RandomActivationByLevel
from pinhead_logging import Logger
import random

from collections import defaultdict

class PinheadModel(Model):
    
    def __init__(
                    self, 
                    n=100, # agents per group 
                    g=20, # number of groups 
                    distrib={"miscreant": 0.25, "deceiver": 0.25, "citizen": 0.25, "saint": 0.25}, # percentage of each strategy
                    mut_distrib=None,
                    benefit=3.5, # benefit of cooperating
                    cost=1, # cost of cooperating 
                    fitness=3, # base amt of fitness received by each agent
                    p_mutation=0.01, # probability of an agent born switching strategy
                    p_con=0.1, # probability of conflict 
                    p_mig=0.1, # probability of migration
                    p_survive=0.8, # probability of surviving for agent of average fitness
                    epsilon=0.05, # probability of going against strategy
                    threshold=0.5, # level of cooperation necessary for citizens to cooperate
                    saintly_group=False, # should it have a single group that has high numbers of cooperators
                    years=1,
                    rand=True, # should random functions be random
                    print_stuff=False,
                    log_basic=False,
                    log_groups=False,
                    until_high=True,
                    until_low=False
                ):
        
        param_dict = {
            "n": n, 
            "g": g,
            "distrib": distrib,
            "benefit": benefit,
            "cost": cost,
            "fitness": fitness,
            "p_mutation": p_mutation,
            "p_con": p_con,
            "p_mig": p_mig,
            "p_survive": p_survive,
            "epsilon": epsilon,
            "threshold": threshold,
            "saintly_group": saintly_group,
            "years": years,
            "rand": rand
        }

      
        # population
        self.n = n
        self.g = g

        # cost and payoff
        self.average_payoff = 0
        self.benefit = benefit
        self.cost = cost
        self.fitness = fitness

        # current id number we're on
        self.curr_group_id = 0
        self.curr_indiv_id = 0

        # tables of individuals
        self.group_table = defaultdict(list)
        self.indiv_table = {}
        self.group_table_dead_indivs = defaultdict(list)
        self.agent_counts = {strat: 0 for strat in Strategy}

        # probabilities of various events 
        self.p_mutation = p_mutation
        self.p_con = p_con
        self.p_mig = p_mig
        self.p_survive = p_survive
        self.epsilon = epsilon
        self.threshold = threshold

        self.diff_sum = 0
        self.schedule = RandomActivationByLevel(self)
        self.years = years

        # logging
        self.log_basic = log_basic
        self.log_groups = log_groups

        if self.log_basic:
            if p_con != 1/13:
                config = f'y{years}_n{n}_g{g}_c{cost}_b{benefit}_pc{p_con}_pm{p_mutation}_ps{p_mig}_r{fitness}_t{threshold}distrib{distrib["citizen"]}_{distrib["saint"]}'
            else:
                config = f'y{years}_n{n}_g{g}_c{cost}_b{benefit}_pm{p_mutation}_ps{p_mig}_r{fitness}_t{threshold}distrib{distrib["citizen"]}_{distrib["saint"]}'
            self.logger = Logger(self, config, param_dict)

        self.distrib = distrib
        if mut_distrib is None:
            self.mut_distrib = distrib
        else:
            self.mut_distrib = mut_distrib

        self.strategies = self.initialize_strategies(distrib, rand)
        self.initialize_groups(saintly_group)

        self.print_stuff = print_stuff

        # termination fields. if until_low, runs until cooperation gets low, and the opposite for until_high
        self.until_low = until_low
        self.until_high = until_high
        self.can_terminate = False # set to true when ready to terminate

    """
    PinheadModel Boolean -> 
    creates groups and puts agents in them. 
    if saintly_group is true, creates a group of all citizens and saints
    **tested**
    """
    def initialize_groups(self, saintly_group):
        for i in range(self.g - saintly_group):
            g_id = "g" + str(self.curr_group_id)
            group_agent = PinheadGroup(g_id, self)
            # self.schedule.add(group_agent) # TODO does this ever get used

            for j in range(self.n):
                i_id = "i" + str(self.curr_indiv_id)
                if i == 0 and saintly_group and j % 2 == 0:
                    strategy = Strategy.CITIZEN
                elif i == 0 and saintly_group and j % 2 == 1:
                    strategy = Strategy.SAINT
                else:
                    strategy = self.strategies[self.curr_indiv_id]
            
                indiv_agent = PinheadAgent(i_id, self, strategy, group_agent, self.fitness)
        
    """
    PinheadModel Boolean -> 
    creates a list of strategies from the distribution of probabilities given
    **tested**
    """
    def initialize_strategies(self, distrib, rand):
        if rand:
            strategies = self.random.choices(
                        population=[Strategy.MISCREANT, Strategy.DECEIVER, Strategy.CITIZEN, Strategy.SAINT],
                        weights=[distrib["miscreant"], distrib["deceiver"], distrib["citizen"], distrib["saint"]],
                        k=self.n*self.g
                        )
        else:
            strategies = []
            possible_strats = [Strategy.MISCREANT, Strategy.DECEIVER, Strategy.CITIZEN, Strategy.SAINT]
            weights = [distrib["miscreant"], distrib["deceiver"], distrib["citizen"], distrib["saint"]]
            for i in range(self.g):
                for j in range(len(possible_strats)):
                    strategies += [possible_strats[j]] * (round(weights[j]*self.n))
        
        return strategies
    
    """
    EvoModel -> None
    Takes a step
    """
    def main(self):
        while self.schedule.year < self.years:
            self.loop()

            if self.can_terminate:
                break

    
    def loop(self):
        self.schedule.step()
        self.refresh_agent_total_counts()

        if self.print_stuff and self.schedule.year % 10 == 0:
            self.print_overall_composition()

    """
    EvoModel -> None
    Pairs all groups, has them fight with probability p_con, and replaces the loser with the
    winner
    **tested**
    """
    def fight_groups(self, rand=True):
        for group in self.group_table:
            group.fought = False
            group.enemy = None

        # pair up groups 
        groups1, groups2 = self.shuffle_and_pair(list(self.group_table.keys()), rand)

        for g1, g2 in zip(groups1, groups2):
            # save enemy for datacollector
            fight = self.random.choices([True, False], weights=[self.p_con, 1 - self.p_con])[0]

            if fight or (not rand):
                g1.enemy = g2
                g2.enemy = g1

                # store whether groups fought for datacollector
                g1.fought = True
                g2.fought = True

                w = self.fight(g1, g2, rand)
                if w:
                    self.replace_group(g1, g2)
                else:
                    self.replace_group(g2, g1)

    """
    EvoModel -> Boolean
    Gets two groups to fight, returns true if g1 wins
    **tested**
    """
    def fight(self, g1, g2, rand=True):
        F1 = g1.average_fitness
        F2 = g2.average_fitness
        
        if F1 + F2 > 0:
            p = F1/(F1 + F2)
        else:
            p = 0.5 # if both have 0 average fitness, then prob of winning is 1/2

        if rand:
            w = self.random.choices([True, False], weights=[p, 1 - p])[0]
        else:
            w = (F1 >= F2)

        return w

    """
    EvoModel GroupAgent GroupAgent -> GroupAgent
    Replaces loser with a new group with exactly the same agent strategy
    distribution as winner
    **tested**
    """
    def replace_group(self, winner, loser):
        loser.kill_group()
        new_group = PinheadGroup("g" + str(self.curr_group_id), self)

        for indiv in self.group_table[winner]:
            new_indiv = PinheadAgent("i" + str(self.curr_indiv_id), self, indiv.strategy, new_group, indiv.fitness)

        return new_group

    """
    EvoModel List -> List List
    Takes a list, shuffles it, and returns two separate lists. Interpret the ith element of
    first list as paired with ith element of second. Throws out one random element if the
    List length is odd
    **Tested**
    """
    def shuffle_and_pair(self, deck, rand=True):  
        if rand:
            self.random.shuffle(deck)

        midpoint = len(deck)//2
        first_half = deck[:midpoint]
        second_half = deck[midpoint:]

        if len(second_half) > len(first_half):
            second_half.pop()

        return first_half, second_half

    """
    EvoModel -> None
    Pairs individuals randomly, and then with probability p, switches their group
    membership
    **tested**
    """
    def recombine_groups(self, rand=True):
        indivs1, indivs2 = self.shuffle_and_pair(list(self.indiv_table.keys()), rand)

        for i1, i2 in zip(indivs1, indivs2):
            i1.migration_partner = i2
            i2.migration_partner = i1

            # if they're already in the same group, they don't move by swapping
            if self.indiv_table[i1] == self.indiv_table[i2]:
                continue

            move = self.random.choices([True, False], weights=[self.p_mig, 1 - self.p_mig])[0]

            if move or (not rand):
                i1.is_new_agent = True
                i1.migrated = True
                i2.is_new_agent = True
                i2.migrated = True
                self.swap_agents(i1, i2)

    """
    EvoModel -> IndivAgent IndivAgent
    Switches the group membership of indiv1 and indiv2
    **tested**
    """
    def swap_agents(self, indiv1, indiv2):
        group1 = self.indiv_table[indiv1]
        group2 = self.indiv_table[indiv2]

        self.indiv_table[indiv1] = group2
        self.indiv_table[indiv2] = group1

        self.group_table[group1].remove(indiv1)
        self.group_table[group1].append(indiv2)
        self.group_table[group2].remove(indiv2)
        self.group_table[group2].append(indiv1)

        group1.dec_strategy_count(indiv1.strategy)
        group1.inc_strategy_count(indiv2.strategy)
        group2.dec_strategy_count(indiv2.strategy)
        group2.inc_strategy_count(indiv1.strategy)

    def refresh_agent_total_counts(self):
        self.agent_counts = {strat: 0 for strat in Strategy}
        for group in self.group_table:
            for strat in Strategy:
                self.agent_counts[strat] += group.agent_counts[strat]

    """
    EvoModel -> None
    Prints out a representation of the groups
    **Tested**
    """
    def print_group_members_with_strategy(self):
        for group, indivs in self.group_table.items():
            print(group.unique_id + "- ", [indiv.unique_id + ": " + str(indiv.strategy.name) for indiv in indivs])

    """
    EvoModel -> None
    Prints out a representation of the groups
    **Tested**
    """
    def print_group_members(self):
        for group, indivs in self.group_table.items():
            print(group.unique_id + "- ", [indiv.unique_id for indiv in indivs])

    """
    EvoModel -> None
    Prints out the numbers of each strategy in each group
    **Tested**
    """
    def print_group_composition(self):

        strr = ""
        for group in self.group_table:
            strr += (group.unique_id + ": ")
            for strat in Strategy:
                strr += f'{strat.name}: {group.agent_counts[strat]}'

        print(strr)

    def print_overall_composition(self):
        self.refresh_agent_total_counts()

        print("Overall:")
        print("miscreants:", self.agent_counts[Strategy.MISCREANT], ",", self.agent_counts[Strategy.MISCREANT]/(self.n * self.g))
        print("deceivers:", self.agent_counts[Strategy.DECEIVER], ",", self.agent_counts[Strategy.DECEIVER]/(self.n * self.g))
        print("citizens:", self.agent_counts[Strategy.CITIZEN], ",", self.agent_counts[Strategy.CITIZEN]/(self.n * self.g))
        print("saints:", self.agent_counts[Strategy.SAINT], ",", self.agent_counts[Strategy.SAINT]/(self.n * self.g))



if __name__ == "__main__":
    pm = PinheadModel(
            n=40, 
            g=80,
            benefit=4,
            cost=1,
            fitness=1,
            distrib={"miscreant": 0.49, "deceiver": 0.49, "citizen": 0.01, "saint": 0.01},
            mut_distrib={"miscreant": 1/4, "deceiver": 1/4, "citizen": 1/4, "saint": 1/4},
            p_mutation=0.01,
            p_con=1/13,
            p_mig=0.03,
            p_survive=0.7,
            epsilon=0.05,
            threshold=0.5,
            print_stuff=True,
            log_basic=True,
            log_groups=False,
            years=10000
        )

    pm.main()