import unittest
import numpy as np
import math

from pinhead_model import PinheadModel
from pinhead_agent import PinheadAgent
from pinhead_agent import Strategy

class PinheadTests(unittest.TestCase):
    
    # PinheadModel tests --------------------------------------------------
    def testInvariants(self):
        pm = PinheadModel(n=100, g=100, p_survive=0.6, p_con=0, p_mig=0.25, benefit=4, cost=1.5, fitness=2, epsilon=0.15, threshold=0.3, print_stuff=True)
        steps = 100
        for i in range(steps):
            pm.loop()

            # indiv_table
            self.assertEqual(len(pm.indiv_table), 10000)
            for indiv in pm.indiv_table:
                for grp, agent_list in pm.group_table.items():
                    if grp == pm.indiv_table[indiv]:
                        self.assertIn(indiv, agent_list)
                    else:
                        self.assertNotIn(indiv, agent_list)
            
            # group_table and agent_counts
            self.assertEqual(len(pm.group_table), 100)
            for grp, agent_list in pm.group_table.items():
                self.assertEqual(len(agent_list), 100)
                strategy_counts = {strat: 0 for strat in Strategy}
                for agent in agent_list:
                    self.assertIn(agent, pm.indiv_table)
                    self.assertEqual(pm.indiv_table[agent], grp)
                    strategy_counts[agent.strategy] += 1
                
                self.assertEqual(strategy_counts, grp.agent_counts)
            
            # cooperation
            expected_counter = {strat: 0 for strat in Strategy}
            total_counter = {strat: 0 for strat in Strategy}

            for group, agent_list in pm.group_table.items():
                for agent in agent_list:
                    total_counter[agent.strategy] += 1
                    if agent.strategy == Strategy.MISCREANT:
                        expected_counter[Strategy.MISCREANT] += (agent.cooperates == False)
                    elif agent.strategy == Strategy.DECEIVER:
                        if agent.is_new_agent:
                            expected_counter[Strategy.DECEIVER] += (agent.cooperates == False)
                        else:
                            expectation = (agent.p_obs * group.old_average_benefit >= 1.5)
                            expected_counter[Strategy.DECEIVER] += (agent.cooperates == expectation)
                    elif agent.strategy == Strategy.CITIZEN:
                        if agent.is_new_agent:
                            expected_counter[Strategy.CITIZEN] += (agent.cooperates == True)
                        else:
                            expectation = (group.old_num_cooperated/100 >= 0.3)
                            expected_counter[Strategy.CITIZEN] += (agent.cooperates == expectation)
                    elif agent.strategy == Strategy.SAINT:
                        expected_counter[Strategy.SAINT] += (agent.cooperates == True)
            
            self.assertGreater(expected_counter[Strategy.SAINT], 0.85 * total_counter[Strategy.SAINT] - 250) # PROB
            self.assertLess(expected_counter[Strategy.SAINT], 0.85 * total_counter[Strategy.SAINT] + 250) # PROB

            self.assertGreater(expected_counter[Strategy.MISCREANT], 0.85 * total_counter[Strategy.MISCREANT] - 250) # PROB
            self.assertLess(expected_counter[Strategy.MISCREANT], 0.85 * total_counter[Strategy.MISCREANT] + 250) # PROB

            self.assertGreater(expected_counter[Strategy.CITIZEN], 0.85 * total_counter[Strategy.CITIZEN] - 250) # PROB
            self.assertLess(expected_counter[Strategy.CITIZEN], 0.85 * total_counter[Strategy.CITIZEN] + 250) # PROB

            self.assertGreater(expected_counter[Strategy.DECEIVER], 0.85 * total_counter[Strategy.DECEIVER] - 250) # PROB
            self.assertLess(expected_counter[Strategy.DECEIVER], 0.85 * total_counter[Strategy.DECEIVER] + 250) # PROB

            # distrib
            for group, agent_list in pm.group_table.items():
                # aggregate number of cooperators
                cooperator_count = 0
                caught_count = 0
                total_fitness = 0
                for indiv in agent_list:
                    total_fitness += indiv.fitness
                    if indiv.observed and not indiv.cooperates:
                        caught_count += 1
                    elif indiv.cooperates:
                        cooperator_count += 1

                self.assertAlmostEqual(group.average_fitness, total_fitness/100)
                self.assertEqual(group.num_cooperated, cooperator_count)
                self.assertAlmostEqual(group.average_benefit, (cooperator_count*4)/(100 - caught_count))

                for indiv in agent_list:
                    if indiv.observed and not indiv.cooperates:
                        self.assertEqual(indiv.fitness, 2)
                    elif not indiv.observed and not indiv.cooperates:
                        self.assertEqual(indiv.fitness, 2 + group.average_benefit)
                    else:
                        # the agent cooperated 
                        self.assertEqual(indiv.fitness, 0.5 + group.average_benefit)

            # death and migration
            if i != 0:
                total_survived = 0
                total_fitness_survived = 0
                same_group_counter = 0
                just_migrated_counter = 0
                is_new_agent_counter = 0
                for indiv in pm.indiv_table:
                    if indiv in old_indiv_table:
                        total_survived += 1
                        total_fitness_survived += indiv.old_fitness

                        if pm.indiv_table[indiv] == old_indiv_table[indiv]:
                            same_group_counter += 1
                            self.assertFalse(indiv.is_new_agent)
                            self.assertFalse(indiv.migrated)
                        else:
                            self.assertTrue(indiv.migrated)
                            self.assertTrue(indiv.is_new_agent)
                    else:
                        self.assertTrue(indiv.is_new_agent)
                    
                    is_new_agent_counter += indiv.is_new_agent
                    just_migrated_counter += indiv.migrated
                    
                
                total_fitness_killed = 0
                total_killed = 0
                for indiv in old_indiv_table:
                    if indiv not in pm.indiv_table:
                        total_fitness_killed += indiv.fitness
                        total_killed += 1
                
                # check that the average fitness of dead is lower than the survived
                self.assertGreater(total_fitness_survived, total_fitness_killed/total_killed)

                # 60% should have died
                self.assertEqual(total_survived, 6000)

                # all the newly born agents (4000) in addition to the old agents that migrated (~1500) should be marked as new agents
                self.assertGreater(is_new_agent_counter, 5500 - 250)
                self.assertLess(is_new_agent_counter, 5500 + 250) 

                # 75% (4500) of the remaining agents should be in the same groups
                self.assertGreater(same_group_counter, 4500 - 250) # PROB
                self.assertLess(same_group_counter, 4500 + 250) # PROB
                
                # 25% of the agents hsould have just migrated
                self.assertGreater(just_migrated_counter, 2500 - 250) # PROB
                self.assertLess(just_migrated_counter, 2500 + 250) # PROB


            # store some old values
            for indiv in pm.indiv_table:
                indiv.old_fitness = indiv.fitness
            
            for group in pm.group_table:
                group.old_average_benefit = group.average_benefit
                group.old_num_cooperated = group.num_cooperated
            
            old_indiv_table = pm.indiv_table.copy()  
        

        pm = PinheadModel(n=5, g=1000, p_survive=0.8, p_con=0.6, p_mig=0.4, benefit=4, cost=1.5, fitness=2, epsilon=0.25, threshold=0.5, print_stuff=True)
        
        less_fit_groups_die_counter = 0
        for i in range(steps):
            pm.loop()

            # indiv_table
            self.assertEqual(len(pm.indiv_table), 5000)
            for indiv in pm.indiv_table:
                for grp, agent_list in pm.group_table.items():
                    if grp == pm.indiv_table[indiv]:
                        self.assertIn(indiv, agent_list)
                    else:
                        self.assertNotIn(indiv, agent_list)
            
            # group_table and agent_counts
            self.assertEqual(len(pm.group_table), 1000)
            for grp, agent_list in pm.group_table.items():
                self.assertEqual(len(agent_list), 5)
                strategy_counts = {strat: 0 for strat in Strategy}
                for agent in agent_list:
                    self.assertIn(agent, pm.indiv_table)
                    self.assertEqual(pm.indiv_table[agent], grp)
                    strategy_counts[agent.strategy] += 1
                
                self.assertEqual(strategy_counts, grp.agent_counts)
            
            # cooperation
            expected_counter = {strat: 0 for strat in Strategy}
            total_counter = {strat: 0 for strat in Strategy}

            for group, agent_list in pm.group_table.items():
                for agent in agent_list:
                    total_counter[agent.strategy] += 1
                    if agent.strategy == Strategy.MISCREANT:
                        expected_counter[Strategy.MISCREANT] += (agent.cooperates == False)
                    elif agent.strategy == Strategy.DECEIVER:
                        if agent.is_new_agent:
                            expected_counter[Strategy.DECEIVER] += (agent.cooperates == False)
                        else:
                            expectation = (agent.p_obs * group.old_average_benefit >= 1.5)
                            expected_counter[Strategy.DECEIVER] += (agent.cooperates == expectation)
                    elif agent.strategy == Strategy.CITIZEN:
                        if agent.is_new_agent:
                            expected_counter[Strategy.CITIZEN] += (agent.cooperates == True)
                        else:
                            expectation = (group.old_num_cooperated/5 >= 0.5)
                            expected_counter[Strategy.CITIZEN] += (agent.cooperates == expectation)
                    elif agent.strategy == Strategy.SAINT:
                        expected_counter[Strategy.SAINT] += (agent.cooperates == True)
            
            self.assertGreater(expected_counter[Strategy.SAINT], 0.75 * total_counter[Strategy.SAINT] - 125) # PROB
            self.assertLess(expected_counter[Strategy.SAINT], 0.75 * total_counter[Strategy.SAINT] + 125) # PROB

            self.assertGreater(expected_counter[Strategy.MISCREANT], 0.75 * total_counter[Strategy.MISCREANT] - 125) # PROB
            self.assertLess(expected_counter[Strategy.MISCREANT], 0.75 * total_counter[Strategy.MISCREANT] + 125) # PROB

            self.assertGreater(expected_counter[Strategy.CITIZEN], 0.75 * total_counter[Strategy.CITIZEN] - 125) # PROB
            self.assertLess(expected_counter[Strategy.CITIZEN], 0.75 * total_counter[Strategy.CITIZEN] + 125) # PROB

            self.assertGreater(expected_counter[Strategy.DECEIVER], 0.75 * total_counter[Strategy.DECEIVER] - 125) # PROB
            self.assertLess(expected_counter[Strategy.DECEIVER], 0.75 * total_counter[Strategy.DECEIVER] + 125) # PROB

            # distrib
            for group, agent_list in pm.group_table.items():
                # aggregate number of cooperators
                cooperator_count = 0
                caught_count = 0
                total_fitness = 0
                for indiv in agent_list:
                    total_fitness += indiv.fitness
                    if indiv.observed and not indiv.cooperates:
                        caught_count += 1
                    elif indiv.cooperates:
                        cooperator_count += 1

                self.assertAlmostEqual(group.average_fitness, total_fitness/5)
                self.assertEqual(group.num_cooperated, cooperator_count)
                
                if caught_count != 5:
                    self.assertAlmostEqual(group.average_benefit, (cooperator_count*4)/(5 - caught_count))
                else:
                    self.assertEqual(group.average_benefit, 0)

                for indiv in agent_list:
                    if indiv.observed and not indiv.cooperates:
                        self.assertEqual(indiv.fitness, 2)
                    elif not indiv.observed and not indiv.cooperates:
                        self.assertEqual(indiv.fitness, 2 + group.average_benefit)
                    else:
                        # the agent cooperated 
                        self.assertEqual(indiv.fitness, 0.5 + group.average_benefit)

            # death, conflict, and migration
            if i != 0:
                total_survived_counter = 0
                fitness_survived_counter = 0
                same_group_counter = 0
                is_new_agent_counter = 0
                just_migrated_counter = 0
                for indiv in pm.indiv_table:
                    if indiv in old_indiv_table:
                        total_survived_counter += 1
                        fitness_survived_counter += indiv.old_fitness
                        
                        if pm.indiv_table[indiv] == old_indiv_table[indiv]:
                            same_group_counter += 1
                            self.assertFalse(indiv.is_new_agent)
                            self.assertFalse(indiv.migrated)
                        else:
                            self.assertTrue(indiv.migrated)
                            self.assertTrue(indiv.is_new_agent)
                    else:
                        self.assertTrue(indiv.is_new_agent)
                    
                    is_new_agent_counter += indiv.is_new_agent
                    just_migrated_counter += indiv.migrated

                total_killed_counter = 0
                fitness_killed_counter = 0
                for indiv in old_indiv_table:
                    # if the agent died from individual causes, and NOT its group dying
                    if indiv not in pm.indiv_table and old_indiv_table[indiv] in pm.group_table:
                        total_killed_counter += 1
                        fitness_killed_counter += indiv.fitness

                self.assertGreater(fitness_survived_counter/total_survived_counter, fitness_killed_counter/total_killed_counter)

                survived_group_counter = 0
                survived_group_fitness = 0
                for group in pm.group_table:
                    if group in old_group_table:
                        survived_group_counter += 1
                        survived_group_fitness += group.old_average_fitness
                
                killed_group_counter = 0
                killed_group_fitness = 0
                for group in old_group_table:
                    if group not in pm.group_table:
                        killed_group_counter += 1
                        killed_group_fitness += group.average_fitness 
                
                fight_counter = sum([group.fought for group in pm.group_table])
                
                # check that the correct numbers of groups survived and died, and that the dead groups have a lower average fitness
                self.assertEqual(killed_group_counter, fight_counter)
                less_fit_groups_die_counter += (survived_group_fitness/survived_group_counter >= killed_group_fitness/killed_group_counter)
                        
                # check that around the expected number of groups fought (that is 3000, since half the fighting groups died) 
                self.assertGreater(fight_counter, 300 - 75) # PROB
                self.assertLess(fight_counter, 300 + 75) # PROB

                # check that the expected number of agents died
                old_agents_died_in_battle = 0.8 * 5 * fight_counter # each fighting group fought a group with 5 agents, 0.8 of whom were old_agents
                expected_old_agents_remaining = 4000 - old_agents_died_in_battle
                # 20% should have died in the mortality step, along with a number equal to the population of groups that fought.
                self.assertEqual(total_survived_counter, expected_old_agents_remaining)

                # the number of new_agents should be the number of new agents plus the number of old agents that migrated
                self.assertGreater(is_new_agent_counter, (1000 + old_agents_died_in_battle) + (4000 - old_agents_died_in_battle)*0.4 - 150)
                self.assertLess(is_new_agent_counter, (1000 + old_agents_died_in_battle) + (4000 - old_agents_died_in_battle)*0.4 + 150)

                # 60% of the remaining agents should be in the same groups as they were before
                self.assertGreater(same_group_counter, 0.6*expected_old_agents_remaining - 150) # PROB
                self.assertLess(same_group_counter, 0.6*expected_old_agents_remaining + 150) # PROB

                # 40% of all the agents should have just migrated
                self.assertGreater(just_migrated_counter, 2000 - 200) # PROB
                self.assertLess(just_migrated_counter, 2000 + 200) # PROB

            # store some old values
            for indiv in pm.indiv_table:
                indiv.old_fitness = indiv.fitness
            
            for group in pm.group_table:
                group.old_average_fitness = group.average_fitness
                group.old_average_benefit = group.average_benefit
                group.old_num_cooperated = group.num_cooperated
            
            old_indiv_table = pm.indiv_table.copy()
            old_group_table = {grp: agents_list.copy() for grp, agents_list in pm.group_table.items()}

        self.assertGreater(less_fit_groups_die_counter, 0.8*steps)

    def testInitializeStrategies(self):
        # check that strategy counts are near the expected value
        pm = PinheadModel(n=1000, g=10, distrib={"miscreant": 0.4, "deceiver": 0.3, "citizen": 0.2, "saint": 0.1})

        counts = {strat: 0 for strat in Strategy}
        for strat in pm.strategies:
            counts[strat] += 1
        
        self.assertTrue(counts[Strategy.MISCREANT] > 3800 and counts[Strategy.MISCREANT] < 4200) # PROB 
        self.assertTrue(counts[Strategy.DECEIVER] > 2800 and counts[Strategy.DECEIVER] < 3200) # PROB 
        self.assertTrue(counts[Strategy.CITIZEN] > 1800 and counts[Strategy.CITIZEN] < 2200) # PROB 
        self.assertTrue(counts[Strategy.SAINT] > 800 and counts[Strategy.SAINT] < 1200) # PROB 


    def testInitializeGroups(self):
        # no saintly group
        pm = PinheadModel(n=25, g=400, distrib={"miscreant": 0.3, "deceiver": 0.4, "citizen": 0.1, "saint": 0.2})
        
        self.assertTrue(pm.agent_counts[Strategy.MISCREANT] > 2800 and pm.agent_counts[Strategy.MISCREANT] < 3200) # PROB
        self.assertTrue(pm.agent_counts[Strategy.DECEIVER] > 3800 and pm.agent_counts[Strategy.MISCREANT] < 4200) # PROB
        self.assertTrue(pm.agent_counts[Strategy.CITIZEN] > 800 and pm.agent_counts[Strategy.CITIZEN] < 1200) # PROB
        self.assertTrue(pm.agent_counts[Strategy.SAINT] > 1800 and pm.agent_counts[Strategy.SAINT] < 2200) # PROB
        
        self.assertEqual(len(pm.group_table), 400)
        self.assertEqual(len(pm.indiv_table), 10000)

        # with a saintly group
        pm = PinheadModel(n=100, g=51, distrib={"miscreant": 0.1, "deceiver": 0.2, "citizen": 0.3, "saint": 0.4}, saintly_group=True)

        # check saintly group counts
        for group in pm.group_table:
            if group.unique_id == "g0":
                saint_group = group
        
        self.assertEqual(saint_group.agent_counts[Strategy.CITIZEN], 50)
        self.assertEqual(saint_group.agent_counts[Strategy.SAINT], 50)
        self.assertEqual(saint_group.agent_counts[Strategy.DECEIVER], 0)
        self.assertEqual(saint_group.agent_counts[Strategy.MISCREANT], 0)

        citizens = 0
        saints = 0
        for agent in pm.group_table[saint_group]:
            if agent.strategy == Strategy.CITIZEN:
                citizens += 1
            if agent.strategy == Strategy.SAINT:
                saints += 1
        
        self.assertEqual(citizens, saint_group.agent_counts[Strategy.CITIZEN])
        self.assertEqual(saints, saint_group.agent_counts[Strategy.SAINT])

        # check the sum from the other groups
        self.assertTrue(pm.agent_counts[Strategy.MISCREANT] > 400 and pm.agent_counts[Strategy.MISCREANT] < 600) # PROB
        self.assertTrue(pm.agent_counts[Strategy.DECEIVER] > 900 and pm.agent_counts[Strategy.DECEIVER] < 1100) # PROB
        self.assertTrue(pm.agent_counts[Strategy.CITIZEN] > 1400 and pm.agent_counts[Strategy.CITIZEN] < 1600) # PROB subtract 50 to account for saintly group
        self.assertTrue(pm.agent_counts[Strategy.SAINT] > 1900 and pm.agent_counts[Strategy.SAINT] < 2100) # PROB subtract 50 to account for saintly group
    
    def testShuffleAndPair(self):
        pm = PinheadModel()
        
        # test for even list
        lst1 = [1, 2, 3, 4, 5, 6]
        fh, sh = pm.shuffle_and_pair(lst1)
        self.assertEqual(len(fh), 3)
        self.assertEqual(len(sh), 3)

        for item in fh:
            self.assertNotIn(item, sh)

        # test for odd list
        lst2 = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
        fh, sh = pm.shuffle_and_pair(lst2)
        self.assertEqual(len(fh), 4)
        self.assertEqual(len(sh), 4)

        for item in fh:
            self.assertNotIn(item, sh)
        
    def testFight(self):
        pm = PinheadModel(g=10)

        # test with different probability combos
        
        # 30 - 70
        group1 = list(pm.group_table.keys())[9]
        group2 = list(pm.group_table.keys())[3]

        group1.average_fitness = 6
        group2.average_fitness = 14

        # 90 - 10
        group3 = list(pm.group_table.keys())[0]
        group4 = list(pm.group_table.keys())[7]
        group3.average_fitness = 27
        group4.average_fitness = 3

        # 50 - 50
        group5 = list(pm.group_table.keys())[1]
        group6 = list(pm.group_table.keys())[8]
        group5.average_fitness = 17
        group6.average_fitness = 17
 
        group1_wins = 0
        group3_wins = 0
        group5_wins = 0
        for i in range(10000):
            group1_wins += pm.fight(group1, group2)
            group3_wins += pm.fight(group3, group4)
            group5_wins += pm.fight(group5, group6)
        
        self.assertTrue(group1_wins > 2500 and group1_wins < 3500) # PROB
        self.assertTrue(group3_wins > 8500 and group3_wins < 9500) # PROB
        self.assertTrue(group5_wins > 4500 and group5_wins < 5500) # PROB

        # deterministic tests
        self.assertEqual(pm.fight(group1, group2, rand=False), False)
        self.assertEqual(pm.fight(group3, group4, rand=False), True)
        self.assertEqual(pm.fight(group5, group6, rand=False), True)

    def testReplaceGroup(self):
        pm = PinheadModel(n=20, g=6)  
        
        # two sets of groups to replace
        group0 = list(pm.group_table.keys())[0]
        group2 = list(pm.group_table.keys())[2]

        group3 = list(pm.group_table.keys())[3]
        group5 = list(pm.group_table.keys())[5]

        # replace group2 with group0
        group2_agents = pm.group_table[group2].copy()
        group6 = pm.replace_group(group0, group2)

        # group_table checks
        self.assertIn(group0, pm.group_table)
        self.assertIn(group6, pm.group_table)
        self.assertNotIn(group2, pm.group_table)

        # indiv_table checks
        for agent in group2_agents:
            self.assertNotIn(agent, pm.indiv_table)
        for agent in pm.group_table[group6]:
            self.assertIn(agent, pm.indiv_table)
            self.assertEqual(pm.indiv_table[agent], group6)
        
        # check that the composition of the new group is the same as the old one 
        self.assertEqual(len(pm.group_table[group6]), len(pm.group_table[group0]))
        self.assertEqual(group6.agent_counts[Strategy.MISCREANT], group0.agent_counts[Strategy.MISCREANT])
        self.assertEqual(group6.agent_counts[Strategy.DECEIVER], group0.agent_counts[Strategy.DECEIVER])
        self.assertEqual(group6.agent_counts[Strategy.CITIZEN], group0.agent_counts[Strategy.CITIZEN])
        self.assertEqual(group6.agent_counts[Strategy.SAINT], group0.agent_counts[Strategy.SAINT])

        # replace group3 with group5 
        group3_agents = pm.group_table[group3].copy()
        group7 = pm.replace_group(group5, group3)

        # group_table checks
        self.assertIn(group5, pm.group_table)
        self.assertIn(group7, pm.group_table)
        self.assertNotIn(group3, pm.group_table)

        # indiv_table checks
        for agent in group3_agents:
            self.assertNotIn(agent, pm.indiv_table)
        for agent in pm.group_table[group7]:
            self.assertIn(agent, pm.indiv_table)
            self.assertEqual(pm.indiv_table[agent], group7)
        
        # check composition is the same 
        self.assertEqual(len(pm.group_table[group7]), len(pm.group_table[group5]))
        self.assertEqual(group7.agent_counts[Strategy.MISCREANT], group5.agent_counts[Strategy.MISCREANT])
        self.assertEqual(group7.agent_counts[Strategy.DECEIVER], group5.agent_counts[Strategy.DECEIVER])
        self.assertEqual(group7.agent_counts[Strategy.CITIZEN], group5.agent_counts[Strategy.CITIZEN])
        self.assertEqual(group7.agent_counts[Strategy.SAINT], group5.agent_counts[Strategy.SAINT])

    def testFightGroups(self):
        # check that a fight occurs about p_con fraction of the time, no repeat opponents
        pm = PinheadModel(n=5, g=10000, p_con=0.3)
        old_groups = list(pm.group_table.keys())
        groups_fought_against = set()
        groups_fought = 0
        pm.fight_groups()

        # check no repeat groups, and that opponents agree on enemy
        for group in old_groups:
            self.assertEqual(group.enemy is not None, group.fought) # enemy is not none <=> fought
            
            if group.fought: 
                groups_fought += 1
                self.assertTrue(group.enemy not in groups_fought_against) # no two groups should have the same opponent
                self.assertEqual(group, group.enemy.enemy)
                groups_fought_against.add(group.enemy)
        
        self.assertTrue(groups_fought > 2500 and groups_fought < 3500)

        # check that expected groups end up replacing and being replaced
        pm = PinheadModel(n=5, g=103)
        old_groups = list(pm.group_table.keys())
        
        # set fitnesses so that even groups win
        for i, group in enumerate(old_groups):
            if i % 2 == 0:
                group.average_fitness = 20
            else:
                group.average_fitness = 13
        
        pm.fight_groups(rand=False)

        # even groups should still be in it, odd groups should have died
        for i, group in enumerate(old_groups):
            if i % 2 == 0:
                self.assertIn(group, pm.group_table)
            else:
                self.assertNotIn(group, pm.group_table)

            # check that opponents are as expected
            if i < 51:
                self.assertEqual(group.enemy, old_groups[i + 51])
                self.assertTrue(group.fought)
            elif i < 102:
                self.assertEqual(group.enemy, old_groups[i - 51])
                self.assertTrue(group.fought)
            elif i == 102:
                self.assertIsNone(group.enemy)
                self.assertFalse(group.fought)
        
        # check that the new_groups list has correct properties
        new_groups = list(pm.group_table.keys())
        self.assertEqual(len(new_groups), 103)
        
        for i, group in enumerate(new_groups):
            id_number = int(group.unique_id[1:])
            
            if id_number < 102:
                # check that corresponding groups have correct composition
                if id_number < 51:
                    corresponding_group_id = "g" + str(id_number + 103) 
                elif id_number < 102:
                    corresponding_group_id = "g" + str(id_number + 52)
                
                self.assertEqual(id_number % 2, 0)
                
                # find corresponding group, check composition
                found = False
                for potential_group in pm.group_table:
                    if potential_group.unique_id == corresponding_group_id:
                        found = True 
                        corresponding_group = potential_group
                self.assertTrue(found)

                self.assertEqual(group.agent_counts[Strategy.MISCREANT], corresponding_group.agent_counts[Strategy.MISCREANT])
                self.assertEqual(group.agent_counts[Strategy.DECEIVER], corresponding_group.agent_counts[Strategy.DECEIVER])
                self.assertEqual(group.agent_counts[Strategy.CITIZEN], corresponding_group.agent_counts[Strategy.CITIZEN])
                self.assertEqual(group.agent_counts[Strategy.SAINT], corresponding_group.agent_counts[Strategy.SAINT])
            

    def testSwapAgents(self):
        pm = PinheadModel(n=500, g=100)

        for i in range(50):
            agents = list(pm.indiv_table.keys())
            agent1 = np.random.choice(agents)
            agents.remove(agent1)
            agent2 = np.random.choice(agents)

            group1 = pm.indiv_table[agent1]
            group2 = pm.indiv_table[agent2]

            if group1 == group2:
                continue

            pm.swap_agents(agent1, agent2)

            self.assertIn(agent1, pm.group_table[group2])
            self.assertIn(agent2, pm.group_table[group1])

            self.assertNotIn(agent1, pm.group_table[group1])
            self.assertNotIn(agent2, pm.group_table[group2])

            self.assertEqual(pm.indiv_table[agent1], group2)
            self.assertEqual(pm.indiv_table[agent2], group1)

            group1_counts = {strat: 0 for strat in Strategy}
            for agent in pm.group_table[group1]:
                group1_counts[agent.strategy] += 1 
            
            self.assertEqual(group1_counts[Strategy.MISCREANT], group1.agent_counts[Strategy.MISCREANT])
            self.assertEqual(group1_counts[Strategy.DECEIVER], group1.agent_counts[Strategy.DECEIVER])
            self.assertEqual(group1_counts[Strategy.CITIZEN], group1.agent_counts[Strategy.CITIZEN])
            self.assertEqual(group1_counts[Strategy.SAINT], group1.agent_counts[Strategy.SAINT])
            
            group2_counts = {strat: 0 for strat in Strategy}
            for agent in pm.group_table[group2]:
                group2_counts[agent.strategy] += 1 
            
            self.assertEqual(group2_counts[Strategy.MISCREANT], group2.agent_counts[Strategy.MISCREANT])
            self.assertEqual(group2_counts[Strategy.DECEIVER], group2.agent_counts[Strategy.DECEIVER])
            self.assertEqual(group2_counts[Strategy.CITIZEN], group2.agent_counts[Strategy.CITIZEN])
            self.assertEqual(group2_counts[Strategy.SAINT], group2.agent_counts[Strategy.SAINT])

    def testRecombineGroups(self):
        # probabilistic test to check that about p_mig fractions of agents migrate
        pm = PinheadModel(n=100, g=100, p_mig=0.15)

        for agent in pm.indiv_table:
            agent.is_new_agent = False

        old_indiv_table = pm.indiv_table.copy()

        pm.recombine_groups()
        
        # count agents that moved in two different ways
        moved_count_table = 0
        moved_count_migrated = 0
        for agent in pm.indiv_table:
            if pm.indiv_table[agent] != old_indiv_table[agent]:
                moved_count_table += 1
            self.assertEqual(agent.is_new_agent, agent.migrated)
            if agent.migrated:
                moved_count_migrated += 1
        
        # check that quantities are equal
        self.assertEqual(moved_count_migrated, moved_count_table)    
        self.assertTrue(moved_count_table > 1250 and moved_count_table < 1750) # PROB
        
        # non-probabilistic test to make sure everyone ended up in the correct group
        pm = PinheadModel(n=6, g=4)
        groups = list(pm.group_table.keys())

        old_group_table = {group: agents.copy() for group, agents in pm.group_table.items()}
        
        pm.recombine_groups(rand=False)

        # check tht agents end up where expected
        for i, group_agents in enumerate(old_group_table.values()):
            for agent in group_agents:
                self.assertTrue(agent.is_new_agent)
                self.assertTrue(agent.migrated)
                corresp_group = i + 2 if i < 2 else i - 2
                self.assertEqual(pm.indiv_table[agent], groups[corresp_group]) # agents are reflected

    # PinheadAgent tests --------------------------------------------------
    def testAddToGroup(self):
        pm = PinheadModel(n=20, g=10)

        for i in range(20):
            group = np.random.choice(list(pm.group_table.keys()))

            # remember old quantities
            old_group_length = len(pm.group_table[group])
            old_group_counts = {strat: count for strat, count in group.agent_counts.items()}
            old_model_counts = {strat: count for strat, count in pm.agent_counts.items()}

            # choose a strategy randomly, make agent
            strat = np.random.choice([Strategy.MISCREANT, Strategy.DECEIVER, Strategy.CITIZEN, Strategy.SAINT])
            new_agent = PinheadAgent("i" + str(pm.curr_indiv_id), pm, strat, group)

            # check that data structures have been updated
            self.assertEqual(old_group_length + 1, len(pm.group_table[group]))
            self.assertIn(new_agent, pm.group_table[group])
            self.assertEqual(group, pm.indiv_table[new_agent])

            # check that strategy counts have been updated
            for strategy in Strategy:
                if strat == strategy:
                    self.assertEqual(old_group_counts[strategy] + 1, group.agent_counts[strategy])
                    self.assertEqual(old_model_counts[strategy] + 1, pm.agent_counts[strategy])        
                else:
                    self.assertEqual(old_group_counts[strategy], group.agent_counts[strategy])
                    self.assertEqual(old_model_counts[strategy], pm.agent_counts[strategy])    

    def testFinalChoice(self):
        # probabilistic test
        probs = [0, 0.2, 0.4, 0.7, 0.9, 1]
        pm = PinheadModel(n=100, g=100)

        for prob in probs: 
            counter = 0    
            for agent in pm.indiv_table.keys():
                counter += agent.final_choice(prob)
            
            expected_counter = 10000*prob
            self.assertTrue(expected_counter - 250 < counter and counter < expected_counter + 250) # PROB

            if prob in [0, 1]:
                self.assertEqual(expected_counter, counter)        
            
        # non random test
        probs = [0, 0.2, 0.4, 0.7, 0.9, 1]
        for prob in probs:
            agent = np.random.choice(list(pm.indiv_table.keys()))
            cooperated = agent.final_choice(prob, rand=False)

            if prob >= 0.5:
                self.assertTrue(cooperated)
            else: 
                self.assertFalse(cooperated)
    
    def testSaintChoice(self):
        pm = PinheadModel(n=50, g=200)

        # probabilistic tests with different epsilons
        epsilons = [0.1, 0.2, 0.3]  
        for epsilon in epsilons:
            pm.epsilon = epsilon

            counter = 0
            for agent in pm.indiv_table.keys():
                counter += agent.saint_choice()
            
            expected_counter = (1 - epsilon) * 10000
            self.assertTrue(expected_counter - 250 < counter and counter < expected_counter + 250) # PROB
    
    def testMiscreantChoice(self):
        pm = PinheadModel(n=50, g=200)

        # probabilistic tests with different epsilons
        epsilons = [0.1, 0.2, 0.3]  
        for epsilon in epsilons:
            pm.epsilon = epsilon

            counter = 0
            for agent in pm.indiv_table.keys():
                counter += agent.miscreant_choice()
            
            expected_counter = epsilon * 10000
            self.assertTrue(expected_counter - 250 < counter and counter < expected_counter + 250) # PROB

    def testDeceiverChoice(self):
        # non probabilistic tests
        pm = PinheadModel(n=10, g=5, epsilon=0.2, cost=10, fitness=15)
        p_obses = [0,0,0,0,0,0,0,0,0,0,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,
                    0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0,0.2,0.4,0.6,0.8]
        avg_benefits = [7,9,11,13,15,17,19,21,23,25,7,9,11,13,15,17,19,21,23,25,7,9,11,13,15,17,19,21,23,25,7,9,11,13,
                        15,17,19,21,23,25,7,9,11,13,15,17,19,21,23,25,17,15,7,12,6]
        is_new_agents = [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
                            False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
                            False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
                            False,False,True,True,True,True,True]
        cooperates = [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
                False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,
                True,True,True,True,True,False,False,False,True,True,True,True,True,True,True,False,False,False,False,False]
        
        for i, agent in enumerate(pm.indiv_table.keys()):
            agent.p_obs = p_obses[i]
            pm.indiv_table[agent].average_benefit = avg_benefits[i]
            agent.is_new_agent = is_new_agents[i]

            cooperated = agent.deceiver_choice(rand=False)
            self.assertEqual(cooperates[i], cooperated)
        
        # probabilistic tests
        # - first round / not first round
        # - high / low obs 
        # - high / low avg_benefit 
        # - high / low cost 

        p_obses = [0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6]
        is_new_agents = [False,False,False,False,True,True,True,True,False,False,False,False,True,True,True,True]
        avg_benefits = [12,12,29,29,12,12,29,29,12,12,29,29,12,12,29,29]
        costs = [4,8,4,8,4,8,4,8,4,8,4,8,4,8,4,8]

        cooperates = [False, False, True, False, False, False, False, False, True, False , True, True, False, False, False, False]

        for p_obs, is_new_agent, average_benefit, cost, expected_coop in zip(p_obses, is_new_agents, avg_benefits, costs, cooperates):
            pm = PinheadModel(n=500, g=20, cost=cost, epsilon=0.15)

            for group in pm.group_table.keys():
                group.average_benefit = average_benefit
            
            as_expected = 0
            for agent in pm.indiv_table.keys():
                agent.p_obs = p_obs
                agent.is_new_agent = is_new_agent

                # should be as expected 15% of the time 
                cooperate = agent.deceiver_choice()
                as_expected += (cooperate == expected_coop)  

            self.assertTrue(8250 < as_expected and as_expected < 8750) # PROB
        
    def testCalcEvs(self):
        pm = PinheadModel(g=20, cost=2, fitness=5)

        groups = list(pm.group_table.keys())

        for group in groups:
            group_index = int(group.unique_id[1:])
            group.average_benefit = group_index/2
        
        for agent in pm.indiv_table.keys():
            agent.p_obs = np.random.uniform(0, 1)
            group = pm.indiv_table[agent]
            group_index = int(group.unique_id[1:])
            ev_coop, ev_def = agent.calc_evs()

            self.assertAlmostEqual(ev_coop, 5 + group_index/2 - 2)
            self.assertAlmostEqual(ev_def, 5 + (1 - agent.p_obs)*group_index/2)

    def testCitizenChoice(self):
        # non-random trial
        num_seen_cooperatings = [4,4,7,7,12,12,18,18,4,4,7,7,12,12,18,18]
        thresholds = [0.32,0.32,0.32,0.32,0.32,0.32,0.32,0.32,0.57,0.57,0.57,0.57,0.57,0.57,0.57,0.57]
        p_obses = [0.3,0.6,0.3,0.6,0.3,0.6,0.3,0.6,0.3,0.6,0.3,0.6,0.3,0.6,0.3,0.6]

        avg_benefits = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

        cooperates = [False,True,True,True,True,True,True,True,False,True,False,True,True,True,True,True]

        pm = PinheadModel(n=20, g=1, cost=0.5)
        agents = list(pm.indiv_table.keys())[:16]
        
        for agent, p_obs, num_seen_cooperating, threshold, avg_benefit, expected_cooperate in zip(agents, p_obses, num_seen_cooperatings, thresholds, avg_benefits, cooperates):
            pm.threshold = threshold
            pm.indiv_table[agent].num_cooperated = num_seen_cooperating
            pm.indiv_table[agent].average_benefit = avg_benefit

            agent.p_obs = p_obs
            agent.is_new_agent = False

            # non-first-round trial 
            agent.is_new_agent = False 
            cooperates = agent.citizen_choice(rand=False)
            self.assertEqual(cooperates, expected_cooperate)

            # first-round trial, acts like a saint
            agent.is_new_agent = True 
            self.assertEqual(agent.citizen_choice(rand=False), True)
        
        # random trial
        num_seen_cooperatings = [6,6,6,6,6,6,6,6,10,10,10,10,10,10,10,10,18,18,18,18,18,18,18,18,27,27,27,27,27,27,27,27]
        thresholds = [0.2,0.2,0.2,0.2,0.55,0.55,0.55,0.55,0.2,0.2,0.2,0.2,0.55,0.55,0.55,0.55,0.2,0.2,0.2,0.2,0.55,0.55,0.55,0.55,0.2,0.2,0.2,0.2,0.55,0.55,0.55,0.55]
        p_obses = [0.3,0.3,0.7,0.7,0.3,0.3,0.7,0.7,0.3,0.3,0.7,0.7,0.3,0.3,0.7,0.7,0.3,0.3,0.7,0.7,0.3,0.3,0.7,0.7,0.3,0.3,0.7,0.7,0.3,0.3,0.7,0.7]

        avg_benefits = [2,1,0.5,1,2,1,0.5,1,2,1,0.5,1,2,1,0.5,1,2,1,0.5,1,2,1,0.5,1,2,1,0.5,1,2,1,0.5,1]

        cooperates = [True,False,False,True,True,False,False,True,True,True,True,True,True,False,False,True,True,True,True,True,True,False,False,True,True,True,True,True,True,True,True,True]

        for p_obs, num_seen_cooperating, threshold, avg_benefit, expected_cooperate in zip(p_obses, num_seen_cooperatings, thresholds, avg_benefits, cooperates):
            pm = PinheadModel(n=40, g=250, cost=0.5, threshold=threshold, epsilon=0.25)

            expected_counter = 0
            expected_counter_saint = 0

            for agent in pm.indiv_table.keys():
                pm.indiv_table[agent].num_cooperated = num_seen_cooperating
                pm.indiv_table[agent].average_benefit = avg_benefit

                agent.p_obs = p_obs
                agent.is_new_agent = False

                # non-first-round trial 
                agent.is_new_agent = False 
                cooperates = agent.citizen_choice(rand=True)
                expected_counter += (cooperates == expected_cooperate)

                # first-round trial, acts like a saint
                agent.is_new_agent = True 
                cooperates = agent.citizen_choice(rand=True)
                expected_counter_saint += (cooperates == True)

            self.assertTrue(7250 < expected_counter and expected_counter < 7750)
            self.assertTrue(7250 < expected_counter_saint and expected_counter_saint < 7750)


    def testKillIndiv(self):
        pm = PinheadModel(n=100, g=30, p_survive=1)
        for group, agents in pm.group_table.items():
            agents_copy = agents.copy()
            for i, agent in enumerate(agents_copy):
                if i % 5 == 0:
                    old_length = len(agents)
                    old_strategy_count = group.agent_counts[agent.strategy]
                    agent.kill_indiv()
                    self.assertNotIn(agent, pm.group_table[group])
                    self.assertNotIn(agent, pm.indiv_table)
                    self.assertEqual(group.agent_counts[agent.strategy], old_strategy_count - 1)
                    self.assertEqual(len(pm.group_table[group]), old_length - 1)
        
        for group, agents in pm.group_table.items():
            self.assertEqual(len(agents), 80)
        
        pm.loop()

        for group, agents in pm.group_table.items():
            agents_copy = agents.copy()
            for i, agent in enumerate(agents_copy):
                if i % 8 == 0:
                    old_length = len(agents)
                    old_strategy_count = group.agent_counts[agent.strategy]
                    agent.kill_indiv()
                    self.assertNotIn(agent, pm.group_table[group])
                    self.assertNotIn(agent, pm.indiv_table)
                    self.assertEqual(group.agent_counts[agent.strategy], old_strategy_count - 1)
                    self.assertEqual(len(pm.group_table[group]), old_length - 1)
        
        for group, agents in pm.group_table.items():
            self.assertEqual(len(agents), 70)
        
    def testMakeChoice(self):
        pm = PinheadModel(n=100, g=30, p_survive=0.9)

        # check that we are using correct agent functions
        for agent in pm.indiv_table:
            strat = agent.strategy
            if strat == Strategy.MISCREANT:
                self.assertEqual(agent.make_choice(rand=False), agent.miscreant_choice(rand=False))
            elif strat == Strategy.DECEIVER:
                self.assertEqual(agent.make_choice(rand=False), agent.deceiver_choice(rand=False))
            elif strat == Strategy.CITIZEN:
                self.assertEqual(agent.make_choice(rand=False), agent.citizen_choice(rand=False))
            elif strat == Strategy.SAINT:
                self.assertEqual(agent.make_choice(rand=False), agent.saint_choice(rand=False))

        for i in range(5):
            pm.loop()
        
        # test that not  all of them are new agents, so we get both arms of the 
        new_agents = 0
        for agent in pm.indiv_table:
            new_agents += agent.is_new_agent
        
        self.assertLess(new_agents, 1000) # PROB

        # check agent functions once we've gotten into non-first-round territory
        for agent in pm.indiv_table:
            strat = agent.strategy
            if strat == Strategy.MISCREANT:
                self.assertEqual(agent.make_choice(rand=False), agent.miscreant_choice(rand=False))
            elif strat == Strategy.DECEIVER:
                self.assertEqual(agent.make_choice(rand=False), agent.deceiver_choice(rand=False))
            elif strat == Strategy.CITIZEN:
                self.assertEqual(agent.make_choice(rand=False), agent.citizen_choice(rand=False))
            elif strat == Strategy.SAINT:
                self.assertEqual(agent.make_choice(rand=False), agent.saint_choice(rand=False))
 
    def testIndivStep(self):
        pm = PinheadModel(n=500, g=100)

        # test five times at different points in the run
        for i in range(4):
            # check that we are drawing the correct distribution
            p_obs_list = []
            for agent in pm.indiv_table:
                agent.step()
                p_obs_list.append(agent.p_obs)
            
            p_obs_mean = sum(p_obs_list)/len(p_obs_list)
            p_obs_var = sum([(p_obs - p_obs_mean)**2 for p_obs in p_obs_list])/len(p_obs_list)
            
            # mean should be 0.5
            self.assertGreater(p_obs_mean, 0.48) # PROB
            self.assertLess(p_obs_mean, 0.52) # PROB

            # var should be 1/12 since this is a uniform distro
            self.assertGreater(p_obs_var, 1/12 - 0.01) # PROB
            self.assertLess(p_obs_var, 1/12 + 0.01) # PROB

            # check that agents in different buckets are caught at the corrects
            caught_count = {0: 0, 0.2: 0, 0.4: 0, 0.6: 0, 0.8: 0}
            total_count = {0: 0, 0.2: 0, 0.4: 0, 0.6: 0, 0.8: 0}
            
            for agent in pm.indiv_table:

                # incrementing buckets
                if agent.p_obs < 0.2:
                    caught_count[0] += agent.observed
                    total_count[0] += 1
                elif agent.p_obs < 0.4:
                    caught_count[0.2]+= agent.observed
                    total_count[0.2] += 1
                elif agent.p_obs < 0.6:
                    caught_count[0.4] += agent.observed
                    total_count[0.4] += 1
                elif agent.p_obs < 0.8:
                    caught_count[0.6] += agent.observed
                    total_count[0.6] += 1
                elif agent.p_obs < 1:
                    caught_count[0.8] += agent.observed
                    total_count[0.8] += 1

            # check distribution is uniform
            for k in [0, 0.2, 0.4, 0.6, 0.8]:
                self.assertGreater(total_count[k], 9000) # PROB
                self.assertLess(total_count[k], 11000) # PROB

            # check that caught rates match
            self.assertGreater(caught_count[0]/total_count[0], 0.07) # PROB
            self.assertLess(caught_count[0]/total_count[0], 0.13) # PROB

            self.assertGreater(caught_count[0.2]/total_count[0.2], 0.27) # PROB
            self.assertLess(caught_count[0.2]/total_count[0.2], 0.33) # PROB

            self.assertGreater(caught_count[0.4]/total_count[0.4], 0.47) # PROB
            self.assertLess(caught_count[0.4]/total_count[0.4], 0.53) # PROB

            self.assertGreater(caught_count[0.6]/total_count[0.6], 0.67) # PROB
            self.assertLess(caught_count[0.6]/total_count[0.6], 0.73) # PROB

            self.assertGreater(caught_count[0.8]/total_count[0.8], 0.87) # PROB
            self.assertLess(caught_count[0.8]/total_count[0.8], 0.93) # PROB

            for j in range(i*3):
                pm.loop()
    
    # PinheadGroup Tests ----------------------------------------------------------
    def testCooperation(self):
        pm = PinheadModel(n=24, g=30, benefit=4.25, cost=1.4, fitness=2)

        for k in range(10):
            # j % 3 == 0 - balanced group
            # j % 3 == 1 - lots of uncaught defectors
            # j % 3 == 2 - lots of cooperators
            for j, (group, agents) in enumerate(pm.group_table.items()):
                for i, agent in enumerate(agents):
                    agent.fitness = 2
                    # balanced
                    if j % 3 == 0:
                        if i % 4 == 0:
                            agent.observed = True 
                            agent.cooperates = True
                        elif i % 4 == 1:
                            agent.observed = True 
                            agent.cooperates = False
                        elif i % 4 == 2:
                            agent.observed = False 
                            agent.cooperates = True
                        elif i % 4 == 3:
                            agent.observed = False 
                            agent.cooperates = False
                    # lots of unobserved defectors
                    elif j % 3 == 1:
                        if i % 6 == 0:
                            agent.observed = True 
                            agent.cooperates = True
                        elif i % 6 == 1:
                            agent.observed = True 
                            agent.cooperates = False
                        elif i % 6 == 2:
                            agent.observed = False 
                            agent.cooperates = True
                        else:
                            agent.observed = False 
                            agent.cooperates = False
                    # lots of cooperators
                    elif j % 3 == 2:
                        if i % 8 == 0:
                            agent.observed = False 
                            agent.cooperates = False
                        elif i % 8 == 1:
                            agent.observed = True 
                            agent.cooperates = False
                        elif i % 8 == 2:
                            agent.observed = False 
                            agent.cooperates = True
                        else:
                            agent.observed = True 
                            agent.cooperates = True

                group.cooperation()
                
                if j % 3 == 0:
                    self.assertEqual(group.num_cooperated, 12)
                    self.assertAlmostEqual(group.average_benefit, 12*4.25/18) # 12 agents cooperated, 18 were not caught defecting    
                elif j % 3 == 1:
                    self.assertEqual(group.num_cooperated, 8)
                    self.assertAlmostEqual(group.average_benefit, 8*4.25/20) # 8 agents cooperated, 20 were not caught defecting
                elif j % 3 == 2:
                    self.assertEqual(group.num_cooperated, 18)
                    self.assertAlmostEqual(group.average_benefit, 18*4.25/21) # 18 agents cooperated, 21 were not caught defecting
                    
                for i, agent in enumerate(agents):
                    if j % 3 == 0:
                        if i % 4 == 0:
                            self.assertAlmostEqual(agent.fitness, 2 - 1.4 + 12*4.25/18)
                        elif i % 4 == 1:
                            self.assertAlmostEqual(agent.fitness, 2)
                        elif i % 4 == 2:
                            self.assertAlmostEqual(agent.fitness, 2 - 1.4 + 12*4.25/18)
                        elif i % 4 == 3:
                            self.assertAlmostEqual(agent.fitness, 2 + 12*4.25/18)
                    # lots of unobserved defectors
                    elif j % 3 == 1:
                        if i % 6 == 0:
                            self.assertAlmostEqual(agent.fitness, 2 - 1.4 + 8*4.25/20)
                        elif i % 6 == 1:
                            self.assertAlmostEqual(agent.fitness, 2)
                        elif i % 6 == 2:
                            self.assertAlmostEqual(agent.fitness, 2 - 1.4 + 8*4.25/20)
                        else:
                            self.assertAlmostEqual(agent.fitness, 2 + 8*4.25/20)
                    # lots of cooperators
                    elif j % 3 == 2:
                        if i % 8 == 0:
                            self.assertAlmostEqual(agent.fitness, 2 + 18*4.25/21)
                        elif i % 8 == 1:
                            self.assertAlmostEqual(agent.fitness, 2)
                        elif i % 8 == 2:
                            self.assertAlmostEqual(agent.fitness, 2 - 1.4 + 18*4.25/21)
                        else:
                            self.assertAlmostEqual(agent.fitness, 2 - 1.4 + 18*4.25/21)
            
            # repeat the test for various stages in the process
            for t in range(k * 3):
                pm.loop()
        
    def testDeath(self):
        pm = PinheadModel(n=10, g=5, p_survive=0.8)

        indiv_fitnesses = [4,12,6,4,5,16,19,3,0,1,3,5,11,2,18,16,6,8,9,20,2,6,6,8,10,4,10,3,7,2,0,2,3,1,4,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0]
        expected_probs = [0.057142857,0.171428571,0.085714286,0.057142857,0.071428571,0.228571429,0.271428571,0.042857143,0,0.014285714,0.030612245,0.051020408,0.112244898,0.020408163,0.183673469,0.163265306,0.06122449,0.081632653,0.091836735,0.204081633,0.034482759,0.103448276,0.103448276,0.137931034,0.172413793,0.068965517,0.172413793,0.051724138,0.120689655,0.034482759,0,0.142857143,0.214285714,0.071428571,0.285714286,0,0.071428571,0.071428571,0.071428571,0.071428571,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]

        for i, group in enumerate(pm.group_table):

            for j, agent in enumerate(pm.group_table[group]):
                agent.fitness = indiv_fitnesses[i*10 + j]

            
            old_agent_list = pm.group_table[group].copy()

            _, s_probs = group.death()
            
            # check that s_probs are as expected 
            for j, agent in enumerate(old_agent_list):
                self.assertEqual(s_probs[j], agent.s_prob)
                self.assertAlmostEqual(agent.s_prob, expected_probs[i*10 + j])
        
        # large group test
        pm = PinheadModel(n=100000, g=5, p_survive=0.05)
        subgroup_fitnesses = [[2, 1, 0.6, 0.4], [8, 4, 2, 1], [4, 3, 2, 1], [3, 2.9, 2.1, 2], [5, 4, 1, 0]]
        approximate_percentages = [[0.5, 0.25, 0.125, 0.125], [8/15, 4/15, 2/15, 1/15], [4/10, 3/10, 2/10, 1/10], [0.3, 0.3, 0.2, 0.2], [0.5, 0.4, 0.1, 0]]

        for i, (group, agents) in enumerate(pm.group_table.items()):
            for j, agent in enumerate(agents):
                index = math.floor(j / 25000)
                agent.fitness = subgroup_fitnesses[i][index]
            
            n_deaths, _ = group.death()

            self.assertEqual(n_deaths, 95000)
            self.assertEqual(len(pm.group_table[group]), 5000)            

            # count up the agents of each fitness level
            counts = {fitness: 0 for fitness in subgroup_fitnesses[i]}
            for agent in pm.group_table[group]:
                counts[agent.fitness] += 1
                self.assertIn(agent, pm.indiv_table)
            
            # count individuals in indiv_table corresponding 
            agents_in_group = 0
            for agnt, grp in pm.indiv_table.items():
                agents_in_group += (grp == group)
            
            self.assertEqual(agents_in_group, 5000)
            
            # check that the prevalence of each agent in the survived group is as expected
            self.assertGreater(counts[subgroup_fitnesses[i][0]], 5000*approximate_percentages[i][0] - 250) # PROB
            self.assertLess(counts[subgroup_fitnesses[i][0]], 5000*approximate_percentages[i][0] + 250) # PROB

            self.assertGreater(counts[subgroup_fitnesses[i][1]], 5000*approximate_percentages[i][1] - 250) # PROB
            self.assertLess(counts[subgroup_fitnesses[i][1]], 5000*approximate_percentages[i][1] + 250) # PROB

            self.assertGreater(counts[subgroup_fitnesses[i][2]], 5000*approximate_percentages[i][2] - 250) # PROB
            self.assertLess(counts[subgroup_fitnesses[i][2]], 5000*approximate_percentages[i][2] + 250) # PROB

            self.assertGreater(counts[subgroup_fitnesses[i][3]], 5000*approximate_percentages[i][3] - 250) # PROB
            self.assertLess(counts[subgroup_fitnesses[i][3]], 5000*approximate_percentages[i][3] + 250) # PROB 
            
        self.assertEqual(len(pm.indiv_table), 5000*5)

        subgroup_fitnesses = [[8, 2], [9, 1], [3, 7], [10, 5], [7, 9]]

        # 5 different types of groups
        for i in range(5):
            # 50 trials for each
            expected_count = 0
            for k in range(200):

                pm = PinheadModel(n=100, g=1, p_survive=0.6)
                group = list(pm.group_table.keys())[0]
                agents = pm.group_table[group]

                for j, agent in enumerate(agents):
                    index = math.floor(j / 50)
                    agent.fitness = subgroup_fitnesses[i][index]
                
                n_deaths, _ = group.death()

                self.assertEqual(n_deaths, 40)
                self.assertEqual(len(pm.group_table[group]), 60)            

                # count up the agents of each fitness level
                counts = {fitness: 0 for fitness in subgroup_fitnesses[i]}
                for agent in pm.group_table[group]:
                    counts[agent.fitness] += 1
                    self.assertIn(agent, pm.indiv_table)
                
                # count individuals in indiv_table corresponding 
                agents_in_group = 0
                for agnt, grp in pm.indiv_table.items():
                    agents_in_group += (grp == group)
                
                self.assertEqual(agents_in_group, 60)

                # count up the agents of each fitness level
                counts = {fitness: 0 for fitness in subgroup_fitnesses[i]}
                for agent in pm.group_table[group]:
                    counts[agent.fitness] += 1
                    self.assertIn(agent, pm.indiv_table)

                # if counts agree with the fitnesses, increment the expectation counter
                if counts[subgroup_fitnesses[i][0]] > counts[subgroup_fitnesses[i][1]] and subgroup_fitnesses[i][0] > subgroup_fitnesses[i][1]:
                    expected_count += 1
                elif counts[subgroup_fitnesses[i][1]] > counts[subgroup_fitnesses[i][0]] and subgroup_fitnesses[i][1] > subgroup_fitnesses[i][0]:
                    expected_count += 1
            
            # check that the expected outcome happened more than 70% of the time
            self.assertGreater(expected_count, 140) # PROB

    def testBirth(self):
        pm = PinheadModel(n=12, g=4)

        # check that r_probs are calculated correctly
        fitnesses = [5,2,3,4,3,5,0,4,4,0,3,4,7,4,0,8,1,7,0,8,8,2,4,5,1,6,0,13,5,11,0,8,7,6,3,0,5,10,19,6,10,15,19,3,13,12,3,8]
        expected_probs = [0.135135135,0.054054054,0.081081081,0.108108108,0.081081081,0.135135135,0,0.108108108,0.108108108,0,0.081081081,0.108108108,0.12962963,0.074074074,0,0.148148148,0.018518519,0.12962963,0,0.148148148,0.148148148,0.037037037,0.074074074,0.092592593,0.016666667,0.1,0,0.216666667,0.083333333,0.183333333,0,0.133333333,0.116666667,0.1,0.05,0,0.040650407,0.081300813,0.154471545,0.048780488,0.081300813,0.12195122,0.154471545,0.024390244,0.105691057,0.097560976,0.024390244,0.06504065]

        for i, (group, agents) in enumerate(pm.group_table.items()):
            for j, agent in enumerate(agents):
                agent.fitness = fitnesses[i * 12 + j]
            
            group.birth(20)

            self.assertEqual(len(pm.group_table[group]), 32)

            for j, agent in enumerate(agents):
                # don't check the new agents
                if j < 12:
                    self.assertAlmostEqual(agent.r_prob, expected_probs[i * 12 + j])
        
        # check that agents reproduce in proportion to their fitnesses
        pm = PinheadModel(n=12000, g=4, p_mutation=0)
        splits = [[3000, 6000, 9000, 12000], [9000, 12000], [4000, 8000, 12000], [2000, 4000, 6000, 8000, 10000, 12000]]
        fitnesses = [[4, 3, 2, 1], [3, 9], [5, 4, 1], [3.5, 2.5, 2, 1.5, 0.5, 0]]

        for i, (group, agents) in enumerate(pm.group_table.items()):
            curr_index = 0
            for j, agent in enumerate(agents):
                if j >= splits[i][curr_index]:
                    curr_index += 1

                agent.fitness = fitnesses[i][curr_index]
        
            group.birth(5000)

            self.assertEqual(len(pm.group_table[group]), 17000)

            fitness_counts = {fitness: 0 for fitness in fitnesses[i]}
            for agent in pm.group_table[group]:
                fitness_counts[agent.fitness] += 1
            
            total_fitness = 0
            for bucket, fitness in enumerate(fitnesses[i]):
                initial_pop = splits[i][bucket] if bucket == 0 else splits[i][bucket] - splits[i][bucket - 1]
                total_fitness += initial_pop * fitness

            for bucket, fitness in enumerate(fitnesses[i]):
                initial_pop = splits[i][bucket] if bucket == 0 else splits[i][bucket] - splits[i][bucket - 1]
                expected_ratio = initial_pop*fitness / total_fitness
                expected_additional_pop = 5000*expected_ratio
                expectation = initial_pop + expected_additional_pop
                self.assertGreater(fitness_counts[fitness], expectation - 100)
                self.assertLess(fitness_counts[fitness], expectation + 100)
        
        # check that around the correct numbers of agents mutate
        pm = PinheadModel(n=1000, g=10, distrib={"miscreant": 0, "deceiver": 0, "citizen": 1, "saint": 0}, 
                            mut_distrib={"miscreant": 0.25, "deceiver": 0.25, "citizen": 0.25, "saint": 0.25}, 
                            p_mutation=0.6)

        for group in pm.group_table:
            group.birth(1000)
        
        counts = {strat: 0 for strat in Strategy}

        for agent in pm.indiv_table:
            counts[agent.strategy] += 1
        
        self.assertEqual(sum(counts.values()), 20000)
        
        # there should be all of the original 10000 citizens, in addition to 40% of the babies. the remaining mutated 60%
        # should be divided evenly among all agent types
        self.assertGreater(counts[Strategy.CITIZEN], 10000 + 4000 + 1500 - 150)
        self.assertLess(counts[Strategy.CITIZEN], 10000 + 4000 + 1500 + 150)

        self.assertGreater(counts[Strategy.SAINT], 1500 - 150)
        self.assertLess(counts[Strategy.SAINT], 1500 + 150)

        self.assertGreater(counts[Strategy.MISCREANT], 1500 - 150)
        self.assertLess(counts[Strategy.MISCREANT], 1500 + 150)

        self.assertGreater(counts[Strategy.DECEIVER], 1500 - 150)
        self.assertLess(counts[Strategy.DECEIVER], 1500 + 150)
        
        for i in range(5):
            pm.loop()
        
        for indiv in pm.indiv_table:
            indiv.fitness = 4
            indiv.strategy = Strategy.DECEIVER
        
        for group in pm.group_table:
            group.birth(2000)

        counts = {strat: 0 for strat in Strategy}

        for agent in pm.indiv_table:
            counts[agent.strategy] += 1
        
        self.assertEqual(sum(counts.values()), 40000)
        
        # there should be all of the original 20000 deceivers, in addition to 40% of the babies. the remaining mutated 60%
        # should be divided evenly among all agent types
        self.assertGreater(counts[Strategy.DECEIVER], 20000 + 8000 + 3000 - 200)
        self.assertLess(counts[Strategy.DECEIVER], 20000 + 8000 + 3000 + 200)

        self.assertGreater(counts[Strategy.SAINT], 3000 - 200)
        self.assertLess(counts[Strategy.SAINT], 3000 + 200)

        self.assertGreater(counts[Strategy.MISCREANT], 3000 - 200)
        self.assertLess(counts[Strategy.MISCREANT], 3000 + 200)

        self.assertGreater(counts[Strategy.CITIZEN], 3000 - 200)
        self.assertLess(counts[Strategy.CITIZEN], 3000 + 200)






if __name__ == "__main__":
    unittest.main()