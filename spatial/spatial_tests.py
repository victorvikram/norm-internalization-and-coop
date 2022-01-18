import unittest
import numpy as np
import random
import math
from collections import defaultdict

from spatial_model import SpatialModel
from spatial_group import SpatialGroup
from spatial_grid import SpatialGrid
from spatial_agent import SpatialAgent

# probabilistic tests are marked with PROB, they may fail
class TestSpatialModelNew(unittest.TestCase):

    # tests that all data structures are in sync
    # sm.forager_grid, sm.grid_group_indices, sm.groups, and each individual group and agent
    def testInvariants(self):
        
        # set up all the different configurations
        ns = [10, 20, 20, 20, 60, 60, 60]
        gs = [5, 10, 10, 10, 50, 50, 50]
        sizes = [3, 10, 10, 10, 10, 10, 10]
        benefits = [15, 70, 70, 70, 140, 140, 140]
        resources = [10, 25, 25, 25, 50, 50, 50]
        cost_coops = [5, 20, 20, 20, 60, 60, 60]
        cost_distants = [5, 5, 5, 5, 15, 15, 15]
        cost_stayin_alives = [5, 2, 2, 1, 2, 2, 1]
        cost_repros = [5, 2, 1, 0.5, 2, 1, 0.5]
        p_mutations = [0.1, 0, 0.05, 0.1, 0, 0.05, 0.1]
        thresholds = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        present_weights = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
        learning_rates = [0.1, 0.05, 0.1, 0.2, 0.05, 0.1, 0.2]
        p_swaps = [0.05, 0, 0.1, 0.2, 0, 0.1, 0.2]
        epsilons = [0.05, 0, 0.05, 0.1, 0, 0.05, 0.1]

        # for each of the 7 configurations, run the tests
        for i in range(7):
            sm = SpatialModel(
                n=ns[i], g=gs[i], size=sizes[i], rand=True, benefit=benefits[i], resources=resources[i], cost_coop=cost_coops[i], cost_distant=cost_distants[i],
                cost_repro=cost_repros[i], cost_stayin_alive=cost_stayin_alives[i], p_mutation=p_mutations[i], threshold=thresholds[i], present_weight=present_weights[i],
                learning_rate=learning_rates[i], p_swap=p_swaps[i], epsilon=epsilons[i], distrib=[1/3, 1/3, 1/3], write_log=False)
            
            # INITIALIZATION

            # grid_group_indices
            for i in range(sm.size):
                for j in range(sm.size):
                    index = sm.grid_group_indices[i, j]
                    if index == -1:
                        for k, group in sm.groups.items():
                            self.assertNotEqual(group.location, (i, j)) # no group is on the square
                    else:
                        self.assertIn(index, sm.groups) # there is a live group with this index 
                        self.assertEqual(sm.groups[index].location, (i, j)) # the group with that index has the correct location
            
            # forager_grid
            for i in range(sm.size):
                for j in range(sm.size):
                    if (sm.forager_grid.grid[i,j,:] == np.zeros(5)).all():
                        self.assertEqual(sm.grid_group_indices[i, j], -1) # if the forager_grid has no agents, then there should be no group
                        for k, group in sm.groups.items():
                            self.assertNotEqual(group.location, (i, j)) # check that no group has this location
                    else:
                        self.assertTrue((sm.forager_grid.grid[i,j,1:] == np.zeros(4)).all()) # all non-home slots should be zero to start
                        self.assertNotEqual(sm.grid_group_indices[i, j], -1) # there should be a group on this square 

                        index = sm.grid_group_indices[i, j]
                        self.assertEqual(sm.groups[index].location, (i, j)) # the group with the corresponding index should be located at this square
                        self.assertEqual(len(sm.groups[index].agents), sm.forager_grid.grid[i, j, 0]) # populations should match
                            
            # check initial group/agent invariants
            for i, group in sm.groups.items():
                row = group.location[0]
                col = group.location[1]
                self.assertEqual(i, sm.grid_group_indices[row, col]) # presence is reflected in grid_group_indices
                self.assertEqual(sm.n, sm.forager_grid.grid[row, col, 0]) # population is reflected in forager_grid

                strategy_counts = {"static": 0, "selfish": 0, "civic": 0}
                for agent in group.agents:
                    strategy_counts[agent.learning] += 1 
                    self.assertEqual(agent.group, group) # group field should match
                    self.assertEqual(agent.foraging_direction, None) # this hasn't been set yet
                    self.assertTrue(agent.first_round) # it is the agent's first round
                
                # check that strategy counts are as expected
                self.assertEqual(strategy_counts["civic"], group.n_agents["civic"])
                self.assertEqual(strategy_counts["static"], group.n_agents["static"])
                self.assertEqual(strategy_counts["selfish"], group.n_agents["selfish"])
                
                self.assertTrue(group.first_round) # it's the group's first round

                # flags in the right place
                self.assertEqual(group.all_civic, strategy_counts["civic"] == len(group.agents))
                self.assertEqual(group.mostly_civic, strategy_counts["civic"] > 0.9*len(group.agents))
                self.assertEqual(group.majority_civic, strategy_counts["civic"] > 0.5*len(group.agents))

            # LOOP
            for j in range(30):
                print(sm.year)
                sm.loop()

                # forager_grid invariants
                self.assertTrue((np.zeros((sm.size, sm.size, 5)) == sm.forager_grid_next.grid).all()) # this should have been reset to zeros
                population = 0
                square_populations = np.zeros((sm.size, sm.size)) # to keep track of the population on each square
                
                for group in sm.groups.values():
                    group_square = group.location
                    up_square = sm.forager_grid.direction_to_coord(group_square[0], group_square[1], 1)
                    down_square = sm.forager_grid.direction_to_coord(group_square[0], group_square[1], 2)
                    left_square = sm.forager_grid.direction_to_coord(group_square[0], group_square[1], 3)
                    right_square = sm.forager_grid.direction_to_coord(group_square[0], group_square[1], 4)

                    stay_population = sm.forager_grid.grid[group_square[0], group_square[1], 0]
                    up_population = sm.forager_grid.grid[up_square[0], up_square[1], 2]
                    down_population = sm.forager_grid.grid[down_square[0], down_square[1], 1]
                    left_population = sm.forager_grid.grid[left_square[0], left_square[1], 4]
                    right_population = sm.forager_grid.grid[right_square[0], right_square[1], 3]
                    sum_population = stay_population + left_population + right_population + up_population + down_population
                    
                    self.assertEqual(len(group.agents), sum_population) # check that the group population matches its presence in teh forager_grid
        
                    location_counts = [0, 0, 0, 0, 0]
                    for agent in group.agents:
                        if not agent.just_migrated:
                            expected_square = sm.forager_grid.direction_to_coord(group_square[0], group_square[1], agent.foraging_direction)
                            location_counts[agent.foraging_direction] += 1
                        else:
                            expected_square = group_square # agents that migrated are always on the group's home square
                            location_counts[0] += 1
                        
                        square_populations[agent.square] += 1 # keep track of the population on the square
                        self.assertEqual(agent.square, expected_square)

                        population += 1
                    
                    # the location info saved in the agent should match the one saved in the grid
                    self.assertEqual(stay_population, location_counts[0])
                    self.assertEqual(up_population, location_counts[1])
                    self.assertEqual(down_population, location_counts[2])
                    self.assertEqual(left_population, location_counts[3])
                    self.assertEqual(right_population, location_counts[4])

                self.assertTrue((sm.forager_grid.grid.sum(axis=2) == square_populations).all()) # squares have expected populations
                
                # test num_foragers 
                for i in range(sm.size):
                    for j in range(sm.size):
                        self.assertEqual(square_populations[i, j] , sm.forager_grid.num_foragers((i, j)))


                for i in range(sm.size):
                    for j in range(sm.size):
                        if sm.grid_group_indices[i, j] == -1:
                            self.assertEqual(sm.forager_grid.grid[i, j, 0], 0) # there should be no home foragers if no group is present
                
                self.assertEqual(population, sm.forager_grid.grid.sum())


                # grid_group_indices
                self.assertEqual((sm.grid_group_indices >= 0).sum(), len(sm.groups)) # number of non-negative entries should match number of groups
                
                indices = []
                for i in range(sm.size):
                    for j in range(sm.size):
                        index = sm.grid_group_indices[i, j]
                        self.assertNotIn(index, indices) # no repeat indices

                        if index == -1:
                            for group in sm.groups.values():
                                self.assertNotEqual(group.location, (i, j)) # no group should be on -1 squares
                        else:
                            indices.append(index)
                            group = sm.groups[index]
                            self.assertEqual(group.location, (i, j)) # corresponding group shoud have correct location
                
                # groups
                for group in sm.groups.values():
                    index = sm.grid_group_indices[group.location[0], group.location[1]]
                    self.assertEqual(group.id, index) # should match with grid_Group_indices
                    
                    # check strategy and cooperator counts
                    self.assertTrue(len(group.agents) > 0)
                    counts = defaultdict(lambda: 0)
                    cooperators = 0
                    for agent in group.agents:
                        counts[agent.learning] += 1
                        cooperators += agent.cooperate
                    
                    self.assertEqual(cooperators/len(group.agents), group.pct_cooperators)
                    
                    self.assertEqual(counts["static"], group.n_agents["static"])
                    self.assertEqual(counts["civic"], group.n_agents["civic"])
                    self.assertEqual(counts["selfish"], group.n_agents["selfish"])
                    self.assertEqual(sum(counts.values()), len(group.agents))

                    # check other flags for the groups
                    self.assertEqual(group.all_civic, counts["civic"] == len(group.agents))
                    self.assertEqual(group.mostly_civic, counts["civic"] > len(group.agents)*0.9)
                    if group.majority_civic and not counts["civic"] > len(group.agents)*0.5:
                        print(group.majority_civic)
                        print(group.id)
                        print(counts["civic"])
                        print(counts["static"])
                        print(counts["selfish"])
                        print(counts["civic"] + counts["static"] + counts["selfish"])
                        print([agent.learning for agent in group.agents])
                    self.assertEqual(group.majority_civic, counts["civic"] > len(group.agents)*0.5)
                
                # agents
                agents_visited = []
                for group in sm.groups.values():
                    group_square = group.location
                    up_square = sm.forager_grid.direction_to_coord(group_square[0], group_square[1], 1)
                    down_square = sm.forager_grid.direction_to_coord(group_square[0], group_square[1], 2)
                    left_square = sm.forager_grid.direction_to_coord(group_square[0], group_square[1], 3)
                    right_square = sm.forager_grid.direction_to_coord(group_square[0], group_square[1], 4)
                    
                    for agent in group.agents:
                        self.assertNotIn(agent, agents_visited) # no agent should be in multiple groups
                        agents_visited.append(agent)
                        self.assertIn(agent.square, [group_square, left_square, right_square, up_square, down_square]) # agent should be adjacent to group

                        if not agent.just_migrated:
                            expected_square = sm.forager_grid.direction_to_coord(group_square[0], group_square[1], agent.foraging_direction) # direction and square should match
                        else:
                            expected_square = group.location # if the agent migrated, they should be on the home square
                        
                        self.assertEqual(expected_square, agent.square)
                        
                        self.assertIn(agent.cooperate, [True, False])
                        self.assertTrue(agent.private_benefit >= 0) # because resources shoudln't be negative
                        self.assertTrue(agent.public_benefit >= 0) # because agents can't contribute negative benefit

                        if not agent.cooperate:
                            self.assertEqual(agent.public_benefit, 0) # non-cooperative agents don't contribute

                        self.assertEqual(agent.group, group)
                        self.assertTrue(agent.age <= agent.lifespan)
                        self.assertTrue(agent.fitness > 0)
    
    # tests that all components are largely functioning as expected
    def testLoop(self):
        # one where everyone should die
        sm = SpatialModel(n=20, g=20, resources=1, benefit=1, cost_coop=1, cost_stayin_alive=20, cost_distant=1, present_weight=0.3, write_log=False)

        # Round 0
        for group in sm.groups.values():
            for agent in group.agents:
                agent.lifespan = 100
                agent.initial_pi = agent.pi

        sm.loop()

        population = 0
        
        for group in sm.groups.values():
            up_square = sm.forager_grid.direction_to_coord(group.location[0], group.location[1], 1)
            down_square = sm.forager_grid.direction_to_coord(group.location[0], group.location[1], 2)
            left_square = sm.forager_grid.direction_to_coord(group.location[0], group.location[1], 3)
            right_square = sm.forager_grid.direction_to_coord(group.location[0], group.location[1], 4)
            total_group_benefit = 0
            caught_count = 0
            all_pis_equal = True

            for agent in group.agents:
                self.assertIn(agent.square, [up_square, down_square, left_square, right_square, group.location]) # agents should be adjacent, square_decision happened 
                self.assertTrue(agent.cooperate == True or agent.public_benefit == 0) # either the agent cooperated, or its public benefit is 0, coop_decision happened
                population += 1
                total_group_benefit += agent.public_benefit 
                self.assertTrue(agent.fitness >= agent.private_benefit) # agent gets at least its private benefit -- distribution happened
                caught_count += agent.caught
                all_pis_equal = all_pis_equal and (agent.pi == agent.initial_pi) # check that pis change (learning happened)

            self.assertFalse(all_pis_equal) # learning happened
            if total_group_benefit != 0:
                self.assertEqual(total_group_benefit/(20 - caught_count), group.avg_benefit) # group_distribution happened
            else:
                self.assertEqual(0, group.avg_benefit)
            
            group.old_avg_benefit = group.avg_benefit # save this so we can compare next round

        self.assertEqual(population, 400) # no one should have died
        self.assertEqual(len(sm.groups), 20) # no groups should have died


        # Round 1
        for group in sm.groups.values():
            for agent in group.agents:
                agent.lifespan = 100
                agent.initial_pi = agent.pi

        sm.loop()

        population = 0
        
        for group in sm.groups.values():
            up_square = sm.forager_grid.direction_to_coord(group.location[0], group.location[1], 1)
            down_square = sm.forager_grid.direction_to_coord(group.location[0], group.location[1], 2)
            left_square = sm.forager_grid.direction_to_coord(group.location[0], group.location[1], 3)
            right_square = sm.forager_grid.direction_to_coord(group.location[0], group.location[1], 4)
            total_group_benefit = 0
            caught_count = 0
            all_pis_equal = True

            for agent in group.agents:
                self.assertIn(agent.square, [up_square, down_square, left_square, right_square, group.location]) # square_decision
                self.assertTrue(agent.cooperate == True or agent.public_benefit == 0) # coop_decision
                population += 1
                total_group_benefit += agent.public_benefit 
                self.assertTrue(agent.fitness >= agent.private_benefit) # group_distribution
                caught_count += agent.caught
                all_pis_equal = all_pis_equal and (agent.pi == agent.initial_pi) # learn

            self.assertFalse(all_pis_equal)
            if total_group_benefit != 0:
                self.assertEqual(total_group_benefit/(20 - caught_count)*0.3 + 0.7*group.old_avg_benefit, group.avg_benefit) # correct avg_benefit update
            else:
                self.assertEqual(0.7*group.old_avg_benefit, group.avg_benefit)
            
            group.old_avg_benefit = group.avg_benefit

        self.assertEqual(population, 400) # no one should have died
        self.assertEqual(len(sm.groups), 20) # no one should have died

        # Round 2
        for group in sm.groups.values():
            for agent in group.agents:
                agent.initial_pi = agent.pi

        sm.loop()

        population = 0
        
        for group in sm.groups.values():
            for agent in group.agents:
                population += 1
 
        self.assertEqual(population, 0) # everyone should have died
        self.assertEqual(len(sm.groups), 0) # groups should be eliminated

        # one where no one should die and everyone should reproduce
        sm = SpatialModel(n=20, g=20, resources=25, benefit=60, cost_coop=20, cost_stayin_alive=0, cost_repro=0, cost_distant=5, write_log=False)

        # Round 0
        for group in sm.groups.values():
            for agent in group.agents:
                agent.lifespan = 100 # increase lifespan so it isn't the bottleneck
                agent.initial_pi = agent.pi

        sm.loop()

        population = 0
        
        for group in sm.groups.values():
            up_square = sm.forager_grid.direction_to_coord(group.location[0], group.location[1], 1)
            down_square = sm.forager_grid.direction_to_coord(group.location[0], group.location[1], 2)
            left_square = sm.forager_grid.direction_to_coord(group.location[0], group.location[1], 3)
            right_square = sm.forager_grid.direction_to_coord(group.location[0], group.location[1], 4)
            total_group_benefit = 0
            caught_count = 0
            all_pis_equal = True

            for agent in group.agents:
                self.assertIn(agent.square, [up_square, down_square, left_square, right_square, group.location]) 
                self.assertTrue(agent.cooperate == True or agent.public_benefit == 0)
                population += 1
                total_group_benefit += agent.public_benefit
                self.assertTrue(agent.fitness >= agent.private_benefit)
                caught_count += agent.caught
                all_pis_equal = all_pis_equal and (agent.pi == agent.initial_pi)

            self.assertFalse(all_pis_equal)
            if total_group_benefit != 0:
                self.assertEqual(total_group_benefit/(20 - caught_count), group.avg_benefit)
            else:
                self.assertEqual(0, group.avg_benefit)
            
            group.old_avg_benefit = group.avg_benefit

        self.assertEqual(population, 400) # no one should have died
        self.assertEqual(len(sm.groups), 20)

        # Round 1
        for group in sm.groups.values():
            for agent in group.agents:
                agent.initial_pi = agent.pi
                agent.lifespan = 100

        sm.loop()

        population = 0
        
        for group in sm.groups.values():
            up_square = sm.forager_grid.direction_to_coord(group.location[0], group.location[1], 1)
            down_square = sm.forager_grid.direction_to_coord(group.location[0], group.location[1], 2)
            left_square = sm.forager_grid.direction_to_coord(group.location[0], group.location[1], 3)
            right_square = sm.forager_grid.direction_to_coord(group.location[0], group.location[1], 4)
            total_group_benefit = 0
            caught_count = 0
            all_pis_equal = True

            for agent in group.agents:
                self.assertIn(agent.square, [up_square, down_square, left_square, right_square, group.location])
                self.assertTrue(agent.cooperate == True or agent.public_benefit == 0)
                population += 1
                total_group_benefit += agent.public_benefit
                self.assertTrue(agent.fitness >= agent.private_benefit)
                caught_count += agent.caught
                if hasattr(agent, "initial_pi"):
                    all_pis_equal = all_pis_equal and (agent.pi == agent.initial_pi)

            self.assertFalse(all_pis_equal)
            if total_group_benefit != 0:
                self.assertEqual(total_group_benefit/(40 - caught_count)*0.3 + 0.7*group.old_avg_benefit, group.avg_benefit)
            else:
                self.assertEqual(0.7*group.old_avg_benefit, group.avg_benefit)
            
            group.old_avg_benefit = group.avg_benefit

        self.assertEqual(population, 800) # no one should have died, everyone reproduced
        self.assertEqual(len(sm.groups), 20) # no group sent more than n agents to a different square

        # Round 4
        # run a few rounds until groups should definitely have started reproducing 
        for i in range(3):
            for group in sm.groups.values():
                for agent in group.agents:
                    agent.initial_pi = agent.pi
                    agent.lifespan = 100

                group.old_avg_benefit = group.avg_benefit
            sm.loop()
        
        population = 0
        
        for group in sm.groups.values():
            up_square = sm.forager_grid.direction_to_coord(group.location[0], group.location[1], 1)
            down_square = sm.forager_grid.direction_to_coord(group.location[0], group.location[1], 2)
            left_square = sm.forager_grid.direction_to_coord(group.location[0], group.location[1], 3)
            right_square = sm.forager_grid.direction_to_coord(group.location[0], group.location[1], 4)
            total_group_benefit = 0
            caught_count = 0
            all_pis_equal = True

            for agent in group.agents:
                self.assertIn(agent.square, [up_square, down_square, left_square, right_square, group.location])
                self.assertTrue(agent.cooperate == True or agent.public_benefit == 0)
                population += 1
                total_group_benefit += agent.public_benefit
                self.assertTrue(agent.fitness >= agent.private_benefit)
                caught_count += agent.caught

                if hasattr(agent, "initial_pi"):
                    all_pis_equal = all_pis_equal and (agent.pi == agent.initial_pi)

            self.assertFalse(all_pis_equal)


        self.assertEqual(population, 6400) # no one should have died, everyone reproduced
        self.assertTrue(len(sm.groups) > 20) # groups should have reproduced by now
        
    # test SpatialModel.sample_points
    def testSamplePoints(self):

        # check that points are ordered as expected
        sm = SpatialModel(size=3, g=9, rand=False, write_log=False)
        group_points = sm.sample_points(9)
        expected_group_points = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
        self.assertEqual(group_points, expected_group_points)
        
        sm = SpatialModel(size=2, g=3, rand=False, write_log=False)
        expected_group_points = [(0, 0), (0, 1), (1, 0)]
        group_points = sm.sample_points(3)
        self.assertEqual(group_points, expected_group_points)

        # check that points are in the correct range
        # and that there are no overlaps
        sm = SpatialModel(size=5, g=22, rand=True, write_log=False)
        group_points = sm.sample_points(22)
        
        for i, point in enumerate(group_points):
            self.assertTrue(point not in group_points[:i])
            self.assertTrue(point not in group_points[i+1:])

            self.assertTrue(point[0] in range(5))
            self.assertTrue(point[1] in range(5))
        
    # test SpatialModel.initialize_groups
    def testInitializeGroups(self):
        sm = SpatialModel(size=3, n=50, g=5, distrib=[1/2, 1/2, 0], rand=False, write_log=False)

        # check each agent list for correct properties
        for index, group in sm.groups.items():
            row = index // sm.size
            col = index % sm.size
            self.assertEqual(sm.grid_group_indices[row, col], index)
            self.assertEqual(sm.forager_grid.grid[row, col, 0], sm.n)
            self.assertEqual(len(group.agents), sm.n)
            for agent in group.agents:
                self.assertTrue(agent.learning == "selfish" or agent.learning == "static")
                self.assertTrue(agent.pi > -1)
                self.assertTrue(agent.pi < 1)
                self.assertEqual(agent.age, 0)


        sm = SpatialModel(size=3, n= 50, g=5, distrib=[1/2, 0, 1/2], rand=True, write_log=False)

        # check each agent list for correct properties
        for index, group in sm.groups.items():
            row = group.location[0]
            col = group.location[1]
            self.assertEqual(sm.grid_group_indices[row, col], index)
            self.assertEqual(sm.forager_grid.grid[row, col, 0], sm.n)
            self.assertEqual(len(group.agents), 50)
            for agent in group.agents:
                self.assertTrue(agent.learning == "static" or agent.learning == "civic")
                self.assertTrue(agent.pi > -1)
                self.assertTrue(agent.pi < 1)
                self.assertEqual(agent.age, 0)
    
    # test SpatialModel.cycle_of_life
    def testCycleOfLife(self):
        sm = SpatialModel(n=8, g=7, cost_stayin_alive=4, write_log=False)

        for j, group in sm.groups.items():
            for i, agent in enumerate(group.agents):
                if j == 3 or j == 6:
                    agent.fitness = 3
                else:
                    if i % 2 == 0:
                        agent.fitness = 3
                    else:
                        agent.fitness = 5
                        agent.lifespan = 100
        
        sm.cycle_of_life()

        # all agents died in 3 and 6
        self.assertNotIn(3, sm.groups)
        self.assertNotIn(6, sm.groups)
        self.assertEqual(len(sm.groups), 5)

        # half the agents died in groups 0, 1, 2, 4, 5
        for group in sm.groups.values():
            self.assertEqual(len(group.agents), 4)
        
    # test SpatialModel.calc_expected_payoff
    def testCalcExpectedPayoffs(self):
        forager_grid = np.array(
                        [[[10,  4,  5, 20,  9],
                            [ 2, 15,  6, 14,  7],
                            [ 9, 11,  1,  0, 14],
                            [12, 16,  1, 20, 10]],

                        [[15,  1, 11, 18, 12],
                            [ 2,  7, 17,  0,  6],
                            [ 0, 21, 17,  4,  1],
                            [12,  3, 10,  8,  9]],

                        [[ 4, 17, 13,  0, 10],
                            [13,  9, 14,  8,  6],
                            [21,  9, 16, 18, 19],
                            [15, 10,  2,  4,  2]],

                        [[13, 10, 14,  4, 15],
                            [ 7, 16,  3, 17, 13],
                            [13, 19, 21,  0,  8],
                            [20, 18, 16,  8, 11]]])

        sm = SpatialModel(g=10, size=4, resources=20, cost_distant=5, write_log=False)
        sm.forager_grid.grid = forager_grid 

        here, outside = sm.calc_expected_payoffs(2, 3)
        self.assertAlmostEqual(0.588235294, here)
        self.assertAlmostEqual(0.243902439, outside)
        

        here, outside = sm.calc_expected_payoffs(3, 0)
        self.assertAlmostEqual(0.350877193, here)
        self.assertAlmostEqual(0.266666667, outside)

        here, outside = sm.calc_expected_payoffs(0, 2)
        self.assertAlmostEqual(0.555555556, here)
        self.assertAlmostEqual(0.28436019, outside)

        forager_grid = np.array(
                        [[[13, 14,  1, 16, 15],
                            [14, 17,  0, 18,  6],
                            [20,  7,  1,  2,  6]],

                        [[ 8,  4, 15,  0, 19],
                            [ 1,  8, 15,  0, 18],
                            [ 7, 19,  8, 15, 16]],

                        [[20,  8, 11,  2,  8],
                            [ 3, 21, 16,  4, 21],
                            [15,  4, 17,  7,  4]]])
        
        sm = SpatialModel(g=6, size=3, resources=16, cost_distant=2, write_log=False)
        sm.forager_grid.grid = forager_grid

        here, outside = sm.calc_expected_payoffs(1, 0)
        self.assertAlmostEqual(0.340425532, here)
        self.assertAlmostEqual(0.255707763, outside)
        

        here, outside = sm.calc_expected_payoffs(1, 1)
        self.assertAlmostEqual(0.372093023, here)
        self.assertAlmostEqual(0.238297872, outside)

        here, outside = sm.calc_expected_payoffs(0, 2)
        self.assertAlmostEqual(0.432432432, here)
        self.assertAlmostEqual(0.243478261, outside)
    
    # SpatialModel.calc_foraging_probs
    def testCalcForagingProbs(self):
        forager_grid = np.array(
                    [[[11, 17,  4,  9, 14],
                        [21, 12, 21,  0,  0],
                        [16,  2,  3,  6, 11],
                        [14,  5,  2,  3,  3]],

                    [[20,  6, 10,  2,  6],
                        [ 4, 15,  9, 20,  5],
                        [15,  9, 14, 13, 19],
                        [ 9, 11, 17, 20,  3]],

                    [[ 6, 12,  1, 20, 12],
                        [ 3, 10, 11,  6,  8],
                        [ 9, 16, 18, 17, 14],
                        [ 4,  7,  1,  5, 10]],

                    [[ 4,  9,  5,  3, 16],
                        [18,  2, 15, 13, 15],
                        [17, 10, 13,  3,  1],
                        [ 5,  3,  1,  3,  0]]])
        
        sm = SpatialModel(g=10, size=4, resources=20, cost_distant=5, write_log=False)
        sm.forager_grid.grid = forager_grid 

        probs = sm.calc_foraging_probs(2, 2)
        self.assertAlmostEqual(probs[0], 0.448529412)
        self.assertAlmostEqual(probs[1], 0.137867647)
        self.assertAlmostEqual(probs[2], 0.137867647)
        self.assertAlmostEqual(probs[3], 0.137867647)
        self.assertAlmostEqual(probs[4], 0.137867647)

        probs = sm.calc_foraging_probs(3, 0)
        self.assertAlmostEqual(probs[0], 0.618729097)
        self.assertAlmostEqual(probs[1], 0.095317726)
        self.assertAlmostEqual(probs[2], 0.095317726)
        self.assertAlmostEqual(probs[3], 0.095317726)
        self.assertAlmostEqual(probs[4], 0.095317726)

        probs = sm.calc_foraging_probs(0, 0)
        self.assertAlmostEqual(probs[0], 0.497005988)
        self.assertAlmostEqual(probs[1], 0.125748503)
        self.assertAlmostEqual(probs[2], 0.125748503)
        self.assertAlmostEqual(probs[3], 0.125748503)
        self.assertAlmostEqual(probs[4], 0.125748503)
        

    def testSquareDecisions(self):
        sm = SpatialModel(n=10, g=10, size=4, p_swap=0, cost_distant=0, write_log=False)

        # check that the forager_grid is in the correct form
        self.assertTrue((sm.forager_grid_next.grid == np.zeros((4, 4, 5))).all())
        self.assertEqual(sm.forager_grid.grid.sum(), 100)
        self.assertEqual(sm.forager_grid.grid[:,:,1:].sum(), 0)
        for i in range(4):
            for j in range(4):
                for k in range(5):
                    if k != 0:
                        self.assertEqual(sm.forager_grid.grid[i, j, k], 0)
        
        sm.square_decisions()

        self.assertEqual(sm.calculations, 10) # check that probs are calculated for each group
        for group in sm.groups.values():
            square = group.location
            stay_counter = 0
            up_counter = 0
            down_counter = 0
            left_counter = 0
            right_counter = 0
            for agent in group.agents:
                left_square = (square[0], sm.forager_grid.modular_add(square[1], -1))
                right_square = (square[0], sm.forager_grid.modular_add(square[1], 1))
                up_square = (sm.forager_grid.modular_add(square[0], -1), square[1])
                down_square = (sm.forager_grid.modular_add(square[0], 1), square[1])
                self.assertIn(agent.square, [square, left_square, right_square, up_square, down_square])

                if agent.square == square:
                    stay_counter += 1
                    self.assertEqual(agent.foraging_direction, 0)
                elif agent.square == up_square:
                    up_counter += 1
                    self.assertEqual(agent.foraging_direction, 1)
                elif agent.square == down_square:
                    down_counter += 1
                    self.assertEqual(agent.foraging_direction, 2)
                elif agent.square == left_square:
                    left_counter += 1
                    self.assertEqual(agent.foraging_direction, 3)
                elif agent.square == right_square:
                    right_counter += 1
                    self.assertEqual(agent.foraging_direction, 4)
            
            self.assertEqual(sm.forager_grid.grid[square[0], square[1], 0], stay_counter)
            self.assertEqual(sm.forager_grid.grid[up_square[0], up_square[1], 2], up_counter)
            self.assertEqual(sm.forager_grid.grid[down_square[0], down_square[1], 1], down_counter)
            self.assertEqual(sm.forager_grid.grid[left_square[0], left_square[1], 4], left_counter)
            self.assertEqual(sm.forager_grid.grid[right_square[0], right_square[1], 3], right_counter)
        
        
        self.assertTrue((sm.forager_grid_next.grid == np.zeros((4, 4, 5))).all())
        self.assertTrue(sm.forager_grid.grid[:,:,1:].sum() > 0)
        self.assertEqual(sm.forager_grid.grid.sum(), 100)
    
        sm = SpatialModel(n=1, g=49, size=7, p_swap=1, cost_distant=0, write_log=False)
        self.assertTrue((sm.forager_grid_next.grid == np.zeros((7, 7, 5))).all())
        self.assertEqual(sm.forager_grid.grid.sum(), 49)
        self.assertEqual(sm.forager_grid.grid[:,:,1:].sum(), 0)

        sm.square_decisions()
        self.assertEqual(sm.calculations, 49)

        for group in sm.groups.values():
            square = group.location
            stay_counter = 0
            up_counter = 0
            down_counter = 0
            left_counter = 0
            right_counter = 0
            for agent in group.agents:
                left_square = (square[0], sm.forager_grid.modular_add(square[1], -1))
                right_square = (square[0], sm.forager_grid.modular_add(square[1], 1))
                up_square = (sm.forager_grid.modular_add(square[0], -1), square[1])
                down_square = (sm.forager_grid.modular_add(square[0], 1), square[1])
                self.assertIn(agent.square, [square, left_square, right_square, up_square, down_square])

                if agent.square == square and not agent.just_migrated:
                    stay_counter += 1
                    self.assertEqual(agent.foraging_direction, 0)
                elif agent.square == up_square and not agent.just_migrated:
                    up_counter += 1
                    self.assertEqual(agent.foraging_direction, 1)
                elif agent.square == down_square and not agent.just_migrated:
                    down_counter += 1
                    self.assertEqual(agent.foraging_direction, 2)
                elif agent.square == left_square and not agent.just_migrated:
                    left_counter += 1
                    self.assertEqual(agent.foraging_direction, 3)
                elif agent.square == right_square and not agent.just_migrated:
                    right_counter += 1
                    self.assertEqual(agent.foraging_direction, 4)
                elif agent.just_migrated:
                    stay_counter += 1

            self.assertEqual(sm.forager_grid.grid[square[0], square[1], 0], stay_counter)
            self.assertEqual(sm.forager_grid.grid[up_square[0], up_square[1], 2], up_counter)
            self.assertEqual(sm.forager_grid.grid[down_square[0], down_square[1], 1], down_counter)
            self.assertEqual(sm.forager_grid.grid[left_square[0], left_square[1], 4], left_counter)
            self.assertEqual(sm.forager_grid.grid[right_square[0], right_square[1], 3], right_counter)

        self.assertTrue((sm.forager_grid_next.grid == np.zeros((7, 7, 5))).all())
        self.assertTrue(len(sm.groups) < 49) # almost certainly true since agents are migrating
        self.assertEqual(sm.forager_grid.grid.sum(), 49)

        # test to see if migration goes as expected
        sm = SpatialModel(n=20, g=9, size=9, p_swap=1, write_log=False)

        # move groups so they're all in a line 
        for i, group in enumerate(sm.groups.values()):
            if not (group.location[0] == 0):
                sm.grid_group_indices[group.location] = -1
            group.location = (0, i)
            sm.grid_group_indices[0, i] = i
        
        sm.square_decisions(default_probs=[0, 0, 0, 0, 1])

        self.assertNotIn(0, sm.groups)
        self.assertEqual(sm.forager_grid.grid.sum(), 9*20)

        test_grid = np.zeros((9, 9, 5))
        test_grid[0, 0, 3] = 20

        for i in range(1, 9):
            test_grid[0, i, 0] = 20
        
        self.assertTrue((test_grid == sm.forager_grid.grid).all())
        
        for i in range(1, 9):
            self.assertIn(i, sm.groups)
        
        # test to see 
        sm = SpatialModel(n=1000, g=9, size=3, p_swap=0, cost_distant=0, write_log=False)
        sm.forager_grid.grid = np.array(
            [[[75, 75, 75, 75, 75],
              [12, 12, 12, 12, 12],
              [25, 25, 25, 25, 25]],

             [[12, 12, 12, 12, 12],
              [99, 99, 99, 99, 99],
              [12, 12, 12, 12, 12]],

             [[12, 12, 12, 12, 12],
              [12, 12, 12, 12, 12],
              [50, 50, 50, 50, 50]]]
        )

        sm.square_decisions()

        # these are probabilistic, so they might fail on occasion
        self.assertTrue(sm.forager_grid.grid[1, 1, 0] < sm.forager_grid.grid[0, 0, 0]) # PROB may fail sometimes
        self.assertTrue(sm.forager_grid.grid[0, 0, 0] < sm.forager_grid.grid[2, 2, 0]) # PROB may fail sometimes
        self.assertTrue(sm.forager_grid.grid[2, 2, 0] < sm.forager_grid.grid[0, 2, 0]) # PROB may fail sometimes

        self.assertEqual(sm.forager_grid.grid.sum(), 1000*9)
    
    # test SpatialModel.coop_decisions
    def testCoopDecisions(self):
        sm = SpatialModel(n=3, g=5, size=6, benefit=70, cost_coop=20, cost_distant=10, resources=25, memory=4, write_log=False)
        sm.year = 3

        group_benefits = [1.3, 2.7, 3.8, 0.16, 1.1]
        group_grid_entries = [[7,3,8,2,17],[2,5,3,4,6],[3,2,5,3,1],[5,2,3,6,2],[12,13,5,1,3]]
        agent_foraging_directions = [[2,1,0,0,0,0],[0,1,2,0,0],[1,0,1,2,0,0,0],[0,4,0,0],[3,2,0]]
        agent_pis = [[-1,-0.3,-0.1,0.2,0.3,0.6],[-0.8,-0.85,-0.4,-0.3,0.9],[-0.9,-0.8,-0.4,-0.2,-0.1,0.1,0.2],[0.8,0.9,0.95,0.99],[-0.94,0.2,0.8]]
        group_populations = [6, 5, 7, 4, 3]

        
        group_initial_n_rounds = [4, 5, 6, 7, 2]

        group_expected_pct_cooperators = [0.5, 0.4, 4/7, 0.5, 2/3]
        group_initial_pct_cooperators_memory_original = [[0.6,0.5,0.3,0.5], [0.6,0.2,0.9], [0.6,0.2], [0.2,0.7,0.3], []]
        group_initial_pct_cooperators_memory = [[0.6,0.5,0.3,0.5], [0.6,0.2,0.9], [0.6,0.2], [0.2,0.7,0.3], []]
        group_expected_avg_pct_cooperators = [0.45, 0.525, 0.45714285714285713, 0.425, 2/3]
            
        
        for i, group in enumerate(sm.groups.values()):
            
            while len(group.agents) < group_populations[i]:
                group.add_agent(SpatialAgent(sm, group))


            square = group.location
            group.avg_benefit = group_benefits[i]
            group.n_rounds = group_initial_n_rounds[i]
            group.pct_cooperators_memory = group_initial_pct_cooperators_memory[i]
            sm.forager_grid.grid[square[0], square[1], :] = group_grid_entries[i]

            for j, agent in enumerate(group.agents):
                agent.foraging_direction = agent_foraging_directions[i][j]
                agent.pi = agent_pis[i][j]
                agent.square = square
        
        sm.coop_decisions(p_obs=0.4, rand=False)

        for i, group in enumerate(sm.groups.values()):
            self.assertEqual(group.pct_cooperators, group_expected_pct_cooperators[i])
            self.assertEqual(group.pct_cooperators_memory, (group_initial_pct_cooperators_memory_original[i] + [group.pct_cooperators])[-4:])
            self.assertEqual(group.avg_pct_cooperators, group_expected_avg_pct_cooperators[i])
    
    # test SpatialModel.split_group
    def testSplitGroup(self):
        sm = SpatialModel(n=25, g=13, size=5, write_log=False)

        # split group index 4, 13 on new_square, 12 on old square 
        old_group = sm.groups[4]
        curr_square = old_group.location 
        
        possible_points = sm.sample_points(25)

        i = 0
        while sm.grid_group_indices[possible_points[i]] != -1:    
            i += 1

        new_square = possible_points[i]

        expected_new_group = []
        expected_old_group = []
        for i, agent in enumerate(old_group.agents):
            if i % 2 == 0:
                agent.square = new_square
                expected_new_group.append(agent)
            else:
                agent.square = curr_square 
                expected_old_group.append(agent)
        
        old_forager_grid = sm.forager_grid.grid.copy()
        sm.split_group(curr_square, new_square)

        self.assertIn(13, sm.groups)
        new_group = sm.groups[13]
        self.assertEqual(sm.grid_group_indices[new_square], 13)
        self.assertEqual(sm.grid_group_indices[curr_square], 4)

        self.assertEqual(expected_new_group, new_group.agents)
        self.assertEqual(expected_old_group, old_group.agents)

        self.assertTrue(new_group.first_round)
        self.assertTrue(old_group.first_round)
        self.assertTrue((old_forager_grid == sm.forager_grid.grid).all()) # shouldn't change until next round

        # split group index 12, all on old square (newly created group should die)
        old_group = sm.groups[12]
        curr_square = old_group.location 
        
        possible_points = sm.sample_points(25)

        i = 0
        while sm.grid_group_indices[possible_points[i]] != -1:    
            i += 1

        new_square = possible_points[i]

        expected_new_group = []
        expected_old_group = []
        for i, agent in enumerate(old_group.agents):
            agent.square = curr_square 
            expected_old_group.append(agent)
        
        old_forager_grid = sm.forager_grid.grid.copy()
        sm.split_group(curr_square, new_square)

        self.assertNotIn(14, sm.groups)
        self.assertEqual(sm.grid_group_indices[new_square], -1)
        self.assertEqual(sm.grid_group_indices[curr_square], 12)

        self.assertEqual(expected_old_group, old_group.agents)
        self.assertEqual(len(old_group.agents), 25)
        self.assertEqual(len(sm.groups), 14)

        self.assertTrue(old_group.first_round)
        self.assertTrue((old_forager_grid == sm.forager_grid.grid).all()) # shouldn't change until next round

        # split group index 0, all on new square (old group should die)
        sm = SpatialModel(n=23, g=8, size=4, write_log=False)
        old_group = sm.groups[0]
        old_group.avg_benefit = 6
        curr_square = old_group.location 
        
        possible_points = sm.sample_points(16)

        i = 0
        while sm.grid_group_indices[possible_points[i]] != -1:    
            i += 1

        new_square = possible_points[i]

        expected_new_group = []
        expected_old_group = []
        for i, agent in enumerate(old_group.agents):
            agent.square = new_square 
            expected_new_group.append(agent)
        
        old_forager_grid = sm.forager_grid.grid.copy()
        sm.split_group(curr_square, new_square)

        self.assertNotIn(0, sm.groups)
        self.assertNotIn(old_group, list(sm.groups.values()))
        self.assertIn(8, sm.groups)
        self.assertEqual(len(sm.groups), 8)

        new_group = sm.groups[8]
        self.assertEqual(sm.grid_group_indices[curr_square], -1)
        self.assertEqual(sm.grid_group_indices[new_square], 8)

        self.assertEqual(expected_new_group, new_group.agents)
        self.assertEqual(len(old_group.agents), 0)
        self.assertEqual(len(new_group.agents), 23)
        
        self.assertTrue(old_group.first_round)
        self.assertTrue(new_group.first_round)
        self.assertEqual(new_group.avg_benefit, 6)
        self.assertTrue((old_forager_grid == sm.forager_grid.grid).all()) # shouldn't change until next round

        # split group on index 8, mix of squares
        old_group = sm.groups[8]
        old_group.avg_benefit = 4
        curr_square = old_group.location 
        
        possible_points = sm.sample_points(16)

        i = 0
        while sm.grid_group_indices[possible_points[i]] != -1:    
            i += 1

        new_square = possible_points[i]

        expected_new_group = []
        expected_old_group = []
        for i, agent in enumerate(old_group.agents):
            if i % 3 == 0:
                agent.square = new_square
                expected_new_group.append(agent)
            elif i % 3 == 1:
                agent.square = (new_square[0], sm.forager_grid.modular_add(new_square[1], 1)) # some other square
                expected_old_group.append(agent)
            else:
                agent.square = curr_square
                expected_old_group.append(agent)
        
        old_forager_grid = sm.forager_grid.grid.copy()
        sm.split_group(curr_square, new_square)

        self.assertIn(8, sm.groups)
        self.assertIn(9, sm.groups)
        self.assertEqual(len(sm.groups), 9)

        new_group = sm.groups[9]
        self.assertEqual(sm.grid_group_indices[curr_square], 8)
        self.assertEqual(sm.grid_group_indices[new_square], 9)

        self.assertEqual(expected_new_group, new_group.agents)
        self.assertEqual(len(old_group.agents), 15)
        self.assertEqual(len(new_group.agents), 8)
        
        self.assertTrue(old_group.first_round)
        self.assertTrue(new_group.first_round)
        self.assertEqual(new_group.avg_benefit, 4)
        self.assertTrue((old_forager_grid == sm.forager_grid.grid).all()) # shouldn't change until next round
    
    # test SpatialModel.bud_groups
    def testBudGroups(self):
        sm = SpatialModel(n=30, g=4, size=3, write_log=False)
        
        # create a map like this 
        # - G - 
        # - G G
        # - G -
        # see if the groups branch as expected
        sm.grid_group_indices = np.array([[-1, 0, -1], [-1, 1, 2], [-1, 3, -1]])
        g0 = sm.groups[0]
        g1 = sm.groups[1]
        g2 = sm.groups[2]
        g3 = sm.groups[3]

        g0.location = (0, 1)
        g1.location = (1, 1)
        g2.location = (1, 2)
        g3.location = (2, 1)

        sm.forager_grid.grid = np.array([
            [
                [ 0,  0,  0,  0, 22],
                [27, 16, 15,  0,  0],
                [ 0,  0, 32, 32,  0],
            ],
            [
                [ 0,  0,  0,  0, 35],
                [23, 33, 15,  0, 34],
                [23,  0,  0, 40,  0]
            ],
            [
                [ 0,  0,  0,  0, 35],
                [23, 25, 32,  0,  0],
                [ 0, 35,  0, 42,  0]
            ]
        ])

        old_forager_grid = sm.forager_grid.grid.copy()

        for i, agent in enumerate(g0.agents):
            if i < 11:
                agent.square = (0, 1)
            elif i < 20:
                agent.square = (0, 2)
            elif i < 30:
                agent.square = (1, 1)

        g5_expected_agents = []
        for i, agent in enumerate(g1.agents):
            if i < 11:
                agent.square = (1, 0)
                g5_expected_agents.append(agent)
            elif i < 20:
                agent.square = (1, 1)
            elif i < 30:
                agent.square = (1, 2)
        
        g4_expected_agents = []
        for i, agent in enumerate(g2.agents):
            if i < 11:
                agent.square = (1, 2)
            elif i < 20:
                agent.square = (1, 1)
            elif i < 26:
                agent.square = (0, 2)
                g4_expected_agents.append(agent)
            elif i < 30:
                agent.square = (2, 2)

        g6_expected_agents = []
        g7_expected_agents = []
        for i, agent in enumerate(g3.agents):
            if i < 11:
                agent.square = (2, 1)
            elif i < 20:
                agent.square = (2, 0)
                g6_expected_agents.append(agent)
            elif i < 30:
                agent.square = (2, 2)
                g7_expected_agents.append(agent)
 
        sm.bud_groups()
        self.assertEqual(len(sm.groups), 8)

        # check that groups are in the right place
        expected_grid = np.array(
            [
                [False, True, True],
                [True, True, True],
                [True, True, True]
            ]
        )
        mask = sm.grid_group_indices >= 0
        self.assertTrue((expected_grid ==  mask).all())

        g4 = sm.groups[sm.grid_group_indices[0, 2]]
        g5 = sm.groups[sm.grid_group_indices[1, 0]] 
        g6 = sm.groups[sm.grid_group_indices[2, 0]]
        g7 = sm.groups[sm.grid_group_indices[2, 2]]

        self.assertEqual(sm.grid_group_indices[(0, 1)], 0)
        self.assertEqual(sm.grid_group_indices[(1, 1)], 1)
        self.assertEqual(sm.grid_group_indices[(1, 2)], 2)
        self.assertEqual(sm.grid_group_indices[(2, 1)], 3)
        
        self.assertEqual(g0.location, (0, 1))
        self.assertEqual(g1.location, (1, 1))
        self.assertEqual(g2.location, (1, 2))
        self.assertEqual(g3.location, (2, 1))
        self.assertEqual(g4.location, (0, 2))
        self.assertEqual(g5.location, (1, 0))
        self.assertEqual(g6.location, (2, 0))
        self.assertEqual(g7.location, (2, 2))

        self.assertEqual(g4.agents, g4_expected_agents)
        self.assertEqual(g5.agents, g5_expected_agents)
        self.assertEqual(g6.agents, g6_expected_agents)
        self.assertEqual(g7.agents, g7_expected_agents)
        
        self.assertTrue((old_forager_grid == sm.forager_grid.grid).all())
    
    # test SpatialModel.add_group
    def testAddGroup(self):
        sm = SpatialModel(n=30, g=4, size=3, write_log=False)

        possible_points = sm.sample_points(5)

        i = 0
        while sm.grid_group_indices[possible_points[i]] != -1:
            i += 1
        
        point = possible_points[i]

        group = SpatialGroup(sm, point, [])

        original_forager_grid = sm.forager_grid.grid.copy()

        sm.add_group(group)

        self.assertEqual(sm.grid_group_indices[point], 4)
        self.assertEqual(len(sm.groups), 5)
        self.assertIn(group, sm.groups.values())
        self.assertTrue((original_forager_grid == sm.forager_grid.grid).all())

        # add a group that's at a point of another grou
        sm = SpatialModel(n=30, g=4, size=3, write_log=False)
        point = sm.groups[3].location

        group = SpatialGroup(sm, point, [])

        original_forager_grid = sm.forager_grid.grid.copy()

        sm.add_group(group)

        self.assertEqual(sm.grid_group_indices[point], 3)
        self.assertEqual(len(sm.groups), 4)
        self.assertNotIn(group, sm.groups.values())
        self.assertTrue((original_forager_grid == sm.forager_grid.grid).all())

        # add a group and change forager_grid
        sm = SpatialModel(n=30, g=4, size=3, write_log=False)
        
        possible_points = sm.sample_points(5)

        i = 0
        while sm.grid_group_indices[possible_points[i]] != -1:
            i += 1
        
        point = possible_points[i]

        group = SpatialGroup(sm, point, [SpatialAgent(sm, group), SpatialAgent(sm, group)])

        original_forager_grid = sm.forager_grid.grid.copy()

        sm.add_group(group, mod_forager_grid=True)

        self.assertEqual(sm.grid_group_indices[point], 4)
        self.assertEqual(len(sm.groups), 5)
        self.assertIn(group, sm.groups.values())

        original_forager_grid[point[0], point[1], 0] += 2
        self.assertTrue((original_forager_grid == sm.forager_grid.grid).all())


    # ------------------------------------------------------------
    # SpatialGroup Tests

    # test SpatialGroup.set_Agents
    def testSetAgents(self):
        sm = SpatialModel(n=22, write_log=False)
        new_agent_list = []

        for i in range(15):
            new_agent_list.append(SpatialAgent(sm, group=None, learning="civic"))
        
        group = sm.groups[0]
        group.set_agents(new_agent_list)
        for agent in new_agent_list:
            self.assertEqual(agent.group, group)
        
        self.assertIn(group, sm.groups.values())
        self.assertEqual(group.agents, new_agent_list)
        self.assertEqual(group.n_agents["civic"], 15)
        self.assertEqual(group.n_agents["selfish"], 0)
        self.assertEqual(group.n_agents["static"], 0)
        self.assertTrue(group.all_civic)
        self.assertTrue(group.mostly_civic)
        self.assertTrue(group.majority_civic)

        sm = SpatialModel(n=25, write_log=False)
        new_agent_list = []

        for i in range(33):
            if i % 11 == 0:
                new_agent_list.append(SpatialAgent(sm, group=None, learning="selfish"))
            else:
                new_agent_list.append(SpatialAgent(sm, group=None, learning="civic"))
        
        group = sm.groups[2]
        group.set_agents(new_agent_list)
        for agent in new_agent_list:
            self.assertEqual(agent.group, group)
        
        self.assertIn(group, sm.groups.values())
        self.assertEqual(group.agents, new_agent_list)
        self.assertEqual(group.n_agents["civic"], 30)
        self.assertEqual(group.n_agents["selfish"], 3)
        self.assertEqual(group.n_agents["static"], 0)
        self.assertTrue(not group.all_civic)
        self.assertTrue(group.mostly_civic)
        self.assertTrue(group.majority_civic)

        sm = SpatialModel(n=25, write_log=False)
        new_agent_list = []

        for i in range(21):
            if i % 2 == 0:
                new_agent_list.append(SpatialAgent(sm, group=None, learning="civic"))
            else:
                new_agent_list.append(SpatialAgent(sm, group=None, learning="static"))
        
        group = sm.groups[3]
        group.set_agents(new_agent_list)
        for agent in new_agent_list:
            self.assertEqual(agent.group, group)
        
        self.assertIn(group, sm.groups.values())
        self.assertEqual(group.agents, new_agent_list)
        self.assertEqual(group.n_agents["civic"], 11)
        self.assertEqual(group.n_agents["selfish"], 0)
        self.assertEqual(group.n_agents["static"], 10)
        self.assertTrue(not group.all_civic)
        self.assertTrue(not group.mostly_civic)
        self.assertTrue(group.majority_civic)

        sm = SpatialModel(n=25, write_log=False)
        new_agent_list = []

        for i in range(22):
            if i % 3 == 0:
                new_agent_list.append(SpatialAgent(sm, group=None, learning="civic"))
            elif i % 3 == 1:
                new_agent_list.append(SpatialAgent(sm, group=None, learning="static"))
            else:
                new_agent_list.append(SpatialAgent(sm, group=None, learning="selfish"))
        
        group = sm.groups[3]
        group.set_agents(new_agent_list)
        for agent in new_agent_list:
            self.assertEqual(agent.group, group)
        
        self.assertIn(group, sm.groups.values())
        self.assertEqual(group.agents, new_agent_list)
        self.assertEqual(group.n_agents["civic"], 8)
        self.assertEqual(group.n_agents["selfish"], 7)
        self.assertEqual(group.n_agents["static"], 7)
        self.assertTrue(not group.all_civic)
        self.assertTrue(not group.mostly_civic)
        self.assertTrue(not group.majority_civic)

        sm = SpatialModel(n=25, g=6, write_log=False)
        new_agent_list = []
        
        group = sm.groups[3]
        group.set_agents(new_agent_list)
        for agent in new_agent_list:
            self.assertEqual(agent.group, group)
        
        self.assertNotIn(group, sm.groups.values())
        self.assertEqual(len(sm.groups), 5)
        self.assertEqual(group.agents, new_agent_list)

        
    def testGroupDistribution(self):
        print("testing group distribution")
        sm = SpatialModel(n=10000, g=1, size=4, write_log=False)

        # check that around 60% of agents get caught when p_obs is fixed
        for agent in sm.groups[0].agents:
            agent.public_benefit = 4
            agent.private_benefit = 2
            agent.p_obs = 0.6
            agent.cooperate = False

        caught_count, _, _ = sm.groups[0].group_distribution(rand=True)
        self.assertTrue(caught_count > 5800 and caught_count < 6200) # PROB 

        sm.square_decisions()
        sm.coop_decisions()

        for agent in sm.groups[0].agents:
            agent.cooperate = False

        caught_count, _, _ = sm.groups[0].group_distribution(rand=True)
        self.assertTrue(caught_count > 4800 and caught_count < 5200) # PROB

        # test caught with mix of cooperators and non-cooperators
        sm = SpatialModel(n=6000, g=1, size=4, write_log=False)

        sm.square_decisions()
        sm.coop_decisions()

        for i, agent in enumerate(sm.groups[0].agents):
            if i % 3 == 0:
                agent.cooperate = True
            else:
                agent.cooperate= False
        
        caught_count, _, _ = sm.groups[0].group_distribution(rand=True)
        self.assertTrue(caught_count > 1900 and caught_count < 2100) # PROB

        # non-random trials to check outputs
        sm = SpatialModel(n=3, g=5, size=4, present_weight=0.7, write_log=False)
        
        # group inputs 
        group_populations = [7, 4, 8, 6, 3]
        group_first_rounds = [True, False, False, True, False]
        group_prev_avg_benefits = [0, 13, 7, 0, 2]

        # agent inputs
        cooperates = [[False, True,False,True,True,True,True],[False,False,True,True],[True,True,False,False,True,True,True,True],[False,False,False,False,True,True],[False,False,False]]
        first_rounds = [[False,False,False,True,True,False,True],[False,False,False,False],[False,False,False,False,True,False,False,True],[False,True,True,False,False,True],[False,False,True]]
        private_benefits = [[2,3,3,2,3,5,5], [3,1,4,4], [5,3,4,1,5,5,2,4],[3,1,4,3,5,5],[4,3,4]]
        public_benefits = [[0,15,0,5,4,8,5],[0,0,18,9],[9,14,0,0,16,14,14,14],[0,0,0,0,20,19],[0,0,0]]
        p_obs = [[0.9,0.7,0.6,0.8,0.4,1.0,0.7],[0.2,0.7,0.0,0.8],[0.0,0.1,0.8,0.9,0.4,1.0,0.3,1.0],[0.7,0.6,0.4,0.2,0.0,0.2],[0.6,1.0,0.4]]
        old_fitnesses = [[20,2,15,12,6,9,3],[17,13,19,14],[16,12,14,7,4,13,2,17],[5,10,11,16,1,8],[4,19,15]]
        old_fitness_diffs = [[5,7,15,0,0,7,0],[5,14,15,5],[15,10,12,13,0,14,9,0],[6,0,0,8,5,0],[12,11,0]]

        # group expected outputs
        group_caught_counts = [2, 1, 2, 2, 2]
        group_public_benefits = [37,27,81,39,0]
        group_shares = [7.4,9,13.5, 9.75, 0]
        group_avg_benefits = [7.4, 10.2, 11.55, 9.75, 0.6]

        # agent expected outputs
        fitness_diffs = [[2,10.4,3,9.4,10.4,12.4,12.4],[12,1,13,13],[18.5,16.5,4,1,18.5,18.5,15.5,17.5],[3,1,13.75,12.75,14.75,14.75],[4,3,4]]
        fitnesses = [[22,12.4,18,21.4,16.4,21.4,15.4],[29,14,32,27],[34.5,28.5,18,8,22.5,31.5,17.5,34.5],[8,11,24.75,28.75,15.75,22.75],[8,22,19]]
        avg_fitness_diffs = [[2.9,9.38,6.6,9.4,10.4,10.78,12.4],[9.9,4.9,13.6,10.6],[17.45,14.55,6.4,4.6,18.5,17.15,13.55,17.5],[3.9,1,13.75,11.325,11.825,14.75],[6.4,5.4,4]]


        for i, group in enumerate(sm.groups.values()):
            
            while len(group.agents) < group_populations[i]:
                group.add_agent(SpatialAgent(sm, group))
            
            group.first_round = group_first_rounds[i]
            group.avg_benefit = group_prev_avg_benefits[i]
        
            for j, agent in enumerate(group.agents):
                agent.cooperate = cooperates[i][j]
                agent.first_round = first_rounds[i][j]
                agent.private_benefit = private_benefits[i][j]
                agent.public_benefit = public_benefits[i][j]
                agent.p_obs = p_obs[i][j]
                agent.fitness = old_fitnesses[i][j]
                agent.avg_fitness_diff = old_fitness_diffs[i][j]
        

            caught_count, share, public_benefit = group.group_distribution(rand=False)
            self.assertEqual(caught_count, group_caught_counts[i])
            self.assertEqual(public_benefit, group_public_benefits[i])
            self.assertAlmostEqual(share, group_shares[i])
            self.assertAlmostEqual(group.avg_benefit, group_avg_benefits[i])

            for j, agent in enumerate(group.agents):
                self.assertAlmostEqual(agent.fitness_diff, fitness_diffs[i][j])
                self.assertAlmostEqual(agent.avg_fitness_diff, avg_fitness_diffs[i][j])
                self.assertAlmostEqual(agent.fitness, fitnesses[i][j])

    # test SpatialGroup.die
    def testDie(self):
        sm = SpatialModel(size=8, g=21, write_log=False)
        
        # kill of some groups
        loc = sm.groups[6].location
        sm.groups[6].die()
        self.assertTrue(6 not in sm.groups)
        self.assertTrue(7 in sm.groups)
        self.assertEqual(sm.grid_group_indices[loc], -1)

        loc = sm.groups[10].location
        sm.groups[10].die()
        self.assertTrue(10 not in sm.groups)
        self.assertTrue(0 in sm.groups)
        self.assertEqual(sm.grid_group_indices[loc], -1)

        for i in range(21):
            if i != 10 and i !=6:
                loc = sm.groups[i].location
                sm.groups[i].die()
                self.assertTrue(i not in sm.groups)
                self.assertEqual(sm.grid_group_indices[loc], -1)
        
        self.assertTrue(not sm.groups)
    
    # test SpatialGroup.add_agent
    def testAddAgent(self):
        sm = SpatialModel(n=10, g=10, size=10, write_log=False)

        group = sm.groups[4]
        agent = SpatialAgent(sm, group)
        old_n_agents = group.n_agents.copy()
        old_agents = group.agents.copy()

        group.add_agent(agent)

        self.assertEqual(len(group.agents), 11)
        self.assertTrue(agent in group.agents)
        self.assertEqual(old_n_agents[agent.learning] + 1, group.n_agents[agent.learning])
        self.assertEqual(old_agents + [agent], group.agents)
        self.assertEqual(agent.group, group)

        group = sm.groups[9]
        agent_list_copy = group.agents.copy()
        for i in range(5):
            agent = group.agents[0]
            n_agents_old = group.n_agents.copy()
            group.remove_agent(agent)

            self.assertEqual(len(group.agents), 9 - i)
            self.assertTrue(agent not in group.agents)
            self.assertEqual(n_agents_old[agent.learning] - 1, group.n_agents[agent.learning])
            self.assertEqual(agent_list_copy[i + 1:], group.agents)

        for i, agent in enumerate(group.agents):
            if i % 2 == 0:
                agent.learning = "static"
            else: 
                agent.learning = "selfish"

        group.recount_agents()

        for i in range(50):
            agent = SpatialAgent(sm, None, learning="civic")
            old_n_agents = group.n_agents.copy()
            old_agents = group.agents.copy()
            group.add_agent(agent)

            self.assertEqual(len(group.agents), 6 + i)
            self.assertTrue(agent in group.agents)
            self.assertEqual(old_n_agents[agent.learning] + 1, group.n_agents[agent.learning])
            self.assertEqual(old_agents + [agent], group.agents)
            self.assertEqual(agent.group, group)

            if i < 5:
                self.assertTrue(not group.majority_civic)
                self.assertTrue(not group.mostly_civic)
                self.assertTrue(not group.all_civic)
                self.assertEqual(group.n_agents["static"], 3)
                self.assertEqual(group.n_agents["selfish"], 2)
                self.assertEqual(group.n_agents["civic"], i + 1)
            elif i >= 5 and i < 45:
                self.assertTrue(group.majority_civic)
                self.assertTrue(not group.mostly_civic)
                self.assertTrue(not group.all_civic)
                self.assertEqual(group.n_agents["static"], 3)
                self.assertEqual(group.n_agents["selfish"], 2)
                self.assertEqual(group.n_agents["civic"], i + 1) 
            elif i >= 45:
                self.assertTrue(group.majority_civic)
                self.assertTrue(group.mostly_civic)
                self.assertTrue(not group.all_civic)
                self.assertEqual(group.n_agents["static"], 3)
                self.assertEqual(group.n_agents["selfish"], 2)
                self.assertEqual(group.n_agents["civic"], i + 1)
        
        group = sm.groups[8]
        agent_list_copy = group.agents.copy()

        for agent in group.agents:
            agent.learning = "civic"
        
        group.recount_agents()

        for i in range(9):
            agent = SpatialAgent(sm, None, learning="civic")
            old_n_agents = group.n_agents.copy()
            old_agents = group.agents.copy()
            group.add_agent(agent)

            self.assertEqual(len(group.agents), 11 + i)
            self.assertTrue(agent in group.agents)
            self.assertEqual(old_n_agents[agent.learning] + 1, group.n_agents[agent.learning])
            self.assertEqual(old_agents + [agent], group.agents)
            self.assertEqual(agent.group, group)

            self.assertTrue(group.majority_civic)
            self.assertTrue(group.mostly_civic)
            self.assertTrue(group.all_civic)
            self.assertEqual(group.n_agents["static"], 0)
            self.assertEqual(group.n_agents["selfish"], 0)
            self.assertEqual(group.n_agents["civic"], 11 + i)
        
        for i in range(21):
            agent = SpatialAgent(sm, None, learning=random.choice(["selfish", "static"]))
            old_n_agents = group.n_agents.copy()
            old_agents = group.agents.copy()
            group.add_agent(agent)

            self.assertEqual(len(group.agents), 20 + i)
            self.assertTrue(agent in group.agents)
            self.assertEqual(old_n_agents[agent.learning] + 1, group.n_agents[agent.learning])
            self.assertEqual(old_agents + [agent], group.agents)
            self.assertEqual(agent.group, group)

            if i < 2:
                self.assertTrue(group.majority_civic)
                self.assertTrue(group.mostly_civic)
                self.assertTrue(not group.all_civic)
                self.assertEqual(group.n_agents["static"] + group.n_agents["selfish"], i + 1)
                self.assertEqual(group.n_agents["civic"], 19)
            elif i >=2 and i < 18:
                self.assertTrue(group.majority_civic)
                self.assertTrue(not group.mostly_civic)
                self.assertTrue(not group.all_civic)
                self.assertEqual(group.n_agents["static"] + group.n_agents["selfish"], i + 1)
                self.assertEqual(group.n_agents["civic"], 19)
            elif i > 18:
                self.assertTrue(not group.majority_civic)
                self.assertTrue(not group.mostly_civic)
                self.assertTrue(not group.all_civic)
                self.assertEqual(group.n_agents["static"] + group.n_agents["selfish"], i + 1)
                self.assertEqual(group.n_agents["civic"], 19)
    
    # test SpatialGroup.add_agent
    def testRemoveAgent(self):
        sm = SpatialModel(size=8, g=21, n=25, write_log=False)
        
        group = sm.groups[5]
        agent = group.agents[4]
        group_agents_old = group.agents.copy()
        n_agents_old = group.n_agents.copy()

        group.remove_agent(agent)
    
        self.assertEqual(len(group.agents), 24)
        self.assertTrue(agent not in group.agents)
        self.assertEqual(n_agents_old[agent.learning] - 1, group.n_agents[agent.learning]) 
        self.assertEqual(group_agents_old[:4] + group_agents_old[5:], group.agents)

        agent_list_copy = group.agents.copy()
        for i, agent in enumerate(agent_list_copy):
            n_agents_old = group.n_agents.copy()
            group.remove_agent(agent)

            self.assertEqual(len(group.agents), 23 - i)
            self.assertTrue(agent not in group.agents)
            self.assertEqual(n_agents_old[agent.learning] - 1, group.n_agents[agent.learning])
            self.assertEqual(agent_list_copy[i + 1:], group.agents)
        
        self.assertTrue(5 not in sm.groups)
        self.assertTrue(group not in sm.groups.values())
        self.assertTrue(not group.agents)
        self.assertEqual(sm.grid_group_indices[group.location], -1)

        sm = SpatialModel(size=8, g=5, n=25, write_log=False)
        group = sm.groups[3]

        for i, agent in enumerate(group.agents):
            if i < 13:
                agent.learning = "static"
            else:
                agent.learning = "civic"
            
        group.recount_agents()
        
        agent_list_copy = group.agents.copy()
        for i, agent in enumerate(agent_list_copy):
            n_agents_old = group.n_agents.copy()
            group.remove_agent(agent)

            self.assertEqual(len(group.agents), 24 - i)
            self.assertTrue(agent not in group.agents)
            self.assertEqual(n_agents_old[agent.learning] - 1, group.n_agents[agent.learning])
            self.assertEqual(agent_list_copy[i + 1:], group.agents)

            if i == 0:
                self.assertTrue(not group.majority_civic)
                self.assertTrue(not group.mostly_civic)
                self.assertTrue(not group.all_civic)
                self.assertEqual(group.n_agents["static"], 12)
                self.assertEqual(group.n_agents["civic"], 12)
            elif i > 0 and i < 11:
                self.assertTrue(group.majority_civic)
                self.assertTrue(not group.mostly_civic)
                self.assertTrue(not group.all_civic)
                self.assertEqual(group.n_agents["static"], 12 - i)
                self.assertEqual(group.n_agents["civic"], 12)
            elif i == 11:
                self.assertTrue(group.majority_civic)
                self.assertTrue(group.mostly_civic)
                self.assertTrue(not group.all_civic)
                self.assertEqual(group.n_agents["static"], 1)
                self.assertEqual(group.n_agents["civic"], 12)
            elif i > 11:
                self.assertTrue(group.majority_civic)
                self.assertTrue(group.mostly_civic)
                self.assertTrue(group.all_civic)
                self.assertEqual(group.n_agents["static"], 0)
                self.assertEqual(group.n_agents["civic"], 24 - i) # start on index 12 with 12 civic learners

        self.assertTrue(3 not in sm.groups)
        self.assertTrue(group not in sm.groups.values())
        self.assertTrue(not group.agents)
        self.assertEqual(sm.grid_group_indices[group.location], -1)

        sm = SpatialModel(size=8, g=5, n=22, write_log=False)
        group = sm.groups[0]

        for i, agent in enumerate(group.agents):
            if i < 20:
                agent.learning = "civic"
            else:
                agent.learning = "static"
        
        group.recount_agents()

        agent_list_copy = group.agents.copy()
        for i, agent in enumerate(agent_list_copy):
            n_agents_old = group.n_agents.copy()
            group.remove_agent(agent)

            self.assertEqual(len(group.agents), 21 - i)
            self.assertTrue(agent not in group.agents)
            self.assertEqual(n_agents_old[agent.learning] - 1, group.n_agents[agent.learning])
            self.assertEqual(agent_list_copy[i + 1:], group.agents)

            if i == 0:
                self.assertTrue(group.majority_civic)
                self.assertTrue(group.mostly_civic)
                self.assertTrue(not group.all_civic)
                self.assertEqual(group.n_agents["static"], 2)
                self.assertEqual(group.n_agents["civic"], 19) # removed 1
            elif i > 0 and i < 17:
                self.assertTrue(group.majority_civic)
                self.assertTrue(not group.mostly_civic)
                self.assertTrue(not group.all_civic)
                self.assertEqual(group.n_agents["static"], 2)
                self.assertEqual(group.n_agents["civic"], 19 - i)
            elif i > 17 and i < 20:
                self.assertTrue(not group.majority_civic)
                self.assertTrue(not group.mostly_civic)
                self.assertTrue(not group.all_civic)
                self.assertEqual(group.n_agents["static"], 2)
                self.assertEqual(group.n_agents["civic"], 19 - i)
            elif i >= 20:
                self.assertTrue(not group.majority_civic)
                self.assertTrue(not group.mostly_civic)
                self.assertTrue(not group.all_civic)
                self.assertEqual(group.n_agents["static"], 21 - i) # start with 1 and on index 20
                self.assertEqual(group.n_agents["civic"], 0)
        
        self.assertTrue(0 not in sm.groups)
        self.assertTrue(group not in sm.groups.values())
        self.assertTrue(not group.agents)
        self.assertEqual(sm.grid_group_indices[group.location], -1)

    # test SpatialGroup.death_and_birth
    def testDeathAndBirth(self):
        print("testing death and birth")
        group_fitnesses =  [[1.5,1.1,1.5,1.8,0.3,0.4,2.2,2.4,0.1,2,0.4,2.7,2.4,1],
                            [1.6,5.6,2.7,2,0.5,1.2,2.2,5.3,4.3,2.3,5.2,1.2],
                            [8.8,9.8,9.3,7.8,9,7.9,7.5,9.2,8.8,7.2,7.8,9.2,8.9],
                            [5.2,4.5,9.8,8.2,9.6,5.3,9.1,5.7,7.2,10.3,6.6,8.2,6.2,3.3],
                            [9.7,1.8,1.6,2.3,0.5,1.2,2.9,4.2,6.9,0.7,8.3,8.5,5.7]]

        super_agents_to_die = []
        super_agents_to_repro = []
        sm = SpatialModel(n=12, g=5, cost_stayin_alive=3, cost_repro=1, write_log=False)

        for i, group in enumerate(sm.groups.values()): 
            while len(group.agents) < len(group_fitnesses[i]):
                group.add_agent(SpatialAgent(sm, group))
            
            agents_to_die = []
            agents_to_repro = []
            for j, agent in enumerate(group.agents):
                agent.lifespan = 3 if agent.lifespan < 3 else agent.lifespan
                agent.fitness = group_fitnesses[i][j]
                
                if agent.fitness < 3:
                    agents_to_die.append(agent)
                
                if agent.fitness >= 7:
                    agents_to_repro.append(agent)
            
            super_agents_to_die.append(agents_to_die)
            super_agents_to_repro.append(agents_to_repro)
            
        # all agents should die
        group0 = sm.groups[0]
        counter = group0.death_and_birth(rand=False)
        self.assertNotIn(group0.id, sm.groups)
        self.assertEqual(len(sm.groups), 4)
        self.assertEqual(counter, 14)

        for agent in super_agents_to_die[0]:
            self.assertNotIn(agent, group0.agents)

        # some agents should die, none should reproduce
        group1 = sm.groups[1]
        agents_to_survive = [agent for agent in group1.agents if agent not in super_agents_to_die[1]]
        counter = group1.death_and_birth(rand=False)
        self.assertIn(group1.id, sm.groups)
        self.assertEqual(len(sm.groups), 4)
        self.assertEqual(counter, 12)
        self.assertEqual(len(group1.agents), 12 - len(super_agents_to_die[1]))
        
        for agent in super_agents_to_die[1]:
            self.assertNotIn(agent, group1.agents)
        
        for agent in agents_to_survive:
            self.assertIn(agent, group1.agents)

        # all agents should survive and reproduce
        group2 = sm.groups[2]

        agents_to_survive = group2.agents.copy()
        agents_to_repro = super_agents_to_repro[2]
        agents_to_die = super_agents_to_die[2]
        
        counter = group2.death_and_birth(rand=False)

        self.assertIn(group2.id, sm.groups)
        self.assertEqual(len(sm.groups), 4)
        self.assertEqual(counter, 13)
        self.assertEqual(len(group2.agents), 26) # this sometimes fails for no apparent reason - it was because of a negative lifespan
        
        for agent in super_agents_to_die[2]:
            self.assertNotIn(agent, group2.agents)
        
        for agent in agents_to_survive:
            self.assertIn(agent, group2.agents)

            pi = agent.pi
            agents_with_same_pi = 0

            for other_agent in group2.agents:
                agents_with_same_pi += (pi == other_agent.pi)
            
            self.assertEqual(agents_with_same_pi, 2)
        
        # some agents should reproduce, all should survive 
        group3 = sm.groups[3]

        agents_to_survive = group3.agents
        agents_to_repro = super_agents_to_repro[3]
        counter = group3.death_and_birth(rand=False)
        self.assertIn(group3.id, sm.groups)
        self.assertEqual(len(sm.groups), 4)
        self.assertEqual(counter, 14)
        self.assertEqual(len(group3.agents), 14 + len(agents_to_repro))
        
        for agent in super_agents_to_die[3]:
            self.assertNotIn(agent, group3.agents)
        
        for agent in agents_to_survive:
            self.assertIn(agent, group3.agents)
        
        for agent in agents_to_repro:
            pi = agent.pi
            agents_with_same_pi = 0

            for other_agent in group3.agents:
                agents_with_same_pi += (pi == other_agent.pi)
            
            self.assertEqual(agents_with_same_pi, 2)
        
        # some agents should die, others should survive, others should survive and reproduce
        group4 = sm.groups[4]

        agents_to_die = super_agents_to_die[4]
        agents_to_survive = set(group4.agents) - set(agents_to_die)
        agents_to_repro = super_agents_to_repro[4]
        counter = group4.death_and_birth(rand=False)
        self.assertIn(group4.id, sm.groups)
        self.assertEqual(len(sm.groups), 4)
        self.assertEqual(counter, 13)
        self.assertEqual(len(group4.agents), 13 - len(agents_to_die) + len(agents_to_repro))
        
        for agent in agents_to_die:
            self.assertNotIn(agent, group4.agents)
        
        for agent in agents_to_survive:
            self.assertIn(agent, group4.agents)
        
        for agent in agents_to_repro:
            pi = agent.pi
            agents_with_same_pi = 0

            for other_agent in group4.agents:
                agents_with_same_pi += (pi == other_agent.pi)
            
            self.assertEqual(agents_with_same_pi, 2)

        # all agents should die
        group1 = sm.groups[1]
        group1_size = len(group1.agents)
        counter = group1.death_and_birth(rand=False)
        self.assertNotIn(group1.id, sm.groups)
        self.assertEqual(len(sm.groups), 3)
        self.assertEqual(counter, group1_size)

            
    # ------------------------------------------------------------
    # SpatialAgent Tests

    # test SpatialAgent.switch_group
    def testSwitchGroup(self):
        sm = SpatialModel(n=21, g=10, size=7, write_log=False)
        origin_group = sm.groups[3]
        origin_old_agents = origin_group.agents.copy()
        origin_old_n_agents = origin_group.n_agents.copy()

        dest_group = sm.groups[5]
        dest_old_agents = dest_group.agents.copy()
        dest_old_n_agents = dest_group.n_agents.copy()

        agent_to_switch = origin_group.agents[3]
        self.assertEqual(agent_to_switch.group, origin_group)
        agent_to_switch.switch_group(5)

        self.assertNotIn(agent_to_switch, origin_group.agents)
        self.assertIn(agent_to_switch, dest_group.agents)

        self.assertEqual(len(origin_group.agents), 20)
        self.assertEqual(len(dest_group.agents), 22)

        self.assertEqual(origin_old_n_agents[agent_to_switch.learning] - 1, origin_group.n_agents[agent_to_switch.learning])
        self.assertEqual(dest_old_n_agents[agent_to_switch.learning] + 1, dest_group.n_agents[agent_to_switch.learning])

        self.assertEqual(origin_group.agents, origin_old_agents[:3] + origin_old_agents[4:])
        self.assertEqual(dest_group.agents, dest_old_agents + [agent_to_switch])

        self.assertEqual(agent_to_switch.group, dest_group)

        origin_group = sm.groups[0]
        dest_group = sm.groups[6]

        for agent in origin_group.agents:
            agent.learning = random.choice(["selfish", "static"])
        
        for agent in dest_group.agents:
            agent.learning = "civic"

        origin_group.recount_agents()
        dest_group.recount_agents()

        origin_old_agents = origin_group.agents.copy()
        dest_old_agents = dest_group.agents.copy()

        self.assertTrue(dest_group.all_civic)
        self.assertTrue(dest_group.mostly_civic)
        self.assertTrue(dest_group.majority_civic)

        for i, agent in enumerate(origin_old_agents):
            self.assertTrue(0 in sm.groups)
            self.assertTrue(origin_group in sm.groups.values())
            self.assertTrue(origin_group.agents)
            self.assertEqual(sm.grid_group_indices[origin_group.location], 0)
            

            self.assertEqual(agent.group, origin_group)
            origin_old_n_agents = origin_group.n_agents.copy()
            dest_old_n_agents = dest_group.n_agents.copy()
            agent.switch_group(6)

            self.assertNotIn(agent, origin_group.agents)
            self.assertIn(agent, dest_group.agents)

            self.assertEqual(len(origin_group.agents), 20 - i)
            self.assertEqual(len(dest_group.agents), 22 + i)

            self.assertEqual(origin_old_n_agents[agent.learning] - 1, origin_group.n_agents[agent.learning])
            self.assertEqual(dest_old_n_agents[agent.learning] + 1, dest_group.n_agents[agent.learning])

            self.assertEqual(dest_group.n_agents["civic"], 21)
            self.assertEqual(dest_group.n_agents["selfish"] + dest_group.n_agents["static"], i + 1)
            self.assertEqual(origin_group.n_agents["selfish"] + origin_group.n_agents["static"], 20 - i)

            self.assertEqual(origin_group.agents, origin_old_agents[i + 1:])
            self.assertEqual(dest_group.agents, dest_old_agents + origin_old_agents[:i + 1])

            self.assertEqual(agent.group, dest_group)

            self.assertTrue(not origin_group.all_civic)
            self.assertTrue(not origin_group.mostly_civic)
            self.assertTrue(not origin_group.majority_civic)

            if i < 2:
                self.assertTrue(not dest_group.all_civic)
                self.assertTrue(dest_group.mostly_civic)
                self.assertTrue(dest_group.majority_civic) 
 
            elif i >= 2 and i < 20:
                self.assertTrue(not dest_group.all_civic)
                self.assertTrue(not dest_group.mostly_civic)
                self.assertTrue(dest_group.majority_civic) 
            elif i == 20:
                self.assertTrue(not dest_group.all_civic)
                self.assertTrue(not dest_group.mostly_civic)
                self.assertTrue(not dest_group.majority_civic) 

        self.assertTrue(0 not in sm.groups)
        self.assertTrue(origin_group not in sm.groups.values())
        self.assertTrue(not origin_group.agents)
        self.assertEqual(sm.grid_group_indices[origin_group.location], -1)

        origin_group = sm.groups[1]
        dest_group = sm.groups[7]

        for agent in origin_group.agents:
            agent.learning = "civic"
        
        for i in range(115 - 21):
            origin_group.add_agent(SpatialAgent(sm, None, learning="civic"))
        
        for i, agent in enumerate(dest_group.agents):
            if i % 2 == 0:
                agent.learning = "selfish"
            else:
                agent.learning = "civic"

        origin_group.recount_agents()
        dest_group.recount_agents()

        origin_old_agents = origin_group.agents.copy()
        dest_old_agents = dest_group.agents.copy()
        
        self.assertTrue(origin_group.all_civic)
        self.assertEqual(len(origin_group.agents), 115)
        for i, agent in enumerate(origin_old_agents):
            self.assertTrue(1 in sm.groups)
            self.assertTrue(origin_group in sm.groups.values())
            self.assertTrue(origin_group.agents)
            self.assertEqual(sm.grid_group_indices[origin_group.location], 1)

            origin_old_n_agents = origin_group.n_agents.copy()
            dest_old_n_agents = dest_group.n_agents.copy()

            agent.switch_group(7)

            self.assertNotIn(agent, origin_group.agents)
            self.assertIn(agent, dest_group.agents)

            self.assertEqual(len(origin_group.agents), 114 - i)
            self.assertEqual(len(dest_group.agents), 22 + i)

            self.assertEqual(origin_old_n_agents[agent.learning] - 1, origin_group.n_agents[agent.learning])
            self.assertEqual(dest_old_n_agents[agent.learning] + 1, dest_group.n_agents[agent.learning])

            self.assertEqual(dest_group.n_agents["civic"], 11 + i)
            self.assertEqual(dest_group.n_agents["selfish"] + dest_group.n_agents["static"], 11)
            self.assertEqual(origin_group.n_agents["civic"], 114 - i)

            self.assertEqual(origin_group.agents, origin_old_agents[i + 1:])
            self.assertEqual(dest_group.agents, dest_old_agents + origin_old_agents[:i + 1])

            self.assertEqual(agent.group, dest_group)

            self.assertTrue(origin_group.all_civic)
            self.assertTrue(origin_group.mostly_civic) # should fail when it gets to 0
            self.assertTrue(origin_group.majority_civic)

            if i < 1:
                self.assertTrue(not dest_group.all_civic)
                self.assertTrue(not dest_group.mostly_civic)
                self.assertTrue(not dest_group.majority_civic)
            elif i >= 1 and i < 89:
                self.assertTrue(not dest_group.all_civic)
                self.assertTrue(not dest_group.mostly_civic)
                self.assertTrue(dest_group.majority_civic)
            elif i >= 89:
                self.assertTrue(not dest_group.all_civic)
                self.assertTrue(dest_group.mostly_civic)
                self.assertTrue(dest_group.majority_civic)
            
        self.assertTrue(1 not in sm.groups)
        self.assertTrue(origin_group not in sm.groups.values())
        self.assertTrue(not origin_group.agents)
        self.assertEqual(sm.grid_group_indices[origin_group.location], -1)

    # test SpatialAgent.choose_square
    def testChooseSquare(self):
        # without migration
        sm = SpatialModel(n=50, g=8, size=4, p_swap=0, write_log=False)
        old_forager_grid_next = np.array([[[17, 15, 21,  9,  4],
        [ 4, 11, 10, 24,  4],
        [21, 23, 17,  9, 22],
        [20, 19, 14,  3,  3]],

       [[ 6, 16,  2, 20,  8],
        [ 7,  3,  3,  3,  7],
        [16,  0, 17, 10, 16],
        [12,  8, 18,  8, 11]],

       [[12,  7, 12, 20, 17],
        [11, 20, 23, 18, 20],
        [12,  9, 19, 21, 16],
        [21, 21, 19,  7,  6]],

       [[ 2, 11, 17,  3,  5],
        [ 8,  0, 10,  6,  8],
        [ 9,  6, 13,  3, 23],
        [20, 23,  1, 14, 24]]])

        sm.forager_grid_next.grid = old_forager_grid_next.copy()
        old_forager_grid = sm.forager_grid.grid.copy()

        group = sm.groups[0]
        square = group.location

        # find the square underneath the selected group, if it has no group on it, move one there
        under_square = sm.forager_grid.direction_to_coord(group.location[0], group.location[1], 2)
        if sm.grid_group_indices[under_square] == -1:
            group_under = sm.groups[1]
            sm.grid_group_indices[group_under.location[0], group_under.location[1]] = -1
            group_under.location = under_square
            sm.grid_group_indices[group_under.location[0], group_under.location[1]] = 1
        else:
            group_under = sm.groups[sm.grid_group_indices[under_square]]
        
        old_group_under_agents = group_under.agents.copy()
        old_group_agents = group.agents.copy()
        
        # find the square to the left of the selected group, if it has a group on it, move it away
        left_square = sm.forager_grid.direction_to_coord(group.location[0], group.location[1], 3)
        if not sm.grid_group_indices[left_square] == -1:
            intruding_group = sm.groups[sm.grid_group_indices[left_square]]
            new_square_raveled = np.argmin(sm.grid_group_indices) # flattened index of the array
            new_square = np.unravel_index(new_square_raveled, sm.grid_group_indices.shape)

            sm.grid_group_indices[left_square] = -1
            intruding_group.location = new_square
            sm.grid_group_indices[new_square] = intruding_group.id
        
        agent = group.agents[3]
        agent.choose_square(probs=[0, 0, 1, 0, 0])
        self.assertEqual(agent.foraging_direction, 2)
        square_down = (sm.forager_grid.modular_add(square[0], 1), square[1]) # going downward
        self.assertEqual(agent.square, square_down)

        old_forager_grid_next[square_down[0], square_down[1], 1] += 1
        self.assertTrue((old_forager_grid_next == sm.forager_grid_next.grid).all()) # increment forager_grid_next
        self.assertTrue((old_forager_grid == sm.forager_grid.grid).all()) # the regular forager grid should stay 

        self.assertEqual(agent.group, group)
        self.assertEqual(group_under.agents, old_group_under_agents)
        self.assertEqual(group.agents, old_group_agents)

        agent = group.agents[8]
        agent.choose_square(probs=[0, 0, 0, 1, 0])
        self.assertEqual(agent.foraging_direction, 3)
        square_left = (square[0], sm.forager_grid.modular_add(square[1], -1))
        self.assertEqual(agent.square, square_left)

        old_forager_grid_next[square_left[0], square_left[1], 4] += 1 # agent coming from the right
        self.assertTrue((old_forager_grid_next == sm.forager_grid_next.grid).all()) # increments forager_grid_next
        self.assertTrue((old_forager_grid == sm.forager_grid.grid).all()) # unchanged
        
        self.assertEqual(agent.group, group)
        self.assertEqual(group_under.agents, old_group_under_agents)
        self.assertEqual(group.agents, old_group_agents)

        agent = group.agents[12]
        agent.choose_square(probs=[1, 0, 0, 0, 0])
        self.assertEqual(agent.foraging_direction, 0)
        self.assertEqual(agent.square, square)

        old_forager_grid_next[square[0], square[1], 0] += 1 # agent coming from the right
        self.assertTrue((old_forager_grid_next == sm.forager_grid_next.grid).all()) # increments forager_grid_next
        self.assertTrue((old_forager_grid == sm.forager_grid.grid).all()) # unchanged
        
        down_counter = 0
        left_counter = 0
        stay_counter = 0
        for agent in group.agents:
            agent.choose_square(probs=[1/3, 0, 1/3, 1/3, 0])
            self.assertIn(agent.foraging_direction, [0, 2, 3])

            if agent.foraging_direction == 2:
                self.assertEqual(agent.square, square_down)
                down_counter += 1
            elif agent.foraging_direction == 3:
                self.assertEqual(agent.square, square_left)
                left_counter +=1
            elif agent.foraging_direction == 0:
                self.assertEqual(agent.square, square)
                stay_counter += 1
        
        self.assertEqual(down_counter + left_counter + stay_counter, 50)
        old_forager_grid_next[square_down[0], square_down[1], 1] += down_counter
        old_forager_grid_next[square_left[0], square_left[1], 4] += left_counter
        old_forager_grid_next[square[0], square[1], 0] += stay_counter

        self.assertTrue((old_forager_grid_next == sm.forager_grid_next.grid).all())
        self.assertTrue((old_forager_grid == sm.forager_grid.grid).all())
        
        self.assertIn(group, sm.groups.values())
        self.assertEqual(len(group.agents), 50)

        # with migration
        sm = SpatialModel(n=50, g=15, size=4, p_swap=1, write_log=False)
        group = sm.groups[3]
        sm.forager_grid.grid = np.array([[[ 1,  5, 13,  4, 14],
        [21,  0, 16, 21, 12],
        [21, 21, 24,  2, 23],
        [ 5, 14,  6,  3, 10]],

       [[17,  6,  9, 24, 13],
        [13, 15,  6, 22, 13],
        [ 4,  3, 19, 11,  6],
        [16, 16,  8,  2, 19]],

       [[18,  2,  8,  9,  8],
        [14,  5, 16,  3, 23],
        [14,  9, 14, 17,  9],
        [12, 19, 13, 23,  4]],

       [[ 4, 10, 15,  4, 20],
        [20, 20, 17,  7, 24],
        [ 9,  2,  3, 20, 11],
        [ 8, 14,  0, 20,  1]]])

        sm.forager_grid_next.grid = old_forager_grid_next.copy()
        old_forager_grid = sm.forager_grid.grid.copy()

        group = sm.groups[0]
        square = group.location

        # find the square underneath the selected group, if it has no group on it, move one there
        over_square = sm.forager_grid.direction_to_coord(group.location[0], group.location[1], 1)
        if sm.grid_group_indices[over_square] == -1:
            group_over = sm.groups[14]
            sm.grid_group_indices[group_over.location[0], group_over.location[1]] = -1
            group_over.location = over_square
            sm.grid_group_indices[over_square] = 14
        else:
            group_over = sm.groups[sm.grid_group_indices[over_square]]
        
        old_group_over_agents = group_over.agents.copy()
        old_group_agents = group.agents.copy()
        
        # find the square to the left of the selected group, if it has a group on it, move it away
        right_square = sm.forager_grid.direction_to_coord(group.location[0], group.location[1], 4)
        if not sm.grid_group_indices[right_square] == -1:
            intruding_group = sm.groups[sm.grid_group_indices[right_square]]
            new_square_raveled = np.argmin(sm.grid_group_indices) # flattened index of the array
            new_square = np.unravel_index(new_square_raveled, sm.grid_group_indices.shape)

            sm.grid_group_indices[right_square] = -1
            intruding_group.location = new_square
            sm.grid_group_indices[new_square] = intruding_group.id
        
        agent = group.agents[23]
        agent.choose_square(probs=[0, 1, 0, 0, 0])
        self.assertEqual(agent.foraging_direction, 1)
        self.assertTrue(agent.just_migrated)
        square_up = (sm.forager_grid.modular_add(square[0], -1), square[1]) # going upward
        self.assertEqual(agent.square, square_up)

        old_forager_grid_next[square_up[0], square_up[1], 0] += 1 # these agents migratec
        self.assertTrue((old_forager_grid_next == sm.forager_grid_next.grid).all()) # increment forager_grid_next
        self.assertTrue((old_forager_grid == sm.forager_grid.grid).all()) # the regular forager grid should stay 

        self.assertEqual(agent.group, group_over)
        self.assertEqual(group_over.agents, old_group_over_agents + [agent])
        self.assertEqual(group.agents, old_group_agents[:23] + old_group_agents[24:])

        agent = group.agents[43]
        agent.choose_square(probs=[0, 0, 0, 0, 1])
        self.assertEqual(agent.foraging_direction, 4)
        self.assertFalse(agent.just_migrated)
        square_right = (square[0], sm.forager_grid.modular_add(square[1], 1))
        self.assertEqual(agent.square, square_right)

        old_forager_grid_next[square_right[0], square_right[1], 3] += 1 # agent coming from the left
        self.assertTrue((old_forager_grid_next == sm.forager_grid_next.grid).all()) # increments forager_grid_next
        self.assertTrue((old_forager_grid == sm.forager_grid.grid).all()) # unchanged

        self.assertEqual(agent.group, group)
        self.assertEqual(group.agents, old_group_agents[:23] + old_group_agents[24:])

        agent = group.agents[12]
        agent.choose_square(probs=[1, 0, 0, 0, 0])
        self.assertEqual(agent.foraging_direction, 0)
        self.assertEqual(agent.square, square)

        old_forager_grid_next[square[0], square[1], 0] += 1 # agent coming from the right
        self.assertTrue((old_forager_grid_next == sm.forager_grid_next.grid).all()) # increments forager_grid_next
        self.assertTrue((old_forager_grid == sm.forager_grid.grid).all()) # unchanged

        self.assertEqual(agent.group, group)
        self.assertEqual(group.agents, old_group_agents[:23] + old_group_agents[24:])

        # self.assertEqual(agent.foraging_direction, 4)        
        up_counter = 0
        right_counter = 0
        stay_counter = 0
        old_group_agents = group.agents.copy()
        for agent in old_group_agents:
            old_group_over_agents = group_over.agents.copy()
            agent.choose_square(probs=[1/3, 1/3, 0, 0, 1/3])
            self.assertIn(agent.foraging_direction, [0, 1, 4])
            
            if agent.foraging_direction == 1:
                up_counter += 1
                self.assertEqual(agent.square, square_up)
                self.assertEqual(agent.group, group_over)
                self.assertEqual(group_over.agents, old_group_over_agents + [agent])
            elif agent.foraging_direction == 4:
                right_counter +=1
                self.assertEqual(agent.square, square_right)
                self.assertEqual(agent.group, group)
                self.assertEqual(len(group.agents), 49 - up_counter)
            elif agent.foraging_direction == 0:
                stay_counter += 1
                self.assertEqual(agent.square, square)
                
            self.assertEqual(len(group.agents), 49 - up_counter)
        
        self.assertEqual(up_counter + right_counter + stay_counter, 49)
        old_forager_grid_next[square_up[0], square_up[1], 0] += up_counter
        old_forager_grid_next[square_right[0], square_right[1], 3] += right_counter
        old_forager_grid_next[square[0], square[1], 0] += stay_counter

        self.assertTrue((old_forager_grid_next == sm.forager_grid_next.grid).all())
        self.assertTrue((old_forager_grid == sm.forager_grid.grid).all())
        
        self.assertEqual(len(group.agents), 49 - up_counter)
        self.assertEqual(len(group_over.agents), 51 + up_counter)

    def testChooseCoop(self):
        sm = SpatialModel(n=100, g=80, size=10, epsilon=0.1, write_log=False)
        
        # test the p_obs line 
        p_obs_arr = [0.5]
        expected_arr = []

        for group in sm.groups.values():
            for agent in group.agents:
                # test of the p_obs are distributed correctly
                agent.square = group.location
                agent.choose_coop()
                p_obs = agent.p_obs
                last_p_obs = p_obs_arr[len(p_obs_arr) - 1]
                self.assertNotEqual(last_p_obs, p_obs)
                p_obs_arr.append(p_obs)

                # check if coop deviates with expected frequency
                expected_coop = agent.pi > 0.5
                expected_arr.append(expected_coop == agent.cooperate)

        
        avg_p_obs = sum(p_obs_arr)/len(p_obs_arr)
        var_p_obs = sum([(p_obs - avg_p_obs)**2 for p_obs in p_obs_arr])/len(p_obs_arr)
        self.assertTrue(avg_p_obs < 0.53 and avg_p_obs > 0.47) # PROB
        self.assertTrue(var_p_obs > 0.073 and var_p_obs < 0.093) # PROB
        
        conform_percent = sum(expected_arr)/len(expected_arr)
        self.assertTrue(conform_percent < 0.92 and conform_percent > 0.88) # PROB
        self.assertTrue(agent.p_obs < 1 and agent.p_obs > -1)

        sm = SpatialModel(n=20, g=10, size=4, epsilon=0.1, write_log=False)
        sm.year = 3

        # config_array
        # [agent_pi, 
        # cost_coop, 
        # cost_distant, 
        # avg_benefit, 
        # p_obs,
        # resources,
        # foraging_direction,
        # forager_grid_entry]
        configs = []
        
        configs.append([-1, 50, 15, 10, 3.2, 0.5, 30, 2, [2,3,4,6,5], True, 0.25, 2.5])
        configs.append([-1, 60, 25, 5, 2.7, 0.2, 35, 0, [20,10,22,12,13], False, 0.454545455, 0])
        configs.append([-0.75, 80, 25, 6, 5.6, 0.4, 27, 4, [2,9,3,1,2], True, 0, 3.952941176])
        configs.append([-0.75, 100, 30, 10, 2.8, 0.8, 15, 0, [2,3,2,1,3], False, 1.363636364, 0])
        configs.append([-0.5, 60, 19, 4, 3.1, 0.6, 30, 0, [2, 4, 8, 1, 1], True, 0.6875, 3.75])
        configs.append([-0.5, 40, 14, 7, 1.4, 0.7, 16, 3, [4,2,1,3,3], False, 0.692307692, 0])
        configs.append([-0.25, 50, 23, 4, 4.5, 0.75, 25, 3, [1, 1, 2, 3, 1], True, 0, 5.706521739])
        configs.append([-0.25, 100, 26, 8, 6.5, 0.5, 20, 4, [1, 0, 0, 2, 1], False, 3, 0])   
        configs.append([0, 60, 22, 12, 3.4, 0.1, 18, 0, [20, 10, 8, 13, 7], True, 0, 0.846394984])
        configs.append([0, 50, 20, 10, 1.1, 0.5, 15, 0, [3, 7, 2, 8, 5], False, 0.6, 0])
        configs.append([0.1, 75, 27, 11, 5.4, 0.12, 18, 2, [2, 3, 1, 0, 4], True, 0, 1.94444444])
        configs.append([0.1, 60, 25, 9, 3.2, 0.1, 17, 3, [3, 5, 7, 2, 4], False, 0.380952381, 0]) 
        configs.append([0.2, 30, 30, 5, 3.1, 0.59, 30, 0, [1, 2, 3, 5, 2], False, 2.307692308, 0])
        configs.append([0.2, 40, 27, 7, 5.6, 0.3, 25, 1, [1,3,3,1,1], True, 0, 2.962962963])
        configs.append([0.3, 60, 32, 11, 2.9, 0.58, 35, 0, [2, 4, 1, 3, 3], False, 2.692307692, 0])
        configs.append([0.3, 50, 10, 20, 7.5, 0.02, 25, 3, [4,7,4,8,5], True, 0, 0.892857143])
        configs.append([0.4, 50, 20, 23, 0.85, 0.9, 29, 0, [3, 1, 4, 2, 6], True, 0.5625, 3.125])
        configs.append([0.4, 67, 25, 22, 2.6, 0.3, 30, 0, [2, 6, 7, 1, 3], False, 1.578947368, 0])
        configs.append([0.5, 34, 20, 6, 2.6, 0.2, 22, 4, [1,2,4,3,5], False, 1.066666667,0])
        configs.append([0.5, 65, 40, 9, 10.1, 0.1, 45, 0, [2,5,1,7,5], True, 0.25, 3.25])
        configs.append([0.6,60, 30, 10, 4.5, 0.03, 35, 1, [23,7,37,3,20], True, 0, 0.5555555556])
        configs.append([0.6, 55, 20, 14, 0.5, 0.18, 40, 3, [12,14,13,11,30], False, 0.325, 0])
        configs.append([0.7, 79, 50, 11, 0.46, 0.2, 30, 2, [12,13,5,13,17],False, 0.316666667,0])
        configs.append([0.7, 40, 55, 10, 2.1, 0.36, 25, 0, [1, 2, 3, 2,2], True, 0, 1.818181818])
        configs.append([0.8, 55, 25, 14, 0.5, 0.7, 29, 0, [3, 2, 5, 1, 4], True, 0.266666667, 3.666666667])
        configs.append([0.8, 40, 30, 16, 2, 0.069, 23, 4, [2, 1, 3, 1, 3], False, 0.7, 0])
        configs.append([0.9, 70, 40, 12, 8.9, 0.05, 20, 0, [1, 3, 0, 1, 0], True, 0, 7])
        configs.append([0.9, 55, 33, 14, 0.3, 0.8, 25, 0, [2, 3, 1, 4, 0], False, 2.5, 0])
        configs.append([1, 30, 20, 13, 2.2, 0.3, 30, 0, [9,8,3,4,6], True, 0.3333333333, 1])
        configs.append([1, 22, 29, 8, 0.5, 0.01, 35, 1, [1, 2,0,2,0], True, 0, 4.096551724])
        configs.append([1, 25, 32, 9, 0.2, 0.02, 40, 0, [1, 1, 2, 0, 2], True, 1.3333333333, 4.166666667])
        
        for config in configs:
            group = sm.groups[random.choice(range(10))]
            agent = group.agents[random.choice(range(20))]

            agent.square = group.location
            agent.pi = config[0]
            sm.benefit = config[1]
            sm.cost_coop = config[2]
            sm.cost_distant = config[3]
            group.avg_benefit = config[4]
            p_obs = config[5]
            sm.resources = config[6]
            agent.foraging_direction = config[7]
            sm.forager_grid.grid[agent.square[0], agent.square[1],:] = config[8]

            agent.choose_coop(rand=False, p_obs=p_obs)

            self.assertEqual(agent.cooperate, config[9])
            self.assertAlmostEqual(agent.private_benefit, config[10])
            self.assertAlmostEqual(agent.public_benefit, config[11])
        
        sm = SpatialModel(n=100, g=80, size=10, epsilon=0.15, write_log=False)
        sm.year = 3
        expectation_arr = []
        for i, config in enumerate(configs):
            if i % 5 == 0:
                for group in sm.groups.values():
                    for agent in group.agents:
                        agent.square = group.location
                        agent.pi = config[0]
                        sm.benefit = config[1]
                        sm.cost_coop = config[2]
                        sm.cost_distant = config[3]
                        group.avg_benefit = config[4]
                        p_obs = config[5]
                        sm.resources = config[6]
                        agent.foraging_direction = config[7]
                        sm.forager_grid.grid[agent.square[0], agent.square[1],:] = config[8]

                        agent.choose_coop(p_obs=p_obs)
                        expectation_arr.append(agent.cooperate == config[9])

                        if agent.cooperate == config[9]:
                            self.assertAlmostEqual(agent.private_benefit, config[10])
                            self.assertAlmostEqual(agent.public_benefit, config[11])
            
        expectation_average = sum(expectation_arr)/len(expectation_arr)
        self.assertTrue(expectation_average < 0.86 and expectation_average > 0.84) # PROB

        sm = SpatialModel(n=100, g=80, size=10, epsilon=0.15, write_log=False)
        sm.year = 0

        for i, config in enumerate(configs):
            group = sm.groups[random.choice(range(10))]
            agent = group.agents[random.choice(range(20))]
            agent.pi = config[0]
            sm.benefit = config[1]
            sm.cost_coop = config[2]
            sm.cost_distant = config[3]
            group.avg_benefit = 0
            p_obs = config[5]
            sm.resources = config[6]
            agent.foraging_direction = config[7]
            agent.square = group.location
            sm.forager_grid.grid[agent.square[0], agent.square[1],:] = config[8]
    
            agent.choose_coop(rand=False, p_obs=p_obs)

            expected_coop = agent.pi > 0.5
            self.assertEqual(expected_coop, agent.cooperate)

    def testLearn(self):
        # run a test as if it's the first iteration of the model
        sm = SpatialModel(n=20, g=5, size=3, write_log=False)

        sm.square_decisions() 
        sm.coop_decisions()
            
        # calculate payoffs and distribute them
        for group in sm.groups.values():
            group.group_distribution()
        
        for group in sm.groups.values():
            for agent in group.agents:
                self.assertAlmostEqual(agent.pi, agent.avg_pi)
                self.assertAlmostEqual(agent.fitness_diff, agent.avg_fitness_diff)
                old_pi = agent.pi
                
                agent.learn(rand=False)

                if agent.learning == "selfish":
                    self.assertAlmostEqual(agent.pi, old_pi)
                    self.assertAlmostEqual(agent.pi, agent.avg_pi)
                elif agent.learning == "civic":
                    if group.pct_cooperators > sm.threshold:
                        expected_pi = sm.learning_rate + (1 - sm.learning_rate)*old_pi
                    else:
                        expected_pi = (1 - sm.learning_rate)*old_pi

                    self.assertAlmostEqual(agent.pi, expected_pi)
                    self.assertAlmostEqual(agent.avg_pi, sm.present_weight*agent.pi+(1 - sm.present_weight)*old_pi)
                
        
        # run a random trial to check that the mean and variance is as expected
        sm = SpatialModel(n=50000, g=1, size=3, learning_rate=0.05, write_log=False)
        group = sm.groups[0]

        pi_inputs =               [0, 0.2, -0.6, -0.4, 0.3,  0.4, -0.3, -0.1, -0.4, 0.8, random.uniform(-1, 1)]
        avg_pi_inputs =           [0, 0.2, -0.6, -0.4, 0.2, -0.1, -0.4,  0.3, -0.6, 0.5, random.uniform(-1, 1)]

        fitness_diff_inputs =     [0.4,  0.1, -0.2, 0.3, -0.2, 0.5, -0.3, 0.5,  0.1, 0, random.uniform(0, 10)]
        avg_fitness_diff_inputs = [0.1, -0.5, -0.6, 0.3, -0.3, 0.8, -0.3, 0.1, -0.6, 0, random.uniform(0, 10)]

        

        for i in range(11):
            pis = []
            pi = pi_inputs[i]
            avg_pi = avg_pi_inputs[i]
            fitness_diff = fitness_diff_inputs[i]
            avg_fitness_diff = avg_fitness_diff_inputs[i]

            for agent in group.agents:
                agent.learning = "selfish"
                agent.pi = pi
                agent.avg_pi = avg_pi
                agent.fitness_diff = fitness_diff
                agent.avg_fitness_diff = avg_fitness_diff

                agent.learn()
                
                self.assertEqual(agent.avg_pi, sm.present_weight*agent.pi + (1 - sm.present_weight)*avg_pi)
                pis.append(agent.pi)
            
            if avg_fitness_diff + fitness_diff == 0:
                vector = 0
            else:
                vector = sm.learning_rate * (avg_pi - pi) * (avg_fitness_diff - fitness_diff) / (avg_fitness_diff + fitness_diff)
            
            expected_pi_mean = pi + vector

            if vector == 0:
                expected_pi_sd = sm.learning_rate**2
            else:
                expected_pi_sd= abs(vector)

            pi_mean = sum(pis)/len(pis)
            pi_sd = math.sqrt(sum([(rand_pi - pi_mean)**2 for rand_pi in pis])/len(pis))

            # print(pi_mean, expected_pi_mean, sm.learning_rate**2)
            # print(pi_sd, expected_pi_sd, sm.learning_rate**3)

            self.assertTrue(pi_mean < expected_pi_mean + sm.learning_rate**2 and pi_mean > expected_pi_mean - sm.learning_rate**2) # PROB
            self.assertTrue(pi_sd < expected_pi_sd + sm.learning_rate**3 and pi_sd > expected_pi_sd - sm.learning_rate**3) # PROB
        
        learning_types = ["selfish",	"selfish",	"selfish",	"selfish",	"selfish",	"selfish",	"selfish",	"selfish",	"selfish",	"selfish",	"selfish",	"selfish",	"selfish",	"selfish",	"selfish",	"selfish",	"civic",	"civic",	"civic",	"civic",	"civic",	"civic", "static", "static", "static"]
        first_rounds = [True,	False,	True,	False,	False,	False,	False,	False,	True,	False,	False,	False,	True,	True,	False,	False,	False,	False,	True,	False,	True,	False, True, False, False]
        fitness_diffs = [0.46,4.46,5.43,1.61,8.19,6.88,4.46,4.58,6.92,2.43,1.83,4.76,0,0,0,0,0.46,1.61,4.46,2.43,0,0,4.7,2.7,7.5]
        avg_fitness_diffs = [0.2,2.16,1.35,1.15,11.05,9.86,5.19,5.35,6.92,2.43,1.83,4.76,0,0,0,0,0.2,1.15,5.19,2.43,0,0,2.9,2.1,2.6]
        initial_pis = [0.24,-0.12,0.4,0,0.26,-0.39,-0.96,0,-0.12,0.29,0.11,0,-0.38,0.68,0.31,0,0.91,-0.4,-0.41,0.43,0.44,0.63,0.6,0.72,-0.18]
        initial_avg_pis = [0.15,0.4,0.4,0,-0.35,-0.1,-0.96,0,-0.8,0.7,0.11,0,-0.45,0.9,0.31,0,-0.76,0.78,0.94,-0.11,0.49,0.19,0.6,0.72,-0.18]
        group_coops = [0.71,0.18,0.74,0.76,0.15,0.1,0.5,0.1,0.07,0.35,0.39,0.14,0.05,0.37,0.1,0.71,0.89,0.05,0.67,0.38,0.35,0.57,0.15,0.39,0.34] 
        learning_rates = [0.68,0.95,0.27,0.46,0.03,0.45,0.93,0.25,0.71,0.42,0.5,0.63,0.67,0.06,0.01,0.87,0.47,0.97,0.29,0.39,0.94,0.26,0.12,0.67,0.35]
        present_weights = [0.75,0.27,0.33,0.25,0.61,0.77,0.04,0.11,0.7,0.87,0.55,0.7,0.15,0.59,1,0.55,0.31,0.45,0.51,0.26,0.14,0.35,0.7,0.44,0.6]
        thresholds = [0.43,0.05,0.25,0.81,0.53,0.57,0.02,0.22,0.4,0.18,0.61,0.52,0.89,0.96,0.68,0.79,0.87,0.32,0.63,0.66,0.06,0.73,0.41,0.36,0.39]
        
        final_pis = [0.26410909,-0.29163142,0.40000000,0.00000000,0.25727973,-0.36676882,-0.96000000,0.00000000,-0.12000000,0.29000000,0.11000000,0.00000000,-0.38000000,0.68000000,0.31000000,0.00000000,0.95230000,-0.01200000,-0.00110000,0.26230000,0.96640000,0.46620000,0.60000000,0.72000000,-0.18000000]
        final_avg_pis = [0.235581818,0.213259517,0.4,0,0.020440635,-0.305411989,-0.96,0,-0.324,0.3433,0.11,0,-0.4395,0.7702,0.31,0,-0.229187,0.4236,0.460039,-0.013202,0.556696,0.28667,0.6,0.72,-0.18]
        
        sm = SpatialModel(n=25, g=6, write_log=False)

        for group in sm.groups.values():    
            for i, agent in enumerate(group.agents):
                agent.learning = learning_types[i]
                agent.first_round = first_rounds[i]
                agent.fitness_diff = fitness_diffs[i]
                agent.avg_fitness_diff = avg_fitness_diffs[i]
                agent.pi = initial_pis[i]
                agent.avg_pi = initial_avg_pis[i]
                group.pct_cooperators = group_coops[i]
                sm.learning_rate = learning_rates[i]
                sm.present_weight = present_weights[i]
                sm.threshold = thresholds[i]

                agent.learn(rand=False)
                self.assertAlmostEqual(agent.pi, final_pis[i])
                self.assertAlmostEqual(agent.avg_pi, final_avg_pis[i])
                
                # check the intuition for selfish learners
                if agent.learning == "selfish":
                    if initial_pis[i] > initial_avg_pis[i] and fitness_diffs[i] > avg_fitness_diffs[i]:
                        self.assertTrue(agent.pi > initial_pis[i])
                    elif initial_pis[i] > initial_avg_pis[i] and fitness_diffs[i] < avg_fitness_diffs[i]:
                        self.assertTrue(agent.pi < initial_pis[i])
                    elif initial_pis[i] < initial_avg_pis[i] and fitness_diffs[i] > avg_fitness_diffs[i]:
                        self.assertTrue(agent.pi < initial_pis[i])
                    elif initial_pis[i] < initial_avg_pis[i] and fitness_diffs[i] < avg_fitness_diffs[i]:
                        self.assertTrue(agent.pi > initial_pis[i])
                    elif initial_pis[i] == initial_avg_pis[i] or fitness_diffs[i] == avg_fitness_diffs[i]:
                        self.assertTrue(agent.pi == initial_pis[i])

    def testSurvives(self):
        print("testing survives")
        fitnesses = [7.1,10.3,19.5,5.7,8.4,6.3,4.6,12.3]
        stayin_alive = [6,8,12,5.7,9,10,5,14]
        lifespan = [24,3,18,30,30,15,12,15]
        age = [19,3,20,31,8,15,15,17]
        final_fitness = [1.1,2.3,7.5,0,-0.6,-3.7,-0.4,-1.7]
        final_survives = [True,True,False,False,False,False,False,False]

        sm = SpatialModel(n=8, g=6, write_log=False)

        for group in sm.groups.values():
            for i, agent in enumerate(group.agents):
                agent.fitness = fitnesses[i]
                sm.cost_stayin_alive = stayin_alive[i]
                agent.lifespan = lifespan[i]
                agent.age = age[i]
                
                survives = agent.survives()

                self.assertAlmostEqual(agent.fitness, final_fitness[i])
                self.assertAlmostEqual(agent.age, age[i])
                self.assertAlmostEqual(survives, final_survives[i])
   
    # test SpatialAgent.child
    def testChild(self):
        learning = ["selfish","selfish","civic" ,"civic" ,"static","static"]
        stayin_alive = [2,1,3,2,2,1]
        repro = [1,0.5,2,1,5,3]
        fitness = [3,0.5,6,2.9,7,2]
        pi = [-0.49,1,0.46,0.5,0.36,0.15]
         
        reproduce = [True, False, True, False, True, False]
        final_fitness = [2,0.5,4,2.9,2,2]
        
        # probabilistic tests. 1000 agents in each group to make sure results converge
        for i in range(6):
            sm = SpatialModel(n=1000, g=1, cost_stayin_alive=stayin_alive[i], cost_repro=repro[i], p_mutation=0.1, learning_rate=0.05, size=4, write_log=False)
            group = sm.groups[0]
            
            children = []
            for agent in group.agents:
                agent.learning = learning[i]
                agent.fitness = fitness[i]
                agent.pi = pi[i]
                
                children.append(agent.child())
                self.assertEqual(agent.fitness, final_fitness[i])
            
            if reproduce[i]:
                main_type_count = sum([child.learning == learning[i] for child in children])
                self.assertTrue(main_type_count > 825 and main_type_count < 975)  # PROB

                mean_child_pi = sum([child.pi for child in children])/len(children)
                sd_child_pi = math.sqrt(sum([(child.pi - mean_child_pi)**2 for child in children])/len(children))

                mean_child_lifespan = sum([child.lifespan for child in children])/len(children)
                sd_child_lifespan = math.sqrt(sum([(child.lifespan - mean_child_lifespan)**2 for child in children])/len(children))

                self.assertTrue(mean_child_pi < pi[i] + 0.005 and mean_child_pi > pi[i] - 0.005) # PROB
                self.assertTrue(sd_child_pi < sm.learning_rate + 0.003 and sd_child_pi > sm.learning_rate - 0.003) # PROB

                self.assertTrue(mean_child_lifespan < 55 and mean_child_lifespan > 45) # PROB
                self.assertTrue(sd_child_lifespan < 23 and sd_child_lifespan > 17) # PROB

                for child in children:
                    self.assertEqual(child.age, 0)
                    self.assertEqual(child.fitness, stayin_alive[i])
                    
            else:
                for child in children:
                    self.assertTrue(child is None)
        
        learning = ["selfish","selfish","selfish","selfish","selfish","selfish","selfish","selfish","civic","civic","civic","civic","civic","civic","civic","civic","static","static","static","static","static","static","static","static"]
        stayin_alive = [1,4,3,5,1,0,1,0,3,3,1,2,5,1,4,4,0,1,1,2,5,1,0,4]
        repro_cost = [2,1,3,4,5,1,5,5,1,1,2,4,2,5,1,3,2,3,5,2,1,4,1,1]
        fitness = [7.8,8.1,4.6,0.1,9.4,6.9,7.3,3.1,0.4,9.5,0.9,7.8,3.4,6.1,8.1,0.6,7.9,2.8,4.2,9.1,9.1,2.4,8.7,8.1]
        pi = [-0.75,0.32,-0.93,-0.21,-0.11,0.88,-0.78,0.35,0.7,0.46,0.66,0.8,-0.45,0.84,-0.84,-0.66,0.22,-0.09,-0.91,-0.33,0.17,0.14,0.26,-0.89]
        reproduce = [True,True,False,False,True,True,True,False,False,True,False,True,False,True,True,False,True,False,False,True,True,False,True,True]
        
        final_fitness = [5.8,7.1,4.6,0.1,4.4,5.9,2.3,3.1,0.4,8.5,0.9,3.8,3.4,1.1,7.1,0.6,5.9,2.8,4.2,7.1,8.1,2.4,7.7,7.1]

        for i in range(24):
            sm = SpatialModel(n=15, g=1, cost_stayin_alive=stayin_alive[i], cost_repro=repro_cost[i], p_mutation=0.1, learning_rate=0.05, size=4, write_log=False)
            group = sm.groups[0]
            
            for agent in group.agents:
                agent.learning = learning[i]
                agent.fitness = fitness[i]
                agent.pi = pi[i]

                child = agent.child(rand=False)
                self.assertAlmostEqual(agent.fitness, final_fitness[i])
            
            if reproduce[i]:
                self.assertAlmostEqual(child.age, 0)
                self.assertAlmostEqual(child.pi, pi[i])
                self.assertAlmostEqual(child.fitness, stayin_alive[i])
                self.assertAlmostEqual(child.lifespan, 50) 

            else:
                self.assertTrue(child is None)
                

    # -------------------------------------------------------------
    # SpatialGrid Tests

    # tests add_group in SpatialGrid
    def testGridAddGroup(self):
        sg = SpatialGrid(6, model=SpatialModel(write_log=False))

        # add a few groups to the map, check that the grid has groups in the right place
        sg.add_group((3, 3), 4)
        sg.add_group((2, 1), 8)
        sg.add_group((4, 4), 5)

        for i in range(6):
            for j in range(6):
                for k in range(5):
                    if (i, j) == (3, 3) and k == 0:
                        self.assertEqual(sg.grid[i, j, k], 4)
                    elif (i, j) == (2, 1) and k == 0:
                        self.assertEqual(sg.grid[i, j, k], 8)
                    elif (i, j) == (4, 4) and k == 0:
                        self.assertEqual(sg.grid[i, j, k], 5)
                    else:
                        self.assertEqual(sg.grid[i, j, k], 0)
        
        sg = SpatialGrid(3, model=SpatialModel(write_log=False))
        sg.grid = np.array(range(45)).reshape((3, 3, 5))

        # add groups to a non_empty grid
        sg.add_group((2, 1), 67)
        sg.add_group((1, 0), 48)
        sg.add_group((2, 2), 93)

        for i in range(3):
            for j in range(3):
                for k in range(5):
                    if (i, j) == (2, 1) and k == 0:
                        self.assertEqual(sg.grid[i, j, k], 67)
                    elif (i, j) == (1, 0) and k == 0:
                        self.assertEqual(sg.grid[i, j, k], 48)
                    elif (i, j) == (2, 2) and k == 0:
                        self.assertEqual(sg.grid[i, j, k], 93)
                    else:
                        self.assertEqual(sg.grid[i, j, k], 15*i+5*j+k) # since we numbered the grid sequentially

    # tests SpatialGrid.group_to_bud, which gives the group that should bud given a forager_grid
    def testGroupToBud(self):
        print("testing group to bud")
        sm = SpatialModel(n=16, g=5, size=6, write_log=False)
        
        # no groups should bud
        square = (5,2)
        sm.forager_grid.grid[square[0], square[1], :] = [2, 3, 14, 9, 15]
        self.assertIsNone(sm.forager_grid.group_to_bud(square, sm.n))

        # one eligible - index 3 should bud 
        square = (3,4)
        sm.forager_grid.grid[square[0], square[1], :] = [5, 9, 14, 16, 14]
        self.assertEqual(sm.forager_grid.group_to_bud(square, sm.n), (3, 3))

        # multiple eligible, no tie - index 2 should bud 
        square = (1,5)
        sm.forager_grid.grid[square[0], square[1], :] = [12, 17, 26, 15, 19]
        self.assertEqual(sm.forager_grid.group_to_bud(square, sm.n), (2, 5))

        # multiple eligible, tie - index 1
        square = (0,3)
        sm.forager_grid.grid[square[0], square[1], :] = [12, 18, 17, 16, 18]
        self.assertEqual(sm.forager_grid.group_to_bud(square, sm.n), (5, 3))

        # index 0 - should never be called in this case, but to see if behavior is correct
        square = (5, 5)
        sm.forager_grid.grid[square[0], square[1], :] = [21, 20, 21, 18, 19]
        self.assertEqual(sm.forager_grid.group_to_bud(square, sm.n), (5, 5))
    
    # test SpatialGrid.direction_to_coord
    def testDirectionToCoord(self):
        print("testing direction to coord")
        sm = SpatialModel(size=5, write_log=False)

        # check all five directions relative to each coord
        coord = sm.forager_grid.direction_to_coord(2, 4, 0)
        self.assertEqual(coord, (2, 4))
        coord = sm.forager_grid.direction_to_coord(2, 4, 1)
        self.assertEqual(coord, (1, 4))
        coord = sm.forager_grid.direction_to_coord(2, 4, 2)
        self.assertEqual(coord, (3, 4))
        coord = sm.forager_grid.direction_to_coord(2, 4, 3)
        self.assertEqual(coord, (2, 3))
        coord = sm.forager_grid.direction_to_coord(2, 4, 4)
        self.assertEqual(coord, (2, 0))

        coord = sm.forager_grid.direction_to_coord(4, 1, 0)
        self.assertEqual(coord, (4, 1))
        coord = sm.forager_grid.direction_to_coord(4, 1, 1)
        self.assertEqual(coord, (3, 1))
        coord = sm.forager_grid.direction_to_coord(4, 1, 2)
        self.assertEqual(coord, (0, 1))
        coord = sm.forager_grid.direction_to_coord(4, 1, 3)
        self.assertEqual(coord, (4, 0))
        coord = sm.forager_grid.direction_to_coord(4, 1, 4)
        self.assertEqual(coord, (4, 2))

        coord = sm.forager_grid.direction_to_coord(0, 4, 0)
        self.assertEqual(coord, (0, 4))
        coord = sm.forager_grid.direction_to_coord(0, 4, 1)
        self.assertEqual(coord, (4, 4))
        coord = sm.forager_grid.direction_to_coord(0, 4, 2)
        self.assertEqual(coord, (1, 4))
        coord = sm.forager_grid.direction_to_coord(0, 4, 3)
        self.assertEqual(coord, (0, 3))
        coord = sm.forager_grid.direction_to_coord(0, 4, 4)
        self.assertEqual(coord, (0, 0))

        coord = sm.forager_grid.direction_to_coord(3, 0, 0)
        self.assertEqual(coord, (3, 0))
        coord = sm.forager_grid.direction_to_coord(3, 0, 1)
        self.assertEqual(coord, (2, 0))
        coord = sm.forager_grid.direction_to_coord(3, 0, 2)
        self.assertEqual(coord, (4, 0))
        coord = sm.forager_grid.direction_to_coord(3, 0, 3)
        self.assertEqual(coord, (3, 4))
        coord = sm.forager_grid.direction_to_coord(3, 0, 4)
        self.assertEqual(coord, (3, 1))
    
    # test SpatialGrid.modular_add
    def testModularAdd(self):
        sm = SpatialModel(size=5, write_log=False)
        self.assertEqual(sm.forager_grid.modular_add(3, 4), 2)
        self.assertEqual(sm.forager_grid.modular_add(1, 2), 3)
        self.assertEqual(sm.forager_grid.modular_add(5, 6), 1)
        
        sm = SpatialModel(size=4, write_log=False)
        self.assertEqual(sm.forager_grid.modular_add(0, 1), 1)
        self.assertEqual(sm.forager_grid.modular_add(3, 2), 1)
        self.assertEqual(sm.forager_grid.modular_add(4, 8), 0)
        self.assertEqual(sm.forager_grid.modular_add(3, 1), 0)

        sm = SpatialModel(g=5, size=3, write_log=False)
        self.assertEqual(sm.forager_grid.modular_add(0, 1), 1)
        self.assertEqual(sm.forager_grid.modular_add(1, 1), 2)
        self.assertEqual(sm.forager_grid.modular_add(2, 1), 0)
        self.assertEqual(sm.forager_grid.modular_add(2, 2), 1)
        self.assertEqual(sm.forager_grid.modular_add(3, 4), 1)
    
    # test SpatialGrid.generate_indices
    def testGenerateIndices(self):
        sm = SpatialModel(size=8, write_log=False)

        # wraps around above on both row and col
        row_indices, col_indices = sm.forager_grid.generate_indices(7, 7)
        self.assertTrue((row_indices == [6, 0, 7, 7]).all())
        self.assertTrue((col_indices == [7, 7, 6, 0]).all())

        # doesn't wrap around
        row_indices, col_indices = sm.forager_grid.generate_indices(4, 5)
        self.assertTrue((row_indices == [3, 5, 4, 4]).all())
        self.assertTrue((col_indices == [5, 5, 4, 6]).all())

        # wraps aroudn below on row
        row_indices, col_indices = sm.forager_grid.generate_indices(0, 2)
        self.assertTrue((row_indices == [7, 1, 0, 0]).all())
        self.assertTrue((col_indices == [2, 2, 1, 3]).all())

        # wraps around above on row
        row_indices, col_indices = sm.forager_grid.generate_indices(7, 5)
        self.assertTrue((row_indices == [6, 0, 7, 7]).all())
        self.assertTrue((col_indices == [5, 5, 4, 6]).all())

        # ...
        row_indices, col_indices = sm.forager_grid.generate_indices(4, 7)
        self.assertTrue((row_indices == [3, 5, 4, 4]).all())
        self.assertTrue((col_indices == [7, 7, 6, 0]).all())

        row_indices, col_indices = sm.forager_grid.generate_indices(6, 0)
        self.assertTrue((row_indices == [5, 7, 6, 6]).all())
        self.assertTrue((col_indices == [0, 0, 7, 1]).all())

        row_indices, col_indices = sm.forager_grid.generate_indices(7, 0)
        self.assertTrue((row_indices == [6, 0, 7, 7]).all())
        self.assertTrue((col_indices == [0, 0, 7, 1]).all())
    
    # SpatialGrid.calculate_n_outside
    def testCalculateNOutside(self):
        forager_grid = np.array([[[18, 17, 21, 17,  2],
                                [11, 10,  4,  4, 13],
                                [17, 11, 11, 15, 11],
                                [17,  2,  9,  0,  6]],

                                [[16,  2,  8,  9,  1],
                                    [ 1,  5, 21, 16,  2],
                                    [18, 13, 15, 10, 15],
                                    [ 4, 13, 15,  8, 16]],

                                [[ 3, 20, 18,  9, 20],
                                    [18, 14,  9,  6,  6],
                                    [10, 19, 14, 17,  6],
                                    [17,  9, 12, 11,  5]],

                                [[ 3,  8, 20, 13,  1],
                                    [ 8, 12,  7, 16,  8],
                                    [21, 13,  1, 12, 14],
                                    [21,  7, 18,  9, 14]]])
        
        sm = SpatialModel(g=10, size=4, write_log=False)
        sm.forager_grid.grid = forager_grid

        n_outside = sm.forager_grid.calculate_n_outside(3, 2)
        self.assertEqual(n_outside, 62.75)

        n_outside = sm.forager_grid.calculate_n_outside(0, 3)
        self.assertEqual(n_outside, 66.25)

        n_outside = sm.forager_grid.calculate_n_outside(2, 1)
        self.assertEqual(n_outside, 58)

        forager_grid = np.array(
                        [[[13,  2, 18, 12, 21],
                        [ 2, 16, 19, 18, 16],
                        [17,  3,  7, 11, 10]],

                        [[18,  6, 16,  1, 21],
                        [ 4, 13,  5,  4, 12],
                        [16,  4, 12,  5,  6]],

                        [[ 1, 19,  3,  6,  8],
                            [ 3, 20,  7,  9,  0],
                            [ 8, 21, 19,  5, 12]]])

        sm = SpatialModel(g=7, size=3, write_log=False)
        sm.forager_grid.grid = forager_grid

        n_outside = sm.forager_grid.calculate_n_outside(1, 1)
        self.assertEqual(n_outside, 53.75)

        n_outside = sm.forager_grid.calculate_n_outside(0, 1)
        self.assertEqual(n_outside, 47.75)

        n_outside = sm.forager_grid.calculate_n_outside(1, 2)
        self.assertEqual(n_outside, 53.25)

        n_outside = sm.forager_grid.calculate_n_outside(2, 0)
        self.assertEqual(n_outside, 58)


if __name__ == "__main__":
    unittest.main()