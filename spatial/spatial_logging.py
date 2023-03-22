import json
import csv
import os
from datetime import datetime

class Logger:
    def __init__(self, model, directory, param_dict):  
        self.model = model

        new_directory = os.path.join("data", directory)
        if not os.path.exists(new_directory):
            os.mkdir(new_directory)
        trial = 1

        if self.model.log_groups:
            while os.path.exists(os.path.join(new_directory, f'deet_stats_{trial}.json')):
                trial += 1
            self.stats_json = f'data/{directory}/deet_stats_{trial}.json'
        else:
            while os.path.exists(os.path.join(new_directory, f'aggr_stats_{trial}.json')):
                trial += 1
            self.stats_json = f'data/{directory}/aggr_stats_{trial}.json'

        self.datadict = {}
        self.datadict["demographics"] = {"total": 0, "age": 0, "migrated": 0}
        self.datadict["params"] = param_dict
    
    # Logs stats for each group
    def log_stats(self):
        strategies = ["civic", "selfish", "static", "coop"]
        year = self.model.year

        total_pop_by_strat = {strat: 0 for strat in strategies}
        total_fitness_by_strat = {strat: 0 for strat in strategies}
        total_coop_by_strat = {strat: 0 for strat in strategies}
        total_pi_by_strat = {strat: 0 for strat in strategies}
        total_new_agents_by_strat = {strat: 0 for strat in strategies}

        self.datadict[year] = {}
        self.datadict[year]["g"] = len(self.model.groups)

        self.datadict[year]["groups"] = {}
        for id, group in self.model.groups.items():
            # Initialize stats for this group
            fitness_by_strat = {strat: 0 for strat in strategies}
            new_agents_by_strat = {strat: 0 for strat in strategies}
            coop_by_strat = {strat: 0 for strat in strategies}
            pi_by_strat = {strat: 0 for strat in strategies}
            obs_by_strat = {strat: 0 for strat in strategies}
            error_by_strat = {strat: 0 for strat in strategies}
            
            # Aggregate stats
            for indiv in group.agents:
                fitness_by_strat[indiv.learning] += indiv.fitness
                new_agents_by_strat[indiv.learning] += indiv.first_round
                coop_by_strat[indiv.learning] += indiv.cooperate
                pi_by_strat[indiv.learning] += indiv.pi
                obs_by_strat[indiv.learning] += indiv.p_obs
                error_by_strat[indiv.learning] += (indiv.cooperate == indiv.coop_strategy)
            
            # Calculate values
            group_stats = {}
            for strat in strategies:
                group_stats[strat[:3]] = {'pop': group.n_agents[strat]}
                if group.n_agents[strat] > 0:
                    group_stats[strat[:3]]['new'] = new_agents_by_strat[strat]
                    group_stats[strat[:3]]['fit'] = round(fitness_by_strat[strat] / group.n_agents[strat], 2)
                    group_stats[strat[:3]]['coop'] = round(coop_by_strat[strat] / group.n_agents[strat], 3)
                    group_stats[strat[:3]]['pi'] = round(pi_by_strat[strat] / group.n_agents[strat], 2)
                    group_stats[strat[:3]]['obs'] = round(obs_by_strat[strat] / group.n_agents[strat], 3)
                    group_stats[strat[:3]]['err'] = round(error_by_strat[strat] / group.n_agents[strat], 3)
    
                total_pop_by_strat[strat] += group.n_agents[strat]
                total_new_agents_by_strat[strat] += new_agents_by_strat[strat]
                total_fitness_by_strat[strat] += fitness_by_strat[strat]
                total_coop_by_strat[strat] += coop_by_strat[strat]
                total_pi_by_strat[strat] += pi_by_strat[strat]

            if self.model.log_groups:
                group_stats["exp"] = None if not group.just_budded else group.budded_to
                group_stats["bud"] = None if not group.just_budded else self.model.grid_group_indices[group.budded_to]
                self.datadict[year]["groups"][id] = group_stats
        
        zero_counter = 3
        for strat in strategies:
            self.datadict[year][strat[:3]] = {}
            self.datadict[year][strat[:3]]["pop"] = total_pop_by_strat[strat]
            self.datadict[year][strat[:3]]["new"] = total_new_agents_by_strat[strat]
            
            if total_pop_by_strat[strat] > 0:
                zero_counter -= 1
                self.datadict[year][strat[:3]]["fit"] = round(total_fitness_by_strat[strat] / total_pop_by_strat[strat], 2)
                self.datadict[year][strat[:3]]["coop"] = round(total_coop_by_strat[strat] / total_pop_by_strat[strat], 3)
                self.datadict[year][strat[:3]]["pi"] = round(total_pi_by_strat[strat] / total_pop_by_strat[strat], 3)
                print(strat, self.datadict[year][strat[:3]]["pop"], self.datadict[year][strat[:3]]["coop"])

        # if only one agent type remains, then we can terminate the model
        if zero_counter == 2 and self.model.p_mutation == 0:
            self.model.can_terminate = True
        
        if year == self.model.years - 1 or self.model.can_terminate:
            with open(self.stats_json, 'w') as f:
                json.dump(self.datadict, f)