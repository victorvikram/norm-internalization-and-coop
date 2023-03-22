from collections import defaultdict
import csv
import os
from datetime import datetime
from pinhead_agent import Strategy
import csv
import os
from datetime import datetime
import json

class Logger:
    def __init__(self, model, directory, param_dict):  
        self.model = model
        self.datadict = {}
        self.datadict["params"] = param_dict
        self.big_coop_counter = 0
        self.small_coop_counter = 0

        new_directory = os.path.join("data", directory)
        if not os.path.exists(new_directory):
            os.mkdir(new_directory)
        trial = 1
        
        # timestamp = datetime.now().strftime("%m%d_%H%M%S")
        
        if self.model.log_groups:
            while os.path.exists(os.path.join(new_directory, f'deet_stats_{trial}.json')):
                trial += 1
            self.stats_json = f'data/{directory}/deet_stats_{trial}.json'
        else:
            while os.path.exists(os.path.join(new_directory, f'aggr_stats_{trial}.json')):
                trial += 1
            self.stats_json = f'data/{directory}/aggr_stats_{trial}.json'

    # Logs stats for each group
    def log_stats(self):
        year = self.model.schedule.year
        total_pop_by_strat = {strat: 0 for strat in Strategy}
        total_fitness_by_strat = {strat: 0 for strat in Strategy}
        total_coop_by_strat = {strat: 0 for strat in Strategy}

        self.datadict[year] = {}
        self.datadict[year]["groups"] = {}
        for group, indivs in self.model.group_table.items():
            # Initialize stats for this group
            fitness_by_strat = {strat: 0 for strat in Strategy}
            coop_by_strat = {strat: 0 for strat in Strategy}
            obs_by_strat = {strat: 0 for strat in Strategy}
            error_by_strat = {strat: 0 for strat in Strategy}

            # Aggregate stats
            for indiv in indivs:
                fitness_by_strat[indiv.strategy] += indiv.fitness
                coop_by_strat[indiv.strategy] += indiv.cooperates
                obs_by_strat[indiv.strategy] += indiv.p_obs
                error_by_strat[indiv.strategy] += (indiv.cooperates == indiv.default_choice)
            
            # Calculate values
            group_stats = {}
            for strat in Strategy:
                group_stats[strat.name.lower()[:3]] = {'pop': group.agent_counts[strat]}
                if group.agent_counts[strat] > 0:
                    # all the average levels for a given strategy
                    group_stats[strat.name.lower()[:3]]['fit'] = round(fitness_by_strat[strat] / group.agent_counts[strat], 2)
                    group_stats[strat.name.lower()[:3]]['coop'] = round(coop_by_strat[strat] / group.agent_counts[strat], 3)
                    group_stats[strat.name.lower()[:3]]['obs'] = round(obs_by_strat[strat] / group.agent_counts[strat], 2)
                    group_stats[strat.name.lower()[:3]]['err'] = round(error_by_strat[strat] / group.agent_counts[strat], 3)

                total_pop_by_strat[strat] += group.agent_counts[strat]
                total_fitness_by_strat[strat] += fitness_by_strat[strat]
                total_coop_by_strat[strat] += coop_by_strat[strat]

            group_stats["beat"] = None if not group.fought else group.enemy.unique_id

            if self.model.log_groups:
                self.datadict[year]["groups"][group.unique_id] = group_stats
        
        zero_counter = 4
        for strat in Strategy:
            self.datadict[year][strat.name.lower()[:3]] = {}
            self.datadict[year][strat.name.lower()[:3]]["pop"] = total_pop_by_strat[strat]
            
            if total_pop_by_strat[strat] > 0:
                zero_counter -= 1
                self.datadict[year][strat.name.lower()[:3]]["fit"] = round(total_fitness_by_strat[strat] / total_pop_by_strat[strat], 2)
                self.datadict[year][strat.name.lower()[:3]]["coop"] = round(total_coop_by_strat[strat] / total_pop_by_strat[strat], 3)
        
        total_coop = sum([total_coop_by_strat[strat] for strat in Strategy])
        total_pop = sum([total_pop_by_strat[strat] for strat in Strategy])
        total_coop_pct = total_coop/total_pop
        self.big_coop_counter += (total_coop_pct > 0.9)
        self.small_coop_counter += (total_coop_pct < 0.25)

        if self.model.print_stuff:
            print(self.model.schedule.year)
            for strat in Strategy:
                strat_name = strat.name.lower()[:3]
                if self.datadict[year][strat_name]["pop"] > 0:
                    print(strat_name, self.datadict[year][strat_name]["pop"] / (self.model.n * self.model.g), self.datadict[year][strat_name]["coop"])


        # if only one agent type remains, then we can terminate the model
        if self.model.until_high and self.big_coop_counter > 1000:
            self.model.can_terminate = True
        
        if self.model.until_low and self.small_coop_counter > 1000:
            self.model.can_terminate = True

        if year == self.model.years - 1 or self.model.can_terminate:
            with open(self.stats_json, 'w') as f:
                json.dump(self.datadict, f)
