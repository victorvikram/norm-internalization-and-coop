from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from collections import defaultdict

class RandomActivationByLevel(RandomActivation):

    def __init__(self, model):
        super().__init__(model)
        self.agents_by_level = defaultdict(list)
        self.year = 0

    """
    Self -> None
    Executes the steps of each agent level in turn
    """
    def step(self):
        if self.model.print_stuff:
            print("Step:", self.year)

        if self.year != 0:
            # reset agent variables
            for indiv in self.model.indiv_table:
                indiv.is_new_agent = False 
                indiv.migrated = False
                indiv.migration_partner = None
            
            all_groups = list(self.model.group_table.keys())
            for group in all_groups:
                group.step_reproduce()

            self.model.fight_groups()
            self.model.recombine_groups()

        all_indivs = list(self.model.indiv_table.keys())
        self.model.random.shuffle(all_indivs)

        cooperator_count = 0 
        for indiv in all_indivs:
            indiv.step()
            cooperator_count += indiv.cooperates

        all_groups = list(self.model.group_table.keys())
        self.model.random.shuffle(all_groups)
        for group in all_groups:
            group.step_distrib()

        if self.model.log_basic:
            self.model.logger.log_stats()

        self.year += 1