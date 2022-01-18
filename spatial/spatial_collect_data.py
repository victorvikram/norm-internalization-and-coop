import pandas as pd
from spatial_model import SpatialModel

from datetime import datetime
import os
from collections import defaultdict

center = {
            "n": 20, 
            "g": 10, 
            "benefit": 65,
            "resources": 20,
            "cost_coop": 20,
            "cost_distant": 5,
            "p_mutation": 0.01,
            "p_swap": 0.003,
            "distrib": [1/3, 1/3, 1/3],
            "years": 20000,
            "size": 10, # CONST
            "cost_stayin_alive": 2, # CONST
            "cost_repro": 2, # CONST
            "threshold": 0.5, # CONST
            "present_weight": 0.3, # CONST
            "learning_rate": 0.05, # CONST
            "epsilon": 0.05, # CONST
            "rand": True # CONST   
        }   
distribs = [[0.49, 0.49, 0.02], [1/2, 1/2, 0]]

for benefit in [55, 60, 65, 70, 75]:
    for distrib in distribs:
            print(benefit, distrib)
            if distrib[2] == 0:
                mut_distrib = [1/2, 1/2, 0]
            else:
                mut_distrib = [1/3, 1/3, 1/3]

            for i in range(3):
                cm = SpatialModel(
                        n=center["n"], 
                        g=center["g"], 
                        benefit=benefit,
                        size=center["size"], 
                        resources=center["resources"],
                        cost_coop=center["cost_coop"],
                        cost_distant=center["cost_distant"],
                        p_mutation=center["p_mutation"],
                        p_swap=center["p_swap"],
                        distrib=distrib,
                        mut_distrib=mut_distrib,
                        years=center["years"],
                        cost_stayin_alive=center["cost_stayin_alive"],
                        cost_repro=center["cost_repro"],
                        threshold=center["threshold"], # CONST
                        present_weight=center["present_weight"], # CONST
                        learning_rate=center["learning_rate"], # CONST
                        epsilon=center["epsilon"], # CONST
                        rand=center["rand"], # CONST 
                        write_log=True,
                        log_groups=(benefit == 65)
                )
                cm.main()