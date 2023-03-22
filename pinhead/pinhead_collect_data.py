import pandas as pd
# from simulation import EvoModel
from pinhead_model import PinheadModel

from datetime import datetime
import os

center = {
    "n": 35, 
    "g": 60,
    "distrib":{"miscreant": 0, "deceiver": 0, "citizen": 0, "saint": 0},
    "benefit": 3.5,
    "cost": 1,
    "fitness": 1,
    "p_mutation": 0.01,
    "p_con": 1/13,
    "p_mig": 0.3,
    "p_survive": 0.7,
    "epsilon": 0.05,
    "threshold": 0.5,
    "saintly_group": False,
    "years": 15000,
    "rand": True
}

# next up, see how low the proliferation goes for 
distribs = [{"miscreant": 0, "deceiver": 0, "citizen": 0, "saint": 0.02, "civic": 0, "selfish": 0.49, "static": 0.49}]

for benefit in [3.5]:
    for mig in [0.6]:
        for distrib in distribs:
            for i in range(6):

                if distrib["civic"] == 0:
                    mut_distrib = {"miscreant": 0, "deceiver": 0, "citizen": 0, "saint": 1/3, "civic": 0, "selfish": 1/3, "static": 1/3}
                elif distrib["saint"] == 0:
                    mut_distrib = {"miscreant": 0, "deceiver": 0, "citizen": 0, "saint": 0, "civic": 1/3, "selfish": 1/3, "static": 1/3}
                
                print(benefit, "\n distrib", distrib, "\n mut", mut_distrib, i)
                
                pm = PinheadModel(
                    n=center["n"],
                    g=center["g"],
                    distrib=distrib,
                    mut_distrib=mut_distrib,
                    benefit=benefit,
                    cost=center["cost"],
                    fitness=center["fitness"],
                    p_mutation=center["p_mutation"],
                    p_con=center["p_con"],
                    p_mig=mig,
                    p_survive=center["p_survive"],
                    epsilon=center["epsilon"],
                    saintly_group=center["saintly_group"],
                    years=center["years"],
                    rand=center["rand"],
                    print_stuff=True,
                    log_basic=True,
                    log_groups=False
                )

                pm.main()