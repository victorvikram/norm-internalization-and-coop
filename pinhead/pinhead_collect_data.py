import pandas as pd
# from simulation import EvoModel
from pinhead_model import PinheadModel

from datetime import datetime
import os

center = {
    "n": 40, 
    "g": 80,
    "distrib":{"miscreant": 0.25, "deceiver": 0.25, "citizen": 0.25, "saint": 0.25},
    "benefit": 3.5,
    "cost": 1,
    "fitness": 1,
    "p_mutation": 0.01,
    "p_con": 1/13,
    "p_mig": 0.03,
    "p_survive": 0.7,
    "epsilon": 0.05,
    "threshold": 0.5,
    "saintly_group": False,
    "years": 50000,
    "rand": True
}

# next up, see how low the proliferation goes for 
distribs = [{"miscreant": 0.49, "deceiver": 0.49, "citizen": 0.00, "saint": 0.02}, {"miscreant": 0.49, "deceiver": 0.49, "citizen": 0.02, "saint": 0.00}, {"miscreant": 0.49, "deceiver": 0.49, "citizen": 0.01, "saint": 0.01}]
for benefit in [3.25]:
    for distrib in distribs:
            for i in range(10):

                if distrib["citizen"] == 0:
                    mut_distrib = {"miscreant": 1/4, "deceiver": 1/4, "citizen": 0, "saint": 1/2}
                elif distrib["saint"] == 0:
                    mut_distrib = {"miscreant": 1/4, "deceiver": 1/4, "citizen": 1/2, "saint": 0}
                else:
                    mut_distrib = {"miscreant": 1/4, "deceiver": 1/4, "citizen": 1/4, "saint": 1/4}
                
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
                    p_mig=center["p_mig"],
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