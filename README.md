# Conscience and cooperation
The official repo for [__Polarize, Catalyze, Stabilize: Conscience and the evolution of cooperation__](https://arxiv.org/abs/2112.11664). What we call the "naturalistic model" in the paper is located in `spatial/` and what we call the "abstract model" is located in `pinhead/`.

The notebook, `data-processing.ipynb` contains all the data analysis that appears in the paper. However, in order to function, it requires data from the model, which is stored [here](https://drive.google.com/drive/folders/191NgPRAGVb0q4hbv9BUPXqfh7lSLAKpv?usp=sharing).

Instructions for transferring folders from the [google drive data folder](https://drive.google.com/drive/folders/191NgPRAGVb0q4hbv9BUPXqfh7lSLAKpv?usp=sharing) to this directory:
1. Copy the `figures` directory to the root. This is where `data-processing.ipynb` will save the figures.
2. Copy the `pickle` directory to the root. This contains a pickled dictionary containing pre-processed data from runs of the model, for easy loading into `data-processing.ipynb`.
3. Copy the `pinhead-data` directory to the `pinhead` directory; rename it `data` -- this is where the pinhead model will save data, and where  `data-processing.ipynb` will read pinhead model data
4. copy the `spatial-data` directory to the `spatial` directory, rename it to `data` -- this is where the spatial model will save data, and where  `data-processing.ipynb` will read spatial model data

The bibtex citation for the paper is below.

```
    @misc{odouard2021polarize,
      title={Polarize, Catalyze, Stabilize: Conscience and the evolution of cooperation}, 
      author={Victor Vikram Odouard and Diana Smirnova and Shimon Edelman},
      year={2021},
      eprint={2112.11664},
      archivePrefix={arXiv},
      primaryClass={q-bio.PE}
    }
````
