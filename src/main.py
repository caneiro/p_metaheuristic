##### LOCAL SEARCH #####

##### ITA | PG-CTE-S | TE-282 - Meta-heurísticas
##### Professor Dr. Angelo Passaro
##### Aluno: Rafael Caneiro de Oliveira
##### Versao: 0.1
##### Data: 22/10/2021

# conjunto de n ativos A = {a1, ..., an}
# possuem retorno = {r1, ..., rn}
# o portfolio é um vetor X = {x1, ..., xn} sendo xi a fraçao do ativo
# 0 <= xi <= 1 e Soma(xn) = 1
# restricoes de cardinalidade -> kmin e kmax ativos no portfolio
# restricoes de quantidade (fracao) de cada asset ->  dmin e dmax

from itertools import product
import numpy as np
import pandas as pd
import time
from datetime import datetime
from pathlib import Path

from harmony_search import ray_harmony_search

np.set_printoptions(linewidth=100000)

PATH = Path.cwd()
LOG_PATH = Path(PATH, "./data/log/")

import ray
import random
from tqdm.auto import tqdm


def benchmarks(tag, seed=None):
    pop_init = ['best']
    max_iter = [10000]
    pop_size = [100000]
    mem_size = [100]
    mem_consider = [0.9]
    par_min = [0.5]
    par_max = [0.9]
    bw_min = [0.5]
    bw_max = [0.9]
    sigma = [1, 3, 5]
    # k = [10]
    # lambda_ = [0.5]
    k = list(range(2,11))
    lambda_ = (np.round(np.array(list(range(101))) * 0.01, 4)).tolist()
    port_n = [1]
    lower = [0.01]
    upper = [1]
    type = ['min']
    seed = [seed]
    tag_ = [tag]
    local_search = [10]

    parameters = [
        max_iter, pop_size, mem_size, mem_consider,
        par_min, par_max, bw_min, bw_max, sigma, k,
        lambda_, port_n, lower, upper, type, seed, tag_,
        pop_init, local_search
    ]

    parameters = list(product(*parameters))
    random.shuffle(parameters)
    print('Number of parameters combinations: {}'.format(len(parameters)))

    futures = [ray_harmony_search.remote(param) for param in parameters]
    logs = ray.get(futures)

def main():
    for i in tqdm(range(10)):
        benchmarks('problem_solving', None)

if __name__ == "__main__":
    main()
    # benchmarks(42) 