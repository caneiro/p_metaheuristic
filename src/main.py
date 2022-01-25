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

    max_iter = [100, 1000, 10000, 100000, 1000000, 10000000]
    mem_size = [100]
    mem_consider = [0.7]
    par = [0.7]
    sigma = [1]
    k = [10]
    # k = list(range(2,11))
    min_return = [0.003]
    port_n = [1]
    lower = [0.01]
    upper = [1]
    type = ['min']
    seed = np.random.choice(range(100), 10, replace=False).tolist()
    tag_ = [tag]

    parameters = [
        max_iter, mem_size, mem_consider, par, 
        sigma, k, min_return, port_n, lower, upper,
        type, seed, tag_
        ]

    parameters = list(product(*parameters))
    random.shuffle(parameters)
    print('Number of parameters combinations: {}'.format(len(parameters)))
    # print(parameters)
    futures = [ray_harmony_search.remote(param) for param in parameters]
    logs = ray.get(futures)

def main():
    for i in tqdm(range(1)):
        benchmarks('tuning_iter', None)

if __name__ == "__main__":
    main()
    # benchmarks(42) 