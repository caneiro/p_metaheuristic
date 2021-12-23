##### HARMONY SEARCH #####

##### ITA | PG-CTE-S | TE-282 - Meta-heurísticas
##### Professor Dr. Angelo Passaro
##### Aluno: Rafael Caneiro de Oliveira
##### Versao: 0.1
##### Data: 28/11/2021

# conjunto de n ativos A = {a1, ..., an}
# possuem retorno = {r1, ..., rn}
# o portfolio é um vetor X = {x1, ..., xn} sendo xi a fraçao do ativo
# 0 <= xi <= 1 e Soma(xn) = 1
# restricoes de cardinalidade -> kmin e kmax ativos no portfolio
# restricoes de quantidade (fracao) de cada asset ->  dmin e dmax

import numpy as np
import pandas as pd
import time
import random
from datetime import datetime

np.set_printoptions(linewidth=100000)

from itertools import combinations, product
from pathlib import Path
import ray

from functools import partial

DEBUG = False
SEED = 42

PATH = Path.cwd()
# LOG_PATH = './log'
# RAW_PATH = '../input/portfolio-optimisation-orlibrary-by-beasley'
LOG_PATH = Path(PATH, "./data/log/")
RAW_PATH = Path(PATH, "./data/raw/")

def load_port_files(n_port):
    start_time = time.time()
    filepath = Path(RAW_PATH, 'port' + str(n_port) + '.txt')
    with open(filepath) as fp:
        # quantidade de ativos no portfolio
        n_assets = int(fp.readline())
        # armazena as estatisticas do ativo
        r_mean = []
        r_std = []
        for n in range(n_assets):
            line = fp.readline()
            r_mean.append(float(line.strip().split()[0]))
            r_std.append(float(line.strip().split()[1]))

        # obtem o restante da matriz de covariancia
        cnt = 32
        i = []
        j = []
        cov = []
        line = fp.readline()
        while line:
            i.append(int(line.strip().split(' ')[0]))
            j.append(int(line.strip().split(' ')[1]))
            cov.append(float(line.strip().split(' ')[2]))
            line = fp.readline()
    fp.close()
    # # retorna dataframe com estatisticas dos ativos do portfolio
    # df_stats = pd.DataFrame({'port':n_port,
    #                          'i':[i_+1 for i_ in range(n_assets)],
    #                          'r_mean':r_mean,
    #                          'r_std':r_std})
    # print(df_stats.shape)

    # # retorna dataframe com matriz de covariancia dos ativos do portfolio
    # df_cov_mx = pd.DataFrame({'port':n_port,
    #                          'i':i,
    #                          'j':j,
    #                          'cov':cov})
    # print(df_cov_mx.shape)
    end_time = time.time()
    exec_time = round(end_time - start_time, 3)
    # print('>>> Arquivo port{}.txt | {} ativos | tempo: {} seg'.format(n_port, n_assets, exec_time))
    r_mean = np.array(r_mean)
    r_std = np.array(r_std)
    cov_mx = np.zeros((n_assets, n_assets))
    for i, j, cov in zip(i, j, cov):
        cov_mx[i-1, j-1] = cov
    return n_assets, r_mean, r_std, cov_mx

def normalize(x, z, lower):
    e = np.zeros(x.shape)
    e[np.where(z==1)[0]] = lower
    fator = (1 - np.sum(e)) / np.sum(x)
    if np.isinf(fator):
        print('fator infinito')
    x = (x * fator) + e
    return x

def generate_population(pop_size, k, n_assets, lower, upper):
    pop_count = 0
    X = []
    Z = []
    while pop_count < pop_size:
        x = np.random.uniform(low=lower, high=upper, size=n_assets)
        z = np.zeros(n_assets)
        z[np.random.choice(n_assets, k, False)] = 1
        x[np.where(z==0)] = 0
        x = normalize(x, z, lower)
        if check_constraints(x, z, lower, upper, k):
            X.append(x)
            Z.append(z)
            pop_count = pop_count + 1

    return np.array(X), np.array(Z)

def check_constraints(x, z, lower, upper, k):
    z1 = np.where(z==1)[0]
    if round(x.sum(), 6)==1 and z.sum()==k and np.all(x[z1]>=lower) and np.all(x<=upper):
        return True
    else:
        return False

def portfolio_risk(X, Z, cor_mx, r_std):
    risk = np.zeros(X.shape[0])
    for r in range(X.shape[0]):
        z1 = np.where(Z[r]==1)[0]
        i, j = list(zip(*list(combinations(z1, 2))))
        risk[r] = np.sum(cor_mx[i, j] * X[r, i] * X[r, j]) # * r_std[list(i)] * r_std[list(j)]
    return np.round(risk, 6)

def portfolio_return(X, Z, r_mean):
    p_return = np.dot(X, r_mean)
    return p_return

def cost_function(Risk, Return, lambda_):
    return lambda_ * Risk - ((1-lambda_) * Return)

def harmony_search(parameters):
    """
    Argumentos:
        max_iter (int): número máximo de iterações 
        pop_size (int): tamanho da população
        xxx
        xxx terminar!!!

    """
    max_iter = parameters[0]
    pop_size = parameters[1]
    mem_size = parameters[2]
    mem_consider = parameters[3]
    par_min = parameters[4]
    par_max = parameters[5]
    bw_min = parameters[6]
    bw_max = parameters[7]
    sigma = parameters[8]
    k = parameters[9]
    lambda_ = parameters[10]
    port_n = parameters[11]
    lower = parameters[12]
    upper = parameters[13]
    type = parameters[14]
    seed = parameters[15]

    l_cost = []
    l_risk = []
    l_return = []
    l_par = []
    l_bw = []
    l_move = []
    l_X = []
    l_Z = []
    l_Q = []
    l_iter = []
    l_improve = []

    np.random.seed(seed)
    
    # Step 1 - Inicialização dos parâmetros
    # demais parâmetros inicializados no cabeçalho da função
    port = load_port_files(port_n)
    n_assets, r_mean, r_std, cor_mx = port
    par = par_min
    bw = bw_min
    move = 0

    # Step 2 - Inicialização da Memória de Harmonias
    X, Z = generate_population(pop_size, k, n_assets, lower, upper)
    Risk = portfolio_risk(X, Z, cor_mx, r_std)
    Return = portfolio_return(X, Z, r_mean)
    C = cost_function(Risk, Return, lambda_)

    # Inicializa a Memória com melhores Harmonias
    idx = np.argsort(C)[:mem_size]
    X = X[idx]
    Z = Z[idx]
    Risk = Risk[idx]
    Return = Return[idx]
    C = C[idx]

    # Step 3 - Processo de Geração de Soluções
    for i in range(max_iter):
        
        x, z = generate_population(1, k, n_assets, lower, upper)
        
        for a in range(n_assets):
            # Verifica se usa a memória de harmonia ou deixa aleatoria
            if np.random.uniform() <= mem_consider:
                rand_idx = np.random.randint(0, mem_size)
                x[0, a] = X[rand_idx, a]
                z[0, a] = Z[rand_idx, a]

            # Calculo do PAR - Pitch Adjustment Rate dinâmico
            par = par_min + (((par_max - par_min) / max_iter) * i)
            if np.random.uniform() <= par:

                # Calculo do BW - Bandwith dinâmico
                c = np.log(bw_min / bw_max) / max_iter
                bw = bw_max * np.exp(c * i)
                move = np.random.normal(scale=sigma) * bw
                x[0, a] = x[0, a] + move
                z[0, a] = 1
                
            # Verifica se valor ficou abaixo do mínimo e elimina a variável
            if x[0, a] < lower:
                x[0, a] = 0
                z[0, a] = 0

        # Normaliza a solução para atender ao critério de restrição
        if z[0].sum()==k:
            x[0] = normalize(x[0], z[0], lower)
            if check_constraints(x[0], z[0], lower, upper, k): ##### Melhorar!!! - testes repetidos

                # print('Gen pop succeed')
                # Step 4 - Verifica se a solução gerada é melhor que a pior 
                # existente na memória, e caso positivo realiza a substituição

                # Obtém pior solução
                h_worst = np.argmax(C) if type=='min' else np.argmin(C)
                cost_worst = C[h_worst]

                # Calculo solução atual
                risk_actual = portfolio_risk(x, z, cor_mx, r_std)
                return_actual = portfolio_return(x, z, r_mean)
                cost_actual = cost_function(risk_actual, return_actual, lambda_)

                # Verifica se a solução atual é melhor que a pior e realiza a substituição
                if type=='min':
                    improve = True if cost_actual < cost_worst else False
                else:
                    improve = True if cost_actual > cost_worst else False
                    
                if improve:
                    X[h_worst] = x[0]
                    Z[h_worst] = z[0]
                    C[h_worst] = cost_actual

        else:
            improve = None
            # print('Gen pop fail')

        h_best = np.argmin(C) if type=='min' else np.argmax(C)
        risk_best = Risk[h_best]
        return_best = Return[h_best]
        cost_best = C[h_best]
            
        # print(improve, cost_best)

        l_cost.append(cost_best)
        l_return.append(return_best)
        l_risk.append(risk_best)
        l_par.append(par)
        l_bw.append(bw)
        l_move.append(move)
        l_X.append(X[h_best])
        l_Z.append(Z[h_best])
        l_Q.append(Z[h_best].sum())

    log = pd.DataFrame({
        'iter':list(range(i+1)),
        'cost':l_cost,
        'risk':l_risk,
        'return':l_return,
        'par':l_par,
        'bw':l_bw,
        'move':l_move,
        'X':l_X,
        'Z':l_Z,
        'Q':l_Q,
    })

    log['max_iter'] = max_iter
    log['pop_size'] = pop_size
    log['mem_consider'] = mem_consider
    log['par_min'] = par_min
    log['par_max'] = par_max
    log['bw_min'] = bw_min
    log['bw_max'] = bw_max
    log['sigma'] = sigma
    log['lambda'] = lambda_
    log['port_n'] = port_n
    log['k'] = k
    log['seed'] = seed

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    mh = 'gls'
    filename = 'log_' + mh + '_' + timestamp + '.csv'
    Path(LOG_PATH).mkdir(parents=True, exist_ok=True)
    log.to_csv(Path(LOG_PATH, filename), index=False, quotechar='"')

    return None

@ray.remote
def ray_harmony_search(params):
    return harmony_search(params)

def main():

    max_iter = 1000
    pop_size = 1000
    mem_size = 50
    mem_consider = 0.7
    par_min = 0.5
    par_max = 0.5
    bw_min = 0.5
    bw_max = 0.5
    sigma = 0.5
    k = 10
    lambda_ = 0.7
    port_n = 1
    lower = 0.01
    upper = 1
    type = 'min'
    seed = 42

    parameters = [
        max_iter, pop_size, mem_size, mem_consider,
        par_min, par_max, bw_min, bw_max, sigma, k,
        lambda_, port_n, lower, upper, type, seed,
    ]

    harmony_search(parameters)

def benchmarks(seed=None):

    max_iter = [1000]
    pop_size = [1000]
    mem_size = [50]
    mem_consider = [0.3, 0.5, 0.7]
    par_min = [0.1, 0.3, 0.5]
    par_max = [0.5, 0.7, 0.9]
    bw_min = [0.1, 0.3, 0.5]
    bw_max = [0.5, 0.7, 0.9]
    sigma = [0.1, 0.3, 0.5]
    k = list(range(2,11))
    lambda_ = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    port_n = [1]
    lower = [0.01]
    upper = [1]
    type = ['min']
    seed = [seed]

    parameters = [
        max_iter, pop_size, mem_size, mem_consider,
        par_min, par_max, bw_min, bw_max, sigma, k,
        lambda_, port_n, lower, upper, type, seed,
    ]

    parameters = list(product(*parameters))
    random.shuffle(parameters)
    print('Number of parameters combinations: {}'.format(len(parameters)))

    ray.init(num_cpus=16)

    futures = [ray_harmony_search.remote(param) for param in parameters]
    logs = ray.get(futures)

    ray.shutdown()

if __name__ == "__main__":
    main()