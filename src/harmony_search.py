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

from turtle import shape
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

def port_risk(X, Z, cor_mx):
    risk = []
    for r in range(X.shape[0]):
        z1 = np.where(Z[r]==1)[0]
        risk_ij = []
        for i, j in combinations(z1, 2):
            risk_ij.append(cor_mx[i, j] * X[r, i] * X[r, j])
        risk_ij = np.array(risk_ij)
        risk.append(risk_ij.sum())
    return np.array(risk)
    
def port_return(x, z, r_mean):
    z1 = np.where(z==1)[0]
    return np.dot(x[z1], r_mean[z1])

def generate_population(pop_size, k, n_assets, lower, upper, min_return, r_mean):
    pop_count = 0
    fail_count = 0
    X = []
    Z = []
    R = []
    while pop_count < pop_size:
        x = np.random.uniform(low=lower, high=upper, size=n_assets)
        z = np.zeros(n_assets)
        z[np.random.choice(n_assets, k, False)] = 1
        x[np.where(z==0)] = 0
        x = normalize(x, z, lower)
        p_return = port_return(x, z, r_mean)
        if check_constraints(x, z, lower, upper, k, min_return, p_return):
            X.append(x)
            Z.append(z)
            R.append(p_return)
            pop_count += 1
        else:
            fail_count += 1
        if fail_count > pop_size * 10:
            pop_count = pop_size


    return np.array(X), np.array(Z), np.array(R)

def check_cardinality(x, z, k):
    z1 = np.where(z==1)[0]
    if len(z1)==k:
        return True
    else:
        return False

def check_quantity(x, z, lower, upper):
    z1 = np.where(z==1)[0]
    if np.all(x[z1]>=lower) and np.all(x[z1]<=upper):
        return True
    else:
        return False

def check_min_return(p_return, min_return):
    if p_return >= min_return:
        return True
    else:
        return False

def check_constraints(x, z, lower, upper, k, min_return, p_return):
    if check_cardinality(x, z, k) and check_quantity(x, z, lower, upper) and check_min_return(p_return, min_return):
        return True
    else:
        return False
        
def harmony_search(parameters):
    """
    Argumentos:
        max_iter (int): número máximo de iterações 
        pop_size (int): tamanho da população
        xxx
        xxx terminar!!!

    """
    max_iter = parameters[0]
    mem_size = parameters[1]
    mem_consider = parameters[2]
    par = parameters[3]
    sigma = parameters[4]
    k = parameters[5]
    min_return = parameters[6]
    port_n = parameters[7]
    lower = parameters[8]
    upper = parameters[9]
    type = parameters[10]
    seed = parameters[11]
    tag = parameters[12]

    l_iter = []
    l_cost = []
    l_risk = []
    l_return = []
    l_par = []
    l_move = []
    l_X = []
    l_Z = []
    l_Q = []

    np.random.seed(seed)
    
    # Step 1 - Inicialização dos parâmetros
    # demais parâmetros inicializados no cabeçalho da função
    port = load_port_files(port_n)
    n_assets, r_mean, r_std, cor_mx = port

    # Step 2 - Inicialização da Memória de Harmonias
    X, Z, Return = generate_population(mem_size, k, n_assets, lower, upper, min_return, r_mean)
    Cost = port_risk(X, Z, cor_mx)

    # Step 3 - Processo de Geração de Soluções
    log_count = 0
    for i in range(max_iter):
        
        x = np.zeros(n_assets)
        z = np.zeros(n_assets)
        m = np.zeros(n_assets)

        for a in range(n_assets):

            # Calculo do move = distribuição normal de variância sigma
            move = np.random.normal(loc=0, scale=sigma)

            # Verifica se usa a memória de harmonia ou deixa aleatoria
            if np.random.uniform() <= mem_consider:
                rand_idx = np.random.randint(0, mem_size)
                x[a] = X[rand_idx, a]
                z[a] = Z[rand_idx, a]
                m[a] = Z[rand_idx, a]

            # Se não, inicializa a variável aleatóriamente
            else:
                x[a] = np.random.uniform(lower, upper)
                z[a] = 1

        # Transformar a solução inviável em viável com relação à cardinalidade
        while z.sum() != k:
            if z.sum() > k:
                a = np.random.choice(np.where(z==1)[0])
                x[a] = 0
                z[a] = 0
                m[a] = 0
            else:
                a = np.random.choice(np.where(z==0)[0])
                x[a] = lower
                z[a] = 1

        # Normaliza a solução para atender ao critério de restrição
        x = normalize(x, z, lower)

        # Verifica se vai aplicar o ajuste do Pitch para as variáveis oriundas da memória    
        for a in np.where(m==1)[0]:
            if np.random.uniform() <= par:
                x[a] = x[a] + move
                z[a] = 1

        # Normaliza a solução para atender ao critério de restrição
        x = normalize(x, z, lower)

        # Calculo do retorno da solução atual
        r = port_return(x, z, r_mean)

        # Verifica novamente as restrições
        if check_constraints(x, z, lower, upper, k, min_return, r):
            
            # Step 4 - Verifica se a solução gerada é melhor que a pior 
            # existente na memória, e caso positivo realiza a substituição

            # Obtém pior solução
            h_worst = np.argmax(Cost) if type=='min' else np.argmin(Cost)
            cost_worst = Cost[h_worst]

            # Calculo solução atual
            cost_actual = port_risk(x.reshape(1,-1), z.reshape(1,-1), cor_mx)

            # Verifica se a solução atual é melhor que a pior e realiza a substituição
            if type=='min':
                improve = True if cost_actual < cost_worst else False
            else:
                improve = True if cost_actual > cost_worst else False
                
            if improve:
                X[h_worst] = x
                Z[h_worst] = z
                Cost[h_worst] = cost_actual
                Return[h_worst] = r

        else:
            improve = None
            # print('Gen pop fail')

        h_best = np.argmin(Cost) if type=='min' else np.argmax(Cost)
        return_best = Return[h_best]
        cost_best = Cost[h_best]
             
        # Log
        if i == 1 or i == max_iter-1:
            l_iter.append(i)
            l_cost.append(cost_best)
            l_return.append(return_best)
            l_par.append(par)
            l_move.append(move)
            l_X.append(X[h_best])
            l_Z.append(Z[h_best])
            l_Q.append(Z[h_best].sum())
            if DEBUG:
                print('{:0>3d} | Q {:.0f} | par {:.3f} | move {:.3f} | cost {:.3f} | return {:.3f}' \
                    .format(i, Z[h_best].sum(), par, move, cost_best, return_best))
        else:
            if log_count >= max_iter / 100:
                log_count = 0
                l_iter.append(i)
                l_cost.append(cost_best)
                l_return.append(return_best)
                l_par.append(par)
                l_move.append(move)
                l_X.append(X[h_best])
                l_Z.append(Z[h_best])
                l_Q.append(Z[h_best].sum())

                if DEBUG:
                    print('{:0>3d} | Q {:.0f} | par {:.3f} | move {:.3f} | cost {:.3f} | return {:.3f}' \
                        .format(i, Z[h_best].sum(), par, move, cost_best, return_best))

        log_count += 1


    log = pd.DataFrame({
        'iter':l_iter,
        'cost':l_cost,
        'return':l_return,
        'par':l_par,
        'move':l_move,
        'X':l_X,
        'Z':l_Z,
        'Q':l_Q,
    })
    log['max_iter'] = max_iter
    log['mem_size'] = mem_size
    log['mem_consider'] = mem_consider
    log['sigma'] = sigma
    log['port_n'] = port_n
    log['k'] = k
    log['seed'] = seed
    log['tag'] = tag

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    mh = 'hs'
    filename = 'log_' + mh + '_' + tag + '_' + timestamp + '.csv'
    Path(LOG_PATH).mkdir(parents=True, exist_ok=True)
    log.to_csv(Path(LOG_PATH, filename), index=False, quotechar='"')

    return log

@ray.remote
def ray_harmony_search(params):
    return harmony_search(params)

def main():

    max_iter = 1000
    mem_size = 100
    mem_consider = 0.7
    par = 0.7
    sigma = 1
    k = 10
    min_return = 0.003
    port_n = 1
    lower = 0.01
    upper = 1
    type = 'min'
    seed = 42
    tag = 'base'

    parameters = [
        max_iter, mem_size, mem_consider, par, 
        sigma, k, min_return, port_n, lower, upper,
        type, seed, tag
        ]

    harmony_search(parameters)

if __name__ == "__main__":
    DEBUG = True
    main()