import time

import ray
from ray import tune
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.optuna import OptunaSearch

from harmony_search import harmony_search

def objective(config):

    pop_init = 'best'
    max_iter = 1000
    pop_size = 100
    mem_size = 10
    local_search = None
    mem_consider = 0.5
    par_min = 0.5
    par_max = 0.5
    bw_min = 0.5
    bw_max = 0.5
    sigma = config['sigma']
    k = 10
    lambda_ = 0.5
    port_n = 1
    lower = 0.01
    upper = 1
    type = 'min'
    seed = 42
    tag = 'base'

    parameters = [
        max_iter, pop_size, mem_size, mem_consider,
        par_min, par_max, bw_min, bw_max, sigma, k,
        lambda_, port_n, lower, upper, type, seed, tag, 
        pop_init, local_search
    ]

    Cost, Risk, Return = harmony_search(parameters)

    return Cost


def run_optuna_tune(smoke_test=False):
    algo = OptunaSearch()
    algo = ConcurrencyLimiter(algo, max_concurrent=32)
    scheduler = AsyncHyperBandScheduler()
    analysis = tune.run(
        objective,
        metric="mean_loss",
        mode="min",
        search_alg=algo,
        scheduler=scheduler,
        num_samples=10 if smoke_test else 100,
        config={
            "steps": 100,
            "width": tune.uniform(0, 20),
            "height": tune.uniform(-100, 100),
            # This is an ignored parameter.
            "activation": tune.choice(["relu", "tanh"])
        })

    print("Best hyperparameters found were: ", analysis.best_config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")
    parser.add_argument(
        "--server-address",
        type=str,
        default=None,
        required=False,
        help="The address of server to connect to if using "
        "Ray Client.")
    args, _ = parser.parse_known_args()
    if args.server_address is not None:
        ray.init(f"ray://{args.server_address}")
    else:
        ray.init(configure_logging=False)

    run_optuna_tune(smoke_test=args.smoke_test)
