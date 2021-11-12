# %%
import pathlib
import os
import multiprocessing as mp
from multi_signal import MultiSignal
from agent_config import agent_configs
from map_config import map_configs
from mdp_config import mdp_configs

from hyperopt import hp, tpe
from hyperopt.pyll.base import scope
from hyperopt.fmin import fmin
from hyperopt import space_eval

import numpy as np
import json

# %%
def run_trial(args, trial, return_rewards=True):
    mdp_config = mdp_configs.get(args['agent'])
    if mdp_config is not None:
        mdp_map_config = mdp_config.get(args['map'])
        if mdp_map_config is not None:
            mdp_config = mdp_map_config
        mdp_configs[args['agent']] = mdp_config

    agt_config = agent_configs[args['agent']]
    agt_map_config = agt_config.get(args['map'])
    if agt_map_config is not None:
        agt_config = agt_map_config
    alg = agt_config['agent']

    if mdp_config is not None:
        agt_config['mdp'] = mdp_config
        management = agt_config['mdp'].get('management')
        if management is not None:    # Save some time and precompute the reverse mapping
            supervisors = dict()
            for manager in management:
                workers = management[manager]
                for worker in workers:
                    supervisors[worker] = manager
            mdp_config['supervisors'] = supervisors

    map_config = map_configs[args['map']]
    num_steps_eps = int((map_config['end_time'] - map_config['start_time']) / map_config['step_length'])
    route = map_config['route']
    if route is not None: route = args['pwd'] + route

    env = MultiSignal(alg.__name__+'-tr'+str(trial),
                        args['map'],
                        args['pwd'] + map_config['net'],
                        agt_config['state'],
                        agt_config['reward'],
                        route=route, step_length=map_config['step_length'], yellow_length=map_config['yellow_length'],
                        step_ratio=map_config['step_ratio'], end_time=map_config['end_time'],
                        max_distance=agt_config['max_distance'], lights=map_config['lights'], gui=args['gui'],
                        log_dir=args['log_dir'], libsumo=args['libsumo'], warmup=map_config['warmup'])

    try:
        agt_config['episodes'] = int(args['eps'] * 0.8)    # schedulers decay over 80% of steps
        agt_config['steps'] = agt_config['episodes'] * num_steps_eps
        agt_config['log_dir'] = args['log_dir'] + env.connection_name + os.sep
        agt_config['num_lights'] = len(env.all_ts_ids)

        # Get agent id's, observation shapes, and action sizes from env
        obs_act = dict()
        for key in env.obs_shape:
            obs_act[key] = [env.obs_shape[key], len(env.phases[key]) if key in env.phases else None]
        agent = alg(agt_config, obs_act, args['map'], trial)

        episodic_rewards = []
        for _ in range(args['eps']):
            ep_reward = 0
            obs = env.reset()
            done = False
            while not done:
                act = agent.act(obs)
                obs, rew, done, info = env.step(act)
                ep_reward += np.asarray(list(rew.values()))
                agent.observe(obs, rew, done, info)
            
            episodic_rewards.append(ep_reward)
                    
        env.close()
    except Exception as e:
        env.close()
        raise e

    if return_rewards:
        return np.asarray(episodic_rewards)

# %%
default_args = dict()

default_args['agent'] = "IDQNPerceiver"
default_args['trials'] = 1
default_args['eps'] = 100
default_args['procs'] = 1
default_args['pwd'] = str(pathlib.Path().absolute()) + os.sep
default_args['log_dir'] = str(pathlib.Path().absolute()) + os.sep + 'logs' + os.sep
default_args['gui'] = False
default_args['libsumo'] = True
default_args['tr'] = 0

maps = ['cologne1', 'cologne3'] # Michal: please, fill in the maps that we wanted to use

# %%
def objective(params):
    args = default_args.copy()
    args.update(**params)
    score = 0
    
    for map in maps:
        args['map'] = map
        rewards = run_trial(args, trial=0, return_rewards=True)
        score += rewards[-5:].mean()

    print("Score {:.3f} params {}".format(score, params))
    return -score

# %%
space = {
    'num_freq_bands': scope.int(hp.quniform("num_freq_bands", 1, 6, 1)),
    'max_freq': hp.uniform('max_freq', 0.1, 10.0),
    'depth': scope.int(hp.quniform("max_depth", 1, 6, 1)),
    'num_latents': scope.int(hp.quniform("num_latents", 4, 64, 1)),
    'latent_dim': scope.int(hp.quniform("latent_dim", 4, 64, 1)),
    'latent_heads': scope.int(hp.quniform("latent_heads", 1, 8, 1)),
    'cross_dim_head': scope.int(hp.quniform("cross_dim_head", 1, 64, 1)),
    'latent_dim_head': scope.int(hp.quniform("latent_dim_head", 1, 64, 1)),
    'attn_dropout': hp.uniform('attn_dropout', 0.0, 0.5),
    'ff_dropout': hp.uniform('ff_dropout', 0.0, 0.5),
    'self_per_cross_attn': scope.int(hp.quniform("self_per_cross_attn", 1, 3, 1)),
}

# %%
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=20 # I don't know how many we can afford to run
        )

# %%
best_params = space_eval(space, best)
print("BEST PARAMS:")
print(best_params)

# %%
with open('best_params.json', 'w') as f:
    json.dump(best_params, f)
