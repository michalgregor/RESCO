import torch
import torch.nn as nn
from perceiver_pytorch import Perceiver
from pfrl.q_functions import DiscreteActionValueHead

from .pfrl_dqn import DQNAgent
from agents.agent import IndependentAgent

class IDQNPerceiver(IndependentAgent):
    def __init__(self, config, obs_act, map_name, thread_number):
        super().__init__(config, obs_act, map_name, thread_number)
        for key in obs_act:
            # obs_space = obs_act[key][0]
            act_space = obs_act[key][1]

            class CustomSlicer(nn.Module):
                def forward(self, x):
                    # y = torch.swapaxes(x, 1, -1).flatten(1, -2) # with axis=1
                    y = torch.swapaxes(x, 1, -1) # with axis=2
                    return y

            model = torch.nn.Sequential(
                CustomSlicer(),
                Perceiver(
                    input_channels = 1,          # number of channels for each token of the input
                    input_axis = 2,              # number of axis for input data (2 for images, 3 for video)
                    num_freq_bands = config["num_freq_bands"],
                    # 2; number of freq bands, with original value (2 * K + 1)
                    max_freq = config["max_freq"],
                    # 5.; maximum frequency, hyperparameter depending on how fine the data is
                    depth = config["depth"],
                    #2; depth of net
                    num_latents = config["num_latents"],
                    # 16; number of latents, or induced set points, or centroids. different papers giving it different names
                    latent_dim = config["latent_dim"],
                    # 16; latent dimension
                    cross_heads = config["cross_heads"],
                    # 1; number of heads for cross attention. paper said 1
                    latent_heads = config["latent_heads"],
                    # 2; number of heads for latent self attention, 8
                    cross_dim_head = config["cross_dim_head"],
                    # 16;
                    latent_dim_head = config["latent_dim_head"],
                    # 8;
                    num_classes = act_space,      # output number of classes
                    attn_dropout = config["attn_dropout"],
                    # 0.5;
                    ff_dropout = config["ff_dropout"],
                    # 0.5;
                    weight_tie_layers = False,    # whether to weight tie layers (optional, as indicated in the diagram)
                    fourier_encode_data = True,  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
                    self_per_cross_attn = config["self_per_cross_attn"]
                    # 1; number of self attention blocks per cross attention
                ),
                DiscreteActionValueHead()
            )
    
            self.agents[key] = DQNAgent(config, act_space, model)
