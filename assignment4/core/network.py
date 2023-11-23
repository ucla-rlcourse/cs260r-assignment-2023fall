import torch
import torch.nn as nn


def normc_initializer(std=1.0):
    def initializer(tensor):
        tensor.data.normal_(0, 1)
        tensor.data *= std / torch.sqrt(tensor.data.pow(2).sum(1, keepdim=True))

    return initializer


class SlimFC(nn.Module):
    """Simple PyTorch version of `linear` function"""

    def __init__(self,
                 in_size,
                 out_size,
                 initializer=None,
                 activation_fn=True,
                 use_bias=True,
                 bias_init=0.0):
        super(SlimFC, self).__init__()
        layers = []
        # Actual Conv2D layer (including correct initialization logic).
        linear = nn.Linear(in_size, out_size, bias=use_bias)
        if initializer:
            initializer(linear.weight)
        if use_bias is True:
            nn.init.constant_(linear.bias, bias_init)
        layers.append(linear)
        if activation_fn:
            activation_fn = nn.ReLU
            layers.append(activation_fn())
        # Put everything in sequence.
        self._model = nn.Sequential(*layers)

    def forward(self, x):
        return self._model(x)


def build_one_mlp(input_size, output_size, hidden_size=256):
    return nn.Sequential(
        SlimFC(
            in_size=input_size,
            out_size=hidden_size,
            initializer=normc_initializer(1.0),
            activation_fn=True
        ),
        SlimFC(
            in_size=hidden_size,
            out_size=hidden_size,
            initializer=normc_initializer(1.0),
            activation_fn=True
        ),
        SlimFC(
            in_size=hidden_size,
            out_size=output_size,
            initializer=normc_initializer(0.01),  # Make the output close to zero, in the beginning!
            activation_fn=False
        )
    )


class PPOModel(nn.Module):
    def __init__(self, input_size, output_size, discrete):
        super(PPOModel, self).__init__()

        # Setup the log std output for continuous action space
        self.discrete = discrete
        self.use_free_logstd = True
        if discrete:
            self.actor_logstd = None
        else:
            output_size = output_size * 2
            self.use_free_logstd = False
        self.policy = build_one_mlp(input_size, output_size)
        self.value = build_one_mlp(input_size, 1)

    def forward(self, input_obs):
        logits = self.policy(input_obs)
        value = self.value(input_obs)
        if self.discrete:
            return logits, value
        else:
            if self.use_free_logstd:
                return logits, self.actor_logstd, value
            else:
                mean, log_std = torch.chunk(logits, 2, dim=-1)
                return mean, log_std, value


class GAILModel(nn.Module):
    def __init__(self, input_size, act_dim, output_size, discrete):
        super(GAILModel, self).__init__()

        # Setup the log std output for continuous action space
        self.discrete = discrete
        self.use_free_logstd = True
        if discrete:
            self.actor_logstd = None
        else:
            output_size = output_size * 2
            self.use_free_logstd = False
        self.policy = build_one_mlp(input_size, output_size)

        self.discriminator = build_one_mlp(input_size + act_dim, 1)  # <<< Add a discriminator network

    def forward(self, input_obs):
        # Unlike PPOModel, we don't return values here! We will compute values when needed.
        logits = self.policy(input_obs)
        if self.discrete:
            return logits
        else:
            if self.use_free_logstd:
                return logits, self.actor_logstd
            else:
                mean, log_std = torch.chunk(logits, 2, dim=-1)
                return mean, log_std

    def compute_prediction(self, obs, act):
        if obs.ndim == 3:
            assert act.ndim == 3
            pred = self.discriminator(torch.concat([obs, act], dim=2))
        else:
            pred = self.discriminator(torch.concat([obs, act], dim=1))
        return torch.sigmoid(pred)  # <<< Output should in [0, 1]

    def get_generator_parameters(self):
        return list(self.policy.parameters())

    def get_discriminator_parameters(self):
        return list(self.discriminator.parameters())
