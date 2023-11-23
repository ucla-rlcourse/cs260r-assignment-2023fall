import pathlib

# Use dot here to denote importing the file in the folder hosting this file.
from .ppo_trainer import PPOTrainer, PPOConfig

FOLDER_ROOT = pathlib.Path(__file__).parent  # The path to the folder hosting this file.


class Policy:
    """
    This class is the interface where the evaluation scripts communicates with your trained agent.

    You can initialize your model and load weights in the __init__ function. At each environment interactions,
    the batched observation `obs`, a numpy array with shape (Batch Size, Obs Dim=161), will be passed into the __call__
    function. You need to generate the action, a numpy array with shape (Batch Size, Act Dim=2), and return it.

    If you use any external package, please import it here and EXPLICITLY describe how to setup package in the REPORT.
    """

    # FILLED YOUR PREFERRED NAME & UID HERE!
    CREATOR_NAME = "YOUR-PREFERRED-NAME"  # Your preferred name here in a string
    CREATOR_UID = "YOUR-UID"  # Your UID here in a string

    def __init__(self):
        config = PPOConfig()
        self.agent = PPOTrainer(config=config)
        # self.agent.load_w(log_dir=FOLDER_ROOT, suffix="iter275")

    def __call__(self, obs):
        value, action, action_log_prob = self.agent.compute_action(obs)
        action = action.detach().cpu().numpy()
        return action
