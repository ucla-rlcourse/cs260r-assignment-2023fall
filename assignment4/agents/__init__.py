import os
from importlib import import_module

import pathlib

base_path = pathlib.Path(__file__).parent

def load_policies():
    """
    Load Policy classes from agent.py in each subfolder in this `agents` folder.

    Returns:
        A dict whose keys are tuples (UID, name, folder_name) and the values are the Policy class in "agent.py".
    """
    policies = {}

    # Iterate over all items in the base directory
    for folder_name in os.listdir(base_path):
        item_path = os.path.join(base_path, folder_name)

        # Check if the item is a directory and contains a agent.py file
        if os.path.isdir(item_path) and "agent.py" in os.listdir(item_path):
            # Dynamically import the policy module from the subfolder
            module_path = f".{folder_name}.agent"
            policy_module = import_module(module_path, package=__package__)

            # Assuming there is a class named 'Policy' in the agent.py file
            policy_class = getattr(policy_module, 'Policy', None)

            # Add the Policy class instance to the dictionary if it exists
            if policy_class:
                policies[folder_name] = policy_class

    return policies