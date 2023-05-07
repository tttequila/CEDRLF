from gym.envs.registration import register

register(
    id = 'MELD-v0',                                   # Format should be xxx-v0, xxx-v1....
    entry_point = 'MELD.envs:MELD',              # Expalined in envs/__init__.py
)
