from gymnasium.envs.registration import register

register(
     id="servo_leg-v0",
     entry_point="leg_robot_gym.env:servo_legEnv",
     max_episode_steps=500,
)
register(
     id="servo_leg-v1",
     entry_point="leg_robot_gym.env:servo_legEnv_v1",
     max_episode_steps=500,
)
register(
     id="servo_leg-v2",
     entry_point="leg_robot_gym.env:servo_legEnv_v2",
     max_episode_steps=500,
)
register(
     id="servo_leg-v3",
     entry_point="leg_robot_gym.env:servo_legEnv_v3",
     max_episode_steps=1000,
)