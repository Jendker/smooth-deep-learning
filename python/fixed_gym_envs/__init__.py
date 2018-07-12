from gym.envs.registration import register

register(
    id='MyContinuousCartPole-v0',
    entry_point='fixed_gym_envs.envs.cartpole:ContinuousCartPoleEnv',
    reward_threshold=-1e-6,
    )

register(
    id='MyCartPole-v0',
    entry_point='fixed_gym_envs.envs.cartpole:CartPoleEnv',
    reward_threshold=-1e-6,
    )

register(
    id='MyContinuousMountainCar-v0',
    entry_point='fixed_gym_envs.envs.mountain_car:ContinuousMountainCarEnv',
    reward_threshold=-1e-6,
    )

register(
    id='MyMountainCar-v0',
    entry_point='fixed_gym_envs.envs.mountain_car:MountainCarEnv',
    reward_threshold=-1e-6,
    )

register(
    id='MyPendulum-v0',
    entry_point='fixed_gym_envs.envs.pendulum:PendulumEnv',
    reward_threshold=-1e-6,
    )
