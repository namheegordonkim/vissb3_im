from typing import List, Any, Type, Sequence, Optional, Union

import gymnasium
import jax
import numpy as np

from utils.env_containers import EnvContainer
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnvObs,
    VecEnvStepReturn,
    VecEnvIndices,
)


class MyVecEnv(VecEnv):

    def __init__(
        self,
        env_container: EnvContainer,
        seed: int,
    ):
        observation_space = gymnasium.spaces.Box(-np.inf, np.inf, (env_container.env.observation_size,))
        action_space = gymnasium.spaces.Box(-1, 1, (env_container.env.action_size,))
        super().__init__(env_container.batch_size, observation_space, action_space)
        self.env_container = env_container
        self.actions = None
        self.state = None
        self.jax_rng = jax.random.PRNGKey(seed)
        self.fake_info = [{} for _ in range(env_container.batch_size)]
        self.trajectory = []

    def reset(self) -> VecEnvObs:
        state = self.env_container.jit_env_reset(self.jax_rng)
        self.state = state
        self.trajectory = [state.pipeline_state.x]
        return np.array(state.obs)

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        next_state = self.env_container.jit_env_step(self.state, self.actions)
        self.state = next_state
        self.trajectory.append(next_state.pipeline_state.x)
        return (
            np.array(self.state.obs),
            np.array(self.state.reward),
            np.array(self.state.done, dtype=bool),
            [{"terminal_observation": to} for to in np.array(self.state.info["terminal_observation"])],
        )

    def close(self) -> None:
        pass

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        pass

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        pass

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs,
    ) -> List[Any]:
        pass

    def env_is_wrapped(self, wrapper_class: Type[gymnasium.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        return [False for _ in range(self.num_envs)]

    def get_images(self) -> Sequence[np.ndarray]:
        pass

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        self.jax_rng = jax.random.PRNGKey(seed)  # Brax takes care of seed changes
        return [seed for _ in range(self.num_envs)]
