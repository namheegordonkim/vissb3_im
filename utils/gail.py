import abc
import dataclasses
from typing import Iterable, Iterator, Mapping, Optional, Type, overload

import numpy as np
import torch as th
from torch.nn import functional as F

from imitation.algorithms import base
from imitation.algorithms.adversarial.gail import RewardNetFromDiscriminatorLogit
from imitation.data import types
from imitation.rewards import reward_nets
from imitation.util import logger, util
from stable_baselines3.common import (
    base_class,
    distributions,
    policies,
)
from stable_baselines3.sac import policies as sac_policies


class MyAdversarialTrainer:
    """Base class for adversarial imitation learning algorithms like GAIL and AIRL."""

    _demo_data_loader: Optional[Iterable[types.TransitionMapping]]
    _endless_expert_iterator: Optional[Iterator[types.TransitionMapping]]

    def __init__(
        self,
        *,
        demonstrations: base.AnyTransitions,
        demo_batch_size: int,
        reward_net: reward_nets.RewardNet,
        gen_algo: base_class.BaseAlgorithm,
        demo_minibatch_size: Optional[int] = None,
        n_disc_updates_per_round: int = 2,
        disc_opt_cls: Type[th.optim.Optimizer] = th.optim.Adam,
        disc_opt_kwargs: Optional[Mapping] = None,
    ):
        """Builds AdversarialTrainer.

        Args:
            demonstrations: Demonstrations from an expert (optional). Transitions
                expressed directly as a `types.TransitionsMinimal` object, a sequence
                of trajectories, or an iterable of transition batches (mappings from
                keywords to arrays containing observations, etc).
            demo_batch_size: The number of samples in each batch of expert data. The
                discriminator batch size is twice this number because each discriminator
                batch contains a generator sample for every expert sample.
            gen_algo: The generator RL algorithm that is trained to maximize
                discriminator confusion. Environment and logger will be set to
                `venv` and `custom_logger`.
            reward_net: a Torch module that takes an observation, action and
                next observation tensors as input and computes a reward signal.
            demo_minibatch_size: size of minibatch to calculate gradients over.
                The gradients are accumulated until the entire batch is
                processed before making an optimization step. This is
                useful in GPU training to reduce memory usage, since
                fewer examples are loaded into memory at once,
                facilitating training with larger batch sizes, but is
                generally slower. Must be a factor of `demo_batch_size`.
                Optional, defaults to `demo_batch_size`.
            n_disc_updates_per_round: The number of discriminator updates after each
                round of generator updates in AdversarialTrainer.learn().
            disc_opt_cls: The optimizer for discriminator training.
            disc_opt_kwargs: Parameters for discriminator training.

        Raises:
            ValueError: if the batch size is not a multiple of the minibatch size.
        """
        self.demo_batch_size = demo_batch_size
        self.demo_minibatch_size = demo_minibatch_size or demo_batch_size
        if self.demo_batch_size % self.demo_minibatch_size != 0:
            raise ValueError("Batch size must be a multiple of minibatch size.")
        self._demo_data_loader = None
        self._endless_expert_iterator = None

        self._global_step = 0
        self._disc_step = 0
        self.n_disc_updates_per_round = n_disc_updates_per_round
        self.set_demonstrations(demonstrations)
        self._reward_net = reward_net.to(gen_algo.device)

        # Create graph for optimising/recording stats on discriminator
        self._disc_opt_cls = disc_opt_cls
        self._disc_opt_kwargs = disc_opt_kwargs or {}
        self._disc_opt = self._disc_opt_cls(
            self._reward_net.parameters(),
            **self._disc_opt_kwargs,
        )

    @abc.abstractmethod
    def logits_expert_is_high(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
        log_policy_act_prob: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        """Compute the discriminator's logits for each state-action sample.

        A high value corresponds to predicting expert, and a low value corresponds to
        predicting generator.

        Args:
            state: state at time t, of shape `(batch_size,) + state_shape`.
            action: action taken at time t, of shape `(batch_size,) + action_shape`.
            next_state: state at time t+1, of shape `(batch_size,) + state_shape`.
            done: binary episode completion flag after action at time t,
                of shape `(batch_size,)`.
            log_policy_act_prob: log probability of generator policy taking
                `action` at time t.

        Returns:
            Discriminator logits of shape `(batch_size,)`. A high output indicates an
            expert-like transition.
        """  # noqa: DAR202

    @property
    @abc.abstractmethod
    def reward_train(self) -> reward_nets.RewardNet:
        """Reward used to train generator policy."""

    @property
    @abc.abstractmethod
    def reward_test(self) -> reward_nets.RewardNet:
        """Reward used to train policy at "test" time after adversarial training."""

    def set_demonstrations(self, demonstrations: base.AnyTransitions) -> None:
        self._demo_data_loader = base.make_data_loader(
            demonstrations,
            self.demo_batch_size,
        )
        self._endless_expert_iterator = util.endless_iter(self._demo_data_loader)

    def _next_expert_batch(self) -> Mapping:
        assert self._endless_expert_iterator is not None
        return next(self._endless_expert_iterator)

    def train_disc(
        self,
        *,
        expert_samples: Optional[Mapping] = None,
        gen_samples: Optional[Mapping] = None,
    ):
        """Perform a single discriminator update, optionally using provided samples.

        Args:
            expert_samples: Transition samples from the expert in dictionary form.
                If provided, must contain keys corresponding to every field of the
                `Transitions` dataclass except "infos". All corresponding values can be
                either NumPy arrays or Tensors. Extra keys are ignored. Must contain
                `self.demo_batch_size` samples. If this argument is not provided, then
                `self.demo_batch_size` expert samples from `self.demo_data_loader` are
                used by default.
            gen_samples: Transition samples from the generator policy in same dictionary
                form as `expert_samples`. If provided, must contain exactly
                `self.demo_batch_size` samples. If not provided, then take
                `len(expert_samples)` samples from the generator replay buffer.

        Returns:
            Statistics for discriminator (e.g. loss, accuracy).
        """
        # compute loss
        self._disc_opt.zero_grad()

        batch_iter = self._make_disc_train_batches(
            gen_samples=gen_samples,
            expert_samples=expert_samples,
        )
        for batch in batch_iter:
            disc_logits = self.logits_expert_is_high(
                batch["state"],
                batch["action"],
                batch["next_state"],
                batch["done"],
                batch["log_policy_act_prob"],
            )
            loss = F.binary_cross_entropy_with_logits(
                disc_logits,
                batch["labels_expert_is_one"].float(),
            )

            # Renormalise the loss to be averaged over the whole
            # batch size instead of the minibatch size.
            assert len(batch["state"]) == 2 * self.demo_minibatch_size
            loss *= self.demo_minibatch_size / self.demo_batch_size
            loss.backward()

        # do gradient step
        self._disc_opt.step()
        self._disc_step += 1

    @overload
    def _torchify_array(self, ndarray: np.ndarray) -> th.Tensor: ...

    @overload
    def _torchify_array(self, ndarray: None) -> None: ...

    def _torchify_array(self, ndarray: Optional[np.ndarray]) -> Optional[th.Tensor]:
        if ndarray is not None:
            return th.as_tensor(ndarray, device=self.reward_train.device)
        return None

    def _get_log_policy_act_prob(
        self,
        obs_th: th.Tensor,
        acts_th: th.Tensor,
    ) -> Optional[th.Tensor]:
        """Evaluates the given actions on the given observations.

        Args:
            obs_th: A batch of observations.
            acts_th: A batch of actions.

        Returns:
            A batch of log policy action probabilities.
        """
        if isinstance(self.policy, policies.ActorCriticPolicy):
            # policies.ActorCriticPolicy has a concrete implementation of
            # evaluate_actions to generate log_policy_act_prob given obs and actions.
            _, log_policy_act_prob_th, _ = self.policy.evaluate_actions(
                obs_th,
                acts_th,
            )
        elif isinstance(self.policy, sac_policies.SACPolicy):
            gen_algo_actor = self.policy.actor
            assert gen_algo_actor is not None
            # generate log_policy_act_prob from SAC actor.
            mean_actions, log_std, _ = gen_algo_actor.get_action_dist_params(obs_th)
            assert isinstance(
                gen_algo_actor.action_dist,
                distributions.SquashedDiagGaussianDistribution,
            )  # Note: this is just a hint to mypy
            distribution = gen_algo_actor.action_dist.proba_distribution(
                mean_actions,
                log_std,
            )
            # SAC applies a squashing function to bound the actions to a finite range
            # `acts_th` need to be scaled accordingly before computing log prob.
            # Scale actions only if the policy squashes outputs.
            assert self.policy.squash_output
            scaled_acts = self.policy.scale_action(acts_th.numpy(force=True))
            scaled_acts_th = th.as_tensor(scaled_acts, device=mean_actions.device)
            log_policy_act_prob_th = distribution.log_prob(scaled_acts_th)
        else:
            return None
        return log_policy_act_prob_th

    def _make_disc_train_batches(
        self,
        *,
        gen_samples: Optional[Mapping] = None,
        expert_samples: Optional[Mapping] = None,
    ) -> Iterator[Mapping[str, th.Tensor]]:
        """Build and return training minibatches for the next discriminator update.

        Args:
            gen_samples: Same as in `train_disc`.
            expert_samples: Same as in `train_disc`.

        Yields:
            The training minibatch: state, action, next state, dones, labels
            and policy log-probabilities.

        Raises:
            RuntimeError: Empty generator replay buffer.
            ValueError: `gen_samples` or `expert_samples` batch size is
                different from `self.demo_batch_size`.
        """
        batch_size = self.demo_batch_size

        if expert_samples is None:
            expert_samples = self._next_expert_batch()

        if not (len(gen_samples["obs"]) == len(expert_samples["obs"]) == batch_size):
            raise ValueError(
                "Need to have exactly `demo_batch_size` number of expert and " "generator samples, each. " f"(n_gen={len(gen_samples['obs'])} " f"n_expert={len(expert_samples['obs'])} " f"demo_batch_size={batch_size})",
            )

        # Guarantee that Mapping arguments are in mutable form.
        expert_samples = dict(expert_samples)
        gen_samples = dict(gen_samples)

        # Convert applicable Tensor values to NumPy.
        for field in dataclasses.fields(types.Transitions):
            k = field.name
            if k == "infos":
                continue
            for d in [gen_samples, expert_samples]:
                if isinstance(d[k], th.Tensor):
                    d[k] = d[k].detach().numpy()
        assert isinstance(gen_samples["obs"], np.ndarray)
        assert isinstance(expert_samples["obs"], np.ndarray)

        # Check dimensions.
        assert batch_size == len(expert_samples["acts"])
        assert batch_size == len(expert_samples["next_obs"])
        assert batch_size == len(gen_samples["acts"])
        assert batch_size == len(gen_samples["next_obs"])

        for start in range(0, batch_size, self.demo_minibatch_size):
            end = start + self.demo_minibatch_size
            # take minibatch slice (this creates views so no memory issues)
            expert_batch = {k: v[start:end] for k, v in expert_samples.items()}
            gen_batch = {k: v[start:end] for k, v in gen_samples.items()}

            # Concatenate rollouts, and label each row as expert or generator.
            obs = np.concatenate([expert_batch["obs"], gen_batch["obs"]])
            acts = np.concatenate([expert_batch["acts"], gen_batch["acts"]])
            next_obs = np.concatenate([expert_batch["next_obs"], gen_batch["next_obs"]])
            dones = np.concatenate([expert_batch["dones"], gen_batch["dones"]])
            # notice that the labels use the convention that expert samples are
            # labelled with 1 and generator samples with 0.
            labels_expert_is_one = np.concatenate(
                [
                    np.ones(self.demo_minibatch_size, dtype=int),
                    np.zeros(self.demo_minibatch_size, dtype=int),
                ],
            )

            # Calculate generator-policy log probabilities.
            with th.no_grad():
                obs_th = th.as_tensor(obs, device=self.gen_algo.device)
                acts_th = th.as_tensor(acts, device=self.gen_algo.device)
                log_policy_act_prob = self._get_log_policy_act_prob(obs_th, acts_th)
                if log_policy_act_prob is not None:
                    assert len(log_policy_act_prob) == 2 * self.demo_minibatch_size
                    log_policy_act_prob = log_policy_act_prob.reshape(
                        (2 * self.demo_minibatch_size,),
                    )
                del obs_th, acts_th  # unneeded

            obs_th, acts_th, next_obs_th, dones_th = self.reward_train.preprocess(
                obs,
                acts,
                next_obs,
                dones,
            )
            batch_dict = {
                "state": obs_th,
                "action": acts_th,
                "next_state": next_obs_th,
                "done": dones_th,
                "labels_expert_is_one": self._torchify_array(labels_expert_is_one),
                "log_policy_act_prob": log_policy_act_prob,
            }

            yield batch_dict


class MyGAIL(MyAdversarialTrainer):
    """Generative Adversarial Imitation Learning (`GAIL`_).

    .. _GAIL: https://arxiv.org/abs/1606.03476
    """

    def __init__(
        self,
        *,
        demonstrations: base.AnyTransitions,
        demo_batch_size: int,
        gen_algo: base_class.BaseAlgorithm,
        reward_net: reward_nets.RewardNet,
        **kwargs,
    ):
        """Generative Adversarial Imitation Learning.

        Args:
            demonstrations: Demonstrations from an expert (optional). Transitions
                expressed directly as a `types.TransitionsMinimal` object, a sequence
                of trajectories, or an iterable of transition batches (mappings from
                keywords to arrays containing observations, etc).
            demo_batch_size: The number of samples in each batch of expert data. The
                discriminator batch size is twice this number because each discriminator
                batch contains a generator sample for every expert sample.
            gen_algo: The generator RL algorithm that is trained to maximize
                discriminator confusion. Environment and logger will be set to
                `venv` and `custom_logger`.
            reward_net: a Torch module that takes an observation, action and
                next observation tensor as input, then computes the logits.
                Used as the GAIL discriminator.
            **kwargs: Passed through to `AdversarialTrainer.__init__`.
        """
        # Raw self._reward_net is discriminator logits
        reward_net = reward_net.to(gen_algo.device)
        # Process it to produce output suitable for RL training
        # Applies a -log(sigmoid(-logits)) to the logits (see class for explanation)
        self._processed_reward = RewardNetFromDiscriminatorLogit(reward_net)
        super().__init__(
            demonstrations=demonstrations,
            demo_batch_size=demo_batch_size,
            gen_algo=gen_algo,
            reward_net=reward_net,
            **kwargs,
        )

    def logits_expert_is_high(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
        log_policy_act_prob: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        r"""Compute the discriminator's logits for each state-action sample.

        Args:
            state: The state of the environment at the time of the action.
            action: The action taken by the expert or generator.
            next_state: The state of the environment after the action.
            done: whether a `terminal state` (as defined under the MDP of the task) has
                been reached.
            log_policy_act_prob: The log probability of the action taken by the
                generator, :math:`\log{P(a|s)}`.

        Returns:
            The logits of the discriminator for each state-action sample.
        """
        del log_policy_act_prob
        logits = self._reward_net(state, action, next_state, done)
        assert logits.shape == state.shape[:1]
        return logits

    @property
    def reward_train(self) -> reward_nets.RewardNet:
        return self._processed_reward

    @property
    def reward_test(self) -> reward_nets.RewardNet:
        return self._processed_reward
