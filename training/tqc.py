from sb3_contrib import TQC
from stable_baselines3.common.save_util import recursive_getattr,recursive_setattr
from stable_baselines3.common.save_util import load_from_zip_file, recursive_getattr, save_to_zip_file
from stable_baselines3.common.utils import (
    check_for_correct_spaces,
    get_device,
    get_schedule_fn,
    get_system_info,
    set_random_seed,
    update_learning_rate,
)
import torch as th
from stable_baselines3.common.vec_env.patch_gym import _convert_space, _patch_env
class My_TQC(TQC):
    def __init__(self, policy, env, kl_coef=0.1, **kwargs):
        """
        :param kl_coef: Coefficient (beta) for the KL divergence penalty.
        """
        super().__init__(policy, env, **kwargs)
        self.kl_coef = kl_coef
        if not hasattr(self, "quantiles_total"):
            # Ensure n_quantiles exists (defaults to 25 in TQC)
            if not hasattr(self, "n_quantiles"):
                self.n_quantiles = kwargs.get("policy_kwargs", {}).get("n_quantiles", 25)
            # Ensure n_critics exists (defaults to 2 in TQC)
            if not hasattr(self, "n_critics"):
                self.n_critics = kwargs.get("n_critics", 2)
            
            self.quantiles_total = self.n_quantiles * self.n_critics
        

    def get_custom_target_log_prob(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """
        P_target is a Standard Gaussian N(0, 1).
        Formula: log(1/sqrt(2pi)) - 0.5 * x^2
        """
        # Constant term for normalization (can technically be ignored for optimization, but good for correctness)
        log_std_norm = -0.5 * np.log(2 * np.pi)
        
        # Calculate -0.5 * action^2
        # Sum over action dimensions (dim=1) because probability of independent joint actions 
        # is the product of marginals (sum of logs).
        log_prob = -0.5 * actions.pow(2) + log_std_norm
    
        return log_prob.sum(dim=1, keepdim=True)
        
    def quantile_huber_loss(self, current_quantiles: th.Tensor, target_quantiles: th.Tensor) -> th.Tensor:
        """
        [cite_start]The quantile-regression loss, as described in the QR-DQN and TQC papers[cite: 1, 2].
        """
        # current_quantiles: (batch_size, n_critics * n_quantiles)
        # target_quantiles: (batch_size, n_target_quantiles, 1)

        # We assume n_quantiles is the number of quantiles per critic
        n_quantiles = self.n_quantiles
        
        # current_quantiles shape: (batch_size, n_critics * n_quantiles) -> (batch_size, n_critics * n_quantiles, 1)
        current_quantiles = current_quantiles.unsqueeze(-1)
        
        # pairwise_delta: (batch_size, n_target_quantiles, n_critics * n_quantiles)
        # Note: target_quantiles is (B, n_target, 1), we transpose current to broadcast
        pairwise_delta = target_quantiles - current_quantiles.transpose(1, 2)
        
        abs_pairwise_delta = th.abs(pairwise_delta)
        huber_loss = th.where(abs_pairwise_delta > 1, abs_pairwise_delta - 0.5, pairwise_delta**2 * 0.5)
        
        # Quantile thresholds (tau)
        # shape: (n_critics * n_quantiles,)
        tau = th.arange(self.quantiles_total, device=self.device).float() / self.quantiles_total + 1 / (2 * self.quantiles_total)
        
        # Loss calculation: sum over quantiles, mean over batch
        loss = (th.abs(tau[None, None, :] - (pairwise_delta.detach() < 0).float()) * huber_loss).mean()
        return loss

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # We need to sample from the policy with reparameterization trick
            # to be able to backpropagate gradients through the actor
            new_actions, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimizing the entropy coefficient
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            # --- CRITIC UPDATE (Standard TQC Logic) ---
            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                
                # Compute and cut quantiles for the next state
                # compute target q-values: (n_critics, batch_size, n_quantiles)
                next_quantiles = self.critic_target(replay_data.next_observations, next_actions)
                
                # Sort and drop top k quantiles to control overestimation [cite: 151, 166]
                n_target_quantiles = self.quantiles_total - self.top_quantiles_to_drop_per_net * self.n_critics
                next_quantiles, _ = th.sort(next_quantiles.reshape(batch_size, -1))
                next_quantiles = next_quantiles[:, :n_target_quantiles]

                # Calculate target: r + gamma * (next_Q - alpha * next_log_prob)
                # Note: entropy term is subtracted inside the target calculation [cite: 95]
                target_quantiles = next_quantiles - ent_coef * next_log_prob.reshape(-1, 1)
                target_quantiles = replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_quantiles
                
                # TQC loss expects (batch_size, n_quantiles, 1)
                target_quantiles.unsqueeze_(-1)

            # Get current Quantile estimates
            current_quantiles = self.critic(replay_data.observations, replay_data.actions)
            
            # Compute Critic Loss (Quantile Huber Loss) [cite: 176]
            critic_loss = self.quantile_huber_loss(current_quantiles, target_quantiles)
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # --- ACTOR (POLICY) UPDATE (MODIFIED) ---
            
            # Compute Q-values for the current policy actions
            q_values = self.critic(replay_data.observations, new_actions)
            
            # TQC specific: take mean over all critics and quantiles 
            q_values_mean = q_values.mean(dim=2).mean(dim=0, keepdim=True)

            # --- CUSTOM KL DIVERGENCE PART STARTS HERE ---
            # Calculate log(P_custom(a|s))
            target_log_prob = self.get_custom_target_log_prob(replay_data.observations, new_actions)
            
            # KL(pi || P_custom) = log_pi - log_P_custom
            kl_div_term = log_prob - target_log_prob

            # Original Loss: ent_coef * log_prob - q_values
            # New Loss: Original + kl_coef * KL_divergence
            actor_loss = (ent_coef * log_prob - q_values_mean + self.kl_coef * kl_div_term).mean()
            # --- CUSTOM KL DIVERGENCE PART ENDS HERE ---

            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))


    @classmethod
    def custom_load(  # noqa: C901
        cls,
        saved_data=None,
        env = None,
        device = "auto",
        custom_objects = None,
        print_system_info = False,
        force_reset = True,
        **kwargs,
    ):
        """
        Load the model from a zip-file.
        Warning: ``load`` re-creates the model from scratch, it does not update it in-place!
        For an in-place load use ``set_parameters`` instead.

        :param path: path to the file (or a file-like) where to
            load the agent from
        :param env: the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model) has priority over any saved environment
        :param device: Device on which the code should run.
        :param custom_objects: Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            ``keras.models.load_model``. Useful when you have an object in
            file that can not be deserialized.
        :param print_system_info: Whether to print system info from the saved model
            and the current system info (useful to debug loading issues)
        :param force_reset: Force call to ``reset()`` before training
            to avoid unexpected behavior.
            See https://github.com/DLR-RM/stable-baselines3/issues/597
        :param kwargs: extra arguments to change the model when loading
        :return: new model instance with loaded parameters
        """
        if print_system_info:
            print("== CURRENT SYSTEM INFO ==")
            get_system_info()
        
        data, params, pytorch_variables = load_from_zip_file(
            saved_data,
            device=device,
            custom_objects=custom_objects,
        )

        assert data is not None, "No data found in the saved file"
        assert params is not None, "No params found in the saved file"

        # Remove stored device information and replace with ours
        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]
            # backward compatibility, convert to new format
            if "net_arch" in data["policy_kwargs"] and len(data["policy_kwargs"]["net_arch"]) > 0:
                saved_net_arch = data["policy_kwargs"]["net_arch"]
                if isinstance(saved_net_arch, list) and isinstance(saved_net_arch[0], dict):
                    data["policy_kwargs"]["net_arch"] = saved_net_arch[0]

        if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data["policy_kwargs"]:
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
            )

        if "observation_space" not in data or "action_space" not in data:
            raise KeyError("The observation_space and action_space were not given, can't verify new environments")

        # Gym -> Gymnasium space conversion
        for key in {"observation_space", "action_space"}:
            data[key] = _convert_space(data[key])  # pytype: disable=unsupported-operands

        if env is not None:
            # Wrap first if needed
            env = cls._wrap_env(env, data["verbose"])
            # Check if given env is valid
            check_for_correct_spaces(env, data["observation_space"], data["action_space"])
            # Discard `_last_obs`, this will force the env to reset before training
            # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
            if force_reset and data is not None:
                data["_last_obs"] = None
            # `n_envs` must be updated. See issue https://github.com/DLR-RM/stable-baselines3/issues/1018
            if data is not None:
                data["n_envs"] = env.num_envs
        else:
            # Use stored env, if one exists. If not, continue as is (can be used for predict)
            if "env" in data:
                env = data["env"]

        # pytype: disable=not-instantiable,wrong-keyword-args
        model = cls(
            policy=data["policy_class"],
            env=env,
            device=device,
            _init_setup_model=False,  # type: ignore[call-arg]
        )
        # pytype: enable=not-instantiable,wrong-keyword-args

        # load parameters
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()

        try:
            # put state_dicts back in place
            model.set_parameters(params, exact_match=True, device=device)
        except RuntimeError as e:
            # Patch to load Policy saved using SB3 < 1.7.0
            # the error is probably due to old policy being loaded
            # See https://github.com/DLR-RM/stable-baselines3/issues/1233
            if "pi_features_extractor" in str(e) and "Missing key(s) in state_dict" in str(e):
                model.set_parameters(params, exact_match=False, device=device)
                
            else:
                raise e
        # put other pytorch variables back in place
        if pytorch_variables is not None:
            for name in pytorch_variables:
                # Skip if PyTorch variable was not defined (to ensure backward compatibility).
                # This happens when using SAC/TQC.
                # SAC has an entropy coefficient which can be fixed or optimized.
                # If it is optimized, an additional PyTorch variable `log_ent_coef` is defined,
                # otherwise it is initialized to `None`.
                if pytorch_variables[name] is None:
                    continue
                # Set the data attribute directly to avoid issue when using optimizers
                # See https://github.com/DLR-RM/stable-baselines3/issues/391
                recursive_setattr(model, f"{name}.data", pytorch_variables[name].data)

        # Sample gSDE exploration matrix, so it uses the right device
        # see issue #44
        if model.use_sde:
            model.policy.reset_noise()  # type: ignore[operator]  # pytype: disable=attribute-error
        return model
    
    def custom_save(self,
        exclude= None,
        include= None,
    ) -> None:
        """
        Save all the attributes of the object and the model parameters in a zip-file.

        :param path: path to the file where the rl agent should be saved
        :param exclude: name of parameters that should be excluded in addition to the default ones
        :param include: name of parameters that might be excluded but should be included anyway
        """
        # Copy parameter list so we don't mutate the original dict
        data = self.__dict__.copy()

        # Exclude is union of specified parameters (if any) and standard exclusions
        if exclude is None:
            exclude = []
        exclude = set(exclude).union(self._excluded_save_params())

        # Do not exclude params if they are specifically included
        if include is not None:
            exclude = exclude.difference(include)

        state_dicts_names, torch_variable_names = self._get_torch_save_params()
        all_pytorch_variables = state_dicts_names + torch_variable_names
        for torch_var in all_pytorch_variables:
            # We need to get only the name of the top most module as we'll remove that
            var_name = torch_var.split(".")[0]
            # Any params that are in the save vars must not be saved by data
            exclude.add(var_name)

        # Remove parameter entries of parameters which are to be excluded
        for param_name in exclude:
            data.pop(param_name, None)

        # Build dict of torch variables
        pytorch_variables = None
        if torch_variable_names is not None:
            pytorch_variables = {}
            for name in torch_variable_names:
                attr = recursive_getattr(self, name)
                pytorch_variables[name] = attr

        # Build dict of state_dicts
        params_to_save = self.get_parameters()
        return data,params_to_save,pytorch_variables

# features_extractor.py
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import time
class Obstacle1DCNNExtractor(BaseFeaturesExtractor):
    """
    Expects a flat obs of size 48 with layout:
      [0:32]  -> AE features (from env)
      [32:36] -> nav (kmh, throttle, steer, turn)
      [36:48] -> obstacle sectors (12 values)
    The 12-dim obstacle slice passes through a 1D CNN that is trained with the policy.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        assert observation_space.shape == (48,), \
            f"Expected obs shape (48,), got {observation_space.shape}"
        super().__init__(observation_space, features_dim)

        self.ae_dim = 32
        self.nav_dim = 4
        self.obs_len = 12
        
        # 1D CNN over the obstacle sectors
        self.obs_cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),  # -> (B,16,1)
        )

        concat_in = self.ae_dim + self.nav_dim + 16  # => 32 + 4 + 16
        self.mlp = nn.Sequential(
            nn.Linear(concat_in, features_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs shape: (B, 48)
        ae   = obs[:, :self.ae_dim]                                 # (B,32)
        nav  = obs[:, self.ae_dim:self.ae_dim+self.nav_dim]         # (B,4)
        obst = obs[:, self.ae_dim+self.nav_dim:self.ae_dim+self.nav_dim+self.obs_len]  # (B,12)

        # (B,12) -> (B,1,12) -> CNN -> (B,16,1) -> (B,16)
        obst_cnn = self.obs_cnn(obst.unsqueeze(1)).squeeze(-1)

        x = torch.cat([ae, nav, obst_cnn], dim=1)                   # (B,32+4+16)
        
        return self.mlp(x)                                          # (B, features_dim)
