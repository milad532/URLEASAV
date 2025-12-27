# trpo.py
from sb3_contrib import TRPO # <-- CHANGED: Import TRPO from sb3_contrib
from stable_baselines3.common.save_util import recursive_getattr,recursive_setattr
from stable_baselines3.common.utils import (
    check_for_correct_spaces,
    get_device,
    get_system_info,
)
from stable_baselines3.common.vec_env.patch_gym import _convert_space

class My_TRPO(TRPO): # <-- CHANGED: Inherit from TRPO
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
        Load the model from saved data. This custom method is preserved from your
        original code and is compatible with the TRPO class.
        """
        if print_system_info:
            print("== CURRENT SYSTEM INFO ==")
            get_system_info()

        data, params, pytorch_variables = saved_data

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
            data[key] = _convert_space(data[key])

        if env is not None:
            # Wrap first if needed
            env = cls._wrap_env(env, data["verbose"])
            # Check if given env is valid
            check_for_correct_spaces(env, data["observation_space"], data["action_space"])
            # Discard `_last_obs`, this will force the env to reset before training
            if force_reset and data is not None:
                data["_last_obs"] = None
            if data is not None:
                data["n_envs"] = env.num_envs
        else:
            if "env" in data:
                env = data["env"]

        model = cls(
            policy=data["policy_class"],
            env=env,
            device=device,
            _init_setup_model=False,
        )

        # load parameters
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()

        try:
            # put state_dicts back in place
            model.set_parameters(params, exact_match=True, device=device)
        except RuntimeError as e:
            # This patch is specific to old PPO models but is harmless to leave in.
            if "pi_features_extractor" in str(e) and "Missing key(s) in state_dict" in str(e):
                model.set_parameters(params, exact_match=False, device=device)
            else:
                raise e
        # put other pytorch variables back in place
        if pytorch_variables is not None:
            for name in pytorch_variables:
                if pytorch_variables[name] is None:
                    continue
                recursive_setattr(model, f"{name}.data", pytorch_variables[name].data)

        if model.use_sde:
            model.policy.reset_noise()
        return model
    
    def custom_save(self,
        exclude= None,
        include= None,
    ):
        """
        Save all the attributes of the object and the model parameters. This custom
        method is preserved from your original code.
        """
        data = self.__dict__.copy()

        if exclude is None:
            exclude = []
        exclude = set(exclude).union(self._excluded_save_params())

        if include is not None:
            exclude = exclude.difference(include)

        state_dicts_names, torch_variable_names = self._get_torch_save_params()
        all_pytorch_variables = state_dicts_names + torch_variable_names
        for torch_var in all_pytorch_variables:
            var_name = torch_var.split(".")[0]
            exclude.add(var_name)

        for param_name in exclude:
            data.pop(param_name, None)

        pytorch_variables = None
        if torch_variable_names is not None:
            pytorch_variables = {}
            for name in torch_variable_names:
                attr = recursive_getattr(self, name)
                pytorch_variables[name] = attr

        params_to_save = self.get_parameters()
        return data,params_to_save,pytorch_variables


# The feature extractor is algorithm-agnostic and does not need any changes.
# It can remain in this file or be moved to its own file.
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

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
            nn.AdaptiveAvgPool1d(1),
        )

        concat_in = self.ae_dim + self.nav_dim + 16
        self.mlp = nn.Sequential(
            nn.Linear(concat_in, features_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        ae   = obs[:, :self.ae_dim]
        nav  = obs[:, self.ae_dim:self.ae_dim+self.nav_dim]
        obst = obs[:, self.ae_dim+self.nav_dim:self.ae_dim+self.nav_dim+self.obs_len]

        obst_cnn = self.obs_cnn(obst.unsqueeze(1)).squeeze(-1)
        x = torch.cat([ae, nav, obst_cnn], dim=1)
        return self.mlp(x)