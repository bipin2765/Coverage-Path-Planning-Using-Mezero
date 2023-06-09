import copy

from src.MuZero.Agent_ import MuZeroAgentParams, MuZeroAgent
from src.CPP.Display import CPPDisplay
from src.CPP.Grid import CPPGrid, CPPGridParams
from src.CPP.Physics import CPPPhysics, CPPPhysicsParams
from src.CPP.State import CPPState
from src.MuZero.Trainer import TrainerParams, MuzeroTrainer
from src.CPP.Rewards import CPPRewardParams, CPPRewards
from src.base.Environment import BaseEnvironment, BaseEnvironmentParams
from src.base.GridActions import GridActions


class CPPEnvironmentParams(BaseEnvironmentParams):
    def __init__(self):
        super().__init__()
        self.grid_params = CPPGridParams()
        self.reward_params = CPPRewardParams()
        self.trainer_params = TrainerParams()
        self.agent_params = MuZeroAgentParams()
        self.physics_params = CPPPhysicsParams()


class CPPEnvironment(BaseEnvironment):

    def __init__(self, params: CPPEnvironmentParams):
        self.display = CPPDisplay()
        super().__init__(params, self.display)

        self.grid = CPPGrid(params.grid_params, self.stats)
        self.rewards = CPPRewards(params.reward_params, stats=self.stats)
        self.physics = CPPPhysics(params=params.physics_params, stats=self.stats)
        self.agent = MuZeroAgent(params.agent_params, self.grid.get_example_state(), self.physics.get_example_action(),
                                 stats=self.stats)
        self.trainer = MuzeroTrainer(params=params.trainer_params, agent=self.agent)
