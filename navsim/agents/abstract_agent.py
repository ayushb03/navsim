from abc import ABC, abstractmethod
from typing import Dict, List, Union

import pytorch_lightning as pl
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.dataclasses import AgentInput, Scene, SensorConfig, Trajectory
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder


class AbstractAgent(torch.nn.Module, ABC):
    """Interface for an agent in NAVSIM."""

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        requires_scene: bool = False,
    ):
        super().__init__()
        self.requires_scene = requires_scene
        self._trajectory_sampling = trajectory_sampling

    @abstractmethod
    def name(self) -> str:
        """
        :return: string describing name of this agent.
        """

    @abstractmethod
    def get_sensor_config(self) -> SensorConfig:
        """
        :return: Dataclass defining the sensor configuration for lidar and cameras.
        """

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize agent
        :param initialization: Initialization class.
        """

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the agent.
        :param features: Dictionary of features.
        :return: Dictionary of predictions.
        """
        raise NotImplementedError

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """
        :return: List of target builders.
        """
        raise NotImplementedError("No feature builders. Agent does not support training.")

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """
        :return: List of feature builders.
        """
        raise NotImplementedError("No target builders. Agent does not support training.")

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        """
        Computes the ego vehicle trajectory.
        :param current_input: Dataclass with agent inputs.
        :return: Trajectory representing the predicted ego's position in future
        """
        self.eval()
        features: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        # add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}

        # forward pass
        with torch.no_grad():
            predictions = self.forward(features)
            poses = predictions["trajectory"].squeeze(0).numpy()

        # extract trajectory
        return Trajectory(poses, self._trajectory_sampling)

    def compute_trajectories_batch(self, agent_inputs: List[AgentInput]) -> List[Trajectory]:
        """
        Computes ego vehicle trajectories for multiple scenes in a single batch.
        :param agent_inputs: List of AgentInput dataclasses for multiple scenes.
        :return: List of Trajectory objects for each scene.
        """
        if not agent_inputs:
            return []
        
        # Use single trajectory computation for single input to maintain compatibility
        if len(agent_inputs) == 1:
            return [self.compute_trajectory(agent_inputs[0])]
        
        self.eval()
        
        # Build features for entire batch using parallel feature builders
        batched_features: Dict[str, torch.Tensor] = {}
        for builder in self.get_feature_builders():
            builder_features = builder.compute_features_batch(agent_inputs)
            batched_features.update(builder_features)
        
        # Single forward pass for entire batch
        with torch.no_grad():
            predictions = self.forward(batched_features)
            batch_poses = predictions["trajectory"].numpy()  # Shape: [batch_size, num_poses, 3]
        
        # Extract individual trajectories
        trajectories = []
        for poses in batch_poses:
            trajectories.append(Trajectory(poses, self._trajectory_sampling))
        
        return trajectories

    def compute_trajectories_batch_with_scenes(self, agent_inputs: List[AgentInput], scenes: List[Scene]) -> List[Trajectory]:
        """
        Computes ego vehicle trajectories for multiple scenes with scene data in a batch.
        This method should be overridden by agents that require scene data and support batching.
        
        :param agent_inputs: List of AgentInput dataclasses for multiple scenes.
        :param scenes: List of Scene objects corresponding to each agent input.
        :return: List of Trajectory objects for each scene.
        """
        if not self.requires_scene:
            # For agents that don't require scenes, delegate to regular batch processing
            return self.compute_trajectories_batch(agent_inputs)
        
        # Default implementation: fall back to serial processing for compatibility
        # Scene-requiring agents should override this method for true batch processing
        trajectories = []
        for agent_input, scene in zip(agent_inputs, scenes):
            trajectory = self.compute_trajectory(agent_input, scene)
            trajectories.append(trajectory)
        return trajectories

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Computes the loss used for backpropagation based on the features, targets and model predictions.
        """
        raise NotImplementedError("No loss. Agent does not support training.")

    def get_optimizers(
        self,
    ) -> Union[torch.optim.Optimizer, Dict[str, Union[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]],]:
        """
        Returns the optimizers that are used by thy pytorch-lightning trainer.
        Has to be either a single optimizer or a dict of optimizer and lr scheduler.
        """
        raise NotImplementedError("No optimizers. Agent does not support training.")

    def get_training_callbacks(self) -> List[pl.Callback]:
        """
        Returns a list of pytorch-lightning callbacks that are used during training.
        See navsim.planning.training.callbacks for examples.
        """
        return []
