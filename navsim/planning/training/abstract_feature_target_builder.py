from abc import abstractmethod
from typing import Dict, List

import torch
from torch import Tensor

from navsim.common.dataclasses import AgentInput, Scene


class AbstractFeatureBuilder:
    """Abstract class of feature builder for agent training."""

    def __init__(self):
        pass

    @abstractmethod
    def get_unique_name(self) -> str:
        """
        :return: Unique name of created feature.
        """

    @abstractmethod
    def compute_features(self, agent_input: AgentInput) -> Dict[str, Tensor]:
        """
        Computes features from the AgentInput object, i.e., without access to ground-truth.
        Outputs a dictionary where each item has a unique identifier and maps to a single feature tensor.
        One FeatureBuilder can return a dict with multiple FeatureTensors.
        """
    
    def compute_features_batch(self, agent_inputs: List[AgentInput]) -> Dict[str, Tensor]:
        """
        Computes features for multiple AgentInput objects in parallel.
        Default implementation falls back to individual processing and stacking.
        Override this method for true parallel feature computation.
        :param agent_inputs: List of AgentInput objects for batch processing.
        :return: Dictionary where each item maps to a batched feature tensor.
        """
        if not agent_inputs:
            return {}
        
        # Default fallback: individual processing + stacking
        batch_features = {}
        for agent_input in agent_inputs:
            features = self.compute_features(agent_input)
            for key, value in features.items():
                if key not in batch_features:
                    batch_features[key] = []
                batch_features[key].append(value)
        
        # Stack into batch tensors
        return {k: torch.stack(v, dim=0) for k, v in batch_features.items()}


class AbstractTargetBuilder:
    def __init__(self):
        pass

    @abstractmethod
    def get_unique_name(self) -> str:
        """
        :return: Unique name of created target.
        """

    @abstractmethod
    def compute_targets(self, scene: Scene) -> Dict[str, Tensor]:
        """
        Computes targets from the Scene object, i.e., with access to ground-truth.
        Outputs a dictionary where each item has a unique identifier and maps to a single target tensor.
        One TargetBuilder can return a dict with multiple TargetTensors.
        """
