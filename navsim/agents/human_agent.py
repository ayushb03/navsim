from typing import List

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, Scene, SensorConfig, Trajectory


class HumanAgent(AbstractAgent):
    """Privileged agent interface of human operator."""

    requires_scene = True

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
    ):
        """
        Initializes the human agent object.
        :param trajectory_sampling: trajectory sampling specification
        """
        self._trajectory_sampling = trajectory_sampling

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig.build_no_sensors()

    def compute_trajectory(self, agent_input: AgentInput, scene: Scene) -> Trajectory:
        """
        Computes the ego vehicle trajectory.
        :param current_input: Dataclass with agent inputs.
        :param scene: Scene object containing ground truth trajectory data.
        :return: Trajectory representing the predicted ego's position in future
        """
        return scene.get_future_trajectory(self._trajectory_sampling.num_poses)

    def compute_trajectories_batch_with_scenes(self, agent_inputs: List[AgentInput], scenes: List[Scene]) -> List[Trajectory]:
        """
        Computes ego vehicle trajectories for multiple scenes in parallel for the HumanAgent.
        This is much faster than processing scenes serially.
        
        :param agent_inputs: List of AgentInput dataclasses for multiple scenes.
        :param scenes: List of Scene objects corresponding to each agent input.
        :return: List of Trajectory objects for each scene.
        """
        if not scenes:
            return []
        
        # Parallel batch processing: extract all future trajectories at once
        trajectories = []
        for scene in scenes:
            trajectory = scene.get_future_trajectory(self._trajectory_sampling.num_poses)
            trajectories.append(trajectory)
        
        return trajectories
