import logging
import os
import pickle
import traceback
from pathlib import Path
from typing import Dict, List

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter, Trajectory
from navsim.common.dataloader import SceneLoader

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_create_submission_pickle"


def process_tokens_batch_submission(
    tokens: List[str], agent: AbstractAgent, input_loader, batch_size: int = 32
) -> Dict[str, Trajectory]:
    """
    Process tokens in batches for faster submission generation.
    """
    trajectories_output: Dict[str, Trajectory] = {}
    
    for i in range(0, len(tokens), batch_size):
        batch_tokens = tokens[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(tokens) + batch_size - 1)//batch_size} with {len(batch_tokens)} tokens")
        
        # Prepare batch data
        batch_agent_inputs = []
        valid_tokens = []
        
        for token in batch_tokens:
            try:
                agent_input = input_loader.get_agent_input_from_token(token)
                batch_agent_inputs.append(agent_input)
                valid_tokens.append(token)
            except Exception as e:
                logger.warning(f"Failed to load data for token {token}: {e}")
        
        if not valid_tokens:
            continue
            
        try:
            # Batch trajectory computation
            trajectories = agent.compute_trajectories_batch(batch_agent_inputs)
            
            # Update results
            for token, trajectory in zip(valid_tokens, trajectories):
                trajectories_output[token] = trajectory
                
        except Exception as e:
            logger.warning(f"Batch processing failed: {e}")
            # Fall back to individual processing
            for token, agent_input in zip(valid_tokens, batch_agent_inputs):
                try:
                    trajectory = agent.compute_trajectory(agent_input)
                    trajectories_output[token] = trajectory
                except Exception:
                    logger.warning(f"----------- Agent failed for token {token}:")
                    traceback.print_exc()
    
    return trajectories_output


def run_test_evaluation(
    cfg: DictConfig,
    agent: AbstractAgent,
    scene_filter: SceneFilter,
    data_path: Path,
    synthetic_sensor_path: Path,
    original_sensor_path: Path,
    synthetic_scenes_path: Path,
) -> Dict[str, Trajectory]:
    """
    Function to create the output file for evaluation of an agent on the testserver
    :param agent: Agent object
    :param data_path: pathlib path to navsim logs
    :param synthetic_sensor_path: pathlib path to sensor blobs
    :param synthetic_scenes_path: pathlib path to synthetic scenes
    :param save_path: pathlib path to folder where scores are stored as .csv
    """
    if agent.requires_scene:
        raise ValueError(
            """
            In evaluation, no access to the annotated scene is provided, but only to the AgentInput.
            Thus, agent.requires_scene has to be False for the agent that is to be evaluated.
            """
        )
    logger.info("Building Agent Input Loader")
    input_loader = SceneLoader(
        data_path=data_path,
        scene_filter=scene_filter,
        synthetic_sensor_path=synthetic_sensor_path,
        original_sensor_path=original_sensor_path,
        synthetic_scenes_path=synthetic_scenes_path,
        sensor_config=agent.get_sensor_config(),
    )
    agent.initialize()

    # Process first stage with batching
    logger.info("Processing first stage with batch processing")
    first_stage_output = process_tokens_batch_submission(
        tokens=input_loader.tokens_stage_one,
        agent=agent,
        input_loader=input_loader,
        batch_size=32
    )

    # Process second stage with batching
    logger.info("Processing second stage with batch processing")
    scene_loader_tokens_stage_two = input_loader.reactive_tokens_stage_two
    second_stage_output = process_tokens_batch_submission(
        tokens=scene_loader_tokens_stage_two,
        agent=agent,
        input_loader=input_loader,
        batch_size=32
    )

    return first_stage_output, second_stage_output


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for submission creation script.
    :param cfg: omegaconf dictionary
    """
    agent = instantiate(cfg.agent)
    data_path = Path(cfg.navsim_log_path)
    synthetic_sensor_path = Path(cfg.synthetic_sensor_path)
    original_sensor_path = Path(cfg.original_sensor_path)
    synthetic_scenes_path = Path(cfg.synthetic_scenes_path)
    save_path = Path(cfg.output_dir)
    scene_filter = instantiate(cfg.train_test_split.scene_filter)

    first_stage_output, second_stage_output = run_test_evaluation(
        cfg=cfg,
        agent=agent,
        scene_filter=scene_filter,
        data_path=data_path,
        synthetic_scenes_path=synthetic_scenes_path,
        synthetic_sensor_path=synthetic_sensor_path,
        original_sensor_path=original_sensor_path,
    )

    submission = {
        "team_name": cfg.team_name,
        "authors": cfg.authors,
        "email": cfg.email,
        "institution": cfg.institution,
        "country / region": cfg.country,
        "first_stage_predictions": [first_stage_output],
        "second_stage_predictions": [second_stage_output],
    }

    # pickle and save dict
    filename = os.path.join(save_path, "submission.pkl")
    with open(filename, "wb") as file:
        pickle.dump(submission, file)
    logger.info(f"Your submission filed was saved to {filename}")


if __name__ == "__main__":
    main()
