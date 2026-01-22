import json
import random
from dataclasses import dataclass
from typing import Dict, List, Type, Any

from Gym.environment import VerifiableEnvironment
from Gym.parameter_controller import ParameterController

from Gym.environments.difference_constraint_system import (
    DifferenceConstraintSystem_Environment as Environment
)
from Gym.parameter_controllers.difference_constraint_system import (
    DifferenceConstraintSystem_ParameterController as ParameterControllerImpl
)


@dataclass
class DatasetConfig:
    num_samples: int = 500
    out_path: str = "./data/HELD-OUT_ENVIRONMENTS/DCS-HELD-OUT.json"
    base_seed: int = 42
    timeout_second: int = 10

    difficulty_range: tuple = (10, 15)

    environment_identifier: str = "DifferenceConstraintSystem"



def generate_one_sample(
    env_identifier: str,
    env_cls: Type[VerifiableEnvironment],
    controller_cls: Type[ParameterController],
    seed: int,
    difficulty: int,
    timeout_second: int,
) -> Dict[str, Any]:

    controller = controller_cls()
    for _ in range(difficulty):
        controller.update()

    parameter_list = controller.get_parameter_list()
    if not parameter_list:
        raise RuntimeError("Empty parameter list from controller")

    parameter = random.choice(parameter_list)

    env = env_cls()
    ok = env.generator(
        seed=seed,
        parameter=parameter,
        timeout_second=timeout_second,
    )
    if not ok:
        raise RuntimeError("env.generator failed")

    user_prompt = env.prompt_generator()

    metadata = dict(
        environment=env_identifier,
        problem_difficulty=difficulty,
        config=env.get_config(),
    )

    return {
        "user_prompt": user_prompt,
        "metadata": json.dumps(metadata, ensure_ascii=False),
    }



def generate_dataset(cfg: DatasetConfig) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []

    rng = random.Random(cfg.base_seed)
    seed = cfg.base_seed

    while len(data) < cfg.num_samples:
        try:
            difficulty = rng.randint(
                cfg.difficulty_range[0],
                cfg.difficulty_range[1],
            )

            sample = generate_one_sample(
                env_identifier=cfg.environment_identifier,
                env_cls=Environment,
                controller_cls=ParameterControllerImpl,
                seed=seed,
                difficulty=difficulty,
                timeout_second=cfg.timeout_second,
            )

            data.append(sample)

        except Exception:
            pass
        finally:
            seed += 1

    return data


def main():
    cfg = DatasetConfig()

    dataset = generate_dataset(cfg)

    with open(cfg.out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(dataset)} samples to {cfg.out_path}")


if __name__ == "__main__":
    main()
