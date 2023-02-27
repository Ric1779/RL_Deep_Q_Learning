from dataclasses import dataclass
import os


@dataclass
class TrainParams:
    """Parameters used for training the agent.
    """

    memory_size: int = 50000
    episodes: int = 2000
    batch_size: int = 32
    target_update_frequency: int = 100
    frequency: int = 1
    gamma: float = 0.95
    lr: float = 1e-4
    epsilon: float = 0.1
    epsilon_end: float = 0.05
    anneal_length: int = 10 ** 4
    num_actions: int = 2


@dataclass
class EvaluateParams:
    """Parameters used for evaluating the agent.
    """

    frequency: int = 25
    episodes: int = 5


@dataclass
class CartPoleConfig:
    """Configuration for the CartPole agent.
    """

    train: TrainParams = TrainParams()
    evaluate: EvaluateParams = EvaluateParams()
    env: str = "CartPole-v1"
    model_path: str = os.path.join(
        os.path.dirname(__file__), "models/CartPole-v1_best.pt"
    )
