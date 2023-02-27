from typing import Dict
from agents.base_agent import BaseAgent
from agents.pong.agent import PongAgent
from agents.cartpole.agent import CartPoleAgent

agent_map: Dict[str, BaseAgent] = {"pong": PongAgent, "cartpole": CartPoleAgent}
