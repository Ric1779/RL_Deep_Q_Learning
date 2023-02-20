from typing import Dict
from agents.base_agent import BaseAgent
from agents.pong.agent import PongAgent

agent_map: Dict[str, BaseAgent] = {"pong": PongAgent}
