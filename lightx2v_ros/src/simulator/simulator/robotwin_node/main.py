from common.contract import get_contract

from ..sim.node import run_simulator_node
from .env import build_robotwin_env


def main(args=None):
    contract = get_contract("robotwin")
    run_simulator_node(contract, build_robotwin_env, node_name="robotwin_node", args=args)


if __name__ == "__main__":
    main()
