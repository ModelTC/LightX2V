from common.contract import get_contract

from ..sim.node import run_simulator_node
from .env import build_libero_env


def main(args=None):
    contract = get_contract("libero")
    run_simulator_node(contract, build_libero_env, node_name="libero_node", args=args)


if __name__ == "__main__":
    main()
