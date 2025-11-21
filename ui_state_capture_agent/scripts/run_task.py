import argparse
from src.agent.orchestrator import run_task_query_blocking


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="Natural language task query for Agent B")
    args = parser.parse_args()

    flow = run_task_query_blocking(args.query)
    print(f"Flow finished: id={flow.id} app={flow.app_name} run_id={flow.run_id} status={flow.status}")


if __name__ == "__main__":
    main()
