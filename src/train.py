"""Train a single model based on the YAML config and save the joblib pickle."""
import argparse
import logging

import yaml

from src import preprocess


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    log.info("loading %s", cfg["data"]["path"])
    df = preprocess.load_csv(cfg["data"]["path"])
    log.info("rows=%d", len(df))


if __name__ == "__main__":
    main()
