"""Safe default: preflight only. Training and inference require explicit subcommands."""
import argparse
import json

from .config import TRAINING_CONDITIONS, configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze", help="One-time creation; refuses an existing protocol freeze")
    preflight = sub.add_parser("preflight", help="No forward/backward passes or optimizer updates")
    preflight.add_argument("--require-mps", action="store_true")
    train = sub.add_parser("train", help="Only after explicit user authorization")
    train.add_argument("--condition", choices=TRAINING_CONDITIONS, required=True)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--stop-after-epoch", type=int)
    evaluation = sub.add_parser("evaluate", help="Full test inference; separately authorized execution")
    evaluation.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "freeze":
        from .provenance import freeze_protocol
        result = freeze_protocol()
        print(json.dumps({k: result[k] for k in ("protocol_version", "freeze_sha256", "historical_digest_match")}, indent=2))
    elif args.command == "preflight":
        from .preflight import preflight
        print(json.dumps(preflight(args.require_mps), indent=2))
    elif args.command == "train":
        from .train import train as execute
        execute(configs()[args.condition], args.resume, args.stop_after_epoch)
    elif args.command == "evaluate":
        from .evaluate import evaluate
        evaluate(args.resume)


if __name__ == "__main__":
    main()
