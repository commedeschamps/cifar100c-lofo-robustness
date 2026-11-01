"""Explicit commands; a full benchmark never runs by default."""
import argparse


def main():
    parser = argparse.ArgumentParser(description="Part 3 official CIFAR-100-C frozen-checkpoint evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    dl = sub.add_parser("download", help="Explicit ~2.9 GB official download; needs >=7 GB free")
    dl.add_argument("--data-root", default="data")
    verify = sub.add_parser("verify", help="Verify existing arrays against the official checksummed archive")
    verify.add_argument("--archive", required=True)
    verify.add_argument("--corruption-root", default="data/CIFAR-100-C")
    plots = sub.add_parser("plots", help="Regenerate figures from saved complete scientific metrics")
    plots.add_argument("--output", default="results/evaluation/seed42")
    for mode in ("preflight", "smoke", "full"):
        run = sub.add_parser(mode)
        run.set_defaults(mode=mode)
        run.add_argument("--data-root", default="data")
        run.add_argument("--corruption-root")
        run.add_argument("--results-root", default="results")
        run.add_argument("--output")
        run.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
        run.add_argument("--batch-size", type=int, default=128)
        run.add_argument("--cpu-threads", type=int, default=4)
        run.add_argument("--benchmark-images", type=int, default=512)
        run.add_argument("--smoke-images", type=int, default=32)
        run.add_argument("--bootstrap", type=int, default=0, help="Optional >=100 paired image-cluster replicates")
        run.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "download":
        from .download import download
        download(args.data_root)
    elif args.command == "verify":
        from .download import verify_archive
        verify_archive(args.archive, args.corruption_root)
    elif args.command == "plots":
        from .plots import plot_saved
        plot_saved(args.output)
    else:
        if args.bootstrap and (args.mode != "full" or args.bootstrap < 100):
            parser.error("Bootstrap requires full mode and >=100 replicates.")
        from .runner import evaluation
        evaluation(args)


if __name__ == "__main__":
    main()
