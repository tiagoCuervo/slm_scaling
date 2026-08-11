import argparse

from .checkpoint import convert_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Convert a training checkpoint to native safetensors")
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    args = parser.parse_args()
    convert_checkpoint(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
