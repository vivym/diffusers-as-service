import argparse

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url", type=str,
        default="http://localhost:8080/predictions/stable-diffusion",
        help="Torchserve inference endpoint",
    )
    parser.add_argument(
        "--prompt", type=str, required=True, help="Prompt for image generation"
    )
    args = parser.parse_args()

    rsp = requests.post(args.url, data={
        "prompt": args.prompt
    })
    print("status_code:", rsp.status_code)
    print("result:", rsp.text)


if __name__ == "__main__":
    main()
