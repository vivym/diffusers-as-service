import torch

from diffusers import StableDiffusionPipeline


def main():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, revision="fp16"
    )
    pipe.save_pretrained("./tmp")


if __name__ == "__main__":
    main()
