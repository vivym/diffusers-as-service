import requests
from celery import Celery

app = Celery(
    __name__,
    broker="redis://localhost",
    backend="redis://localhost",
)


@app.task
def text_to_image(
    prompt: str,
    strength: float = 0.8,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
) -> str:
    rsp = requests.post(
        "http://localhost:8080/predictions/stable-diffusion",
        data={
            "prompt": prompt,
            "strength": strength,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
        },
    )
    return rsp.text


@app.task
def image_to_image(
    init_image,
    prompt: str,
    strength: float = 0.8,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
) -> str:
    rsp = requests.post(
        "http://localhost:8080/predictions/stable-diffusion",
        data={
            "prompt": prompt,
            "strength": strength,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
        },
        files={
            "init_image": open(f"uploaded_images/{init_image}", "rb"),
        }
    )
    return rsp.text


@app.task
def inpaint_image(
    init_image,
    mask_image,
    prompt: str,
    strength: float = 0.8,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
) -> str:
    rsp = requests.post(
        "http://localhost:8080/predictions/stable-diffusion",
        data={
            "prompt": prompt,
            "strength": strength,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
        },
        files={
            "init_image": open(f"uploaded_images/{init_image}", "rb"),
            "mask_image": open(f"uploaded_images/{mask_image}", "rb"),
        }
    )
    return rsp.text