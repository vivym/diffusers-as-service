from uuid import uuid4
from pathlib import Path

import aiofiles
from celery.result import AsyncResult
from fastapi import FastAPI, UploadFile

from worker import text_to_image, image_to_image, inpaint_image

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    task_result = AsyncResult(task_id)

    return {
        "task_id": task_id,
        "status": task_result.status,
        "result": task_result.result,
    }


@app.get("/diffusers/text2img")
async def diffusers_text2img(
    prompt: str,
    strength: float = 0.8,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
):
    task = text_to_image.delay(
        prompt=prompt,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    )

    return {
        "task_id": task.id,
        "prompt": prompt,
    }


async def save_upload_file(upload_file: UploadFile) -> str:
    file_name = f"{uuid4()}{Path(upload_file.filename).suffix}"
    async with aiofiles.open(f"/uploaded_images/{file_name}", "wb") as f:
        while content := await upload_file.read(4 * 1024):
            await f.write(content)

    return file_name


@app.post("/diffusers/img2img")
async def diffusers_img2img(
    init_image: UploadFile,
    prompt: str,
    strength: float = 0.8,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
):
    init_image_path = await save_upload_file(init_image)

    task = image_to_image.delay(
        init_image=init_image_path,
        prompt=prompt,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    )

    return {
        "task_id": task.id,
        "prompt": prompt,
        "init_image": init_image_path,
    }


@app.post("/diffusers/inpaint")
async def diffusers_inpaint(
    init_image: UploadFile,
    mask_image: UploadFile,
    prompt: str,
    strength: float = 0.8,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
):
    init_image_path = await save_upload_file(init_image)
    mask_image_path = await save_upload_file(mask_image)

    task = inpaint_image.delay(
        init_image=init_image_path,
        mask_image=mask_image_path,
        prompt=prompt,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    )

    return {
        "task_id": task.id,
        "prompt": prompt,
        "init_image": init_image_path,
        "mask_image": mask_image_path,
    }
