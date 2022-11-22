import logging
import zipfile
import io
import hashlib
from abc import ABC


import diffusers
import torch
from PIL import Image
from ts.torch_handler.base_handler import BaseHandler

from unified_stable_diffusion import UnifiedStableDiffusionPipeline

logger = logging.getLogger(__name__)
logger.info("Diffusers version %s", diffusers.__version__)


class DiffusersHandler(BaseHandler, ABC):
    """
    Diffusers handler class for text to image generation.
    """

    def __init__(self):
        self.initialized = False

    def initialize(self, ctx):
        """In this initialize function, the Stable Diffusion model is loaded and
        initialized here.
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artefacts parameters.
        """
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        # Loading the model and tokenizer from checkpoint and config files based on the user's choice of mode
        # further setup config can be added.
        logger.info(f"Loading models from {model_dir + '/model.zip'}")
        with zipfile.ZipFile(model_dir + "/model.zip", "r") as zip_ref:
            zip_ref.extractall(model_dir + "/model")

        self.pipe = UnifiedStableDiffusionPipeline.from_pretrained(model_dir + "/model")
        self.pipe.to(self.device)
        logger.info("Unified diffusion model from path %s loaded successfully", model_dir)

        self.initialized = True

    def preprocess(self, requests):
        """Basic text preprocessing, of the user's prompt.
        Args:
            requests (str): The Input data in the form of text is passed on to the preprocess
            function.
        Returns:
            list : The preprocess function returns a list of prompts.
        """
        inputs = []
        for _, data in enumerate(requests):
            prompt = data.get("prompt")
            init_image = data.get("init_image")
            mask_image = data.get("mask_image")
            num_outputs = data.get("num_outputs")
            height = data.get("height")
            width = data.get("width")
            strength = data.get("strength") # 0.8
            guidance_scale = data.get("guidance_scale") # 7.5
            num_inference_steps = data.get("num_inference_steps")   # 50

            logger.info(f"prompt: {prompt}, {type(prompt)}")
            logger.info(f"init_image: {type(init_image)}")
            logger.info(f"mask_image: {type(mask_image)}")
            logger.info(f"num_outputs: {num_outputs} {type(num_outputs)}")
            logger.info(f"height: {height} {type(height)}")
            logger.info(f"width: {width} {type(width)}")
            logger.info(f"strength: {strength}, {type(strength)}")
            logger.info(f"guidance_scale: {guidance_scale}, {type(guidance_scale)}")
            logger.info(f"num_inference_steps: {num_inference_steps}, {type(num_inference_steps)}")

            if isinstance(prompt, (bytes, bytearray)):
                prompt = prompt.decode("utf-8")

            if init_image is not None:
                init_image = Image.open(io.BytesIO(init_image)).convert("RGB")
                init_image = init_image.resize((768, 512))

            if mask_image is not None:
                mask_image = Image.open(io.BytesIO(mask_image)).convert("RGB")
                mask_image = mask_image.resize((768, 512))

            if isinstance(num_outputs, (bytes, bytearray)):
                num_outputs = num_outputs.decode("utf-8")
                num_outputs = int(num_outputs)
            else:
                num_outputs = 1

            if isinstance(height, (bytes, bytearray)):
                height = height.decode("utf-8")
                height = int(height)
            else:
                height = 512

            if isinstance(width, (bytes, bytearray)):
                width = width.decode("utf-8")
                width = int(width)
            else:
                width = 512

            if isinstance(strength, (bytes, bytearray)):
                strength = strength.decode("utf-8")
                strength = float(strength)
            else:
                strength = 0.8

            if isinstance(guidance_scale, (bytes, bytearray)):
                guidance_scale = guidance_scale.decode("utf-8")
                guidance_scale = float(guidance_scale)
            else:
                guidance_scale = 7.5

            if isinstance(num_inference_steps, (bytes, bytearray)):
                num_inference_steps = num_inference_steps.decode("utf-8")
                num_inference_steps = int(num_inference_steps)
            else:
                num_inference_steps = 50

            logger.info("Received text: '%s'", prompt)
            inputs.append((
                prompt, init_image, mask_image, num_outputs, height, width,
                strength, guidance_scale, num_inference_steps,
            ))
        return inputs

    def inference(self, inputs):
        """Generates the image relevant to the received text.
        Args:
            input_batch (list): List of Text from the pre-process function is passed here
        Returns:
            list : It returns a list of the generate images for the input text
        """
        # Handling inference for sequence_classification.
        images = []
        for (
            prompt, init_image, mask_image, num_outputs, height, width,
            strength, guidance_scale, num_inference_steps,
         ) in inputs:
            images.append(
                self.pipe(
                    prompt=[prompt] * num_outputs,
                    init_image=init_image,
                    mask_image=mask_image,
                    height=height,
                    width=width,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                ).images
            )

        logger.info("Generated image: '%s'", images)
        return images

    def postprocess(self, inference_output):
        """Post Process Function converts the generated image into Torchserve readable format.
        Args:
            inference_output (list): It contains the generated image of the input text.
        Returns:
            (list): Returns a list of the images.
        """
        results = []
        for images in inference_output:
            paths = []
            for image in images:
                try:
                    m = hashlib.sha1()
                    with io.BytesIO() as memf:
                        image.save(memf, "PNG")
                        data = memf.getvalue()
                        m.update(data)

                    file_name = f"{m.hexdigest()}.png"
                    with open(f"/home/model-server/generated_images/{file_name}", "wb") as f:
                        f.write(data)
                    paths.append(f"/static/{file_name}")
                except Exception as e:
                    logger.error(f"Postprocess error: {e}")
                    paths.append(None)
            results.append(paths)

        return results
