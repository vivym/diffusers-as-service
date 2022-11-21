# diffusers-as-service

## Usage

```bash
cd model-deploy
python download_models.py

cd tmp
zip -r ../model.zip *
cd ..
rm -rf tmp
```

```bash
torch-model-archiver --model-name stable-diffusion --version 1.0 --handler stable_diffusion_handler.py --extra-files model.zip,unified_stable_diffusion.py -r requirements.txt --export-path model-store/ -f
```

```bash
docker run --rm -it -p 8080:8080 -p 8081:8081 --name stable-diffusion --gpus '"device=2,3"' -v $(pwd)/config.properties:/home/model-server/config.properties -v $(pwd)/model-store:/home/model-server/model-store -v $(pwd)/../generated_images:/home/model-server/generated_images pytorch/torchserve:latest-gpu
```
