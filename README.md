# diffusers-as-service

## Usage

```bash
cd model-deploy
python download_models.py

cd tmp
zip -r ../model.zip *
cd ..
rm -rf tmp

torch-model-archiver --model-name stable-diffusion --version 1.0 --handler stable_diffusion_handler.py --extra-files model.zip,unified_stable_diffusion.py -r requirements.txt --export-path model-store/ -f

cd ..
docker compose up -d
```
