# Retina regression — demo

Bu repo `RetinaMNIST` (medmnist) fotosuratlaridan tibbiy severity (0..4) bashorat qilish uchun starter.  
Maqsad — imaging-based regression pipeline’ni o‘rganish: data loader, model, trening, baholash.

## Tez boshlash
```bash
cd src
pip install -r requirements.txt
python train.py --epochs 8 --batch_size 128 --model_dir ../runs/retina_demo --use_tensorboard
python eval.py --checkpoint ../runs/retina_demo/best.pth
