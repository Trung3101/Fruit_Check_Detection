# A2 RF-DETR-L Training

A2 cung cap bo script train RF-DETR-L co:
- CLI args day du de tuy chinh train.
- Wandb logging cho config, metrics va artifacts.
- Augmentation profile tuong tu huong A1 va co the bat/tat.
- Preset phan cung cho 1x RTX 4090 24GB, 48GB RAM, 32 CPU cores.

## Files
- `train_rf_detr_wandb.py`: script train chinh.
- `augment.py`: profile augment va pipeline Albumentations (tuy chon).
- `hardware_profile.py`: preset thong so may.
- `run_4090.sh`: lenh chay nhanh cho cau hinh 4090.

## Cai dat goi

Vi API RF-DETR co the khac nhau theo version, script da co lop tuong thich linh hoat.
Can cai toi thieu:

```bash
pip install wandb rfdetr
```

Neu dung Albumentations:

```bash
pip install albumentations
```

## Vi du chay

```bash
python train/A2/train_rf_detr_wandb.py \
  --model-variant RF-DETR-L \
  --dataset-dir Fruit-Dataset-11 \
  --output-dir runs/rfdetr_a2_4090 \
  --hardware-profile 1x-rtx4090-24gb \
  --epochs 180 \
  --batch-size 8 \
  --grad-accum-steps 4 \
  --img-size 640 \
  --workers 16 \
  --learning-rate 2e-4 \
  --weight-decay 1e-4 \
  --warmup-epochs 5 \
  --early-stopping 30 \
  --amp true \
  --cache true \
  --device cuda:0 \
  --use-augment true \
  --use-albumentations false \
  --wandb-project RFDETR_A2 \
  --run-name A2_rfdetr_l_4090 \
  --wandb-key "$WANDB_API_KEY"
```

## Ghi chu
- Script se luu kwargs thuc te vao `train_kwargs_used.json` de tai lap.
- Neu package RF-DETR thay doi ten tham so train, script tu dong loc kwargs hop le theo signature hien tai.
- Ban co the bat `--use-albumentations true` neu backend RF-DETR cua ban ho tro external augment pipeline.
