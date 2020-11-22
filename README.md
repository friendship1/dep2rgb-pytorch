# gf2rgb
geometric feature to RGB

# commands
python train.py --is_test=True  --resume=best_val_0022.ckpt

# commands for refine
python train.py --trainer=refine --coarsenet_pth=model/checkpoints/coarsenet.pth
nohup python train.py --trainer=refine --coarsenet_pth=model/checkpoints/coarsenet.pth --resume=outputs/checkpoint0001.pth &> run0?.log &

python train.py --trainer=refine --coarsenet_pth=model/checkpoints/coarsenet.pth --is_test=true --out_dir=out_test --resume=path/checkpoint.pth



# commands for 3d conv
python train.py --trainer=3d --out_dir=outputs_3d