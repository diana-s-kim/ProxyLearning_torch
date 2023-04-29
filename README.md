train: python -u train.py --img_dir "/ibex/scratch/kimds/Research/P2/data/wikiart_resize/" --csv_dir "./data/" --backbone_net "vgg16" --save_model_dir "./model/"

evaluation: python -u train.py --img_dir "/ibex/scratch/kimds/Research/P2/data/wikiart_resize/" --csv_dir "./data/" --backbone_net "vgg16" --save_model_dir "./model/" --do_eval "./model/proxy_0.pt"

resume training: python -u train.py --img_dir "/ibex/scratch/kimds/Research/P2/data/wikiart_resize/" --csv_dir "./data/" --backbone_net "vgg16" --save_model_dir "./model/" --resume "./model/proxy_0.pt"

collect embedding (train / val): python -u train.py --img_dir "/ibex/scratch/kimds/Research/P2/data/wikiart_resize/" --csv_dir "./data/" --backbone_net "vgg16" --save_model_dir "./model/" --do_cllct "./model/proxy_0.pt" "val" "./cllct_embedding/"
