DukeMTMC

python demo/visualize_result.py --config-file logs/dukemtmc/mgn_R50-ibn/config.yaml --parallel --vis-label --dataset-name 'DukeMTMC' --output logs/mgn_duke_vis --opts MODEL.WEIGHTS logs/dukemtmc/mgn_R50-ibn/model_final.pth

python demo/visualize_result.py --config-file logs/dukemtmc/bagtricks_R50/config.yaml --parallel --vis-label --dataset-name 'DukeMTMC' --output logs/bot_duke_vis --opts MODEL.WEIGHTS logs/dukemtmc/bagtricks_R50/model_final.pth

python demo/visualize_result.py --config-file logs/dukemtmc/agw_R50/config.yaml --parallel --vis-label --dataset-name 'DukeMTMC' --output logs/agw_duke_vis --opts MODEL.WEIGHTS logs/dukemtmc/agw_R50/model_best.pth 

Market1501

python demo/visualize_result.py --config-file logs/market1501/bagtricks_R50/config.yaml --parallel --vis-label --dataset-name 'Market1501' --output logs/bot_market_vis --opts MODEL.WEIGHTS logs/market1501/bagtricks_R50/model_final.pth

python demo/visualize_result.py --config-file logs/market1501/mgn_R50-ibn/config.yaml --parallel --vis-label --dataset-name 'Market1501' --output logs/mgm_market_vis --opts MODEL.WEIGHTS logs/market1501/mgn_R50-ibn/model_final.pth

python demo/visualize_result.py --config-file logs/market1501/agw_R50/config.yaml --parallel --vis-label --dataset-name 'Market1501' --output logs/agw_market_vis --opts MODEL.WEIGHTS logs/market1501/agw_R50/model_final.pth
