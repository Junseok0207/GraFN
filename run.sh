


device=0

for label_rate in 0.15 0.2 0.25
do
python main.py --dataset computers --folds 20 --label_rate $label_rate --tau 0.1 --thres 0.8 --decay 1e-2 --lr 0.005 --lam 0.5 --lam2 0.5 --df_1 0.5 --de_1 0.5 --df_2 0.2 --de_2 0.2 --device $device
python main.py --dataset photo --folds 20 --label_rate $label_rate --tau 0.1 --thres 0.8 --decay 1e-2 --lr 0.1 --lam 1.0 --lam2 0.5 --df_1 0.5 --de_1 0.5 --df_2 0.2 --de_2 0.2 --device $device
done

for label_rate in 0.5 1 2
do
python main.py --dataset cora --folds 20 --label_rate $label_rate --tau 0.1 --thres 0.9 --decay 5e-4 --lr 0.001 --lam 1.0 --lam2 0.5 --df_1 0.5 --de_1 0.5 --df_2 0.2 --de_2 0.2 --device $device
python main.py --dataset citeseer --folds 20 --label_rate $label_rate --tau 0.1 --thres 0.9 --decay 1e-2 --lr 0.001 --lam 0.5 --lam2 2.0 --df_1 0.5 --de_1 0.5 --df_2 0.4 --de_2 0.4 --device $device
done

for label_rate in 0.03 0.06 0.1
do
python main.py --dataset pubmed --folds 20 --label_rate $label_rate --tau 0.1 --thres 0.9 --decay 5e-4 --lr 0.1 --lam 0.5 --lam2 1.0 --df_1 0.5 --de_1 0.5 --df_2 0.2 --de_2 0.2 --device $device
done


