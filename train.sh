
#python train_word2vec.py --save_dir /mnt/binhna/summary2/w2v --dim 300 --data_dir /mnt/binhna/summary2/finished_files

#python make_extraction_labels.py --data_dir /mnt/binhna/summary2/finished_files
echo "Train abstractor\n"
python train_abstractor.py --data_dir /mnt/binhna/summary2/finished_files --path /mnt/binhna/summary2/abs_model_old --w2v /mnt/binhna/summary2/w2v/word2vec.300d.225k.bin --batch 8
echo "Train extractor\n"
python train_extractor_ml.py --data_dir /mnt/binhna/summary2/finished_files --path /mnt/binhna/summary2/ext_model --w2v /mnt/binhna/summary2/w2v/word2vec.300d.225k.bin --batch 8
echo "Train full\n"
python train_full_rl.py --path='/mnt/binhna/summary2/model' --abs_dir='/mnt/binhna/summary2/abs_model' --ext_dir='/mnt/binhna/summary2/ext_model' --data_dir='/mnt/binhna/summary2/finished_files' --batch=8
echo "Decode full\n"
python decode_full_model.py --path=/mnt/binhna/summary2/decoded_files --model_dir=/mnt/binhna/summary2/model --beam=5 --test --data_dir=/mnt/binhna/summary2/finished_files
echo "make eval\n"
python make_eval_references.py --data_dir /mnt/binhna/summary2/finished_files
echo "eval full\n"
python eval_full_model.py --rouge --decode_dir=/mnt/binhna/summary2/decoded_files --data_dir=/mnt/binhna/summary2/finished_files --rouge_path=/mnt/binhna/summary/pyrouge/tools/ROUGE-1.5.5


