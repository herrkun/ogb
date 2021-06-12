### 1 Training Model for Ensemble
   Training two different models for capturing multy features
#### 1.1 Training PairRE Model
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 dglke_train_val --model_name PairRE --hidden_dim 100 --gamma 10 --lr 0.1 --regularization_coef 1e-9 --mlp_lr 0.00001 --valid -adv --num_proc 4 --num_thread 4 --gpu 0 1 2 3 --force_sync_interval 100000 --max_step 800000 --eval_interval 100000 --train_with_val --neg_sample_size 100 --eval_percent 0.1 --topk 20  --print_on_screen --encoder_model_name concat -de -dr  --save_path "xxxxxxxx"
#### 1.2 Training RotatE Model
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 dglke_train_val --model_name RotatE --hidden_dim 100 --gamma 10 --lr 0.1 --regularization_coef 1e-9 --mlp_lr 0.00001 --valid -adv --num_proc 4 --num_thread 4 --gpu 4 5 6 7 --force_sync_interval 100000 --max_step 1100000 --eval_interval 100000 --train_with_val --neg_sample_size 100 --eval_percent 0.1 --topk 20  --print_on_screen --encoder_model_name concat -de  --save_path "xxxxxxxxx"
  
#### 1.3 Training Ensemble Model

### 2 Training Model With More Data
   Upsampling some datasets that appear frequently on the validation set but appear less on the training set as Model can not learn well on these datsets.
   
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 dglke_train_val --model_name PairRE --hidden_dim 100 --gamma 10 --lr 0.1 --regularization_coef 1e-9 --mlp_lr 0.00001 --valid -adv --num_proc 4 --num_thread 4 --gpu 4 5 6 7 --force_sync_interval 100000 --max_step 2500000 --eval_interval 100000 --train_with_val --neg_sample_size 100 --eval_percent 0.1 --topk 30 --test --print_on_screen --encoder_model_name concat -de -dr  --save_path "xxxxxxxx"
   
### 3 Retrain Model Generised on Step 2
   Through analysis, We found there are fewer samples of prediction errors, but more of them are in prediction top 10. It will be helpfull to retain model on these datasets which filtered by prediction top_10 tails in Training dataset
   
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 dglke_retrain --num_proc 4 --gpu 4 5 6 7  --save_path "xxxxxxxxx"
   
### 4 Reranking Result on A Specific dataset
  As Ensemble Model has higher Recall and Upsampling Model has higher Accuary, Finaly Result is improved by rescore ensemble model prediction with Upsampling Model's result
  python "xxxxxxxx"
