# Data Process


###  Data Upsamping
After analyzing the results of several models, we found that the model generally has poor standard intervals on a specific data set. The confidence of this part of the data is low, so in order to improve the model in this part For data representation, we upsample this part of the data, and in the final test scoring stage, we add the validation set into the training set.

```python ./dglke/generate_data_upsample.py```


# Model Training
### 1 Training Model for Ensemble
   Training two different models for capturing multy features
#### 1.1 Training PairRE Model
  ```CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 dglke_train_val --model_name PairRE --hidden_dim 100 --gamma 10 --lr 0.1 --regularization_coef 1e-9 --mlp_lr 0.00001 --valid -adv --num_proc 4 --num_thread 4 --gpu 0 1 2 3 --force_sync_interval 100000 --max_step 800000 --eval_interval 100000 --train_with_val --neg_sample_size 100 --eval_percent 0.1 --topk 20  --print_on_screen --encoder_model_name concat -de -dr  --save_path ${save_pairre_model_path}```
#### 1.2 Training RotatE Model
  ```CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 dglke_train_val --model_name RotatE --hidden_dim 100 --gamma 10 --lr 0.1 --regularization_coef 1e-9 --mlp_lr 0.00001 --valid -adv --num_proc 4 --num_thread 4 --gpu 4 5 6 7 --force_sync_interval 100000 --max_step 1100000 --eval_interval 100000 --train_with_val --neg_sample_size 100 --eval_percent 0.1 --topk 20  --print_on_screen --encoder_model_name concat -de  --save_path ${save_rotate_model_path}```

#### 1.3 Training Model with Node Features
``` 
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 dglke_ensemble_train --model_name ${model_name} --hidden_dim 100 --gamma 10 --lr 0.08 --regularization_coef 1e-9 --mlp_lr 0.00001 --valid -adv --num_proc 4 --num_thread 4 --gpu 0 1 2 3 --force_sync_interval 100000 --max_step 1500000 --eval_interval 100000 --print_on_screen --encoder_model_name concat -de -dr --save_path ${save_path}
```

#### 1.4 Training Ensemble Model
``` 
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python dglke/ensemble_on_valid.py --num_proc 4 --gpu 0 1 2 3 --embed_version feat --m1_path ${model1_path} --m2_path ${model2_path} --save_path ${ensemble_model_path}
```


### 2 Training Model With More Data
   Upsampling some datasets that appear frequently on the validation set but appear less on the training set as Model can not learn well on these datsets.
   
   ```CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 dglke_train_val --model_name PairRE --hidden_dim 100 --gamma 10 --lr 0.1 --regularization_coef 1e-9 --mlp_lr 0.00001 --valid -adv --num_proc 4 --num_thread 4 --gpu 4 5 6 7 --force_sync_interval 100000 --max_step 2500000 --eval_interval 100000 --train_with_val --neg_sample_size 100 --eval_percent 0.1 --topk 30 --test --print_on_screen --encoder_model_name concat -de -dr  --save_path ${upsample_model_path}```
   
### 3 Retrain Model Generised on Step 2
   Through analysis, We found there are fewer samples of prediction errors, but more of them are in prediction top 10. It will be helpfull to retain model on these datasets which filtered by prediction top_10 tails in Training dataset
   
   ```CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 dglke_retrain --num_proc 4 --gpu 4 5 6 7  --save_path ${retrain_model_path}```
   
### 4 Reranking Result on A Specific dataset
  As Ensemble Model has higher Recall and Upsampling Model has higher Accuary, Finaly Result is improved by rescore ensemble model prediction with Upsampling Model's result
 
 ``` python ./dglke/generate_finaly_res.py --path /ogb/link_level/dataset/wikikg90m_kddcup2021/processed/ --ensem_v1_path ${ensemble_model_path_1} --ensem_v2_path ${ensemble_model_path_1} --pure_model_path ${upsample_model_path} --t_threshold ${t_threshold} --v_threshold ${v_threshold} ```
