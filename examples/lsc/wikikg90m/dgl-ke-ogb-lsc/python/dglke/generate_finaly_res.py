import numpy as np
import pandas as pd
import torch as th
import tqdm
import os
import time
from math import ceil
import argparse 

def general_finally_res_test(step, proc_num, path, ensem_v1_path, ensem_v2_path, pure_model_path, t_threshold,v_threshold):
    res_val, tmp_res = [], []
    
    train_tail_score = np.load(os.path.join(path, '/trian_tail_score_wyk.npy'))
    val_tail_score = np.load(os.path.join(path, '/val_tail_score_wyk.npy'))
    val_candi = np.load(os.path.join(path, '/val_t_candidate.npy'))
    
    global_index = 0
    err_cnt, rh_cnt, top9_cnt = 0, 0, 0
    top1_score, mid_score, final_res, final_res_tmp = [], [], [], []
    rr_cnt, rr1_cnt, rr2_cnt = 0, 0, 0
    
    for proc_no in range(proc_num):
        
        ori_res = th.load(os.path.join(ensem_v1_path, f"/load_ensemble_test_rerank_{proc_no}.pkl"), map_location=th.device('cpu'))
        ori_res_pred_top10 = ori_res['h,r->t']['t_pred_top10'][:,:10].numpy()
        print(ori_res['h,r->t']['t_pred_top10'].shape)
        
        res = th.load(os.path.join(ensem_v2_path, f"/load_ensemble_test_{proc_no}.pkl"), map_location=th.device('cpu'))
        print(res['h,r->t']['t_pred_top10'].shape)
        res_pred_top10 = res['h,r->t']['t_pred_top10'][:,:10].numpy()
        
        res_3 = th.load(os.path.join(pure_model_path, f"/test_{proc_no}_{step}.pkl"), map_location=th.device('cpu'))
        res_pred_top10_3 = res_3['h,r->t']['t_pred_top10'][:,:10].numpy()
        fileInfo = os.stat(os.path.join(pure_model_path, f"/test_{proc_no}_{step}.pkl"))    
        
        for index in tqdm.tqdm(range(res_pred_top10.shape[0])): 
            final_res_tmp =[]
            
            tmp_index = val_candi[global_index][res_pred_top10[index][0]]
            
            if train_tail_score[tmp_index]<=t_threshold + val_tail_score[tmp_index]<=v_threshold:
                embd_res, aft_res, aft_res_2 = ori_res_pred_top10[index].tolist(), res_pred_top10[index].tolist(), res_pred_top10_3[index].tolist()
                final_res_tmp = []
                for tail_index in range(10):
                    if val_tail_score[val_candi[global_index][embd_res[tail_index]]]>v_threshold:
                        rr_cnt += 1
                        final_res_tmp.append(embd_res[tail_index])
                        if embd_res[tail_index] in aft_res: 
                            rr1_cnt += 1
                            aft_res.remove(embd_res[tail_index])
                            
                if len(final_res_tmp)<=10: 
                    final_res_tmp.extend(aft_res)
                    final_res_tmp = final_res_tmp[:10]
                
                embd_res, final_res_tmp = final_res_tmp, []
                for tail_index in range(10):
                    if aft_res_2[tail_index] in embd_res:
                        rr2_cnt += 1
                        final_res_tmp.append(aft_res_2[tail_index])
                        embd_res.remove(aft_res_2[tail_index])
                        
                if len(embd_res)>0: final_res_tmp.extend(embd_res)
                assert len(final_res_tmp)==10,'not equal to 10'
                
                final_res.append(final_res_tmp)
                
            else:
                final_res.append(res_pred_top10[index].tolist())
                
            global_index += 1
    
    input_dict = res     
                
    from ogb.lsc import WikiKG90MEvaluator
    evaluator = WikiKG90MEvaluator()
    
    dir_path = os.path.join(ensem_v2_path, '/ensemble2_res/609new')
    finaly_res = th.Tensor(np.array(final_res))
    print(finaly_res.shape)
    input_dict['h,r->t']['t_pred_top10'] = finaly_res
    evaluator.save_test_submission(input_dict = input_dict, dir_path = dir_path)

    return 0


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--path',  help='path dir of data')
    parser.add_argument('--ensem_v1_path',  help='path of model 1 of ensemble') 
    parser.add_argument('--ensem_v2_path',  help='path of model 2 of ensemble') 
    parser.add_argument('--pure_model_path',  help='path of new model which trained by upsampleing training dataset') 
    parser.add_argument('--t_threshold',  help='train score threshold')
    parser.add_argument('--v_threshold',  help='val score threshold')

    args = parser.parse_args() 
    path = args.path
    ensem_v1_path = args.ensem_v1_path
    ensem_v2_path = args.ensem_v2_path
    pure_model_path = args.pure_model_path
    general_finally_res_test(step, proc_num, path, ensem_v1_path, ensem_v2_path, pure_model_path, t_threshold, v_threshold)

