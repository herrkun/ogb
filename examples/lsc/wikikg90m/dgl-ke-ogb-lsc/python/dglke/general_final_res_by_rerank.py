import numpy as np
import pandas as pd
import torch as th
import tqdm
import os
import time
from math import ceil


def secondsToStr(seconds):
    x = time.localtime(seconds)
    return time.strftime("%Y-%m-%d %X", x)
    
def general_finally_res_test(step, proc_num, path, ensem_v1_path, ensem_v2_path, pure_model_path):
    res_val, tmp_res = [], []
    
    train_tail_cnt = np.load(os.path.join(path, '/trian_tail_cnt_wyk.npy'))
    
    test_tail_count = np.load(os.path.join(path, '/test_tail_cnt_wyk.npy'))
    
    
    test_candi = np.load(os.path.join(path, '/test_t_candidate.npy'))
    
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
        print('file size:{}k'.format(ceil(fileInfo.st_size/1024)))
        print('creat time:{}'.format(secondsToStr(fileInfo.st_ctime)))
        print('read time:{}'.format(secondsToStr(fileInfo.st_atime)))
        print('modify time:{}'.format(secondsToStr(fileInfo.st_mtime)))
        
        
        for index in tqdm.tqdm(range(res_pred_top10.shape[0])): 
            final_res_tmp =[]
            
            tmp_index = test_candi[global_index][res_pred_top10[index][0]]
            
            if train_tail_cnt[tmp_index]<=6000 and test_tail_count[tmp_index]<=32:
                embd_res, aft_res, aft_res_2 = ori_res_pred_top10[index].tolist(), res_pred_top10[index].tolist(), res_pred_top10_3[index].tolist()
                final_res_tmp = []
                for tail_index in range(10):
                    if test_tail_count[test_candi[global_index][embd_res[tail_index]]]>32:
                        rr_cnt += 1
                        final_res_tmp.append(embd_res[tail_index])
                        if embd_res[tail_index] in aft_res: 
                            rr1_cnt += 1
                            aft_res.remove(embd_res[tail_index])
                            
                if len(final_res_tmp)<=10: 
                    final_res_tmp.extend(aft_res)
                    final_res_tmp = final_res_tmp[:10]
                assert len(final_res_tmp)==10,'not equal to 10'
                
                # rerank 2
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
    
    input_dict = res      # just for tmp use, just init dict     
                
    from ogb.lsc import WikiKG90MEvaluator
    evaluator = WikiKG90MEvaluator()
    
    dir_path = os.path.join(ensem_v2_path, '/ensemble2_res/609new')
    finaly_res = th.Tensor(np.array(final_res))
    print(finaly_res.shape)
    input_dict['h,r->t']['t_pred_top10'] = finaly_res
    evaluator.save_test_submission(input_dict = input_dict, dir_path = dir_path)

    return 0


if __name__ == '__main__':
    general_finally_res_test(step, proc_num, path, ensem_v1_path, ensem_v2_path, pure_model_path)
