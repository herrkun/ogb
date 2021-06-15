import numpy as np
import pandas as pd
import torch as th
import tqdm
import os

# Count the number of head/tail occurrences
def get_head_cnt_train(reload=False):
    saved_path = './dataset/wikikg90m_kddcup2021/processed/trian_head_cnt_wyk.npy'
    if reload and os.path.isfile(saved_path):
        return np.load(saved_path)
    trip_list = np.load('./dataset/wikikg90m_kddcup2021/processed/train_hrt.npy')
    n_sample = trip_list.shape[0]
    head_count = np.zeros((87143637, 1))
    for i in range(n_sample):
        if i % 10000000 == 0: print(i)
        head_count[trip_list[i][0]] += 1
    np.save(saved_path, head_count)
    return head_count

def get_tail_cnt_train(reload=False):
    saved_path = './dataset/wikikg90m_kddcup2021/processed/trian_tail_cnt_wyk.npy'
    if reload and os.path.isfile(saved_path):
        return np.load(saved_path)
    trip_list = np.load('./dataset/wikikg90m_kddcup2021/processed/train_hrt.npy')
    n_sample = trip_list.shape[0]
    tail_count = np.zeros((87143637, 1))
    for i in range(n_sample):
        if i % 10000000 == 0: print(i)
        tail_count[trip_list[i][2]] += 1
    np.save(saved_path, tail_count)
    return tail_count

def get_head_cnt_val(reload=False):
    saved_path = './dataset/wikikg90m_kddcup2021/processed/val_head_cnt_wyk.npy'
    if reload and os.path.isfile(saved_path):
        return np.load(saved_path)
    candidate_list = np.load('./dataset/wikikg90m_kddcup2021/processed/val_hr.npy')
    n_sample = candidate_list.shape[0]
    head_count = np.zeros((87143637, 1))
    for i in tqdm.tqdm(range(n_sample)):
        head_count[candidate_list[i][0]] += 1
    np.save(saved_path, head_count)
    print('head count have saved........')
    return head_count

def get_tail_cnt_val(reload=False):
    saved_path = './dataset/wikikg90m_kddcup2021/processed/val_tail_cnt_wyk.npy'
    if reload and os.path.isfile(saved_path):
        return np.load(saved_path)
    candidate_list = np.load('./dataset/wikikg90m_kddcup2021/processed/val_t_candidate.npy')
    n_sample = candidate_list.shape[0]
    n_candidate = candidate_list.shape[1]
    tail_count = np.zeros((87143637, 1))
    for i in range(n_sample):
        if i % 100000 == 0: print(i)
        tail_count[candidate_list[i]] += 1
    np.save(saved_path, tail_count)
    return tail_count



def generator_new_train(include_val=True, head_upsample_epochs=2):
    
    train_tail_count = get_tail_cnt_train()
    val_tail_count = get_tail_cnt_val()
    
    train_trip = np.load('./dataset/wikikg90m_kddcup2021/processed/train_hrt.npy')
    val_trip = np.load('./dataset/wikikg90m_kddcup2021/processed/val_hrt_wyk.npy')
    
    new_train_hrt = []
    
    # 筛选出现在  val  block4 上的样本， top1 或者 top10
    val_candi = np.load('./dataset/wikikg90m_kddcup2021/processed/val_t_candidate.npy')    
    global_index = 0
    filter_topk = 10
    
    counter1, counter2, counter3, counter4 = 0,0,0,0
    
    proc_num = 4
    for proc_no in range(proc_num):
        ori_res = th.load(f"./ensemble_valid_v2_backup/load_ensemble_valid_rerank_{proc_no}.pkl", map_location=th.device('cpu'))
        ori_res_pred_top10 = ori_res['h,r->t']['t_pred_top10'][:,:10].numpy()
        print(ori_res['h,r->t']['t_pred_top10'].shape)
        res_correct_index = ori_res['h,r->t']['t_correct_index'].numpy()
        
        for index in tqdm.tqdm(range(res_correct_index.shape[0])):  
            for topk_index in range(filter_topk):  # 取预测前 top k 的样本进行采样
                tmp_index = val_candi[global_index][ori_res_pred_top10[index][topk_index]]
                if train_tail_count[tmp_index]<=6000 and val_tail_count[tmp_index]<=39:
                    val_tail_count[tmp_index] = -1
                    counter1 += 1
            global_index += 1
    print("validation topk have fitered %d samples" % counter1)
    
    global_index = 0
    
    # 各自筛选 <6000 的样本 在 val   上预测的 candicate tail的出现次数
    for i in tqdm.tqdm(range(train_trip.shape[0])):
        tmp_train = train_trip[i]
        if (train_tail_count[tmp_train[2]]+val_tail_count[tmp_train[2]] <=6000) or val_tail_count[tmp_train[2]]==-1:
            new_train_hrt.append(tmp_train)
            
    print('after add 6000 filter and topK filter')
    print(len(new_train_hrt))    
    
    # 筛选出所有 head 出现在 val
    head_apper_count = 0
    val_head_count = get_head_cnt_val()
    for epoch in range(head_upsample_epochs):
        for i in tqdm.tqdm(range(train_trip.shape[0])):
            tmp_train = train_trip[i]
            if val_head_count[tmp_train[0]] > 0:
                new_train_hrt.append(tmp_train)
                head_apper_count += 1
                counter3 += 1
    print(head_apper_count)  
    print(counter3,counter4)
            
    np.random.shuffle(new_train_hrt)
    try:
        print(len(new_train_hrt))   
    except:
        print('haha')
    np.save('./dataset/wikikg90m_kddcup2021/processed/trian_val_topk_add_h.npy', new_train_hrt)
    return 0

def generator_new_val(include_val=True, head_upsample_epochs=2):
    
    train_tail_count = get_tail_cnt_train()
    val_tail_count = get_tail_cnt_val()
    
    train_trip = np.load('./dataset/wikikg90m_kddcup2021/processed/train_hrt.npy')
    val_trip = np.load('./dataset/wikikg90m_kddcup2021/processed/val_hrt_wyk.npy')
    
    new_train_hrt = []
    
    # 筛选出现在  val  block4 上的样本， top1 或者 top10
    val_candi = np.load('./dataset/wikikg90m_kddcup2021/processed/val_t_candidate.npy')
    
    global_index = 0
    filter_topk = 10
    
    counter1, counter2, counter3, counter4 = 0,0,0,0
    
    proc_num = 4
    for proc_no in range(proc_num):
        ori_res = th.load(f"./ensemble_valid_v2_backup/load_ensemble_valid_rerank_{proc_no}.pkl", map_location=th.device('cpu'))
        ori_res_pred_top10 = ori_res['h,r->t']['t_pred_top10'][:,:10].numpy()
        print(ori_res['h,r->t']['t_pred_top10'].shape)
        res_correct_index = ori_res['h,r->t']['t_correct_index'].numpy()
        
        for index in tqdm.tqdm(range(res_correct_index.shape[0])):  
            for topk_index in range(filter_topk):  # 取预测前 top k 的样本进行采样
                tmp_index = val_candi[global_index][ori_res_pred_top10[index][topk_index]]
                if train_tail_count[tmp_index]<=6000 and val_tail_count[tmp_index]<=39:
                    val_tail_count[tmp_index] = -1
                    counter1 += 1
            global_index += 1
    print("validation topk have fitered %d samples" % counter1)
    
    # 各自筛选 <6000 的样本 在 val  上预测的 candicate tail的出现次数
    for epoch in range(head_upsample_epochs):
        for i in tqdm.tqdm(range(val_trip.shape[0])):
            tmp_val = val_trip[i]
            if train_tail_count[tmp_val[2]]+val_tail_count[tmp_val[2]] <=6000 or val_tail_count[tmp_val[2]]==-1:
                new_train_hrt.append(tmp_val)
            
    print('after add 6000 filter and topK filter')
    print(len(new_train_hrt))   
            
    np.random.shuffle(new_train_hrt)
    try:
        print(len(new_train_hrt)) 
    except:
        print('haha')
    np.save('./dataset/wikikg90m_kddcup2021/processed/upsample_on_val_wyk.npy', new_train_hrt)
    return 0

if __name__ == '__main__':
	generator_new_train()
	generator_new_val()
