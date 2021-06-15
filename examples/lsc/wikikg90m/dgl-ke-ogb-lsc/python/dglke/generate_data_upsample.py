import numpy as np
import pandas as pd
import torch as th
import tqdm
import os

def get_head_score_train(reload=False):
    saved_path = './dataset/wikikg90m_kddcup2021/processed/trian_head_score_wyk.npy'
    if reload and os.path.isfile(saved_path):
        return np.load(saved_path)
    trip_list = np.load('./dataset/wikikg90m_kddcup2021/processed/train_hrt.npy')
    n_sample = trip_list.shape[0]
    head_score = np.zeros((87143637, 1))
    for i in range(n_sample):
        head_score[trip_list[i][0]] += 1
    np.save(saved_path, head_score)
    return head_score

def get_tail_score_train(reload=False):
    saved_path = './dataset/wikikg90m_kddcup2021/processed/trian_tail_score_wyk.npy'
    if reload and os.path.isfile(saved_path):
        return np.load(saved_path)
    trip_list = np.load('./dataset/wikikg90m_kddcup2021/processed/train_hrt.npy')
    n_sample = trip_list.shape[0]
    tail_score = np.zeros((87143637, 1))
    for i in range(n_sample):
        tail_score[trip_list[i][2]] += 1
    np.save(saved_path, tail_score)
    return tail_score

def get_head_score_val(reload=False):
    saved_path = './dataset/wikikg90m_kddcup2021/processed/val_head_score_wyk.npy'
    if reload and os.path.isfile(saved_path):
        return np.load(saved_path)
    candidate_list = np.load('./dataset/wikikg90m_kddcup2021/processed/val_hr.npy')
    n_sample = candidate_list.shape[0]
    head_score = np.zeros((87143637, 1))
    for i in tqdm.tqdm(range(n_sample)):
        head_score[candidate_list[i][0]] += 1
    np.save(saved_path, head_score)
    print('head count have saved        ')
    return head_score

def get_tail_score_val(reload=False):
    saved_path = './dataset/wikikg90m_kddcup2021/processed/val_tail_score_wyk.npy'
    if reload and os.path.isfile(saved_path):
        return np.load(saved_path)
    candidate_list = generator_val_hrt()
    n_sample = candidate_list.shape[0]
    n_candidate = candidate_list.shape[1]
    tail_score = np.zeros((87143637, 1))
    for i in range(n_sample):
        tail_score[candidate_list[i][2]] += 1
    np.save(saved_path, tail_score)
    return tail_score

def generator_val_hrt():
    val_candis = np.load('./dataset/wikikg90m_kddcup2021/processed/val_t_candidate.npy')
    val_correct_index = np.load('./dataset/wikikg90m_kddcup2021/processed/val_t_correct_index.npy')
    val_hr = np.load('./dataset/wikikg90m_kddcup2021/processed/val_hr.npy')
    print('loaded val true candi')
    val_t = []
    for i in tqdm.tqdm(range(val_candis.shape[0])):
        val_t.append(val_candis[i][val_correct_index[i]])
    val_t = np.array(val_t).reshape(len(val_t),1)
    val_hrt = np.concatenate((val_hr, val_t), axis = 1)
    np.save('./dataset/wikikg90m_kddcup2021/processed/val_hrt_wyk.npy', val_hrt)
    return val_hrt


def generator_new_train(include_val=True, head_upsample_epochs=2,t_threshold,v_threshold):
    
    train_tail_score = get_tail_score_train()
    val_tail_score = get_tail_score_val()
    
    train_trip = np.load('./dataset/wikikg90m_kddcup2021/processed/train_hrt.npy')
    val_trip = np.load('./dataset/wikikg90m_kddcup2021/processed/val_hrt_wyk.npy')
    
    new_train_hrt = []
    
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
            for topk_index in range(filter_topk):  
                tmp_index = val_candi[global_index][ori_res_pred_top10[index][topk_index]]
                if train_tail_score[tmp_index]<=t_threshold+val_tail_score[tmp_index]<=v_threshold:
                    val_tail_score[tmp_index] = -1
                    counter1 += 1
            global_index += 1
    
    global_index = 0
    
    for i in tqdm.tqdm(range(train_trip.shape[0])):
        tmp_train = train_trip[i]
        if (train_tail_score[tmp_train[2]]+val_tail_score[tmp_train[2]] <=t_threshold) or val_tail_score[tmp_train[2]]==-1:
            new_train_hrt.append(tmp_train)
    
    val_head_score = get_head_score_val()
    for epoch in range(head_upsample_epochs):
        for i in tqdm.tqdm(range(train_trip.shape[0])):
            tmp_train = train_trip[i]
            if val_head_score[tmp_train[0]] > 0:
                new_train_hrt.append(tmp_train)

    np.random.shuffle(new_train_hrt)
    try:
        print(len(new_train_hrt))   
    except:
        print('errors')
    np.save('./dataset/wikikg90m_kddcup2021/processed/trian_val_topk_add_h.npy', new_train_hrt)
    return 0

def generator_new_val(include_val=True,head_upsample_epochs=2,t_threshold,v_threshold):
    
    train_tail_score = get_tail_score_train()
    val_tail_score = get_tail_score_val()
    
    train_trip = np.load('./dataset/wikikg90m_kddcup2021/processed/train_hrt.npy')
    val_trip = np.load('./dataset/wikikg90m_kddcup2021/processed/val_hrt_wyk.npy')
    
    new_train_hrt = []
    
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
            for topk_index in range(filter_topk):  
                tmp_index = val_candi[global_index][ori_res_pred_top10[index][topk_index]]
                if train_tail_score[tmp_index]<=t_threshold+val_tail_score[tmp_index]<=v_threshold:
                    val_tail_score[tmp_index] = -1
                    counter1 += 1
            global_index += 1
    
    for epoch in range(head_upsample_epochs):
        for i in tqdm.tqdm(range(val_trip.shape[0])):
            tmp_val = val_trip[i]
            if train_tail_score[tmp_val[2]]+val_tail_score[tmp_val[2]] <=t_threshold or val_tail_score[tmp_val[2]]==-1:
                new_train_hrt.append(tmp_val)
            
    np.random.shuffle(new_train_hrt)
    try:
        print(len(new_train_hrt)) 
    except:
        print('errors')
    np.save('./dataset/wikikg90m_kddcup2021/processed/upsample_on_val_wyk.npy', new_train_hrt)
    return 0

def generator_for_finefune(include_val=True,head_upsample_epochs=1,t_threshold,v_threshold):
    
    train_tail_score = get_tail_score_train()
    val_tail_score = get_tail_score_val()
    
    train_trip = np.load('./dataset/wikikg90m_kddcup2021/processed/train_hrt.npy')
    val_trip = np.load('./dataset/wikikg90m_kddcup2021/processed/val_hrt_wyk.npy')
    
    new_train_hrt = []
    
    val_candi = np.load('./dataset/wikikg90m_kddcup2021/processed/val_t_candidate.npy')
    
    global_index = 0
    filter_topk = 10
    
    proc_num = 4
    for proc_no in range(proc_num):
        ori_res = th.load(f"./ensemble_valid_v2_backup/load_ensemble_valid_rerank_{proc_no}.pkl", map_location=th.device('cpu'))
        ori_res_pred_top10 = ori_res['h,r->t']['t_pred_top10'][:,:10].numpy()
        print(ori_res['h,r->t']['t_pred_top10'].shape)
        res_correct_index = ori_res['h,r->t']['t_correct_index'].numpy()
        
        for index in tqdm.tqdm(range(res_correct_index.shape[0])):  
            for topk_index in range(filter_topk): 
                tmp_index = val_candi[global_index][ori_res_pred_top10[index][topk_index]]
                if train_tail_score[tmp_index]<=t_threshold+val_tail_score[tmp_index]<=v_threshold:
                    val_tail_score[tmp_index] = -1
                    counter1 += 1
            global_index += 1

    for i in tqdm.tqdm(range(train_trip.shape[0])):
        tmp_train = train_trip[i]
        if val_tail_score[tmp_train[2]]==-1:
            new_train_hrt.append(tmp_train)

    head_apper_count = 0
    val_head_score = np.load('./dataset/wikikg90m_kddcup2021/processed/val_head_score_wyk.npy')
    for epoch in range(head_upsample_epochs):
        for i in tqdm.tqdm(range(train_trip.shape[0])):
            tmp_train = train_trip[i]
            if val_head_score[tmp_train[0]] > 0:
                new_train_hrt.append(tmp_train)
            
    np.random.shuffle(new_train_hrt)
    try:
        print(len(new_train_hrt))  
    except:
        print('errors')
    np.save('./dataset/wikikg90m_kddcup2021/processed/trian_for_finetune.npy', new_train_hrt)
    return 0

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--include_val',  help='whether include val')
    parser.add_argument('--head_upsample_epochs',  help='upample rate') 
    parser.add_argument('--t_threshold',  help='train score threshold')
    parser.add_argument('--v_threshold',  help='val score threshold')
    generator_val_hrt()
    generator_new_train(args.include_val,args.head_upsample_epochs,args.t_threshold,args.v_threshold)
    generator_new_val(args.include_val,args.head_upsample_epochs,args.t_threshold,args.v_threshold)
    generator_for_finefune(args.include_val,args.head_upsample_epochs,args.t_threshold,args.v_threshold)
