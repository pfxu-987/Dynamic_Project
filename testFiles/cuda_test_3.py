from tqdm import tqdm
import torch
# from torch._C import _load_for_lite_interpreter
from transformers.modeling_bert import BertModel, BertConfig
import numpy
import turbo_transformers
import sys
import os
import time
import random
from turbo_transformers.layers import PET_Types, PETBertModel, PETBertConfig
import statistics

class PET_Server:
    def __init__(self) -> None:
        self.base_torch_model = None
        self.base_tt_model = None
        self.test_device = torch.device('cuda:0')
        self.cfg = BertConfig(num_hidden_layers=12)
        self.task_torch_models = []

        self.num_task_a = None
        self.num_task_b = None
        self.num_k = None

    def load_torch_model(self):
        self.base_torch_model = BertModel(self.cfg)
        self.base_torch_model.eval()
        if torch.cuda.is_available():
            self.base_torch_model.to(self.test_device)

    def load_shared_w(self):
        base_turbo_model = turbo_transformers.SharedBertModel.from_torch(
            self.base_torch_model, self.test_device, "turbo")
        self.base_tt_model = base_turbo_model
        turbo_transformers.set_num_threads(4)
    
    def load_new_task(self, pet_type, model_path = None):
        """
        Load PETs 
        """
        pet_bert_config = PETBertConfig(pet_type = pet_type, num_hidden_layers=12)
        pet_bert_model = PETBertModel(pet_bert_config)
        pet_bert_model.eval()

        # load the shared parts from base torch model
        for k,v in self.base_torch_model.named_parameters():
            if pet_type == PET_Types.bitfit:
                # exclude the bias and layer norm params
                if ("bias" in k) or ("LayerNorm" in k):
                    continue
            elif pet_type == PET_Types.diff_pruning:
                if ("bias" in k) or ("LayerNorm" in k):
                    continue
            pet_bert_model.state_dict()[k].copy_(v.clone())
            
        if torch.cuda.is_available():
                pet_bert_model.to(self.test_device)
        
        self.task_torch_models.append(pet_bert_model)
        self.base_tt_model.load_new_task_from_torch(pet_bert_model)

    def init(self):
        print("Init shared Model:")
        self.load_torch_model()
        self.load_shared_w()

    def warmup(self):
        for i in range(10):
            input_ids = torch.randint(low=0,
                                  high=self.cfg.vocab_size - 1,
                                  size=(4, 128),
                                  dtype=torch.long,
                                  device=self.test_device)
            task_ids = torch.LongTensor([0,1,0,1])
            n_samples = torch.LongTensor([1,1,1,1])
            self.base_tt_model(input_ids, task_ids = task_ids, n_samples = n_samples)

    def prepare_inputs(self, batch_size, seq_len,nums_task0):
        self.batch_size = batch_size
        self.seq_len = seq_len
        
        task_ids = torch.LongTensor([0]*nums_task0 + [1] * (self.batch_size-nums_task0))


        n_samples = torch.LongTensor([1 for _ in range(self.batch_size)])
        
        input_ids = torch.randint(low=0,
                                 high=self.cfg.vocab_size - 1,
                                 size=(self.batch_size, self.seq_len),
                                 dtype=torch.long,
                                 device=self.test_device)

        return [input_ids, task_ids, n_samples]

    def run(self, inputs, num_streams):
        turbo_transformers.set_num_streams(num_streams)
        
        if not hasattr(self, 'warmed_up'):
            self.warmup()
            self.warmed_up = True
        
        # 记录开始时间
        torch.cuda.synchronize()
        start_time = time.time()
        
        # 执行推理
        for i in range(10):
            pet_output = self.base_tt_model(inputs[0], 
                                        task_ids=inputs[1], 
                                        n_samples=inputs[2])[0]
        
        # 确保所有操作完成
        torch.cuda.synchronize()
        end_time = time.time()
        
        # 打印性能信息
        return end_time - start_time

if __name__ == '__main__':

    server = PET_Server()
    # load the shared model
    server.init()

    # load for tasks:
    print("Loading PET tasks...")

    for pet_type in tqdm([PET_Types.maskbert,PET_Types.adapters]):
        server.load_new_task(pet_type)
    
    seed = 411

    for batch_size in [128]:
        for seq_len in [64,128]:
            for nums_task0 in [1,4,8,16,32,64,127]:
                random.seed(seed)
                torch.manual_seed(seed)
                torch.torch.cuda.manual_seed(seed)  # 当前设备
                inputs = server.prepare_inputs(batch_size, seq_len,nums_task0)
                numstreams_1_time_list= []
                numstreams_2_time_list= []
            
                for i in range(9):
                    for numstreams in [2,1]:
                        ans = server.run(inputs,numstreams)
                        if numstreams == 1:
                            numstreams_1_time_list.append(ans)
                        else:
                            numstreams_2_time_list.append(ans)

                # Calculate median times for both stream configurations
                median_time_1stream = statistics.median(numstreams_1_time_list)
                median_time_2stream = statistics.median(numstreams_2_time_list)
                print(f"nums_task0:{nums_task0},batch_size:{batch_size},seq_len:{seq_len},numstreams:1,time:{median_time_1stream:.4f}")
                print(f"nums_task0:{nums_task0},batch_size:{batch_size},seq_len:{seq_len},numstreams:2,time:{median_time_2stream:.4f}")
                print(f"{'='*50}")
                # Log results to file
                with open('cuda_test_3_results.log', 'a') as f:
                    f.write(f"nums_task0:{nums_task0},batch_size:{batch_size},seq_len:{seq_len},numstreams:1,time:{median_time_1stream:.4f}\n")
                    f.write(f"nums_task0:{nums_task0},batch_size:{batch_size},seq_len:{seq_len},numstreams:2,time:{median_time_2stream:.4f}\n")
