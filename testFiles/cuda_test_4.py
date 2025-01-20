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
        #设定 0任务数量为nums_task0，其余任务数量为(self.batch_size-nums_task0)//7
        task_ids = torch.LongTensor([0]*nums_task0 + [i for i in range(1,8)]*((self.batch_size-nums_task0)//7))


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
#同一pet多流实验
    server = PET_Server()
    # load the shared model
    server.init()

    # load for tasks:
    print("Loading PET tasks...")
    for i in tqdm(range(8)):
    #for pet_type in tqdm([PET_Types.maskbert,PET_Types.maskbert]):
    # for pet_type in tqdm([PET_Types.maskbert,PET_Types.adapters]):
    # for pet_type in tqdm([PET_Types.maskbert,PET_Types.bitfit]):
    # for pet_type in tqdm([PET_Types.diff_pruning,PET_Types.adapters]):
    # for pet_type in tqdm([PET_Types.diff_pruning,PET_Types.bitfit]):
    # for pet_type in tqdm([PET_Types.adapters,PET_Types.bitfit]):
        server.load_new_task(PET_Types.maskbert)
    
    seed = 411

    for batch_size in [128]:
        for seq_len in [64,128]:
            for nums_task0 in [128-i*7 for i in range(0,19,4)]:
                #[128, 100, 72, 44, 16]
                #[0, 4, 8, 12, 16]
                random.seed(seed)
                torch.manual_seed(seed)
                torch.torch.cuda.manual_seed(seed)  # 当前设备
                inputs = server.prepare_inputs(batch_size, seq_len,nums_task0)
                numstreams_time_list = {numstreams: [] for numstreams in [8, 4, 2, 1]}
            
                for i in range(5):
                    for numstreams in [8, 4, 2, 1]:
                        ans = server.run(inputs, numstreams)
                        numstreams_time_list[numstreams].append(ans)

                # Calculate median times for each stream configuration
                median_times = {numstreams: statistics.median(times) for numstreams, times in numstreams_time_list.items()}

                for numstreams in [8, 4, 2, 1]:
                    print(f"nums_task0:{nums_task0},batch_size:{batch_size},seq_len:{seq_len},numstreams:{numstreams},time:{median_times[numstreams]:.4f}")
                    # Log results to file
                    with open('cuda_test_4_results.log', 'a') as f:
                        f.write(f"nums_task0:{nums_task0},batch_size:{batch_size},seq_len:{seq_len},numstreams:{numstreams},time:{median_times[numstreams]:.4f}\n")
                print(f"{'='*50}")
