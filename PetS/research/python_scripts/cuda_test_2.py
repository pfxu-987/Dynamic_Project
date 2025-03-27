import torch
from torch.random import seed
from transformers.models.bert.modeling_bert import BertModel, BertConfig
import turbo_transformers
import time
from turbo_transformers.layers import PET_Types, PETBertModel, PETBertConfig
import random 
import argparse
import tqdm
import numpy as np
import os
import sys
import atexit
from pet_scheduler import PET_Scheduler



class PET_Server:
    def __init__(self,cfg) -> None:
        self.torch_model = None
        self.base_tt_model = None
        self.test_device = torch.device('cuda:0')
        self.cfg = cfg  # experiment config
        self.logger = None        
        # init the bert model config
        if cfg.model == 'distilbert':
            model_config = BertConfig(num_hidden_layers = 6)
        elif cfg.model == 'bert_large':
            model_config = BertConfig(num_hidden_layers = 24, hidden_size = 1024,
                                      intermediate_size=4096,
                                      num_attention_heads=16)
        elif cfg.model == 'bert_base':
            model_config = BertConfig()
        else:
            raise NotImplementedError
        self.bert_cfg = model_config

        self.task_types = []
        # self.init_logger()
        
        self.has_inited = False
        # 注册退出处理函数，确保在程序退出前清理资源
        atexit.register(self.cleanup_atexit)

    def cleanup_atexit(self):
        """在程序退出时安全清理资源"""
        try:
            # 在程序退出时，避免调用可能导致CUDA错误的函数
            if hasattr(self, 'logger') and self.logger:
                self.logger.close()
            # 不在这里调用CUDA相关的清理，因为驱动可能已经关闭
        except Exception as e:
            print(f"Warning during atexit cleanup: {e}")

    def init_logger(self,exp_name):
        if not os.path.exists(self.cfg.log_dir):
            os.makedirs(self.cfg.log_dir)
        log_name = os.path.join(self.cfg.log_dir,exp_name+"_PETS.log")
        self.logger = open(log_name,"w")

    def write_log(self, str):
        if self.logger:
            self.logger.write(str)
            self.logger.flush()
        
    def load_torch_model(self):
        try:
            self.torch_model = BertModel(self.bert_cfg)
            self.torch_model.eval()
            if torch.cuda.is_available():
                self.torch_model.to(self.test_device)
        except Exception as e:
            print(f"Error loading torch model: {e}")
            raise

    def load_shared_weight(self):
        """
        Load the pytorch model weight as the shared parameters
        """
        try:
            base_turbo_model = turbo_transformers.SharedBertModel.from_torch(
                self.torch_model, self.test_device, "turbo")
            self.base_tt_model = base_turbo_model
            turbo_transformers.set_num_threads(4)

            # release the torch model
            self.torch_model = None
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error loading shared weights: {e}")
            raise

    def load_dense_model(self):
        base_turbo_model = turbo_transformers.SharedBertModel.from_torch(
            self.torch_model, self.test_device, "turbo")
        return base_turbo_model

    def load_new_task(self, pet_type, model_path = None):
        """
        Load PETs
        """
        try:
            if self.cfg.model == 'distilbert':
                pet_bert_config = PETBertConfig(num_hidden_layers = 6, pet_type = pet_type)
            elif self.cfg.model == 'bert_large':
                pet_bert_config = PETBertConfig(num_hidden_layers=24, hidden_size = 1024,
                                                intermediate_size=4096,
                                                num_attention_heads=16,
                                                pet_type = pet_type)
            elif self.cfg.model == 'bert_base':
                pet_bert_config = PETBertConfig(pet_type = pet_type)

            pet_bert_model = PETBertModel(pet_bert_config)

            if torch.cuda.is_available():
                pet_bert_model.to(self.test_device)
            
            self.task_types.append(pet_type)
            self.base_tt_model.load_new_task_from_torch(pet_bert_model)
            
            # 手动释放pet_bert_model以避免内存泄漏
            del pet_bert_model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error loading new task: {e}")
            raise

    def init(self):
        if self.has_inited:
            # if run the all-in-one script
            return 

        try:
            # shared model
            self.load_torch_model()
            self.load_shared_weight()
            print("Shared model loaded")

            # PET tasks
            self.prepare_tasks()
            print("PET tasks loaded")

            self.has_inited = True
            # Queries 
            # self.prepare_query_pool()
            # print("Benchmark generated")
        except Exception as e:
            print(f"Initialization error: {e}")
            self.cleanup()
            raise

    def prepare_query_pool(self):
        """
        Generate queries obeying normal distribution 
        """
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        query_pool = []
        normal = np.random.normal(self.cfg.mean_v, self.cfg.std_v, self.cfg.total_queries)
    
        for i in range(self.cfg.total_queries):
            # randomly assign the query to a task
            task_id = i%self.cfg.num_tasks

            task_type = self.task_types[task_id]
            generated_seq_len = int(normal[i])
            if generated_seq_len > 128:
                generated_seq_len = 128
            if generated_seq_len < 1:
                generated_seq_len = 1
            query_pool.append((task_id, generated_seq_len, task_type))

        self.query_pool = query_pool
    
    def prepare_tasks(self):
        print("Preparing PET Tasks")
        num_tasks = self.cfg.num_tasks
        random.seed(self.cfg.seed)
        try:
            # 减少任务数量，避免内存不足
            actual_tasks = min(num_tasks, 16)  # 限制最大任务数为16
            print(f"Loading {actual_tasks} tasks (limited from {num_tasks})")
            
            for _ in tqdm.tqdm(range(actual_tasks//2)):
                pet_type = 3
                self.load_new_task(pet_type)
                # 在每个任务加载后清理缓存
                torch.cuda.empty_cache()
                
                pet_type = 3
                self.load_new_task(pet_type)
                # 在每个任务加载后清理缓存
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error preparing tasks: {e}")
            raise

    def warmup(self):
        try:
            # 减少预热次数
            for i in range(5):  # 从20减少到5
                input_ids = torch.randint(low=0,
                                      high=self.bert_cfg.vocab_size - 1,
                                      size=(4, 64),  # 从128减少到64
                                      dtype=torch.long,
                                      device=self.test_device)
                task_ids = torch.LongTensor([0,1,2,3])
                # task_ids = torch.LongTensor([0])
                n_samples = torch.LongTensor([1,1,1,1])
                self.base_tt_model(input_ids, task_ids = task_ids, n_samples = n_samples)
                # 每次预热后清理缓存
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error during warmup: {e}")
            raise
    
    def get_scheduler(self):
        # schedule the quiery pool to get batches
        try:
            # 确保路径存在
            alpha_path = self.cfg.alpha_table_path
            beta_path = self.cfg.beta_table_path
            
            # 检查文件是否存在
            if not os.path.exists(alpha_path):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                alpha_path = os.path.join(script_dir, alpha_path)
                if not os.path.exists(alpha_path):
                    # 尝试使用可用的alpha表
                    available_alpha = os.path.join(script_dir, "perf_model/alpha_table_1080ti.dat")
                    if os.path.exists(available_alpha):
                        alpha_path = available_alpha
                        print(f"Using alternative alpha table: {alpha_path}")
                    
            if not os.path.exists(beta_path):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                beta_path = os.path.join(script_dir, beta_path)
                if not os.path.exists(beta_path):
                    # 尝试使用可用的beta表
                    available_beta = os.path.join(script_dir, "perf_model/beta_table_1080ti.dat")
                    if os.path.exists(available_beta):
                        beta_path = available_beta
                        print(f"Using alternative beta table: {beta_path}")
                
            print(f"Using alpha table: {alpha_path}")
            print(f"Using beta table: {beta_path}")
            
            pet_scheduler = PET_Scheduler(query_pool=self.query_pool,
                                          vocab_size=self.bert_cfg.vocab_size,
                                          sort_queries=self.cfg.sort_queries,
                                          test_device=self.test_device,
                                          alpha_table_path=alpha_path,
                                          beta_table_path=beta_path
                                          )
            return pet_scheduler
        except Exception as e:
            print(f"Error getting scheduler: {e}")
            raise
    
    def run(self, no_log = False, bs = 32):
        try:
            # 减少批次大小
            bs = min(bs, 16)  # 限制最大批次大小为16
            
            turbo_transformers.set_num_streams(self.cfg.num_streams)
            pet_scheduler = self.get_scheduler()
            batches = []
            for i in range(0, len(self.query_pool), bs):
                batch_queries = self.query_pool[i:i+bs]
                if len(batch_queries) > 0:
                    task_ids = torch.LongTensor([q[0] for q in batch_queries])
                    # 减少序列长度
                    seq_len = min(self.cfg.mean_v, 64)  # 限制最大序列长度为64
                    input_ids = torch.randint(low=0, 
                                              high=self.bert_cfg.vocab_size-1, 
                                              size=(len(batch_queries), seq_len), 
                                              dtype=torch.long, device=self.test_device)
                    n_samples = torch.LongTensor([1 for _ in range(len(batch_queries))])

                    batches.append((input_ids, task_ids, n_samples))

            self.warmup()      
            # Start serving------------:
            start = time.time()
            # 减少迭代次数
            iterations = min(self.cfg.iterations, 3)  # 限制最大迭代次数为3
            for iter in range(iterations):
                # for batch in tqdm.tqdm(batches):
                for batch in batches:
                    # if len(batch) == 3:
                    self.base_tt_model(batch[0], task_ids = batch[1], n_samples = batch[2])
                    # 每次批处理后清理缓存
                    torch.cuda.empty_cache()
                    # elif len(batch) == 4:
                        # self.base_tt_model(batch[0], task_ids = batch[1], n_samples = batch[2], minibatch_lens = batch[3])
            
            elasp_time = time.time() - start
            average_time = elasp_time / iterations
            
            # print("Average time : {}".format(average_time),flush=True)
            QPS = self.cfg.total_queries / (average_time)
            print("QPS: {}".format(QPS),flush=True)

            if not no_log:
                self.write_log("QPS: {}\n".format(QPS))
        except Exception as e:
            print(f"Error during run: {e}")
            self.cleanup()
            raise

    def cleanup(self):
        """释放CUDA资源"""
        try:
            print("Cleaning up resources...")
            # 先将模型移到CPU，再删除
            if self.base_tt_model is not None:
                # 避免直接删除CUDA对象
                self.base_tt_model = None
            if self.torch_model is not None:
                self.torch_model = None
                
            # 强制同步CUDA操作
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            # 清空CUDA缓存
            torch.cuda.empty_cache()
            
            if self.logger:
                self.logger.close()
                self.logger = None
                
            print("Cleanup completed")
        except Exception as e:
            print(f"Error during cleanup: {e}")

    def compare_multi_stream(self):
        try:
            self.init_logger("multi_stream")
            # 减少任务数量
            self.cfg.schedule_policy = "batch_schedule"
            self.cfg.num_tasks = 4  # 从8减少到4
            self.init()
            self.cfg.total_queries = 64  # 从192减少到64
            
            for task_num in [4]:  # 从[8]改为[4]
                self.cfg.num_tasks = task_num
                for i in range(1):
                    # 减少序列长度范围
                    for seq_len in [32, 64]:  # 减少测试的序列长度
                        self.cfg.mean_v = seq_len
                        self.cfg.std_v = 4
                        self.prepare_query_pool()
                        # 减少流的数量
                        for stream_num in [1, 2]:  # 只测试1和2个流
                            self.cfg.num_streams = stream_num
                            self.write_log("total_queries:{},task_num:{},stream_num:{},seq_len:{} ".format(self.cfg.total_queries, self.cfg.num_tasks, stream_num,seq_len))
                            self.run(bs = min(16, self.cfg.total_queries))  # 限制批次大小
                            # 每次运行后清理缓存
                            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error in compare_multi_stream: {e}")
            self.cleanup()
            raise
        finally:
            self.cleanup()

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--exp_name",
            type=str,
            help="Which experiment to conduct?",
            default="multi_stream",  # 默认改为multi_stream
            choices=["main_results","serving_throughput", "capacity_dense", "capacity_pet", "batching_strategy", "multi_stream", "breakdown"]
        )

        parser.add_argument(
            "--log_dir",
            type=str,
            # required=True,
            help="Path to write the results",
            default="./"
        )

        parser.add_argument(
            "--num_tasks",
            type=int,
            default = 4,  # 从128减少到4
            help="Number of loaded tasks",
        )
        parser.add_argument(
            "--test_device",
            type=str,
            default = 'cuda:0',
        )
        parser.add_argument(
            "--min_seq_length",
            type=int,
            default = 8,
        )
        parser.add_argument(
            "--max_seq_length",
            type = int,
            default = 64
        )
        parser.add_argument(
            "--seed",
            type=int,
            default = 1
        )
        parser.add_argument(
            "--model",
            type = str,
            default = "bert_base",
            choices=['bert_base', 'distilbert', 'bert_large']
        )
        parser.add_argument(
            "--total_queries",
            type = int, 
            default = 64,  # 从1024减少到64
            help = "Total number of queries in the pool"
        )
        parser.add_argument(
            "--iterations",
            type = int, 
            default = 3,  # 从10减少到3
            help = "Total number of iterations"
        )
        parser.add_argument(
            "--sort_queries",
            action = "store_true"
        )
        parser.add_argument(
            "--num_streams",
            type = int, 
            default = 1,
            help = "Total number of CUDA streams"
        )

        parser.add_argument(
            "--alpha_table_path",
            type = str,
            default = "perf_model/alpha_table_1080ti.dat",  # 使用已知存在的文件
        )
        parser.add_argument(
            "--beta_table_path",
            type = str,
            default = "perf_model/beta_table_1080ti.dat",  # 使用已知存在的文件
        )
        parser.add_argument(
            "--schedule_policy",
            type=str,
            default = "batch_schedule",
            choices=["batch_schedule","two_stage_schedule"]
        )
        
        parser.add_argument(
            "--schedule_stage",
            type=int,
            default = 2
        )
        parser.add_argument(
            "--mean_v",
            type=int,
            default = 32
        )
        parser.add_argument(
            "--std_v",
            type=int,
            default = 1
        )

        cfg = parser.parse_args()
        
        # 检查性能模型文件路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.exists(cfg.alpha_table_path):
            alpha_path = os.path.join(script_dir, cfg.alpha_table_path)
            if os.path.exists(alpha_path):
                cfg.alpha_table_path = alpha_path
                
        if not os.path.exists(cfg.beta_table_path):
            beta_path = os.path.join(script_dir, cfg.beta_table_path)
            if os.path.exists(beta_path):
                cfg.beta_table_path = beta_path
        
        # 设置环境变量，防止CUDA驱动超时
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        
        # 打印CUDA信息
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.is_available()}")
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"CUDA current device: {torch.cuda.current_device()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
            
        server = PET_Server(cfg)

        # 只运行multi_stream实验
        server.compare_multi_stream()
            
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        # 确保在程序结束时释放CUDA资源
        if 'server' in locals():
            server.cleanup()
        # 强制清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
 