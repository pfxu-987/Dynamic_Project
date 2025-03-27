from re import L
from turtle import color, write_docstringdict
from matplotlib import pyplot as plt
import argparse
import os
import numpy as np
plt.style.use('ggplot')
def plot_figure_7(cfg):
    model = cfg.model
    results_path = cfg.results_path
    qps_log_file_path = os.path.join(results_path,"pets",model,"serving_throughput_PETS.log")
    if not os.path.exists(qps_log_file_path):
        print(qps_log_file_path,"does not exist!")
        return

    qps_log = open(qps_log_file_path,"r").readlines()

    task_numbers = [1,16,32,64]
    bs_seqlen_configs = [(1,128),(2,64),(4,32)]

    plot_data = []
    for bs,seq_len in bs_seqlen_configs:
        data = []
        for task_number in task_numbers:
            for line in qps_log:
                if "task_num:{},".format(task_number) in line:
                    if("bs:{},seq_len:{},".format(bs,seq_len) in line):
                        qps = int(eval(line.strip().split("QPS: ")[-1]))
                        data.append(qps)
                        # print(bs,seq_len,task_number, qps)
        plot_data.append(data)
    
    plot_data = np.array(plot_data)

    #norm to single task
    normed_plot_data = np.ones_like(plot_data).astype(np.float)
    for i in range(3):
        for j in range(4):
            normed_plot_data[i][j] = plot_data[i][j]/plot_data[i][0]

    plot_data = np.transpose(normed_plot_data)
    total_width, n = 0.8, 4

    x = np.arange(3)
    width = total_width / n
    x = x - (total_width - width) / 2

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.bar(x, plot_data[0], width= width, label = "Single Task")
    ax.bar(x + width, plot_data[1], width= width, label = "Task={}".format(task_numbers[1]))
    ax.bar(x + 2 * width, plot_data[2], width= width, label ="Task={}".format(task_numbers[2]))
    ax.bar(x + 3 * width, plot_data[3], width= width, label = "Task={}".format(task_numbers[3]))

    ax.legend()
    ax.set_xticks(np.arange(3))
    ax.set_ylabel('Norm. QPS')
    ax.set_xticklabels(["{1,128}","{2,64}","{4,32}"])
    print("reproduced_figures/fig_7.jpg")
    plt.savefig("reproduced_figures/fig_7.jpg",bbox_inches='tight')

def plot_figure_8(cfg):
    model = cfg.model
    results_path = cfg.results_path
    pets_qps_log_file_path = os.path.join(results_path,"pets",model,"serving_throughput_PETS.log")
    if not os.path.exists(pets_qps_log_file_path):
        print(pets_qps_log_file_path,"does not exist!")
        return
    pars_qps_log_file_path = os.path.join(results_path,"pars",model,"serving_throughput_PARS.log")
    if not os.path.exists(pars_qps_log_file_path):
        print(pars_qps_log_file_path,"does not exist!")
        return
    seqs_qps_log_file_path = os.path.join(results_path,"seqs",model,"serving_throughput_SEQS.log")
    if not os.path.exists(seqs_qps_log_file_path):
        print(seqs_qps_log_file_path,"does not exist!")
        return

    bs, seq_len = 1, 128
    tasks = [2,4,8,16]
    pars_qps = []
    pars_qps_log = open(pars_qps_log_file_path,"r").readlines()
    for line in pars_qps_log:
        if "QPS" in line:
            qps = line.strip().split(" ")[-1]
            pars_qps.append(eval(qps))
    
    pets_qps = []
    pets_qps_log = open(pets_qps_log_file_path,"r").readlines()
    for n_task in tasks:
        for line in pets_qps_log:
            if("task_num:{},bs:{},seq_len:{},".format(n_task,bs,seq_len) in line):
                qps = line.strip().split(" ")[-1]
                pets_qps.append(eval(qps))

    seq_qps = []
    seqs_qps_log = open(seqs_qps_log_file_path,"r").readlines()
    lines = len(seqs_qps_log)
    for i in range(lines//2):
        exp_cfg = seqs_qps_log[2*i]
        qps = seqs_qps_log[2*i+1]
        if("bs:{},seq_len:{}".format(bs,seq_len) in exp_cfg):
            qps = eval(qps.strip().split(" ")[-1])
            seq_qps.append(qps)
    
    total_width, n = 0.8, 3

    x = np.arange(4)
    width = total_width / n
    x = x - (total_width - width) / 2

    fig, ax = plt.subplots(figsize=(6,5))

    ax.bar(x, seq_qps, width= width, label = "SEQS")
    ax.bar(x + width, pars_qps, width= width, label = "PARS")
    ax.bar(x + 2 * width, pets_qps, width= width, label ="PETS")

    ax.legend()
    ax.set_xticks(np.arange(4))
    ax.set_ylabel('QPS')
    ax.set_xticklabels(tasks)
    ax.set_xlabel('# of tasks (BL,SL = {1,128})')
    print("reproduced_figures/fig_8.jpg")
    plt.savefig("reproduced_figures/fig_8.jpg",bbox_inches='tight')

def plot_figure_11(cfg):
    model = cfg.model
    results_path = cfg.results_path
    pets_qps_log_file_path = os.path.join(results_path,"pets",model,"multi_stream_PETS.log")
    if not os.path.exists(pets_qps_log_file_path):
        print(pets_qps_log_file_path,"does not exist!")
        return

    qps_log = open(pets_qps_log_file_path,"r").readlines()
    
    n_tasks = [128,64,32]
    streams = [1,2,4,8,16,32]
    seq_lens = [64,32,16,8,4]
    subplots = [131,132,133]
    
    markers = ["o","^","s","d","_"]
    # colors = ["grey","blue","orange","yellow","grey"]
    fig = plt.figure(figsize=(15,5))
    for id, n_task in enumerate(n_tasks):
        plot_data = []
        for seq_len in seq_lens:
            data = []
            normed_data = []
            for n_stream in streams:
                for line in qps_log:
                    if("task_num:{},stream_num:{},seq_len:{}".format(n_task,n_stream,seq_len) in line):
                        qps = line.strip().split(" ")[-1]
                        data.append(eval(qps))
                        normed_data.append(eval(qps)/data[0])
            plot_data.append(normed_data)

        ax = fig.add_subplot(subplots[id])
        for i, data in enumerate(plot_data):
            ax.plot(streams, data, label = "seq={}".format(seq_lens[i]),marker=markers[i])
            ax.legend()
        plt.ylim((0.6,1.2))
        plt.xscale("log")
        ax.set_xticks(streams)
        ax.set_xticklabels(streams)
        ax.set_xlabel('# of Streams (Task = {})'.format(n_task))
    print("reproduced_figures/fig_11.jpg")
    plt.savefig("reproduced_figures/fig_11.jpg",bbox_inches='tight')

def plot_figure_12(cfg):
    model = cfg.model
    results_path = cfg.results_path
    pets_qps_log_file_path = os.path.join(results_path,"pets",model,"compare_batching_PETS.log")
    if not os.path.exists(pets_qps_log_file_path):
        print(pets_qps_log_file_path,"does not exist!")
        return

    qps_log = open(pets_qps_log_file_path,"r").readlines()
    stds = [1,2,4,8]
    stages = [0,1,2,3]
    tasks = [128,64,32]
    plot_data = []
    for std in stds:
        raw_data_std = []
        plot_data_std = []
        for stage in stages:
            raw_data_task = []
            for task_number in tasks:
                for line in qps_log:
                    if "task_num:{},mean_v:32,std_v:{},stage:{}".format(task_number,std,stage) in line:
                        qps = int(eval(line.strip().split("QPS: ")[-1]))
                        raw_data_task.append(qps)

            raw_data_std.append(raw_data_task)
            plot_data_task = []
            for i in range(len(tasks)):
                plot_data_task.append(raw_data_task[i]/raw_data_std[0][i])
            plot_data_std.append(plot_data_task)                    
        plot_data.append(plot_data_std)

    plot_data = np.array(plot_data)
    total_width, n = 0.8, 4
    x = np.arange(3)
    width = total_width / n
    x = x - (total_width - width) / 2

    fig = plt.figure(figsize=(20,5))
    subplots = [141,142,143,144]

    for i, std in enumerate(stds):
        ax = fig.add_subplot(subplots[i])
        ax.bar(x, plot_data[i][0], width= width, label = "Fixed")
        ax.bar(x + width, plot_data[i][1], width= width, label = "beta-only")
        ax.bar(x + 2 * width, plot_data[i][2], width= width, label ="alpha-only")
        ax.bar(x + 3 * width, plot_data[i][3], width= width, label ="CB")

        ax.legend()
        ax.set_xticks(np.arange(3))
        ax.set_ylabel('QPS')
        ax.set_xticklabels(tasks)
        ax.set_xlabel('# of tasks (std = {})'.format(std))
        plt.ylim((0,2.0))

    print("reproduced_figures/fig_12.jpg")
    plt.savefig("reproduced_figures/fig_12.jpg",bbox_inches='tight')

def plot_adapter_results(cfg):
    model = cfg.model
    results_path = cfg.results_path
    adapter_log_file_path = os.path.join(results_path,"pets",model,"multi_stream_PETS.log")
    if not os.path.exists(adapter_log_file_path):
        print(adapter_log_file_path,"does not exist!")
        return

    # 读取日志文件
    with open(adapter_log_file_path, 'r') as f:
        lines = f.readlines()

    # 提取数据
    data = {}
    for line in lines:
        if 'QPS:' in line:
            parts = line.split(' QPS: ')[0].split(',')
            # 解析参数
            task_num = int(parts[1].split(':')[1])
            stream_num = int(parts[2].split(':')[1])
            seq_len = int(parts[3].split(':')[1])
            
            # 获取QPS值
            qps = float(line.split('QPS: ')[1])
            
            if seq_len not in data:
                data[seq_len] = {}
            data[seq_len][stream_num] = qps

    # 准备绘图数据
    seq_lens = sorted(data.keys())
    stream_nums = [1, 2, 4, 8]  # 固定stream_num顺序
    
    # 创建两个图表：原始数据和归一化数据
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 为每个序列长度绘制一条线
    markers = ["o","^","s","d","*","x","v","<",">"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    # 绘制原始QPS数据
    for i, seq_len in enumerate(seq_lens):
        x_values = []
        y_values = []
        for stream in stream_nums:
            if stream in data[seq_len]:
                x_values.append(stream)
                y_values.append(data[seq_len][stream])
        
        if x_values:  # 只有在有数据的情况下才绘制
            ax1.plot(x_values, y_values, 
                    marker=markers[i % len(markers)],
                    color=colors[i % len(colors)],
                    label=f'seq_len={seq_len}',
                    linewidth=2)
    
    # 设置原始数据图表属性
    ax1.set_xlabel('Number of Streams')
    ax1.set_ylabel('QPS (Queries Per Second)')
    ax1.set_title('Adapter Performance (Raw QPS)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='best')
    ax1.set_xticks(stream_nums)
    
    # 绘制归一化数据
    for i, seq_len in enumerate(seq_lens):
        x_values = []
        y_values = []
        # 获取stream_num=1的基准值
        base_value = data[seq_len].get(1, None)
        if base_value is None:
            continue  # 如果没有stream_num=1的数据，则跳过
            
        for stream in stream_nums:
            if stream in data[seq_len]:
                x_values.append(stream)
                # 归一化: 当前值/基准值
                y_values.append(data[seq_len][stream] / base_value)
        
        if x_values:  # 只有在有数据的情况下才绘制
            ax2.plot(x_values, y_values, 
                    marker=markers[i % len(markers)],
                    color=colors[i % len(colors)],
                    label=f'seq_len={seq_len}',
                    linewidth=2)
    
    # 设置归一化图表属性
    ax2.set_xlabel('Number of Streams')
    ax2.set_ylabel('Normalized QPS (relative to stream_num=1)')
    ax2.set_title('Adapter Performance (Normalized)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='best')
    ax2.set_xticks(stream_nums)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    output_path = os.path.join("reproduced_figures", "adapter_results.jpg")
    print(output_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # 单独保存归一化图表
    plt.figure(figsize=(10, 6))
    
    # 重新绘制归一化数据（单独图表）
    for i, seq_len in enumerate(seq_lens):
        x_values = []
        y_values = []
        # 获取stream_num=1的基准值
        base_value = data[seq_len].get(1, None)
        if base_value is None:
            continue  # 如果没有stream_num=1的数据，则跳过
            
        for stream in stream_nums:
            if stream in data[seq_len]:
                x_values.append(stream)
                # 归一化: 当前值/基准值
                y_values.append(data[seq_len][stream] / base_value)
        
        if x_values:  # 只有在有数据的情况下才绘制
            plt.plot(x_values, y_values, 
                    marker=markers[i % len(markers)],
                    color=colors[i % len(colors)],
                    label=f'seq_len={seq_len}',
                    linewidth=2)
    
    # 设置归一化单独图表属性
    plt.xlabel('Number of Streams')
    plt.ylabel('Normalized QPS (relative to stream_num=1)')
    plt.title('Adapter Performance (Normalized to stream_num=1)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.xticks(stream_nums)
    
    # 保存归一化图表
    norm_output_path = os.path.join("reproduced_figures", "adapter_results_normalized.jpg")
    print(norm_output_path)
    plt.savefig(norm_output_path, dpi=300, bbox_inches='tight')
    
    plt.close('all')

def plot(cfg):
    exp_name = cfg.exp_name
    if exp_name == "figure_7":
        plot_figure_7(cfg)
    elif exp_name == "figure_8":
        plot_figure_8(cfg)
    elif exp_name == "figure_11":
        plot_figure_11(cfg)
    elif exp_name == "figure_12":
        plot_figure_12(cfg)
    elif exp_name == "adapter":
        plot_adapter_results(cfg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default='bert_base',
        choices=['bert_base',"bert_large","distil_bert"]
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="/workspace/research/exp_results"
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        choices=['figure_7','figure_8','figure_11','figure_12','adapter']
    )

    cfg = parser.parse_args()
    plot(cfg)