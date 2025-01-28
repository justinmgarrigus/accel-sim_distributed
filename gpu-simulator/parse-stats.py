import matplotlib.pyplot as plt 
import os, sys

# r = how many cycles to average. r=1 means we report all stats on each cycle, 
# while r=1000 means we average every 1000 cycles together. 
r = 1000

fig, axs = plt.subplots(4) 
fig.set_figwidth(16)
fig.set_figheight(9)
axs[0].set_title(f'BERT Utilization per {r} Cycles') 

f = open('warp_occupancy_stats.dat', 'r') 
warp_data_cycles = [float(part) for part in f.read().split(',') if len(part) > 0]
data = [
    sum(warp_data_cycles[i*r : (i+1)*r]) / r 
    for i in range(len(warp_data_cycles) // r)
] 
xaxis = [i for i in range(len(data))] 
f.close()
axs[0].set_ylabel('Warp utilization\n(percent)') 
axs[0].plot(xaxis, data)
print(f'Warp utilization: {len(data)} points') 

f = open('sm_occupancy_stats.dat', 'r') 
sm_data_cycles = [float(part) for part in f.read().split(',') if len(part) > 0] 
data = [
    sum(sm_data_cycles[i*r : (i+1)*r]) / r 
    for i in range(len(sm_data_cycles) // r)
] 
data = data[:len(xaxis)] 
f.close()
axs[1].set_ylabel('SM utilization\n(percent)') 
axs[1].set_ylim(-0.1, 1.1)
axs[1].plot(xaxis, data)
print(f'SM utilization: {len(data)} points') 

f = open('tensor_cycle_stats.dat', 'r') 
tensor_data_cycles = [int(part) for part in f.read().split(',') if len(part) > 0] 
yvals = [0] * len(xaxis) * r
for d in tensor_data_cycles: 
    if d < len(yvals): 
        yvals[d] += 1
tensor_data_cycles = yvals 
yvals = [
    sum(yvals[i*r : (i+1)*r]) // r 
    for i in range(len(yvals) // r)
] 
f.close()
axs[2].set_ylabel('Tensor core utilization\n(instructions issued)') 
axs[2].plot(xaxis, yvals)  
print(f'Tensor core utilization: {len(yvals)} points') 
tensor_ylim = axs[2].get_ylim() 

f = open('dram_access_stats.dat', 'r') 
data = [int(part) for part in f.read().split(',') if len(part) > 0] 
dram_data_cycles = [data[i+1] - data[i] for i in range(len(data)-1)] + [0]
data = [
    sum(dram_data_cycles[i*r : (i+1)*r]) // r 
    for i in range(len(dram_data_cycles) // r)
] 
dram_data = data[:len(xaxis)]
f.close() 
axs[3].set_ylabel('DRAM utilization\n(accesses)') 
axs[3].set_xlabel(f'{r} cycles') 
axs[3].plot(xaxis, dram_data)
print(f'DRAM utilization: {len(dram_data)} points') 
dram_ylim = axs[3].get_ylim() 

f = open('kernel_time_stats.dat', 'r') 
kernel_data = [line.strip().split(',') for line in f.readlines() if len(line) > 0] 

f.close() 
for idx, item in enumerate(kernel_data): 
    kernel_name = item[0].lower()  
    if 'gemm' in kernel_name: kernel_name = 'gemm'
    elif 'elementwise' in kernel_name: kernel_name = 'elementwise'
    elif 'softmax' in kernel_name: kernel_name = 'softmax'
    elif 'dropout' in kernel_name: kernel_name = 'dropout' 
    elif 'layer_norm' in kernel_name: kernel_name = 'layer_norm'
    kernel_data[idx][0] = kernel_name 

    start_cycle = int(item[1]) // r 
    end_cycle = int(item[2]) // r
    axs[0].text(
        start_cycle + (end_cycle - start_cycle) // 2,
        0.5, 
        kernel_name,
        rotation='vertical',
        verticalalignment='center',
        horizontalalignment='center'
    )

    color = (0, 1, 0, 0.2) if kernel_name == 'gemm' else (1, 0, 0, 0.2) 
    axs[0].fill_between([start_cycle, end_cycle], 0, 1, color=color)
    axs[1].fill_between([start_cycle, end_cycle], 0, 1, color=color)
    axs[2].fill_between([start_cycle, end_cycle], 0, max(yvals), color=color)
    axs[3].fill_between([start_cycle, end_cycle], 0, max(dram_data), 
        color=color)

fig.savefig(f'per-{r}.png')
print(f'Saved figure "per-{r}.png".') 

# Per-kernel averages 
fig, axs = plt.subplots(4) 
fig.set_figwidth(16)
fig.set_figheight(9)
axs[0].set_title(f'BERT Utilization per layer') 
n_cycles = len(warp_data_cycles) 
xaxis = [i for i in range(n_cycles)] 
warp_yaxis = [0] * n_cycles 
sm_yaxis = [0] * n_cycles 
tensor_yaxis = [0] * n_cycles 
dram_yaxis = [0] * n_cycles
for item in kernel_data:  
    start_cycle = int(item[1])
    end_cycle = int(item[2])
    duration = end_cycle - start_cycle 
    
    warp_average = sum(warp_data_cycles[start_cycle:end_cycle+1]) / \
        sum(1 if p > 0 else 0 for p in warp_data_cycles[start_cycle:end_cycle+1])
    sm_average = sum(sm_data_cycles[start_cycle:end_cycle+1]) / duration 
    tensor_average = sum(tensor_data_cycles[start_cycle:end_cycle+1]) / duration 
    dram_average = sum(dram_data_cycles[start_cycle:end_cycle+1]) / duration 
    for i in range(start_cycle, end_cycle): 
        warp_yaxis[i] = warp_average
        sm_yaxis[i] = sm_average
        tensor_yaxis[i] = tensor_average
        dram_yaxis[i] = dram_average
    
    kernel_name = item[0] 
    axs[0].text(
        start_cycle + (end_cycle - start_cycle) // 2,
        0.5, 
        kernel_name,
        rotation='vertical',
        verticalalignment='center',
        horizontalalignment='center'
    )
        
    color = (0, 1, 0, 0.2) if kernel_name == 'gemm' else (1, 0, 0, 0.2) 
    axs[0].fill_between([start_cycle, end_cycle], 0, 1, color=color)
    axs[1].fill_between([start_cycle, end_cycle], 0, 1, color=color)
    axs[2].fill_between([start_cycle, end_cycle], 0, max(yvals), color=color)
    axs[3].fill_between([start_cycle, end_cycle], 0, max(dram_data), 
        color=color)
axs[0].set_ylabel('Warp utilization\n(percent)') 
axs[0].plot(xaxis, warp_yaxis) 
axs[1].set_ylabel('SM utilization\n(percent)') 
axs[1].plot(xaxis, sm_yaxis) 
axs[2].set_ylabel('Tensor core utilization\n(instructions issued)') 
axs[2].set_ylim(tensor_ylim[0], tensor_ylim[1]) 
axs[2].plot(xaxis, tensor_yaxis) 
axs[3].set_ylabel('DRAM utilization\n(accesses)')
axs[3].set_ylim(dram_ylim[0], dram_ylim[1])
axs[3].set_xlabel(f'{r} cycles') 
axs[3].plot(xaxis, dram_yaxis) 
fig.savefig('per-layer.png')
print('Saved figure "per-layer.png".') 
