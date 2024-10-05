import numpy as np
import random
import matplotlib.pyplot as plt

# 绘制甘特图
def plot_gantt_chart(schedule, processing_data):
    num_jobs = processing_data.shape[0]
    num_machines = processing_data.shape[1] // 2
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab20.colors  # 颜色映射
    for job_id, job_schedule in enumerate(schedule):
        for op_id, (start_time, machine) in enumerate(job_schedule):
            duration = processing_data[job_id, 2 * op_id + 1]
            ax.barh(y=machine, width=duration, left=start_time, height=0.8,
                    color=colors[job_id % len(colors)], edgecolor='black')
            ax.text(start_time + duration / 2, machine, f'Job {job_id + 1}-{op_id + 1}',
                    va='center', ha='center', color='white', fontsize=8)

    ax.set_xlabel('Time')
    ax.set_ylabel('Machine')
    ax.set_title('Gantt Chart')
    ax.set_yticks(range(num_machines))
    ax.set_yticklabels([f'Machine {i}' for i in range(num_machines)])
    ax.invert_yaxis()  # 机器编号从上到下
    plt.tight_layout()
    plt.show()

# 调度工件
def schedule_jobs(individual, processing_data):
    num_jobs = processing_data.shape[0]
    num_machines = processing_data.shape[1] // 2
    num_operations = num_jobs * num_machines

    machine_available_time = [0] * num_machines
    job_next_operation_idx = [0] * num_jobs
    job_available_time = [0] * num_jobs
    job_schedule = [[] for _ in range(num_jobs)]

    for operation_job_id in individual:
        op_idx = job_next_operation_idx[operation_job_id]
        if op_idx >= num_machines:
            continue  # 该作业的所有工序已调度

        machine = int(processing_data[operation_job_id, 2 * op_idx])
        duration = processing_data[operation_job_id, 2 * op_idx + 1]
        start_time = max(machine_available_time[machine], job_available_time[operation_job_id])

        job_schedule[operation_job_id].append((start_time, machine))
        machine_available_time[machine] = start_time + duration
        job_available_time[operation_job_id] = start_time + duration

        job_next_operation_idx[operation_job_id] += 1

    makespan = max(job_available_time)
    return job_schedule, makespan

# 遗传算法
def genetic_algorithm(processing_data, population_size=100, generations=2000, crossover_rate=0.8, mutation_rate=0.3):
    num_jobs = processing_data.shape[0]
    num_machines = processing_data.shape[1] // 2
    num_operations = num_jobs * num_machines

    # 初始化种群
    operations = []
    for job_id in range(num_jobs):
        operations.extend([job_id] * num_machines)

    population = [random.sample(operations, len(operations)) for _ in range(population_size)]

    best_makespan_history = []
    global_best_individual = None
    global_best_makespan = float('inf')

    # 初始化动态绘图
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel("Generation")
    ax.set_ylabel("Makespan")
    ax.set_title("Best Makespan Over Generations")
    line, = ax.plot([], [], label="Best Makespan")
    ax.legend()

    for gen in range(generations):
        # 计算适应度
        fitness = []
        for individual in population:
            _, makespan = schedule_jobs(individual, processing_data)
            fitness.append(makespan)

        # 找出这一代的最佳个体
        best_makespan_gen = min(fitness)
        best_individual_gen = population[fitness.index(best_makespan_gen)]
        best_makespan_history.append(best_makespan_gen)

        # 更新全局最优解
        if best_makespan_gen < global_best_makespan:
            global_best_makespan = best_makespan_gen
            global_best_individual = best_individual_gen

        # 选择（锦标赛选择）
        tournament_size = 3  # 锦标赛大小，可以根据需要调整
        selected = []
        for _ in range(population_size):
            # 随机选择锦标赛参与者
            participants = random.sample(list(zip(population, fitness)), tournament_size)
            # 选择适应度最好的个体作为胜者
            winner = min(participants, key=lambda x: x[1])  # 适应度（完工时间）越小越好
            selected.append(winner[0])

        # 交叉
        new_population = []
        for i in range(0, population_size, 2):
            parent1 = selected[i]
            if i + 1 < population_size:
                parent2 = selected[i + 1]
            else:
                parent2 = selected[0]
            if random.random() < crossover_rate:
                # 均匀交叉
                child1, child2 = [], []
                counts1 = {job_id: 0 for job_id in range(num_jobs)}
                counts2 = {job_id: 0 for job_id in range(num_jobs)}
                for pos in range(len(parent1)):
                    gene1 = parent1[pos] if random.random() < 0.5 else parent2[pos]
                    gene2 = parent2[pos] if random.random() < 0.5 else parent1[pos]

                    if counts1[gene1] < num_machines:
                        child1.append(gene1)
                        counts1[gene1] += 1
                    else:
                        # 填充剩余的作业
                        for job_id in counts1:
                            if counts1[job_id] < num_machines:
                                child1.append(job_id)
                                counts1[job_id] += 1
                                break

                    if counts2[gene2] < num_machines:
                        child2.append(gene2)
                        counts2[gene2] += 1
                    else:
                        for job_id in counts2:
                            if counts2[job_id] < num_machines:
                                child2.append(job_id)
                                counts2[job_id] += 1
                                break
                new_population.extend([child1, child2])
            else:
                new_population.extend([parent1, parent2])

        # 变异
        for i in range(len(new_population)):
            if random.random() < mutation_rate:
                idx1, idx2 = random.sample(range(num_operations), 2)
                new_population[i][idx1], new_population[i][idx2] = new_population[i][idx2], new_population[i][idx1]

        # 更新种群
        population = new_population

        # 每 1000 代绘制一次曲线
        if (gen + 1) % 100 == 0:
            print(f"Generation {gen + 1}: Best makespan = {global_best_makespan}")
            ax.clear()
            ax.set_xlabel("Generation")
            ax.set_ylabel("Makespan")
            ax.set_title("Best Makespan Over Generations")
            ax.plot(range(len(best_makespan_history)), best_makespan_history, label="Best Makespan")
            ax.legend()
            plt.draw()
            plt.pause(0.1)

    plt.ioff()  # 关闭动态模式

    # 最后输出全局最优解并绘制甘特图
    print(f"Global best makespan: {global_best_makespan}")
    print(f"Best individual: {global_best_individual}")
    best_schedule, _ = schedule_jobs(global_best_individual, processing_data)
    plot_gantt_chart(best_schedule, processing_data)

    return global_best_individual, global_best_makespan

# 运行遗传算法示例
if __name__ == "__main__":
    processing_data = np.array([
        [2, 1, 0, 3, 1, 6, 3, 7, 5, 3, 4, 6],
        [1, 8, 2, 5, 4, 10, 5, 10, 0, 10, 3, 4],
        [2, 5, 3, 4, 5, 8, 0, 9, 1, 1, 4, 7],
        [1, 5, 0, 5, 2, 5, 3, 3, 4, 8, 5, 9],
        [2, 9, 1, 3, 4, 5, 5, 4, 0, 3, 3, 1],
        [1, 3, 3, 3, 5, 9, 0, 10, 4, 4, 2, 1]
    ])

    best_individual, best_makespan = genetic_algorithm(processing_data)
