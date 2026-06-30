# NullspaceOfLLM: Multi-Task RL with Dynamic NSP and Delta-Loss Replay

本文档总结 `recipe/continual_ns_grpo` 中实现的多任务强化学习算法设计。该算法面向“多个任务顺序训练同一个策略模型”的场景，核心目标不是单纯提升某一个任务的 reward，而是在持续学习过程中缓解典型的“跷跷板”问题：当模型适应新任务时，旧任务性能往往同步下滑；而如果过度保护旧任务，又会压制新任务的学习效率。

围绕这个目标，我们从持续学习角度建模，针对多任务学习的稳定性（stability）和可塑性（plasticity）设计了两大核心机制：

1. 动态零空间投影算法（Dynamic Null-Space Projection, NSP）；
2. 基于 Delta Loss 和相关性的 replay 策略（Delta-Loss Replay）；

方法的核心思想是：NSP 负责限制“哪些方向不能乱动”，而 Delta-Loss Replay 负责提醒模型“哪些旧任务样本已经开始被忘”以及“哪些旧任务样本对新任务有正向知识迁移”。二者结合后，可以同时对抗“更新冲突”和“历史退化”，提升正向迁移作用，从而更稳定地支撑多任务强化学习。

## 1. 华为揭榜难题第四期问题2.3——训练精度跷跷板问题：如何设计混合数据训练策略

### 1.1 背景与验收标准
大语言模型通常需要兼顾数学、代码、通用推理、创意生成等多方面能力，然而现有的混合数据训练策略通常为多阶段接续训练或直接混合数据训练，训练精度容易出现”跷跷板”的问题，难以保证多能力项均匀提升。

难题揭榜对应的**验收标准**为：
给出一套推荐的混合数据训练策略，基于Qwen 7B作为基座模型上进行多能力项后训练，在AIME2025、LiveCodeBench(2024.10~2025.05)和Arena Hard的数学、代码和通用等测试集上，保持AIME和LCB测试精度为SOTA的情况下， Arena Hard测试精度不低于SOTA精度的95%。

### 1.2 实验设计与结果
实验上，根据揭榜要求，我们选择Qwen2.5-7B基座模型（未经过instruction tuning和RL的模型）作为基础网络，对比的基准为DeepSeek-R1-Distill-Qwen-7B。在进行RL之前，我们进行了大规模SFT。SFT数据包括0.9M AM_SFT数据（筛选长度2k左右）、1m Nemontron-Cascade的SFT Math数据（筛选长度在2k-4k左右）、1m 混合数据(长度分布在4k-8k)。

RL训练中使用code数据和math数据分别从公开数据集中筛选得到，训练采用1k prompt+8k completion的训练长度配置。RL过程为Code RL+NSP+replay->trajectory distillation->Math RL+NSP+replay。实验结果如下所示：

| Setting                         | AIME 2025        | LiveCodeBench(2024.10~2025.05) | Arena Hard v0.1            |
|---------------------------------|:--------------:|:---:|:--------------------------:|
| R1-Distill-Qwen-7B (baseline)   | 30.42 / 56.67   | 25.81    | 14.2 (-1.2 / +1.3)  |
| After SFT                       | 25.69 / 50      | 19.65    | 31.2                |
| Code + replay                   | 25.28 / 53.33   | 29.03    | -                   |
| Math (train alone)              | 28.33 / 56.67   | 21.70    | **33.4**                |
| Code->Math, sequential training | 26.53 / 56.67   | 26.39    | -                   |
| Math->Code, sequential training | 22.50 / 56.67   | 26.39    | -                   |
| Code->Math, NSP+replay          | **30.83 / 66.67**  | **29.33**    | 28.1                |

**核心结论**

我们在AIME 25、LCB和Arena Hard v0.1的测试集上都达到了验收标准，且在AIME 25和LCB上相较于基线有明显提升。需要注意的是，虽然在Arena Hard v0.1上的表现略有下降，但仍然满足了不低于SOTA精度95%的要求。

## 2. 方法总览

`recipe/continual_ns_grpo` 保留了标准 GRPO/PPO 的 on-policy 强化学习主循环：

1. 当前任务采样 rollout。
2. 计算 reward。
3. 计算 advantage。
4. 更新 actor/critic。

在这个基础上，我们额外插入两条持续学习机制：

1. 动态 NSP：基于激活协方差构造旧任务敏感子空间，并把后续参数更新投影到其零空间中，减少对旧知识的直接破坏。
2. Delta-Loss Replay：从历史任务候选样本中动态识别“当前最受干扰”的簇，再把这些样本以额外 SFT loss 的形式注入 actor backward，强化模型对旧任务关键模式的保留。

可以把这套方法理解为：

1. NSP 负责限制新的task的gradient更新方向，保持任务学习的稳定性。
2. Replay 负责筛选“受干扰最大的样本”以及“对当前任务训练有正向迁移作用的历史样本”，抗遗忘的同时提升新任务学习的可塑性。

## 3. 核心算法1：动态零空间投影

### 3.1 核心思想

对任意被选中的线性层，设其参数为 $W$，anchor 表征矩阵为 $X$，新任务原始梯度为

$$
G = \frac{\partial \mathcal{L}_{\text{new}}}{\partial W}.
$$

NSP 的目标是让更新尽量不破坏 anchor 任务响应，即

$$
X(W + \Delta W)^\top \approx XW^\top \quad \Longrightarrow \quad X\Delta W^\top \approx 0.
$$

因此我们不直接使用 $G$，而是构造校正梯度 $\widetilde{G}$，使其尽可能满足

$$
X\widetilde{G}^\top \approx 0.
$$

若当前训练 Code、anchor 取自 Math，则约束可以理解为

$$
H_{\ell,b}^{\text{math}}\left(\Delta W_{\ell,b}^{\text{code}}\right)^\top \approx 0,
$$

即 Code 的更新尽量不要沿着会破坏 Math 表征的方向移动。

### 3.2 动态性的 7 个设计点

1. 选择性应用：只在选定层集合

$$
\mathcal{L}_{\text{NSP}} = \mathcal{L}_{\text{MLP}} \cup \mathcal{L}_{\text{Attn}}
$$

上施加约束，通常优先中后部 MLP 和最后若干层 Attention；未选中层保持原始梯度不变。研究发现[1, 2]LLM中的事实性知识通常存储于中间层的MLP中，而后续层的attention主要关联事实性知识并做出回答。

2. 周期性刷新：在刷新时刻 $t_r \in \{0, T, 2T, \dots\}$，从 anchor 数据中采样 $\mathcal{A}_{t_r}$，并对每个被选层收集输入表征

$$
H_\ell^{(t_r)} \in \mathbb{R}^{N_\ell \times d_\ell}.
$$

两个刷新点之间固定使用 $P^{(t_r)}$，下一次刷新再重估子空间。

3. 分段矫正：将高维特征按块切分为

$$
H_\ell^{(t_r)} = [H_{\ell,1}^{(t_r)}, \dots, H_{\ell,p_\ell}^{(t_r)}],
$$

并对每个 block 构造协方差与投影器

$$
C_{\ell,b}^{(t_r)} = \frac{1}{N_\ell}(H_{\ell,b}^{(t_r)})^\top H_{\ell,b}^{(t_r)}, \qquad P_{\ell,b}^{(t_r)} = U_{\ell,b}^{(t_r)}(U_{\ell,b}^{(t_r)})^\top.
$$

随后按块修正梯度：$\widetilde{G}_{\ell,b}^{(t)} = G_{\ell,b}^{(t)} P_{\ell,b}^{(t_r)}$。

4. Token 级自适应：不是所有 token 都同样重要，可给 token 赋权

$$
w_t \propto \exp(\gamma s_t), \qquad s_t \in \{\|h_t\|_2, |\langle h_t, g_t \rangle|\},
$$

并用加权协方差

$$
C = \frac{1}{\sum_t w_t} \sum_t w_t h_t h_t^\top
$$

优先保护关键 token 对应的表征方向。

5. 层级自适应：对每层计算重要性得分 $s_\ell$，例如 Fisher 或激活统计，只对 top-$K$ 或 top-ratio 的层启用 NSP：**

$$
\mathcal{L}_{\text{NSP}} = {Top}_K(\{s_\ell\}_{\ell=1}^{L}).
$$

6. 非均匀分段：对高重要性特征用更细粒度保护。若特征重要性为 $a_j$，则可只对高重要性维度投影：

$$
\widetilde{G} = [G_{\text{high}} P_{\text{high}}, \; G_{\text{low}}],
$$

或对高重要性维采用更小 block、低重要性维采用更大 block。

7. 软投影：在原始梯度与投影梯度之间插值，平衡稳定性和可塑性：

$$
\widetilde{G}_{\ell,b} = \alpha G_{\ell,b} + (1-\alpha) G_{\ell,b} P_{\ell,b}, \qquad \alpha \in [0,1].
$$

[1] Kevein Meng, et al, "Locating and Editing Factual Associations in GPT", NeurIPS 2022.

[2] Junfeng Fang, et al. "AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models", NeurIPS 2025.

### 3.3 在当前实现中的执行流程

当前实现由 trainer、worker 和 optimizer 三层协同完成：

1. Trainer 决定刷新时机，以及何时同步统计量和更新投影矩阵。
2. Worker 在选定的 attention/MLP 线性层上注册 hook，收集输入特征并累计协方差。
3. 多卡场景下对协方差做 all-reduce，得到全局 anchor 统计。
4. Optimizer 对各 block 做特征分解，构造 $P_{\ell,b}$，并在 `optimizer.step()` 中对梯度更新执行投影。



## 4. 核心算法2：基于 Delta Loss和相关性 的 Replay 策略

### 4.1 核心思想

仅仅回放历史样本还不够，因为并不是所有旧样本在当前时刻都同样重要。更有效的做法是优先回放“正在被当前任务显著干扰”或者“与当前任务高度相关”的那部分旧知识。为此，我们引入两个指标来度量干扰和迁移能力：

**(1) Delta Loss**：

$$
\Delta_i = \ell_i^{\text{current}} - \ell_i^{\text{baseline}}
$$

其中：

1. $\ell_i^{\text{baseline}}$ 表示某个历史簇对应 sentinel 样本在参考时刻的 SFT loss，是通过在新任务训练前对 sentinel 样本做一次 forward 评估得到的。
2. $\ell_i^{\text{current}}$ 表示当前训练时刻该 sentinel 的 SFT loss，是通过定期评估 sentinel 样本得到的。
3. **$\Delta_i$ 越大，说明该簇代表的历史知识受到的干扰越严重**。

因此，Delta Loss replay 的核心不是“平均回放过去”，而是“优先修复当前退化最明显的历史模式”。

**(2) 基于相关性的迁移**：

结合数据的prototype可以判断哪些sentinel样本与当前训练任务高度相关，从而可以召回以及强化当前任务的学习。具体做法是：

1. 对指定任务的当前 batch embedding 做 EMA，形成 prototype。
2. 计算 sentinel embedding 与 prototype 的余弦相似度。
3. 将标准化后的 Delta Loss 和余弦相似度线性组合成最终打分。

因此最终用于筛选replay样本的打分形式可写为：

$$
s_i = \alpha \cdot z(\Delta_i) + \beta \cdot z(\cos_i)
$$

其中 $z(\cdot)$ 表示 z-score 标准化，$\alpha$ 和 $\beta$ 控制“干扰强度”和“原型相关性”两类信号的权重。

$s_i$ 越大，说明该簇代表的历史模式既受干扰又与当前任务相关，因此越值得优先回放修复；越小则说明replay样本未被遗忘或与当前任务关系不大，优先级较低。

### 4.2 Replay 候选池如何构建

replay候选池对应的`jsonl`种子样本筛选策略：

1. 从**训练任务的生成轨迹**中筛选：对任一训练样本的rollouts，根据其reward优先选择正确率低但至少存在一个正确答案的样本，以保证replay数据的质量和多样性。

2. 每条候选样本会被整理成一个 SFT 形式的 `(prompt, target)` 对，并缓存为 replay example。这样做的原因是：当前 replay 不是再走一次完整 RL rollout，而是以额外监督损失的形式，更直接地约束模型保留旧任务输出能力。

注： 为简化本次验证程序，我们将提供已经预先筛选好的replay数据。对于code RL，replay数据来自SFT阶段的science & reasoning数据；对于math RL, replay数据来自reasoning & science & code数据，其中code数据为上述流程1筛选出来的样本。
 
### 4.3 从候选池到 sentinel pool

为了避免直接在所有历史样本上做全量评估，当前实现先把候选样本映射到隐藏态表示空间：

1. 用 actor 提取指定隐藏层的 mean-pooled embedding。
2. 对候选 embedding 做 k-means 聚类。
3. 每个簇选择一个最接近簇中心的样本作为 sentinel。

这样，sentinel 就变成了每个历史模式的代表点。后续只需要跟踪这些代表点的 loss 变化，就能近似判断哪些历史区域正在被遗忘。

### 4.4 Replay策略如何驱动样本选择

训练进入新任务后，系统会先对每个 sentinel 记录一份 baseline loss。之后每隔若干 step：

1. 重新计算 sentinel 当前的 SFT loss。
2. 用当前 loss 减去 baseline，得到每个簇的 Delta Loss；并计算cosine similarity，得到每个簇的相关性分数。
3. 按 $s_i$ 从高到低排序，优先选择受干扰最严重的簇，从每个簇中筛选出一定数量的样本，构成 replay batch。



### 4.5 Replay 如何注入训练

当系统选出 top-k 高分簇后，会从每个簇中再抽取若干条 replay 样本，构造一个 replay payload。这个 payload 不替换原有 RL 更新，而是附加到 actor 的最终 backward 中：

1. 当前任务仍然执行标准 GRPO/PPO 的策略梯度更新。
2. 在 actor update 的最后阶段，额外对 replay 样本计算 SFT loss。
3. 该损失乘上 `replay_loss_lambda` 后进行一次额外反向传播。

于是 actor 的总更新实际上同时包含两部分：

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{RL}} + \lambda_{\text{replay}} \mathcal{L}_{\text{replay}}
$$

其中 $\mathcal{L}_{\text{replay}}$ 只在被识别为“当前最值得修复”的历史样本上触发。


## 6. 算法参数解析

这一节按算法设计解释超参数含义。从方法设计上，它们分别控制 NSP 与 replay 的保护强度、更新频率、选择粒度和计算开销。

### 6.1 NSP 相关参数

下面给出的“默认值”以 `recipe/continual_ns_grpo/nsp_config.py` 中 `NullSpaceProjConfig` 的 dataclass 默认配置为准。需要注意，实际训练脚本通常会在此基础上覆盖部分参数；例如为了更强保护或更积极筛选重要 token / block，实验脚本里常会把 `soft_projection_alpha`、`token_importance_top_ratio`、`high_freq_block_ratio`、`anchor_max_examples` 等值调大。实验中使用的参数值在脚本中已给出。

1. `enable_nullspace_projection`：NSP 总开关，决定是否启用整套零空间约束，默认值为 `False`。默认关闭的原因是 NSP 会额外引入激活统计、特征分解和梯度投影开销；把它设计成显式开启，更适合避免在普通训练里误开高成本逻辑。

2. `NSP_mlp_layer_range`、`NSP_attention_layer_range`：对应动态点 1，用来选择哪些 MLP 层和 Attention 层进入 NSP。范围越大，保护覆盖越广，但新任务可更新空间越小。默认值分别为 `(7, 12)` 和 `(20, 23)`。这个默认选择偏向中后层：一方面避免对全部层施加约束导致可塑性明显下降，另一方面也和“中间层 MLP 与后部 attention 更承载任务特征”的经验观察一致。

3. `anchor_task_id`、`anchor_dataset_list`：对应动态点 2，决定 anchor 数据来自哪个历史任务，或直接来自哪些外部数据文件。前者适合固定参考任务，后者适合显式指定需要保护的数据分布。默认值分别为 `0` 和 `None`。这意味着默认优先以第 0 个任务作为 anchor，而不是强制依赖外部 jsonl；这样便于在多任务顺序训练中直接复用历史任务数据流。

4. `anchor_max_examples`、`anchor_batch_size`、`anchor_shuffle`：对应动态点 2，控制每次刷新时用多少 anchor 样本、每个 batch 多大、是否打乱顺序。默认值分别为 `512`、`4` 和 `True`。这组默认值相对保守：`512` 个样本已经足够提供一份可用的协方差估计，而 `batch_size=4` 更有利于控制显存；默认打乱则可以降低统计对局部样本顺序的偏置。

5. `projection_update_mode`：对应动态点 2，决定投影器是在 `periodic` 模式下按步数刷新，还是在 `task_end` 模式下按任务结束刷新。默认值为 `periodic`，采用周期刷新。

6. `update_projections_every`、`anchor_update_interval`：对应动态点 2，控制重建投影矩阵的节奏。默认值分别为 `4` 和 `100`。前者较小，表示在开启统计更新时会比较频繁地重算投影矩阵；后者较大，表示 anchor 数据本身不必每几步都重新采样。这样的组合本质上是在“投影更新灵敏度”和“anchor 重采样成本”之间做折中。

7. `cov_update_every`、`inner_steps_for_update_cov`：对应动态点 2，控制协方差统计的累积频率。默认值分别为 `1` 和 `16`。`cov_update_every=1` 表示一旦进入统计阶段就按步累积，以免错过表示变化；`inner_steps_for_update_cov=16` 则限制一次更新周期内的统计跨度，避免无限制累积导致统计过旧。

8. `do_update_statistics`、`should_update_NSP_first_step`、`reset_stats_on_task_start`：对应动态点 2，分别控制是否更新统计、首步是否强制刷新、任务切换时是否清空旧缓存。默认值都为 `True`。这一组默认值体现的是“任务边界优先保证统计有效性”而不是“尽量复用旧缓存”：新任务刚开始时往往最容易发生更新冲突，因此默认首步刷新并重置统计更稳妥。

9. `block_null_space_projection`、`max_feature_width_allow`：对应动态点 3，决定是否采用分段投影以及每个 block 的最大宽度。默认值分别为 `True` 和 `1536`。默认启用 block 版 NSP，是因为对大模型线性层做整块特征分解的代价太高；`1536` 则是一个兼顾分解稳定性和计算成本的块宽上限。

10. `svd_thres`、`num_eigen`：对应动态点 3，控制零空间基的构造强度。默认值分别为 `1e-3` 和 `100`。这个默认设置属于较常见的温和约束：阈值不会大到把太多方向都视为零空间，而 `100` 个最小特征值方向通常足够形成一组有效保护子空间，又不会把更新空间压得过窄。

11. `token_importance`、`token_importance_top_ratio`、`cov_update_batch_size`：对应动态点 4，控制是否做 token 级加权、保留多少高重要性 token，以及做这类统计时的批处理大小。默认值分别为 `True`、`0.3` 和 `0`。默认启用 token importance，是为了避免把大量低信息 token 等权写入协方差；`0.3` 表示只保留前 30% 高重要性 token 参与统计，偏向“聚焦关键 token”；`0` 则表示默认不再额外分 chunk，只有在显存紧张时再手动设置更小批量。

12. `layer_adaptivity`、`layer_selection_metric`、`layer_top_k`、`layer_top_ratio`：对应动态点 5，控制是否做层级筛选、按什么指标给层打分，以及最终保留多少层进入 NSP。默认值分别为 `True`、`"mean"`、`0` 和 `0.75`。保证 layer adaptivity 能力，但默认不用 `top-k`；而 `mean` 指标也比 Fisher 更便宜，不需要额外依赖 backward 统计。

13. `feature_activity_metric`、`high_freq_block_ratio`、`high_freq_min_blocks`：对应动态点 6，控制如何给特征或 block 打活跃度分数，以及只对多少高活跃 block 施加投影。默认值分别为 `"variance"`、`0.25` 和 `1`。默认用方差衡量活跃度，是因为它对“哪些维度变化更明显”更敏感；只保留前 25% block 进入 NSP，则是为了把计算集中到更可能承载历史知识的高活跃部分，同时保证至少保留 1 个 block，避免某层被完全跳过。

14. `soft_projection_alpha`：对应动态点 7，控制软投影强度，可理解为更新从 $gP$ 平滑过渡到 $g$。默认值为 `0.2`。这意味着 dataclass 默认行为更接近硬投影，优先强调旧任务稳定性；在实际实验里，如果发现新任务学习受抑制，常会把它调大到如 `0.2` 一类的值，以混合原始梯度和矫正梯度。

15. `svd_lr`、`bn_lr`：控制采用 NSP 时不同参数组的学习率。默认值都为 `1e-6`。这个默认值明显偏小，核心考虑是：一旦对梯度做了额外投影，再叠加较大学习率，训练容易不稳定；先用小步长保证受约束优化可控，再在具体任务里逐步放大更安全。

16. `nsp_log_details`、`nsp_log_level`：控制 NSP 的日志粒度，用于观察层选择、block 数量、统计刷新和投影构造过程。默认值分别为 `False` 和 `"basic"`。默认只保留基础日志，是为了减少训练时的 I/O 与控制台噪声；在调试阶段再切到 `detail` 或 `trace` 更合适。

17. `save_covariance_every`、`resume_fea_in_and_num_for_cov_update`、`path_for_fea_in`：控制协方差统计的保存与恢复。默认值分别为 `100`、`False` 和 `None`。也就是说，默认每 100 步落一次统计，但不主动从磁盘恢复已有统计；这是比较自然的“可恢复但不强依赖外部缓存”的默认形态，更适合常规从头训练。

### 6.2 Replay 相关参数

1. `enable_interference_recall`：Replay 总开关，决定是否启用历史样本回放。

2. `seed_jsonl_path`、`seed_jsonl_path_list`、`seed_jsonl_max_examples_per_file`：控制是否从外部 jsonl 预加载历史样本，以及每个文件最多读取多少条。这组参数决定 replay 是否只依赖已训练任务的数据，还是额外引入外部种子候选池。

3. `sentinel_selection_max_examples`、`sentinel_candidate_cap`、`sentinel_shuffle_candidates`：控制候选池规模和候选样本顺序。前两者分别限制单任务候选数和全局候选容量，后者决定聚类前是否打乱。候选池越大，覆盖越充分，但 embedding 编码和聚类成本也越高。推荐`sentinel_selection_max_examples`设置2000或者更大。

4. `sentinel_sample_count`、`sentinel_kmeans_iters`：控制用多少个 sentinel 簇代表历史模式，以及 k-means 做多少轮迭代。簇数越多，表示越细，但噪声更大；迭代越多，聚类更稳，但耗时更高。

5. `sentinel_hidden_state_batch_size`、`sentinel_sft_max_seq_length`：控制提取候选 embedding 和评估 sentinel loss 时的吞吐与长度上限。它们主要影响 replay 的显存占用、速度和可覆盖样本长度。

6. `selected_cluster_num_k`、`num_replayed_examples_per_cluster`：控制每轮从多少高分簇中取样，以及每个簇回放多少条样本。前者决定“修多少类历史知识”，后者决定“每类修多深”。

7. `sft_replay_batch_size`、`sft_target_key`：控制 replay backward 的微批大小和监督目标字段。前者影响反向传播吞吐与显存，后者决定用样本中的哪一项文本作为 SFT 监督标签。

8. `replay_loss_lambda`：控制 replay loss 在总目标中的权重，可理解为总损失从 $L_{RL}$ 向 $L_{RL} + \lambda L_{replay}$ 的偏移强度。它越大，旧任务保护越强，但越可能干扰当前任务收敛。推荐设置`0.1`。

9. `start_decay_step`、`sentinel_eval_interval`：控制 replay 强度和干扰评估的时间节奏。`start_decay_step` 决定何时开始减弱 replay，`sentinel_eval_interval` 决定多久重算一次 Delta Loss。前者更偏优化后期策略，后者更偏监控频率。

10. `score_alpha`、`score_beta`：控制最终打分中 Delta Loss 信号与 prototype 相似度信号的相对权重。`score_alpha` 更大时更偏抗遗忘，`score_beta` 更大时更偏向与当前任务语义相关的正迁移样本。默认都为`1.0`，即两者同等重要。

11. `prototype_hidden_layer_index`、`prototype_ema_decay`、`prototype_update_interval`、`prototype_task_ids`：控制 prototype 的构建方式。分别对应从哪一层取表征、EMA 平滑强度、多长时间更新一次、哪些任务参与构造 prototype。它们共同决定“相关性”信号是否稳定、是否贴近当前任务状态。


## 7. 代码位置

相关实现主要位于：

1. `recipe/continual_ns_grpo/main_continual_ns_grpo.py`
2. `recipe/continual_ns_grpo/continual_ray_trainer.py`
3. `recipe/continual_ns_grpo/nsp_fsdp_worker.py`
4. `recipe/continual_ns_grpo/ns_adamw.py`
5. `recipe/continual_ns_grpo/config/continual_ns_grpo_trainer.yaml`

## 8.程序运行指南

1. 下载SFT模型和回放数据集；
   - 筛选后的数学RL数据: `sieved_DAPO_13khard_8k_AIME.parquet`  
    [Download Link](https://drive.google.com/file/d/1jgQNLk7t2Axcg7Q2859ysDMWJtRl9N6k/view?usp=sharing)
   - 筛选后的Math RL数据: `code_train.parquet`  
 [Download Link](https://drive.google.com/file/d/123T5CRBQ5pRdW3rcNjI6e9WYryY_dPou/view?usp=sharing)
    - AIME 25评估数据: `math_eval.parquet`  
 [Download Link](https://drive.google.com/file/d/1D4t9tK736ff3mR6oVV8rozHzJsyl7fSY/view?usp=sharing)
    - Code RL的replay数据: `science_reasoning_replay_data.jsonl`  
 [Download Link](https://drive.google.com/file/d/1-UdUlXCCrpP-JBCY6DS9oGega5D4-QrN/view?usp=sharing)
    - Math RL的replay数据: `science_reasoning_code_replay_data.jsonl`  
 [Download Link](https://drive.google.com/file/d/1kj1euMG2yzh_YPYM_PAK0vHeeQobrFck/view?usp=sharing)
    - SFT模型：`randylo/SFT_model_qwen2.5_7B`  
  [model link](https://www.modelscope.cn/models/randylo/SFT_model_qwen2.5_7B)
1. 修改脚本`recipe/continual_ns_grpo/run_ns_grpo_code.sh`中的数据路径并运行脚本训练代码任务；
2. 修改脚本`recipe/continual_ns_grpo/run_ns_grpo_math.sh`中的数据路径并运行脚本训练数学任务；