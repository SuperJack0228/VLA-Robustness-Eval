# MiniVLA 鲁棒性评估项目总览

更新时间：2026-08-07

## 1. 一句话结论

项目已经完成一个在 MuJoCo/robosuite 中运行的轻量级多任务 VLA 闭环系统。冻结的 MiniVLA V3 在六任务、240 个严格配对 Clean 回合中达到 97.92% 成功率，并在动态目标平移测试中把成功率点估计保持在 80% 的位移边界从 V2 的 2 cm 推进到 7 cm。ACT chunk size = 1/5/10/20 的两种子、1,920 回合消融评估已经完成；结果显示 4 cm 扰动下存在显著的整体 chunk effect，但性能呈非单调关系，经典 temporal 模式下 chunk 10 最优。视觉噪声、物理漂移、OOD 干扰物和相机外参测试尚未形成正式曲线。

## 2. 项目边界

- 已完成：MuJoCo/robosuite 仿真、多任务专家数据、语言条件策略、ACT 闭环执行、配对 Clean 测试、动态瞬移鲁棒性曲线、Failure Taxonomy。
- 未完成：真实机器人、sim-to-real、安全认证，以及覆盖所有扰动类型的综合鲁棒性结论。
- 正确表述：轻量级 MiniVLA 策略系统，不是从零训练的通用基础大模型。
- 当前所有成功率均为 MuJoCo 仿真结果。

## 3. 环境与任务

### 3.1 仿真配置

- Python 3.10。
- 官方 `mujoco` Python 包，不使用 `mujoco-py`。
- robosuite + Panda 机械臂。
- 控制器：`OSC_POSE`。
- 控制动作：7D 增量动作，而不是绝对世界坐标。
- 双视角：`agentview` 和 `robot0_eye_in_hand`。
- 图像分辨率：112 x 112 RGB。
- 单回合最大长度：200 步。

### 3.2 场景物体

| ID | 物体 |
|---|---|
| A | red cube |
| B | blue ball |
| C | green cylinder |

### 3.3 六个语言任务

- `pick_A`：抓取红色方块。
- `pick_B`：抓取蓝色球。
- `pick_C`：抓取绿色圆柱。
- `push_A`：推开红色方块。
- `push_B`：推开蓝色球。
- `push_C`：推开绿色圆柱。

## 4. 数据契约

每条 episode 在磁盘中固定填充到 200 步，并通过 `valid_mask` 与 `trajectory_length` 标记有效区间。

### 4.1 主要张量

| 字段 | 形状 | 类型 | 含义 |
|---|---:|---|---|
| `image_agentview` | `(200,112,112,3)` | `uint8` | 全局相机 |
| `image_wrist` | `(200,112,112,3)` | `uint8` | 腕部相机 |
| `state` | `(200,17)` | `float32` | 机器人状态 |
| `previous_action` | `(200,7)` | `float32` | 上一步动作 |
| `action` | `(200,7)` | `float32` | 专家动作 |
| `object_pose` | `(200,3,7)` | `float32` | 三物体标注 |
| `object_contact` | `(200,3)` | `uint8` | 接触标注 |
| `object_grasped` | `(200,3)` | `uint8` | 抓取标注 |
| `expert_phase` | `(200,)` | `uint8` | 专家状态机阶段 |

### 4.2 17D 状态

| 维度 | 内容 |
|---:|---|
| 0:3 | 末端 XYZ |
| 3:7 | 末端四元数 XYZW |
| 7:9 | 夹爪 qpos |
| 9:11 | 夹爪 qvel |
| 11:14 | 末端线速度 |
| 14:17 | 末端角速度 |

推理输入使用连续 5 步状态历史，形状为 `(5,17)`。

### 4.3 7D 动作

| 维度 | 内容 |
|---:|---|
| 0:3 | Delta XYZ |
| 3:6 | Delta orientation / RPY rotation-error vector |
| 6 | Gripper |

动作输入范围被约束到 `[-1,1]`，执行前根据训练统计量反归一化连续位姿维度。

## 5. 数据集

### 5.1 V2 Clean

- 总计：1,200 episodes。
- 六任务完全平衡：每任务 200。
- 每任务：160 train、20 validation、20 test。
- 总划分：960 train、120 validation、120 test。
- 本地体积：约 5.2 GB。
- 数据门禁：Preflight PASS。

### 5.2 V3 Recovery

- 总计：600 episodes。
- 六任务完全平衡：每任务 100。
- 每任务：80 train、10 validation、10 test。
- 总划分：480 train、60 validation、60 test。
- 本地体积：约 2.6 GB。
- 内容：目标位移后的重新定位、接触恢复、重新抓取和继续推送轨迹。
- 数据门禁：Preflight PASS。

### 5.3 合并训练规模

- 原始轨迹：1,800 episodes。
- 实际训练 episodes：1,440。
- 训练滑窗：92,160。
- Combined validation 滑窗：8,640。
- Clean validation 滑窗：5,760。
- DataLoader 每个 batch 至少混合 16 条 episode；门禁首批检测到 30 条不同语言指令。
- 数据总大小：约 7.8 GB。

## 6. 语言数据

- 语言编码器：冻结的 `distilbert-base-uncased`。
- 六任务每任务 60 条训练表达、30 条留出评估表达。
- 全部训练表达：360。
- 全部留出表达：180。
- 总语言表达：540。
- 语言目录版本：`v3.language.1`。
- DistilBERT 不更新梯度。
- 180 条留出表达的独立闭环语义泛化基准尚未完成，因此目前不能宣称开放词汇语言泛化已经被充分验证。

## 7. MiniVLA V3 架构

### 7.1 模态输入

- 语言：DistilBERT tokens。
- 视觉：共享视觉骨干处理 agentview 与 wrist 图像。
- 状态：5 x 17 状态历史。
- 当前控制上下文与上一动作也进入融合流程。

### 7.2 核心配置

| 配置 | 数值 |
|---|---:|
| Hidden dimension | 512 |
| State history | 5 |
| Encoder layers | 2 |
| Decoder layers | 3 |
| Attention heads | 8 |
| Dropout | 0.15 |
| Final V3 chunk size | 20 |
| Pose output | 6D continuous |
| Gripper output | binary probability |
| Architecture version | 4 |
| Checkpoint format | 8 |

### 7.3 实测参数量

- 总参数：101,784,921，约 1.018 亿。
- 可训练参数：33,977,113，约 3,398 万。
- 冻结语言模型参数：66,362,880。
- 视觉模块参数：9,094,208，其中可训练 7,649,280。
- Multimodal Encoder 参数：6,305,792。
- V3 policy checkpoint：约 135 MB。

### 7.4 输出头

- 连续 6D 位姿动作头。
- 二元夹爪头。
- Target grounding 辅助头。
- Target class 辅助头。
- Expert phase / phase family 辅助头。
- Contact / grasp interaction 辅助头。

## 8. V3 训练

- HPC job：`1349625`。
- GPU：单张 NVIDIA L40S，不是 H100。
- Epochs：30。
- Batch size：32。
- Workers：8。
- 冻结 V2 Clean policy 全权重 warm-start。
- 最佳 selection epoch：22。
- 最终 train total：0.16972。
- 最终 combined validation total：0.24732。
- Combined validation normalized XYZ MAE：0.01890。
- Gripper accuracy：99.23%。
- Phase accuracy：97.11%。
- Grounding error：0.785 cm。
- Target selection accuracy：100%。
- 训练后 6 回合 Postflight：6/6 成功，仅作为运行门禁，不作为正式统计。

## 9. Clean 配对评估

### 9.1 总结果

| 模型 | 成功 | 成功率 |
|---|---:|---:|
| V2 Clean | 222/240 | 92.50% |
| V3 | 235/240 | 97.92% |

- V3 绝对提升：+5.42 percentage points。
- V3-only successes：17。
- V2-only successes：4。
- Exact paired McNemar：`p=0.0072`。
- V2 mean steps：94.35；V3：86.57。
- Grounding error：0.865 cm -> 0.725 cm。
- Safety intervention：5.19% -> 4.20%。
- Action clipping：6.89% -> 7.76%，这是一个需要如实报告的代价。

### 9.2 六任务 Clean 成功率

| 任务 | V2 | V3 |
|---|---:|---:|
| pick_A | 90.0% | 100.0% |
| pick_B | 100.0% | 100.0% |
| pick_C | 97.5% | 100.0% |
| push_A | 82.5% | 100.0% |
| push_B | 95.0% | 92.5% |
| push_C | 90.0% | 95.0% |

V3 总体显著提升，但 Clean `push_B` 比 V2 低 2.5 个百分点，不能隐藏这一局部回退。

## 10. 动态目标瞬移鲁棒性

### 10.1 协议

- 位移尺度：0/1/2/3/4/5/6/7/8 cm。
- 两套评估种子。
- 每尺度每模型 120 episodes。
- 同场景、同任务、同目标、同语言、同瞬移方向严格配对。
- Temporal 和 latest-only 两种执行模式。
- 原始 episode rows：4,320。
- V2/V3 paired model outcomes：2,160。
- 主分析使用 Temporal。
- 所有非零 Temporal 回合完成了请求的瞬移。
- 策略执行不使用模拟器特权状态；特权状态只用于扰动注入和评估计分。

### 10.2 主衰减曲线

| 位移 | V2 | V3 | V3增益 |
|---:|---:|---:|---:|
| 0 cm | 90.8% | 97.5% | +6.7 pp |
| 1 cm | 90.8% | 98.3% | +7.5 pp |
| 2 cm | 86.7% | 97.5% | +10.8 pp |
| 3 cm | 57.5% | 94.2% | +36.7 pp |
| 4 cm | 40.0% | 90.8% | +50.8 pp |
| 5 cm | 27.5% | 87.5% | +60.0 pp |
| 6 cm | 21.7% | 83.3% | +61.7 pp |
| 7 cm | 20.0% | 80.0% | +60.0 pp |
| 8 cm | 16.7% | 75.0% | +58.3 pp |

- V2 robustness AUC：0.4974。
- V3 robustness AUC：0.8974。
- AUC 绝对提升：0.4000。
- 80% 成功率点估计边界：V2 2 cm，V3 7 cm。
- 95% Wilson 下界仍不低于 80% 的保守边界：V2 1 cm，V3 5 cm。
- 所有尺度上的 V3/V2 配对差异经 Holm 校正后仍显著。

### 10.3 视觉恢复证据

4 cm：

- V2 target contact 98.3%，0.5 cm reacquisition 79.2%，最终成功 40.0%。
- V3 target contact 99.2%，0.5 cm reacquisition 90.8%，最终成功 90.8%。
- V2 成功重定位延迟 0.86 s；V3 0.69 s。

8 cm：

- V2 target contact 72.5%，0.5 cm reacquisition 55.0%，最终成功 16.7%。
- V3 target contact 95.8%，0.5 cm reacquisition 76.7%，最终成功 75.0%。
- V2 成功重定位延迟 2.09 s；V3 1.35 s。

V2 在 3-4 cm 时经常仍能重新看到并接触目标，但无法稳定完成抓取或推送，因此其崩溃不只是视觉定位问题，也包括接触后的动作恢复问题。

## 11. Failure Taxonomy

在全部九个 Temporal 尺度上汇总：

| 失败类型 | V2 | V3 |
|---|---:|---:|
| grasp_failed_after_contact | 173 | 1 |
| insufficient_push_distance | 160 | 52 |
| wrong_object_contact | 75 | 4 |
| missed_grasp | 49 | 0 |
| target_not_contacted | 28 | 12 |
| target_toppled_before_grasp | 17 | 0 |
| recovery_limit | 10 | 28 |
| object_dropped | 9 | 0 |
| object_launched | 6 | 0 |
| workspace_stall | 6 | 0 |
| gripper_never_closed | 2 | 10 |
| insufficient_lift | 2 | 6 |

结论：V3 基本消除了“接触后抓取崩溃、抓空、碰错目标、抓起后掉落”等 V2 主导问题；剩余主要瓶颈转为大位移情况下的推送距离不足和恢复次数耗尽。

## 12. ACT chunk-size 消融

### 12.1 模型状态

| Chunk | 训练状态 | 使用权重 |
|---:|---|---|
| 1 | HPC完成，SHA256通过 | job 1526873 task 0 |
| 5 | HPC完成，SHA256通过 | job 1526873 task 1 |
| 10 | HPC完成，SHA256通过 | job 1526873 task 2 |
| 20 | 使用冻结V3参考模型 | job 1349625 |

四个权重都已在 Mac 上完成真实 MuJoCo 加载与前向推理冒烟。

### 12.2 评估协议

- Clean 与 4 cm 瞬移。
- `legacy temporal`：测试经典 ACT 历史预测重叠带来的惯性。
- `latest-only`：去除重叠集成，作为训练预测跨度对照。
- 每种子每条件 60 回合。
- 两套种子，每条件 120 回合。
- 总计：1,920 episodes；协议审计通过，4 cm 瞬移注入合规率 100%。

### 12.3 最终结果

| Chunk | Temporal Clean | Temporal 4 cm | Latest Clean | Latest 4 cm |
|---:|---:|---:|---:|---:|
| 1 | 96.67% | 85.00% | 96.67% | 85.00% |
| 5 | 97.50% | 89.17% | 95.83% | 88.33% |
| 10 | 98.33% | **95.00%** | **99.17%** | 92.50% |
| 20 | **99.17%** | 90.83% | 97.50% | **94.17%** |

- Clean：temporal Cochran Q `p=0.2898`，latest-only `p=0.1718`，没有显著整体 chunk effect。
- 4 cm：temporal `p=0.0112`，latest-only `p=0.004511`，存在显著整体 chunk effect。
- “chunk 越短，动态鲁棒性越高”的简单假设不成立；chunk 1 在 4 cm 下最弱。
- 经典 temporal 下 chunk 10 的 Clean/4 cm 平衡最好，4 cm 成功率 95.00%。
- latest-only 下 chunk 20 最好，4 cm 成功率 94.17%。
- chunk 20 从 latest-only 的 94.17% 降至 temporal 的 90.83%，方向上符合历史惯性，但配对差异不显著（McNemar `p=0.388`）。
- 4 cm temporal 的主要困难仍是 `push_B`；各 chunk 成功率依次为 75%、70%、85%、85%。

完整报告、统计表与 10 组 PNG/PDF 图：

`/Users/superjack/VLA-Robustness-Eval/final_report/03_chunk_size_ablation`

### 12.4 消融解释边界

- chunk 1/5/10 的 Action Queries 为重新初始化。
- 冻结 chunk 20 V3 从 V2 完整 warm-start，包括 Action Queries。
- 因此这是一项实用系统对比，而不是初始化方式绝对相同的纯控制变量实验。
- 正式图中应标记：`Chunk 20: frozen V3 reference, full warm-start`。

## 13. 已完成图表

正式 V2/V3 动态对比目录：

`/Users/superjack/VLA-Robustness-Eval/final_report/02_dynamic_displacement`

包含：

1. V2/V3 Temporal 成功率衰减。
2. Temporal/latest-only 四曲线对比。
3. V3 配对增益与显著性。
4. 归一化成功率保持率。
5. Pick/Push 任务族对比。
6. 六任务增益热图。
7. Failure Taxonomy。
8. Reacquisition rate/latency。
9. Recovery cost/wrong contact。

Clean 与训练汇报图：

`/Users/superjack/VLA-Robustness-Eval/final_report/01_clean_baseline and final_report/04_training_diagnostics`

## 14. 关键文件

### 14.1 模型和数据

- `models/mini_vla_v2.py`：当前 V2/V3 共用模型架构。
- `utils/v2_schema.py`：数据契约。
- `utils/training_dataset_v2.py`：多任务、语言增强、滑窗训练数据集。
- `scripts/collect_data_v2.py`：V2 Clean 专家采集。
- `scripts/collect_data_v3.py`：V3 recovery 采集。
- `scripts/train_v3.py`：V3 训练。

### 14.2 评估与鲁棒性

- `utils/evaluation_core_v2.py`：闭环执行、反归一化、安全边界、时序集成和失败分类。
- `utils/perturbations_v2.py`：扰动生命周期和目标瞬移。
- `scripts/benchmark_robustness_v2.py`：配对鲁棒性总控。
- `scripts/analyze_v2_v3_displacement_comparison.py`：V2/V3统计与图表。
- `scripts/run_chunk_ablation_evaluation_mac.sh`：chunk消融Mac批量评估。

### 14.3 冻结权重

- V2 Clean：`artifacts/v2-clean-rc1/mini_vla_v2_clean_policy.pth`。
- V3 Final：`artifacts/v3-clean-rc1/mini_vla_v3_policy.pth`。
- Chunk 1/5/10：`artifacts/chunk-ablation-v3/chunk_*/mini_vla_v3_policy.pth`。

## 15. 当前完成度

| 项目 | 状态 |
|---|---|
| 仿真环境与7D控制 | 完成 |
| 多物体六任务专家数据 | 完成 |
| 双视角+语言+状态历史ACT | 完成 |
| 数据门禁和因果清洗 | 完成 |
| V3训练与冻结 | 完成 |
| 配对Clean基准 | 完成 |
| 0-8 cm动态位移曲线 | 完成 |
| Failure Taxonomy | 完成 |
| Chunk size消融训练 | 1/5/10完成，20使用冻结V3 |
| Chunk size正式评估 | 完成，1,920 episodes，协议审计PASS |
| 视觉噪声/模糊正式曲线 | 未完成 |
| 质量/摩擦物理漂移 | 未完成 |
| OOD干扰物 | 未完成 |
| 相机外参偏移 | 未完成 |
| 留出语言闭环基准 | 未完成 |
| 真实机器人验证 | 未完成 |

## 16. 下一步优先级

1. 运行视觉高斯噪声、模糊和亮度衰减曲线。
2. 运行物体质量、物体摩擦和桌面摩擦漂移。
3. 完成 OOD distractor 与 camera extrinsic shift。
4. 使用 180 条留出语言完成无显式任务标签泄漏的闭环语义测试。
5. 补充参数量、Mac/HPC推理延迟、控制频率、内存和checkpoint尺寸表。
6. 把最新结果写入论文 Methods、Experiments、Failure Analysis、Ablation 和 Limitations。
7. 提交当前尚未进入 Git 的鲁棒性分析、HPC消融和汇报脚本。

## 17. 论文可安全使用的结论

- V3 在严格配对 Clean 场景中显著优于 V2：97.92% vs 92.50%，McNemar p=0.0072。
- V3 在动态目标位移中显著提高闭环恢复能力，并把 80% 成功率点估计边界从 2 cm 推进到 7 cm。
- V3 的主要贡献不只是视觉重新定位，还包括接触后抓取稳定性、错误目标抑制和恢复动作质量。
- V3 在大位移下的主要剩余瓶颈是推送距离不足。
- 这些结论仅适用于当前 MuJoCo/robosuite 协议，不能直接外推到真实机器人或所有OOD扰动。
