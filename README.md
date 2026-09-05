# AlphaFactorService

AlphaBlocks统一的数据研究服务，负责因子、Qlib模型训练与推理、模型信号和快速回测。

## 职责边界

- Studio 负责展示、编辑、创建任务、查看结果。
- AlphaFactorService 负责保存因子定义、管理任务、执行计算、查询结果。
- AlphaFactorService 内置研究调度器，负责LightGBM、XGBoost、CatBoost和PyTorch MLP训练与每日推理。
- 因子API和研究调度器运行在同一个服务进程、共用8100端口；原生模型训练由短生命周期子进程隔离。
- ClickHouse 的 `ab_factor` 库负责保存因子定义、计算任务和因子结果。
- 不再用本地 SQLite 或 YAML 保存因子库，元数据和结果统一由 ClickHouse 承载。

## 第一阶段能力

- `GET /factors` 查询因子定义。
- `POST /factors` 创建因子定义。
- `PUT /factors/{factor_id}` 更新因子定义。
- `DELETE /factors/{factor_id}` 停用因子定义。
- `POST /factor-jobs` 创建计算任务。
- `GET /factor-jobs` 查询任务。
- `POST /factor-jobs/{job_id}/run` 执行指定任务。
- `POST /factor-jobs/run-pending` 批量执行 pending 任务。
- `GET /factor-values` 查询因子结果。
- `GET /factor-values/coverage` 查询覆盖率框架。
- `POST /factor-values/sync-states` 批量查询因子规格的真实持久化覆盖范围，供自动全量/增量同步规划使用。
- `POST /model-inference/availability` 按冻结因子、数据截止时间和历史交易日检查每日推理可用性。
- `GET /model-signals` 返回指定不可变模型版本、交易日的PIT安全TopN信号，供AlphaBlocks正式策略回测读取。
- `POST /model-backtests/jobs` 支持用可选 `benchmark_code` 将选股范围与报告基准分离；基准只能取系统已配置指数，未传时仍使用股票池默认基准。
- `POST /model-research/jobs` 创建训练任务，`POST /model-research/jobs/{job_id}/dispatch` 在本服务中调度执行；`execution.max_runtime_minutes`冻结60至1440分钟的任务上限，超时后终止隔离进程并记录`training_timeout`。
- `DELETE /model-research/models/{model_id}/versions/{version}` 永久删除未被主模型、运行任务、部署、架构或其他冻结模型引用的版本，并清理其预测、回测与独占产物。
- 删除模型时会自动核对 `paper` 部署快照对应的模拟盘是否仍存在；仅当策略已删除时，才清理孤儿部署引用并继续删除模型。仍存在的策略、非 `paper` 部署或无法核对的引用都会保留并返回冲突。
- `DELETE /model-research/architectures/{architecture_id}` 永久删除未激活且没有运行中回测的模型架构，并清理其引擎引用和ClickHouse回测证据。
- `GET /research/ready` 和 `GET /research/status` 查询内置研究调度器状态。

聚宽因子迁移实施参考见 [`docs/joinquant-factor-catalog.md`](docs/joinquant-factor-catalog.md)，可通过
`rtk .venv/bin/python scripts/export_joinquant_factor_catalog.py` 重新生成公开目录与详情快照。

当前数据源和公式引擎的逐因子兼容性检测见
[`docs/joinquant-factor-compatibility.md`](docs/joinquant-factor-compatibility.md)，可通过
`rtk .venv/bin/python scripts/audit_joinquant_factor_compatibility.py` 重新实测字段覆盖率并生成报告。

将审计通过的 84 个聚宽因子幂等导入当前因子库：

```bash
rtk .venv/bin/python -m scripts.import_joinquant_ready_factors
rtk .venv/bin/python -m scripts.import_joinquant_ready_factors --apply
```

第一条命令只校验；第二条才写入定义，不会自动创建计算任务或启动全历史同步。重复执行不会增加版本；只有显式追加
`--update-existing` 才会覆盖同名但定义不同的因子并创建新版本。
- `POST /factor-analysis/jobs` 创建 Alphalens 标准分析任务。
- `POST /factor-analysis/jobs/{analysis_job_id}/run` 执行分析任务。
- `GET /factor-analysis/summary` 查询 IC、分位收益、换手等汇总结果。
- `GET /factor-analysis/ic` 查询每日 IC 序列。
- `GET /factor-analysis/quantile-returns` 查询分位收益序列。
- `GET /factor-analysis/turnover` 查询分位换手和 Rank 自相关序列。

当前计算引擎先支持默认股票日频因子的 ClickHouse SQL 计算，包括均值、区间涨跌幅、涨停次数、N 日首次涨停。结果由 worker 写入 ClickHouse。
因子分析使用 `alphalens-reloaded` 做标准化计算，分析结果仍落在 ClickHouse。

每次日频因子同步会完成整条横截面处理流水线：

```text
raw_value -> rank_value -> percentile -> score
```

- `raw_value` 是公式原始结果。
- `rank_value` 是当日横截面降序名次，原值最大者排名 1，并列值使用相同名次。
- `percentile` 是当日横截面百分位，范围 0 到 1。默认升序时原值越大越接近 1；rank 或分位分层配置为降序时原值越大越接近 0。
- `score` 是模型输入值。布尔因子保留 0/1；连续因子默认使用 1%/99% 缩尾后的横截面 Z-score，也可由因子的 `data_processing` 配置改为 MAD/中位数去极值、rank 标准化、每日横截面分位分层或不标准化。分位分层使用 `{"standardize":"quantile","quantiles":5,"direction":"asc"}`，其中 `quantiles` 可配置为 2 至 20；`direction` 可配置为 `asc`（原始值从小到大，第 1 档为低值端）或 `desc`（原始值从大到小，第 1 档为高值端）。旧的 `quantile5` 仍兼容为 5 档升序分层。

新批次成功落库后，worker 会清理同一因子版本、参数、实体和日期范围内的旧批次，避免重算后同时读到旧原始值与新标准分。写入失败时不会提前删除旧结果。

当前日频源尚未绑定行业与市值暴露。配置行业或市值中性化的任务会明确失败，避免将未执行的中性化误报为成功。

## 启动

```bash
cd /Users/zhao/Desktop/git/AlphaFactorService
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp config/runtime.example.yaml config/runtime.local.yaml
alpha-factor-service
```

默认服务地址：

```text
http://127.0.0.1:8100
```

## PM2 启动

```bash
cd /Users/zhao/Desktop/git/AlphaFactorService
python3.11 -m venv .venv
.venv/bin/pip install -e .
pm2 start ecosystem.config.js
```

远程GPU节点不使用`runtime.local.yaml`。节点普通字段保存在PostgreSQL
`model_execution_nodes`，SSH密码、SSH私钥正文和AutoDL Token经AES-GCM加密后保存在
`model_execution_node_secrets`。首次启用前生成并导出固定主密钥，再用`--update-env`启动PM2：

```bash
export ALPHA_REMOTE_NODE_SECRET_KEY="$(openssl rand -base64 32)"
pm2 startOrReload ecosystem.config.js --update-env
```

所有AlphaFactorService副本必须使用同一主密钥；接口不会返回任何秘密明文。

PM2只启动一个常驻进程`alpha-factor-service`，统一提供因子、模型信号、回测、
Qlib多模型训练和每日推理。模型训练在任务期间启动隔离子进程，完成后自动退出。
PyTorch MLP使用显式`hidden_layers`数组冻结逐层宽度，例如`[64, 128, 256]`表示
`输入维度 → 64 → 128 → 256 → 1`；层结构会进入任务配置和训练manifest，保证可复现。
PyTorch LSTM使用Qlib `TSDatasetH`按股票构建因果历史窗口，默认结构为
`60交易日 × 因子维度 → 2层LSTM(128) → 1`。训练、Walk-Forward、测试段预测和每日推理
使用同一`lookback_window`，不允许把不同股票的数据拼成一条序列。
Transformer+LSTM混合模型在同一窗口上先执行带因果遮罩的Transformer Encoder，再把
时序表示交给LSTM输出截面预测。默认结构为`60日 → Transformer 64D/4头/2层 →
LSTM(128) → 1`，所有结构参数随模型版本冻结。

研究能力已经融合到`factor_service.research`模块，不再存在第二个顶层业务包。当前处于开发阶段，
训练产物只接受新的`factor_service.research.models`类路径，不保留旧Worker包兼容层。

默认只监听本机：

```text
http://127.0.0.1:8100
```

如果AlphaBlocks运行在另一台机器，部署机的`runtime.local.yaml`需要改为：

```yaml
service:
  host: 0.0.0.0
  port: 8100
```

同时把AlphaBlocks的`external_services.factor_service.base_url`设为这台部署机的局域网地址。

监听地址、ClickHouse、数据源与研究存储位置统一配置在：

```text
config/runtime.local.yaml
```

`runtime.local.yaml`已被Git忽略。部署时可以用`ALPHA_FACTOR_RUNTIME_CONFIG`显式选择
另一份YAML；除Python解释器选择外，PM2不再通过环境变量覆盖业务配置。

统一服务使用Python 3.11或3.12，并复用同一份`clickhouse`配置。研究配置示例：

```yaml
research:
  scheduler:
    enabled: true
    refresh_seconds: 60
  storage:
    work_root: /Volumes/QuantData/alphafactor/research-work
    model_artifacts_root: /Volumes/QuantData/alphafactor/model-artifacts
    dataset_cache_retention_hours: 24
    dataset_cache_cleanup_interval_seconds: 3600
```

每日推理计划由同一个AlphaFactorService进程按`research.scheduler.refresh_seconds`检查，
不需要AlphaBlocks再运行独立的模型推理调度进程。

`research.storage.work_root`只保存训练暂存文件、预测Parquet、MLflow记录、日志和任务状态；
`research.storage.model_artifacts_root`保存FactorService本地正式制品。可选的
`research.storage.object_store`会在本地SHA256校验完成后，把`bundle`和
`walk_forward_series`自动归档到S3兼容对象存储。只有远端上传及二次校验成功后，训练任务
才会进入成功状态；对象URI、Bucket版本号和SHA256会随模型制品登记到PostgreSQL。成功的
普通训练会直接登记为不可变`candidate`模型版本，不再等待用户确认；参数实验只自动登记验证集
入选trial。模型验证、默认版本、自动推理和交易发布仍分别受原有门槛控制。

推理机器优先读取`model_artifacts_root`中的本地正式制品；本机缺失时会使用PostgreSQL登记的
对象URI和版本号从MinIO下载到`.object-cache/{sha256}`，并校验登记大小、远端元数据SHA256和
下载内容SHA256。多台FactorService只需配置同一个控制库、MinIO endpoint、Bucket和读取凭据，
不需要挂载训练主机的本地模型目录。

对象存储凭据放在Git忽略的文件中，例如：

```dotenv
MINIO_APP_ACCESS_KEY=service-account
MINIO_APP_SECRET_KEY=replace-me
S3_ENDPOINT=http://10.126.126.5:9000
S3_BUCKET=alphablocks-models
S3_REGION=us-east-1
```

远程训练已经生成完整`remote_result.json`，但下载或发布阶段被取消时，可在完成文件哈希核验后执行
只发布、不重训的恢复命令：

```bash
python -m factor_service.research.artifact_recovery <job_id> --source-attempt <ordinal>
```

恢复操作只接受`failed`或`canceled`训练任务及已终止的来源Attempt。系统会新建一个本地
`uploading` Attempt，继续本地正式制品、MinIO、预测结果和候选模型登记，并保留原失败或取消
Attempt的审计记录；不会把原Attempt改写为成功，也不会再次训练模型。

`research.storage.model_artifacts_root`保存经过SHA256校验并原子发布的正式模型产物，以及
`datasets/{dataset_hash}`下的单份训练数据集缓存。启用MinIO后，`dataset.parquet`、
`dataset_raw.parquet`和`dataset_manifest.json`在数据构建完成时就归档，不等待训练成功。
对象键使用`{prefix}/datasets/{dataset_hash}/{file_sha256}/{file_name}`，跨任务与模型版本复用。
上传后完整流式回读并校验大小和SHA256，三份文件全部通过后原子登记到PostgreSQL
`model_dataset_archives`，并补齐历史`model_artifacts`中的对象身份。必须先应用AlphaBlocks控制库
迁移`0041_model_dataset_archives.sql`。归档独立于模型任务，删除单个模型不会删除共享数据归档。

Qlib、远程推送、数据预览、模型诊断和数据下载只读取校验过的本地快照。缺失时优先按数据库记录的
精确对象版本从MinIO恢复，不能因对象损坏、网络失败或权限错误悄悄重算。首次没有归档、也没有
本地文件时才按冻结Dataset Spec生成。本地缓存最多保留一份Dataset Hash：任务/读取结束后保留当前
快照，同Hash直接复用；换Hash时先核验旧数据的MinIO归档和本地文件，再删除旧缓存，之后才生成或
恢复新数据。单份指一组处理后Parquet、原始Parquet与清单，不是只保留其中一个文件。
缓存准入锁与跨进程使用锁覆盖构建、训练（包括隔离子进程）、预览、诊断和下载的完整使用期；同Hash
可共享，不同Hash遇到正在使用的缓存时明确提示稍后重试，不能先写第二份。排队但尚未使用本地文件的
任务不占用缓存，轮到它时可以从MinIO恢复。未归档、上传/登记失败、远端核验失败或含未知文件时保留
本地数据并拒绝替换，不强行满足数量上限。后台仅清理可安全删除的历史副本，最近一份不再按TTL过期；
历史未归档副本可用下面的维护命令逐一补归档。源ClickHouse范围暂存的24小时TTL不变。
未启用MinIO的独立离线部署保留原24小时本地缓存策略。

已有数据可以使用以下维护命令归档、清理并测试按需恢复；它不会创建训练任务、重新计算因子或
重新登记模型。`--evict`只删除通过远端校验、没有活动使用者的本地数据集副本；`--verify-restore`
恢复校验完成后按单份策略保留该缓存：

```bash
python -m factor_service.research.dataset_archive <existing_job_id> --evict --verify-restore
```

成功发布并校验本地规范快照后删除`dataset_staging`重复副本；随后上传或登记中断时保留该规范快照。
模型Bundle继续自动归档到对象存储，不受数据集缓存清理影响。相对路径从项目根目录解析；需要
放到外置磁盘或指定数据盘时建议直接填写绝对路径。AlphaBlocks只保存产物元数据。

AlphaBlocks只配置`external_services.factor_service.base_url`。没有单独的研究服务地址或端口。
模型任务、事件、版本和产物元数据由AlphaFactorService直接写入`control_database`，
AlphaFactorService不通过AlphaBlocks API回写模型状态或元数据。只有当因子公式引用的字段尚未
物化到本地因子源视图时，计算器才通过AlphaBlocks只读统一数据SDK查询股票实体资产的日频复合
视图；字段授权、实体关系和财务PIT对齐仍由AlphaBlocks统一数据层负责。

同一Dataset中的多个因子共享复合实体资产时，FactorService优先调用AlphaBlocks内部日期范围暂存
接口，并对完整Dataset区间只建立一次来源暂存；因子输出仍按有界日期块计算，但暂存绑定不会再把
每个块的输出边界覆盖成完整Dataset范围。`alphablocks.dataset-pipeline.v9`会在每个日期块内把
同源因子编译为一次宽表查询，日期和股票键只传输一份；因子各自的公式、PIT截止、缩尾和标准化
保持独立。v9的因子级分位缩尾使用因子参数Hash作为确定性采样盐，保证单因子与宽表计算一致。
v8及更早冻结任务继续走逐因子兼容查询，用原口径精确重建，不会被v9静默改写。主行情字段由
ClickHouse按范围直接物化，财务PIT字段由AlphaBlocks一次扫描范围内的报表披露，生成稀疏事件后
通过`ASOF JOIN`展开到交易日，不再按日重复扫描三张报表。返回绑定还会由FactorService自己的
ClickHouse连接二次确认。暂存身份包含Dataset、交易日、字段、`data_cutoff`、节点版本、实体资产
版本和PIT事件合同，并由24小时TTL清理。
当旧版AlphaBlocks尚未提供该内部接口时，FactorService保留按日查询兼容路径；兼容路径同样必须
传递冻结`data_cutoff`，不能使用执行时的当前时间代替训练计划时点。

单模型LightGBM、XGBoost和CatBoost支持冻结`optuna`配置后执行10至100次TPE搜索。非滚动任务
把固定验证段切成连续子窗口；Walk-Forward任务在正式样本外起点之前生成多个内层调参折，每折
独立移动训练/验证区间并重新拟合。两种模式都跨多个随机种子计算“平均Rank ICIR减去波动惩罚”，优先选择达到正向
窗口比例门槛的参数。搜索trial、窗口/种子指标与最佳参数写入模型Manifest和
`optuna_trials.json`，测试段不参与选择；最佳参数随后用于正式训练。验证长度为0、增量续训、
参数实验、多模型对比和Stacking不允许
同时开启Optuna。内层调参完成后把最佳参数冻结到全部外层Walk-Forward窗口，外层测试不参与选参。

LightGBM完整迭代指标通过当前Qlib Recorder的MLflow客户端，每批最多1000条同步写入，
不再逐指标进入异步队列。普通训练、Optuna内层折、Walk-Forward及Stacking中的LightGBM
共享这条路径；指标名、trial/窗口前缀、step和数值保持不变。每批之间响应取消，写入错误
直接上报，进度显示`training_metrics_writing`/`training_metrics_written`及已保存条数。
不关闭SQLite同步落盘，不丢弃训练曲线，也不改变模型、早停或选参结果。此优化只对加载
新版程序的新任务生效，不热修改正在运行的进程或已有MLflow数据库。

可用`PYTHONPATH=. python scripts/benchmark_training_metrics.py --work-dir <新的临时目录>`
对比旧Qlib逐条队列写入（包含结束时排空队列）和新批量写入。只生成合成指标与独立SQLite库，
反向交换两轮执行顺序并回读每条曲线；`report.json`中的加速比仅代表指标保存，不是训练总耗时。

### Walk-Forward滚动评估

远程节点使用独立、轻量的内存监护进程启动训练子进程。有效资源取宿主机可用内存、
cgroup v1/v2及可见父级限制的较小值，CPU同时受亲和性和配额限制，不使用宿主机标称配置
替代容器容量。规划估算仍预留有效内存的15%（至少512 MiB），仅作提示而非启动准入条件。
文件缓存中的inactive_file可回收部分不计入活跃工作集，避免上传数据后误判内存耗尽。

macOS CPU节点可使用`direct_python`运行；所选Python环境需安装研究依赖（包括Optuna）。
资源探测使用`sysctl`和`vm_stat`，按实际页大小计算可用内存，不把压缩内存或swap计入预算；
进程组通过Python POSIX接口创建，不依赖Linux的`setsid`命令。连接测试与训练监护共用资源
读取逻辑，无法读取真实内存时拒绝启动；macOS同样保留运行期安全余量。

启动前会按冻结快照的行数、特征数估算工作集；超过规划预算时记录警告，仍尝试加载和训练。
该估算不是峰值保证：监护进程每秒检查实时内存，每10秒回传资源状态，即使模型底层计算
不回报迭代进度也能检查。实际可用内存连续3次低于运行期底线（有效内存的5%，最少512 MiB、
最多1 GiB），或一次不足等于256 MiB时，仅终止本任务训练进程组，返回
`node_memory_budget_exceeded`；紧急不足时不启动子进程。确认系统OOM杀进程或Python内存分配
失败返回`node_out_of_memory`。多节点任务可转交其他所选节点（见下文）；单节点不会自动循环
重试，用户释放资源后可显式重试原任务，原失败记录保留。
一秒内的突发分配仍可能先触发系统OOM，不能把监护阈值当作内核级硬内存限制。

远程节点的`cache/datasets`最多保留一套训练快照（两份Parquet及清单），不按任务重复下载。
相同Dataset Hash且三个文件的SHA256与大小一致时直接复用；不同Hash先核对所有旧缓存的
MinIO归档身份及对象可读性，再删除旧数据、传入新数据，全部校验成功后才允许训练。
已有多个Hash缓存会在下次训练时按此规则收敛；未归档的数据、未知文件及符号链接不会强删。
传输中断只留下当前一套未完成缓存，下一次同Hash任务可重新校验并续传，不会使用半成品。
代码缓存、历史任务产物以及MinIO正式归档不受单份数据缓存策略影响。

上传、训练、回传期间独占节点工作目录下的`cache/.dataset-use.lock`。其他任务不得替换缓存；
旧版本训练进程或数据传输仍活动、节点失联时同样拒绝清理。任务正常结束或确认停止后释放锁。
控制端异常退出且无法确认远程任务结束时保留占用锁，不按时间强制过期；需先核实旧任务及
传输均已结束再处理遗留锁，避免自动重试误删正在使用的数据。该限制是磁盘缓存策略，不改变
模型内存预算或训练数据范围，也不实现多节点联合训练。

滚动训练每窗完成后立即保存模型、预测和带SHA256的manifest，释放旧DatasetH、DataFrame与
LightGBM原生训练矩阵；只保留最后一窗供根模型兼容别名及诊断使用，不随窗口数保留训练数据。
Optuna在各折/试验之间回收内存。不完整序列不注册模型、不发布评分；失败时已保存的窗口文件
仍保留供核验，本次内存改造不自动跨Attempt复用窗口，也不缩减冻结股票池、特征或日期范围。

可在节点执行`PYTHONPATH=. python scripts/check_training_memory.py --work-dir <新的临时目录>`
进行小规模合成数据回归检查；该脚本不访问行情、不登记模型、不运行用户的全A训练任务。

训练任务可选开启严格Walk-Forward评估。窗口长度全部使用实际交易日，默认依次执行
`756日训练 → 隔离 → 60日验证 → 隔离 → 20日独立测试`，隔离长度不得小于标签周期。
用户显式选择样本外起止日，服务生成覆盖整个区间的全部窗口，不再使用最大窗口数截断。
逐窗缺失值中位数仅使用该窗训练段拟合；每窗模型、预处理参数、指标和生效区间都随模型版本
持久化，拼接后的样本外分数写入模型信号库。

```json
{
  "walk_forward": {
    "enabled": true,
    "strategy": "rolling",
    "train_sessions": 756,
    "valid_sessions": 60,
    "test_sessions": 20,
    "step_sessions": 20,
    "embargo_sessions": 5,
    "oos_date_start": "2022-01-04",
    "oos_date_end": "2025-12-31"
  }
}
```

为保证样本外日期完整且预测唯一，`step_sessions`必须等于`test_sessions`；末窗不足一个标准
测试周期时按剩余交易日生成。`expanding`策略固定首个训练日并逐窗扩展训练段；`rolling`策略
保持训练长度不变并向前滚动。

安装后可用统一命令执行环境诊断：

```bash
alpha-factor-service doctor
```

如果以后需要独立后台消费 pending 任务，可以单独运行：

```bash
python -m factor_service.worker --limit 5 --poll-interval 60
```

## ClickHouse 初始化

ClickHouse连接信息写在本项目`config/runtime.local.yaml`：

```yaml
clickhouse:
  host: 10.126.126.3
  port: 8123
  username: default
  password: ""
  secure: false
  factor_database: ab_factor
  model_database: ab_model
```

其中`factor_database`是因子服务自己的库，默认`ab_factor`。

因子计算的数据源单独配置，默认优先读取本地物化视图
`ab_factor.stock_daily_factor_source`：

```yaml
sources:
  factor:
    database: ab_factor
    stock_daily_table: stock_daily_factor_source
    stock_code_column: code
    stock_date_column: trade_time
    stock_price_column: close
    stock_basic_table: stock_basic_factor_source
    stock_basic_type_column: type
    stock_basic_stock_type_value: "1"
    entity_asset_api_base_url: http://127.0.0.1:8001/api/data-sdk
    entity_asset_query_timeout_seconds: 120
    entity_asset_query_concurrency: 4
```

当前股票日频因子需要从多张实体资产表组合字段：行情来自 `starlight.ad_market_kline_daily`，
涨跌停和状态来自 `starlight.ad_history_stock_status`，换手率等字段来自 `baostock.bs_daily_kline`。
先创建因子计算源视图：

```bash
python - <<'PY'
from pathlib import Path
from factor_service.clickhouse import client

for statement in [item.strip() for item in Path("scripts/create_factor_source_views.sql").read_text().split(";") if item.strip()]:
    client().command(statement)
PY
```

公式只引用上述物化视图已有字段时继续走ClickHouse快路径。若公式还引用了股票实体资产中已授权、
但尚未物化的字段，worker优先请求统一数据SDK生成可复用的范围暂存，并继续复用同一套公式编译、
去极值和标准化流程。旧版AlphaBlocks不支持范围暂存时才回退到按日查询。公式计算目前只接受数值
或布尔字段；文本字段即使可在字段目录中查看，也会在计算时返回明确错误。

然后将计算源指向视图：

```bash
clickhouse-client < scripts/init_clickhouse.sql
```

也可以直接在 ClickHouse 控制台执行 `scripts/init_clickhouse.sql`。

如果需要导入当前 AlphaBlocks 里已有的几个默认因子：

```bash
clickhouse-client < scripts/seed_default_factors.sql
```

## 因子结果校验

可以用脚本检查源表、公式编译、worker dry-run 和已落库结果是否一致：

```bash
python scripts/validate_factor_outputs.py
```

## 数据存储

### 模型任务的多节点执行

Studio 多选远程训练节点时，FactorService 使用一个父任务协调执行：Optuna 按 trial 并行，
每个 trial 的验证折/随机种子顺序执行；正式 Walk-Forward 按独立窗口并行。
节点只运行一个子任务并复用单份数据集缓存，父任务最终汇总为一个模型版本。
这不是单个模型的数据并行训练。

多节点执行合同为 `execution.mode=distributed`、`node_id=distributed`、
`node_ids=[已配置节点ID...]`；`max_runtime_minutes` 为整个任务的总时限。
当前最多 16 个远程节点，不混用 `local`，不支持 Stacking 或增量续训。
没有 Optuna 或独立滚动窗口时应选一个节点。

部署前运行 AlphaBlocks 控制库迁移 `0042_model_training_subtasks.sql`。
子任务参数、状态和归档身份由 PostgreSQL 记录，产物在 MinIO 上传并完整校验后才能标记完成。
节点不需要 PostgreSQL/MinIO 凭据。恢复任务时，完成项直接复用；未完成 trial 沿用原参数，
新提案重新播种（异步 TPE 不保证与连续单节点搜索逐位一致）。全局 trial 预算不会按节点数翻倍。
`GET /model-research/jobs/{job_id}` 的 `subtasks` 字段可查看各节点执行状态。

内存估算不足先尝试执行；实际触发运行期内存保护或OOM后，原子任务使用相同冻结参数/窗口
转交其他所选节点。失败节点在本次父任务Attempt内暂时停用；健康节点繁忙时等待，不叠加
同节点并发。每个内存失败节点至多尝试一次；所有所选节点均失败才结束父任务并保留已完成
子任务。显式重试会重新尝试所选节点并复用完成项，不改变数据集、全局trial预算或已提案参数。
临时网络等错误仍最多两次尝试；无效参数/数据等确定性错误不按内存失败转交。
`distributed_node_memory_failed`和`distributed_task_reassigned`事件记录失败与承接节点。

SSH认证、连接中断和传输超时单独分类为`node_ssh_authentication_failed`或
`node_ssh_connection_failed`。确认SSH失败后，本轮暂时停用该节点，将同一子任务交给其他
所选节点；重试范围有界于所选节点，不在同一节点反复尝试密码（每次连接最多一次密码提示）。
全部节点不可用时返回`node_execution_unavailable`，保留完成项供显式恢复。其他临时故障
仍沿用最多两次尝试。主机身份校验失败不放宽校验，数据/文件权限错误不冒充网络错误。
断连后不重复清理或解锁，标记`remote_cleanup_pending`；下次节点恢复后仍需确认旧进程已
结束才能恢复同父任务缓存锁。`distributed_node_ssh_failed`记录故障，承接事件标注`reason=ssh`。
隔离进程保留结构化异常，页面展示节点与具体原因，完整堆栈保留在原任务日志。旧版被包装为
`unexpected_error`的SSH失败仅在较长错误记录中能确认最终SSH异常时允许显式续跑；不重写审计记录。
每个节点/父Attempt使用独立SSH ControlMaster连接，缓存检查、传输和进度查询复用认证会话，
不在每次轮询重新认证。控制套接字放在本机私有0700临时目录，不跨节点/任务共享；任务结束
显式关闭，异常遗留连接空闲60秒后退出。不修改服务器SSH配置或保存新的持久认证凭据。

分布式执行要求 MinIO 启用 `bundle` 归档。子任务归档使用独立的 `distributed_<hash>` 对象前缀，
不会写入正式模型注册表。自动关机只在父任务结束且节点确认空闲后执行；用户手动管理的节点不关机。
断连遗留锁只允许在确认旧进程退出且属于同一父任务时恢复，禁止按年龄强制删除。

训练进度按父任务Attempt记录`phase`、`phase_seconds`和`attempt_elapsed_seconds`，
百分比在同一次Attempt内不倒退；完成态统一返回100%。分布式进度额外包含各类型任务的
完成/执行中/等待/失败/复用数量，以及逐节点资源快照。每个子任务的执行耗时与最后资源采样
写入子任务账本的`execution_observation`；RSS峰值是监护进程每秒采样所得，不声称是内核级
精确峰值。节点RSS在开始下一个任务时清空，避免不同任务串用。旧任务没有采样的字段不回填。
主任务的MinIO上传回调每两秒至多报告一次字节进度，保留取消检查和原有对象身份校验。

新滚动模型只在`walk_forward_series.tar.gz`中保存窗口目录；主模型包通过
`manifest.walk_forward_artifact`固定该包的SHA256和大小，不再重复打包窗口。
启用MinIO的滚动训练必须同时开启`walk_forward_series`归档。每日推理冻结独立序列的
artifact_id/SHA256/大小并验证与根Manifest一致，缺失或不匹配明确失败，不能退回最新窗口。
不支持旧版内嵌窗口的Bundle；滚动模型缺少独立序列包清单时，提交推理和执行推理都会明确失败。
非滚动模型正常读取主模型包，不要求序列包。不自动转换、重写或删除已登记模型及MinIO对象；
旧滚动模型已经发布的预测数据保留，新增推理需使用独立序列包格式的模型。
以上只改执行观测和制品布局，不改变数据、参数、训练窗口、预算及选参规则。

所有数据都落在 ClickHouse 的 `ab_factor` 库：

```text
ab_factor.factor_definitions
ab_factor.factor_compute_jobs
ab_factor.factor_values_daily
ab_factor.factor_analysis_jobs
ab_factor.factor_analysis_summary
ab_factor.factor_analysis_ic_daily
ab_factor.factor_analysis_quantile_returns
ab_factor.factor_analysis_turnover_daily
```

其中：

- `factor_definitions` 保存因子定义、版本、参数、表达式。
- `factor_compute_jobs` 保存计算任务和执行状态。
- `factor_values_daily` 保存日频因子结果。
- `factor_analysis_jobs` 保存因子评价任务。
- `factor_analysis_*` 保存 Alphalens 生成的 IC、分位收益、换手和汇总指标。

`factor_values_daily` 同时保存两套时间：`event_available_at` 是行情事件理论可用时间，`computed_at` 是因子批次实际生成时间。策略查询默认按 `computed_at` 和 DataCutoff 做严格截断；只有显式历史重建研究才按 `event_available_at` 查询。`source_vintage` 记录计算源和任务批次。Alphalens 对日频收盘因子统一延迟一个交易日后再计算 forward return，避免用当天收盘数据又假设能在当天收盘成交。

模型训练不要求先写入`factor_values_daily`：任务会锁定因子版本、公式参数和`params_hash`，
直接从源数据即时计算，先在`work_root`暂存，再发布为不可变Parquet。因子中心的同步、评价和
单因子回测仍继续使用`factor_values_daily`，两条流程互不替代。

`model-signals`只返回`feature_cutoff_at <= T日15:00`的模型预测，并携带
`dataset_hash`和`inference_run_id`。接口只提供因果安全的信号截面；正式策略仍由AlphaBlocks
执行T+1成交、涨跌停/停牌、费用和持仓规则，FactorService不会在这里复制策略引擎。

后续分钟级结果可以单独增加：

```text
ab_factor.factor_values_intraday
```
