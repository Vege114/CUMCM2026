# 机器学习与 CUDA 环境

## 环境划分

代码和数据仍在同一仓库。两个独立 `uv` 项目用于避免 TensorFlow 和 PyTorch 对 NVIDIA 库的版本要求冲突，不要把两个环境的包混装，也不要将 Windows 与 Linux 虚拟环境混用。

| 用途 | 依赖锁 | 本机 WSL 环境 | GPU 运行库 |
| --- | --- | --- | --- |
| TensorFlow / Keras 预测、神经网络训练、自写 TF 策略 | 根目录 `pyproject.toml`、`uv.lock` | `.venv-wsl/` | TensorFlow 2.18.1，CUDA 12.5，cuDNN 9.3 |
| Gymnasium 环境、SB3 的 PPO / DQN / SAC 等算法 | `environments/rl/pyproject.toml`、`uv.lock` | `.venv-rl-wsl/` | SB3 2.9.0，PyTorch 2.8.0+cu128，CUDA 12.8，cuDNN 9.10 |
| 队友 macOS / Windows 原生 CPU | 根目录 `pyproject.toml`、`uv.lock` | `.venv/` | Apple Silicon 自动装 Metal；Windows 原生 TF 使用 CPU |

Python 为 3.12，精确包版本以各自锁文件为准。主环境原有科学计算、Excel、绘图、JupyterLab 依赖保留。RL 环境也包含 NumPy、SciPy、pandas、scikit-learn、openpyxl、Matplotlib、TensorBoard 和 ipykernel。

## 本机安装和运行

Windows 已有 Ubuntu 24.04 WSL2 和 NVIDIA 驱动。2026-09-13 检测到 RTX 4070 Laptop GPU，显存 8188 MiB，驱动 610.62。`nvidia-smi` 显示的 CUDA 版本是驱动支持能力，不代表项目已安装那个版本的 Toolkit。环境由官方 Python wheels 提供所需 CUDA/cuDNN；不在 WSL 安装第二套显卡驱动。

在仓库根目录 PowerShell 执行：

```powershell
.\scripts\setup_ml.ps1                  # 安装两套环境并依次自检
.\scripts\setup_ml.ps1 -Profile tf      # 只安装 TensorFlow
.\scripts\setup_ml.ps1 -Profile rl      # 只安装强化学习

.\scripts\run_ml.ps1 tf python scripts/check_environment.py
.\scripts\run_ml.ps1 rl python scripts/check_rl_environment.py
.\scripts\run_ml.ps1 tf python -m unittest discover -s tests -v

# 所有任务参数使用 WSL 路径或相对仓库根目录的路径
.\scripts\run_ml.ps1 tf python path/to/task.py
.\scripts\run_ml.ps1 rl python path/to/rl_task.py
```

运行脚本默认使用名为 `Ubuntu` 的发行版。其他发行版可用 `setup_ml.ps1 -Distribution <名称>` 安装，并在该发行版终端使用下面的 Bash 命令。若机器还没有 WSL2，先按照 [Microsoft 安装说明](https://learn.microsoft.com/windows/wsl/install) 完成安装和首次用户初始化；Linux 中需已有 [uv](https://docs.astral.sh/uv/getting-started/installation/)。

在 Linux/WSL2 终端中执行（本机仓库路径为 `/mnt/d/CUMCM2026/CUMCM2026`）：

```bash
cd /mnt/d/CUMCM2026/CUMCM2026
bash scripts/setup_ml.sh all
bash scripts/run_ml.sh tf python scripts/check_environment.py
bash scripts/run_ml.sh rl python scripts/check_rl_environment.py
```

启动器始终使用 `--locked`，从仓库根目录运行命令，并设置 TensorFlow 显存按需增长、CUDA/Triton 编译缓存、环境内动态库搜索路径及 XLA 的 `ptxas`/`libdevice` 路径。无需全局修改 `PATH`、`CUDA_PATH` 或 `LD_LIBRARY_PATH`。通常使用上述入口即可；直接在 WSL 使用 `uv` 时，必须保持环境和扩展一致，例如：

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv sync --locked --extra cuda
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run --locked --extra cuda python scripts/check_environment.py
```

## Notebook 和日志

```powershell
.\scripts\run_ml.ps1 tf jupyter lab --no-browser --ip=127.0.0.1
.\scripts\run_ml.ps1 tf tensorboard --logdir experiments --host 127.0.0.1
```

打开命令打印的本地浏览器地址。默认 Notebook kernel 属于 TensorFlow 环境；如需在同一 JupyterLab 中使用 RL kernel，可将 kernel 注册到被忽略的项目缓存，然后重启 JupyterLab：

```bash
bash scripts/run_ml.sh rl python -m ipykernel install \
  --prefix "$PWD/.cache/jupyter-rl" --name cumcm2026-rl --display-name 'CUMCM2026 RL CUDA'
JUPYTER_PATH="$PWD/.cache/jupyter-rl/share/jupyter" \
  bash scripts/run_ml.sh tf jupyter lab --no-browser --ip=127.0.0.1
```

VS Code / Cursor 的 WSL 窗口中，选择 `.venv-wsl/bin/python` 或 `.venv-rl-wsl/bin/python`。Windows 原生编辑器中的 Windows Python 不能加载 Linux CUDA 环境。

## 自检范围与开始训练

TensorFlow 自检覆盖 GPU 矩阵运算、Dense 和 Conv1D/GRU 正反向传播、权重更新、Keras 模型保存恢复、TensorBoard 写入、Gymnasium 交互、附件 1–4 读取、Excel 读写及绘图。RL 自检在指定 GPU 上执行矩阵和卷积反向传播，再执行 256 步 DQN 训练，检查参数更新、保存恢复和日志。自检不证明策略已收敛，也不产生正式实验结果。

2026-09-13 在上述 RTX 4070 Laptop / Ubuntu WSL2 / Python 3.12.3 上完成验证：

- TensorFlow 全部环境自检通过，实际加载 CUDA 12.5.1 与 cuDNN 9.3。
- 现有训练入口的 MLP、GRU、TCN 均通过合成数据训练，输出位于 `GPU:0`；CUDA 环境元数据可正常保存。
- XLA 编译检查通过，环境内 `ptxas` 和 `libdevice` 可用。
- RL 自检通过，实际加载 CUDA 12.8 与 cuDNN 9.10.2；DQN 完成 256 步、56 次更新，策略位于 `cuda:0`，保存恢复及 TensorBoard 日志均通过。
- `python -m unittest discover -s tests -v` 的 8 项现有测试全部通过；相关 Python 文件 Ruff 检查、PowerShell/Bash 语法及 Git 忽略规则检查通过。

旧训练入口的环境记录现已兼容 CUDA，保留模型结构。此次修改改变了 `train.py` 的源代码签名，新的训练应使用新的 `--run-id`，不要接续旧实验缓存。例如小规模试跑：

```powershell
.\scripts\run_ml.ps1 tf python -m experiments.common.neural_v1.train --run-id cuda-smoke-001 --months 2 --variants gru --seeds 42 --epochs 2
```

TensorFlow 和 RL 可以分别运行。8 GB 显卡建议先逐个运行任务，按实际显存使用调整 batch size。SB3 的小型 MLP / PPO 可选择 `device="cpu"`；GPU 可用性由 RL 自检验证，具体任务再选择适合的设备。

## Git 和依赖维护

提交两个项目的 `pyproject.toml`、`uv.lock` 及安装/启动/自检脚本。`.venv*`、下载与 CUDA 缓存、TensorBoard/W&B 日志、`runs/`、`checkpoints/` 和常见模型权重文件均被忽略。SB3 的 `.zip` 模型和 replay buffer 放在 `runs/` 或 `checkpoints/` 下；原始数据、论文、正式工作簿和可复现报告继续按现有约定管理。

```bash
uv add <tensorflow环境的新依赖>
uv add --project environments/rl <强化学习环境的新依赖>
bash scripts/setup_ml.sh all
```

更新任意 CUDA 框架版本后都应重跑对应 GPU 自检，不要使用 `pip install --upgrade` 绕过锁文件。

## 故障定位

- PowerShell 的 `nvidia-smi` 和 `wsl -d Ubuntu -- nvidia-smi` 都应显示 GPU。Windows 正常但 WSL 异常时，检查 WSL2 与驱动集成。
- TensorFlow 在 Windows 原生 Python 下没有 GPU 属于版本支持范围；请使用 WSL 入口。仅 CPU 调试时才加 `--allow-cpu`。
- 出现 CUDA/cuDNN 加载错误时，先运行对应 `setup_ml` 命令同步锁文件，再用 `run_ml` 启动；不要复用另一个环境的 `LD_LIBRARY_PATH`。
- 分配显存失败时，用 `nvidia-smi` 检查已有 GPU 占用并降低 batch size。启动器已启用 TensorFlow 显存按需增长。

官方依据：[TensorFlow pip / WSL2 安装](https://www.tensorflow.org/install/pip)、[TensorFlow CUDA/cuDNN 版本表](https://www.tensorflow.org/install/source#gpu)、[NVIDIA WSL CUDA 指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)、[Stable-Baselines3 安装](https://stable-baselines3.readthedocs.io/en/master/guide/install.html)、[PyTorch 官方历史版本](https://pytorch.org/get-started/previous-versions/)、[uv 配置 PyTorch 索引](https://docs.astral.sh/uv/guides/integration/pytorch/)。
