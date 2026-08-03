# 内力抑制模块 - 安装与依赖说明

## 📋 依赖项

### 核心依赖（必需）

```bash
pip install numpy
pip install pyyaml
pip install pinocchio  # 机器人动力学库
```

### 可选依赖

```bash
# 用于测试
pip install pytest

# 用于可视化
pip install matplotlib

# 用于MuJoCo仿真
pip install mujoco
```

---

## 🔧 安装方法

### 方法 1: 直接使用（推荐用于开发）

将 `src` 目录添加到 Python 路径：

**Linux/Mac:**
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/dual-tron1-mujoco/src"
```

**Windows PowerShell:**
```powershell
$env:PYTHONPATH = "E:\Robot\Sustech-多人形协作\Multiple_Humanoid_Cooperative_Transportation\dual-tron1-mujoco\src"
```

**Windows CMD:**
```cmd
set PYTHONPATH=E:\Robot\Sustech-多人形协作\Multiple_Humanoid_Cooperative_Transportation\dual-tron1-mujoco\src
```

### 方法 2: 开发模式安装

在项目根目录创建 `setup.py`:

```python
from setuptools import setup, find_packages

setup(
    name="internal_force_suppression",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.20.0",
        "pyyaml>=5.4.0",
        "pin>=2.6.0",  # pinocchio
    ],
    extras_require={
        "test": ["pytest>=7.0.0"],
        "viz": ["matplotlib>=3.5.0"],
    }
)
```

然后安装：
```bash
pip install -e .
```

---

## 🐍 Pinocchio 安装

Pinocchio 是机器人动力学计算的核心库，但安装可能比较复杂。

### 选项 1: Conda（推荐）

```bash
conda install pinocchio -c conda-forge
```

### 选项 2: pip（Linux/Mac）

```bash
pip install pin
```

### 选项 3: 从源码编译

如果上述方法失败，请参考：
https://stack-of-tasks.github.io/pinocchio/download.html

### Windows 用户注意

Pinocchio 在 Windows 上的安装可能需要：
1. 使用 conda
2. 或使用预编译的二进制文件
3. 或使用 WSL (Windows Subsystem for Linux)

---

## ✅ 验证安装

### 测试 1: 验证项目结构

```bash
python validate_ifsm_structure.py
```

**预期输出**:
```
✓ Found files: 18
✗ Missing files: 0
🎉 SUCCESS!
```

### 测试 2: 验证依赖导入

创建测试脚本 `test_imports.py`:

```python
import sys
from pathlib import Path

# 添加 src 到路径
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

print("Testing imports...")

try:
    import numpy as np
    print("✓ numpy")
except ImportError:
    print("✗ numpy - pip install numpy")

try:
    import yaml
    print("✓ pyyaml")
except ImportError:
    print("✗ pyyaml - pip install pyyaml")

try:
    import pinocchio as pin
    print("✓ pinocchio")
except ImportError:
    print("✗ pinocchio - conda install pinocchio -c conda-forge")

try:
    from internal_force_suppression import __version__
    print(f"✓ internal_force_suppression (v{__version__})")
except ImportError as e:
    print(f"✗ internal_force_suppression - {e}")

print("\nDone!")
```

运行：
```bash
python test_imports.py
```

### 测试 3: 运行基础测试（无需Pinocchio）

```bash
# 这个会失败，因为需要 pinocchio
python -m pytest tests/internal_force_suppression/test_basic.py -v
```

---

## 🚨 常见问题

### Q: ModuleNotFoundError: No module named 'internal_force_suppression'

**解决方案**:
```bash
# 设置 PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# 或使用绝对路径
export PYTHONPATH="${PYTHONPATH}:/full/path/to/dual-tron1-mujoco/src"
```

### Q: ModuleNotFoundError: No module named 'pinocchio'

**解决方案**:
```bash
# 使用 conda（推荐）
conda install pinocchio -c conda-forge

# 或使用特定环境
conda create -n ifsm python=3.10
conda activate ifsm
conda install pinocchio -c conda-forge
pip install pyyaml pytest matplotlib
```

### Q: 测试失败

**原因**: 测试代码中直接导入了 pinocchio

**临时解决方案**: 使用示例代码而非单元测试验证功能

```bash
# 这个示例也需要 pinocchio
python examples/internal_force_suppression/demo_force_estimation.py
```

---

## 📦 完整安装流程（推荐）

### For Linux/Mac:

```bash
# 1. 创建虚拟环境
conda create -n ifsm python=3.10
conda activate ifsm

# 2. 安装依赖
conda install pinocchio -c conda-forge
pip install pyyaml pytest matplotlib

# 3. 设置路径
cd /path/to/dual-tron1-mujoco
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# 4. 验证
python validate_ifsm_structure.py
python examples/internal_force_suppression/demo_force_estimation.py
```

### For Windows:

```powershell
# 1. 创建虚拟环境
conda create -n ifsm python=3.10
conda activate ifsm

# 2. 安装依赖
conda install pinocchio -c conda-forge
pip install pyyaml pytest matplotlib

# 3. 设置路径
cd E:\Robot\Sustech-多人形协作\Multiple_Humanoid_Cooperative_Transportation\dual-tron1-mujoco
$env:PYTHONPATH = "$PWD\src"

# 4. 验证
python validate_ifsm_structure.py
python examples\internal_force_suppression\demo_force_estimation.py
```

---

## 🎯 最小工作示例（无需完整安装）

如果你只想查看代码结构而不运行：

```bash
# 1. 验证文件结构
python validate_ifsm_structure.py

# 2. 查看文档
cat src/internal_force_suppression/README.md
cat QUICKSTART.md

# 3. 查看配置
cat configs/internal_force_suppression/default_ifsm.yaml

# 4. 浏览代码
ls -R src/internal_force_suppression/
```

---

## 📝 后续步骤

安装完成后：

1. **阅读文档**: `QUICKSTART.md`
2. **运行验证**: `python validate_ifsm_structure.py`
3. **查看示例**: `examples/internal_force_suppression/`
4. **开始集成**: 参考 `src/internal_force_suppression/README.md`

---

## 🆘 需要帮助？

如果安装遇到问题：

1. 检查 Python 版本（推荐 3.8-3.10）
2. 尝试使用 conda 环境
3. 查看 Pinocchio 官方文档
4. 联系项目维护者

---

**更新日期**: 2026-08-03
