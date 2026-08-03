"""
诊断脚本 - 检查模块导入问题
"""

import sys
from pathlib import Path

print("=" * 70)
print("Python 环境诊断")
print("=" * 70)

print(f"\nPython 可执行文件: {sys.executable}")
print(f"Python 版本: {sys.version}")

print("\n当前 sys.path:")
for i, p in enumerate(sys.path):
    print(f"  [{i}] {p}")

print("\n" + "=" * 70)
print("尝试导入模块")
print("=" * 70)

# 尝试1: 直接导入
print("\n1. 尝试直接导入...")
try:
    import internal_force_suppression
    print("   ✓ 成功导入 internal_force_suppression")
    print(f"   模块位置: {internal_force_suppression.__file__}")
    print(f"   版本: {internal_force_suppression.__version__}")
except ImportError as e:
    print(f"   ✗ 失败: {e}")

# 尝试2: 导入子模块
print("\n2. 尝试导入核心模块...")
try:
    from internal_force_suppression.core import force_estimator
    print("   ✓ 成功导入 force_estimator")
except ImportError as e:
    print(f"   ✗ 失败: {e}")

# 尝试3: 手动添加路径后导入
print("\n3. 手动添加 src 到路径后尝试...")
src_path = Path(__file__).parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))
    print(f"   添加路径: {src_path}")
    try:
        import internal_force_suppression
        print("   ✓ 成功导入 internal_force_suppression")
    except ImportError as e:
        print(f"   ✗ 仍然失败: {e}")
else:
    print(f"   ✗ src 目录不存在: {src_path}")

# 检查文件结构
print("\n" + "=" * 70)
print("检查文件结构")
print("=" * 70)

src_ifsm = Path(__file__).parent / "src" / "internal_force_suppression"
if src_ifsm.exists():
    print(f"\n✓ 找到模块目录: {src_ifsm}")
    print("\n文件列表:")
    for item in src_ifsm.rglob("*.py"):
        rel_path = item.relative_to(src_ifsm.parent)
        print(f"  {rel_path}")
else:
    print(f"\n✗ 未找到模块目录: {src_ifsm}")

print("\n" + "=" * 70)
print("建议")
print("=" * 70)

print("""
如果导入失败，请执行以下命令之一：

方法1 - 安装为可编辑包（推荐）:
    pip install -e .

方法2 - 设置 PYTHONPATH（临时）:
    # PowerShell
    $env:PYTHONPATH = "$PWD\\src"

    # 然后运行 pytest
    python -m pytest tests/internal_force_suppression/test_basic.py -v

方法3 - 直接运行（不用pytest）:
    # 添加路径到测试脚本开头
    python tests/internal_force_suppression/test_basic.py
""")

print("=" * 70)
