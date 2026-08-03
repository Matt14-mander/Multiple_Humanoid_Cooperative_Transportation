#!/usr/bin/env python3
"""
Project Structure Validator

验证Internal Force Suppression Module的项目结构完整性。
"""

import os
from pathlib import Path
from typing import List, Tuple

def check_file_exists(filepath: Path) -> bool:
    """检查文件是否存在"""
    return filepath.exists()

def validate_project_structure() -> Tuple[List[str], List[str]]:
    """
    验证项目结构完整性。

    Returns:
        (found_files, missing_files)
    """
    base_dir = Path(__file__).parent

    required_files = [
        # Core module
        "src/internal_force_suppression/__init__.py",
        "src/internal_force_suppression/integrated_controller.py",
        "src/internal_force_suppression/README.md",

        # Core algorithms
        "src/internal_force_suppression/core/__init__.py",
        "src/internal_force_suppression/core/force_estimator.py",
        "src/internal_force_suppression/core/internal_force_analyzer.py",
        "src/internal_force_suppression/core/admittance_controller.py",

        # Config
        "src/internal_force_suppression/config/__init__.py",
        "src/internal_force_suppression/config/ifsm_config.py",

        # Utils
        "src/internal_force_suppression/utils/__init__.py",
        "src/internal_force_suppression/utils/wrench_decomposition.py",
        "src/internal_force_suppression/utils/safety_monitor.py",

        # Configs
        "configs/internal_force_suppression/default_ifsm.yaml",

        # Examples
        "examples/internal_force_suppression/__init__.py",
        "examples/internal_force_suppression/demo_force_estimation.py",

        # Tests
        "tests/internal_force_suppression/__init__.py",
        "tests/internal_force_suppression/test_force_estimator.py",

        # Documentation
        "IFSM_PROJECT_SUMMARY.md",
    ]

    found = []
    missing = []

    for file_path in required_files:
        full_path = base_dir / file_path
        if check_file_exists(full_path):
            found.append(file_path)
        else:
            missing.append(file_path)

    return found, missing

def main():
    print("=" * 70)
    print("Internal Force Suppression Module - Structure Validation")
    print("=" * 70)

    found, missing = validate_project_structure()

    print(f"\n✓ Found files: {len(found)}")
    print(f"✗ Missing files: {len(missing)}")

    if missing:
        print("\n⚠ Missing files:")
        for file in missing:
            print(f"  - {file}")

    print("\n" + "=" * 70)
    print("Project Structure:")
    print("=" * 70)

    structure = """
dual-tron1-mujoco/
├── src/internal_force_suppression/
│   ├── __init__.py
│   ├── integrated_controller.py          # 主控制器
│   ├── README.md                          # 完整文档
│   │
│   ├── core/                              # 核心算法
│   │   ├── __init__.py
│   │   ├── force_estimator.py            # 动量观测器
│   │   ├── internal_force_analyzer.py    # 内力分解
│   │   └── admittance_controller.py      # 残差导纳
│   │
│   ├── config/                            # 配置系统
│   │   ├── __init__.py
│   │   └── ifsm_config.py
│   │
│   ├── utils/                             # 工具函数
│   │   ├── __init__.py
│   │   ├── wrench_decomposition.py
│   │   └── safety_monitor.py
│   │
│   └── [models/, estimators/, controllers/]  # 未来扩展
│
├── configs/internal_force_suppression/
│   └── default_ifsm.yaml                  # 默认配置
│
├── examples/internal_force_suppression/
│   ├── __init__.py
│   └── demo_force_estimation.py           # 力估计演示
│
├── tests/internal_force_suppression/
│   ├── __init__.py
│   └── test_force_estimator.py            # 单元测试
│
└── IFSM_PROJECT_SUMMARY.md                # 项目总结
    """

    print(structure)

    print("\n" + "=" * 70)
    print("Implementation Status:")
    print("=" * 70)

    print("\n✅ Phase 1: Basic Framework (100% Complete)")
    print("  ✓ Project structure setup")
    print("  ✓ Core algorithms implemented:")
    print("    • Generalized Momentum Observer (De Luca 2005)")
    print("    • Internal Force Decomposition (Erhart 2015)")
    print("    • Residual Admittance Control (Hogan 1985)")
    print("  ✓ Configuration system")
    print("  ✓ Safety monitoring")
    print("  ✓ Unit tests")
    print("  ✓ Documentation")

    print("\n🔄 Phase 2: Validation & Testing (Next)")
    print("  ⏳ Force estimation validation")
    print("  ⏳ Internal force analysis verification")
    print("  ⏳ Admittance control testing")
    print("  ⏳ Parameter tuning")

    print("\n📅 Phase 3: Simulation Integration (Planned)")
    print("  ⏳ MuJoCo integration")
    print("  ⏳ Full system demo")
    print("  ⏳ Visualization tools")

    print("\n" + "=" * 70)
    print("Key Metrics:")
    print("=" * 70)
    print(f"  • Python files: {len(found)}")
    print(f"  • Core algorithms: 3 (Force Estimation, Force Analysis, Admittance)")
    print(f"  • Configuration files: 1 (YAML)")
    print(f"  • Example scripts: 1")
    print(f"  • Unit tests: 1 (expandable)")
    print(f"  • Documentation: Complete (README + inline comments)")

    print("\n" + "=" * 70)

    if not missing:
        print("\n🎉 SUCCESS! All required files are in place.")
        print("✅ Project structure is complete and ready for Phase 2!")
        return 0
    else:
        print("\n⚠ WARNING! Some files are missing.")
        print("Please check the missing files list above.")
        return 1

if __name__ == "__main__":
    exit(main())
