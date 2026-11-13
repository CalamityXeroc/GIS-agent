#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GIS Agent 一键安装脚本
自动查找 ArcGIS Pro Python 环境并安装
"""

import os
import sys
import subprocess
from pathlib import Path


def find_arcgis_python():
    """查找 ArcGIS Pro Python 环境

    直接用 ``gis_cli.arcpy_bridge`` 的发现链：这里以前自己抄了一份候选表
    （第一条是 ``E:\ArcGISPro3.6``），升级 Pro 之后安装脚本找不到环境、
    CLI 却找得到 —— 两套路径逻辑不该同时存在。
    """
    print("🔍 查找 ArcGIS Pro Python 环境...")

    # 通常在仓库根目录直接跑，包可能还没 pip install：先把源码目录挂进 sys.path
    src_dir = Path(__file__).resolve().parent / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    try:
        from gis_cli.arcpy_bridge import discover_arcgis_pro_python

        info = discover_arcgis_pro_python()
    except Exception as exc:  # 没装 Pro / 源码缺文件都不该让安装脚本崩在 import 上
        print(f"⚠️  未找到 ArcGIS Pro Python：{exc}")
        return None

    print(f"✅ 找到: {info.python_executable}（{info.source}）")
    return info.python_executable


def check_python_version(python_exe):
    """检查 Python 版本"""
    try:
        result = subprocess.run(
            [python_exe, "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        version = result.stdout.strip() or result.stderr.strip()
        print(f"✅ Python 版本: {version}")
        return True
    except Exception as e:
        print(f"❌ Python 不可用: {e}")
        return False


def install_dependencies(python_exe):
    """安装依赖"""
    print("\n📦 安装依赖包...")
    
    # 基础依赖
    packages = [
        "typer",
        "pydantic",
        "rich",
        "openai",
    ]
    
    # 升级 pip
    print("  ⬆️  升级 pip...")
    try:
        subprocess.run(
            [python_exe, "-m", "pip", "install", "--upgrade", "pip"],
            capture_output=True,
            check=True
        )
    except:
        print("  ⚠️  pip 升级失败，继续...")
    
    # 安装依赖
    for package in packages:
        print(f"  📥 安装 {package}...")
        try:
            subprocess.run(
                [python_exe, "-m", "pip", "install", package],
                capture_output=True,
                check=True
            )
        except Exception as e:
            print(f"  ⚠️  {package} 安装失败: {e}")
    
    # 安装项目
    print("\n  📥 安装 GIS Agent...")
    try:
        subprocess.run(
            [python_exe, "-m", "pip", "install", "-e", "."],
            check=True
        )
        print("✅ GIS Agent 安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ GIS Agent 安装失败")
        return False


def save_python_path(python_exe):
    """保存 Python 路径"""
    path_file = Path(".python_path")
    path_file.write_text(python_exe)
    print(f"\n✅ Python 路径已保存到: {path_file}")


def create_launcher(python_exe):
    """创建启动脚本"""
    if sys.platform == "win32":
        # Windows 批处理
        bat_content = f'''@echo off
chcp 65001 >nul 2>&1
title GIS Agent

echo 启动 GIS Agent...
echo.

"{python_exe}" -m gis_cli.agent.cli chat

if errorlevel 1 (
    echo.
    echo [错误] 启动失败
    pause
)
'''
        launcher = Path("快速启动.bat")
        launcher.write_text(bat_content, encoding="utf-8")
        print(f"✅ 启动脚本已创建: {launcher}")
    else:
        # Unix shell
        sh_content = f'''#!/bin/bash
echo "启动 GIS Agent..."
echo

"{python_exe}" -m gis_cli.agent.cli chat
'''
        launcher = Path("快速启动.sh")
        launcher.write_text(sh_content)
        launcher.chmod(0o755)
        print(f"✅ 启动脚本已创建: {launcher}")


def main():
    """主函数"""
    print("=" * 60)
    print("  GIS Agent 一键安装")
    print("=" * 60)
    print()
    
    # 检查项目目录
    if not Path("src/gis_cli").exists():
        print("❌ 错误: 请在项目根目录运行此脚本")
        print(f"   当前目录: {Path.cwd()}")
        return 1
    
    # 查找 Python
    python_exe = find_arcgis_python()
    
    if not python_exe:
        print("\n选项:")
        print("  1. 使用系统 Python（部分功能受限）")
        print("  2. 手动指定路径")
        print("  3. 退出")
        
        choice = input("\n请选择 [1-3]: ").strip()
        
        if choice == "1":
            python_exe = sys.executable
            print(f"使用系统 Python: {python_exe}")
        elif choice == "2":
            manual_path = input("请输入 Python 路径: ").strip()
            if Path(manual_path).exists():
                python_exe = manual_path
            else:
                print("❌ 路径无效")
                return 1
        else:
            return 0
    
    # 检查 Python
    if not check_python_version(python_exe):
        return 1
    
    # 安装依赖
    if not install_dependencies(python_exe):
        print("\n❌ 安装失败")
        print("\n请尝试手动安装:")
        print(f'  {python_exe} -m pip install -e .')
        return 1
    
    # 保存配置
    save_python_path(python_exe)
    create_launcher(python_exe)
    
    # 完成
    print("\n" + "=" * 60)
    print("  ✅ 安装完成！")
    print("=" * 60)
    print(f"\nPython 路径: {python_exe}")
    print("\n下次启动请使用:")
    if sys.platform == "win32":
        print("  双击 快速启动.bat")
    else:
        print("  ./快速启动.sh")
    print("\n或直接运行:")
    print(f'  {python_exe} -m gis_cli.agent.cli chat')
    print("=" * 60)
    
    # 询问是否启动
    launch = input("\n是否立即启动 GIS Agent? [y/N]: ").strip().lower()
    if launch == 'y':
        print("\n启动中...\n")
        subprocess.run([python_exe, "-m", "gis_cli.agent.cli", "chat"])
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
