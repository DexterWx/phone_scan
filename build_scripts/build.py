#!/usr/bin/env python3

import sys
import os
import subprocess
import re
from pathlib import Path

# 统一构建脚本 - 跨平台兼容
# 使用方法: python build.py [platform]
# 平台: windows, android, ios

def main():
    platform = sys.argv[1] if len(sys.argv) > 1 else 'windows'
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    print('========================================')
    print('Phone Scan Build Script')
    print('========================================')
    print(f'Platform: {platform}')
    print('Build Type: release')
    print('========================================')
    
    os.chdir(project_root)
    
    def run_command(cmd):
        print(f'Executing: {cmd}')
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f'Command failed with return code {result.returncode}')
            if result.stderr:
                print(f'Error: {result.stderr}')
            raise subprocess.CalledProcessError(result.returncode, cmd)
    
    def load_env_from_powershell(script_path):
        """从PowerShell脚本加载环境变量"""
        # 创建一个临时的PowerShell脚本文件来获取环境变量
        temp_script = script_dir / "temp_env.ps1"
        with open(temp_script, 'w', encoding='utf-8') as f:
            f.write(f'''
# 执行原始脚本
& "{script_path}"

# 输出环境变量
Get-ChildItem Env: | Where-Object {{$_.Name -match "OPENCV|LIBCLANG|ANDROID|PATH"}} | ForEach-Object {{ 
    Write-Output "$($_.Name)=$($_.Value)"
}}
''')
        
        try:
            cmd = f'"{ps_paths[0]}" -ExecutionPolicy Bypass -File "{temp_script}"'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f'PowerShell script failed: {result.stderr}')
                raise subprocess.CalledProcessError(result.returncode, cmd)
            
            # 解析环境变量并设置到当前进程
            for line in result.stdout.strip().split('\n'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    # 验证环境变量名是否有效（只包含字母、数字、下划线）
                    if key and re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', key):
                        try:
                            os.environ[key] = value
                            # 只显示重要的环境变量
                            if key.startswith(('OPENCV', 'LIBCLANG', 'ANDROID')):
                                print(f'Set {key}={value}')
                        except ValueError as e:
                            print(f'Skipped invalid environment variable: {key} (error: {e})')
        finally:
            # 清理临时文件
            if temp_script.exists():
                temp_script.unlink()
    
    if platform == 'windows':
        print('Setting up Windows environment...')
        if os.name == 'nt':
            # Windows系统，使用Windows PowerShell的完整路径
            ps_paths = [
                'C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe',
                'powershell.exe',
                'pwsh'
            ]
            # 先尝试加载环境变量
            try:
                load_env_from_powershell(f'{script_dir}\\windows_env.ps1')
            except subprocess.CalledProcessError:
                print('Failed to load environment variables from PowerShell script')
                sys.exit(1)
        else:
            run_command(f'pwsh -ExecutionPolicy Bypass -File "{script_dir}/windows_env.ps1"')
        run_command('cargo build --release')
        
    elif platform == 'android':
        print('Setting up Android environment...')
        if os.name == 'nt':
            # Windows系统，使用Windows PowerShell的完整路径
            ps_paths = [
                'C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe',
                'powershell.exe',
                'pwsh'
            ]
            # 先尝试加载环境变量
            try:
                load_env_from_powershell(f'{script_dir}\\android_env.ps1')
            except subprocess.CalledProcessError:
                print('Failed to load environment variables from PowerShell script')
                sys.exit(1)
        else:
            run_command(f'pwsh -ExecutionPolicy Bypass -File "{script_dir}/android_env.ps1"')
        run_command('cargo ndk -t arm64-v8a build --release')
        
    elif platform == 'ios':
        print('Setting up iOS environment...')
        if sys.platform == 'darwin':
            run_command(f'bash -c "source {script_dir}/ios_env.sh && cargo build --release --target aarch64-apple-ios"')
        else:
            print('Error: iOS builds can only be performed on macOS')
            sys.exit(1)
    else:
        print(f'Error: Unknown platform "{platform}"')
        print('Supported platforms: windows, android, ios')
        sys.exit(1)
    
    print('Build completed successfully!')

if __name__ == '__main__':
    main()
