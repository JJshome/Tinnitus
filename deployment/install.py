#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
이명 치료 시스템 설치 스크립트

이 스크립트는 이명 치료 시스템의 설치 및 초기 설정을 자동화합니다.
"""

import os
import sys
import platform
import subprocess
import shutil
import logging
import json
import argparse
from pathlib import Path
import pkg_resources

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("install.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("installer")

# 경로 설정
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"

# 설치 구성
DEFAULT_CONFIG = {
    "audio": {
        "sample_rate": 44100,
        "channels": 2,
        "device_id": "default"
    },
    "eeg": {
        "device_type": "auto",
        "channels": 32,
        "sample_rate": 256
    },
    "visual": {
        "enabled": True,
        "device_id": "auto"
    },
    "haptic": {
        "enabled": True,
        "port": "auto"
    },
    "system": {
        "data_dir": str(DATA_DIR),
        "log_level": "INFO",
        "use_gpu": True
    },
    "user_interface": {
        "theme": "dark",
        "language": "ko"
    }
}

def check_python_version():
    """파이썬 버전 확인"""
    logger.info(f"Python 버전 확인: {sys.version}")
    major, minor, _ = platform.python_version_tuple()
    if int(major) < 3 or (int(major) == 3 and int(minor) < 9):
        logger.error("Python 3.9 이상이 필요합니다.")
        return False
    return True

def check_system_compatibility():
    """시스템 호환성 확인"""
    system = platform.system()
    logger.info(f"운영체제 확인: {system} {platform.release()}")
    
    if system == "Windows" and int(platform.release()) < 10:
        logger.warning("Windows 10 이상을 권장합니다.")
    
    if system == "Darwin":  # macOS
        version = platform.mac_ver()[0]
        if tuple(map(int, version.split('.'))) < (11, 0):
            logger.warning("macOS 11.0(Big Sur) 이상을 권장합니다.")
    
    if system == "Linux":
        # 리눅스 배포판 확인
        try:
            with open('/etc/os-release') as f:
                os_info = {}
                for line in f:
                    if '=' in line:
                        key, value = line.rstrip().split('=', 1)
                        os_info[key] = value.strip('"')
            
            logger.info(f"Linux 배포판: {os_info.get('NAME', 'Unknown')} {os_info.get('VERSION_ID', 'Unknown')}")
        except:
            logger.warning("Linux 배포판을 확인할 수 없습니다.")
    
    # 메모리 확인
    import psutil
    total_memory = psutil.virtual_memory().total / (1024**3)  # GB 단위
    logger.info(f"시스템 메모리: {total_memory:.2f} GB")
    if total_memory < 8:
        logger.warning("8GB 이상의 메모리를 권장합니다.")
    
    # CPU 확인
    cpu_info = platform.processor()
    logger.info(f"CPU: {cpu_info}")
    
    return True

def install_dependencies():
    """의존성 패키지 설치"""
    requirements_file = SCRIPT_DIR / "requirements.txt"
    if not requirements_file.exists():
        logger.error("requirements.txt 파일을 찾을 수 없습니다.")
        return False
    
    logger.info("의존성 패키지 설치 중...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
        logger.info("의존성 패키지 설치 완료")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"패키지 설치 중 오류 발생: {e}")
        return False

def check_hardware():
    """하드웨어 연결 확인"""
    logger.info("하드웨어 연결 확인 중...")
    
    # 오디오 장치 확인
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        logger.info(f"발견된 오디오 장치: {len(devices)}")
        for i, device in enumerate(devices):
            logger.info(f"[{i}] {device['name']}")
    except Exception as e:
        logger.warning(f"오디오 장치 확인 중 오류: {e}")
    
    # 시리얼 포트 확인 (촉각 자극 장치)
    try:
        import serial.tools.list_ports
        ports = list(serial.tools.list_ports.comports())
        logger.info(f"발견된 시리얼 포트: {len(ports)}")
        for port in ports:
            logger.info(f"{port.device} - {port.description}")
    except Exception as e:
        logger.warning(f"시리얼 포트 확인 중 오류: {e}")
    
    # GPU 확인
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        logger.info(f"CUDA 사용 가능: {gpu_available}")
        if gpu_available:
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    except Exception as e:
        logger.warning(f"GPU 확인 중 오류: {e}")
    
    return True

def create_config():
    """초기 설정 파일 생성"""
    if not CONFIG_DIR.exists():
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    
    config_file = CONFIG_DIR / "system_config.json"
    
    # 이미 설정 파일이 있으면 백업
    if config_file.exists():
        backup_file = CONFIG_DIR / f"system_config.json.bak"
        shutil.copy2(config_file, backup_file)
        logger.info(f"기존 설정 파일 백업: {backup_file}")
    
    # 설정 파일 생성
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(DEFAULT_CONFIG, f, indent=2, ensure_ascii=False)
    
    logger.info(f"설정 파일 생성 완료: {config_file}")
    return True

def create_directories():
    """필요한 디렉토리 생성"""
    dirs = [
        DATA_DIR,
        DATA_DIR / "patients",
        DATA_DIR / "recordings",
        DATA_DIR / "models",
        DATA_DIR / "logs",
        DATA_DIR / "stimuli",
        DATA_DIR / "export"
    ]
    
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"디렉토리 생성: {directory}")
    
    return True

def finalize_installation():
    """설치 마무리"""
    logger.info("설치 완료!")
    logger.info("\n===== 이명 치료 시스템 설치 요약 =====")
    logger.info(f"설치 경로: {PROJECT_ROOT}")
    logger.info(f"설정 파일: {CONFIG_DIR / 'system_config.json'}")
    logger.info(f"데이터 경로: {DATA_DIR}")
    logger.info("\n시스템을 시작하려면 다음 명령어를 실행하세요:")
    logger.info(f"python {PROJECT_ROOT / 'src' / 'main.py'}")
    logger.info("=====================================")
    return True

def verify_installation():
    """설치 확인"""
    try:
        # 필수 패키지 로드 테스트
        import numpy
        import scipy
        import torch
        import pandas
        import matplotlib
        
        # 설정 파일 확인
        config_file = CONFIG_DIR / "system_config.json"
        if not config_file.exists():
            logger.error("설정 파일이 존재하지 않습니다.")
            return False
        
        # 데이터 디렉토리 확인
        if not DATA_DIR.exists():
            logger.error("데이터 디렉토리가 존재하지 않습니다.")
            return False
        
        logger.info("설치 검증 완료")
        return True
    except ImportError as e:
        logger.error(f"패키지 로드 실패: {e}")
        return False
    except Exception as e:
        logger.error(f"설치 검증 중 오류 발생: {e}")
        return False

def main():
    """설치 메인 함수"""
    parser = argparse.ArgumentParser(description="이명 치료 시스템 설치 스크립트")
    parser.add_argument("--skip-deps", action="store_true", help="의존성 패키지 설치 건너뛰기")
    parser.add_argument("--skip-hw-check", action="store_true", help="하드웨어 확인 건너뛰기")
    args = parser.parse_args()
    
    logger.info("==== 이명 치료 시스템 설치 시작 ====")
    
    steps = [
        ("파이썬 버전 확인", check_python_version),
        ("시스템 호환성 확인", check_system_compatibility),
        ("의존성 패키지 설치", lambda: True if args.skip_deps else install_dependencies),
        ("하드웨어 연결 확인", lambda: True if args.skip_hw_check else check_hardware),
        ("초기 설정 파일 생성", create_config),
        ("디렉토리 구조 생성", create_directories),
        ("설치 검증", verify_installation),
        ("설치 마무리", finalize_installation)
    ]
    
    success = True
    for step_name, step_func in steps:
        logger.info(f"\n=== {step_name} ===")
        try:
            if not step_func():
                logger.error(f"{step_name} 실패")
                success = False
                break
        except Exception as e:
            logger.error(f"{step_name} 중 오류 발생: {e}", exc_info=True)
            success = False
            break
    
    if success:
        logger.info("\n설치가 성공적으로 완료되었습니다!")
        return 0
    else:
        logger.error("\n설치 중 오류가 발생했습니다. 로그 파일을 확인하세요.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
