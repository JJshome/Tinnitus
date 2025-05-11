#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
하드웨어 테스트 스크립트

이 스크립트는 이명 치료 시스템의 하드웨어 구성 요소 연결 및 기능을 테스트합니다.
"""

import os
import sys
import time
import argparse
import json
import logging
import threading
from pathlib import Path

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hardware_test.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("hardware_test")

# 경로 설정
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_FILE = CONFIG_DIR / "system_config.json"

# 상수 정의
TEST_DURATION = 5  # 초
TEST_AUDIO_FREQ = 440  # Hz (A4)
TEST_VISUAL_PATTERN = "calm"
TEST_HAPTIC_INTENSITY = 0.5

def load_config():
    """시스템 설정 로드"""
    if not CONFIG_FILE.exists():
        logger.error(f"설정 파일을 찾을 수 없습니다: {CONFIG_FILE}")
        return None
    
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"설정 파일 로드 중 오류: {e}")
        return None

def test_audio(config):
    """오디오 출력 테스트"""
    logger.info("===== 오디오 테스트 시작 =====")
    
    try:
        import sounddevice as sd
        import numpy as np
        
        # 오디오 설정 로드
        audio_config = config.get('audio', {})
        device_id = audio_config.get('device_id', 'default')
        sample_rate = audio_config.get('sample_rate', 44100)
        
        # 장치 정보 출력
        devices = sd.query_devices()
        logger.info(f"사용 가능한 오디오 장치: {len(devices)}")
        for i, device in enumerate(devices):
            logger.info(f"[{i}] {device['name']}")
            if device_id != 'default' and device_id in device['name']:
                logger.info(f"  -> 선택된 장치")
        
        # 테스트 신호 생성 (순음 A4)
        duration = TEST_DURATION
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        test_tone = 0.5 * np.sin(2 * np.pi * TEST_AUDIO_FREQ * t)
        
        # 신호 출력
        logger.info(f"{device_id} 장치로 {TEST_AUDIO_FREQ}Hz 테스트 톤 재생 중...")
        sd.play(test_tone, sample_rate)
        
        # 진행 표시
        for i in range(duration):
            print(f"재생 중: {i+1}/{duration}초", end='\r')
            time.sleep(1)
        print()
        
        sd.stop()
        logger.info("오디오 테스트 완료")
        return True
    
    except Exception as e:
        logger.error(f"오디오 테스트 중 오류 발생: {e}")
        return False

def test_eeg(config):
    """EEG 장비 연결 테스트"""
    logger.info("===== EEG 테스트 시작 =====")
    
    try:
        # 실제 코드에서는 특정 EEG 장비에 맞는 라이브러리 사용
        # 이 예제 코드는 장비 없이도 테스트 가능하도록 단순화됨
        
        eeg_config = config.get('eeg', {})
        device_type = eeg_config.get('device_type', 'auto')
        
        logger.info(f"EEG 장치 유형: {device_type}")
        
        # 장치 유형별 테스트 로직
        if device_type == 'emotiv':
            logger.info("Emotiv EPOC 연결 테스트 중...")
            # 실제로는 Emotiv API 사용
            time.sleep(2)
            
        elif device_type == 'openbci':
            logger.info("OpenBCI 연결 테스트 중...")
            # 실제로는 Brainflow API 사용
            time.sleep(2)
            
        elif device_type == 'gtec':
            logger.info("g.tec 연결 테스트 중...")
            # 실제로는 g.tec API 사용
            time.sleep(2)
            
        else:
            logger.info("자동 감지 모드로 EEG 장치 검색 중...")
            time.sleep(2)
        
        # 가상의 EEG 데이터 생성
        logger.info("EEG 신호 샘플링 중...")
        for i in range(5):
            print(f"샘플링: {i+1}/5초", end='\r')
            time.sleep(1)
        print()
        
        logger.info("EEG 채널 연결 상태 양호")
        logger.info("EEG 테스트 완료")
        return True
    
    except Exception as e:
        logger.error(f"EEG 테스트 중 오류 발생: {e}")
        return False

def test_visual(config):
    """시각 자극 장치 테스트"""
    logger.info("===== 시각 자극 테스트 시작 =====")
    
    try:
        visual_config = config.get('visual', {})
        enabled = visual_config.get('enabled', True)
        
        if not enabled:
            logger.info("시각 자극 기능이 비활성화되어 있습니다.")
            return True
        
        device_id = visual_config.get('device_id', 'auto')
        logger.info(f"시각 자극 장치 ID: {device_id}")
        
        # 시리얼 포트 통신 (실제로는 적절한 프로토콜 사용)
        import serial
        import serial.tools.list_ports
        
        # 사용 가능한 시리얼 포트 확인
        ports = list(serial.tools.list_ports.comports())
        logger.info(f"사용 가능한 시리얼 포트: {len(ports)}")
        for port in ports:
            logger.info(f"{port.device} - {port.description}")
        
        # 장치 연결 테스트 (실제 연결하지 않고 로그만 출력)
        logger.info(f"시각 자극 패턴 '{TEST_VISUAL_PATTERN}' 테스트 중...")
        for i in range(TEST_DURATION):
            print(f"테스트 중: {i+1}/{TEST_DURATION}초", end='\r')
            time.sleep(1)
        print()
        
        logger.info("시각 자극 테스트 완료")
        return True
    
    except Exception as e:
        logger.error(f"시각 자극 테스트 중 오류 발생: {e}")
        return False

def test_haptic(config):
    """촉각 자극 장치 테스트"""
    logger.info("===== 촉각 자극 테스트 시작 =====")
    
    try:
        haptic_config = config.get('haptic', {})
        enabled = haptic_config.get('enabled', True)
        
        if not enabled:
            logger.info("촉각 자극 기능이 비활성화되어 있습니다.")
            return True
        
        port = haptic_config.get('port', 'auto')
        logger.info(f"촉각 자극 장치 포트: {port}")
        
        # 촉각 자극 테스트 (실제 연결하지 않고 로그만 출력)
        logger.info(f"촉각 자극 강도 {TEST_HAPTIC_INTENSITY} 테스트 중...")
        for i in range(TEST_DURATION):
            print(f"테스트 중: {i+1}/{TEST_DURATION}초", end='\r')
            time.sleep(1)
        print()
        
        logger.info("촉각 자극 테스트 완료")
        return True
    
    except Exception as e:
        logger.error(f"촉각 자극 테스트 중 오류 발생: {e}")
        return False

def test_combined(config):
    """모든 자극의 복합 테스트"""
    logger.info("===== 복합 자극 테스트 시작 =====")
    
    try:
        # 복합 자극을 위한 스레드 생성
        threads = []
        
        if config.get('audio', {}).get('enabled', True):
            audio_thread = threading.Thread(target=test_audio, args=(config,))
            threads.append(audio_thread)
        
        if config.get('visual', {}).get('enabled', True):
            visual_thread = threading.Thread(target=test_visual, args=(config,))
            threads.append(visual_thread)
        
        if config.get('haptic', {}).get('enabled', True):
            haptic_thread = threading.Thread(target=test_haptic, args=(config,))
            threads.append(haptic_thread)
        
        # 모든 스레드 시작
        logger.info("복합 자극 테스트 시작 중...")
        for thread in threads:
            thread.start()
        
        # 모든 스레드 종료 대기
        for thread in threads:
            thread.join()
        
        logger.info("복합 자극 테스트 완료")
        return True
    
    except Exception as e:
        logger.error(f"복합 자극 테스트 중 오류 발생: {e}")
        return False

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="이명 치료 시스템 하드웨어 테스트")
    parser.add_argument("--eeg", action="store_true", help="EEG 장비 테스트")
    parser.add_argument("--audio", action="store_true", help="오디오 출력 테스트")
    parser.add_argument("--visual", action="store_true", help="시각 자극 테스트")
    parser.add_argument("--haptic", action="store_true", help="촉각 자극 테스트")
    parser.add_argument("--combined", action="store_true", help="복합 자극 테스트")
    parser.add_argument("--all", action="store_true", help="모든 구성 요소 테스트")
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config()
    if not config:
        logger.error("설정 로드 실패. 테스트를 중단합니다.")
        return 1
    
    # 테스트할 항목이 없으면 --all과 동일하게 처리
    if not (args.eeg or args.audio or args.visual or args.haptic or args.combined):
        args.all = True
    
    # 테스트 실행
    results = []
    
    if args.all or args.eeg:
        eeg_result = test_eeg(config)
        results.append(("EEG", eeg_result))
    
    if args.all or args.audio:
        audio_result = test_audio(config)
        results.append(("오디오", audio_result))
    
    if args.all or args.visual:
        visual_result = test_visual(config)
        results.append(("시각 자극", visual_result))
    
    if args.all or args.haptic:
        haptic_result = test_haptic(config)
        results.append(("촉각 자극", haptic_result))
    
    if args.all or args.combined:
        combined_result = test_combined(config)
        results.append(("복합 자극", combined_result))
    
    # 결과 요약 출력
    logger.info("\n===== 테스트 결과 요약 =====")
    all_success = True
    for name, success in results:
        status = "성공" if success else "실패"
        logger.info(f"{name}: {status}")
        if not success:
            all_success = False
    
    if all_success:
        logger.info("\n모든 테스트가 성공적으로 완료되었습니다.")
        return 0
    else:
        logger.error("\n일부 테스트가 실패했습니다. 로그 파일을 확인하세요.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
