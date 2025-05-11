"""
복합 자극 출력 모듈

이 모듈은 청각, 시각, 촉각 자극을 동기화하여 출력하는 기능을 제공합니다.
하드웨어 장치와의 인터페이스 및 자극 동기화 메커니즘을 담당합니다.
"""

import numpy as np
import time
import threading
import logging
import json
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import os
import soundfile as sf
import matplotlib.pyplot as plt
import serial
import socket
import bluetooth

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/stimulus_output.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("stimulus_output")


@dataclass
class OutputConfig:
    """출력 모듈 설정"""
    audio_device: str = "default"  # 오디오 출력 장치
    bone_conduction: bool = True  # 골전도 헤드폰 사용 여부
    eyemask_device_id: str = ""  # 스마트 안대 장치 ID
    vibration_device_port: str = ""  # 진동 장치 시리얼 포트
    audio_latency: float = 0.02  # 오디오 출력 지연 시간 (초)
    visual_latency: float = 0.05  # 시각 출력 지연 시간 (초)
    tactile_latency: float = 0.03  # 촉각 출력 지연 시간 (초)
    sync_tolerance: float = 0.01  # 동기화 허용 오차 (초)
    output_dir: str = "output/data"  # 출력 데이터 저장 디렉토리


class AudioOutputModule:
    """청각 자극 출력 모듈"""
    
    def __init__(self, config: OutputConfig):
        """
        초기화
        
        Args:
            config: 출력 모듈 설정
        """
        self.config = config
        self.current_playback = None
        self.is_playing = False
        self.playback_thread = None
        
        # 오디오 출력 장치 초기화
        try:
            import sounddevice as sd
            self.sd = sd
            self.sd.query_devices()
            logger.info(f"Audio output initialized with device: {config.audio_device}")
        except Exception as e:
            logger.error(f"Failed to initialize audio output: {e}")
            self.sd = None
    
    def play_audio(self, audio_data: np.ndarray, sample_rate: int = 44100, 
                 blocking: bool = False) -> bool:
        """
        오디오 자극 재생
        
        Args:
            audio_data: 오디오 데이터 (샘플 × 채널)
            sample_rate: 샘플링 레이트 (Hz)
            blocking: 블로킹 모드 여부
            
        Returns:
            성공 여부
        """
        if self.sd is None:
            logger.error("Audio output not available")
            return False
        
        # 이전 재생 중지
        self.stop_audio()
        
        # 재생 상태 업데이트
        self.is_playing = True
        self.current_playback = audio_data
        
        # 블로킹 모드에서는 직접 재생
        if blocking:
            try:
                self.sd.play(audio_data, sample_rate, device=self.config.audio_device)
                self.sd.wait()
                self.is_playing = False
                return True
            except Exception as e:
                logger.error(f"Error playing audio: {e}")
                self.is_playing = False
                return False
        
        # 비블로킹 모드에서는 스레드 생성
        def play_thread():
            try:
                self.sd.play(audio_data, sample_rate, device=self.config.audio_device)
                self.sd.wait()
            except Exception as e:
                logger.error(f"Error in audio playback thread: {e}")
            finally:
                self.is_playing = False
        
        self.playback_thread = threading.Thread(target=play_thread)
        self.playback_thread.daemon = True
        self.playback_thread.start()
        
        # 지연 시간만큼 대기 (하드웨어 지연 보상)
        time.sleep(self.config.audio_latency)
        
        return True
    
    def stop_audio(self) -> bool:
        """
        오디오 재생 중지
        
        Returns:
            성공 여부
        """
        if self.sd is None:
            return False
        
        if self.is_playing:
            try:
                self.sd.stop()
                self.is_playing = False
                if self.playback_thread and self.playback_thread.is_alive():
                    self.playback_thread.join(0.5)  # 최대 0.5초 대기
                return True
            except Exception as e:
                logger.error(f"Error stopping audio: {e}")
                return False
        
        return True
    
    def set_volume(self, volume: float) -> bool:
        """
        볼륨 설정
        
        Args:
            volume: 볼륨 (0.0 ~ 1.0)
            
        Returns:
            성공 여부
        """
        if self.sd is None:
            return False
        
        # 시스템 의존적인 볼륨 제어 필요
        # 여기서는 시스템 의존성을 피하기 위해 다음 재생 시 적용할 볼륨 저장
        self.volume = max(0.0, min(1.0, volume))
        logger.info(f"Volume set to {self.volume:.2f}")
        
        return True


class VisualOutputModule:
    """시각 자극 출력 모듈"""
    
    def __init__(self, config: OutputConfig):
        """
        초기화
        
        Args:
            config: 출력 모듈 설정
        """
        self.config = config
        self.is_connected = False
        self.device = None
        self.current_pattern = None
        
        # 스마트 안대 연결 시도
        self.connect_device()
    
    def connect_device(self) -> bool:
        """
        스마트 안대 장치 연결
        
        Returns:
            연결 성공 여부
        """
        if not self.config.eyemask_device_id:
            logger.warning("No eyemask device ID configured")
            return False
        
        try:
            # 블루투스 연결 (실제 구현은 하드웨어에 따라 다름)
            # 여기서는 가상 연결로 대체
            self.device = {
                "id": self.config.eyemask_device_id,
                "connected": True,
                "brightness": 0.5,
                "pattern": "none",
                "color": (255, 255, 255)  # RGB
            }
            self.is_connected = True
            logger.info(f"Connected to eyemask device: {self.config.eyemask_device_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to eyemask device: {e}")
            self.is_connected = False
            return False
    
    def display_pattern(self, pattern: str, brightness: float = 0.5, 
                      color: Tuple[int, int, int] = (255, 255, 255),
                      duration: float = 0) -> bool:
        """
        시각 패턴 표시
        
        Args:
            pattern: 패턴 이름 ('calm', 'transient', 'subtle', 'off' 등)
            brightness: 밝기 (0.0 ~ 1.0)
            color: RGB 색상 (0-255)
            duration: 지속 시간 (초), 0이면 무기한
            
        Returns:
            성공 여부
        """
        if not self.is_connected:
            logger.error("Eyemask device not connected")
            return False
        
        try:
            # 패턴 설정
            self.device["pattern"] = pattern
            self.device["brightness"] = max(0.0, min(1.0, brightness))
            self.device["color"] = color
            
            # 명령 전송 (실제 구현은 하드웨어에 따라 다름)
            # 여기서는 로깅으로 대체
            logger.info(f"Display pattern: {pattern}, brightness: {brightness:.2f}, color: {color}")
            
            # 현재 패턴 저장
            self.current_pattern = {
                "pattern": pattern,
                "brightness": brightness,
                "color": color
            }
            
            # 지연 시간만큼 대기 (하드웨어 지연 보상)
            time.sleep(self.config.visual_latency)
            
            # 지정된 지속 시간이 있으면 타이머 설정
            if duration > 0:
                timer = threading.Timer(duration, self.turn_off)
                timer.daemon = True
                timer.start()
            
            return True
        except Exception as e:
            logger.error(f"Error displaying pattern: {e}")
            return False
    
    def turn_off(self) -> bool:
        """
        시각 자극 끄기
        
        Returns:
            성공 여부
        """
        return self.display_pattern("off", brightness=0, color=(0, 0, 0))
    
    def set_brightness(self, brightness: float) -> bool:
        """
        밝기 설정
        
        Args:
            brightness: 밝기 (0.0 ~ 1.0)
            
        Returns:
            성공 여부
        """
        if not self.is_connected or not self.current_pattern:
            return False
        
        return self.display_pattern(
            self.current_pattern["pattern"],
            brightness,
            self.current_pattern["color"]
        )


class TactileOutputModule:
    """촉각 자극 출력 모듈"""
    
    def __init__(self, config: OutputConfig):
        """
        초기화
        
        Args:
            config: 출력 모듈 설정
        """
        self.config = config
        self.is_connected = False
        self.device = None
        self.current_vibration = None
        
        # 진동 장치 연결 시도
        self.connect_device()
    
    def connect_device(self) -> bool:
        """
        진동 장치 연결
        
        Returns:
            연결 성공 여부
        """
        if not self.config.vibration_device_port:
            logger.warning("No vibration device port configured")
            return False
        
        try:
            # 시리얼 연결 (실제 구현은 하드웨어에 따라 다름)
            # 여기서는 가상 연결로 대체
            self.device = {
                "port": self.config.vibration_device_port,
                "connected": True,
                "intensity": 0.0,
                "frequency": 0.0
            }
            self.is_connected = True
            logger.info(f"Connected to vibration device: {self.config.vibration_device_port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to vibration device: {e}")
            self.is_connected = False
            return False
    
    def apply_vibration(self, intensity: float, frequency: float, 
                      pattern: str = "continuous", duration: float = 0) -> bool:
        """
        진동 적용
        
        Args:
            intensity: 진동 강도 (0.0 ~ 1.0)
            frequency: 진동 주파수 (Hz)
            pattern: 진동 패턴 ('continuous', 'pulse', 'irregular' 등)
            duration: 지속 시간 (초), 0이면 무기한
            
        Returns:
            성공 여부
        """
        if not self.is_connected:
            logger.error("Vibration device not connected")
            return False
        
        try:
            # 진동 설정
            self.device["intensity"] = max(0.0, min(1.0, intensity))
            
            # 'irregular' 패턴인 경우 랜덤 주파수 변동 추가
            if pattern == "irregular":
                base_frequency = frequency
                # 실제 구현에서는 여기서 불규칙한 진동 패턴 생성
                freq_variation = f"irregular({base_frequency:.1f}Hz)"
                self.device["frequency"] = freq_variation
            else:
                self.device["frequency"] = frequency
            
            # 명령 전송 (실제 구현은 하드웨어에 따라 다름)
            # 여기서는 로깅으로 대체
            logger.info(f"Apply vibration: intensity: {intensity:.2f}, frequency: {frequency:.1f}Hz, pattern: {pattern}")
            
            # 현재 진동 저장
            self.current_vibration = {
                "intensity": intensity,
                "frequency": frequency,
                "pattern": pattern
            }
            
            # 지연 시간만큼 대기 (하드웨어 지연 보상)
            time.sleep(self.config.tactile_latency)
            
            # 지정된 지속 시간이 있으면 타이머 설정
            if duration > 0:
                timer = threading.Timer(duration, self.stop_vibration)
                timer.daemon = True
                timer.start()
            
            return True
        except Exception as e:
            logger.error(f"Error applying vibration: {e}")
            return False
    
    def stop_vibration(self) -> bool:
        """
        진동 중지
        
        Returns:
            성공 여부
        """
        return self.apply_vibration(0.0, 0.0, "continuous")
    
    def set_intensity(self, intensity: float) -> bool:
        """
        진동 강도 설정
        
        Args:
            intensity: 진동 강도 (0.0 ~ 1.0)
            
        Returns:
            성공 여부
        """
        if not self.is_connected or not self.current_vibration:
            return False
        
        return self.apply_vibration(
            intensity,
            self.current_vibration["frequency"],
            self.current_vibration["pattern"]
        )


class StimulusOutputManager:
    """복합 자극 출력 관리자"""
    
    def __init__(self, config: OutputConfig = None):
        """
        초기화
        
        Args:
            config: 출력 모듈 설정. None인 경우 기본값 사용
        """
        self.config = config or OutputConfig()
        
        # 출력 디렉토리 생성
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # 모듈 초기화
        self.audio_module = AudioOutputModule(self.config)
        self.visual_module = VisualOutputModule(self.config)
        self.tactile_module = TactileOutputModule(self.config)
        
        # 자극 상태
        self.current_stimuli = None
        self.stimuli_start_time = None
    
    def apply_stimuli(self, stimuli: Dict, sync: bool = True) -> bool:
        """
        복합 자극 적용
        
        Args:
            stimuli: 자극 설정 딕셔너리
                {
                    'audio': {'data': np.ndarray, 'sample_rate': int, ...},
                    'visual': {'pattern': str, 'brightness': float, ...},
                    'tactile': {'intensity': float, 'frequency': float, ...}
                }
            sync: 동기화 모드 여부
            
        Returns:
            성공 여부
        """
        self.current_stimuli = stimuli
        self.stimuli_start_time = time.time()
        success = True
        
        # 각 모듈별 자극 적용
        if 'audio' in stimuli:
            audio_success = self._apply_audio_stimulus(stimuli['audio'])
            success = success and audio_success
        
        if 'visual' in stimuli:
            visual_success = self._apply_visual_stimulus(stimuli['visual'])
            success = success and visual_success
        
        if 'tactile' in stimuli:
            tactile_success = self._apply_tactile_stimulus(stimuli['tactile'])
            success = success and tactile_success
        
        # 동기화 모드에서는 모든 자극이 성공적으로 적용될 때까지 대기
        if sync and success:
            # 동기화 검증 (실제 구현에서는 하드웨어 피드백 필요)
            # 여기서는 단순히 지연 시간으로 근사
            max_latency = max(
                self.config.audio_latency,
                self.config.visual_latency,
                self.config.tactile_latency
            )
            time.sleep(max_latency + self.config.sync_tolerance)
        
        return success
    
    def _apply_audio_stimulus(self, audio_stimulus: Dict) -> bool:
        """
        청각 자극 적용
        
        Args:
            audio_stimulus: 청각 자극 설정
            
        Returns:
            성공 여부
        """
        if 'data' in audio_stimulus:
            # 오디오 데이터가 직접 제공된 경우
            sample_rate = audio_stimulus.get('sample_rate', 44100)
            return self.audio_module.play_audio(audio_stimulus['data'], sample_rate)
        else:
            # 파라미터만 제공된 경우 (실제 구현에서는 여기서 오디오 생성)
            volume = audio_stimulus.get('volume', 0.5)
            self.audio_module.set_volume(volume)
            logger.info(f"Applied audio stimulus: {audio_stimulus}")
            return True
    
    def _apply_visual_stimulus(self, visual_stimulus: Dict) -> bool:
        """
        시각 자극 적용
        
        Args:
            visual_stimulus: 시각 자극 설정
            
        Returns:
            성공 여부
        """
        pattern = visual_stimulus.get('pattern', 'calm')
        brightness = visual_stimulus.get('brightness', 0.5)
        
        # 색상 처리
        color_name = visual_stimulus.get('color', 'white')
        color_map = {
            'white': (255, 255, 255),
            'blue': (0, 0, 255),
            'amber': (255, 191, 0),
            'red': (255, 0, 0),
            'deep_red': (139, 0, 0),
            'green': (0, 255, 0),
            'none': (0, 0, 0)
        }
        color = color_map.get(color_name, (255, 255, 255))
        
        duration = visual_stimulus.get('duration', 0)
        
        return self.visual_module.display_pattern(pattern, brightness, color, duration)
    
    def _apply_tactile_stimulus(self, tactile_stimulus: Dict) -> bool:
        """
        촉각 자극 적용
        
        Args:
            tactile_stimulus: 촉각 자극 설정
            
        Returns:
            성공 여부
        """
        intensity = tactile_stimulus.get('intensity', 0.5)
        
        # 주파수 처리
        frequency = tactile_stimulus.get('frequency', 10)
        if frequency == 'irregular':
            # 불규칙한 진동은 기본 주파수를 기반으로 패턴만 변경
            frequency = 10
            pattern = 'irregular'
        else:
            try:
                frequency = float(frequency)
                pattern = tactile_stimulus.get('pattern', 'continuous')
            except (ValueError, TypeError):
                frequency = 10
                pattern = 'continuous'
        
        duration = tactile_stimulus.get('duration', 0)
        
        return self.tactile_module.apply_vibration(intensity, frequency, pattern, duration)
    
    def stop_all_stimuli(self) -> bool:
        """
        모든 자극 중지
        
        Returns:
            성공 여부
        """
        audio_success = self.audio_module.stop_audio()
        visual_success = self.visual_module.turn_off()
        tactile_success = self.tactile_module.stop_vibration()
        
        return audio_success and visual_success and tactile_success
    
    def save_stimuli_config(self, filename: str) -> str:
        """
        자극 설정 저장
        
        Args:
            filename: 파일명
            
        Returns:
            저장된 파일 경로
        """
        if not self.current_stimuli:
            logger.warning("No current stimuli to save")
            return ""
        
        # 저장 가능한 형태로 변환
        save_data = {
            'timestamp': time.time(),
            'stimuli': {}
        }
        
        # 각 모듈별 설정 복사 (numpy 배열은 제외)
        if 'audio' in self.current_stimuli:
            audio_config = self.current_stimuli['audio'].copy()
            if 'data' in audio_config:
                # 데이터 배열은 저장하지 않고 메타데이터만 저장
                audio_config['data'] = f"array(shape={audio_config['data'].shape})"
            save_data['stimuli']['audio'] = audio_config
        
        if 'visual' in self.current_stimuli:
            save_data['stimuli']['visual'] = self.current_stimuli['visual'].copy()
        
        if 'tactile' in self.current_stimuli:
            save_data['stimuli']['tactile'] = self.current_stimuli['tactile'].copy()
        
        # 파일 경로 생성
        if not filename.endswith('.json'):
            filename += '.json'
        
        filepath = os.path.join(self.config.output_dir, filename)
        
        # 파일 저장
        try:
            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=4)
            logger.info(f"Stimuli configuration saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving stimuli configuration: {e}")
            return ""
    
    def generate_stimuli_report(self, output_file: str = None) -> Dict:
        """
        자극 보고서 생성
        
        Args:
            output_file: 출력 파일 경로 (선택적)
            
        Returns:
            보고서 데이터
        """
        if not self.current_stimuli:
            logger.warning("No stimuli data available for report")
            return {}
        
        # 보고서 데이터 구성
        report = {
            'date': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.stimuli_start_time)) if self.stimuli_start_time else time.strftime("%Y-%m-%d %H:%M:%S"),
            'duration': time.time() - self.stimuli_start_time if self.stimuli_start_time else 0,
            'stimuli_summary': {}
        }
        
        # 각 모듈별 요약 추가
        if 'audio' in self.current_stimuli:
            audio = self.current_stimuli['audio']
            report['stimuli_summary']['audio'] = {
                'volume': audio.get('volume', 'N/A'),
                'freq_mod': audio.get('freq_mod', 'N/A'),
                'amp_mod': audio.get('amp_mod', 'N/A')
            }
        
        if 'visual' in self.current_stimuli:
            visual = self.current_stimuli['visual']
            report['stimuli_summary']['visual'] = {
                'pattern': visual.get('pattern', 'N/A'),
                'brightness': visual.get('brightness', 'N/A'),
                'color': visual.get('color', 'N/A')
            }
        
        if 'tactile' in self.current_stimuli:
            tactile = self.current_stimuli['tactile']
            report['stimuli_summary']['tactile'] = {
                'intensity': tactile.get('intensity', 'N/A'),
                'frequency': tactile.get('frequency', 'N/A'),
                'pattern': tactile.get('pattern', 'N/A') if 'pattern' in tactile else 'continuous'
            }
        
        # 파일 출력 (선택적)
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    json.dump(report, f, indent=4)
                logger.info(f"Stimuli report saved to {output_file}")
            except Exception as e:
                logger.error(f"Error saving stimuli report: {e}")
        
        return report


# 예제 사용법
if __name__ == "__main__":
    # 설정 객체 생성
    config = OutputConfig(
        audio_device="default",
        eyemask_device_id="EYEMASK001",
        vibration_device_port="COM3"  # Windows 기준, Linux에서는 "/dev/ttyUSB0" 등
    )
    
    # 자극 출력 관리자 초기화
    output_manager = StimulusOutputManager(config)
    
    # 샘플 자극 설정
    sample_stimuli = {
        'audio': {
            'volume': 0.6,
            'freq_mod': 'alpha',
            'amp_mod': 'low',
            'data': np.random.normal(0, 0.1, (44100, 2))  # 1초 스테레오 노이즈
        },
        'visual': {
            'pattern': 'calm',
            'brightness': 0.4,
            'color': 'blue',
            'duration': 5.0
        },
        'tactile': {
            'intensity': 0.3,
            'frequency': 8.5,
            'duration': 5.0
        }
    }
    
    # 복합 자극 적용
    output_manager.apply_stimuli(sample_stimuli)
    
    # 5초 유지
    time.sleep(5.0)
    
    # 모든 자극 중지
    output_manager.stop_all_stimuli()
    
    # 자극 설정 저장
    output_manager.save_stimuli_config("sample_stimuli")
    
    # 자극 보고서 생성
    report = output_manager.generate_stimuli_report("stimuli_report.json")
    print(f"Stimuli Report: {report}")
