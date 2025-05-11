"""
수면 단계 분석 및 관리 모듈

이 모듈은 실시간 뇌파 분석을 통해 수면 단계를 판별하고, 
각 수면 단계에 최적화된 이명 치료 자극을 제공하는 기능을 담당합니다.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import simps
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import time
import json
import logging
from datetime import datetime
import os

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/sleep_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("sleep_analysis")


@dataclass
class SleepStageConfig:
    """수면 단계 분석 설정"""
    sample_rate: int = 256  # Hz
    window_size: float = 30.0  # 초
    window_step: float = 1.0  # 초
    delta_band: Tuple[float, float] = (0.5, 4.0)  # Hz
    theta_band: Tuple[float, float] = (4.0, 8.0)  # Hz
    alpha_band: Tuple[float, float] = (8.0, 13.0)  # Hz
    beta_band: Tuple[float, float] = (13.0, 30.0)  # Hz
    spindle_band: Tuple[float, float] = (12.0, 14.0)  # Hz
    slow_osc_band: Tuple[float, float] = (0.3, 1.0)  # Hz
    eog_threshold: float = 50.0  # μV
    emg_threshold: float = 20.0  # μV
    model_path: str = "models/sleep_stage_classifier.pkl"


class SleepStageDetector:
    """수면 단계 감지 모듈"""
    
    def __init__(self, config: SleepStageConfig = None):
        """
        초기화
        
        Args:
            config: 수면 단계 분석 설정. None인 경우 기본값 사용
        """
        self.config = config or SleepStageConfig()
        self.history = []
        self.current_stage = None
        
        # 스코어 평활화를 위한 필터 설계
        self.smoothing_filter = np.ones(10) / 10
        
        # 모델 로드 시도
        try:
            self._load_model()
            logger.info("Sleep stage classifier model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load sleep stage model: {e}. Will use rule-based classification")
            self.model = None
    
    def _load_model(self):
        """수면 단계 분류 모델 로드"""
        import joblib
        
        if os.path.exists(self.config.model_path):
            self.model = joblib.load(self.config.model_path)
        else:
            logger.warning(f"Model file not found at {self.config.model_path}")
            self.model = None
    
    def detect_stage(self, eeg_data: np.ndarray, eog_data: np.ndarray = None, 
                   emg_data: np.ndarray = None) -> str:
        """
        현재 수면 단계 감지
        
        Args:
            eeg_data: EEG 데이터 (채널 × 샘플)
            eog_data: 안전도(EOG) 데이터 (선택적)
            emg_data: 근전도(EMG) 데이터 (선택적)
            
        Returns:
            수면 단계 ('awake', 'N1', 'N2', 'N3', 'REM')
        """
        # 충분한 데이터가 있는지 확인
        window_samples = int(self.config.window_size * self.config.sample_rate)
        if eeg_data.shape[1] < window_samples:
            logger.warning(f"Insufficient EEG data for detection: {eeg_data.shape[1]} samples, need {window_samples}")
            return "unknown"
        
        # 특성 추출
        features = self._extract_features(eeg_data, eog_data, emg_data)
        
        # 모델 기반 예측 또는 규칙 기반 예측
        if self.model is not None:
            # 모델 기반 예측
            stage = self._predict_with_model(features)
        else:
            # 규칙 기반 예측
            stage = self._predict_with_rules(features)
        
        # 히스토리 업데이트 및 평활화 적용
        self.history.append(stage)
        if len(self.history) > 30:  # 최대 30초 히스토리 유지
            self.history.pop(0)
        
        # 최종 단계 결정 (다수결)
        from collections import Counter
        stage_counter = Counter(self.history[-10:])  # 최근 10초 데이터만 사용
        self.current_stage = stage_counter.most_common(1)[0][0]
        
        return self.current_stage
    
    def _extract_features(self, eeg_data: np.ndarray, eog_data: np.ndarray = None, 
                        emg_data: np.ndarray = None) -> Dict[str, float]:
        """
        수면 단계 분류를 위한 특성 추출
        
        Args:
            eeg_data: EEG 데이터 (채널 × 샘플)
            eog_data: 안전도(EOG) 데이터 (선택적)
            emg_data: 근전도(EMG) 데이터 (선택적)
            
        Returns:
            특성 딕셔너리
        """
        features = {}
        
        # EEG 채널 평균 (다중 채널인 경우)
        if len(eeg_data.shape) > 1 and eeg_data.shape[0] > 1:
            # 여러 채널이 있는 경우 평균
            eeg = np.mean(eeg_data, axis=0)
        else:
            # 단일 채널인 경우
            eeg = eeg_data.ravel()
        
        # 주파수 대역 파워 계산
        features.update(self._calculate_band_powers(eeg))
        
        # EOG 특성 (안구 움직임)
        if eog_data is not None:
            eog = eog_data.ravel() if len(eog_data.shape) > 1 else eog_data
            features['eog_max'] = np.max(np.abs(eog))
            features['eog_std'] = np.std(eog)
            features['eog_movements'] = self._detect_eye_movements(eog)
        
        # EMG 특성 (근육 긴장도)
        if emg_data is not None:
            emg = emg_data.ravel() if len(emg_data.shape) > 1 else emg_data
            features['emg_rms'] = np.sqrt(np.mean(emg**2))
            features['emg_above_threshold'] = np.mean(np.abs(emg) > self.config.emg_threshold)
        
        # 추가 특성 - 수면 방추 검출
        features['spindles'] = self._detect_sleep_spindles(eeg)
        
        # 추가 특성 - K-complex 검출
        features['k_complexes'] = self._detect_k_complexes(eeg)
        
        # 추가 특성 - 서파 진동 검출
        features['slow_oscillations'] = self._detect_slow_oscillations(eeg)
        
        return features
    
    def _calculate_band_powers(self, eeg: np.ndarray) -> Dict[str, float]:
        """
        주파수 대역별 파워 계산
        
        Args:
            eeg: EEG 신호
            
        Returns:
            대역별 파워 딕셔너리
        """
        # 주파수 영역 변환 (PSD 계산)
        freqs, psd = signal.welch(eeg, fs=self.config.sample_rate, 
                                nperseg=min(4096, len(eeg)),
                                scaling='density')
        
        # 각 대역의 주파수 인덱스 찾기
        freq_idx = lambda band: np.logical_and(freqs >= band[0], freqs <= band[1])
        
        # 각 대역 파워 계산 (영역 적분)
        delta_power = simps(psd[freq_idx(self.config.delta_band)], 
                          freqs[freq_idx(self.config.delta_band)])
        theta_power = simps(psd[freq_idx(self.config.theta_band)], 
                          freqs[freq_idx(self.config.theta_band)])
        alpha_power = simps(psd[freq_idx(self.config.alpha_band)], 
                          freqs[freq_idx(self.config.alpha_band)])
        beta_power = simps(psd[freq_idx(self.config.beta_band)], 
                         freqs[freq_idx(self.config.beta_band)])
        
        # 총 파워
        total_power = delta_power + theta_power + alpha_power + beta_power
        
        # 상대 파워 계산
        rel_delta = delta_power / total_power if total_power > 0 else 0
        rel_theta = theta_power / total_power if total_power > 0 else 0
        rel_alpha = alpha_power / total_power if total_power > 0 else 0
        rel_beta = beta_power / total_power if total_power > 0 else 0
        
        # 비율 계산
        theta_alpha_ratio = theta_power / alpha_power if alpha_power > 0 else 0
        theta_beta_ratio = theta_power / beta_power if beta_power > 0 else 0
        
        return {
            'delta_power': delta_power,
            'theta_power': theta_power,
            'alpha_power': alpha_power,
            'beta_power': beta_power,
            'rel_delta': rel_delta,
            'rel_theta': rel_theta,
            'rel_alpha': rel_alpha,
            'rel_beta': rel_beta,
            'theta_alpha_ratio': theta_alpha_ratio,
            'theta_beta_ratio': theta_beta_ratio
        }
    
    def _detect_eye_movements(self, eog: np.ndarray) -> float:
        """
        안구 움직임 검출
        
        Args:
            eog: EOG 신호
            
        Returns:
            초당 안구 움직임 수
        """
        # 미분 계산
        diff = np.diff(eog)
        
        # 임계값을 초과하는 급격한 변화 검출
        movements = np.where(np.abs(diff) > self.config.eog_threshold)[0]
        
        # 최소 간격 적용 (동일한 움직임이 중복 카운트되는 것 방지)
        if len(movements) > 1:
            min_gap = 0.1 * self.config.sample_rate  # 최소 0.1초 간격
            grouped_movements = [movements[0]]
            
            for i in range(1, len(movements)):
                if movements[i] - movements[i-1] > min_gap:
                    grouped_movements.append(movements[i])
            
            movements = grouped_movements
        
        # 초당 움직임 수 계산
        movement_rate = len(movements) / (len(eog) / self.config.sample_rate)
        
        return movement_rate
    
    def _detect_sleep_spindles(self, eeg: np.ndarray) -> float:
        """
        수면 방추(sleep spindle) 검출
        
        Args:
            eeg: EEG 신호
            
        Returns:
            초당 수면 방추 수
        """
        # 방추 대역 필터링
        b, a = signal.butter(3, [self.config.spindle_band[0] / (self.config.sample_rate/2),
                               self.config.spindle_band[1] / (self.config.sample_rate/2)], 
                            btype='band')
        filtered = signal.filtfilt(b, a, eeg)
        
        # 방추 진폭 계산
        amplitude_envelope = np.abs(signal.hilbert(filtered))
        
        # 임계값 설정 (신호 RMS의 3배)
        threshold = 3 * np.sqrt(np.mean(filtered**2))
        
        # 방추 검출 (0.5~3초 지속 시간 조건)
        min_duration_samples = int(0.5 * self.config.sample_rate)
        max_duration_samples = int(3.0 * self.config.sample_rate)
        
        # 임계값 초과 구간 찾기
        above_threshold = amplitude_envelope > threshold
        
        # 임계값 초과 구간의 시작과 끝 인덱스 찾기
        onset_indices = np.where(np.diff(above_threshold.astype(int)) > 0)[0]
        offset_indices = np.where(np.diff(above_threshold.astype(int)) < 0)[0]
        
        # 시작과 끝 인덱스 개수 맞추기
        if len(onset_indices) > 0 and len(offset_indices) > 0:
            if onset_indices[0] > offset_indices[0]:
                offset_indices = offset_indices[1:]
            if len(onset_indices) > len(offset_indices):
                onset_indices = onset_indices[:len(offset_indices)]
        
        # 유효한 방추 카운트
        spindle_count = 0
        for start, end in zip(onset_indices, offset_indices):
            duration = end - start
            if min_duration_samples <= duration <= max_duration_samples:
                spindle_count += 1
        
        # 초당 방추 수 계산
        spindle_rate = spindle_count / (len(eeg) / self.config.sample_rate)
        
        return spindle_rate
    
    def _detect_k_complexes(self, eeg: np.ndarray) -> float:
        """
        K-complex 검출
        
        Args:
            eeg: EEG 신호
            
        Returns:
            초당 K-complex 수
        """
        # 0.5-5Hz 대역 필터링 (K-complex 주파수 범위)
        b, a = signal.butter(3, [0.5 / (self.config.sample_rate/2),
                               5.0 / (self.config.sample_rate/2)], 
                            btype='band')
        filtered = signal.filtfilt(b, a, eeg)
        
        # K-complex 검출 (큰 음의 피크 다음 큰 양의 피크)
        # 신호 RMS의 4배 이상 진폭 조건
        threshold = 4 * np.sqrt(np.mean(filtered**2))
        
        # 피크 검출
        peaks, _ = signal.find_peaks(filtered, height=threshold, distance=0.5*self.config.sample_rate)
        troughs, _ = signal.find_peaks(-filtered, height=threshold, distance=0.5*self.config.sample_rate)
        
        # 유효한 K-complex 카운트 (음의 피크 다음 양의 피크가 0.5~1.5초 내에 발생)
        k_complex_count = 0
        min_gap = int(0.5 * self.config.sample_rate)
        max_gap = int(1.5 * self.config.sample_rate)
        
        for trough in troughs:
            for peak in peaks:
                gap = peak - trough
                if min_gap <= gap <= max_gap:
                    k_complex_count += 1
                    break
        
        # 초당 K-complex 수 계산
        k_complex_rate = k_complex_count / (len(eeg) / self.config.sample_rate)
        
        return k_complex_rate
    
    def _detect_slow_oscillations(self, eeg: np.ndarray) -> float:
        """
        서파 진동(slow oscillation) 검출
        
        Args:
            eeg: EEG 신호
            
        Returns:
            초당 서파 진동 수
        """
        # 서파 대역 필터링
        b, a = signal.butter(3, [self.config.slow_osc_band[0] / (self.config.sample_rate/2),
                               self.config.slow_osc_band[1] / (self.config.sample_rate/2)], 
                            btype='band')
        filtered = signal.filtfilt(b, a, eeg)
        
        # 임계값 설정 (신호 RMS의 2배)
        threshold = 2 * np.sqrt(np.mean(filtered**2))
        
        # 피크 검출
        peaks, _ = signal.find_peaks(filtered, height=threshold, distance=0.8*self.config.sample_rate)
        
        # 초당 서파 진동 수 계산
        oscillation_rate = len(peaks) / (len(eeg) / self.config.sample_rate)
        
        return oscillation_rate
    
    def _predict_with_model(self, features: Dict[str, float]) -> str:
        """
        모델을 사용한 수면 단계 예측
        
        Args:
            features: 특성 딕셔너리
            
        Returns:
            수면 단계 ('awake', 'N1', 'N2', 'N3', 'REM')
        """
        # 모델 입력 형식으로 변환
        feature_names = list(features.keys())
        feature_values = [features[name] for name in feature_names]
        X = np.array(feature_values).reshape(1, -1)
        
        # 모델 예측
        try:
            stage_id = self.model.predict(X)[0]
            # 숫자 ID를 단계 이름으로 변환
            stage_map = {0: 'awake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
            stage = stage_map.get(stage_id, 'unknown')
        except Exception as e:
            logger.error(f"Error during model prediction: {e}")
            stage = self._predict_with_rules(features)  # 폴백으로 규칙 기반 사용
        
        return stage
    
    def _predict_with_rules(self, features: Dict[str, float]) -> str:
        """
        규칙 기반 수면 단계 예측
        
        Args:
            features: 특성 딕셔너리
            
        Returns:
            수면 단계 ('awake', 'N1', 'N2', 'N3', 'REM')
        """
        # 1. 각성 상태 (awake) 검출
        if features.get('rel_alpha', 0) > 0.3 or features.get('rel_beta', 0) > 0.4:
            return 'awake'
        
        # 2. N3 수면 (깊은 수면) 검출
        if features.get('rel_delta', 0) > 0.5:
            return 'N3'
        
        # 3. REM 수면 검출
        if (features.get('eog_movements', 0) > 2.0 and 
            features.get('emg_rms', 100) < self.config.emg_threshold and
            features.get('theta_alpha_ratio', 0) > 1.5):
            return 'REM'
        
        # 4. N2 수면 검출
        if (features.get('spindles', 0) > 1.0 or 
            features.get('k_complexes', 0) > 0.5):
            return 'N2'
        
        # 5. N1 수면 (기본값)
        return 'N1'


class SleepOptimizer:
    """수면 최적화 모듈"""
    
    def __init__(self, config: SleepStageConfig = None):
        """
        초기화
        
        Args:
            config: 수면 단계 분석 설정. None인 경우 기본값 사용
        """
        self.config = config or SleepStageConfig()
        self.stage_detector = SleepStageDetector(self.config)
        self.sleep_quality = {
            'total_sleep_time': 0,
            'sleep_efficiency': 0,
            'awake_time': 0,
            'n1_time': 0,
            'n2_time': 0,
            'n3_time': 0,
            'rem_time': 0,
            'stage_transitions': 0
        }
        self.stimuli_config = {
            'awake': {
                'audio': {'volume': 0.7, 'freq_mod': 'alpha', 'amp_mod': 'low'},
                'visual': {'brightness': 0.6, 'pattern': 'calm', 'color': 'blue'},
                'tactile': {'intensity': 0.5, 'frequency': 10}
            },
            'N1': {
                'audio': {'volume': 0.5, 'freq_mod': 'theta', 'amp_mod': 'medium'},
                'visual': {'brightness': 0.3, 'pattern': 'transient', 'color': 'amber'},
                'tactile': {'intensity': 0.4, 'frequency': 6}
            },
            'N2': {
                'audio': {'volume': 0.3, 'freq_mod': 'delta', 'amp_mod': 'high'},
                'visual': {'brightness': 0.1, 'pattern': 'off', 'color': 'none'},
                'tactile': {'intensity': 0.2, 'frequency': 3}
            },
            'N3': {
                'audio': {'volume': 0.2, 'freq_mod': 'delta', 'amp_mod': 'high'},
                'visual': {'brightness': 0, 'pattern': 'off', 'color': 'none'},
                'tactile': {'intensity': 0.1, 'frequency': 2}
            },
            'REM': {
                'audio': {'volume': 0.4, 'freq_mod': 'custom', 'amp_mod': 'irregular'},
                'visual': {'brightness': 0.05, 'pattern': 'subtle', 'color': 'deep_red'},
                'tactile': {'intensity': 0.3, 'frequency': 'irregular'}
            }
        }
        self.history = []
        self.last_stage = None
        self.start_time = None
        self.stage_start_times = {stage: None for stage in ['awake', 'N1', 'N2', 'N3', 'REM']}
    
    def start_monitoring(self):
        """수면 모니터링 시작"""
        self.start_time = time.time()
        self.history = []
        self.last_stage = None
        self.sleep_quality = {
            'total_sleep_time': 0,
            'sleep_efficiency': 0,
            'awake_time': 0,
            'n1_time': 0,
            'n2_time': 0,
            'n3_time': 0,
            'rem_time': 0,
            'stage_transitions': 0
        }
        self.stage_start_times = {stage: None for stage in ['awake', 'N1', 'N2', 'N3', 'REM']}
        logger.info("Sleep monitoring started")
    
    def stop_monitoring(self):
        """수면 모니터링 중지 및 결과 분석"""
        if self.start_time is None:
            logger.warning("Monitoring was not started")
            return None
        
        end_time = time.time()
        total_time = end_time - self.start_time
        
        # 총 수면 시간 계산
        sleep_time = (self.sleep_quality['n1_time'] + 
                     self.sleep_quality['n2_time'] + 
                     self.sleep_quality['n3_time'] + 
                     self.sleep_quality['rem_time'])
        
        # 수면 효율 계산
        if total_time > 0:
            self.sleep_quality['sleep_efficiency'] = (sleep_time / total_time) * 100
        
        # 총 수면 시간 업데이트
        self.sleep_quality['total_sleep_time'] = sleep_time
        
        logger.info(f"Sleep monitoring stopped. Total time: {total_time:.1f}s, Sleep efficiency: {self.sleep_quality['sleep_efficiency']:.1f}%")
        
        return self.sleep_quality
    
    def update(self, eeg_data: np.ndarray, eog_data: np.ndarray = None, 
             emg_data: np.ndarray = None) -> Dict:
        """
        수면 상태 업데이트 및 최적 자극 설정 반환
        
        Args:
            eeg_data: EEG 데이터
            eog_data: 안전도(EOG) 데이터 (선택적)
            emg_data: 근전도(EMG) 데이터 (선택적)
            
        Returns:
            현재 수면 단계에 최적화된 자극 설정
        """
        # 수면 단계 감지
        current_stage = self.stage_detector.detect_stage(eeg_data, eog_data, emg_data)
        
        # 현재 시간
        current_time = time.time()
        
        # 수면 단계 전환 감지
        if self.last_stage is not None and current_stage != self.last_stage:
            # 이전 단계 시간 업데이트
            if self.stage_start_times[self.last_stage] is not None:
                stage_duration = current_time - self.stage_start_times[self.last_stage]
                
                if self.last_stage == 'awake':
                    self.sleep_quality['awake_time'] += stage_duration
                elif self.last_stage == 'N1':
                    self.sleep_quality['n1_time'] += stage_duration
                elif self.last_stage == 'N2':
                    self.sleep_quality['n2_time'] += stage_duration
                elif self.last_stage == 'N3':
                    self.sleep_quality['n3_time'] += stage_duration
                elif self.last_stage == 'REM':
                    self.sleep_quality['rem_time'] += stage_duration
            
            # 단계 전환 카운트 증가
            self.sleep_quality['stage_transitions'] += 1
            
            # 새 단계 시작 시간 기록
            self.stage_start_times[current_stage] = current_time
            
            logger.info(f"Sleep stage transition: {self.last_stage} -> {current_stage}")
        
        # 첫 단계 시작 시 시작 시간 기록
        elif self.last_stage is None:
            self.stage_start_times[current_stage] = current_time
        
        # 현재 단계 저장
        self.last_stage = current_stage
        
        # 히스토리에 현재 단계 추가
        self.history.append({
            'timestamp': current_time,
            'stage': current_stage
        })
        
        # 현재 단계에 최적화된 자극 설정 반환
        return self.get_optimized_stimuli(current_stage)
    
    def get_optimized_stimuli(self, sleep_stage: str) -> Dict:
        """
        수면 단계에 최적화된 자극 설정 반환
        
        Args:
            sleep_stage: 수면 단계 ('awake', 'N1', 'N2', 'N3', 'REM')
            
        Returns:
            자극 설정 딕셔너리
        """
        # 기본 설정 가져오기
        if sleep_stage in self.stimuli_config:
            stimuli = self.stimuli_config[sleep_stage]
        else:
            # 알 수 없는 단계는 N1 설정 사용
            stimuli = self.stimuli_config['N1']
            logger.warning(f"Unknown sleep stage '{sleep_stage}', using N1 stimuli settings")
        
        # 추가 최적화 로직 (필요시 구현)
        # 예: 수면 이력, 이명 특성 등에 따른 미세 조정
        
        return stimuli
    
    def generate_sleep_report(self, output_file: str = None) -> Dict:
        """
        수면 보고서 생성
        
        Args:
            output_file: 출력 파일 경로 (선택적)
            
        Returns:
            수면 보고서 데이터
        """
        if not self.history:
            logger.warning("No sleep data available for report")
            return {}
        
        # 기본 통계 계산
        stage_counts = {'awake': 0, 'N1': 0, 'N2': 0, 'N3': 0, 'REM': 0}
        for entry in self.history:
            if entry['stage'] in stage_counts:
                stage_counts[entry['stage']] += 1
        
        total_records = sum(stage_counts.values())
        
        # 백분율 계산
        stage_percentages = {}
        for stage, count in stage_counts.items():
            stage_percentages[stage] = (count / total_records * 100) if total_records > 0 else 0
        
        # 보고서 데이터 구성
        report = {
            'date': datetime.now().strftime("%Y-%m-%d"),
            'start_time': datetime.fromtimestamp(self.start_time).strftime("%H:%M:%S") if self.start_time else "Unknown",
            'total_sleep_time': f"{self.sleep_quality['total_sleep_time'] / 60:.1f} minutes",
            'sleep_efficiency': f"{self.sleep_quality['sleep_efficiency']:.1f}%",
            'stage_percentages': stage_percentages,
            'stage_transitions': self.sleep_quality['stage_transitions'],
            'stage_durations': {
                'awake': f"{self.sleep_quality['awake_time'] / 60:.1f} minutes",
                'N1': f"{self.sleep_quality['n1_time'] / 60:.1f} minutes",
                'N2': f"{self.sleep_quality['n2_time'] / 60:.1f} minutes",
                'N3': f"{self.sleep_quality['n3_time'] / 60:.1f} minutes",
                'REM': f"{self.sleep_quality['rem_time'] / 60:.1f} minutes"
            }
        }
        
        # 파일 출력 (선택적)
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    json.dump(report, f, indent=4)
                logger.info(f"Sleep report saved to {output_file}")
            except Exception as e:
                logger.error(f"Error saving sleep report: {e}")
        
        return report
    
    def plot_hypnogram(self, output_file: str = None):
        """
        수면도(hypnogram) 그래프 생성
        
        Args:
            output_file: 출력 파일 경로 (선택적)
        """
        if not self.history:
            logger.warning("No sleep data available for hypnogram")
            return
        
        # 데이터 준비
        timestamps = [entry['timestamp'] - self.start_time for entry in self.history]
        stages = [entry['stage'] for entry in self.history]
        
        # 수면 단계를 숫자로 변환
        stage_map = {'awake': 0, 'REM': 1, 'N1': 2, 'N2': 3, 'N3': 4}
        stage_values = [stage_map.get(stage, 0) for stage in stages]
        
        # 그래프 생성
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, stage_values, 'b-')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Sleep Stage')
        plt.yticks([0, 1, 2, 3, 4], ['Awake', 'REM', 'N1', 'N2', 'N3'])
        plt.title('Sleep Hypnogram')
        plt.grid(True)
        
        # 저장 또는 표시
        if output_file:
            plt.savefig(output_file)
            logger.info(f"Hypnogram saved to {output_file}")
        else:
            plt.show()
        
        plt.close()


# 예제 사용법
if __name__ == "__main__":
    # 설정 객체 생성
    config = SleepStageConfig(
        sample_rate=256,
        window_size=30.0
    )
    
    # 수면 최적화 모듈 초기화
    optimizer = SleepOptimizer(config)
    
    # 수면 모니터링 시작
    optimizer.start_monitoring()
    
    # 샘플 데이터 생성 (실제로는 EEG 장비에서 데이터 수집)
    import time
    
    for i in range(100):  # 100초 시뮬레이션
        # 샘플 EEG 데이터 생성
        samples = 256 * 30  # 30초 데이터
        eeg_data = np.random.normal(0, 50, (1, samples))  # 1채널 EEG
        
        # 수면 상태 업데이트 및 최적 자극 가져오기
        stimuli = optimizer.update(eeg_data)
        print(f"Time {i}s: Sleep stage = {optimizer.last_stage}, Stimuli = {stimuli}")
        
        time.sleep(1)  # 1초 대기
    
    # 수면 모니터링 중지 및 결과 분석
    sleep_quality = optimizer.stop_monitoring()
    print(f"Sleep Quality: {sleep_quality}")
    
    # 수면 보고서 생성
    report = optimizer.generate_sleep_report("sleep_report.json")
    
    # 수면도 생성
    optimizer.plot_hypnogram("hypnogram.png")
