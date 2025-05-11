"""
상태 특성 추출 모듈

이 모듈은 이명 환자의 생체신호, 이명 특성, 수면 상태 등의 데이터로부터
강화학습에 필요한 상태 특성을 추출하는 기능을 담당합니다.
"""

import numpy as np
from scipy import signal
from typing import Dict, List, Tuple, Optional, Union
from .config import RLConfig


class StateFeatureExtractor:
    """상태 특성 추출기"""
    
    def __init__(self, config: RLConfig):
        """
        초기화
        
        Args:
            config: 강화학습 설정
        """
        self.config = config
    
    def extract_features(self, eeg_data: np.ndarray = None, 
                       hrv_data: np.ndarray = None,
                       tinnitus_data: Dict = None,
                       sleep_data: Dict = None) -> Dict[str, float]:
        """
        상태 특성 추출
        
        Args:
            eeg_data: EEG 데이터
            hrv_data: 심박 변이도 데이터
            tinnitus_data: 이명 특성 데이터
            sleep_data: 수면 상태 데이터
            
        Returns:
            특성 딕셔너리
        """
        features = {}
        
        # EEG 특성 추출
        if eeg_data is not None:
            eeg_features = self._extract_eeg_features(eeg_data)
            features.update(eeg_features)
        
        # HRV 특성 추출
        if hrv_data is not None:
            hrv_features = self._extract_hrv_features(hrv_data)
            features.update(hrv_features)
        
        # 이명 특성 추출
        if tinnitus_data is not None:
            tinnitus_features = self._extract_tinnitus_features(tinnitus_data)
            features.update(tinnitus_features)
        
        # 수면 특성 추출
        if sleep_data is not None:
            sleep_features = self._extract_sleep_features(sleep_data)
            features.update(sleep_features)
        
        return features
    
    def _extract_eeg_features(self, eeg_data: np.ndarray) -> Dict[str, float]:
        """
        EEG 특성 추출
        
        Args:
            eeg_data: EEG 데이터
            
        Returns:
            EEG 특성 딕셔너리
        """
        # 예시 EEG 특성 추출 로직
        # 실제 구현에서는 주파수 대역별 파워, 비대칭성, 연결성 등 다양한 특성 추출 필요
        
        # 간소화된 구현 (실제로는 MNE, PyEEG 등의 라이브러리 활용 권장)
        eeg = eeg_data.ravel() if len(eeg_data.shape) > 1 else eeg_data
        
        # 주파수 변환
        sample_rate = 256  # 가정된 샘플링 레이트
        freqs, psd = signal.welch(eeg, fs=sample_rate, nperseg=min(4096, len(eeg)))
        
        # 주파수 대역 인덱스
        delta_idx = np.logical_and(freqs >= 0.5, freqs <= 4)
        theta_idx = np.logical_and(freqs >= 4, freqs <= 8)
        alpha_idx = np.logical_and(freqs >= 8, freqs <= 13)
        beta_idx = np.logical_and(freqs >= 13, freqs <= 30)
        gamma_idx = np.logical_and(freqs >= 30, freqs <= 100)
        
        # 대역별 파워 계산
        delta_power = np.mean(psd[delta_idx])
        theta_power = np.mean(psd[theta_idx])
        alpha_power = np.mean(psd[alpha_idx])
        beta_power = np.mean(psd[beta_idx])
        gamma_power = np.mean(psd[gamma_idx]) if np.any(gamma_idx) else 0
        
        # 총 파워
        total_power = delta_power + theta_power + alpha_power + beta_power + gamma_power
        
        # 상대 파워
        rel_delta = delta_power / total_power if total_power > 0 else 0
        rel_theta = theta_power / total_power if total_power > 0 else 0
        rel_alpha = alpha_power / total_power if total_power > 0 else 0
        rel_beta = beta_power / total_power if total_power > 0 else 0
        rel_gamma = gamma_power / total_power if total_power > 0 else 0
        
        # 비율
        theta_alpha_ratio = theta_power / alpha_power if alpha_power > 0 else 0
        
        # 간단한 통계량
        eeg_mean = np.mean(eeg)
        eeg_std = np.std(eeg)
        eeg_min = np.min(eeg)
        eeg_max = np.max(eeg)
        
        return {
            'eeg_rel_delta': rel_delta,
            'eeg_rel_theta': rel_theta,
            'eeg_rel_alpha': rel_alpha,
            'eeg_rel_beta': rel_beta,
            'eeg_rel_gamma': rel_gamma,
            'eeg_theta_alpha_ratio': theta_alpha_ratio,
            'eeg_mean': eeg_mean,
            'eeg_std': eeg_std,
            'eeg_range': eeg_max - eeg_min
        }
    
    def _extract_hrv_features(self, hrv_data: np.ndarray) -> Dict[str, float]:
        """
        심박 변이도 특성 추출
        
        Args:
            hrv_data: RR 간격 데이터
            
        Returns:
            HRV 특성 딕셔너리
        """
        # 예시 HRV 특성 추출 로직
        # 실제 구현에서는 시간 도메인, 주파수 도메인, 비선형 분석 등 다양한 특성 추출 필요
        
        # RR 간격(msec)을 가정
        rr_intervals = hrv_data.ravel()
        
        # 시간 도메인 지표
        mean_rr = np.mean(rr_intervals)
        sdnn = np.std(rr_intervals)
        rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
        
        # NN50 및 pNN50
        nn50 = np.sum(np.abs(np.diff(rr_intervals)) > 50)
        pnn50 = (nn50 / len(rr_intervals)) * 100 if len(rr_intervals) > 0 else 0
        
        # 주파수 도메인 분석을 위한 시계열 재샘플링
        # (이 부분은 실제 구현 시 적절한 리샘플링 방법 사용 필요)
        
        # 간단한 주파수 분석
        sample_rate = 4.0  # 가정된 샘플링 레이트 (Hz)
        freqs, psd = signal.welch(rr_intervals, fs=sample_rate, nperseg=min(256, len(rr_intervals)))
        
        # 주파수 대역 인덱스
        vlf_idx = np.logical_and(freqs >= 0.0033, freqs <= 0.04)
        lf_idx = np.logical_and(freqs >= 0.04, freqs <= 0.15)
        hf_idx = np.logical_and(freqs >= 0.15, freqs <= 0.4)
        
        # 대역별 파워 계산
        vlf_power = np.sum(psd[vlf_idx]) if np.any(vlf_idx) else 0
        lf_power = np.sum(psd[lf_idx]) if np.any(lf_idx) else 0
        hf_power = np.sum(psd[hf_idx]) if np.any(hf_idx) else 0
        
        # 총 파워
        total_power = vlf_power + lf_power + hf_power
        
        # 정규화된 파워 및 비율
        norm_lf = lf_power / (lf_power + hf_power) if (lf_power + hf_power) > 0 else 0
        norm_hf = hf_power / (lf_power + hf_power) if (lf_power + hf_power) > 0 else 0
        lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0
        
        return {
            'hrv_mean_rr': mean_rr,
            'hrv_sdnn': sdnn,
            'hrv_rmssd': rmssd,
            'hrv_pnn50': pnn50,
            'hrv_norm_lf': norm_lf,
            'hrv_norm_hf': norm_hf,
            'hrv_lf_hf_ratio': lf_hf_ratio
        }
    
    def _extract_tinnitus_features(self, tinnitus_data: Dict) -> Dict[str, float]:
        """
        이명 특성 추출
        
        Args:
            tinnitus_data: 이명 특성 데이터
            
        Returns:
            이명 특성 딕셔너리
        """
        features = {}
        
        # 이명 주파수
        primary_freq = tinnitus_data.get('primary_frequency')
        if primary_freq is not None:
            features['tinnitus_freq'] = float(primary_freq)
        
        # 이명 강도
        intensity = tinnitus_data.get('intensity')
        if intensity is not None:
            features['tinnitus_intensity'] = float(intensity)
        
        # 이명 대역폭
        bandwidth = tinnitus_data.get('bandwidth')
        if bandwidth is not None:
            features['tinnitus_bandwidth'] = float(bandwidth)
        
        # 이명 지속시간 (분 단위)
        duration = tinnitus_data.get('duration')
        if duration is not None:
            features['tinnitus_duration'] = float(duration)
        
        # 자가 보고 심각도 (0-10 스케일)
        severity = tinnitus_data.get('severity')
        if severity is not None:
            features['tinnitus_severity'] = float(severity)
        
        # 이명의 일상 방해 정도 (0-10 스케일)
        annoyance = tinnitus_data.get('annoyance')
        if annoyance is not None:
            features['tinnitus_annoyance'] = float(annoyance)
        
        return features
    
    def _extract_sleep_features(self, sleep_data: Dict) -> Dict[str, float]:
        """
        수면 특성 추출
        
        Args:
            sleep_data: 수면 상태 데이터
            
        Returns:
            수면 특성 딕셔너리
        """
        features = {}
        
        # 수면 단계
        stage = sleep_data.get('stage')
        if stage is not None:
            # 수면 단계를 숫자로 변환
            stage_map = {'awake': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'REM': 4}
            features['sleep_stage'] = float(stage_map.get(stage, 0))
        
        # 수면 효율
        efficiency = sleep_data.get('sleep_efficiency')
        if efficiency is not None:
            features['sleep_efficiency'] = float(efficiency)
        
        # 수면 시간 (시간 단위)
        total_sleep_time = sleep_data.get('total_sleep_time')
        if total_sleep_time is not None:
            features['total_sleep_time'] = float(total_sleep_time)
        
        # 수면 중 각성 횟수
        awakenings = sleep_data.get('awakenings')
        if awakenings is not None:
            features['sleep_awakenings'] = float(awakenings)
        
        # N3 수면 비율
        n3_percentage = sleep_data.get('n3_percentage')
        if n3_percentage is not None:
            features['n3_percentage'] = float(n3_percentage)
        
        # REM 수면 비율
        rem_percentage = sleep_data.get('rem_percentage')
        if rem_percentage is not None:
            features['rem_percentage'] = float(rem_percentage)
        
        return features
    
    def normalize_features(self, features: Dict[str, float],
                         feature_means: Dict[str, float] = None,
                         feature_stds: Dict[str, float] = None) -> Dict[str, float]:
        """
        특성 정규화
        
        Args:
            features: 원본 특성 딕셔너리
            feature_means: 특성별 평균값 (None이면 현재 데이터로 계산)
            feature_stds: 특성별 표준편차 (None이면 현재 데이터로 계산)
            
        Returns:
            정규화된 특성 딕셔너리, 특성 평균, 특성 표준편차
        """
        # 원본 특성 복사
        normalized = {}
        
        # 평균, 표준편차가 제공되지 않으면 현재 데이터로 계산
        if feature_means is None:
            feature_means = {key: val for key, val in features.items()}
        
        if feature_stds is None:
            feature_stds = {key: 1.0 for key in features}
        
        # Z-점수 정규화 적용
        for key, value in features.items():
            mean = feature_means.get(key, 0)
            std = feature_stds.get(key, 1)
            
            if std > 0:
                normalized[key] = (value - mean) / std
            else:
                normalized[key] = value - mean
        
        return normalized, feature_means, feature_stds
