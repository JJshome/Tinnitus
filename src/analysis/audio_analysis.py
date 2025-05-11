"""
고해상도 청력 검사 및 이명 주파수 분석 모듈

이 모듈은 125Hz부터 16kHz까지 1/24 옥타브 간격으로 청력 역치를 측정하고,
이명 주파수를 0.1kHz 단위로 정밀하게 측정하여 3차원 청각 지도를 작성하는 기능을 제공합니다.
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union
import pandas as pd


@dataclass
class AudioAnalysisConfig:
    """오디오 분석 설정"""
    min_frequency: float = 125.0  # Hz
    max_frequency: float = 16000.0  # Hz
    octave_fraction: int = 24  # 1/24 옥타브 간격
    min_level: float = -10.0  # dB HL
    max_level: float = 120.0  # dB HL
    level_step: float = 1.0  # dB
    tinnitus_resolution: float = 0.1  # kHz
    sample_rate: int = 44100  # Hz


class HighResolutionAudioAnalysis:
    """고해상도 청력 검사 및 이명 주파수 분석 클래스"""
    
    def __init__(self, config: AudioAnalysisConfig = None):
        """
        초기화
        
        Args:
            config: 오디오 분석 설정. None인 경우 기본값 사용
        """
        self.config = config or AudioAnalysisConfig()
        self.frequencies = self._generate_frequencies()
        self.levels = self._generate_levels()
        
    def _generate_frequencies(self) -> np.ndarray:
        """
        1/24 옥타브 간격의 주파수 배열 생성
        
        Returns:
            주파수 배열
        """
        octaves = np.log2(self.config.max_frequency / self.config.min_frequency)
        n_frequencies = int(octaves * self.config.octave_fraction) + 1
        
        return self.config.min_frequency * np.power(
            2, np.linspace(0, octaves, n_frequencies)
        )
    
    def _generate_levels(self) -> np.ndarray:
        """
        음압 레벨 배열 생성
        
        Returns:
            음압 레벨 배열
        """
        n_levels = int((self.config.max_level - self.config.min_level) / 
                        self.config.level_step) + 1
        
        return np.linspace(
            self.config.min_level, 
            self.config.max_level, 
            n_levels
        )
    
    def generate_test_tone(self, frequency: float, level: float, duration: float = 0.5) -> np.ndarray:
        """
        지정된 주파수와 레벨의 검사용 순음 생성
        
        Args:
            frequency: 주파수 (Hz)
            level: 음압 레벨 (dB HL)
            duration: 지속 시간 (초)
            
        Returns:
            검사용 순음 신호
        """
        t = np.linspace(0, duration, int(self.config.sample_rate * duration), False)
        
        # dB HL을 amplitude로 변환 (0 dB HL = 0.00002 Pa = 0.00002 N/m²)
        # 디지털 신호에서는 1.0을 기준으로 하므로, 변환 공식 적용
        amplitude = 10 ** (level / 20) * 0.00002 * 100
        
        # 진폭을 -1.0에서 1.0 사이로 제한
        amplitude = min(amplitude, 1.0)
        
        # 사인파 생성
        tone = amplitude * np.sin(2 * np.pi * frequency * t)
        
        # 톤의 시작과 끝에 10ms 페이드 인/아웃 적용
        fade_len = int(0.01 * self.config.sample_rate)
        fade_in = np.linspace(0, 1, fade_len)
        fade_out = np.linspace(1, 0, fade_len)
        
        if len(tone) > 2 * fade_len:
            tone[:fade_len] *= fade_in
            tone[-fade_len:] *= fade_out
        
        return tone
    
    def measure_hearing_threshold(self, frequency: float, 
                                  response_callback=None) -> Optional[float]:
        """
        특정 주파수에서 청력 역치 측정 (실제 측정을 위한 골격 함수)
        실제 구현에서는 하드웨어 연동 및 환자 반응 처리 로직 필요
        
        Args:
            frequency: 측정할 주파수 (Hz)
            response_callback: 사용자 응답을 받는 콜백 함수
            
        Returns:
            청력 역치 (dB HL) 또는 측정 실패 시 None
        """
        if response_callback is None:
            # 실제 구현에서는 GUI 또는 하드웨어 인터페이스를 통해 응답 수집
            print(f"Warning: No response callback provided for threshold measurement")
            return None
        
        # 변형된 계단식 측정 방법 구현
        # 먼저 큰 레벨에서 시작하여 반응이 있으면 내려가고, 없으면 올라감
        # 첫 번째 반전 후에는 작은 단계로 변경
        
        # 이 함수는 실제 측정 로직의 뼈대만 제공하며,
        # 실제 구현에서는 복잡한 청력 검사 프로토콜 적용 필요
        
        level = 40.0  # 시작 레벨, 보통 정상 청력 기준으로 선택
        step = 10.0   # 초기 단계
        reversals = 0
        max_reversals = 6
        last_response = None
        threshold_levels = []
        
        while reversals < max_reversals:
            # 테스트 톤 생성
            tone = self.generate_test_tone(frequency, level)
            
            # 톤 재생 및 응답 확인 (실제 구현 필요)
            # response는 환자가 소리를 들었는지 여부 (True/False)
            response = response_callback(tone, frequency, level)
            
            if last_response is not None and response != last_response:
                # 반응 변화 = 반전점
                reversals += 1
                if reversals == 1:
                    # 첫 번째 반전 후 작은 단계로 변경
                    step = 5.0
                if reversals == 2:
                    # 두 번째 반전 후 더 작은 단계로 변경
                    step = 2.0
                if reversals > 2:
                    # 두 번째 반전 이후부터 역치 계산에 포함
                    threshold_levels.append(level)
            
            # 다음 레벨 결정
            if response:  # 소리를 들었으면 레벨 낮춤
                level -= step
            else:  # 소리를 듣지 못했으면 레벨 높임
                level += step
                
            # 범위 제한
            level = min(max(level, self.config.min_level), self.config.max_level)
            
            last_response = response
        
        # 마지막 반전점들의 평균으로 역치 계산
        if threshold_levels:
            return np.mean(threshold_levels)
        else:
            return None
        
    def measure_full_audiogram(self, response_callback=None) -> pd.DataFrame:
        """
        전체 주파수 범위에서 청력도 측정
        
        Args:
            response_callback: 사용자 응답을 받는 콜백 함수
            
        Returns:
            주파수별 청력 역치를 담은 DataFrame
        """
        results = []
        
        for freq in self.frequencies:
            threshold = self.measure_hearing_threshold(freq, response_callback)
            results.append({
                'frequency': freq,
                'threshold': threshold
            })
            
        return pd.DataFrame(results)
    
    def analyze_tinnitus_frequency(self, audiogram: pd.DataFrame,
                               user_feedback_callback=None) -> Dict[str, float]:
        """
        이명 주파수 분석
        
        Args:
            audiogram: 청력도 DataFrame (주파수, 역치)
            user_feedback_callback: 사용자 피드백을 수집하는 콜백 함수
            
        Returns:
            이명 특성 정보 (주파수, 강도, 대역폭 등)
        """
        # 이 함수는 실제 이명 주파수 분석 로직의 뼈대만 제공
        # 실제 구현에서는 다양한 이명 특성 측정 및 분석 알고리즘 필요
        
        # 샘플 구현: 환자가 직접 이명과 가장 유사한 주파수를 찾도록 유도
        
        if user_feedback_callback is None:
            print("Warning: No user feedback callback provided for tinnitus analysis")
            return {'primary_frequency': None, 'intensity': None, 'bandwidth': None}
        
        # 이명의 주요 특성 측정
        tinnitus_data = {}
        
        # 1. 주요 주파수 측정 - 정밀한 주파수 조정으로 측정
        start_freq = 1000.0  # 초기 추정 주파수 (Hz)
        tinnitus_data['primary_frequency'] = self._find_matching_frequency(
            start_freq, user_feedback_callback)
        
        # 2. 이명 강도 측정 - 유사한 강도의 외부 소리와 비교
        tinnitus_data['intensity'] = self._measure_tinnitus_intensity(
            tinnitus_data['primary_frequency'], user_feedback_callback)
        
        # 3. 대역폭/주파수 범위 측정
        tinnitus_data['bandwidth'] = self._measure_tinnitus_bandwidth(
            tinnitus_data['primary_frequency'], user_feedback_callback)
        
        return tinnitus_data
    
    def _find_matching_frequency(self, start_freq: float, 
                               user_feedback_callback) -> float:
        """
        환자의 이명과 가장 유사한 주파수 찾기
        
        Args:
            start_freq: 시작 주파수 (Hz)
            user_feedback_callback: 사용자 피드백 콜백
            
        Returns:
            이명과 가장 유사한 주파수 (Hz)
        """
        # 이진 탐색과 유사한 접근 방식으로 주파수 범위 좁히기
        # 실제 구현에서는 더 복잡한 알고리즘 필요
        
        freq = start_freq
        step = 1000.0  # 초기 단계
        
        # 사용자가 "매우 비슷함"으로 응답할 때까지 반복
        while step > 10.0:  # 최소 10Hz 정밀도까지
            # 현재 주파수 재생
            tone = self.generate_test_tone(freq, 40.0, 2.0)
            
            # 사용자 피드백 (높음/낮음/유사함)
            feedback = user_feedback_callback(tone, freq)
            
            # 피드백에 따라 주파수 조정
            if feedback == "higher":
                freq += step
            elif feedback == "lower":
                freq -= step
            elif feedback == "very_similar":
                break
            
            # 다음 반복에서 작은 단계 사용
            step /= 2
        
        return freq
    
    def _measure_tinnitus_intensity(self, frequency: float, 
                                  user_feedback_callback) -> float:
        """
        이명 강도 측정
        
        Args:
            frequency: 이명 주파수 (Hz)
            user_feedback_callback: 사용자 피드백 콜백
            
        Returns:
            이명 강도 (dB HL)
        """
        # 강도 비교 방식으로 이명 강도 추정
        level = 0.0
        step = 5.0
        
        while True:
            tone = self.generate_test_tone(frequency, level)
            feedback = user_feedback_callback(tone, level, "intensity")
            
            if feedback == "louder":
                level += step
            elif feedback == "softer":
                level -= step
            elif feedback == "equal":
                break
                
            if step > 1.0 and (feedback == "almost_equal"):
                step = 1.0
            
            # 범위 제한
            level = min(max(level, -10), 100)
            
            if abs(step) < 1.0:
                break
        
        return level
    
    def _measure_tinnitus_bandwidth(self, center_freq: float, 
                                 user_feedback_callback) -> float:
        """
        이명 대역폭 측정
        
        Args:
            center_freq: 이명 중심 주파수 (Hz)
            user_feedback_callback: 사용자 피드백 콜백
            
        Returns:
            이명 대역폭 (Hz)
        """
        # 밴드패스 노이즈의 대역폭을 조정하며 이명과 가장 유사한 소리 찾기
        # 실제 구현에서는 더 정교한 방법 필요
        
        bandwidth = center_freq / 3.0  # 초기 대역폭 추정값 (1/3 옥타브)
        step = bandwidth / 2.0
        
        while step > 10.0:
            # 현재 대역폭으로 밴드패스 노이즈 생성
            noise = self._generate_bandpass_noise(center_freq, bandwidth)
            
            # 사용자 피드백 (넓음/좁음/유사함)
            feedback = user_feedback_callback(noise, bandwidth, "bandwidth")
            
            # 피드백에 따라 대역폭 조정
            if feedback == "wider":
                bandwidth += step
            elif feedback == "narrower":
                bandwidth -= step
            elif feedback == "very_similar":
                break
            
            # 다음 반복에서 작은 단계 사용
            step /= 2
            
            # 대역폭 범위 제한
            bandwidth = max(bandwidth, 10.0)  # 최소 10Hz
        
        return bandwidth
    
    def _generate_bandpass_noise(self, center_freq: float, bandwidth: float) -> np.ndarray:
        """
        특정 중심 주파수와 대역폭을 갖는 밴드패스 노이즈 생성
        
        Args:
            center_freq: 중심 주파수 (Hz)
            bandwidth: 대역폭 (Hz)
            
        Returns:
            밴드패스 노이즈 신호
        """
        # 백색 노이즈 생성
        duration = 2.0  # 2초
        samples = int(self.config.sample_rate * duration)
        white_noise = np.random.normal(0, 0.1, samples)
        
        # 노이즈에 밴드패스 필터 적용
        low_freq = max(center_freq - bandwidth/2, 20)  # 최소 20Hz
        high_freq = min(center_freq + bandwidth/2, self.config.sample_rate/2 - 1)
        
        # 필터 설계
        nyquist = self.config.sample_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # 4차 버터워스 필터 적용
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_noise = signal.lfilter(b, a, white_noise)
        
        # 정규화
        filtered_noise = filtered_noise / np.max(np.abs(filtered_noise)) * 0.7
        
        return filtered_noise

    def create_3d_auditory_map(self, audiogram: pd.DataFrame, 
                            tinnitus_data: Dict[str, float]) -> Dict:
        """
        3차원 청각 지도 생성
        
        Args:
            audiogram: 청력도 데이터
            tinnitus_data: 이명 특성 데이터
            
        Returns:
            3차원 청각 지도 데이터
        """
        # 이 함수는 실제 3D 청각 지도 생성 로직의 뼈대만 제공
        # 실제 구현에서는 더 복잡한 데이터 처리 및 시각화 필요
        
        # 청력도와 이명 데이터를 결합하여 3D 지도 생성
        map_data = {
            'frequencies': audiogram['frequency'].tolist(),
            'thresholds': audiogram['threshold'].tolist(),
            'tinnitus_frequency': tinnitus_data.get('primary_frequency'),
            'tinnitus_intensity': tinnitus_data.get('intensity'),
            'tinnitus_bandwidth': tinnitus_data.get('bandwidth')
        }
        
        return map_data
    
    def plot_audiogram(self, audiogram: pd.DataFrame, 
                    tinnitus_data: Dict[str, float] = None,
                    show_tinnitus: bool = True):
        """
        청력도 및 이명 데이터 시각화
        
        Args:
            audiogram: 청력도 데이터
            tinnitus_data: 이명 특성 데이터 (옵션)
            show_tinnitus: 이명 데이터 표시 여부
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 로그 스케일로 주파수 표시
        ax.semilogx(audiogram['frequency'], audiogram['threshold'], 'o-', markersize=8)
        
        # 축 반전 (청력도는 위로 갈수록 청력이 나빠짐)
        ax.invert_yaxis()
        
        # 격자 및 축 레이블 설정
        ax.grid(True, which="both", ls="-", alpha=0.5)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Hearing Level (dB HL)')
        ax.set_title('High-Resolution Audiogram')
        
        # 주요 주파수 표시
        major_freqs = [125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        ax.set_xticks(major_freqs)
        ax.set_xticklabels([str(f) for f in major_freqs])
        
        # 청력 등급 표시
        hearing_levels = [-10, 0, 20, 40, 60, 80, 100, 120]
        for level in hearing_levels:
            if level == 0:
                ax.axhline(y=level, color='g', linestyle='-', alpha=0.3, linewidth=2)
            elif level == 20:
                ax.axhline(y=level, color='y', linestyle='-', alpha=0.3, linewidth=2)
            elif level == 40:
                ax.axhline(y=level, color='orange', linestyle='-', alpha=0.3, linewidth=2)
            elif level >= 60:
                ax.axhline(y=level, color='r', linestyle='-', alpha=0.3, linewidth=2)
        
        # 이명 데이터 표시
        if show_tinnitus and tinnitus_data and tinnitus_data.get('primary_frequency'):
            primary_freq = tinnitus_data.get('primary_frequency')
            intensity = tinnitus_data.get('intensity', 0)
            bandwidth = tinnitus_data.get('bandwidth', 0)
            
            # 이명 주파수 지점 표시
            ax.scatter([primary_freq], [intensity], color='r', s=100, marker='*', 
                     label=f'Tinnitus: {primary_freq:.1f} Hz')
            
            # 이명 대역폭 표시
            if bandwidth:
                low_freq = primary_freq - bandwidth/2
                high_freq = primary_freq + bandwidth/2
                ax.axvspan(low_freq, high_freq, alpha=0.2, color='red', 
                          label=f'Bandwidth: {bandwidth:.1f} Hz')
        
        ax.legend()
        plt.tight_layout()
        plt.show()
        
        return fig


# 예제 사용법
if __name__ == "__main__":
    # 설정 객체 생성
    config = AudioAnalysisConfig(
        min_frequency=125.0,
        max_frequency=16000.0,
        octave_fraction=24
    )
    
    # 분석 객체 생성
    analyzer = HighResolutionAudioAnalysis(config)
    
    # 샘플 데이터 생성 (실제 측정 데이터 대체)
    sample_frequencies = [125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    sample_thresholds = [10, 15, 20, 25, 30, 45, 60, 70]
    
    sample_audiogram = pd.DataFrame({
        'frequency': sample_frequencies,
        'threshold': sample_thresholds
    })
    
    sample_tinnitus = {
        'primary_frequency': 6000.0,
        'intensity': 35.0,
        'bandwidth': 800.0
    }
    
    # 청력도 시각화
    analyzer.plot_audiogram(sample_audiogram, sample_tinnitus)
