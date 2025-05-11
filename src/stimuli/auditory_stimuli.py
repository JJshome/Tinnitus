"""
맞춤형 청각 자극 생성 모듈

이 모듈은 이명 환자의 특성에 맞춘 청각 자극을 생성합니다.
주요 기능:
1. 환자의 이명 주파수를 중심으로 한 노치 필터링
2. 환자의 뇌파 리듬과 동기화된 주파수 변조
3. 수면 유도를 위한 델타파 리듬 강화
"""

import numpy as np
from scipy import signal
import soundfile as sf
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union
import os


@dataclass
class AuditoryStimConfig:
    """청각 자극 생성 설정"""
    sample_rate: int = 44100  # Hz
    duration: float = 3600.0  # 1시간 (초)
    fadeout_duration: float = 10.0  # 페이드아웃 시간 (초)
    notch_width_octaves: float = 1.0  # 노치 필터 폭 (옥타브)
    notch_depth: float = 40.0  # 노치 필터 감쇠량 (dB)
    alpha_freq: float = 10.0  # 알파파 주파수 (Hz)
    delta_freq: float = 2.0  # 델타파 주파수 (Hz)
    output_dir: str = "output/auditory"


class NotchFilterModule:
    """노치 필터 모듈"""
    
    def __init__(self, config: AuditoryStimConfig):
        """
        초기화
        
        Args:
            config: 청각 자극 생성 설정
        """
        self.config = config
        
    def design_notch_filter(self, tinnitus_freq: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        이명 주파수에 맞춘 노치 필터 설계
        
        Args:
            tinnitus_freq: 이명 주파수 (Hz)
            
        Returns:
            필터 계수 (b, a)
        """
        # 노치 폭 계산 (옥타브 기반)
        lower_freq = tinnitus_freq / (2 ** (self.config.notch_width_octaves / 2))
        upper_freq = tinnitus_freq * (2 ** (self.config.notch_width_octaves / 2))
        
        # 정규화된 주파수 계산 (0.0 ~ 1.0)
        nyquist = self.config.sample_rate / 2.0
        low = lower_freq / nyquist
        high = upper_freq / nyquist
        
        # IIR 노치 필터 설계
        b, a = signal.iirnotch(tinnitus_freq / nyquist, 30, self.config.notch_depth)
        
        return b, a
    
    def apply_notch_filter(self, audio: np.ndarray, tinnitus_freq: float) -> np.ndarray:
        """
        노치 필터 적용
        
        Args:
            audio: 입력 오디오 신호
            tinnitus_freq: 이명 주파수 (Hz)
            
        Returns:
            필터링된 오디오 신호
        """
        b, a = self.design_notch_filter(tinnitus_freq)
        filtered_audio = signal.filtfilt(b, a, audio)
        
        return filtered_audio
    
    def plot_filter_response(self, tinnitus_freq: float):
        """
        노치 필터 주파수 응답 시각화
        
        Args:
            tinnitus_freq: 이명 주파수 (Hz)
        """
        b, a = self.design_notch_filter(tinnitus_freq)
        
        # 주파수 응답 계산
        w, h = signal.freqz(b, a)
        
        # Hz 단위로 변환
        freqs = w * self.config.sample_rate / (2 * np.pi)
        
        # 크기 응답 (dB)
        magnitude_db = 20 * np.log10(abs(h))
        
        plt.figure(figsize=(10, 6))
        plt.semilogx(freqs, magnitude_db)
        plt.axvline(x=tinnitus_freq, color='r', linestyle='--', 
                   label=f'Tinnitus: {tinnitus_freq} Hz')
        
        plt.title('Notch Filter Frequency Response')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True, which="both", ls="-")
        plt.xlim([20, 20000])
        plt.ylim([-60, 5])
        plt.legend()
        
        plt.tight_layout()
        plt.show()


class FrequencyModulationModule:
    """주파수 변조 모듈"""
    
    def __init__(self, config: AuditoryStimConfig):
        """
        초기화
        
        Args:
            config: 청각 자극 생성 설정
        """
        self.config = config
    
    def apply_alpha_modulation(self, audio: np.ndarray, 
                             alpha_freq: float = None,
                             modulation_depth: float = 0.2) -> np.ndarray:
        """
        알파파 리듬으로 주파수 변조 적용
        
        Args:
            audio: 입력 오디오 신호
            alpha_freq: 알파파 주파수 (Hz), None이면 설정값 사용
            modulation_depth: 변조 깊이 (0.0 ~ 1.0)
            
        Returns:
            변조된 오디오 신호
        """
        if alpha_freq is None:
            alpha_freq = self.config.alpha_freq
            
        # 시간 배열 생성
        t = np.linspace(0, len(audio) / self.config.sample_rate, len(audio), False)
        
        # 알파파 리듬 생성 (사인파)
        alpha_wave = np.sin(2 * np.pi * alpha_freq * t)
        
        # 변조 깊이 조정
        alpha_wave = alpha_wave * modulation_depth
        
        # 원본 신호에 변조 적용
        # 여기서는 진폭 변조 형태로 구현 (곱셈)
        modulated_audio = audio * (1.0 + alpha_wave)
        
        # 클리핑 방지를 위한 정규화
        modulated_audio = modulated_audio / np.max(np.abs(modulated_audio))
        
        return modulated_audio
    
    def apply_frequency_shift(self, audio: np.ndarray, 
                            shift_amount: float = 1.0,
                            modulation_freq: float = None) -> np.ndarray:
        """
        주파수 이동/변조 적용
        
        Args:
            audio: 입력 오디오 신호
            shift_amount: 주파수 이동량 (Hz)
            modulation_freq: 변조 주파수 (Hz), None이면 설정값 사용
            
        Returns:
            주파수 이동된 오디오 신호
        """
        if modulation_freq is None:
            modulation_freq = self.config.alpha_freq
            
        # 시간 배열 생성
        t = np.linspace(0, len(audio) / self.config.sample_rate, len(audio), False)
        
        # 힐버트 변환을 통한 해석적 신호 생성
        analytic_signal = signal.hilbert(audio)
        
        # 주파수 이동 - 곱셈을 통한 주파수 영역 이동
        # shift_amount가 양수면 주파수가 증가, 음수면 감소
        shifted_signal = analytic_signal * np.exp(2j * np.pi * shift_amount * t)
        
        # 변조 주파수로 주파수 변조
        if modulation_freq > 0:
            modulation = np.sin(2 * np.pi * modulation_freq * t)
            shifted_signal = shifted_signal * np.exp(2j * np.pi * modulation)
        
        # 실수부만 취함
        result = np.real(shifted_signal)
        
        # 클리핑 방지를 위한 정규화
        result = result / np.max(np.abs(result))
        
        return result


class AmplitudeModulationModule:
    """진폭 변조 모듈"""
    
    def __init__(self, config: AuditoryStimConfig):
        """
        초기화
        
        Args:
            config: 청각 자극 생성 설정
        """
        self.config = config
    
    def apply_delta_modulation(self, audio: np.ndarray, 
                             delta_freq: float = None,
                             modulation_depth: float = 0.3) -> np.ndarray:
        """
        델타파 리듬으로 진폭 변조 적용
        
        Args:
            audio: 입력 오디오 신호
            delta_freq: 델타파 주파수 (Hz), None이면 설정값 사용
            modulation_depth: 변조 깊이 (0.0 ~ 1.0)
            
        Returns:
            변조된 오디오 신호
        """
        if delta_freq is None:
            delta_freq = self.config.delta_freq
            
        # 시간 배열 생성
        t = np.linspace(0, len(audio) / self.config.sample_rate, len(audio), False)
        
        # 델타파 리듬 생성 (사인파)
        # 수면 유도를 위해 양의 반파만 사용 (진폭이 0 이하로 내려가지 않음)
        delta_wave = np.sin(2 * np.pi * delta_freq * t)
        delta_wave = np.maximum(0, delta_wave)
        
        # 변조 깊이 조정 및 DC 오프셋 추가
        # 신호가 완전히 묻히지 않도록 (1-modulation_depth)의 최소값 유지
        delta_wave = (1 - modulation_depth) + modulation_depth * delta_wave
        
        # 원본 신호에 변조 적용
        modulated_audio = audio * delta_wave
        
        return modulated_audio
    
    def apply_binaural_beats(self, audio: np.ndarray, 
                          base_freq: float = 220.0,
                          beat_freq: float = 5.0,
                          mix_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """
        양이 진동(Binaural Beats) 효과 적용
        
        Args:
            audio: 입력 오디오 신호
            base_freq: 기본 주파수 (Hz)
            beat_freq: 비트 주파수 (Hz) - 왼쪽과 오른쪽의 주파수 차이
            mix_ratio: 원본 오디오와 비트 신호의 믹스 비율 (0.0~1.0)
            
        Returns:
            (왼쪽 채널, 오른쪽 채널) 오디오 신호
        """
        # 시간 배열 생성
        t = np.linspace(0, len(audio) / self.config.sample_rate, len(audio), False)
        
        # 왼쪽과 오른쪽 채널의 주파수 계산
        left_freq = base_freq
        right_freq = base_freq + beat_freq
        
        # 좌우 채널 신호 생성
        left_beat = np.sin(2 * np.pi * left_freq * t)
        right_beat = np.sin(2 * np.pi * right_freq * t)
        
        # 원본 오디오가 모노인 경우 스테레오로 변환
        if len(audio.shape) == 1:
            audio_left = audio
            audio_right = audio
        else:
            audio_left = audio[:, 0]
            audio_right = audio[:, 1] if audio.shape[1] > 1 else audio[:, 0]
        
        # 비트 신호와 원본 오디오 믹싱
        left_channel = (1 - mix_ratio) * audio_left + mix_ratio * left_beat
        right_channel = (1 - mix_ratio) * audio_right + mix_ratio * right_beat
        
        # 클리핑 방지를 위한 정규화
        max_val = max(np.max(np.abs(left_channel)), np.max(np.abs(right_channel)))
        left_channel = left_channel / max_val
        right_channel = right_channel / max_val
        
        return left_channel, right_channel


class AuditoryStimGenerator:
    """청각 자극 생성기"""
    
    def __init__(self, config: AuditoryStimConfig = None):
        """
        초기화
        
        Args:
            config: 청각 자극 생성 설정. None인 경우 기본값 사용
        """
        self.config = config or AuditoryStimConfig()
        
        # 모듈 초기화
        self.notch_filter = NotchFilterModule(self.config)
        self.freq_modulation = FrequencyModulationModule(self.config)
        self.amp_modulation = AmplitudeModulationModule(self.config)
        
        # 출력 디렉토리 생성
        os.makedirs(self.config.output_dir, exist_ok=True)
    
    def generate_white_noise(self, duration: float = None) -> np.ndarray:
        """
        백색 소음 생성
        
        Args:
            duration: 지속 시간 (초), None이면 설정값 사용
            
        Returns:
            백색 소음 신호
        """
        if duration is None:
            duration = self.config.duration
            
        samples = int(self.config.sample_rate * duration)
        noise = np.random.normal(0, 0.1, samples)
        
        # 클리핑 방지를 위한 정규화
        noise = noise / np.max(np.abs(noise)) * 0.9
        
        return noise
    
    def generate_pink_noise(self, duration: float = None) -> np.ndarray:
        """
        핑크 소음 생성
        
        Args:
            duration: 지속 시간 (초), None이면 설정값 사용
            
        Returns:
            핑크 소음 신호
        """
        if duration is None:
            duration = self.config.duration
            
        samples = int(self.config.sample_rate * duration)
        
        # 백색 소음 생성
        white_noise = np.random.normal(0, 1, samples)
        
        # FFT를 이용한 주파수 도메인 변환
        X = np.fft.rfft(white_noise)
        
        # 주파수에 따른 1/f 스케일 적용
        S = np.sqrt(np.arange(len(X)) + 1.)  # +1 to avoid division by zero
        S[0] = 1  # DC 성분은 변경하지 않음
        y = np.fft.irfft(X / S, n=samples)
        
        # 클리핑 방지를 위한 정규화
        pink_noise = y / np.max(np.abs(y)) * 0.9
        
        return pink_noise
    
    def generate_natural_sound(self, sound_type: str = "rain",
                             duration: float = None) -> np.ndarray:
        """
        자연음 생성 (샘플 구현 - 실제로는 자연음 파일을 로드하거나 합성 필요)
        
        Args:
            sound_type: 자연음 유형 ("rain", "stream", "forest", "waves")
            duration: 지속 시간 (초), None이면 설정값 사용
            
        Returns:
            자연음 신호
        """
        if duration is None:
            duration = self.config.duration
            
        samples = int(self.config.sample_rate * duration)
        
        # 여기서는 간단히 필터링된 소음으로 자연음 대체
        # 실제 구현에서는 실제 자연음 샘플 로드 또는 더 정교한 합성 필요
        
        if sound_type == "rain":
            # 비 소리: 고주파가 풍부한 백색 소음
            noise = self.generate_white_noise(duration)
            
        elif sound_type == "stream":
            # 시냇물 소리: 중간 주파수 대역의 왜곡된 소음
            noise = self.generate_pink_noise(duration)
            # 특정 주파수 대역 강화
            b, a = signal.butter(2, [0.1, 0.6], btype='band')
            noise = signal.lfilter(b, a, noise)
            
        elif sound_type == "forest":
            # 숲 소리: 저주파 배경 + 간헐적 소리
            noise = self.generate_pink_noise(duration)
            # 저주파 강화
            b, a = signal.butter(2, 0.2, btype='low')
            noise = signal.lfilter(b, a, noise)
            
            # 간헐적 소리 추가 (새 소리 등 대체)
            t = np.linspace(0, duration, samples, False)
            chirps = np.zeros_like(noise)
            
            # 임의의 시간에 간헐적 소리 배치
            n_chirps = int(duration / 10)  # 평균 10초에 한 번
            for _ in range(n_chirps):
                chirp_time = np.random.uniform(0, duration)
                chirp_idx = int(chirp_time * self.config.sample_rate)
                chirp_len = int(0.2 * self.config.sample_rate)  # 0.2초 길이
                
                if chirp_idx + chirp_len < samples:
                    chirp = signal.chirp(
                        t=np.linspace(0, 0.2, chirp_len, False),
                        f0=2000,
                        f1=4000,
                        t1=0.2,
                        method='logarithmic'
                    )
                    
                    # 소리 크기 조정 및 적용
                    chirp = chirp * 0.2  # 볼륨 낮춤
                    chirps[chirp_idx:chirp_idx+chirp_len] += chirp
            
            # 배경 소음과 간헐적 소리 혼합
            noise = noise * 0.8 + chirps
            
        elif sound_type == "waves":
            # 파도 소리: 저주파 변조가 있는 백색 소음
            noise = self.generate_white_noise(duration)
            
            # 파도 리듬 생성 (약 0.1-0.2 Hz)
            t = np.linspace(0, duration, samples, False)
            wave_rhythm = np.sin(2 * np.pi * 0.1 * t) * 0.5 + 0.5  # 0~1 범위
            
            # 파도 리듬으로 변조
            noise = noise * wave_rhythm
            
        else:
            # 기본값: 핑크 소음
            noise = self.generate_pink_noise(duration)
        
        # 클리핑 방지를 위한 정규화
        noise = noise / np.max(np.abs(noise)) * 0.9
        
        return noise
    
    def create_notched_sound(self, tinnitus_freq: float, 
                          sound_type: str = "pink_noise",
                          duration: float = None) -> np.ndarray:
        """
        이명 주파수에 맞춘 노치 필터링된 소리 생성
        
        Args:
            tinnitus_freq: 이명 주파수 (Hz)
            sound_type: 소리 유형 ("white_noise", "pink_noise", "rain", "stream", "forest", "waves")
            duration: 지속 시간 (초), None이면 설정값 사용
            
        Returns:
            노치 필터링된 소리
        """
        if duration is None:
            duration = self.config.duration
        
        # 기본 소리 생성
        if sound_type == "white_noise":
            sound = self.generate_white_noise(duration)
        elif sound_type == "pink_noise":
            sound = self.generate_pink_noise(duration)
        else:
            sound = self.generate_natural_sound(sound_type, duration)
        
        # 노치 필터 적용
        filtered_sound = self.notch_filter.apply_notch_filter(sound, tinnitus_freq)
        
        return filtered_sound
    
    def create_personalized_stimuli(self, tinnitus_freq: float, 
                                 brain_data: Dict[str, float],
                                 sound_type: str = "pink_noise",
                                 duration: float = None,
                                 sleep_stage: str = None) -> np.ndarray:
        """
        개인 맞춤형 청각 자극 생성
        
        Args:
            tinnitus_freq: 이명 주파수 (Hz)
            brain_data: 뇌파 데이터 (알파파, 델타파 주파수 등)
            sound_type: 소리 유형
            duration: 지속 시간 (초), None이면 설정값 사용
            sleep_stage: 수면 단계 ('awake', 'N1', 'N2', 'N3', 'REM'), None이면 각성 상태 가정
            
        Returns:
            맞춤형 청각 자극 (스테레오 신호)
        """
        if duration is None:
            duration = self.config.duration
        
        # 환자의 뇌파 리듬 추출
        alpha_freq = brain_data.get('alpha_freq', self.config.alpha_freq)
        delta_freq = brain_data.get('delta_freq', self.config.delta_freq)
        
        # 1. 노치 필터링된 기본 소리 생성
        sound = self.create_notched_sound(tinnitus_freq, sound_type, duration)
        
        # 2. 수면 단계에 따른 처리 최적화
        if sleep_stage is None or sleep_stage == 'awake':
            # 각성 상태: 알파파 리듬 강화
            sound = self.freq_modulation.apply_alpha_modulation(sound, alpha_freq, 0.2)
            left, right = self.amp_modulation.apply_binaural_beats(sound, 
                                                                base_freq=200, 
                                                                beat_freq=alpha_freq, 
                                                                mix_ratio=0.15)
            
        elif sleep_stage == 'N1':
            # N1 수면 단계: 세타파 리듬 강화 (알파파에서 세타파로 전이)
            theta_freq = (alpha_freq + delta_freq) / 2  # 약 6Hz 근처
            sound = self.freq_modulation.apply_alpha_modulation(sound, theta_freq, 0.25)
            left, right = self.amp_modulation.apply_binaural_beats(sound, 
                                                                base_freq=150, 
                                                                beat_freq=theta_freq, 
                                                                mix_ratio=0.2)
            
        elif sleep_stage in ['N2', 'N3']:
            # N2, N3 수면 단계: 델타파 리듬 강화
            sound = self.amp_modulation.apply_delta_modulation(sound, delta_freq, 0.35)
            # 낮은 볼륨의 바이노럴 비트
            left, right = self.amp_modulation.apply_binaural_beats(sound, 
                                                                base_freq=100, 
                                                                beat_freq=delta_freq, 
                                                                mix_ratio=0.1)
            
        elif sleep_stage == 'REM':
            # REM 수면 단계: 이명 억제에 집중
            # 이명 주파수에 가까운 주파수의 소리로 마스킹
            # 위상 반전을 통한 상쇄 간섭 유도
            t = np.linspace(0, duration, int(self.config.sample_rate * duration), False)
            mask_signal = np.sin(2 * np.pi * tinnitus_freq * t) * 0.15
            
            # 소리에 마스킹 신호 추가
            sound = sound * 0.85 + mask_signal
            
            # 약한 불규칙 변조로 청각 피질의 동기화 방해
            irregular_mod = np.random.normal(0, 0.05, len(sound))
            left = sound * (1 + irregular_mod)
            right = sound * (1 - irregular_mod)
        
        # 3. 신호 정규화 및 페이드아웃 적용
        max_val = max(np.max(np.abs(left)), np.max(np.abs(right)))
        left = left / max_val * 0.9
        right = right / max_val * 0.9
        
        # 페이드아웃 적용
        if self.config.fadeout_duration > 0:
            fadeout_samples = int(self.config.sample_rate * self.config.fadeout_duration)
            if fadeout_samples < len(left):
                fadeout = np.linspace(1.0, 0.0, fadeout_samples)
                left[-fadeout_samples:] *= fadeout
                right[-fadeout_samples:] *= fadeout
        
        # 4. 스테레오 신호 생성
        stereo = np.column_stack((left, right))
        
        return stereo
    
    def save_stimuli(self, stimuli: np.ndarray, filename: str, 
                  sample_rate: int = None) -> str:
        """
        자극 신호를 파일로 저장
        
        Args:
            stimuli: 자극 신호 (모노 또는 스테레오)
            filename: 파일명
            sample_rate: 샘플링 레이트 (Hz), None이면 설정값 사용
            
        Returns:
            저장된 파일 경로
        """
        if sample_rate is None:
            sample_rate = self.config.sample_rate
            
        # 파일 경로 생성
        if not filename.endswith(('.wav', '.flac')):
            filename += '.wav'
            
        filepath = os.path.join(self.config.output_dir, filename)
        
        # 파일 저장
        sf.write(filepath, stimuli, sample_rate)
        
        return filepath


# 예제 사용법
if __name__ == "__main__":
    # 설정 객체 생성
    config = AuditoryStimConfig(
        sample_rate=44100,
        duration=60.0,  # 60초 샘플
        fadeout_duration=5.0
    )
    
    # 자극 생성기 초기화
    generator = AuditoryStimGenerator(config)
    
    # 샘플 뇌파 데이터
    brain_data = {
        'alpha_freq': 10.2,  # Hz
        'delta_freq': 1.8    # Hz
    }
    
    # 샘플 이명 주파수
    tinnitus_freq = 5000.0  # Hz
    
    # 다양한 수면 단계에 맞춘 자극 생성
    for stage in ['awake', 'N1', 'N2', 'N3', 'REM']:
        stimuli = generator.create_personalized_stimuli(
            tinnitus_freq=tinnitus_freq,
            brain_data=brain_data,
            sound_type='pink_noise',
            duration=30.0,  # 30초 샘플
            sleep_stage=stage
        )
        
        # 자극 저장
        filepath = generator.save_stimuli(stimuli, f'tinnitus_therapy_{stage}')
        print(f'저장된 파일: {filepath}')
    
    # 노치 필터 응답 시각화
    generator.notch_filter.plot_filter_response(tinnitus_freq)
