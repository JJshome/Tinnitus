"""
자극 액션 공간 모듈

이 모듈은 강화학습 에이전트가 선택할 수 있는 자극 액션들의 공간을 정의합니다.
청각, 시각, 촉각 자극의 다양한 파라미터 조합을 관리합니다.
"""

import random
import logging
from typing import Dict, List, Tuple, Any

# 로깅 설정
logger = logging.getLogger("action_space")


class ActionSpace:
    """자극 액션 공간"""
    
    def __init__(self):
        """액션 공간 초기화"""
        # 청각 자극 파라미터 공간
        self.audio_params = {
            'volume': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            'freq_mod': ['none', 'delta', 'theta', 'alpha', 'custom'],
            'amp_mod': ['none', 'low', 'medium', 'high', 'irregular']
        }
        
        # 시각 자극 파라미터 공간
        self.visual_params = {
            'brightness': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            'pattern': ['off', 'calm', 'transient', 'subtle'],
            'color': ['none', 'white', 'blue', 'amber', 'red', 'deep_red', 'green']
        }
        
        # 촉각 자극 파라미터 공간
        self.tactile_params = {
            'intensity': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            'frequency': [0, 2, 4, 6, 8, 10, 'irregular']
        }
        
        # 액션 공간 계산
        self.audio_space_size = (
            len(self.audio_params['volume']) *
            len(self.audio_params['freq_mod']) *
            len(self.audio_params['amp_mod'])
        )
        
        self.visual_space_size = (
            len(self.visual_params['brightness']) *
            len(self.visual_params['pattern']) *
            len(self.visual_params['color'])
        )
        
        self.tactile_space_size = (
            len(self.tactile_params['intensity']) *
            len(self.tactile_params['frequency'])
        )
        
        # 총 액션 공간 크기 (모든 조합)
        self.total_space_size = (
            self.audio_space_size *
            self.visual_space_size *
            self.tactile_space_size
        )
        
        logger.info(f"Action space initialized with {self.total_space_size} possible combinations")
        logger.info(f"Audio space: {self.audio_space_size}, Visual space: {self.visual_space_size}, Tactile space: {self.tactile_space_size}")
    
    def sample_random_action(self) -> Dict:
        """
        무작위 액션 샘플링
        
        Returns:
            자극 설정 딕셔너리
        """
        action = {
            'audio': {
                'volume': random.choice(self.audio_params['volume']),
                'freq_mod': random.choice(self.audio_params['freq_mod']),
                'amp_mod': random.choice(self.audio_params['amp_mod'])
            },
            'visual': {
                'brightness': random.choice(self.visual_params['brightness']),
                'pattern': random.choice(self.visual_params['pattern']),
                'color': random.choice(self.visual_params['color'])
            },
            'tactile': {
                'intensity': random.choice(self.tactile_params['intensity']),
                'frequency': random.choice(self.tactile_params['frequency'])
            }
        }
        
        return action
    
    def encode_action(self, action: Dict) -> int:
        """
        액션을 정수 인덱스로 인코딩
        
        Args:
            action: 자극 설정 딕셔너리
            
        Returns:
            액션 인덱스
        """
        # 각 파라미터의 인덱스 찾기
        audio_vol_idx = self.audio_params['volume'].index(action['audio']['volume'])
        audio_freq_idx = self.audio_params['freq_mod'].index(action['audio']['freq_mod'])
        audio_amp_idx = self.audio_params['amp_mod'].index(action['audio']['amp_mod'])
        
        visual_bright_idx = self.visual_params['brightness'].index(action['visual']['brightness'])
        visual_pattern_idx = self.visual_params['pattern'].index(action['visual']['pattern'])
        visual_color_idx = self.visual_params['color'].index(action['visual']['color'])
        
        tactile_intensity_idx = self.tactile_params['intensity'].index(action['tactile']['intensity'])
        tactile_freq_idx = self.tactile_params['frequency'].index(action['tactile']['frequency'])
        
        # 인덱스 계산 (다차원 배열의 1차원 인덱스 계산 방식)
        audio_idx = (
            audio_vol_idx * len(self.audio_params['freq_mod']) * len(self.audio_params['amp_mod']) +
            audio_freq_idx * len(self.audio_params['amp_mod']) +
            audio_amp_idx
        )
        
        visual_idx = (
            visual_bright_idx * len(self.visual_params['pattern']) * len(self.visual_params['color']) +
            visual_pattern_idx * len(self.visual_params['color']) +
            visual_color_idx
        )
        
        tactile_idx = (
            tactile_intensity_idx * len(self.tactile_params['frequency']) +
            tactile_freq_idx
        )
        
        # 최종 인덱스 계산
        action_idx = (
            audio_idx * self.visual_space_size * self.tactile_space_size +
            visual_idx * self.tactile_space_size +
            tactile_idx
        )
        
        return action_idx
    
    def decode_action(self, action_idx: int) -> Dict:
        """
        인덱스를 액션으로 디코딩
        
        Args:
            action_idx: 액션 인덱스
            
        Returns:
            자극 설정 딕셔너리
        """
        # 유효한 인덱스 범위 확인
        if action_idx < 0 or action_idx >= self.total_space_size:
            logger.warning(f"Invalid action index: {action_idx}, using random action instead")
            return self.sample_random_action()
        
        # 각 모듈별 인덱스 계산
        tactile_size = self.tactile_space_size
        visual_size = self.visual_space_size
        
        audio_idx = action_idx // (visual_size * tactile_size)
        remainder = action_idx % (visual_size * tactile_size)
        
        visual_idx = remainder // tactile_size
        tactile_idx = remainder % tactile_size
        
        # 오디오 파라미터 인덱스 계산
        audio_amp_size = len(self.audio_params['amp_mod'])
        audio_freq_size = len(self.audio_params['freq_mod'])
        
        audio_vol_idx = audio_idx // (audio_freq_size * audio_amp_size)
        audio_remainder = audio_idx % (audio_freq_size * audio_amp_size)
        
        audio_freq_idx = audio_remainder // audio_amp_size
        audio_amp_idx = audio_remainder % audio_amp_size
        
        # 시각 파라미터 인덱스 계산
        visual_color_size = len(self.visual_params['color'])
        visual_pattern_size = len(self.visual_params['pattern'])
        
        visual_bright_idx = visual_idx // (visual_pattern_size * visual_color_size)
        visual_remainder = visual_idx % (visual_pattern_size * visual_color_size)
        
        visual_pattern_idx = visual_remainder // visual_color_size
        visual_color_idx = visual_remainder % visual_color_size
        
        # 촉각 파라미터 인덱스 계산
        tactile_freq_size = len(self.tactile_params['frequency'])
        
        tactile_intensity_idx = tactile_idx // tactile_freq_size
        tactile_freq_idx = tactile_idx % tactile_freq_size
        
        # 액션 딕셔너리 구성
        action = {
            'audio': {
                'volume': self.audio_params['volume'][audio_vol_idx],
                'freq_mod': self.audio_params['freq_mod'][audio_freq_idx],
                'amp_mod': self.audio_params['amp_mod'][audio_amp_idx]
            },
            'visual': {
                'brightness': self.visual_params['brightness'][visual_bright_idx],
                'pattern': self.visual_params['pattern'][visual_pattern_idx],
                'color': self.visual_params['color'][visual_color_idx]
            },
            'tactile': {
                'intensity': self.tactile_params['intensity'][tactile_intensity_idx],
                'frequency': self.tactile_params['frequency'][tactile_freq_idx]
            }
        }
        
        return action
    
    def get_sleep_optimized_action(self, sleep_stage: str) -> Dict:
        """
        수면 단계에 최적화된 자극 액션 반환
        
        Args:
            sleep_stage: 수면 단계 ('awake', 'N1', 'N2', 'N3', 'REM')
            
        Returns:
            자극 설정 딕셔너리
        """
        # 수면 단계별 최적 자극 설정
        if sleep_stage == 'awake':
            action = {
                'audio': {
                    'volume': 0.7,
                    'freq_mod': 'alpha',
                    'amp_mod': 'low'
                },
                'visual': {
                    'brightness': 0.5,
                    'pattern': 'calm',
                    'color': 'blue'
                },
                'tactile': {
                    'intensity': 0.5,
                    'frequency': 10
                }
            }
        elif sleep_stage == 'N1':
            action = {
                'audio': {
                    'volume': 0.5,
                    'freq_mod': 'theta',
                    'amp_mod': 'medium'
                },
                'visual': {
                    'brightness': 0.3,
                    'pattern': 'transient',
                    'color': 'amber'
                },
                'tactile': {
                    'intensity': 0.4,
                    'frequency': 6
                }
            }
        elif sleep_stage == 'N2':
            action = {
                'audio': {
                    'volume': 0.3,
                    'freq_mod': 'delta',
                    'amp_mod': 'high'
                },
                'visual': {
                    'brightness': 0.1,
                    'pattern': 'off',
                    'color': 'none'
                },
                'tactile': {
                    'intensity': 0.2,
                    'frequency': 4
                }
            }
        elif sleep_stage == 'N3':
            action = {
                'audio': {
                    'volume': 0.2,
                    'freq_mod': 'delta',
                    'amp_mod': 'high'
                },
                'visual': {
                    'brightness': 0.0,
                    'pattern': 'off',
                    'color': 'none'
                },
                'tactile': {
                    'intensity': 0.1,
                    'frequency': 2
                }
            }
        elif sleep_stage == 'REM':
            action = {
                'audio': {
                    'volume': 0.4,
                    'freq_mod': 'custom',
                    'amp_mod': 'irregular'
                },
                'visual': {
                    'brightness': 0.1,
                    'pattern': 'subtle',
                    'color': 'deep_red'
                },
                'tactile': {
                    'intensity': 0.3,
                    'frequency': 'irregular'
                }
            }
        else:
            # 알 수 없는 단계는 N1 설정 사용
            action = {
                'audio': {
                    'volume': 0.5,
                    'freq_mod': 'theta',
                    'amp_mod': 'medium'
                },
                'visual': {
                    'brightness': 0.3,
                    'pattern': 'transient',
                    'color': 'amber'
                },
                'tactile': {
                    'intensity': 0.4,
                    'frequency': 6
                }
            }
        
        return action
