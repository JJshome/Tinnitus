"""
보상 함수 모듈

이 모듈은 환자의 상태 변화와 이명 증상 변화에 기반하여
강화학습 에이전트를 위한 보상 신호를 계산합니다.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging

# 로깅 설정
logger = logging.getLogger("reward_function")


class RewardFunction:
    """보상 함수 클래스"""
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        초기화
        
        Args:
            weights: 보상 계산에 사용되는 가중치 딕셔너리
        """
        # 기본 가중치 설정
        self.default_weights = {
            'tinnitus_intensity': -0.5,  # 이명 강도 감소가 주요 목표
            'tinnitus_annoyance': -0.3,  # 이명으로 인한 불편함 감소
            'sleep_efficiency': 0.3,     # 수면 효율 증가
            'eeg_rel_alpha': 0.2,        # 알파파 상대 파워 증가
            'hrv_lf_hf_ratio': -0.2,     # 자율신경계 균형 개선
            'comfort': 0.3,              # 환자의 편안함
            'duration_adjustment': 0.1,  # 자극 지속시간에 따른 조정
            'action_similarity': -0.1    # 급격한 자극 변화 페널티
        }
        
        # 사용자 정의 가중치가 있는 경우 업데이트
        if weights:
            for key, value in weights.items():
                if key in self.default_weights:
                    self.default_weights[key] = value
    
    def calculate_reward(self, 
                        current_state: Dict[str, float],
                        previous_state: Dict[str, float],
                        action: Dict, 
                        previous_action: Dict = None,
                        feedback: Dict[str, float] = None) -> float:
        """
        보상 계산
        
        Args:
            current_state: 현재 환자 상태 특성
            previous_state: 이전 환자 상태 특성
            action: 현재 취한 자극 액션
            previous_action: 이전 자극 액션 (None인 경우 무시)
            feedback: 환자 피드백 데이터 (None인 경우 무시)
            
        Returns:
            계산된 보상 값
        """
        reward = 0.0
        
        # 상태 변화 기반 보상 계산
        state_reward = self._calculate_state_reward(current_state, previous_state)
        
        # 액션 기반 보상 계산
        action_reward = self._calculate_action_reward(action, previous_action)
        
        # 환자 피드백 기반 보상 계산
        feedback_reward = self._calculate_feedback_reward(feedback) if feedback else 0.0
        
        # 총 보상 계산
        reward = state_reward + action_reward + feedback_reward
        
        logger.debug(f"Reward components: state={state_reward:.2f}, action={action_reward:.2f}, feedback={feedback_reward:.2f}")
        logger.info(f"Total reward: {reward:.4f}")
        
        return reward
    
    def _calculate_state_reward(self, current_state: Dict[str, float], 
                              previous_state: Dict[str, float]) -> float:
        """
        상태 변화 기반 보상 계산
        
        Args:
            current_state: 현재 환자 상태 특성
            previous_state: 이전 환자 상태 특성
            
        Returns:
            상태 변화 기반 보상
        """
        reward = 0.0
        
        # 상태 키가 없는 경우 대비
        if not previous_state or not current_state:
            return 0.0
        
        # 주요 지표들에 대한 변화 계산 및 가중치 적용
        for key, weight in self.default_weights.items():
            # 'action_similarity'와 'duration_adjustment'는 별도 처리
            if key in ['action_similarity', 'duration_adjustment']:
                continue
                
            if key in current_state and key in previous_state:
                # 변화량 계산
                delta = current_state[key] - previous_state[key]
                
                # 가중치에 변화량을 곱해 보상 계산
                reward += weight * delta
        
        return reward
    
    def _calculate_action_reward(self, action: Dict, previous_action: Dict = None) -> float:
        """
        액션 기반 보상 계산
        
        Args:
            action: 현재 취한 자극 액션
            previous_action: 이전 자극 액션
            
        Returns:
            액션 기반 보상
        """
        reward = 0.0
        
        # 이전 액션이 없으면 액션 유사성 검사 생략
        if previous_action is None:
            return reward
        
        # 액션 유사성 계산 (급격한 변화에 페널티)
        action_similarity_weight = self.default_weights.get('action_similarity', -0.1)
        similarity_score = self._calculate_action_similarity(action, previous_action)
        reward += action_similarity_weight * (1.0 - similarity_score)  # 유사성이 낮을수록 페널티
        
        # 지속 시간에 따른 조정 (미세 조정을 위한 부분)
        duration_weight = self.default_weights.get('duration_adjustment', 0.1)
        
        # 예: 자극 볼륨의 지속시간에 따른 조정
        # 낮은 볼륨이 장시간 유지될 경우 보상 감소
        audio_volume = action.get('audio', {}).get('volume', 0)
        if audio_volume < 0.3:  # 낮은 볼륨
            reward -= duration_weight  # 지속 시간에 따른 페널티
        
        return reward
    
    def _calculate_feedback_reward(self, feedback: Dict[str, float]) -> float:
        """
        환자 피드백 기반 보상 계산
        
        Args:
            feedback: 환자 피드백 데이터
            
        Returns:
            피드백 기반 보상
        """
        reward = 0.0
        
        # 환자 피드백이 없는 경우
        if not feedback:
            return reward
        
        # 이명 강도 감소 피드백
        if 'tinnitus_intensity_change' in feedback:
            intensity_weight = self.default_weights.get('tinnitus_intensity', -0.5)
            reward += -intensity_weight * feedback['tinnitus_intensity_change']  # 감소할수록 보상 증가
        
        # 편안함 피드백
        if 'comfort' in feedback:
            comfort_weight = self.default_weights.get('comfort', 0.3)
            reward += comfort_weight * feedback['comfort']  # 높을수록 보상 증가
        
        # 수면 개선 피드백
        if 'sleep_improvement' in feedback:
            sleep_weight = self.default_weights.get('sleep_efficiency', 0.3)
            reward += sleep_weight * feedback['sleep_improvement']  # 개선될수록 보상 증가
        
        return reward
    
    def _calculate_action_similarity(self, action: Dict, previous_action: Dict) -> float:
        """
        현재 액션과 이전 액션의 유사성 계산
        
        Args:
            action: 현재 액션
            previous_action: 이전 액션
            
        Returns:
            유사성 점수 (0에서 1 사이, 1이 완전 동일)
        """
        # 유사성 점수 초기화
        similarity = 0.0
        total_factors = 0
        
        # 청각 자극 유사성
        if 'audio' in action and 'audio' in previous_action:
            audio_sim = 0.0
            audio_factors = 0
            
            # 볼륨 유사성
            if 'volume' in action['audio'] and 'volume' in previous_action['audio']:
                vol_diff = abs(action['audio']['volume'] - previous_action['audio']['volume'])
                audio_sim += 1.0 - min(vol_diff / 0.8, 1.0)  # 최대 볼륨 차이를 0.8로 가정
                audio_factors += 1
            
            # 주파수 조절 유사성
            if 'freq_mod' in action['audio'] and 'freq_mod' in previous_action['audio']:
                freq_sim = 1.0 if action['audio']['freq_mod'] == previous_action['audio']['freq_mod'] else 0.0
                audio_sim += freq_sim
                audio_factors += 1
            
            # 진폭 조절 유사성
            if 'amp_mod' in action['audio'] and 'amp_mod' in previous_action['audio']:
                amp_sim = 1.0 if action['audio']['amp_mod'] == previous_action['audio']['amp_mod'] else 0.0
                audio_sim += amp_sim
                audio_factors += 1
            
            # 청각 자극 전체 유사성
            if audio_factors > 0:
                similarity += audio_sim / audio_factors
                total_factors += 1
        
        # 시각 자극 유사성
        if 'visual' in action and 'visual' in previous_action:
            visual_sim = 0.0
            visual_factors = 0
            
            # 밝기 유사성
            if 'brightness' in action['visual'] and 'brightness' in previous_action['visual']:
                bright_diff = abs(action['visual']['brightness'] - previous_action['visual']['brightness'])
                visual_sim += 1.0 - min(bright_diff / 0.6, 1.0)  # 최대 밝기 차이를 0.6으로 가정
                visual_factors += 1
            
            # 패턴 유사성
            if 'pattern' in action['visual'] and 'pattern' in previous_action['visual']:
                pattern_sim = 1.0 if action['visual']['pattern'] == previous_action['visual']['pattern'] else 0.0
                visual_sim += pattern_sim
                visual_factors += 1
            
            # 색상 유사성
            if 'color' in action['visual'] and 'color' in previous_action['visual']:
                color_sim = 1.0 if action['visual']['color'] == previous_action['visual']['color'] else 0.0
                visual_sim += color_sim
                visual_factors += 1
            
            # 시각 자극 전체 유사성
            if visual_factors > 0:
                similarity += visual_sim / visual_factors
                total_factors += 1
        
        # 촉각 자극 유사성
        if 'tactile' in action and 'tactile' in previous_action:
            tactile_sim = 0.0
            tactile_factors = 0
            
            # 강도 유사성
            if 'intensity' in action['tactile'] and 'intensity' in previous_action['tactile']:
                intensity_diff = abs(action['tactile']['intensity'] - previous_action['tactile']['intensity'])
                tactile_sim += 1.0 - min(intensity_diff / 0.5, 1.0)  # 최대 강도 차이를 0.5로 가정
                tactile_factors += 1
            
            # 주파수 유사성
            if 'frequency' in action['tactile'] and 'frequency' in previous_action['tactile']:
                # 불규칙 패턴의 경우 처리
                if action['tactile']['frequency'] == 'irregular' or previous_action['tactile']['frequency'] == 'irregular':
                    freq_sim = 1.0 if action['tactile']['frequency'] == previous_action['tactile']['frequency'] else 0.0
                else:
                    # 숫자 주파수인 경우
                    try:
                        freq_diff = abs(float(action['tactile']['frequency']) - float(previous_action['tactile']['frequency']))
                        freq_sim = 1.0 - min(freq_diff / 10.0, 1.0)  # 최대 주파수 차이를 10Hz로 가정
                    except (ValueError, TypeError):
                        freq_sim = 0.0
                
                tactile_sim += freq_sim
                tactile_factors += 1
            
            # 촉각 자극 전체 유사성
            if tactile_factors > 0:
                similarity += tactile_sim / tactile_factors
                total_factors += 1
        
        # 최종 유사성 계산
        return similarity / max(total_factors, 1)
