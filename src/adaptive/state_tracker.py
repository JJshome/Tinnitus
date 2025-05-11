"""
상태 추적 모듈

이 모듈은 환자의 상태와 과거 이력, 자극 반응을 추적하고
이를 강화학습 에이전트에 제공하는 기능을 담당합니다.
"""

import time
import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
from .feature_extractor import StateFeatureExtractor

# 로깅 설정
logger = logging.getLogger("state_tracker")


class StateTracker:
    """환자 상태 추적기"""
    
    def __init__(self, feature_extractor: StateFeatureExtractor, 
                history_size: int = 48, 
                feature_storage_path: str = "data/features/"):
        """
        초기화
        
        Args:
            feature_extractor: 상태 특성 추출기 인스턴스
            history_size: 유지할 상태 이력의 크기
            feature_storage_path: 특성 저장 경로
        """
        self.feature_extractor = feature_extractor
        self.history_size = history_size
        self.feature_storage_path = feature_storage_path
        
        # 상태 이력 초기화
        self.state_history = deque(maxlen=history_size)
        self.action_history = deque(maxlen=history_size)
        self.reward_history = deque(maxlen=history_size)
        self.feedback_history = deque(maxlen=history_size)
        self.timestamp_history = deque(maxlen=history_size)
        
        # 특성 정규화를 위한 통계
        self.feature_means = {}
        self.feature_stds = {}
        
        # 현재 상태와 액션
        self.current_state = None
        self.current_action = None
        
        # 로깅
        logger.info(f"StateTracker initialized with history size: {history_size}")
    
    def update_state(self, eeg_data: np.ndarray = None,
                  hrv_data: np.ndarray = None,
                  tinnitus_data: Dict = None,
                  sleep_data: Dict = None) -> Dict[str, float]:
        """
        환자 상태 업데이트
        
        Args:
            eeg_data: EEG 데이터
            hrv_data: 심박 변이도 데이터
            tinnitus_data: 이명 특성 데이터
            sleep_data: 수면 상태 데이터
            
        Returns:
            정규화된 상태 특성 딕셔너리
        """
        # 이전 상태 저장
        previous_state = self.current_state
        
        # 특성 추출
        features = self.feature_extractor.extract_features(
            eeg_data=eeg_data,
            hrv_data=hrv_data,
            tinnitus_data=tinnitus_data,
            sleep_data=sleep_data
        )
        
        # 특성 정규화
        normalized_features, self.feature_means, self.feature_stds = self.feature_extractor.normalize_features(
            features=features,
            feature_means=self.feature_means,
            feature_stds=self.feature_stds
        )
        
        # 현재 상태 업데이트
        self.current_state = normalized_features
        
        # 타임스탬프 기록
        timestamp = time.time()
        
        # 이력 업데이트
        if previous_state is not None:
            self.state_history.append(previous_state)
            self.timestamp_history.append(timestamp)
        
        logger.debug(f"State updated with {len(features)} features")
        
        return normalized_features
    
    def record_action(self, action: Dict) -> None:
        """
        자극 액션 기록
        
        Args:
            action: 자극 설정 딕셔너리
        """
        # 이전 액션 저장
        previous_action = self.current_action
        
        # 현재 액션 업데이트
        self.current_action = action
        
        # 이력 업데이트
        if previous_action is not None:
            self.action_history.append(previous_action)
        
        logger.debug(f"Action recorded: {action}")
    
    def record_reward(self, reward: float) -> None:
        """
        보상 기록
        
        Args:
            reward: 보상 값
        """
        # 이력 업데이트
        self.reward_history.append(reward)
        
        logger.debug(f"Reward recorded: {reward}")
    
    def record_feedback(self, feedback: Dict[str, Any]) -> None:
        """
        환자 피드백 기록
        
        Args:
            feedback: 환자 피드백 데이터
        """
        # 이력 업데이트
        self.feedback_history.append(feedback)
        
        logger.debug(f"Feedback recorded: {feedback}")
    
    def get_current_state(self) -> Dict[str, float]:
        """
        현재 상태 반환
        
        Returns:
            현재 상태 특성 딕셔너리
        """
        return self.current_state
    
    def get_previous_state(self) -> Optional[Dict[str, float]]:
        """
        이전 상태 반환
        
        Returns:
            이전 상태 특성 딕셔너리 또는 None
        """
        if len(self.state_history) > 0:
            return self.state_history[-1]
        return None
    
    def get_current_action(self) -> Dict:
        """
        현재 액션 반환
        
        Returns:
            현재 자극 액션 딕셔너리
        """
        return self.current_action
    
    def get_previous_action(self) -> Optional[Dict]:
        """
        이전 액션 반환
        
        Returns:
            이전 자극 액션 딕셔너리 또는 None
        """
        if len(self.action_history) > 0:
            return self.action_history[-1]
        return None
    
    def get_episode_data(self, episode_length: int = 24) -> Tuple[List, List, List]:
        """
        에피소드 데이터 반환
        
        Args:
            episode_length: 에피소드 길이
            
        Returns:
            상태, 액션, 보상 리스트의 튜플
        """
        # 최대 가능한 길이 계산
        max_length = min(len(self.state_history), len(self.action_history), len(self.reward_history))
        length = min(max_length, episode_length)
        
        if length == 0:
            return [], [], []
        
        # 가장 최근 데이터부터 필요한 길이만큼 추출
        states = list(self.state_history)[-length:]
        actions = list(self.action_history)[-length:]
        rewards = list(self.reward_history)[-length:]
        
        return states, actions, rewards
    
    def get_time_window_data(self, window_hours: int = 24) -> Tuple[List, List, List, List]:
        """
        시간 윈도우 데이터 반환
        
        Args:
            window_hours: 윈도우 시간 (시간 단위)
            
        Returns:
            상태, 액션, 보상, 타임스탬프 리스트의 튜플
        """
        if len(self.timestamp_history) == 0:
            return [], [], [], []
        
        # 현재 시간 기준으로 window_hours 이내의 데이터 필터링
        current_time = time.time()
        window_seconds = window_hours * 3600
        
        # 데이터와 타임스탬프 함께 저장
        data = list(zip(
            self.state_history, 
            self.action_history, 
            self.reward_history,
            self.timestamp_history
        ))
        
        # 타임스탬프 기준 필터링
        filtered_data = [(s, a, r, t) for s, a, r, t in data if current_time - t <= window_seconds]
        
        # 결과 분리
        if filtered_data:
            states, actions, rewards, timestamps = zip(*filtered_data)
            return list(states), list(actions), list(rewards), list(timestamps)
        else:
            return [], [], [], []
    
    def get_recent_trend(self, metric: str, window_size: int = 10) -> Tuple[float, float]:
        """
        최근 트렌드 계산
        
        Args:
            metric: 트렌드를 계산할 지표 이름
            window_size: 트렌드 계산 윈도우 크기
            
        Returns:
            평균값, 변화 기울기의 튜플
        """
        # 해당 지표의 최근 값들 추출
        values = []
        
        # 상태 이력에서 지표 값 추출
        for state in list(self.state_history)[-window_size:]:
            if metric in state:
                values.append(state[metric])
        
        # 충분한 데이터가 없는 경우
        if len(values) < 2:
            return 0.0, 0.0
        
        # 평균 계산
        avg_value = sum(values) / len(values)
        
        # 선형 회귀로 기울기 계산
        x = np.arange(len(values))
        A = np.vstack([x, np.ones(len(x))]).T
        
        try:
            # 최소 제곱법으로 회귀선 계산
            slope, _ = np.linalg.lstsq(A, values, rcond=None)[0]
        except np.linalg.LinAlgError:
            slope = 0.0
        
        return avg_value, slope
    
    def save_features(self, patient_id: str) -> bool:
        """
        특성 데이터 저장
        
        Args:
            patient_id: 환자 식별자
            
        Returns:
            저장 성공 여부
        """
        try:
            import os
            
            # 저장 디렉토리 확인 및 생성
            save_dir = os.path.join(self.feature_storage_path, patient_id)
            os.makedirs(save_dir, exist_ok=True)
            
            # 현재 타임스탬프 생성
            current_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            
            # 파일 경로 생성
            features_file = os.path.join(save_dir, f"features_{current_time}.json")
            stats_file = os.path.join(save_dir, "feature_stats.json")
            
            # 상태 이력 저장
            data_to_save = {
                "states": list(self.state_history),
                "actions": list(self.action_history),
                "rewards": list(self.reward_history),
                "feedback": list(self.feedback_history),
                "timestamps": list(self.timestamp_history)
            }
            
            with open(features_file, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=2)
            
            # 특성 통계 저장
            stats_to_save = {
                "feature_means": self.feature_means,
                "feature_stds": self.feature_stds
            }
            
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats_to_save, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Features saved for patient {patient_id}: {features_file}, {stats_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving features: {str(e)}")
            return False
    
    def load_features(self, patient_id: str) -> bool:
        """
        특성 데이터 로드
        
        Args:
            patient_id: 환자 식별자
            
        Returns:
            로드 성공 여부
        """
        try:
            import os
            
            # 로드 디렉토리 경로
            load_dir = os.path.join(self.feature_storage_path, patient_id)
            
            # 특성 통계 파일 경로
            stats_file = os.path.join(load_dir, "feature_stats.json")
            
            # 특성 통계 로드
            if os.path.exists(stats_file):
                with open(stats_file, 'r', encoding='utf-8') as f:
                    stats_data = json.load(f)
                
                self.feature_means = stats_data.get("feature_means", {})
                self.feature_stds = stats_data.get("feature_stds", {})
                
                logger.info(f"Feature statistics loaded for patient {patient_id}")
                return True
            else:
                logger.warning(f"Feature statistics file not found for patient {patient_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading features: {str(e)}")
            return False
