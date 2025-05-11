"""
적응형 제어 통합 모듈

이 모듈은 강화학습 에이전트, 상태 추적기, 보상 함수 등을
통합하여 이명 치료 시스템의 적응형 제어를 담당합니다.
"""

import time
import logging
import threading
import json
import os
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from .config import RLConfig
from .action_space import ActionSpace
from .feature_extractor import StateFeatureExtractor
from .state_tracker import StateTracker
from .reward_function import RewardFunction
from .agent import RLAgent

# 로깅 설정
logger = logging.getLogger("adaptive_controller")


class AdaptiveController:
    """이명 치료 적응형 제어 컨트롤러"""
    
    def __init__(self, config_path: str = None):
        """
        초기화
        
        Args:
            config_path: 설정 파일 경로 (None이면 기본 설정 사용)
        """
        # 설정 로드
        self.config = self._load_config(config_path)
        
        # 하위 모듈 초기화
        self.action_space = ActionSpace()
        self.feature_extractor = StateFeatureExtractor(self.config)
        self.state_tracker = StateTracker(self.feature_extractor)
        self.reward_function = RewardFunction()
        self.agent = RLAgent(self.config, self.action_space)
        
        # 제어 상태
        self.is_running = False
        self.control_thread = None
        self.last_action_time = time.time()
        self.patient_id = None
        
        # 환자별 모델 저장 경로
        self.patient_model_dir = "models/patients/"
        
        logger.info("Adaptive Controller initialized")
    
    def start(self, patient_id: str) -> bool:
        """
        적응형 제어 시작
        
        Args:
            patient_id: 환자 식별자
            
        Returns:
            시작 성공 여부
        """
        if self.is_running:
            logger.warning("Adaptive controller is already running. Stop it first.")
            return False
        
        # 환자 ID 설정
        self.patient_id = patient_id
        
        # 환자별 모델 로드 시도
        patient_model_path = os.path.join(self.patient_model_dir, f"{patient_id}_model.pkl")
        if os.path.exists(patient_model_path):
            self.agent.load_model(patient_model_path)
            logger.info(f"Loaded model for patient {patient_id}")
        
        # 환자별 특성 통계 로드 시도
        self.state_tracker.load_features(patient_id)
        
        # 제어 시작
        self.is_running = True
        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()
        
        logger.info(f"Adaptive controller started for patient {patient_id}")
        return True
    
    def stop(self) -> bool:
        """
        적응형 제어 중지
        
        Returns:
            중지 성공 여부
        """
        if not self.is_running:
            logger.warning("Adaptive controller is not running")
            return False
        
        # 제어 중지
        self.is_running = False
        
        # 스레드 종료 대기
        if self.control_thread:
            self.control_thread.join(timeout=3.0)
        
        # 환자별 모델 저장
        if self.patient_id:
            patient_model_path = os.path.join(self.patient_model_dir, f"{self.patient_id}_model.pkl")
            os.makedirs(os.path.dirname(patient_model_path), exist_ok=True)
            self.agent.save_model(patient_model_path)
            
            # 특성 데이터 저장
            self.state_tracker.save_features(self.patient_id)
        
        logger.info("Adaptive controller stopped")
        return True
    
    def update_state(self, eeg_data: np.ndarray = None,
                  hrv_data: np.ndarray = None,
                  tinnitus_data: Dict = None,
                  sleep_data: Dict = None) -> Dict:
        """
        환자 상태 업데이트
        
        Args:
            eeg_data: EEG 데이터
            hrv_data: 심박 변이도 데이터
            tinnitus_data: 이명 특성 데이터
            sleep_data: 수면 상태 데이터
            
        Returns:
            현재 상태
        """
        # 상태 업데이트
        current_state = self.state_tracker.update_state(
            eeg_data=eeg_data,
            hrv_data=hrv_data,
            tinnitus_data=tinnitus_data,
            sleep_data=sleep_data
        )
        
        logger.debug("Patient state updated")
        return current_state
    
    def get_action(self, sleep_stage: str = None) -> Dict:
        """
        최적 자극 액션 얻기
        
        Args:
            sleep_stage: 수면 단계
            
        Returns:
            자극 설정 딕셔너리
        """
        # 현재 상태
        current_state = self.state_tracker.get_current_state()
        
        # 상태가 없으면 기본 액션 반환
        if current_state is None:
            if sleep_stage:
                return self.action_space.get_sleep_optimized_action(sleep_stage)
            else:
                return self.action_space.sample_random_action()
        
        # 액션 선택
        action = self.agent.select_action(current_state, sleep_stage)
        
        # 액션 기록
        self.state_tracker.record_action(action)
        
        # 액션 타임스탬프 갱신
        self.last_action_time = time.time()
        
        logger.debug(f"Selected action: {action}")
        return action
    
    def provide_feedback(self, feedback: Dict[str, Any]) -> float:
        """
        환자 피드백 제공
        
        Args:
            feedback: 환자 피드백 데이터
            
        Returns:
            계산된 보상 값
        """
        # 현재 및 이전 상태
        current_state = self.state_tracker.get_current_state()
        previous_state = self.state_tracker.get_previous_state()
        
        # 현재 및 이전 액션
        current_action = self.state_tracker.get_current_action()
        previous_action = self.state_tracker.get_previous_action()
        
        # 보상 계산
        reward = self.reward_function.calculate_reward(
            current_state=current_state,
            previous_state=previous_state,
            action=current_action,
            previous_action=previous_action,
            feedback=feedback
        )
        
        # 보상 기록
        self.state_tracker.record_reward(reward)
        
        # 피드백 기록
        self.state_tracker.record_feedback(feedback)
        
        # 에이전트 경험 저장
        if previous_state is not None and previous_action is not None:
            self.agent.store_experience(
                state=previous_state,
                action=previous_action,
                reward=reward,
                next_state=current_state,
                done=False
            )
        
        logger.debug(f"Feedback processed. Reward: {reward}")
        return reward
    
    def train(self) -> Dict[str, float]:
        """
        에이전트 모델 학습
        
        Returns:
            학습 통계 딕셔너리
        """
        # 에이전트 학습
        train_stats = self.agent.train(self.state_tracker, self.reward_function)
        
        # 타겟 네트워크 업데이트
        self.agent.update_target_network()
        
        logger.debug(f"Model training completed. Stats: {train_stats}")
        return train_stats
    
    def save_model(self, filepath: str = None) -> bool:
        """
        모델 저장
        
        Args:
            filepath: 저장 경로 (None이면 기본 경로 사용)
            
        Returns:
            저장 성공 여부
        """
        return self.agent.save_model(filepath)
    
    def load_model(self, filepath: str = None) -> bool:
        """
        모델 로드
        
        Args:
            filepath: 로드 경로 (None이면 기본 경로 사용)
            
        Returns:
            로드 성공 여부
        """
        return self.agent.load_model(filepath)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        제어 통계 정보 얻기
        
        Returns:
            통계 정보 딕셔너리
        """
        # 에피소드 데이터
        states, actions, rewards = self.state_tracker.get_episode_data()
        
        # 보상 통계
        reward_stats = {
            'mean': np.mean(rewards) if rewards else 0.0,
            'std': np.std(rewards) if rewards else 0.0,
            'min': np.min(rewards) if rewards else 0.0,
            'max': np.max(rewards) if rewards else 0.0,
            'recent': rewards[-5:] if rewards else []
        }
        
        # 특성 중요도
        feature_importance = self.agent.get_feature_importance()
        important_features = {k: v for k, v in feature_importance.items() 
                             if v >= self.config.feature_importance_threshold}
        
        # 전체 통계
        stats = {
            'rewards': reward_stats,
            'exploration_rate': self.agent.exploration_rate,
            'episode_length': len(rewards),
            'feature_importance': important_features,
            'agent_type': 'DQN' if hasattr(self.agent, 'model') else 'Q-Table'
        }
        
        return stats
    
    def _control_loop(self) -> None:
        """
        적응형 제어 루프 (별도 스레드에서 실행)
        """
        try:
            logger.info("Control loop started")
            
            training_interval = 60  # 1분마다 학습
            save_interval = 600     # 10분마다 모델 저장
            
            last_train_time = time.time()
            last_save_time = time.time()
            
            while self.is_running:
                current_time = time.time()
                
                # 주기적 학습
                if current_time - last_train_time >= training_interval:
                    self.train()
                    last_train_time = current_time
                
                # 주기적 저장
                if self.patient_id and current_time - last_save_time >= save_interval:
                    patient_model_path = os.path.join(self.patient_model_dir, f"{self.patient_id}_model.pkl")
                    self.agent.save_model(patient_model_path)
                    last_save_time = current_time
                
                # CPU 사용량 최소화
                time.sleep(1.0)
                
        except Exception as e:
            logger.error(f"Error in control loop: {str(e)}")
            self.is_running = False
    
    def _load_config(self, config_path: str = None) -> RLConfig:
        """
        설정 로드
        
        Args:
            config_path: 설정 파일 경로
            
        Returns:
            설정 객체
        """
        # 기본 설정 생성
        config = RLConfig()
        
        # 설정 파일이 있으면 로드
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # 설정 업데이트
                for key, value in config_data.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
                
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration: {str(e)}")
        
        return config
