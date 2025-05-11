"""
강화학습 에이전트 모듈

이 모듈은 이명 치료를 위한 자극 액션을 선택하는 
강화학습 에이전트를 구현합니다.
"""

import os
import random
import numpy as np
import pickle
import logging
from typing import Dict, List, Tuple, Any, Optional
from collections import deque

# 외부 라이브러리
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    
from .config import RLConfig
from .action_space import ActionSpace
from .state_tracker import StateTracker
from .reward_function import RewardFunction

# 로깅 설정
logger = logging.getLogger("agent")


class ReplayMemory:
    """경험 재생 메모리"""
    
    def __init__(self, capacity: int):
        """
        초기화
        
        Args:
            capacity: 메모리 용량
        """
        self.memory = deque(maxlen=capacity)
    
    def push(self, state: Dict[str, float], action: Dict, 
           reward: float, next_state: Dict[str, float], 
           done: bool) -> None:
        """
        경험 저장
        
        Args:
            state: 현재 상태
            action: 취한 액션
            reward: 받은 보상
            next_state: 다음 상태
            done: 에피소드 종료 여부
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List:
        """
        경험 샘플링
        
        Args:
            batch_size: 배치 크기
            
        Returns:
            경험 배치
        """
        return random.sample(self.memory, min(len(self.memory), batch_size))
    
    def __len__(self) -> int:
        """메모리 크기 반환"""
        return len(self.memory)


# PyTorch 모델 정의 (PyTorch가 설치된 경우에만 사용)
if TORCH_AVAILABLE:
    class DQN(nn.Module):
        """Deep Q-Network 모델"""
        
        def __init__(self, input_size: int, output_size: int, hidden_size: int = 128):
            """
            초기화
            
            Args:
                input_size: 입력 차원 (상태 특성 수)
                output_size: 출력 차원 (액션 공간 크기)
                hidden_size: 은닉층 크기
            """
            super(DQN, self).__init__()
            
            self.network = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
            )
        
        def forward(self, x):
            """순전파"""
            return self.network(x)


class RLAgent:
    """강화학습 에이전트"""
    
    def __init__(self, config: RLConfig, action_space: ActionSpace):
        """
        초기화
        
        Args:
            config: 강화학습 설정
            action_space: 액션 공간
        """
        self.config = config
        self.action_space = action_space
        
        # 탐색 파라미터
        self.exploration_rate = config.exploration_rate
        self.min_exploration_rate = config.min_exploration_rate
        self.exploration_decay = config.exploration_decay
        
        # 학습 파라미터
        self.learning_rate = config.learning_rate
        self.discount_factor = config.discount_factor
        self.batch_size = config.batch_size
        
        # 경험 재생 메모리
        self.memory = ReplayMemory(config.memory_size)
        
        # 특성 중요도 임계값
        self.feature_importance_threshold = config.feature_importance_threshold
        
        # 모델 초기화
        self.initialize_model()
        
        logger.info("RL Agent initialized")
    
    def initialize_model(self) -> None:
        """모델 초기화"""
        # PyTorch를 사용할 경우 DQN 초기화
        if TORCH_AVAILABLE:
            logger.info("Using PyTorch for RL model")
            
            # 입력 차원 (예: 20개 특성)
            # 실제 구현에서는 실제 특성 수에 맞게 조정 필요
            input_size = 20
            
            # 출력 차원 (액션 공간 크기)
            output_size = self.action_space.total_space_size
            
            # 모델 초기화
            self.model = DQN(input_size, output_size)
            self.target_model = DQN(input_size, output_size)
            self.target_model.load_state_dict(self.model.state_dict())
            
            # 옵티마이저 설정
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
            # 손실 함수 설정
            self.criterion = nn.SmoothL1Loss()  # Huber 손실
            
            # 학습 장치 설정
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.target_model.to(self.device)
            
            logger.info(f"DQN initialized with input_size={input_size}, output_size={output_size}")
            logger.info(f"Using device: {self.device}")
            
        else:
            # PyTorch가 없는 경우, 간단한 Q-테이블 사용
            logger.info("PyTorch not available, using simple Q-table")
            
            # Q-테이블 초기화
            # 간소화를 위해 상태를 몇 개의 이산적인 버킷으로 나눔
            # 실제 구현에서는 더 정교한 상태 표현 방법 필요
            self.q_table = {}
    
    def state_to_tensor(self, state: Dict[str, float]) -> torch.Tensor:
        """
        상태를 텐서로 변환
        
        Args:
            state: 상태 딕셔너리
            
        Returns:
            상태 텐서
        """
        if not TORCH_AVAILABLE:
            return None
        
        # 상태 특성을 배열로 변환
        # 주의: 일관된 순서로 특성을 유지해야 함
        # 실제 구현에서는 미리 정의된 특성 목록 사용 권장
        feature_list = []
        
        # 이명 특성
        feature_list.append(state.get('tinnitus_freq', 0.0))
        feature_list.append(state.get('tinnitus_intensity', 0.0))
        feature_list.append(state.get('tinnitus_bandwidth', 0.0))
        feature_list.append(state.get('tinnitus_severity', 0.0))
        feature_list.append(state.get('tinnitus_annoyance', 0.0))
        
        # EEG 특성
        feature_list.append(state.get('eeg_rel_delta', 0.0))
        feature_list.append(state.get('eeg_rel_theta', 0.0))
        feature_list.append(state.get('eeg_rel_alpha', 0.0))
        feature_list.append(state.get('eeg_rel_beta', 0.0))
        feature_list.append(state.get('eeg_rel_gamma', 0.0))
        feature_list.append(state.get('eeg_theta_alpha_ratio', 0.0))
        
        # HRV 특성
        feature_list.append(state.get('hrv_sdnn', 0.0))
        feature_list.append(state.get('hrv_rmssd', 0.0))
        feature_list.append(state.get('hrv_lf_hf_ratio', 0.0))
        
        # 수면 특성
        feature_list.append(state.get('sleep_stage', 0.0))
        feature_list.append(state.get('sleep_efficiency', 0.0))
        feature_list.append(state.get('n3_percentage', 0.0))
        feature_list.append(state.get('rem_percentage', 0.0))
        
        # 기타 특성
        feature_list.append(state.get('hrv_mean_rr', 0.0))
        feature_list.append(state.get('sleep_awakenings', 0.0))
        
        # 텐서로 변환
        tensor = torch.FloatTensor(feature_list).unsqueeze(0)
        return tensor.to(self.device)
    
    def select_action(self, state: Dict[str, float], sleep_stage: str = None) -> Dict:
        """
        액션 선택
        
        Args:
            state: 현재 상태
            sleep_stage: 수면 단계 (None이면 무시)
            
        Returns:
            선택된 자극 액션
        """
        # 수면 단계가 주어진 경우 수면 최적화 액션 반환
        if sleep_stage is not None:
            return self.action_space.get_sleep_optimized_action(sleep_stage)
        
        # 탐색 여부 결정
        if random.random() < self.exploration_rate:
            # 무작위 탐색
            selected_action = self.action_space.sample_random_action()
            logger.debug("Exploration: Random action selected")
        else:
            # 최적 액션 선택
            if TORCH_AVAILABLE:
                # DQN을 이용한 액션 선택
                with torch.no_grad():
                    state_tensor = self.state_to_tensor(state)
                    q_values = self.model(state_tensor)
                    action_idx = q_values.max(1)[1].item()
                    selected_action = self.action_space.decode_action(action_idx)
            else:
                # 간단한 Q-테이블을 이용한 액션 선택
                state_key = self._get_state_key(state)
                
                if state_key in self.q_table:
                    # 최고 Q값을 가진 액션 선택
                    best_action_idx = max(self.q_table[state_key], key=self.q_table[state_key].get)
                    selected_action = self.action_space.decode_action(int(best_action_idx))
                else:
                    # Q값이 없는 상태는 무작위 액션
                    selected_action = self.action_space.sample_random_action()
            
            logger.debug("Exploitation: Optimal action selected")
        
        # 탐색률 감소
        self.exploration_rate = max(
            self.min_exploration_rate, 
            self.exploration_rate * self.exploration_decay
        )
        
        return selected_action
    
    def _get_state_key(self, state: Dict[str, float]) -> str:
        """
        상태를 키 문자열로 변환 (Q-테이블용)
        
        Args:
            state: 상태 딕셔너리
            
        Returns:
            상태 키 문자열
        """
        # 주요 특성만 선택하여 양자화
        key_parts = []
        
        # 이명 강도 (5개 구간으로 양자화)
        if 'tinnitus_intensity' in state:
            intensity_level = min(4, max(0, int(state['tinnitus_intensity'] * 5)))
            key_parts.append(f"i{intensity_level}")
        
        # 수면 단계 (5개 단계)
        if 'sleep_stage' in state:
            key_parts.append(f"s{int(state['sleep_stage'])}")
        
        # EEG 알파파 상대 파워 (4개 구간으로 양자화)
        if 'eeg_rel_alpha' in state:
            alpha_level = min(3, max(0, int(state['eeg_rel_alpha'] * 4)))
            key_parts.append(f"a{alpha_level}")
        
        # HRV LF/HF 비율 (4개 구간으로 양자화)
        if 'hrv_lf_hf_ratio' in state:
            lf_hf_level = min(3, max(0, int(min(3, state['hrv_lf_hf_ratio']) * 4/3)))
            key_parts.append(f"h{lf_hf_level}")
        
        return "_".join(key_parts)
    
    def train(self, state_tracker: StateTracker, reward_function: RewardFunction) -> Dict[str, float]:
        """
        모델 학습
        
        Args:
            state_tracker: 상태 추적기
            reward_function: 보상 함수
            
        Returns:
            학습 통계 딕셔너리
        """
        # 경험 메모리에 충분한 데이터가 없으면 학습 생략
        if len(self.memory) < self.batch_size:
            logger.debug(f"Skipping training: not enough samples in memory ({len(self.memory)} < {self.batch_size})")
            return {"loss": 0.0, "avg_q": 0.0, "max_q": 0.0, "min_q": 0.0}
        
        # 미니배치 샘플링
        batch = self.memory.sample(self.batch_size)
        
        if TORCH_AVAILABLE:
            # PyTorch DQN 학습
            return self._train_dqn(batch)
        else:
            # 간단한 Q-테이블 학습
            return self._train_q_table(batch)
    
    def _train_dqn(self, batch: List) -> Dict[str, float]:
        """
        DQN 모델 학습
        
        Args:
            batch: 경험 배치
            
        Returns:
            학습 통계 딕셔너리
        """
        # 배치 데이터 준비
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for state, action, reward, next_state, done in batch:
            states.append(self.state_to_tensor(state).squeeze(0))
            actions.append(self.action_space.encode_action(action))
            rewards.append(reward)
            next_states.append(self.state_to_tensor(next_state).squeeze(0))
            dones.append(float(done))
        
        # 텐서로 변환
        states = torch.stack(states)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.stack(next_states)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 현재 Q값
        current_q_values = self.model(states).gather(1, actions)
        
        # 타겟 Q값
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.discount_factor * next_q_values
        
        # 손실 계산
        loss = self.criterion(current_q_values, target_q_values.unsqueeze(1))
        
        # 역전파 및 최적화
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 학습 통계
        avg_q = current_q_values.mean().item()
        max_q = current_q_values.max().item()
        min_q = current_q_values.min().item()
        
        return {
            "loss": loss.item(),
            "avg_q": avg_q,
            "max_q": max_q,
            "min_q": min_q
        }
    
    def _train_q_table(self, batch: List) -> Dict[str, float]:
        """
        Q-테이블 학습
        
        Args:
            batch: 경험 배치
            
        Returns:
            학습 통계 딕셔너리
        """
        total_loss = 0.0
        q_values = []
        
        for state, action, reward, next_state, done in batch:
            # 상태 키 생성
            state_key = self._get_state_key(state)
            next_state_key = self._get_state_key(next_state)
            
            # 액션 인덱스
            action_idx = str(self.action_space.encode_action(action))
            
            # Q-테이블에 없는 상태/액션 초기화
            if state_key not in self.q_table:
                self.q_table[state_key] = {}
            
            if action_idx not in self.q_table[state_key]:
                self.q_table[state_key][action_idx] = 0.0
            
            # 다음 상태의 최대 Q값
            next_q_max = 0.0
            if not done and next_state_key in self.q_table:
                next_actions = self.q_table[next_state_key]
                if next_actions:
                    next_q_max = max(next_actions.values())
            
            # 현재 Q값
            current_q = self.q_table[state_key][action_idx]
            
            # 타겟 Q값
            target_q = reward + self.discount_factor * next_q_max * (1 - done)
            
            # Q값 업데이트
            new_q = current_q + self.learning_rate * (target_q - current_q)
            self.q_table[state_key][action_idx] = new_q
            
            # 손실 계산 (MSE)
            loss = (target_q - current_q) ** 2
            total_loss += loss
            
            # Q값 통계용
            q_values.append(new_q)
        
        # 학습 통계
        avg_loss = total_loss / len(batch)
        avg_q = sum(q_values) / len(q_values) if q_values else 0.0
        max_q = max(q_values) if q_values else 0.0
        min_q = min(q_values) if q_values else 0.0
        
        return {
            "loss": avg_loss,
            "avg_q": avg_q,
            "max_q": max_q,
            "min_q": min_q
        }
    
    def update_target_network(self) -> None:
        """타겟 네트워크 업데이트"""
        if TORCH_AVAILABLE:
            self.target_model.load_state_dict(self.model.state_dict())
            logger.debug("Target network updated")
    
    def store_experience(self, state: Dict[str, float], action: Dict, 
                      reward: float, next_state: Dict[str, float], 
                      done: bool) -> None:
        """
        경험 저장
        
        Args:
            state: 현재 상태
            action: 취한 액션
            reward: 받은 보상
            next_state: 다음 상태
            done: 에피소드 종료 여부
        """
        self.memory.push(state, action, reward, next_state, done)
    
    def save_model(self, filepath: str = None) -> bool:
        """
        모델 저장
        
        Args:
            filepath: 저장 경로 (None이면 기본 경로 사용)
            
        Returns:
            저장 성공 여부
        """
        if filepath is None:
            filepath = self.config.model_path
        
        try:
            # 저장 디렉토리 확인
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            if TORCH_AVAILABLE:
                # PyTorch 모델 저장
                model_state = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'exploration_rate': self.exploration_rate
                }
                torch.save(model_state, filepath)
            else:
                # Q-테이블 저장
                data = {
                    'q_table': self.q_table,
                    'exploration_rate': self.exploration_rate
                }
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f)
            
            logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, filepath: str = None) -> bool:
        """
        모델 로드
        
        Args:
            filepath: 로드 경로 (None이면 기본 경로 사용)
            
        Returns:
            로드 성공 여부
        """
        if filepath is None:
            filepath = self.config.model_path
        
        if not os.path.exists(filepath):
            logger.warning(f"Model file not found: {filepath}")
            return False
        
        try:
            if TORCH_AVAILABLE:
                # PyTorch 모델 로드
                checkpoint = torch.load(filepath, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.exploration_rate = checkpoint.get('exploration_rate', self.exploration_rate)
                
                # 타겟 네트워크 업데이트
                self.target_model.load_state_dict(self.model.state_dict())
            else:
                # Q-테이블 로드
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                
                self.q_table = data.get('q_table', {})
                self.exploration_rate = data.get('exploration_rate', self.exploration_rate)
            
            logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        특성 중요도 계산
        
        Returns:
            특성별 중요도 딕셔너리
        """
        # 간소화된 구현: Q-테이블 기반으로 상태 키에 사용된 특성 카운트
        if not TORCH_AVAILABLE:
            feature_counts = {}
            total_states = len(self.q_table)
            
            if total_states == 0:
                return {}
            
            # 각 상태 키 분석
            for state_key in self.q_table:
                parts = state_key.split('_')
                for part in parts:
                    feature_type = part[0]  # 첫 글자가 특성 유형
                    if feature_type in feature_counts:
                        feature_counts[feature_type] += 1
                    else:
                        feature_counts[feature_type] = 1
            
            # 특성 중요도 계산 (출현 빈도 기준)
            importance = {}
            for feature_type, count in feature_counts.items():
                importance[feature_type] = count / total_states
            
            return importance
        
        # PyTorch 모델은 좀 더 복잡한 방법 필요 (현재 간소화된 구현)
        return {}
