"""
강화학습 설정 모듈

이 모듈은 강화학습 알고리즘의 설정 파라미터를 정의합니다.
"""

from dataclasses import dataclass


@dataclass
class RLConfig:
    """강화학습 설정"""
    learning_rate: float = 0.1  # 학습률
    discount_factor: float = 0.95  # 할인 계수
    exploration_rate: float = 0.2  # 탐색률
    min_exploration_rate: float = 0.01  # 최소 탐색률
    exploration_decay: float = 0.995  # 탐색률 감소 계수
    batch_size: int = 32  # 배치 크기
    memory_size: int = 10000  # 메모리 크기
    update_frequency: int = 5  # 모델 업데이트 주기(에피소드 단위)
    episode_length: int = 24  # 하나의 에피소드 길이(시간 단위)
    model_path: str = "models/rl_model.pkl"  # 모델 저장 경로
    feature_importance_threshold: float = 0.05  # 특성 중요도 임계값
