"""
적응형 제어 컨트롤러 사용 예제

이 예제는 개인 맞춤형 복합 자극 기반 이명 치료 시스템의 
적응형 제어 부분을 사용하는 방법을 보여줍니다.
"""

import os
import time
import numpy as np
import logging
import json
from src.adaptive import AdaptiveController

# 로깅 설정
logging.basicConfig(level=logging.INFO, 
                  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 예제 데이터 디렉토리
EXAMPLE_DATA_DIR = "examples/data/"
os.makedirs(EXAMPLE_DATA_DIR, exist_ok=True)

def load_example_data(filename):
    """예제 데이터 로드"""
    try:
        filepath = os.path.join(EXAMPLE_DATA_DIR, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    except Exception as e:
        print(f"Error loading example data: {str(e)}")
        return None

def save_example_data(data, filename):
    """예제 데이터 저장"""
    try:
        filepath = os.path.join(EXAMPLE_DATA_DIR, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Error saving example data: {str(e)}")
        return False

def generate_example_eeg_data(duration_sec=5, fs=256):
    """예제 EEG 데이터 생성 (가상 데이터)"""
    num_samples = duration_sec * fs
    
    # 다양한 주파수 성분 생성
    t = np.arange(num_samples) / fs
    
    # 기본 신호 (알파파 주성분, 8-12Hz)
    alpha = 50 * np.sin(2 * np.pi * 10 * t)
    
    # 델타파 성분 (1-4Hz)
    delta = 20 * np.sin(2 * np.pi * 2 * t)
    
    # 베타파 성분 (13-30Hz)
    beta = 10 * np.sin(2 * np.pi * 20 * t)
    
    # 약간의 노이즈 추가
    noise = 5 * np.random.randn(num_samples)
    
    # 최종 EEG 신호
    eeg = alpha + delta + beta + noise
    
    return eeg

def generate_example_hrv_data(duration_sec=300):
    """예제 HRV 데이터 생성 (가상 데이터)"""
    # 평균 심박수 분당 70회, 표준편차 5회로 가정
    mean_hr = 70
    std_hr = 5
    
    # 초당 심박수를 생성
    heart_rates = np.random.normal(mean_hr, std_hr, duration_sec)
    
    # 심박수를 RR 간격으로 변환 (ms)
    rr_intervals = 60000 / heart_rates
    
    return rr_intervals

def generate_example_tinnitus_data(severity=5.0):
    """예제 이명 데이터 생성"""
    return {
        'primary_frequency': 4000 + np.random.randint(-200, 200),  # 4kHz 주변
        'intensity': severity / 10.0,  # 0-1 스케일
        'bandwidth': 0.2 + np.random.random() * 0.3,  # 0.2-0.5 범위
        'duration': 60 + np.random.randint(0, 60),  # 60-120분
        'severity': severity,  # 0-10 스케일
        'annoyance': severity + np.random.randint(-1, 2)  # 약간의 변동
    }

def generate_example_sleep_data(stage='awake'):
    """예제 수면 데이터 생성"""
    # 수면 단계별 수면 품질 설정
    if stage == 'awake':
        efficiency = 0.0
        n3_percentage = 0.0
        rem_percentage = 0.0
        awakenings = 0
    elif stage == 'N1':
        efficiency = 0.5 + np.random.random() * 0.2
        n3_percentage = 0.0
        rem_percentage = 0.0
        awakenings = np.random.randint(1, 3)
    elif stage == 'N2':
        efficiency = 0.7 + np.random.random() * 0.2
        n3_percentage = 0.1 + np.random.random() * 0.1
        rem_percentage = 0.0
        awakenings = np.random.randint(0, 2)
    elif stage == 'N3':
        efficiency = 0.9 + np.random.random() * 0.1
        n3_percentage = 0.7 + np.random.random() * 0.3
        rem_percentage = 0.0
        awakenings = 0
    elif stage == 'REM':
        efficiency = 0.8 + np.random.random() * 0.2
        n3_percentage = 0.0
        rem_percentage = 0.8 + np.random.random() * 0.2
        awakenings = np.random.randint(0, 2)
    else:
        efficiency = 0.5
        n3_percentage = 0.0
        rem_percentage = 0.0
        awakenings = 2
    
    return {
        'stage': stage,
        'sleep_efficiency': efficiency,
        'total_sleep_time': 6.0 + np.random.random() * 2.0,  # 6-8 시간
        'awakenings': awakenings,
        'n3_percentage': n3_percentage,
        'rem_percentage': rem_percentage
    }

def simulate_patient_feedback(action, severity):
    """환자 피드백 시뮬레이션"""
    # 이명 강도 변화 (-1 ~ +0.5)
    intensity_change = -0.5 + np.random.random() * 1.5
    
    # 높은 볼륨은 편안함 감소
    if action.get('audio', {}).get('volume', 0) > 0.7:
        comfort = 0.2 + np.random.random() * 0.3  # 0.2-0.5 범위
    else:
        comfort = 0.5 + np.random.random() * 0.5  # 0.5-1.0 범위
    
    # 수면 개선
    sleep_improvement = 0.3 + np.random.random() * 0.7  # 0.3-1.0 범위
    
    # 예상보다 높은 이명 강도는 수면 개선 감소
    if severity > 7.0:
        sleep_improvement *= 0.5
    
    return {
        'tinnitus_intensity_change': intensity_change,
        'comfort': comfort,
        'sleep_improvement': sleep_improvement
    }

def main():
    """메인 실행 함수"""
    print("이명 치료 적응형 제어 시스템 예제 시작")
    
    # 적응형 컨트롤러 초기화
    controller = AdaptiveController()
    
    # 환자 ID 설정
    patient_id = "patient001"
    
    # 컨트롤러 시작
    controller.start(patient_id)
    
    try:
        # 시뮬레이션 반복
        print("\n시뮬레이션 시작...\n")
        
        # 다양한 상황 시뮬레이션
        scenarios = [
            {"stage": "awake", "severity": 6.0, "duration": 5},
            {"stage": "N1", "severity": 5.5, "duration": 3},
            {"stage": "N2", "severity": 5.0, "duration": 3},
            {"stage": "N3", "severity": 4.0, "duration": 3},
            {"stage": "REM", "severity": 6.5, "duration": 3},
            {"stage": "awake", "severity": 5.0, "duration": 3},
        ]
        
        for i, scenario in enumerate(scenarios):
            print(f"\n[시나리오 {i+1}] {scenario['stage']} 단계, 이명 강도: {scenario['severity']}")
            
            # 해당 시나리오 반복
            for j in range(scenario["duration"]):
                # 상태 데이터 생성
                eeg_data = generate_example_eeg_data()
                hrv_data = generate_example_hrv_data()
                tinnitus_data = generate_example_tinnitus_data(scenario["severity"])
                sleep_data = generate_example_sleep_data(scenario["stage"])
                
                # 상태 업데이트
                controller.update_state(
                    eeg_data=eeg_data,
                    hrv_data=hrv_data,
                    tinnitus_data=tinnitus_data,
                    sleep_data=sleep_data
                )
                
                # 액션 얻기
                action = controller.get_action(sleep_stage=scenario["stage"])
                
                # 액션 정보 출력
                print(f"\n  [액션 {j+1}]")
                print(f"    청각: 볼륨={action['audio']['volume']}, 주파수 조절={action['audio']['freq_mod']}, 진폭 조절={action['audio']['amp_mod']}")
                print(f"    시각: 밝기={action['visual']['brightness']}, 패턴={action['visual']['pattern']}, 색상={action['visual']['color']}")
                print(f"    촉각: 강도={action['tactile']['intensity']}, 주파수={action['tactile']['frequency']}")
                
                # 피드백 시뮬레이션
                feedback = simulate_patient_feedback(action, scenario["severity"])
                
                # 피드백 출력
                print(f"    피드백: 이명 변화={feedback['tinnitus_intensity_change']:.2f}, 편안함={feedback['comfort']:.2f}, 수면 개선={feedback['sleep_improvement']:.2f}")
                
                # 피드백 제공
                reward = controller.provide_feedback(feedback)
                print(f"    보상: {reward:.4f}")
                
                # 모델 학습
                if j % 2 == 0:  # 2회마다 학습
                    train_stats = controller.train()
                    print(f"    학습: 손실={train_stats['loss']:.4f}, 평균Q={train_stats['avg_q']:.4f}")
                
                # 잠시 대기
                time.sleep(0.5)
        
        # 통계 출력
        print("\n[최종 통계]")
        stats = controller.get_statistics()
        print(f"  보상 평균: {stats['rewards']['mean']:.4f}")
        print(f"  보상 최소/최대: {stats['rewards']['min']:.4f} / {stats['rewards']['max']:.4f}")
        print(f"  탐색률: {stats['exploration_rate']:.4f}")
        print(f"  에피소드 길이: {stats['episode_length']}")
        print(f"  에이전트 유형: {stats['agent_type']}")
        
    except KeyboardInterrupt:
        print("\n사용자에 의해 시뮬레이션이 중단되었습니다.")
    except Exception as e:
        print(f"\n오류 발생: {str(e)}")
    finally:
        # 컨트롤러 중지
        controller.stop()
        print("\n시뮬레이션 종료")

if __name__ == "__main__":
    main()
