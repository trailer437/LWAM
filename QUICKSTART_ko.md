# LWAM v0.1 실행 가이드

초기 연구용 구현입니다. 단일 RGB 카메라 PushT, 목표 이미지 조건, Frozen DINOv2 CLS, 소형 MLP 3종으로 구성됩니다. 실제 로봇 구동은 포함하지 않습니다.

## 설치 (Ubuntu, Python 3.12 권장)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# https://pytorch.org/get-started/locally/ 에서 드라이버에 맞는 CUDA PyTorch 설치
pip install -r requirements.txt
pip install -e .
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
pytest -q
```

## 실제 데이터 파이프라인

프로젝트 루트에서 실행합니다. 처음 실행하면 인터넷이 필요합니다. DINOv2 Torch Hub 코드 및 모델과 Hugging Face 데이터셋을 다운로드합니다. LeRobot 0.4 계열/v3 데이터셋을 대상으로 작성했지만 외부 다운로드 및 GPU 경로는 작성 환경에서 테스트하지 못했습니다. Python 및 패키지 설치 가능 여부는 사용하는 OS에 따라 달라집니다.

```bash
python scripts/download_dataset.py
python scripts/build_latent_cache.py --max-episodes 20
python -m lwam.training.train_world_model --epochs 2
python -m lwam.training.train_action_model --epochs 2
python -m lwam.training.train_joint --mode refiner --epochs 2
python -m lwam.training.train_joint --mode joint --resume-refiner --epochs 2
python scripts/evaluate.py
```

20개 Episode와 2 epochs는 동작 점검용입니다. 전체 데이터로 확장하려면 새로운 cache_dir 및 checkpoint_dir를 YAML에서 설정하고 `--max-episodes`를 생략합니다. 기존 캐시와 섞지 않습니다. 각 단계의 `--epochs`를 생략하면 YAML의 30 epochs를 사용합니다.

## 인터넷 없는 합성 데이터 테스트

PyTorch, NumPy, PyYAML, pytest만 미리 설치되어 있으면 가능합니다. **별도의 YAML을 복사하여** data.cache_dir을 `data/synthetic_cache`, training.checkpoint_dir을 `checkpoints/synthetic`로 설정하세요. 아래는 `configs/synthetic.yaml`로 저장했다고 가정합니다. 합성 데이터는 로봇 학습 결과가 아닙니다.

```bash
export PYTHONPATH="$PWD/src"
python scripts/build_latent_cache.py --config configs/synthetic.yaml --synthetic
python -m lwam.training.train_world_model --config configs/synthetic.yaml --epochs 1
python -m lwam.training.train_action_model --config configs/synthetic.yaml --epochs 1
python -m lwam.training.train_joint --config configs/synthetic.yaml --mode refiner --epochs 1
python -m lwam.training.train_joint --config configs/synthetic.yaml --mode joint --resume-refiner --epochs 1
python scripts/evaluate.py --config configs/synthetic.yaml --checkpoint checkpoints/synthetic/joint.pt
```

## 오프라인 추론

실제 PushT 캐시로 학습한 checkpoint와 동일한 DINOv2 전처리를 사용합니다. state는 원본 데이터의 단위로 전달합니다. 출력은 정규화 해제된 행동이며 실제 실행 전에 환경의 유효 범위 검사가 필요합니다.

```bash
python scripts/infer.py --image current.png --goal goal.png --state 100 200
```

## 설계 및 제약

- 최종 관측을 goal로 사용하는 hindsight goal 방식입니다. 종료 관측이 실제 성공 상태인지 별도 확인해야 합니다. 기본 PushT 시연을 전문가 데이터로 가정하며 성공/실패 혼합 데이터에는 imitation mask가 추가로 필요합니다.
- Episode 마지막 행의 행동은 다음 관측이 없어 제외합니다. Episode 경계를 넘는 학습 샘플은 만들지 않습니다.
- Train/validation/test는 episode 단위 분리이며 state/action 정규화는 train 통계만 사용합니다.
- World Model은 visual latent와 robot state를 함께 예측합니다. 예측된 state를 다음 rollout에 전달합니다.
- World Loss에는 기록된 행동만 사용합니다. Refiner에는 initial head가 제안한 행동의 미래 예측을 입력합니다.
- Joint training은 별도 연구 단계이며 action gradient가 dynamics를 변화시킬 수 있습니다. 독립적인 dynamics 오차와 frozen refiner 비교를 유지하세요.
- `evaluate.py`는 held-out MSE를 측정합니다. 실제 PushT closed-loop 성공률, random-action baseline, action-refiner ablation, 실물 로봇 제어는 후속 구현입니다.
- CLS token은 공간 정보를 잃을 수 있습니다. Patch token, 2-camera 입력, LoRA/QLoRA, TensorRT는 미구현입니다.
- 체크포인트는 검증 손실 최저 모델 가중치와 정규화 통계를 저장합니다. optimizer 재개 상태는 저장하지 않습니다. 변경된 데이터에 기존 체크포인트를 재사용하지 마세요.

## 검증 결과

작성 환경: PyTorch 2.10 CPU, CUDA 없음. 단위 테스트 3개 통과. 12개 합성 Episode, hidden_dim=32에서 4개 학습 단계 및 held-out 평가 실행 완료. 실제 DINOv2 대신 단위 테스트에서는 모의 backbone을 사용했습니다. 외부 패키지 설치는 네트워크 제한으로 검증하지 못했습니다.
