# GT 없이 달리는 장애물 회피 MPC (LiDAR 위치 추정 + CorridorMPC)

ForzaETH F1TENTH 시뮬레이터(ROS 2 Humble)에서 **LiDAR 스캔과 지도만으로** 자기 위치를 추정하면서,
랩마다 무작위로 놓이는 정지 장애물을 피해 달리는 주행기다.
시뮬레이터의 정답 위치(`/car_state/pose_GT`, `/car_state/odom_GT`)는 시작할 때 한 번 받는 초기 자세 말고는 쓰지 않는다.

```
/scan ──► LidarLocDriver (위치 추정) ──x, y, yaw, v, r──► CorridorMPC (장애물 인식 → DP → 통로 → LTV-MPC → 속도) ──► /drive
             ▲                                                      │
             └──────────── 실제로 보낸 명령 (조향각, 속도) ◄────────┘
```

![RViz: berlin, 장애물 4개/랩](docs/rviz_berlin.png)

빨간 선은 맵 레이싱라인, 초록 선은 주행기의 레이싱라인, 색 있는 선은 실제로 달린 경로다(파랑 2 m/s → 빨강 10 m/s).
주황색 박스는 장애물이고, 글자는 LiDAR 위치 오차(정답과 비교, 표시용)다.

## 결과 (berlin 맵, 기본 시뮬레이터 파라미터)

오프라인(f1tenth_gym + race_manager 장애물 배치, 3랩/에피소드). 주행기마다 장애물 있는 랩 900개다.
"GT"는 같은 MPC에 시뮬레이터 정답 상태를 준 것이다.

| 조건 | GT: 충돌 / 멈춤 / 평균 랩 | LiDAR(GT 없음): 충돌 / 멈춤 / 평균 랩 | LiDAR 위치 오차 최대 |
|---|---|---|---|
| 장애물 없음 | 0 / 0 / 7.25 s | 0 / 0 / 7.26 s | 8.0 mm |
| 장애물 4개, 시드 0-99 | 1 / 2 / 8.53 s | 1 / 1 / 8.45 s | 12.0 mm |
| 장애물 5개, 시드 0-99 | 0 / 0 / 8.66 s | 0 / 1 / 8.61 s | 11.0 mm |
| 장애물 4개, 처음 보는 시드 900-999 | 2 / 3 / 8.56 s | 1 / 3 / 8.70 s | 12.0 mm |
| ROS 타이밍 흉내(실시간 0.9배, 스텝 흔들림, 지연) 전체 | 충돌 9 | 충돌 4 | ≤ 12.3 mm |

ROS 실시뮬(장애물 4개, 시드 7):

| 실행 | 랩 | 충돌 | 랩타임 |
|---|---|---|---|
| CPU 여유 있음 | 42 | 0 | 평균 8.74 s, 베스트 8.06 s |
| 처음 19랩은 다른 학습 작업과 CPU를 나눠 씀 | 23 | 0 | 나눠 쓸 때 9.5-11.7 s, 학습을 멈춘 뒤 7.87-8.52 s. 헤어핀 시케인에서 1번 멈춤(손으로 다시 놓음) |

- 주행기 한 번 계산 시간(`act()`)은 평균 약 4 ms다(Ryzen 7, 다른 작업 없음). 위치 추정 몫은 0.5-1 ms다.
- 채점기는 벽시계 시간으로 랩을 잰다. CPU가 부족하면 시뮬레이터가 실제 시간보다 느리게 돌아서 랩타임이 늘어난다.

## 1. 위치 추정: `drivers/lidar_loc.py` (`LidarLocDriver`)

다른 주행기를 감싸는 wrapper다. `act(scan, x, y, yaw, v, yaw_rate, t)` 형태의 주행기라면 무엇이든 안에 넣을 수 있다.
안쪽 주행기에는 시뮬레이터 정답 대신 자기가 추정한 상태를 넘긴다.

**쓰는 정보**
- LiDAR 스캔: 1080개 빔, 시야 4.7 rad, 차 기준점보다 0.27 m 앞에 달려 있다.
- 점유 지도: `stack_master/maps/<map>`의 이미지와 임계값. 시뮬레이터가 레이캐스트하는 그 지도다.
- 자기가 보낸 명령(조향각, 속도)과 차량 모델.
- 초기 자세 1개: `/initialpose`(운영자가 차를 놓은 자세), 맵 시작점(`init="start"`), 또는 전역 탐색(`init="global"`).

**한 번의 계산(`act()`)마다 하는 일**

1. **시계:** 실제 ROS 브리지는 물리 스텝을 흔들리는 벽시계 타이머로 돌린다. 그래서 두 스캔 사이에 10 ms 스텝이 1-3개 들어간다.
   스텝 수는 스캔이 직접 판단한다: 가능한 스텝 수마다 예측하고 정합해서 가장 잘 맞는 것을 고른다. 필터, 명령 이력, 지연 추정, 안쪽 주행기는 모두 이 시뮬레이터 시계로 돈다.
2. **예측:** EKF 상태는 `s = [x, y, yaw, v, r, beta]`다.
   - 보낸 명령을 시뮬레이터와 같은 모델로 10 ms씩 적분한다(RK4). 모델에는 2스텝 조향 버퍼, 조향 서보 3.2 rad/s, 속도 P제어기, `sim_params.yaml`의 single-track 모델이 들어간다.
   - 0.5 m/s 미만이나 후진에서는 시뮬레이터처럼 기구학 모델로 바꾼다.
3. **측정 갱신:** 스캔을 지도에 맞추는 정합을 iterated EKF로 한다(사전 분포 + robust 스캔 비용의 Gauss-Newton).
   - 빔 끝점을 지도의 부호 있는 거리장(SDF)에 bilinear로 맞춘다.
   - Huber 가중치를 쓰고, 첫 반복은 0.25 m, 이후는 약 5σ에서 자른다. 그래서 장애물이나 지도에 없는 물체는 무시된다.
   - 빔 오차는 지도 픽셀을 통해 서로 상관돼 있어서, 스캔 정보량을 줄여서(`info_scale`) 쓴다.
4. **지연 다중 모델:** 명령 지연 0-150 ms 가설마다 EKF를 하나씩 둔다. 스캔 근거(Laplace 근사)를 망각 계수와 함께 누적해서 가장 그럴듯한 가설을 고른다(히스테리시스).
   이 지연 추정값은 안쪽 MPC에도 넘긴다.
5. **품질과 재위치 추정:** 매 스캔마다 inlier 비율을 본다. 장애물처럼 뭉친 free-space 끝점은 빼고 센다.
   나쁜 스캔이 이어지면 예측 주변을 correlative search로 다시 찾고, 그래도 안 되면 지도 전체를 탐색한다(약 0.3 s).

`forbid_gt=True`이면 GT를 읽으려는 순간 예외가 난다. 그리고 `avoid_node.py`는 GT 대신 NaN을 넘긴다.
초기 자세 뒤로는 GT 인자에 NaN이나 난수를 넣어도 명령이 비트 단위로 같다는 것을 시험으로 확인했다.

## 2. 주행기: `drivers/cmpc_berlin_tt.py` + `drivers/cmpc_berlin_obs.py` (`CorridorMPC`)

`cmpc_berlin_tt.py`는 기본 MPC다(시간 기록용 설정). `cmpc_berlin_obs.py`는 그것을 상속해서 장애물 기동을 더한다.
추가 기능은 모두 kwarg로 켜고 끈다. 다 끄면 기본 MPC와 비트 단위로 같다.

### 2.1 상태 추정과 지연 보상
- 조향 서보와 속도 P제어기를 자기가 보낸 명령으로 시뮬레이션해서, 지금 실제 조향각과 가속도를 안다.
- CG 횡미끄럼각(측정 안 됨)은 측정 요레이트로 굴린 single-track 모델과 연속 위치의 진행각으로 추정한다.
- 상태를 새 조향 명령이 실제로 먹히는 시점까지 비선형 모델로 예측한다.
- **온라인 추정:**
  - 명령 지연: 0-150 ms 후보, 다중 모델.
  - 타이어 강성 배율 `c`: 강성 × 추가 지연 격자의 병렬 single-track 모델을 측정 요레이트와 비교해서 고른다.
  - 측정 잡음: 적응형 Kalman 이득.
  - 호출 주기.
  - 이렇게 추정한 값으로 모델 전체를 맞춘다. 마찰이나 타이어가 20% 약해져도 벽에 안 박게 하려는 것이다.

### 2.2 장애물 인식
- 스캔 끝점을 지도 좌표로 옮긴다. 지도 벽에서 `obs_edt`(9 cm)보다 먼 점은 장애물 셀이다(지도 거리변환 사용).
- 장애물 셀은 랩 단위로 기억한다. race_manager가 랩마다 배치를 바꾸기 때문이다.
- 빔이 지나간 셀은 비어 있는 것으로 지운다(carving).
- 셀을 묶어서 Frenet 좌표의 박스로 만든다. 안 보이는 뒷면은 최대 크기(`hidden_depth` 0.43 m)로 채운다.
- 가려진 옆면은 빔 순서로 구분해서, 보일 때까지 최대 폭으로 늘려둔다.

### 2.3 DP: 장애물마다 왼쪽/오른쪽 고르기
- Frenet 격자(0.5 m 간격, 횡방향 5 cm) 위에서 보이는 모든 장애물에 대해 한 번에 일관된 쪽을 고른다.
- 비용 항목:
  - 레이싱라인 이탈
  - 기울기와 곡률 변화
  - 차가 꺾을 수 있는 기하 곡률 한계(`dp_kgeo`)
  - 벽과 장애물 근접
- 한번 고른 쪽을 바꾸면 벌점을 준다(히스테리시스). 장애물에 가까울수록 거의 바꿀 수 없게 된다.
- 벽까지 틈이 좁은 쪽은 후보에서 뺀다(`min_gap_w`: 기본 6 cm, 장애물 모드 10 cm, 이미 고른 쪽은 3 cm 덜 엄격).
- 한쪽이 좁고 반대쪽이 확실히 넓으면 좁은 쪽을 닫는다(`g_narrow`, `g_diff`).
- 고른 쪽들로 차체 기준선의 정확한 통로 `lb(s) ≤ d ≤ ub(s)`가 정해진다. 장애물과 벽 경계는 충돌 박스 반폭과 여유만큼 이미 부풀려져 있다.

### 2.4 공간 LTV-MPC
- **모델:** 시뮬레이터와 같은 선형 타이어 single-track 모델을 Frenet 좌표로 쓴다.
  - 상태: `d`(횡오차), `mu`(라인 대비 헤딩), `beta`, `r`, `delta`
  - 입력: 조향 속도(|·| ≤ 3.2 rad/s)
- **이산화:** 시간이 아니라 거리 단위다. 0.2 m 간격 40 스테이션, 즉 8 m 앞까지 본다.
  계획 속도 프로파일을 따라 스테이션마다 정확히 이산화한다(행렬 지수). 계획 가속도에서 하중 이동도 반영한다.
- **제약:**
  - 충돌 박스의 세 점(CG, +0.22 m, +0.435 m)이 통로 안에 있어야 한다. soft 제약이고 벌점이 크다.
  - 횡가속도 |v r| ≤ `a_lat_hard`도 soft 제약이다.
- **솔버:** HPIPM(CasADi에 들어 있는 `libhpipm.so`를 ctypes로 호출)이다. QP 데이터는 numba 커널이 솔버 버퍼에 바로 쓴다.
  새 속도 프로파일이 가까운 미래의 가속도를 2 m/s² 넘게 바꾸면 한 번 더 푼다.

### 2.5 속도 프로파일
- MPC 경로 위에서 계산한다.
- 횡가속도 한계를 지킨다. 지연에 따라 적응하는데, 명령 지연이 60-70 ms로 커지면 14 → 12 m/s²로 줄인다.
- 계획 제동은 조향이 유지되는(요 안정) 감속도 안에서만 한다.
- **비상 envelope**는 두 가지다:
  - 막힌 통로 앞에서 설 수 있는 속도
  - 스캔이 아직 못 본 첫 경로점에 `v_vis_safe` 이하로 도달할 수 있는 속도
- 트랙 폭이 2.8 m보다 좁은 구간에서는, 시야 끝에 장애물이 나타나도 멈출 수 있게 미리 감속한다(`w_front`, `v_hz`).
- 두 장애물을 반대쪽으로 연달아 지나가는 S자(weave)는, 그 곡률을 낼 수 있는 속도로 제한한다(`a_weave`).

### 2.6 저속 기동 (`cmpc_berlin_obs`)
1. **느린 DP:** DP가 길을 못 찾으면 `v_slow`(1.5 m/s)로 꺾을 수 있는 곡률로 다시 계획한다. 그리고 좁은 곳 앞에서 그 속도까지 줄인다.
2. **확인된 기어가기:** 0.6 m/s로 기어간다. 조향 호를 따라 20 cm 앞까지 차체 점들이 장애물(측정 셀)에서 4 cm, 벽에서 3 cm 떨어져야 한다.
   3 m/s 미만에서는 계획 경로에도 같은 검사를 한다. 통과하지 못하면 정지한다.
3. **후진:** 정지한 뒤 1 s 동안 갈 길이 없으면 0.3 m/s로 후진한다.
   - 0.5 m/s 미만은 시뮬레이터의 기구학 영역이라 후진이 안정적이다.
   - 조향은 차체 여유가 가장 큰 쪽으로 한다.
   - 거리는 0.5, 0.75, 1.0, … m로 늘리고, 한 장소에서 최대 5번까지 한다.
   - 후진 중에도 벽 5 cm, 장애물 8 cm 검사를 한다.
4. **재시도:** 후진 뒤에는 그 장소를 지날 때까지 통로 여유를 줄여서(`tight_kw`) 첫 박스에 붙어 다시 시도한다.
5. **헤딩 가드:** 라인 대비 헤딩이 1.6 rad를 넘으면 선다. Frenet 선형화가 깨지기 때문이다. 그다음 후진으로 차를 편다.

### 2.7 레이싱라인과 장애물 모드
- `maps_ext/berlin/tt/mtr_def_0.2.json`은 최소 시간 레이싱라인이다(46.6 m, 맵 기본 라인은 48.8 m).
  주행기 안에서만 맵 라인을 대신한다. s = 0은 맵 라인의 결승선에 다시 맞춘다.
- 장애물 모드(`obs_kw`)는 첫 랩, 박스를 본 다음 랩, 그리고 그 랩의 첫 박스부터 켜진다. 비어 있는 트랙은 시간 기록용 파라미터로 달린다.

### 2.8 주요 설정 (`profiles/berlin_obs.json`)

| 파라미터 | 값 | 뜻 |
|---|---|---|
| `a_lat`, `a_lat_hard` | 14 m/s² | 횡가속도 한계(계획 / soft 제약) |
| `lat_alat` | [0.06, 0.07, 12] | 명령 지연이 60 → 70 ms일 때 a_lat 14 → 12 |
| `a_acc`, `jerk`, `v_max` | 7.5 m/s², 120 m/s³, 11 m/s | 가속, 저크, 최고속도 |
| `obs_kw.v_vis_safe` | 2.0 m/s | 못 본 경로점 도달 속도 |
| `w_front`, `v_hz` | 2.8 m, 0.5 m/s | 좁은 구간 시야 끝 정지 준비 |
| `v_slow` | 1.5 m/s | 느린 DP 속도 |
| `creep_chk`, `prox_v` | 0.2 m, 3 m/s | 기어가기 / 근접 검사 |
| `back_dist`, `back_grow`, `back_max` | 0.5 m, 0.5, 5 | 후진 거리, 증가율, 최대 횟수 |
| `mu_max` | 1.6 rad | 헤딩 가드 |

`profiles/berlin_tt.json`(FAST)과 `berlin_tt_safe.json`(a_lat 12)은 장애물 없는 시간 기록용이다. 둘 다 장애물 인식은 켜져 있다.

## 3. 설치와 세팅

### 3.1 필요한 것
- Ubuntu 22.04와 ROS 2 Humble
- ForzaETH race stack의 ROS 2 시뮬레이터를 `~/forza_ws/race_stack`에 빌드해둔 상태(`colcon build`)
  - 주행기는 지도를 `~/forza_ws/race_stack/stack_master/maps/<map>/`에서 읽는다(`<map>.yaml`, `.png`, `global_waypoints.json`). 이 경로는 코드에 박혀 있다(`drivers/*.py`의 `MAPS` / `MAPDIR`). 다른 곳에 있으면 심볼릭 링크를 만든다.
  - 무작위 장애물과 랩 채점은 `race_manager` 패키지가 한다. 이 패키지는 수업용 race stack에 들어 있고 공개돼 있지 않다. 없으면 `run_avoid.sh`가 장애물 없이 돌린다.
- 시뮬레이터 차량 파라미터는 기본값(`sim_params.yaml`)이다. MPC의 모델 상수(`drivers/cmpc_berlin_tt.py` 상단)가 이 값과 같아야 한다.

### 3.2 설치
```bash
git clone https://github.com/delboy11/forza-lidar-corridor-mpc.git
cd forza-lidar-corridor-mpc
pip install --user -r requirements.txt
chmod +x run_avoid.sh
```
첫 실행 때는 numba 커널을 컴파일하느라 1-2분 걸린다. 결과는 `drivers/__pycache__`에 캐시되고, 그다음부터는 몇 초면 뜬다.

### 3.3 실행
GT 없는 장애물 MPC(LiDAR 위치 추정 + `cmpc_berlin_obs`), 장애물 4개, 시드 7, berlin 맵:
```bash
./run_avoid.sh drivers.lidar_loc:LidarLocDriver \
  '{"init": "start", "forbid_gt": true, "inner": "drivers.cmpc_berlin_obs:CorridorMPC", "profile": "profiles/berlin_obs.json"}' \
  4 7 berlin
```
- 인자 순서: `<module:Class> '<json kwargs>' [장애물 수] [시드(-1이면 무작위)] [맵]`
- 순서대로 시뮬레이터, race_manager, 시작 자세(`/initialpose`), 주행 노드(`avoid_node.py`, 50 Hz)를 띄운다. 종료는 Ctrl-C다.
- 로그는 `~/race_logs/`에 남는다(`av_driver.log`: 계산 시간, 지연, 속도).
- 차가 서버리거나 부딪히면 RViz의 **2D Pose Estimate**로 차를 다시 놓는다. 그 자세가 위치 추정의 새 초기 자세가 된다.

다른 조합:
```bash
# 같은 MPC, 시뮬레이터 GT 상태 사용
./run_avoid.sh drivers.cmpc_berlin_obs:CorridorMPC "$(cat profiles/berlin_obs.json)" 4 7 berlin
# 장애물 없는 시간 기록(GT 없음, FAST 설정)
./run_avoid.sh drivers.lidar_loc:LidarLocDriver '{"init": "start", "forbid_gt": true}' 0 -1 berlin
```

### 3.4 RViz 표시
```bash
source /opt/ros/humble/setup.bash && source ~/forza_ws/race_stack/install/setup.bash
python3 tools/loc_viz.py &      # /lidar_loc/viz: 추정 자세 화살표, GT 대비 오차(x100 확대), 오차 글자 (표시 전용)
python3 tools/trail_viz.py &    # /viz/trail: 최근 14 s 주행 경로, 속도별 색
rviz2 -d rviz/berlin_view.rviz  # berlin 위에서 내려다보는 시점 + 위 두 표시
```
`loc_viz.py`는 평가용으로만 GT와 비교한다. 주행기는 GT를 받지 않는다.

### 3.5 다른 맵
- `map_name`으로 아무 맵이나 줄 수 있다(`run_avoid.sh`가 맵 인자를 자동으로 넣는다).
- 하지만 `berlin_obs.json`의 레이싱라인 파일과 파라미터는 berlin용이다.
- 다른 맵에서는 `raceline_file`을 빼서(맵 라인 사용) 시작하고, `a_lat`이나 여유 값을 맵에 맞춰 조정해야 한다.

## 4. 알려진 한계
- **헤어핀 시케인 멈춤:** 좁은 헤어핀(berlin s ≈ 23-25 m) 안에 박스 두 개가 시케인으로 놓이면 문제가 생긴다. 후진 5번을 다 써도 각이 안 나와서 서버릴 수 있다. 오프라인에서는 900랩에 약 5번, GT를 써도 같은 빈도다. 실시뮬에서는 운영자가 다시 놓아야 한다.
- **CPU와 랩타임:** 랩타임은 PC 성능과 부하에 따라 달라진다(채점이 벽시계 기준). 주행기 자체는 1코어로 충분하다.
- **검증 범위:** 시뮬레이터 기본 파라미터와 berlin 맵에 맞춰 검증했다. 명령 지연 50 ms와 자세 잡음은 시험했지만 그 조건에 맞춰 튜닝하지는 않았다. 실제 차에서는 시험하지 않았다.
- **코드 주석의 `work/...` 경로:** 개발 중 보고서를 가리킨다. 이 저장소에는 들어 있지 않다.

## 5. 파일

| 파일 | 내용 |
|---|---|
| `drivers/lidar_loc.py` | `LidarLocDriver`: LiDAR + 지도 + 자기 명령 위치 추정 wrapper |
| `drivers/cmpc_berlin_tt.py` | `CorridorMPC` 본체: 인식, DP, 통로, LTV-MPC(HPIPM), 속도 프로파일, 온라인 추정, numba 커널 |
| `drivers/cmpc_berlin_obs.py` | 장애물 기동 추가(느린 DP, 기어가기, 후진, 재시도, 헤딩 가드, weave 제한, 장애물 모드) |
| `profiles/*.json` | 주행기 kwargs 프로파일 |
| `maps_ext/berlin/tt/mtr_def_0.2.json` | berlin 최소 시간 레이싱라인 |
| `avoid_node.py` | ROS 2 노드: `/scan`(+ GT 모드면 `/car_state/*_GT`) → `/drive`, `/initialpose` 처리, 추정 자세 발행 |
| `run_avoid.sh` | 시뮬레이터 + race_manager + 노드 실행 스크립트 |
| `tools/loc_viz.py`, `tools/trail_viz.py`, `rviz/berlin_view.rviz` | RViz 표시 |
