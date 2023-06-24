# Qualifying Round
국민대학교 자율주행경진대회 예선전

## 과제
- (1) 주행 시뮬레이터 환경에서 차선을 벗어나지 않고 목적지까지 주행하는 자율주행SW를 구현합니다.
- (2) 주차 시뮬레이터 환경에서 AR태그를 이용하여 정확하게 주차하는 자율주차SW를 구현합니다.
- (3) 다양한 미션을 수행하는 자율주행SW를 제작하는데 필요한 SW설계서를 작성합니다.

## 평가
- (1)번 과제: 차선을 벗어나지 않으며 목적지에 최대한 가까이 도달하는지와 주행 품질로 평가
- (2)번 과제: 4개 지정위치에서 출발하여 주차구역에 정확하게 주차하는지 여부로 평가
- (3)번 과제: SW 설계서의 완성도, 구체성, 타당성, 창의성 검토하여 평가

## 결과물
**parking**

![parking](https://github.com/2022-autonomous-pegasus/qualifying-round/assets/87895999/39330650-8e75-42b4-be00-83a3d73f4aa7)


**driving.py**

![driving](https://github.com/2022-autonomous-pegasus/qualifying-round/assets/87895999/2413f67a-1998-41e3-bf1e-d787e1641536)

## 실행 방법
xycar 제공 시뮬레이터 실행 방법입니다.
1. driving simulator
```bash
# 첫 번째 창
$ cm
$ roslaunch rosbridge_server rosbridge_websocket.launch

# 두 번째 창
$ cd ~/catkin_ws/src/xycar_sim_driving
$ ./xycar3Dsimulator.x86_64

#세 번째 창
$ cm
$ roslaunch assignment1 driving.launch
```
( 혹시 차가 움직이지 않는다면 cm을 입력한 후 다시 해보세요. )

2. parking simulator
```bash
$ cm
$ roslaunch assignment2 parking.launch
```


3. bashrc 파일 수정
```bash
# ================== DON'T TOUCH!!!! ==================
echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
alias cm='cd ~/catkin_ws && catkin_make'
alias eb='nano ~/.bashrc'
alias sb='source ~/.bashrc'
alias gs='git status'
alias gp='git pull'
alias cw='cd ~/catkin_ws'
alias cs='cd ~/catkin_ws/src'
alias cm='cd ~/catkin_ws && catkin_make && source /opt/ros/melodic/setup.bash && source ~/catkin_ws/devel/setup.bash'
alias dis1920='xrandr --newmode "1920x1080" 172.80 1920 2040 2248 2576 1080 1081 1084 1118 -HSync +Vsync && xrandr --addmode DP-1 1920x1080 && xrandr -s 1920x1080'
source /opt/ros/melodic/setup.bash
export ROS_MASTER_URI=http://localhost:11311
export ROS_HOSTNAME=localhost
# =====================================================
```

