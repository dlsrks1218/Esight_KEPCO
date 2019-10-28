# Esight_KEPCO
2019 Energy × Security 해커톤

# 사용 알고리즘 및 개념
* NILM - classification for each appliance
* Differential Privacy, DP - perturbation for statistical data distribution
* RNN - prediction for future power consumption of each appliance

# 사용 언어
* python3
* HTML, css, javascript

# 사용 라이브러리 및 프레임워크
* Flask - 웹 프레임워크
* Bokeh - 시각화
* jQuery, AJAX - HTML의 클라이언트 사이드 데이터 처리
* Tensorflow - 예측

# Issues    
* static 안의 파일을 수정할 경우 앱을 새로 구동해도 업데이트 되지 않는 경우가 발생. 인터넷 사용 기록 지우고 다시 앱 구동할 것 

# NILM + RNN + DP
* 총 전력 데이터를 NILM TK을 활용해 가전 기기별로 분류
* 분류된 전력 데이터를 기반으로 미래의 가전 기기별 소비 전력량을 예측
* 데이터 분석가에게 제공 시 차분 프라이버시 (Differential Privacy, DP)를 적용하여 변조된 통계 데이터를 보냄
