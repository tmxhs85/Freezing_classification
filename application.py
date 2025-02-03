import pickle
from flask import Flask, render_template, request, jsonify
import requests
from collections import defaultdict
from datetime import datetime
import os
from azure.storage.blob import BlobServiceClient 
import pandas as pd
import base64


app = Flask(__name__)

# Load the prediction model
# with open("C:\\py\\freeze_azure_ml\\freezing_model", "rb") as model_file:
#     model = pickle.load(model_file)
    
# 기상청 API 설정
SERVICE_KEY = "u/tFOWu9xDYgBc2n6zUlZ+6PpZ3tLIUrjTcPxNnHPWQE8y4w2XzU3fHUre1ZEyB9hzPSDgN+KIEqIHB4U16Y6w=="  # 발급받은 서비스 키 입력
BASE_URL = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
DATA_TYPE = "JSON"  # 요청 데이터 형식

# 지역별 nx, ny 좌표 설정
LOCATION_COORDS = {
    "경기도": {
        "화성시": {"nx": 57, "ny": 119, "위도":37.196816, "경도":126.833530},
        "수원시": {"nx": 60, "ny": 121, "위도":37.301011, "경도":127.012222},
        "성남시": {"nx": 63, "ny": 124, "위도":37.447491, "경도":127.147719},
        "의정부시": {"nx": 61, "ny": 130, "위도":37.735288, "경도":127.035841},
        "안양시": {"nx": 59, "ny": 123, "위도":37.383777, "경도":126.934500},
        "부천시": {"nx": 57, "ny": 125, "위도":37.496592, "경도":126.786997},
        "광명시": {"nx": 58, "ny": 125, "위도":37.475750, "경도":126.866708},
        "평택시": {"nx": 62, "ny": 114, "위도":36.989438, "경도":127.114655},
        "동두천시": {"nx": 61, "ny": 134, "위도":37.900916, "경도":127.062652},
        "안산시": {"nx": 58, "ny": 121, "위도":37.298519, "경도":126.846819},
        "고양시": {"nx": 57, "ny": 128, "위도":37.634583, "경도":126.834197},
        "과천시": {"nx": 60, "ny": 124, "위도":37.4263722, "경도":126.9898},
        "구리시": {"nx": 62, "ny": 127, "위도":37.591625, "경도":127.131863},
        "남양주시": {"nx": 64, "ny": 128, "위도":37.633177, "경도":127.218633},
        "오산시": {"nx": 62, "ny": 118, "위도":37.146913, "경도":127.079641},
        "시흥시": {"nx": 57, "ny": 123, "위도":37.377319, "경도":126.805077},
        "군포시": {"nx": 59, "ny": 122, "위도":37.358658, "경도":126.9375},
        "의왕시": {"nx": 60, "ny": 122, "위도":37.34195, "경도":126.970388},
        "하남시": {"nx": 64, "ny": 126, "위도":37.536497, "경도":127.217},
        "용인시": {"nx": 64, "ny": 119, "위도":37.231477, "경도":127.203844},
        "파주시": {"nx": 56, "ny": 131, "위도":37.757083, "경도":126.781952},
        "이천시": {"nx": 68, "ny": 121, "위도":37.275436, "경도":127.443219},
        "안성시": {"nx": 65, "ny": 115, "위도":37.005175, "경도":127.28184},
        "김포시": {"nx": 55, "ny": 128, "위도":37.612458, "경도":126.717777},
        "광주시": {"nx": 65, "ny": 123, "위도":37.414505, "경도":127.257786},
        "양주시": {"nx": 61, "ny": 131, "위도":37.78245, "경도":127.04781},
        "포천시": {"nx": 64, "ny": 134, "위도":37.892155, "경도":127.20241},
        "여주시": {"nx": 71, "ny": 121, "위도":37.295358, "경도":127.639622},
        "연천군": {"nx": 61, "ny": 138, "위도":38.093363, "경도":127.077066},
        "가평군": {"nx": 69, "ny": 133, "위도":37.828830, "경도":127.511777},
        "양평군": {"nx": 69, "ny": 125, "위도":37.488936, "경도":127.489886},
    },
    "서울특별시": {
        "--": {"nx": 60, "ny": 127, "위도":126.980008, "경도":37.563569},
    },
    "인천광역시": {
        "--": {"nx": 55, "ny": 124, "위도":126.707352, "경도":	37.453233},
    },
}


# Azure Blob Storage 설정
AZURE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=sonw006428547489;AccountKey=rUMn7ZlgjmoXH5x9t7FymbUN5tBB/ySp0tC2zN90xbfmy7ICz8Pdr6PDVv/pSic/10ESyhoQW5Lp+AStY7VvoA==;EndpointSuffix=core.windows.net"  # Azure Storage의 연결 문자열
BLOB_CONTAINER_NAME = "freezing-ml"         # 컨테이너 이름
MODEL_BLOB_NAME = "LGB_Model"                  # 모델 파일 이름
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # FlaskApp 디렉토리 경로

LOCAL_MODEL_PATH = os.path.join(BASE_DIR, MODEL_BLOB_NAME)  # 모델 경로
#LOCAL_MODEL_PATH = "C:\\py\\freezing_model.pkl"        # 로컬에 저장될 모델 경로

SCALER_BLOB_NAME = "scaler.pkl"
LOCAL_SCALER_PATH = os.path.join(BASE_DIR, SCALER_BLOB_NAME)

# Blob Storage에서 모델 다운로드
def download_model_from_blob():
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    blob_client = blob_service_client.get_blob_client(container=BLOB_CONTAINER_NAME, blob=MODEL_BLOB_NAME)

    # Blob 다운로드 및 로컬에 저장
    with open(LOCAL_MODEL_PATH, "wb") as model_file:
        model_file.write(blob_client.download_blob().readall())

def download_scaler_from_blob():
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    blob_client = blob_service_client.get_blob_client(container=BLOB_CONTAINER_NAME, blob=SCALER_BLOB_NAME)

    # Blob 다운로드 및 로컬에 저장
    with open(LOCAL_SCALER_PATH, "wb") as scaler_file:
        scaler_file.write(blob_client.download_blob().readall())
        
# Flask 앱 초기화 시 모델 로드
with app.app_context():
    if not os.path.exists(LOCAL_MODEL_PATH):
        download_model_from_blob()
    if not os.path.exists(LOCAL_SCALER_PATH):
        download_scaler_from_blob()
        
    global model
    global scaler
    
    with open(LOCAL_MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
    print("모델이 성공적으로 로드되었습니다!")
    print(LOCAL_MODEL_PATH)
    
    with open(LOCAL_SCALER_PATH, "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    print("Scaler가 성공적으로 로드되었습니다!")
    print(LOCAL_SCALER_PATH)
        
    
@app.route('/')
def main():
    return render_template('main.html', locations=LOCATION_COORDS)

@app.route('/index')
def index(): 
    print("dff")
    return render_template('index.html', locations=LOCATION_COORDS)

@app.route("/predict_freezing", methods=['POST'])
def predict_freezing():
    req_data = request.json
    
    region = req_data.get("region")
    city = req_data.get("city")
    date = req_data.get("day")  # Format: YYYYMMDD
    #print(req_data)
    
    # 요청 데이터 검증
    if not region or not city or not date:
        return jsonify({"error": "Missing region, city, or day"}), 400

    if region not in LOCATION_COORDS or city not in LOCATION_COORDS[region]:
        return jsonify({"error": "Invalid region or city"}), 400
    
    try:
        # `day` 값 처리
        year = int(date[:4])
        month = int(date[4:6])
        day = int(date[6:8])
        hour = 5  # 기본 예보 시간 05:00
    except ValueError:
        return jsonify({"error": "Invalid date format"}), 400
    
    # 선택한 지역의 nx, ny 값 가져오기
    coords = LOCATION_COORDS[region][city]
    nx, ny, latitude, longitude = coords["nx"], coords["ny"], coords["위도"], coords["경도"]
    
    # 오늘 날짜 가져오기
    today = datetime.now()
    # YYYYMMDD 형식으로 변환
    today_str = today.strftime("%Y%m%d")
    
    # 요청 파라미터 설정
    params = {
        "serviceKey": SERVICE_KEY,
        "numOfRows": 1000,
        "pageNo": 1,
        "dataType": DATA_TYPE,
        "base_date": today_str, #f"{year:04}{month:02}{day:02}",  # YYYYMMDD
        "base_time": f"{hour:02}00", #"0500",
        "nx": nx,
        "ny": ny,
    }
    #print(f"{hour:02}00",nx,ny,)

    # API 호출
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code == 200:
        weather_data = response.json()
        #print(weather_data)
        
        
    # Extract relevant data from API response
    try:
        items = weather_data['response']['body']['items']['item']
        #print(items)
        filtered_items = [item for item in items if item['fcstDate'] == date]
        #print(filtered_items)
        # 시간별로 데이터 그룹화
        time_grouped_data = {}
        for item in filtered_items:
            fcst_time = item['fcstTime']
            
            if fcst_time not in time_grouped_data:
                time_grouped_data[fcst_time] = {
                    'TMP': None,
                    'REH': None,
                    'WSD': None
                }
            
            if item['category'] in ['TMP', 'REH', 'WSD']:
                time_grouped_data[fcst_time][item['category']] = float(item['fcstValue'])
                
        #temperature = float(next(item['fcstValue'] for item in items if item['category'] == 'TMP'))
        #humidity = float(next(item['fcstValue'] for item in items if item['category'] == 'REH'))
        #wind_speed = float(next(item['fcstValue'] for item in items if item['category'] == 'WSD'))
        
    except Exception as e:
        return jsonify({"error": "Failed to parse weather data", "details": str(e)}), 500

    #print(time_grouped_data)
    # 각 시간대별로 결빙 예측
    results = {}
    
    # Prepare input for the model
    for fcst_time, data in time_grouped_data.items():
        if all(v is not None for v in data.values()):
            input_features = [latitude, longitude, year, month, day, int(fcst_time[:2]),
                              data['WSD'],
                              data['TMP'],
                              data['REH']
                              ]
        df = pd.DataFrame(columns=['GRID_X','GRID_Y','Year','Month','Day','Hour','WS','TA_C','HM'])
        df.loc[0] = input_features
        x_predict_scaled = scaler.transform(df)
        prediction = model.predict(x_predict_scaled)
        freezing_status = int(prediction[0])
        
        # 결과 저장
        results[fcst_time] = {
            "fcst_time": f"{date} {fcst_time}",
            "temperature": data['TMP'],
            "humidity": data['REH'],
            "wind_speed": data['WSD'],
            "freezing_status": {
                0: "freezing",
                1: "Freezing possible",
                2: "No Freezing"
            }[freezing_status]
        }
    #print(results)
    return jsonify(results)

# Blob 데이터 가져오기 함수
def get_blob_data(file_name):
    # Blob 서비스 클라이언트 생성
    blob_service_client = azure.storage.blob.BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    # 컨테이너 클라이언트 생성
    container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)
    # Blob 클라이언트 생성
    blob_client = container_client.get_blob_client(file_name)
    
    # Blob 데이터 다운로드
    blob_data = blob_client.download_blob().readall()
    return blob_data

@app.route('/load-model-data', methods=['GET'])
def load_model_data():
    # 요청에서 모델 이름(파일명) 받기
    model_file_name = request.args.get('model')  # 쿼리 파라미터로 받은 모델 이름 (파일명)
    
    try:
        # Azure Blob에서 해당 모델 파일 가져오기
        blob_data = get_blob_data(model_file_name)
        
        # 가져온 데이터를 Base64로 인코딩하여 응답
        encoded_data = base64.b64encode(blob_data).decode('utf-8')
        
        return jsonify(result='success', data=encoded_data)
    except Exception as e:
        return jsonify(result='error', message=str(e))
        
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
