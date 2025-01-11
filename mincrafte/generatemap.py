import os
import json
from collections import defaultdict
import sys

cx, cy = map(int, input().split())
# 데이터 저장용 딕셔너리
data = defaultdict(lambda: defaultdict(dict))
# 데이터 작성
for x in range(-8, 9):
    for z in range(-8, 9):
        key = f"{16*cx+x},{16*cy+z}"
        data[key][0] = {
            "block": "grass_block"
        }
for x in range(-8, 9):
    for y in range(-1, -3, -1):
        for z in range(-8, 9):
            key = f"{16*cx+x},{16*cy+z}"
            data[key][y] = {
                "block": "dirt"
            }
# 파일 경로 설정
dir_path = f".\\assets\\map\\c{cx}-{cy}.json"
os.makedirs(os.path.dirname(dir_path), exist_ok=True)
# JSON 파일 쓰기
with open(dir_path, 'w') as f:
    json.dump(data, f, indent=4)
print('finished')