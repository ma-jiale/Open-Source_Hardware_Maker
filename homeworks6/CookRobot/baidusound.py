from aip import AipSpeech
import os

APP_ID = '58314125'
API_KEY = "KHwzMGtTte6P1ZS53aBi9Kto"
SECRET_KEY = "gjVkKDnXByh4pAuow9XXBDqrl7ciUFqO"

client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

voice = client.synthesis("上海交通大学学生创新中心", 'zh', 6, {'vol': 15, 'per': 3, 'spd': 5})
with open("playback.mp3", 'wb') as fp:
    fp.write(voice)
os.system("playback.mp3")
