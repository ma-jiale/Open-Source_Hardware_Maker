# 利用录音，语音识别，大模型，设计一个做菜教练，
# 能够根据大模型给出的菜谱，利用关键词匹配，利用opencv将材料和调料显示在一张虚拟桌子图片上。
# 比如西红柿1000克，一般一个西红柿重200克，就摆2个西红柿在桌上。
# 1.将录音转换成文字
# 2.将文字发送给大模型获得结果
# 3.对大模型返回结果进行关键字匹配显示相应食材和调料
from aip import AipSpeech
import pyaudio
import wave
import os
import requests
import json
import cv2

LLM_API_KEY = "xhPH2zJMNQLn20aTYkN30Vle"
LLM_SECRET_KEY = "5WlZE1Gw3LfawdW0lNDIImrv9N4VdFcd"
SOUND_APP_ID = '58315193'
SOUND_API_KEY = 'GZkpX29llXlCTtefPUwQMEaf'
SOUND_SECRET_KEY = 'WGTbzCJXfKpbDvtLTt4Ur36wpKUQD3aP'
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 8000
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "audio.wav"

client = AipSpeech(SOUND_APP_ID, SOUND_API_KEY, SOUND_SECRET_KEY)

egg = cv2.imread("ingredients/egg.png", cv2.IMREAD_COLOR)
tomato = cv2.imread("ingredients/tomato.png", cv2.IMREAD_COLOR)


def mxrecord():
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    stream.start_stream()
    print("* 开始录音......")

    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


def mxwav2char():
    with open('audio.wav', 'rb') as fp:
        wave = fp.read()

    print("* 正在识别......", len(wave))
    result = client.asr(wave, 'wav', 16000, {'dev_pid': 1536})
    # print(result)
    if result["err_no"] == 0:
        for t in result["result"]:
            print(t)
            return t
    else:
        print("没有识别到语音\n", result["err_no"])


def mxTTS(char):
    voice = client.synthesis(char, 'zh', 6, {'vol': 15, 'per': 3, 'spd': 5})
    with open("playback.mp3", 'wb') as fp:
        fp.write(voice)
    os.system("playback.mp3")


def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": LLM_API_KEY, "client_secret": LLM_SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))


def cooking_coach(char):
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant?access_token=" + get_access_token()
    s = "你是一个家庭做菜教练，请你根据用户给出的菜名，只列出这道菜需要的材料和调料，单位为克。不需要其他额外信息。用户：" + char
    # 注意message必须是奇数条
    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": s
            }
        ]
    })
    headers = {
        'Content-Type': 'application/json'
    }

    res = requests.request("POST", url, headers=headers, data=payload).json()
    print(res['result'])
    return res['result']


# 展示食材
def show_ingredients(res):
    ingredients = {'鸡蛋': 'ingredients/egg.png', '西红柿': 'ingredients/tomato.png', '番茄': 'ingredients/tomato.png',
                   '黄瓜': 'ingredients/cucumber.jpg', '胡萝卜': 'ingredients/carrot.jpg',
                   '土豆': 'ingredients/potato.jpg', '马铃薯': 'ingredients/potato.jpg', '葱': 'ingredients/onion.jpg',
                   '油': 'ingredients/oil.png',
                   '盐': 'ingredients/salt.png', '酱油': 'ingredients/soy sauce.png',
                   '老抽': 'ingredients/soy sauce.png', '生抽': 'ingredients/soy sauce.png',
                   '牛肉': 'ingredients/beef.jpg', '鸡肉': 'ingredients/chicken.jpg',
                   '鸡胸肉': 'ingredients/chicken.jpg', '猪肉': 'ingredients/pork.jpg'}
    table = cv2.imread("ingredients/table.jpg", cv2.IMREAD_COLOR)
    cv2.imshow('table', table)
    for key in ingredients:
        if key in res:
            img = cv2.imread(ingredients[key], cv2.IMREAD_COLOR)
            if img is not None:
                target_width = 200
                resized_img = scale_img(img, target_width)
                cv2.imshow(key, resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def scale_img(img, target_width):
    scale_ratio = target_width / img.shape[1]
    target_height = int(img.shape[0] * scale_ratio)
    resized_img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    return resized_img


def main():
    mxrecord()
    char = mxwav2char()
    result = cooking_coach(char)
    show_ingredients(result)


if __name__ == '__main__':
    main()
