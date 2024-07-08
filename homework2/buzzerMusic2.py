import serial
import time

ser = serial.Serial("COM4")
time.sleep(3)

try:
    file = open('record.csv', 'r', encoding='utf-8-sig')
except FileNotFoundError:
    # 文件不存在的错误处理
    print("File not found.")
    exit(1)

line1 = file.readline()
line2 = file.readline()
line3 = file.readline()
record1 = line1.split(',')
record2 = line2.split(',')
record3 = line3.split(',')

name = input("Please enter a name of a song: ")

if name == record1[0]:
    song = [int(item) for item in record1[1:]]
elif name == record2[0]:
    song = [int(item) for item in record2[1:]]
elif name == record3[0]:
    song = [int(item) for item in record3[1:]]
else:
    print('no such song')
    exit(1)
for note in song:
    a = str(note)
    ser.write(a.encode())
    time.sleep(2)
