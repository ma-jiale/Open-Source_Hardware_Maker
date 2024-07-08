import serial
import time

ser = serial.Serial("COM4")

time.sleep(3)
song1 = [1, 1, 5, 5, 6, 6, 5, 4, 4, 3, 3, 2, 2, 1]
song2 = [1, 2, 3, 1, 1, 2, 3, 1, 3, 4, 5]
song3 = [1, 1, 3, 5, 5]
record = {'tickle': 1, 'tiger': 2, 'bug': 3}
name = input("Please enter a name of a song: ")

if record[name] == 1:
    song = song1
elif record[name] == 2:
    song = song2
elif record[name] == 3:
    song = song3
else:
    print('no such song')
    exit(1)
for note in song:
    a = str(note)
    ser.write(a.encode())
    time.sleep(2)
