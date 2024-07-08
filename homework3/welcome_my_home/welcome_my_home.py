from mcpi.minecraft import Minecraft
import time

mc = Minecraft.create()
pos = mc.player.getTilePos()
print("player pos is", pos)

# mc.setBlock(pos.x,pos.y,pos.z,1)

stayed_time = 0
while True:
    print("stay_time" + str(stayed_time))
    time.sleep(1)
    pos = mc.player.getTilePos()
    mc.postToChat("please go to home x=201 y=-5 z=335 for 15s to fly")
    mc.postToChat("x:" + str(pos.x) + "y:" + str(pos.y) + "z:" + str(pos.z))
    # 修改为方块交大中我的宿舍的地址
    if pos.x == 104 and pos.z == -550 and pos.y == 4:
        mc.postToChat("welcome home,count down" + str(15 - stayed_time))
        stayed_time = stayed_time + 1
        if stayed_time >= 15:
            mc.player.setTilePos(201, 5, 335)
            stayed_time = 0
    else:
        stayed_time = 0
