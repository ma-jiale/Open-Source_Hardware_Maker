import house
import mcpi.minecraft as minecraft

mc = minecraft.Minecraft.create()

pos = mc.player.getTilePos()
street = []
for i in range(5):
    for j in range(5):
        street.append(house.House(pos.x + 15 * i, pos.y, pos.z + 15 * j))

for i in range(25):
    street[i].buildAll()

while True:
    pos = mc.player.getTilePos()
    for i in range(25):
        street[i].isInside(pos)


