import random
import mcpi.block as block
import mcpi.minecraft as minecraft

mc = minecraft.Minecraft.create()


class House:
    def __init__(self, x, y, z):
        names = ["JiaShuo", "Zangyujie", "Zhangsen", "Majiale", "Kevin", "Peter", "Tom", "Jerry"]
        self.name = random.choice(names)
        self.x = x
        self.y = y
        self.z = z
        self.size = random.randint(4, 10)
        self.color = random.randint(1, 15)

    def __buildWall__(self):
        mc.setBlocks(self.x, self.y, self.z, self.x + self.size, self.y + self.size, self.z + self.size, block.WOOL.id,
                     self.color)
        mc.setBlocks(self.x + 1, self.y + 1, self.z + 1, self.x + self.size - 1, self.y + self.size,
                     self.z + self.size - 1, block.AIR.id)

    def __buildRoof(self):
        mc.setBlocks(self.x, self.y + self.size, self.z, self.x + self.size, self.y + self.size, self.z + self.size,
                     block.WOOL.id,
                     self.color)

    def buildAll(self):
        self.__buildWall__()
        self.__buildRoof()

    def isInside(self, pos):
        if self.x < pos.x < self.x + self.size and self.y < pos.y < self.y + self.size and self.z < pos.z < self.z + self.size:
            mc.postToChat("Welcome %s's house!" % self.name)


class RoundHouse(House):
    def __init__(self, x, y, z, r):
        super(RoundHouse, self).__init__(self, x, y, z)
        self.r = r
        print("I will build a RoundHouse, r is", r)

    def __buildRoof(self):
        print("I will build a round roof, the Radius is", self.r)

    def buildAll(self):
        super(RoundHouse, self).__buildWall__()
        self.__buildRoof()
