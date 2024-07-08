import mcpi.minecraft as minecraft
import mcpi.block as block

mc = minecraft.Minecraft.create()


def build(x, y, z):
    for i in range(3):
        for j in range(3):
            mc.setBlock(x + i, y, z + j, block.GOLD_BLOCK.id)


def house(x, y, z):
    SIZE = 5
    mc.setBlocks(x, y, z, x + SIZE, y + SIZE, z + SIZE, block.WOOD_PLANKS.id)
    mc.setBlocks(x + 1, y, z + 1, x + SIZE - 1, y + SIZE - 1, z + SIZE - 1, block.AIR.id)
    mc.setBlocks(x + 2, y, z, x + 3, y + 2, z, block.AIR.id)
    mc.setBlocks(x, y + 1, z + 2, x, y + 2, z + 3, block.GLASS.id)
    mc.setBlocks(x + 5, y + 1, z + 2, x + 5, y + 2, z + 3, block.GLASS.id)


def main():
    pos = mc.player.getTilePos()
    x = 201 - 1
    y = -6
    z = 335 - 1
    for i in range(3):
        for j in range(3):
            house(pos.x + 10 * i, pos.y, pos.z + 10 * j)


if __name__ == "__main__":
    main()
