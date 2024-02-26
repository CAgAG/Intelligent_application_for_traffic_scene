class IDPool:
    def __init__(self, length: int):
        """
        :param length: 长度
         pool: 0: 未使用  1: 已使用
        """
        self.pool = {}
        self.pool[0] = [i for i in range(length)]
        self.pool[1] = []

    def getNewID(self):
        if not self.pool[0]:
            return None
        num = self.pool[0].pop(0)
        self.pool[1].append(num)
        return num

    def releaseID(self, id: int):
        if id in self.pool[1]:
            self.pool[1].remove(id)
        self.pool[0].append(id)


