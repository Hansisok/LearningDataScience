import numpy as np

'''
# 框架列表下标与原集合的对应方式：
1. 集合的取舍可以用二进制表示，取为1，舍为0.
2. 每个元素对应二进制表示中的一位，从左到右依次对应二进制的从低到高位.
3. 例如：集合{A, B, C}，取A舍B取C，则对应二进制为101，即十进制的5.

'''
class DStheory:
    # Constructor
    def __init__(self, Events: list):
        self.Events = Events
        self.lenFrame = 2**len(Events)
        self.BPAs = dict()

    # add a basic probability assignment named 'name' to the frame
    def addBPA(self, A: list, m: list, name: str):
        if len(A) != len(m):
            raise ValueError("The length of A and m must be the same.")
        if sum(m) != 1:
            raise ValueError("The sum of m must be 1.")
        if name in self.BPAs:
            raise ValueError("The name already exists.")
        
        mList = list(np.zeros( self.lenFrame ) / (self.lenFrame) )

        for i in range(len(A)):
            try:
                print(self.__getIndex_A(A[i]), A[i], m[i])
                mList[self.__getIndex_A(A[i])] = m[i]  
            except:
                raise ValueError("The event in A must be in the frame.")
        
        self.BPAs[name] = mList

    # remove a basic probability assignment named 'name' from the frame
    def removeBPA(self, name: str):
        self.mList.pop(name)

    # Belief
    def belief(self, nameLists: list, name: str):
        pass

    # Plausible
    def plausible(self, nameLists: list, name: str):
        pass

    # Combine
    def combine(self, names: list, name: str):
        combineBPA = list(np.zeros( self.lenFrame ) / (self.lenFrame) )
        for name in names:
            pass
        pass

    def __findIntersection (self, index: int, length: int):
        # 从self.lenFrame个集合中找到所有的长度为length的集合，该集合需满足它们的交集只有self.Events[index]指向的那个事件.
        # 最简单的遍历方法。遍历每一种组合，然后验证它们是否满足条件。
        group = []
        for i in range(length):
            hashtable = dict()
            turn = 0
            for j in range(self.lenFrame):
                if j & (1<<i):
                    hashtable[turn] = j
                    turn += 1
            for j in range(turn):
                for k in range(j+1, turn):
                    if hashtable[j] & hashtable[k] == 0:
                        group.append([hashtable[j], hashtable[k]])
        return group

    def __findUnion (self, index: int, length: int):
        # 从self.lenFrame个集合中找到所有的长度为length的集合，该集合需满足它们的并集包含self.Events[index]指向的那个事件.
        pass
    

    # Show the frame
    def showFrame(self):
        print("Frame: ", self.Events)
        print("BPAs: ", self.BPAs)

    def __getIndex_A(self, A: list):
        index = 0
        for i in range(len(A)):
            index += 2**self.Events.index(A[i]) # 找到集合A在框架列表中的对应索引
        return index



# Main
if __name__ == "__main__":
    Events = ["事件A", "事件B", "事件C"]
    ds = DStheory(Events)
    ds.addBPA(A=[["事件A", "事件B"], ["事件A", "事件C"]], m=[0.3, 0.7], name="BPA1")
    ds.addBPA(A=[["事件A"], ["事件B"], ["事件C"]], m=[0.1, 0.2, 0.7], name="BPA2")
    ds.showFrame()
    

