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
        self.mList = dict()

    # Basic probability assignment
    def addBPA(self, event_m: dict, name: str):
        self.mList[name] = event_m
        print("Add BPA: ", name)

    def removeBPA(self, name: str):
        self.mList.pop(name)
        print("Remove BPA: ", name)

    def combineBPA(self, names: list):
        for name in names:
            
            pass
        pass

    # Belief
    def belief(self, name: str):
        
        pass


# Main
if __name__ == "__main__":
    Events = ["事件A", "事件B", "事件C"]
    ds = DStheory(Events)

    
    

