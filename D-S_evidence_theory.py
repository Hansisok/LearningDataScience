import numpy as np

'''
# 框架列表下标与原集合的对应方式：
1. 集合的取舍可以用二进制表示，取为1，舍为0.
2. 每个元素对应二进制表示中的一位，从左到右依次对应二进制的从低到高位.
3. 例如：集合{A, B, C}，取A舍B取C，则对应二进制为101，即十进制的5.

'''
class DStheory:
    # Constructor
    def __init__(self, Phi: list):
        self.Phi = Phi
        self.A_list = []
        self.m = list(np.ones( 2**len(Phi) ) / (2**len(Phi)) )

    # Basic probability assignment
    def bpa(self, A: list, m: int):
        self.m[self.get_Aindex(A)] = m  
        pass

    # Belief
    def belief(self, Phi: list, X: list):
        pass

    def get_Aindex(self, A: list):
        index = 0
        for i in range(len(A)):
            index += 2**self.Phi.index(A[i]) # 找到集合A在框架列表中的对应索引
        return index


# Main
if __name__ == "__main__":
    Phi = ["事件A", "事件B", "事件C"]
    ds = DStheory(Phi)
    print(ds.m)
    ds.bpa(["事件A"], 0.2)
    print(ds.m)

    
    

