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
        ''' add a basic probability assignment named 'name' to the frame 
        '''
        if len(A) != len(m):
            raise ValueError("The length of A and m must be the same.")
        if sum(m) != 1:
            raise ValueError("The sum of m must be 1.")
        if name in self.BPAs:
            raise ValueError("The name already exists.")
        
        mList = list(np.zeros( self.lenFrame ) / (self.lenFrame) )

        for i in range(len(A)):
            try:
                # print(self.__getIndex_A(A[i]), A[i], m[i])
                mList[self.__getIndex_A(A[i])] = m[i]  
            except:
                raise ValueError("The event in A must be in the frame.")
        
        self.BPAs[name] = mList
        print("Add BPA: ", name, self.BPAs[name])

    # remove a basic probability assignment named 'name' from the frame
    def removeBPA(self, name: str):
        self.mList.pop(name)

    # Belief
    def belief(self, event_f: list, name: str):
        ''' 计算信任度, 取所有被event_f包含的集合的m之和
            event_f: 事件集合
            name: BPA的名称
            
        '''
        event = 0
        event_f = self.__getIndex_A(event_f)
        print("event_f: ", event_f)
        index_in = getBinaryOneIndex(event_f)
        events = []
        print("index_in: ", index_in)
        for i in range(2**index_in[0] if index_in != [] else 0, event_f+1):
            if i & event_f == event_f:
                events.append(i)
        return sum([self.BPAs[name][i] for i in events])


    # Plausible
    def plausible(self, event_f: list, name: str):
        ''' 计算怀疑度, 取所有与event_f的交集不为空的集合的m之和
        '''
        index_inter = self.__findIntersection(self.__getIndex_A(event_f), 1)
        print("index_inter: ", index_inter)
        return sum([self.BPAs[name][i[0]] for i in index_inter])

    # Combine
    def combineBPA(self, names: list, name: str) -> list:
        ''' 框架中每一个需要都有一个新的信任度m。
        '''
        combinedBPA = list(np.zeros( self.lenFrame ) / (self.lenFrame) )

        # 计算比例系数K
        K = 1 - self.__sumIntersection(0, names)

        # 计算所有组合的交集的乘积和
        for i in range(len(combinedBPA)): 
                combinedBPA[i] += self.__sumIntersection(i, names) / K
        self.BPAs[name] = combinedBPA
        print("Combined BPA: ", name, self.BPAs[name])
        return self.BPAs[name]
    
    def __sumIntersection (self, index: int, names: list):
        """ 
        参数:  
            index: int, 指向的框架中的事件集合的编号
            names: list, 需要找到的集合的长度
        return:
            sum: int, 所有满足条件的集合的乘积和
        """
        
        sum = 0

        # 计算所有组合的交集的乘积和
        inters = self.__findIntersection(index, len(names))
        # print("inters: ", inters)

        for j in range(len(inters)):
            mul = 1
            for k in range(len(names)):
                mul *= self.BPAs[names[k]][inters[j][k]]
                # print("inters[j][k]: ", inters[j][k], self.BPAs[names[k]][inters[j][k]])
                # print("j, mul, k: ", j, mul, k)
            sum += mul 
            # print("sum: ", sum)
        
        return sum

    def __findIntersection (self, index: int, length: int):
        """ 从self.lenFrame个集合中找到所有的长度为length的集合, 该集合需满足它们的交集只有self.Events[index]指向的那个事件.    
            最简单的遍历方法. 遍历每一种组合, 然后验证它们是否满足条件.  
            顺序: 1. 获取所有包含事件A的集合; 2. 按特定顺序遍历, 找到所有满足条件的组.  
            index: int, 指向的框架中的事件集合的编号  
            length: int, 需要找到的集合的长度
        """

        # 获取所有包含事件A的集合
        index_f = getBinaryOneIndex(index)
        
        index_all = list(range(len(self.Events)))
        index_else = [i for i in index_all if i not in index_f]

        # print("index:", index)
        # print("index_all: ", index_all)
        # print("index_else: ", index_else)
        # print("index_f: ", index_f, self.getFrameElement(getBinaryFromIndexList(index_f)))

        index_inter = distribute_unique(index_else, length)
        # print("index_inter: ", index_inter)

        index_inter_1 = []
        # 组合index_f和index_inter，得到真正的index
        for i in index_inter:
            index_inter_2 = []
            for j in i:
                # print("j: ", j, "j+index_f: ",j+index_f, getBinaryFromIndexList(j+index_f), self.getFrameElement(getBinaryFromIndexList(j+index_f)))
                index_inter_2.append(getBinaryFromIndexList(j+index_f))
            index_inter_1.append(index_inter_2)


        # print("index_inter: ", index_inter)
        # print("index: ", index, ", 指向的事件: ", self.getFrameElement(index))
        # print("index_inter: ", index_inter_1)

        return index_inter_1


    def __findUnion (self, index: int, length: int):
        # 从self.lenFrame个集合中找到所有的长度为length的集合，该集合需满足它们的并集包含self.Events[index]指向的那个事件.
        pass
        
    def getFrameElement(self, index: int):
        ''' 获取框架中下标为index的对应元素集合  
            index: int  
            return: list
        '''
        element = []
        for i in getBinaryOneIndex(index):
            element.append(self.Events[i])
        return element

    # Show the frame
    def showFrame(self):
        print("Frame: ", self.Events)
        print("BPAs: ", self.BPAs)
        print("Frame length: ", self.lenFrame)
        event = 0 # 在框架中的事件编号
        length = 2
        # print("Intersection: ", self.__findIntersection(event, length))
        print("sumIntersection: ", self.__sumIntersection(event, ["BPA1", "BPA2"]))

    def __getIndex_A(self, A: list):
        ''' 获取集合A在框架列表中的对应索引 '''
        index = 0
        for i in range(len(A)):
            index += 2**self.Events.index(A[i]) # 找到集合A在框架列表中的对应索引
        return index

# 递归实现组合
def comb_recursive(arr, m):
    """从 arr 中取 m 个元素的组合"""
    if m == 0:
        return [[]]
    if len(arr) < m:
        return []
    with_first = [[arr[0]] + rest for rest in comb_recursive(arr[1:], m-1)]
    without_first = comb_recursive(arr[1:], m)
    return with_first + without_first

def distribute_unique(arr: list, n: int):
    result = []

    def helper(current, depth, remaining):
        if depth == n:
            result.append(current)
            return
        # 每个人可以拿0个到len(remaining)个元素的任意组合
        for i in range(2 ** len(remaining)):
            subset = [remaining[j] for j in range(len(remaining)) if (i >> j) & 1]
            new_remaining = [x for x in remaining if x not in subset]
            helper(current + [subset], depth + 1, new_remaining)

    helper([], 0, arr)
    return result

def distribute_same(arr: list,  n: int):
    # 列表的顺序不重要
    result = []

    def helper(current, depth, remaining):
        pass
        

def getBinaryOneIndex(num: int):
    # 获取二进制数为1的位数
    # print(bin(num))
    count = 0
    index = []
    while num != 0:
        if num%2 == 1:
            index.append(count)
        num = num//2
        # print(num)
        count += 1        
    return index

def getBinaryFromIndexList(index: list):
    # 从index中获取二进制数
    result = 0
    for i in index:
        result += 2**i
    return result

# Main
if __name__ == "__main__":
    # 对象测试
    Events = ["事件A", "事件B", "事件C"]
    ds = DStheory(Events)
    ds.addBPA(A=[["事件A", "事件B"], ["事件A", "事件C"]], m=[0.3, 0.7], name="BPA1")
    ds.addBPA(A=[["事件A"], ["事件B"], ["事件C"]], m=[0.1, 0.2, 0.7], name="BPA2")
    event_f = ["事件A", "事件B"]
    print([ds.belief(event_f, name="BPA1"), ds.plausible(event_f, name="BPA1")])
    # ds.showFrame()
    ds.combineBPA(names=["BPA1", "BPA2"], name="BPA3")
    

    # 函数测试
    # print(getBinaryOneIndex(0b110))
    # print(distribute_unique([1, 2, 3, 4, 5], 3))
    # print(getBinaryFromIndex([1, 2, 3]), bin(getBinaryFromIndex([1, 2, 3])))
    

