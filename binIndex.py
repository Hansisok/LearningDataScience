# 获取1个二进制数为1的位数。
def getOneNum(num: int):
    print(bin(num))
    count = 0
    index = []
    while num != 0:
        if num%2 == 1:
            index.append(count)
        num = num//2
        print(num)
        count += 1        
    return index