def comb_recursive(arr, m):
    """从 arr 中取 m 个元素的组合"""
    if m == 0:
        return [[]]
    if len(arr) < m:
        return []
    with_first = [[arr[0]] + rest for rest in comb_recursive(arr[1:], m-1)]
    without_first = comb_recursive(arr[1:], m)
    return with_first + without_first

# 示例
arr = [1, 2, 3, 4, 5]
result = comb_recursive(arr, 3)
print(result)


# 用递归写真是太直观太优雅了。我想到了这个思路，但没想到能直接实现。它在我的基础上，就只有最终结果的区别。