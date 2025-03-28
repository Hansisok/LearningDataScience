def distribute_unique(arr, n):
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


# 示例
arr = [1, 2, 3]
n = 6
combinations = distribute_unique(arr, n)

for combo in combinations:
    print(combo)
