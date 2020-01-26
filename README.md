# 算法
    leetCode 以及 剑指 offer 算法相关题目
    
-----------

## 超时问题
> 超时问题往往代表代码正常    
由于存在重复计算问题导致超时、    
故需考虑记录已经计算的值或者状态、减少次数

> [除法问题](https://leetcode.com/problems/divide-two-integers/)

> [数据流中的中位数](https://leetcode.com/problems/find-median-from-data-stream/)
key point: 考虑使用双队列来维持查询操作
 
## 深度、广度优先问题
> 深度遍历靠栈来实现

> 宽度遍历靠队列实现

> 深度、广度遍历核心还是回溯法、回溯法需要注意更改回溯状态

> 遍历得考虑好边界退出条件

> [课程调度]()

> [八皇后问题](https://leetcode.com/problems/n-queens/)

> [找到被包围的区域三种写法](https://leetcode.com/problems/surrounded-regions/)

> [符号添加不同方式 即遍历方式](https://leetcode.com/problems/different-ways-to-add-parentheses/)

> [墙和门问题](https://www.lintcode.com/problem/walls-and-gates/description)

> [生命游戏](https://leetcode.com/problems/game-of-life/solution/)


## 连续序列问题
> [数组中连续最大乘积 keyCase: 考虑到连续序列](https://leetcode.com/problems/maximum-product-subarray/)

## Dp问题
>**动态规划解题关键**

> 第一个要点需要考虑 子问题方程式

> 第二个要点需要考虑方程式连续问题。由于子问题决定下一个问题的最优解。故动态规划方程式必须连贯起来

> [魔法匹配问题](https://www.cnblogs.com/grandyang/p/4401196.html
)

> [正则表达式匹配](https://leetcode.com/problems/regular-expression-matching/discuss/5651/Easy-DP-Java-Solution-with-detailed-Explanation)

> [KMP算法](https://leetcode.com/problems/implement-strstr/discuss/12956/C%2B%2B-Brute-Force-and-KMP)

> [获取八皇后个数](https://leetcode.com/problems/n-queens-ii/)

> [游戏棋盘生命值最少问题](https://leetcode.com/submissions/detail/226063539/)

> [房屋大盗II](https://leetcode.com/problems/house-robber-ii/)

> [求二维数组正方形最大面积](https://leetcode.com/problems/maximal-square/)
key point: dp[i-1][j-1], dp[i-1][j], dp[i][j-1];

> [房子刷漆问题](https://www.lintcode.com/problem/paint-house/description)

## 贪心问题


## 分治问题

## 位操问题
    
* **两个不同数^ ==相当于 无进位加法**
* **两个不同数& 相当于 判断是否有进位**
* **& (偶数- 1) 相当于 取模操作**
* **n & 1 相当于 对2取模操作 相当于每次获取二进制最末尾一位数字值**
* **n & 1 并且 n 无符号右移相当于二进制数据反转**
> [二进制数据反转并且满足32数](https://leetcode.com/problems/reverse-bits/)



## 滑动窗口问题问题

* **考虑使用双指针**
* **考虑使用队列**
* **[滑动窗口模板](https://leetcode.com/problems/minimum-window-substring/discuss/26808/Here-is-a-10-line-template-that-can-solve-most-'substring'-problems)**

> 滑动窗口关键值

> **窗口的大小 当窗口太小时 窗口往右扩展。窗口太大时缩小窗口**

> **窗口左右边界问题。确定如何移除窗口条件**

> **窗口需要进行初始化。慎重考虑初始化条件**

> **可以考虑使用list存储窗口的值**

> 滑动窗口经典题目  

* [滑动窗口最大值](https://www.nowcoder.com/practice/1624bc35a45c42c0bc17d17fa0cba788?tpId=13&tqId=11217&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking&tPage=4) 使用队列    

* [滑动窗口最多两个无差别子串](http://www.cnblogs.com/grandyang/p/5185561.html)

* [窗口最小数量](https://leetcode.com/problems/minimum-size-subarray-sum/)

* [会议室区间数量]()


# 二分搜索
> trick 数组移动
> 从左往右 (left + right) / 2 + 1
> 从右往左 (left + right) / 2 -1

> 边界值left <= right 或者 left < right 取决于代码思路

* [34. Find First and Last Position of Element in Sorted Array](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/)
* [154. Find Minimum in Rotated Sorted Array II](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array-ii/)
* [162. Find Peak Element](https://leetcode.com/problems/find-peak-element/)




----

## 比较难题目
> [两个数取中位数](https://leetcode.com/problems/median-of-two-sorted-arrays/)


-----
### 常用数学理论以及算法

> 空集是任何一个集合的子集

> [约瑟夫环](https://www.nowcoder.com/practice/f78a359491e64a50bce2d89cff857eb6?tpId=13&tqId=11199&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking&tPage=3)

> [格雷码以及二进制编码成格雷码](https://baike.baidu.com/item/%E6%A0%BC%E9%9B%B7%E7%A0%81/6510858?fr=aladdin)

> [求解质数的个数](https://leetcode.com/submissions/detail/121785675/)

> [约瑟夫环三种解法](https://blog.csdn.net/weixin_38214171/article/details/80352921)

> [计算素数个数](https://leetcode.com/problems/count-primes/)

> [字典树](https://leetcode.com/submissions/detail/226455100/)

> [摩尔投票法](https://www.jianshu.com/p/c19bb428f57a) **关键在于找出候选者**

> [位运算](https://leetcode.com/problems/single-number-ii/) 需要使用两位

> [逆波兰数](https://leetcode.com/problems/reverse-words-in-a-string/)

> [求斜率 keyCase:求斜率](https://leetcode.com/submissions/detail/186532613/)

> [小数到循环小数](https://leetcode.com/problems/fraction-to-recurring-decimal/)

> [计算数字0的个数](https://leetcode.com/problems/factorial-trailing-zeroes/)

> [城市天际图](https://leetcode.com/problems/the-skyline-problem/)

> [数字中1的个数](https://leetcode.com/problems/number-of-digit-one/) todo

> [一个数位数的个数](https://leetcode.com/problems/add-digits/discuss/68580/Accepted-C%2B%2B-O(1)-time-O(1)-space-1-Line-Solution-with-Detail-Explanations)

> [数组中查找两个不同的数字](https://leetcode.com/submissions/detail/194395415/)

key point: 将数组分割成两个不同的部分

key point: digit = 1 + (n-1) mod 9

> [求出第n个丑数的值](https://leetcode.com/problems/ugly-number-ii/)

> [将正数转换为英文单词](https://www.lintcode.com/problem/integer-to-english-words/description)

> [在数组中找到重复的数字](https://leetcode.com/problems/find-the-duplicate-number/solution/)
key point: Floyd's Tortoise and Hare
不允许使用任务额外空间

> [最佳见面地点](http://www.lintcode.com/problem/best-meeting-point/description)
key point:曼哈顿距离法
 
----
## 位运算
------

> [二进制反转](https://leetcode.com/problems/reverse-bits/)

思路: 进行位运算

---
# 遍历问题

> [树层次遍历O(1)空间](https://leetcode.com/problems/populating-next-right-pointers-in-each-node/)

> [树层次遍历II O(1)空间](https://leetcode.com/problems/populating-next-right-pointers-in-each-node-ii/)

> [判断搜索树的先序遍历是否符合规范](https://www.lintcode.com/problem/verify-preorder-sequence-in-binary-search-tree/description)

# 数据结构
树的定义: tree is an undirected graph in which any two vertices are connected by  exactly  one path. In other words, any connected graph without simple cycles is a tree.”
> [判断无向图是否可以组装成树](https://github.com/grandyang/leetcode/issues/261)
key point: 1、可以考虑使用深度和宽度 2、树的顶点 - 1 = 边数

> [异形字典](https://www.lintcode.com/problem/alien-dictionary/description)
key point: 有向连接图 排序


# 优化问题

> [移动数组中0的个数](https://leetcode.com/problems/move-zeroes/submissions/)
key point: 考虑用下标移动来优化