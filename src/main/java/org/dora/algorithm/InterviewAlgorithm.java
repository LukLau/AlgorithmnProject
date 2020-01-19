package org.dora.algorithm;

import org.dora.algorithm.datastructe.TreeNode;

import java.util.*;

/**
 * 常见算法
 *
 * @author liulu
 * @date 2019-03-20
 */
public class InterviewAlgorithm {

    /**
     * 0-1背包问题
     *
     * @param weights
     * @param w
     * @param values
     * @param v
     * @return
     */
    public int knapsack(int[] weights, int w, int[] values, int v) {
        if (weights == null || values == null) {
            return 0;
        }
        int[][] dp = new int[v + 1][w + 1];
        for (int i = 1; i <= v; i++) {
            for (int j = 1; j <= w; j++) {
                if (weights[i - 1] > j) {
                    dp[i][j] = dp[i - 1][j];
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i - 1][j - weights[i - 1]] + values[i - 1]);
                }
            }
        }
        return dp[v][w];
    }

    /**
     * 房屋抢劫
     *
     * @param nums
     * @return
     */
    public int hourseRob(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int[] dp = new int[nums.length + 1];
        for (int i = 1; i <= nums.length; i++) {
            if (i == 1) {
                dp[i] = Math.max(0, nums[i - 1]);
            } else {
                dp[i] = Math.max(dp[i - 2] + nums[i - 1], dp[i - 1]);
            }
        }
        return dp[nums.length];
    }

    /**
     * 416、Partition Equal Subset Sum
     * 一个数组 分成值 相等的两个部分
     *
     * @param nums
     * @return
     */
    public boolean canPartition(int[] nums) {
        if (nums == null || nums.length == 0) {
            return false;
        }
        int sum = 0;
        for (int num : nums) {
            sum += num;
        }
        if ((sum & 1) == 1) {
            return false;
        }
        sum = sum / 2;
        boolean[][] dp = new boolean[nums.length + 1][sum + 1];

        dp[0][0] = true;

        for (int i = 1; i < nums.length + 1; i++) {
            dp[i][0] = true;
        }
        for (int j = 1; j < sum + 1; j++) {
            dp[0][j] = false;
        }

        for (int i = 1; i < nums.length + 1; i++) {
            for (int j = 1; j < sum + 1; j++) {
                if (j >= nums[i - 1]) {
                    dp[i][j] = (dp[i - 1][j] || dp[i - 1][j - nums[i - 1]]);
                } else {
                    dp[i][j] = dp[i - 1][j];
                }
            }
        }

        return dp[nums.length][sum];
    }

    /**
     * random7 转换成 random10
     *
     * @return
     */
    public int randomConvert() {
        return -1;
    }

    /**
     * 字符串中最长无重复K个字符
     *
     * @param s
     * @param k
     * @return
     */
    public int lengthOfLongestSubstringKDistinct(String s, int k) {
        if (s == null) {
            return 0;
        }
        int result = 0;
        HashMap<Character, Integer> map = new HashMap<>();
        int left = 0;
        for (int i = 0; i < s.length(); i++) {
            map.put(s.charAt(i), i);
            while (map.size() > k) {
                if (map.get(s.charAt(i)) == left) {
                    map.remove(s.charAt(i));
                }
                left++;
            }
            result = Math.max(result, i - left + 1);
        }
        return result;
    }


    /**
     * kmp 算法
     *
     * @param s
     * @param t
     * @return
     */
    public int kmpAlgorithm(String s, String t) {
        return -1;
    }

    /**
     * 魔法匹配
     *
     * @param s
     * @param p
     * @return
     */
    public boolean isMatch(String s, String p) {
        return true;
    }

    public boolean isMatchII(String s, String p) {
        return false;
    }

    /**
     * 51. N-Queens
     *
     * @param n
     * @return
     */
    public List<List<String>> solveNQueens(int n) {
        if (n <= 0) {
            return Collections.emptyList();
        }
        char[][] nQueens = new char[n][n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                nQueens[i][j] = '.';
            }
        }
        List<List<String>> ans = new ArrayList<>();

        this.checkQueens(ans, 0, nQueens);

        return ans;
    }

    private void checkQueens(List<List<String>> ans, int row, char[][] nQueens) {
        if (row == nQueens.length) {
            List<String> tmp = new ArrayList<>();

            for (char[] rowValue : nQueens) {
                tmp.add(String.valueOf(rowValue));
            }
            ans.add(tmp);
        }

        for (int j = 0; j < nQueens.length; j++) {
            if (this.isValidQueens(nQueens, j, row)) {
                nQueens[row][j] = 'Q';
                this.checkQueens(ans, row + 1, nQueens);
                nQueens[row][j] = '.';
            }
        }
    }

    private boolean isValidQueens(char[][] nQueens, int column, int row) {
        for (int i = row - 1; i >= 0; i--) {
            if (nQueens[i][column] == 'Q') {
                return false;
            }
        }
        for (int i = row - 1, j = column - 1; i >= 0 && j >= 0; i--, j--) {
            if (nQueens[i][j] == 'Q') {
                return false;
            }
        }
        for (int i = row - 1, j = column + 1; i >= 0 && j < nQueens.length; i--, j++) {
            if (nQueens[i][j] == 'Q') {
                return false;
            }
        }
        return true;
    }


    /**
     * 52. N-Queens II
     *
     * @param n
     * @return
     */
    public int totalNQueens(int n) {
        if (n <= 0) {
            return 0;
        }
        int[] dp = new int[n];
        return this.totalNQueens(dp, 0, n);
    }

    private int totalNQueens(int[] dp, int row, int n) {
        int count = 0;
        if (row == n) {
            count++;
            return count;
        }
        for (int i = 0; i < n; i++) {
            if (this.isValidTotalNQueens(dp, i, row)) {
                dp[row] = i;
                count += this.totalNQueens(dp, row + 1, n);
                dp[row] = -1;
            }
        }
        return count;
    }

    private boolean isValidTotalNQueens(int[] dp, int column, int row) {
        for (int i = row - 1; i >= 0; i--) {
            if (dp[i] == column || Math.abs(i - row) == Math.abs(dp[i] - column)) {
                return false;
            }
        }
        return true;
    }

    /**
     * 今日头条: S型打印二叉树
     *
     * @param root
     * @return
     */
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        List<List<Integer>> ans = new ArrayList<>();
        LinkedList<TreeNode> linkedList = new LinkedList<>();
        linkedList.add(root);
        boolean leftToRight = true;
        while (!linkedList.isEmpty()) {
            int size = linkedList.size();
            Integer[] tmp = new Integer[size];
            for (int i = 0; i < size; i++) {

                TreeNode node = linkedList.poll();

                int index = leftToRight ? i : size - 1 - i;

                tmp[index] = node.val;

                if (node.left != null) {
                    linkedList.add(node.left);
                }
                if (node.right != null) {
                    linkedList.add(node.right);
                }
            }
            leftToRight = !leftToRight;
            ans.add(Arrays.asList(tmp));
        }
        return ans;
    }

    public static void main(String[] args) {

    }


}
