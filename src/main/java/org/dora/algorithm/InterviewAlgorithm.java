package org.dora.algorithm;

import java.util.HashMap;

/**
 * 常见算法
 *
 * @author liulu
 * @date 2019-03-20
 */
public class InterviewAlgorithm {

    public static void main(String[] args) {
        InterviewAlgorithm interviewAlgorithm = new InterviewAlgorithm();
        System.out.println(interviewAlgorithm.lengthOfLongestSubstringKDistinct("abbcda", 2));
    }

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
     * 字符串中最长无K个子串
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

        int left = 0;

        int end = 0;


        HashMap<Character, Integer> map = new HashMap<>();
        while (end < s.length()) {
            int num = map.getOrDefault(s.charAt(end), 0);

            map.put(s.charAt(end), ++num);

            while (map.size() > k) {
                int edge = map.getOrDefault(s.charAt(left), 0);
                edge--;
                if (edge == 0) {
                    map.remove(s.charAt(left));
                } else {
                    map.put(s.charAt(left), edge);
                }
                left++;

            }

            result = Math.max(result, end - left + 1);
        }


        return result;
    }

}
