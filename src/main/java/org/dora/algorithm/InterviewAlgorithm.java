package org.dora.algorithm;

import org.dora.algorithm.datastructe.ListNode;
import org.dora.algorithm.datastructe.TreeNode;

import java.util.HashMap;

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
     * 有序链表 组成 BST树
     *
     * @param head
     * @return
     */
    public TreeNode sortedListToBST(ListNode head) {
        if (head == null) {
            return null;
        }
        return buildBST(head, null);
    }

    private TreeNode buildBST(ListNode head, ListNode end) {
        if (head == end) {
            return null;
        }
        ListNode slow = head;
        ListNode fast = head;
        while (fast != end && fast.next != end) {
            fast = fast.next.next;
            slow = slow.next;
        }
        TreeNode root = new TreeNode(slow.val);
        root.left = buildBST(head, slow);
        root.right = buildBST(slow.next, end);
        return root;
    }

    /**
     * 排序数组中查找中位数
     *
     * @param nums1
     * @param nums2
     * @return
     */
    public double findMediaSortedArray(int[] nums1, int[] nums2) {
        if (nums1 == null || nums2 == null) {
            return 0;
        }
        int m = nums1.length;
        int n = nums2.length;
        if (m < n) {
            return findMediaSortedArray(nums2, nums1);
        }
        int imin = 0;
        int imax = nums1.length;
        int left = 0;
        int right = 0;
        while (imin <= imax) {
            int i = imin + (imax - imin) / 2;
            int j = (m + n + 1) / 2 - i;
            if (i < imax && nums1[i] < nums2[j - 1]) {
                imin = i + 1;
            } else if (i > 0 && nums1[i - 1] > nums2[j]) {
                imax = i - 1;
            } else {
                if (i == 0) {
                    left = nums2[j - 1];
                } else if (j == 0) {
                    left = nums1[i - 1];
                } else {
                    left = Math.max(nums1[i - 1], nums2[j - 1]);
                }
                if ((m + n) % 1 != 0) {
                    return left;
                }
                if (i == m) {
                    right = nums2[j];
                } else if (j == n) {
                    right = nums1[i];
                } else {
                    right = Math.min(nums1[i], nums1[j]);
                }
                return (left + right) / 2.0;
            }
        }
        return -1;
    }

    /**
     * 最长无重复字串
     * 有三种解法
     *
     * @param s
     * @return
     */
    public String longestPalindrome(String s) {
        if (s == null || s.length() == 0) {
            return "";
        }
        int m = s.length();
        int left = 0;
        int len = 0;
        boolean[][] dp = new boolean[m][m];
        for (int i = 0; i < s.length(); i++) {
            for (int j = 0; j <= i; j++) {
                if (i - j < 2) {
                    dp[j][i] = s.charAt(i) == s.charAt(j);
                } else {
                    dp[j][i] = s.charAt(i) == s.charAt(j) && dp[j + 1][i - 1];
                }
                if (dp[j][i] && i - j + 1 > len) {
                    left = j;
                    len = i - j + 1;
                }
            }
        }
        if (len > 0) {
            return s.substring(left, left + len);
        }
        return s;
    }

    /**
     * 正则表达式匹配
     *
     * @param s
     * @param p
     * @return
     */
    public boolean isMatch(String s, String p) {
        if (s == null && p == null) {
            return false;
        } else if (s == null) {
            return true;
        }
        int m = s.length();
        int n = p.length();
        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[0][0] = true;
        for (int j = 1; j <= n; j++) {
            dp[0][j] = p.charAt(j - 1) == '*' && dp[0][j - 2];
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (p.charAt(j - 1) == s.charAt(i - 1) || p.charAt(j - 1) == '.') {
                    dp[i][j] = dp[i - 1][j - 1];
                } else if (p.charAt(j - 1) == '*') {
                    if (s.charAt(i - 1) != p.charAt(j - 1) && p.charAt(j - 2) != '.') {
                        dp[i][j] = dp[i][j - 2];
                    } else {
                        dp[i][j] = dp[i - 1][j] || dp[i][j - 1] || dp[i][j - 2];
                    }
                }
            }
        }
        return dp[m][n];
    }

    /**
     * 魔法匹配
     *
     * @param s
     * @param p
     * @return
     */
    public boolean isMatchII(String s, String p) {
        if (s == null && p == null) {
            return false;
        } else if (s == null) {
            return true;
        }
        return false;
    }

    /**
     * 判断二叉搜索树后序遍历是否正确
     *
     * @param sequence
     * @return
     */
    public boolean VerifySquenceOfBST(int[] sequence) {
        if (sequence == null || sequence.length == 0) {
            return false;
        }
        return VerifySquenceOfBST(sequence, 0, sequence.length - 1);
    }

    private boolean VerifySquenceOfBST(int[] sequence, int start, int end) {
        if (start == end) {
            return true;
        }
        int tmp = start;
        while (tmp < end && sequence[tmp] < sequence[end]) {
            tmp++;
        }
        int mid = tmp;
        while (mid < end && sequence[mid] > sequence[end]) {
            mid++;
        }
        if (mid < end) {
            return false;
        }
        if (tmp == start || mid == end) {
            return true;
        }
        return VerifySquenceOfBST(sequence, start, tmp - 1) && VerifySquenceOfBST(sequence, tmp, end - 1);
    }


}
