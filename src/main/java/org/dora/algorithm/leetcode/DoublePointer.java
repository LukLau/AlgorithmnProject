package org.dora.algorithm.leetcode;

/**
 * @author dora
 * @date 2019/11/5
 */
public class DoublePointer {

    public static void main(String[] args) {
        DoublePointer pointer = new DoublePointer();
        String s = "ADOBECODEBANC";

        String t = "ABC";

        pointer.minWindow(s, t);
    }

    /**
     * 75. Sort Colors
     *
     * @param nums
     */
    public void sortColors(int[] nums) {
        if (nums == null || nums.length == 0) {
            return;
        }
        int zeroNum = 0;

        int twoNum = nums.length - 1;

        for (int i = 0; i < nums.length; i++) {
            while (nums[i] == 2 && i < twoNum) {
                this.swap(nums, i, twoNum--);
            }
            while (nums[i] == 0 && i > zeroNum) {
                this.swap(nums, i, zeroNum++);
            }
        }

    }

    private void swap(int[] nums, int i, int j) {
        int val = nums[i];
        nums[i] = nums[j];
        nums[j] = val;
    }


    /**
     * 左右边界移动 形成窗口
     * 移动窗口的时候 判断窗口与题意的雅秋
     * 76. Minimum Window Substring
     *
     * @param s
     * @param t
     * @return
     */
    public String minWindow(String s, String t) {
        if (s == null || t == null) {
            return "";
        }
        int m = s.length();

        int count = t.length();

        int[] hash = new int[256];

        for (int i = 0; i < count; i++) {
            hash[t.charAt(i) - '0']++;
        }

        int end = 0;

        int begin = 0;

        int head = 0;

        int result = Integer.MAX_VALUE;

        while (end < m) {
            // 为什么不可以使用while
            // 因为每次移动窗口 需要判断是否达到边界条件
//            while (hash[s.charAt(end++) - '0']-- > 0) {
//                count--;
//            }
            if (hash[s.charAt(end++) - '0']-- > 0) {
                count--;
            }
            while (count == 0) {
                if (end - begin < result) {

                    head = begin;

                    result = end - begin;
                }

                if (hash[s.charAt(begin++) - '0']++ == 0) {
                    count++;
                }
            }
        }
        if (result != Integer.MAX_VALUE) {
            return s.substring(head, head + result);
        }
        return "";
    }
}
