package org.dora.algorithm.leetcode.v3;

import java.util.HashMap;
import java.util.Stack;

/**
 * @author lauluk
 * @date 2020/3/26
 */
public class LeetCodeHard {


    /**
     * 5. Longest Palindromic Substring
     *
     * @param s
     * @return
     */
    private int palindromeResult = Integer.MIN_VALUE;
    private int palindromeHead = 0;

    /**
     * 3. Longest Substring Without Repeating Characters
     *
     * @param s
     * @return
     */
    public int lengthOfLongestSubstring(String s) {
        if (s == null | s.isEmpty()) {
            return 0;
        }
        HashMap<Character, Integer> map = new HashMap<>();
        int left = 0;
        int result = 0;
        for (int i = 0; i < s.length(); i++) {
            if (map.containsKey(s.charAt(i))) {
                left = Math.max(left, map.get(s.charAt(i)) + 1);
            }
            map.put(s.charAt(i), i);
            result = Math.max(result, i - left + 1);
        }
        return result;
    }

    public int lengthOfLongestSubstringII(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        }
        int left = 0;
        int result = 0;
        int[] hash = new int[256];
        int len = s.length();
        for (int i = 0; i < len; i++) {
            left = Math.max(left, hash[s.charAt(i)]);
            result = Math.max(result, i - left + 1);

            hash[s.charAt(i)] = i + 1;
        }
        return result;

    }

    /**
     * 5. Longest Palindromic Substring
     *
     * @param s
     * @return
     */
    public String longestPalindrome(String s) {
        if (s == null || s.isEmpty()) {
            return "";
        }
        int len = s.length();
        for (int i = 0; i < len; i++) {
            intervalPalindrome(i, i + 1, s);
            intervalPalindrome(i, i, s);
        }
        if (palindromeResult != Integer.MIN_VALUE) {
            return s.substring(palindromeHead, palindromeHead + palindromeResult);
        }
        return "";

    }

    private void intervalPalindrome(int j, int k, String s) {
        while (j >= 0 && k < s.length() && s.charAt(j) == s.charAt(k)) {
            if (k - j + 1 > palindromeResult) {
                palindromeHead = j;
                palindromeResult = k - j + 1;
            }
            j--;
            k++;
        }
    }

    public String longestPalindromeII(String s) {
        if (s == null || s.isEmpty()) {
            return "";
        }
        int len = s.length();
        boolean[][] dp = new boolean[len][len];
        int result = Integer.MIN_VALUE;
        int begin = 0;
        for (int i = 0; i < len; i++) {
            for (int j = 0; j <= i; j++) {
                if (s.charAt(j) == s.charAt(i)) {
                    if (i - j < 2) {
                        dp[j][i] = true;
                    } else {
                        dp[j][i] = dp[j + 1][i - 1];
                    }
                }
                if (dp[j][i] && i - j + 1 > result) {
                    begin = j;
                    result = i - j + 1;
                }
            }
        }
        if (result != Integer.MIN_VALUE) {
            return s.substring(begin, begin + result);
        }
        return "";


    }


    /**
     * 10. Regular Expression Matching
     *
     * @param s
     * @param p
     * @return
     */
    public boolean isMatch(String s, String p) {
        if (s == null) {
            return true;
        }
        if (p == null) {
            return false;
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
                if (s.charAt(i - 1) == p.charAt(j - 1) || p.charAt(j - 1) == '.') {
                    dp[i][j] = dp[i - 1][j - 1];
                } else if (p.charAt(j - 1) == '*') {
                    if (p.charAt(j - 2) != '.' && p.charAt(j - 2) != s.charAt(i - 1)) {
                        dp[i][j] = dp[i][j - 2];
                    } else {
                        dp[i][j] = dp[i - 1][j] || dp[i][j - 1] || dp[i][j - 2];
                    }

                }

            }
        }
        return dp[m][n];
    }


    public boolean isMatchV2(String s, String p) {
        if (s.isEmpty()) {
            return p.isEmpty();
        }
        boolean firstMatch = !p.isEmpty() && (s.charAt(0) == p.charAt(0) || p.charAt(0) == '.');

        if (s.length() >= 2 && p.charAt(1) == '*') {
            return isMatchV2(s, p.substring(2)) ||
                    (firstMatch && isMatchV2(s.substring(1), p));
        } else {
            return firstMatch && isMatchV2(s.substring(1), p.substring(1));
        }


    }

    /**
     * 32. Longest Valid Parentheses
     *
     * @param s
     * @return
     */
    public int longestValidParentheses(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        }
        Stack<Integer> stack = new Stack<>();
        int result = 0;
        int left = 0;
        int len = s.length();
        for (int i = 0; i < len; i++) {
            char word = s.charAt(i);
            if (word == '(') {
                stack.push(i);
            } else {
                if (stack.isEmpty()) {
                    left = i;
                } else {
                    stack.pop();
                }
                if (stack.isEmpty()) {
                    result = Math.max(result, i - left);
                } else {
                    result = Math.max(result, i - stack.peek());
                }
            }
        }
        return result;
    }

    public int longestValidParenthesesV2(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        }
        Stack<Integer> stack = new Stack<>();
        int len = s.length();
        for (int i = 0; i < len; i++) {
            char word = s.charAt(i);
            if (stack.isEmpty() || word == '(') {
                stack.push(i);
            } else {
                if (s.charAt(stack.peek()) == '(') {
                    stack.pop();
                } else {
                    stack.push(i);
                }
            }
        }
        if (stack.isEmpty()) {
            return s.length();
        }
        int close = s.length();
        int result = 0;
        while (!stack.isEmpty()) {
            int side = stack.pop();
            result = Math.max(result, close - side - 1);
            close = side;
        }
        result = Math.max(result, close);
        return result;

    }


    /**
     * 33. Search in Rotated Sorted Array
     *
     * @param nums
     * @param target
     * @return
     */
    public int search(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return -1;
        }
        return 0;
    }


}
