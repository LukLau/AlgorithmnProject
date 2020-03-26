package org.dora.algorithm.leetcode.v3;

import java.util.HashMap;

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


}
