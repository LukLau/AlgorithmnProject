package org.dora.algorithm.leetcode;

import org.dora.algorithm.datastructe.TreeNode;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * @author dora
 * @date 2019-04-30
 */
public class VipPage {

    public static void main(String[] args) {
        VipPage vipPage = new VipPage();
//        System.out.println(vipPage.isOneEditDistance("aa", "b"));
//        System.out.println(vipPage.isOneEditDistance("a", "b"));
//        System.out.println(vipPage.isOneEditDistance("ab", "b"));
//        System.out.println(vipPage.isOneEditDistance("a", "ba"));
        vipPage.compareVersion("0.1", "1.1");
        vipPage.reversePower(8);

    }

    /**
     * 156、Binary Tree Upside Down
     * <a href="https://www.lintcode.com/problem/binary-tree-upside-down/description">
     * 从低到上旋转树</a>
     *
     * @param root
     * @return
     */
    public TreeNode upsideDownBinaryTree(TreeNode root) {
        // write your code here
        if (root == null || root.left == null) {
            return root;
        }

        TreeNode left = root.left;

        TreeNode right = root.right;

        TreeNode ans = this.upsideDownBinaryTree(left);

        left.left = right;

        left.right = root;

        root.left = null;

        root.right = null;
        return ans;

    }

    /**
     * 159、Longest Substring with At Most Two Distinct Characters
     * <a href="http://www.cnblogs.com/grandyang/p/5185561.html>最大两个差异字符</a>
     * todo 滑动窗口 不懂
     *
     * @param s
     * @return
     */
    public int lengthOfLongestSubstringTwoDistinct(String s) {

        int left = 0;
        int end = 0;
        int result = 0;
        HashMap<Character, Integer> map = new HashMap<>();
        while (end < s.length()) {

            int count = map.getOrDefault(s.charAt(end), 0);

            map.put(s.charAt(end++), ++count);

            while (map.size() > 2) {
                int num = map.getOrDefault(s.charAt(left), 0);

                --num;
                if (num == 0) {
                    map.remove(s.charAt(left));
                    left++;
                } else {
                    map.put(s.charAt(left), num);
                }
            }
            result = Math.max(result, end - left);
        }
//        for (int i = 1; i < s.length(); ++i) {
//            if (s[i] == s[i - 1]) {
//                continue;
//            }
//            if (right >= 0 && s[right] != s[i]) {
//                res = Math.max(res, i - left);
//                left = right + 1;
//            }
//            right = i - 1;
//        }
        return Math.max(s.length() - left, result);
    }

    /**
     * 161 One Edit Distance
     * <a href="https://www.cnblogs.com/grandyang/p/5184698.html">one edit distance</a>
     *
     * @param s
     * @param t
     * @return
     */
    public boolean isOneEditDistance(String s, String t) {
        if (s == null || t == null) {
            return false;
        }
        if (s.length() < t.length()) {
            return this.isOneEditDistance(t, s);
        }
        int m = s.length();
        int n = t.length();
        int diff = m - n;
        if (diff >= 2) {
            return false;
        } else if (diff == 1) {
            for (int i = 0; i < n; i++) {
                if (s.charAt(i) != t.charAt(i)) {
                    return s.charAt(i + 1) == t.charAt(i);
                }
            }
        } else {
            return s.charAt(0) == t.charAt(0);
        }
        return false;
    }


    /**
     * 163 Missing Ranges
     * <a href="">https://www.lintcode.com/problem/missing-ranges/my-submissions</a>
     * todo 不懂
     *
     * @param nums
     * @param lower
     * @param upper
     * @return
     */
    public List<String> findMissingRanges(int[] nums, int lower, int upper) {
        List<String> ranges = new ArrayList<>();
        int prev = lower - 1;
        for (int i = 0; i <= nums.length; i++) {
            int curr = (i == nums.length) ? upper + 1 : nums[i];
            if (curr - prev >= 2) {
                ranges.add(this.getRange(prev + 1, curr - 1));
            }
            prev = curr;
        }
        return ranges;
    }

    private String getRange(int from, int to) {
        return (from == to) ? String.valueOf(from) : from + "->" + to;
    }


    /**
     * 164. Maximum Gap
     * todo 不懂
     *
     * @param nums
     * @return
     */
    public int maximumGap(int[] nums) {
        return 0;
    }

    /**
     * 165. Compare Version Numbers
     *
     * @param version1
     * @param version2
     * @return
     */
    public int compareVersion(String version1, String version2) {
        if (version1 == null || version2 == null) {
            return 0;
        }
        String[] v1 = version1.split("\\.");
        String[] v2 = version2.split("\\.");
        int i = 0;
        int j = 0;
        while (i < v1.length || j < v2.length) {

            int value1 = i < v1.length ? Integer.parseInt(v1[i]) : 0;

            int value2 = j < v2.length ? Integer.parseInt(v2[j]) : 0;

            if (value1 < value2) {
                return -1;
            } else if (value1 > value2) {
                return 1;
            } else {
                i++;
                j++;
            }
        }
        return 0;
    }


    /**
     * 167. Two Sum II - Input array is sorted
     *
     * @param numbers
     * @param target
     * @return
     */
    public int[] twoSum(int[] numbers, int target) {
        if (numbers == null || numbers.length == 0) {
            return new int[]{};
        }
        int[] ans = new int[2];
        int left = 0;
        int right = numbers.length - 1;
        while (left < right) {
            int value = numbers[left] + numbers[right];
            if (value == target) {
                ans[0] = left + 1;
                ans[1] = right + 1;
                return ans;
            } else if (value < target) {
                left++;
            } else {
                right--;
            }
        }
        return ans;
    }


    /**
     * 168. Excel Sheet Column Title
     *
     * @param n
     * @return
     */
    public String convertToTitle(int n) {
        if (n <= 0) {
            return "";
        }
        String s = "";
        while (n > 0) {
            char c = (char) ((n - 1) % 26 + 'A');
            s = c + s;
            n = (n - 1) / 26;
        }
        return s;
    }


    /**
     * 169. Majority Element
     *
     * @param nums
     * @return
     */
    public int majorityElement(int[] nums) {
        if (nums == null || nums.length == 0) {
            return -1;
        }
        HashMap<Integer, Integer> hash = new HashMap<>();
        for (int num : nums) {
            int count = hash.getOrDefault(num, 0);

            hash.put(num, ++count);
        }
        for (int num : nums) {
            int count = hash.get(num);
            if (2 * count > nums.length) {
                return num;
            }
        }
        return -1;
    }


    /**
     * 186、Reverse Words in a String II
     *
     * @param str
     * @return
     */
    public char[] reverseWords(char[] str) {
        // write your code here
        if (str == null || str.length == 0) {
            return new char[]{};
        }
        String string = String.valueOf(str);

        StringBuilder sb = new StringBuilder();

        int endIndex = string.length() - 1;
        while (endIndex >= 0) {

            if (string.charAt(endIndex) == ' ') {
                endIndex--;
                continue;
            }

            int startIndex = string.lastIndexOf(" ", endIndex);


            String tmp = string.substring(startIndex + 1, endIndex + 1);

            sb.append(tmp);

            if (startIndex > 0) {
                sb.append(" ");
            }
            endIndex = startIndex - 1;
        }
        return sb.toString().toCharArray();
    }

    public int reversePower(int n) {
        int result = 0;
        while (n != 0) {
            result += n & 1;
            n >>>= 1;
            result <<= 1;
        }
        return result;
    }


    /**
     * 243、Shortest Word Distance
     * <a href="https://www.cnblogs.com/grandyang/p/5187041.html">最短单词距离</a>
     *
     * @param words
     * @param word1
     * @param word2
     * @return
     */
    public int shortestDistance(String[] words, String word1, String word2) {
        if (words == null || words.length == 0) {
            return -1;
        }
        int index1 = -1;
        int index2 = -1;
        int result = 0;
        for (int i = 0; i < words.length; i++) {
            if (word1.equals(words[i])) {
                index1 = i;
            } else if (word2.equals(words[i])) {
                index2 = i;
            }
            if (index1 != -1 && index2 != -1) {
                result = Math.max(result, Math.abs(index1 - index2));
            }
        }
        return result;
    }

    /**
     * 255、Verify Preorder Sequence in Binary Search Tree
     * <a href="http://www.cnblogs.com/grandyang/p/5327635.html">验证二叉树的先序遍历</a>
     *
     * @param preorder
     * @return
     */
    public boolean verifyPreorder(int[] preorder) {
        if (preorder == null || preorder.length == 0) {
            return false;
        }
        int size = preorder.length - 1;


        while (size > 0) {
            int tmp = size;
            while (tmp > 0 && preorder[tmp] > preorder[0]) {
                tmp--;
            }
            while (tmp > 0 && preorder[tmp] < preorder[0]) {
                tmp--;
            }
            if (tmp != 0) {
                return false;
            }
            size--;
        }
        return true;
    }


}
