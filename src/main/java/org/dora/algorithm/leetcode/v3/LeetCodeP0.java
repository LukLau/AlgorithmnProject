package org.dora.algorithm.leetcode.v3;

import org.dora.algorithm.datastructe.ListNode;

import java.util.*;

/**
 * @author liulu12@xiaomi.com
 * @date 2020/2/22
 */
public class LeetCodeP0 {


    public static void main(String[] args) {
        LeetCodeP0 codeV3 = new LeetCodeP0();

        int[] nums = new int[]{-1, 0, 1, 2, -1, -4};
        codeV3.threeSum(nums);
    }

    /**
     * 1. Two Sum
     *
     * @param nums
     * @param target
     * @return
     */
    public int[] twoSum(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return new int[]{};
        }
        HashMap<Integer, Integer> hash = new HashMap<>();
        int[] ans = new int[2];
        for (int i = 0; i < nums.length; i++) {
            if (hash.containsKey(target - nums[i])) {
                ans[0] = hash.get(target - nums[i]);
                ans[1] = i;
                return ans;
            }
            hash.put(nums[i], i);
        }
        return ans;
    }

    /**
     * 2. Add Two Numbers
     *
     * @param l1
     * @param l2
     * @return
     */
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        if (l1 == null && l2 == null) {
            return null;
        }
        ListNode root = new ListNode(0);
        ListNode dummy = root;
        int carry = 0;
        while (l1 != null || l2 != null || carry != 0) {
            int tmp = (l1 == null ? 0 : l1.val) + (l2 == null ? 0 : l2.val) + carry;

            carry = tmp / 10;

            ListNode node = new ListNode(tmp % 10);


            dummy.next = node;

            dummy = dummy.next;

            l1 = l1 == null ? null : l1.next;
            l2 = l2 == null ? null : l2.next;
        }
        return root.next;
    }

    /**
     * 3. Longest Substring Without Repeating Characters
     *
     * @param s
     * @return
     */
    public int lengthOfLongestSubstring(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        }
        int result = 0;

        int left = 0;

        int len = s.length();

        HashMap<Character, Integer> map = new HashMap<>();
        for (int i = 0; i < len; i++) {
            if (map.containsKey(s.charAt(i))) {
                left = Math.max(left, map.get(s.charAt(i)) + 1);
            }
            map.put(s.charAt(i), i);
            result = Math.max(result, i - left + 1);
        }
        return result;
    }

    public int lengthOfLongestSubstringV2(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        }
        int[] hash = new int[256];

        int result = 0;

        int left = 0;

        for (int i = 0; i < s.length(); i++) {

            left = Math.max(left, hash[s.charAt(i)]);

            result = Math.max(result, i - left + 1);

            hash[s.charAt(i)] = i + 1;

        }
        return result;
    }

    /**
     * 4. Median of Two Sorted Arrays
     *
     * @param nums1
     * @param nums2
     * @return
     */
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        if (nums1 == null || nums2 == null) {
            return Integer.MAX_VALUE;
        }
        return -1;
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
        int result = Integer.MIN_VALUE;
        int beginIndex = 0;
        boolean[][] dp = new boolean[len][len];
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
                    beginIndex = j;
                    result = i - j + 1;
                }
            }
        }
        if (result != Integer.MIN_VALUE) {
            return s.substring(beginIndex, beginIndex + result);
        }
        return "";
    }

    public String longestPalindromeV2(String s) {
        if (s == null || s.isEmpty()) {
            return "";
        }
        int[] result = new int[2];
        for (int i = 0; i < s.length(); i++) {
            intervalPalindrome(i, i, s, result);
            intervalPalindrome(i, i + 1, s, result);
        }
        if (result[1] != 0) {
            return s.substring(result[0], result[0] + result[1]);
        }
        return "";

    }

    private void intervalPalindrome(int j, int i, String s, int[] result) {
        while (j >= 0 && i < s.length() && s.charAt(j) == s.charAt(i)) {
            if (i - j + 1 > result[1]) {
                result[1] = i - j + 1;
                result[0] = j;
            }
            j--;
            i++;
        }
    }

    /**
     * 6. ZigZag Conversion
     *
     * @param s
     * @param numRows
     * @return
     */
    public String convert(String s, int numRows) {
        if (s == null || s.isEmpty()) {
            return "";
        }
        StringBuilder[] builders = new StringBuilder[numRows];

        char[] words = s.toCharArray();

        for (int i = 0; i < builders.length; i++) {
            builders[i] = new StringBuilder();
        }

        int index = 0;
        while (index < words.length) {
            for (int i = 0; i < numRows && index < words.length; i++) {
                builders[i].append(words[index++]);
            }
            for (int i = numRows - 2; i >= 1 && index < words.length; i--) {
                builders[i].append(words[index++]);
            }
        }
        for (int i = 1; i < builders.length; i++) {
            builders[0].append(builders[i]);
        }
        return builders[0].toString();
    }

    /**
     * 7. Reverse Integer
     *
     * @param x
     * @return
     */
    public int reverse(int x) {
        int result = 0;
        while (x != 0) {
            if (result > Integer.MAX_VALUE / 10 || result < Integer.MIN_VALUE / 10) {
                return 0;
            }
            result = result * 10 + x % 10;
            x /= 10;
        }
        return result;
    }

    /**
     * 8. String to Integer (atoi)
     *
     * @param str
     * @return
     */
    public int myAtoi(String str) {
        if (str == null) {
            return 0;
        }
        str = str.trim();
        if (str.isEmpty()) {
            return 0;
        }
        int sign = 1;

        int index = 0;

        if (str.charAt(index) == '-' || str.charAt(index) == '+') {
            sign = str.charAt(index) == '-' ? -1 : 1;
            index++;
        }
        Long result = 0L;
        while (index < str.length() && Character.isDigit(str.charAt(index))) {

            result = result * 10 + Character.getNumericValue(str.charAt(index));

            if (result > Integer.MAX_VALUE / 10) {
                return sign == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
            }
            index++;
        }
        return sign * result.intValue();
    }

    /**
     * 9. Palindrome Number
     *
     * @param x
     * @return
     */
    public boolean isPalindrome(int x) {
        if (x == 0) {
            return true;
        }
        if (x < 0) {
            return false;
        }
        if (x % 10 == 0) {
            return false;
        }
        int result = 0;

        while (x > result) {
            result = result * 10 + x % 10;

            x /= 10;
        }
        return result / 10 == x || result == x;

    }

    /**
     * 10. Regular Expression Matching
     *
     * @param s
     * @param p
     * @return
     */
    public boolean isMatch(String s, String p) {
        if (s == null || p == null) {
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
                    if (s.charAt(i - 1) != p.charAt(j - 2) && p.charAt(j - 2) != '.') {
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
     * 11. Container With Most Water
     *
     * @param height
     * @return
     */
    public int maxArea(int[] height) {
        if (height == null || height.length == 0) {
            return 0;
        }
        int left = 0;
        int right = height.length - 1;
        int result = 0;
        while (left < right) {
            result = Math.max(result, (right - left) * Math.min(height[right], height[left]));

            if (height[left] < height[right]) {
                left++;
            } else {
                right--;
            }
        }
        return result;
    }

    /**
     * todo
     * 12. Integer to Roman
     *
     * @param num
     * @return
     */
    public String intToRoman(int num) {
        return "";
    }

    /**
     * todo
     * 13. Roman to Integer
     *
     * @param s
     * @return
     */
    public int romanToInt(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        }
        return -1;
    }

    /**
     * 14. Longest Common Prefix
     *
     * @param strs
     * @return
     */
    public String longestCommonPrefix(String[] strs) {
        if (strs == null || strs.length == 0) {
            return "";
        }
        String prefix = strs[0];

        for (int i = 1; i < strs.length; i++) {

            String str = strs[i];


            while (str.indexOf(prefix) != 0) {
                prefix = prefix.substring(0, prefix.length() - 1);
            }
        }
        return prefix;
    }

    /**
     * 15. 3Sum
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> threeSum(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new ArrayList<>();
        }
        Arrays.sort(nums);
        List<List<Integer>> ans = new ArrayList<>();
        for (int i = 0; i < nums.length - 2; i++) {

            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            int left = i + 1;
            int right = nums.length - 1;
            while (left < right) {
                int value = nums[i] + nums[left] + nums[right];
                if (value == 0) {
                    ans.add(Arrays.asList(nums[i], nums[left], nums[right]));

                    while (left < right && nums[left] == nums[left + 1]) {
                        left++;
                    }
                    while (left < right && nums[right] == nums[right - 1]) {
                        right--;
                    }
                    left++;
                    right--;
                } else if (value < 0) {
                    left++;
                } else {
                    right--;
                }
            }

        }
        return ans;
    }


    /**
     * 16. 3Sum Closest
     *
     * @param nums
     * @param target
     * @return
     */
    public int threeSumClosest(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return Integer.MAX_VALUE;
        }
        Arrays.sort(nums);

        int result = nums[0] + nums[1] + nums[2];
        for (int i = 0; i < nums.length - 2; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            int left = i + 1;
            int right = nums.length - 1;
            while (left < right) {
                int value = nums[i] + nums[left] + nums[right];
                if (value == target) {
                    return value;
                } else if (value < target) {
                    left++;
                } else {
                    right--;
                }

                if (Math.abs(result - target) > Math.abs(value - target)) {
                    result = value;
                }
            }
        }
        return result;
    }


    /**
     * 17. Letter Combinations of a Phone Number
     *
     * @param digits
     * @return
     */
    public List<String> letterCombinations(String digits) {
        if (digits == null || digits.isEmpty()) {
            return new ArrayList<>();
        }
        String[] map = new String[]{"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};

        LinkedList<String> deque = new LinkedList<>();
        deque.add("");
        int length = digits.length();
        for (int i = 0; i < length; i++) {
            int index = Character.getNumericValue(digits.charAt(i));

            String word = map[index];

            while (deque.peekFirst().length() == i) {

                String poll = deque.poll();

                char[] chars = word.toCharArray();

                for (char tmp : chars) {
                    deque.add(poll + tmp);
                }
            }
        }
        return deque;
    }


    /**
     * 18. 4Sum
     *
     * @param nums
     * @param target
     * @return
     */
    public List<List<Integer>> fourSum(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return new ArrayList<>();
        }
        Arrays.sort(nums);
        List<List<Integer>> ans = new ArrayList<>();

        for (int i = 0; i < nums.length - 3; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }

            for (int j = i + 1; j < nums.length - 2; j++) {
                if (j > i + 1 && nums[j] == nums[j - 1]) {
                    continue;

                }
                int left = j + 1;
                int right = nums.length - 1;
                while (left < right) {
                    int value = nums[i] + nums[j] + nums[left] + nums[right];

                    if (value == target) {
                        ans.add(Arrays.asList(nums[i], nums[j], nums[left], nums[right]));
                        while (left < right && nums[left] == nums[left + 1]) {
                            left++;
                        }
                        while (left < right && nums[right] == nums[right - 1]) {
                            right--;
                        }
                        left++;
                        right--;
                    } else if (value < target) {
                        left++;
                    } else {
                        right--;
                    }
                }
            }
        }
        return ans;
    }


    /**
     * 19. Remove Nth Node From End of List
     *
     * @param head
     * @param n
     * @return
     */
    public ListNode removeNthFromEnd(ListNode head, int n) {
        if (head == null || n <= 0) {
            return null;
        }
        ListNode root = new ListNode(0);
        root.next = head;
        ListNode fast = root;
        ListNode slow = root;
        for (int i = 0; i < n; i++) {
            fast = fast.next;
        }
        while (fast.next != null) {
            slow = slow.next;
            fast = fast.next;
        }
        ListNode node = slow.next;

        slow.next = node.next;


        node.next = null;

        return root.next;
    }


    /**
     * 20. Valid Parentheses
     *
     * @param s
     * @return
     */
    public boolean isValid(String s) {
        if (s == null || s.isEmpty()) {
            return false;
        }
        Stack<Character> stack = new Stack<>();

        for (int i = 0; i < s.length(); i++) {

            char character = s.charAt(i);

            if (character == '(') {
                stack.push(')');
            } else if (character == '[') {
                stack.push(']');
            } else if (character == '{') {
                stack.push('}');
            } else {
                if (stack.isEmpty() || stack.peek() != character) {
                    return false;
                }
                stack.pop();
            }
        }
        return stack.isEmpty();
    }

    /**
     * 21. Merge Two Sorted Lists
     *
     * @param l1
     * @param l2
     * @return
     */
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null && l2 == null) {
            return null;
        }
        if (l1 == null || l2 == null) {
            return l1 == null ? l2 : l1;
        }
        if (l1.val <= l2.val) {
            l1.next = mergeTwoLists(l1.next, l2);

            return l1;
        } else {
            l2.next = mergeTwoLists(l1, l2.next);
            return l2;
        }
    }


    /**
     * 22. Generate Parentheses
     *
     * @param n
     * @return
     */
    public List<String> generateParenthesis(int n) {
        if (n <= 0) {
            return new ArrayList<>();
        }
        List<String> result = new ArrayList<>();
        intervalGenerate(result, 0, 0, "", n);
        return result;
    }

    private void intervalGenerate(List<String> result, int open, int close, String str, int n) {
        if (str.length() == 2 * n) {
            result.add(str);
            return;
        }
        if (open < n) {
            intervalGenerate(result, open + 1, close, str + "(", n);
        }
        if (close < open) {
            intervalGenerate(result, open, close + 1, str + ")", n);
        }
    }


    /**
     * 23. Merge k Sorted Lists
     *
     * @param lists
     * @return
     */
    public ListNode mergeKLists(ListNode[] lists) {
        if (lists == null || lists.length == 0) {
            return null;
        }
        PriorityQueue<ListNode> priorityQueue = new PriorityQueue<>(lists.length, Comparator.comparingInt(o -> o.val));

        for (ListNode list : lists) {
            priorityQueue.offer(list);
        }
        ListNode root = new ListNode(0);
        ListNode dummy = root;

        while (!priorityQueue.isEmpty()) {
            ListNode poll = priorityQueue.poll();
            dummy.next = poll;

            dummy = dummy.next;

            if (poll.next != null) {
                priorityQueue.offer(poll.next);
            }
        }
        return root.next;
    }


    /**
     * 24. Swap Nodes in Pairs
     *
     * @param head
     * @return
     */
    public ListNode swapPairs(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        return null;
    }


}
