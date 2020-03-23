package org.dora.algorithm.leetcode.v3;

import org.dora.algorithm.datastructe.ListNode;
import org.dora.algorithm.datastructe.TreeNode;

import java.util.*;

/**
 * @author liulu12@xiaomi.com
 * @date 2020/2/22
 */
public class LeetCodeP0 {

    public static void main(String[] args) {
        LeetCodeP0 codeV3 = new LeetCodeP0();
        String s = "25525511135";
        codeV3.restoreIpAddresses(s);
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
        ListNode root = new ListNode(0);
        root.next = head;
        ListNode dummy = root;

        while (dummy.next != null && dummy.next.next != null) {

            ListNode slowNode = dummy.next;

            ListNode fastNode = dummy.next.next;

            dummy.next = fastNode;

            slowNode.next = fastNode.next;

            fastNode.next = slowNode;

            dummy = dummy.next.next;
        }
        return root.next;
    }

    /**
     * 25. Reverse Nodes in k-Group
     *
     * @param head
     * @param k
     * @return
     */
    public ListNode reverseKGroup(ListNode head, int k) {
        if (head == null || head.next == null) {
            return head;
        }
        int count = 0;
        ListNode current = head;
        while (current != null && count != k) {
            current = current.next;
            count++;
        }
        if (count == k) {
            ListNode reverseNode = reverseKGroup(current, k);
            while (count-- > 0) {
                ListNode tmp = head.next;
                head.next = reverseNode;
                reverseNode = head;

                head = tmp;
            }
            head = reverseNode;
        }
        return head;
    }

    public ListNode reverseKGroupV2(ListNode head, int k) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode fast = head;
        for (int i = 0; i < k; i++) {
            if (fast == null) {
                return head;
            }
            fast = fast.next;
        }
        ListNode newHead = reverseListNode(head, fast);

        head.next = reverseKGroup(fast, k);

        return newHead;
    }

    private ListNode reverseListNode(ListNode start, ListNode end) {
        ListNode prev = end;
        while (start != end) {
            ListNode tmp = start.next;

            start.next = prev;

            prev = start;

            start = tmp;
        }
        return prev;
    }

    /**
     * 26. Remove Duplicates from Sorted Array
     *
     * @param nums
     * @return
     */
    public int removeDuplicates(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int index = 1;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] != nums[i - 1]) {
                nums[index++] = nums[i];
            }
        }
        return index;
    }

    /**
     * 27. Remove Element
     *
     * @param nums
     * @param val
     * @return
     */
    public int removeElement(int[] nums, int val) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int index = 0;
        for (int i = 0; i < nums.length; i++) {

            if (nums[i] != val) {
                nums[index++] = nums[i];
            }
        }
        return index;
    }

    /**
     * kmp
     * 28. Implement strStr()
     *
     * @param haystack
     * @param needle
     * @return
     */
    public int strStr(String haystack, String needle) {
        if (haystack == null || needle == null) {
            return -1;
        }
        return Integer.MAX_VALUE;
    }

    /**
     * 29. Divide Two Integers
     *
     * @param dividend
     * @param divisor
     * @return
     */
    public int divide(int dividend, int divisor) {
        if (dividend == Integer.MIN_VALUE && divisor == -1) {
            return Integer.MAX_VALUE;
        }
        if (dividend == Integer.MAX_VALUE && divisor == 1) {
            return dividend;
        }
        if (dividend == 0) {
            return 0;
        }
        int sign = -1;
        if (dividend > 0 && divisor > 0) {
            sign = 1;
        }
        if (dividend < 0 && divisor < 0) {
            sign = 1;
        }
        long absDividend = Math.abs(Long.valueOf(dividend));
        long absDivisor = Math.abs(Long.valueOf(divisor));
        int result = 0;
        while (absDividend >= absDivisor) {
            long tmp = absDivisor;
            int num = 1;
            while (absDividend >= (tmp <<= 1)) {
                tmp <<= 1;
                num <<= 1;
            }
            absDividend -= tmp;
            result += num;
        }
        return result * sign;
    }

    /**
     * 31. Next Permutation
     *
     * @param nums
     */
    public void nextPermutation(int[] nums) {
        if (nums == null || nums.length == 0) {
            return;
        }

        int index = nums.length - 1;
        while (index > 0) {
            if (nums[index] > nums[index - 1]) {
                break;
            }
            index--;
        }
        if (index == 0) {
            reverseArrays(nums, 0, nums.length - 1);
        } else {
            int preValue = nums[index - 1];
            int j = nums.length - 1;
            while (j > (index - 1)) {
                if (nums[j] > preValue) {
                    swapValue(nums, index - 1, j);
                    break;
                }
                j--;
            }
            reverseArrays(nums, index, nums.length - 1);
        }

    }

    public void swapValue(int[] nums, int i, int j) {
        int value = nums[i];
        nums[i] = nums[j];
        nums[j] = value;
    }

    public void reverseArrays(int[] nums, int start, int end) {
        if (start > end) {
            return;
        }
        for (int i = start; i <= (start + end) / 2; i++) {
            int value = nums[i];
            nums[i] = nums[start + end - i];
            nums[start + end - i] = value;
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
        int left = -1;
        for (int i = 0; i < s.length(); i++) {
            char character = s.charAt(i);
            if (character == '(') {
                stack.push(i);
            } else {
                if (stack.isEmpty()) {
                    left = i;
                } else {
                    stack.pop();
                }
                result = Math.max(result, stack.isEmpty() ? i - left : i - 1 - stack.peek());
            }
        }
        return result;
    }

    public int longestValidParenthesesV2(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        }
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(' || stack.isEmpty()) {
                stack.push(i);
            } else if (s.charAt(stack.peek()) == '(') {
                stack.pop();
            } else {
                stack.push(i);
            }
        }
        if (stack.isEmpty()) {
            return s.length();
        }

        int result = 0;
        int longest = s.length();
        while (!stack.isEmpty()) {
            Integer peek = stack.pop();
            result = Math.max(result, longest - 1 - peek);
            longest = peek;
        }
        result = Math.max(result, longest);
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

        int left = 0;
        int right = nums.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                return mid;
            }
            if (nums[left] <= nums[mid]) {
                if (target < nums[mid] && nums[left] <= target) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            } else {
                if (target > nums[mid] && target <= nums[right]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }
        return nums[left] == target ? left : -1;
    }

    /**
     * 34. Find First and Last Position of Element in Sorted Array
     *
     * @param nums
     * @param target
     * @return
     */
    public int[] searchRange(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return new int[]{-1, -1};
        }
        int[] ans = new int[]{-1, -1};
        int left = 0;
        int right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] > target) {
                right = mid - 1;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else {
                int firstIndex = mid;
                int secondIndex = mid;
                while (firstIndex > 0 && nums[firstIndex] == nums[firstIndex - 1]) {
                    firstIndex--;
                }
                while (secondIndex < nums.length - 1 && nums[secondIndex] == nums[secondIndex + 1]) {
                    secondIndex++;
                }
                ans[0] = firstIndex;
                ans[1] = secondIndex;
                return ans;
            }
        }
        return ans;
    }

    public int[] searchRangeV2(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return new int[]{-1, -1};
        }
        int[] ans = new int[]{-1, -1};
        int left = 0;
        int right = nums.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        if (nums[left] != target) {
            return ans;
        }
        ans[0] = left;

        right = nums.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2 + 1;
            if (nums[mid] > target) {
                right = mid - 1;
            } else {
                left = mid;
            }
        }
        ans[1] = left;
        return ans;
    }

    /**
     * 35. Search Insert Position
     *
     * @param nums
     * @param target
     * @return
     */
    public int searchInsert(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int left = 0;
        int right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {

                return mid;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return left;
    }

    /**
     * 39. Combination Sum
     *
     * @param candidates
     * @param target
     * @return
     */
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        if (candidates == null || candidates.length == 0) {
            return new ArrayList<>();
        }
        List<List<Integer>> ans = new ArrayList<>();
        Arrays.sort(candidates);
        intervalCombinationSum(ans, new ArrayList<>(), 0, candidates, target);
        return ans;
    }

    private void intervalCombinationSum(List<List<Integer>> ans, List<Integer> tmp, int start, int[] candidates, int target) {
        if (target == 0) {
            ans.add(new ArrayList<>(tmp));
            return;
        }
        for (int i = start; i < candidates.length && candidates[i] <= target; i++) {

            tmp.add(candidates[i]);

            intervalCombinationSum(ans, tmp, i, candidates, target - candidates[i]);

            tmp.remove(tmp.size() - 1);
        }
    }

    /**
     * 40. Combination Sum II
     *
     * @param candidates
     * @param target
     * @return
     */
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        if (candidates == null || candidates.length == 0) {
            return new ArrayList<>();
        }
        List<List<Integer>> ans = new ArrayList<>();
        Arrays.sort(candidates);
        intervalCombinationSumV2(ans, new ArrayList<Integer>(), 0, target, candidates);
        return ans;
    }

    private void intervalCombinationSumV2(List<List<Integer>> ans, ArrayList<Integer> integers, int start, int target, int[] candidates) {
        if (target == 0) {
            ans.add(new ArrayList<>(integers));
            return;
        }
        for (int i = start; i < candidates.length && candidates[i] <= target; i++) {
            if (i > start && candidates[i] == candidates[i - 1]) {
                continue;
            }
            integers.add(candidates[i]);
            intervalCombinationSumV2(ans, integers, i + 1, target - candidates[i], candidates);
            integers.remove(integers.size() - 1);
        }
    }

    /**
     * 41. First Missing Positive
     *
     * @param nums
     * @return
     */
    public int firstMissingPositive(int[] nums) {
        for (int i = 0; i < nums.length; i++) {
            while (nums[i] >= 0 && nums[i] <= nums.length && nums[i] != nums[nums[i] - 1]) {
                swapValue(nums, i, nums[i] - 1);
            }
        }
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != i + 1) {
                return i + 1;
            }
        }
        return nums.length + 1;
    }

    /**
     * 42. Trapping Rain Water
     *
     * @param height
     * @return
     */
    public int trap(int[] height) {
        if (height == null || height.length == 0) {
            return 0;
        }
        int result = 0;
        int left = 0;
        int right = height.length - 1;
        int leftValue = 0;
        int rightValue = 0;
        while (left < right) {
            if (height[left] <= height[right]) {
                if (height[left] >= leftValue) {
                    leftValue = height[left];
                } else {
                    result += leftValue - height[left];
                }
                left++;
            } else {
                if (height[right] >= rightValue) {
                    rightValue = height[right];
                } else {
                    result += rightValue - height[right];
                }
                right--;
            }
        }
        return result;
    }

    public int trapV2(int[] height) {
        if (height == null || height.length == 0) {
            return 0;
        }
        int result = 0;
        int left = 0;
        int right = height.length - 1;
        while (left < right) {
            while (left < right && height[left] == 0) {
                left++;
            }
            while (left < right && height[right] == 0) {
                right--;
            }
            int minValue = Math.min(height[left], height[right]);
            for (int i = left; i <= right; i++) {
                int value = height[i];

                if (value >= minValue) {
                    height[i] = value - minValue;
                } else {
                    result += minValue - height[i];
                    height[i] = 0;
                }
            }
        }
        return result;
    }

    /**
     * 43. Multiply Strings
     *
     * @param num1
     * @param num2
     * @return
     */
    public String multiply(String num1, String num2) {
        if (num1 == null || num2 == null || num1.isEmpty() || num2.isEmpty()) {
            return "";
        }
        int m = num1.length();
        int n = num2.length();
        int[] position = new int[m + n];
        for (int i = m - 1; i >= 0; i--) {
            for (int j = n - 1; j >= 0; j--) {
                int value = (Character.getNumericValue(num1.charAt(i))) * (Character.getNumericValue(num2.charAt(j))) + position[i + j + 1];

                position[i + j + 1] = value % 10;

                position[i + j] += value / 10;
            }
        }
        StringBuilder builder = new StringBuilder();
        for (int number : position) {
            if (!(number == 0 && builder.length() == 0)) {
                builder.append(number);
            }
        }
        return builder.length() == 0 ? "0" : builder.toString();
    }

    /**
     * 44. Wildcard Matching
     *
     * @param s
     * @param p
     * @return
     */
    public boolean isMatchV2(String s, String p) {
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
            dp[0][j] = p.charAt(j - 1) == '*' && dp[0][j - 1];
        }
        for (int i = 1; i < dp.length; i++) {
            for (int j = 1; j < dp[i].length; j++) {
                if (p.charAt(j - 1) == s.charAt(i - 1) || p.charAt(j - 1) == '?') {
                    dp[i][j] = dp[i - 1][j - 1];
                } else if (p.charAt(j - 1) == '*') {
                    dp[i][j] = dp[i - 1][j] || dp[i][j - 1];
                }
            }
        }
        return dp[m][n];
    }

    /**
     * 45. Jump Game II
     *
     * @param nums
     * @return
     */
    public int jump(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int step = 0;
        int furthest = 0;
        int current = 0;
        for (int i = 0; i < nums.length - 1; i++) {
            furthest = Math.max(i + nums[i], furthest);
            if (i == current) {
                step++;
                current = furthest;
            }

        }
        return step;
    }

    /**
     * 46. Permutations
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> permute(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new ArrayList<>();
        }
        List<List<Integer>> ans = new ArrayList<>();
        boolean[] used = new boolean[nums.length];
        intervalPermute(ans, new ArrayList<Integer>(), used, nums);
        return ans;
    }

    private void intervalPermute(List<List<Integer>> ans, List<Integer> integers, boolean[] used, int[] nums) {
        if (integers.size() == nums.length) {
            ans.add(new ArrayList<>(integers));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (used[i]) {
                continue;
            }
            used[i] = true;
            integers.add(nums[i]);
            intervalPermute(ans, integers, used, nums);
            used[i] = false;
            integers.remove(integers.size() - 1);
        }
    }

    /**
     * 47. Permutations II
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> permuteUnique(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new ArrayList<>();
        }
        Arrays.sort(nums);
        List<List<Integer>> ans = new ArrayList<>();
        boolean[] used = new boolean[nums.length];
        interPermuteUnique(ans, new ArrayList<Integer>(), used, nums);
        return ans;
    }

    private void interPermuteUnique(List<List<Integer>> ans, ArrayList<Integer> integers, boolean[] used, int[] nums) {
        if (integers.size() == nums.length) {
            ans.add(new ArrayList<>(integers));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (i > 0 && nums[i] == nums[i - 1] && !used[i - 1]) {
                continue;
            }
            if (used[i]) {
                continue;
            }
            used[i] = true;
            integers.add(nums[i]);
            interPermuteUnique(ans, integers, used, nums);
            used[i] = false;
            integers.remove(integers.size() - 1);
        }

    }

    /**
     * todo
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> permuteUniqueV2(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new ArrayList<>();
        }
        Arrays.sort(nums);
        List<List<Integer>> ans = new ArrayList<>();
        intervalPermuteUniqueV2(ans, 0, nums);
        return ans;
    }

    private void intervalPermuteUniqueV2(List<List<Integer>> ans, int start, int[] nums) {
        if (start == nums.length - 1) {
            List<Integer> tmp = new ArrayList<>();
            for (int num : nums) {
                tmp.add(num);
            }
            ans.add(tmp);
            return;
        }
        for (int i = start; i < nums.length; i++) {
            if (i != start && nums[i] == nums[start]) {
                continue;
            }
            swapValue(nums, i, start);
            intervalPermuteUniqueV2(ans, start + 1, nums);
            swapValue(nums, i, start);
        }
    }

    /**
     * 48. Rotate Image
     *
     * @param matrix
     */
    public void rotate(int[][] matrix) {
        if (matrix == null || matrix.length == 0) {
            return;
        }
        int row = matrix.length;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < i; j++) {
                swapMatrix(matrix, i, j);
            }
        }
        for (int i = 0; i < row; i++) {
            reverseArrays(matrix[i], 0, matrix[i].length - 1);
        }
    }

    public void swapMatrix(int[][] matrix, int i, int j) {
        int value = matrix[i][j];
        matrix[i][j] = matrix[j][i];

        matrix[j][i] = value;
    }

    /**
     * 49. Group Anagrams
     *
     * @param strs
     * @return
     */
    public List<List<String>> groupAnagrams(String[] strs) {
        if (strs == null || strs.length == 0) {
            return new ArrayList<>();
        }
        List<List<Integer>> ans = new ArrayList<>();
        Map<String, List<String>> map = new HashMap<>();
        for (String str : strs) {
            char[] chars = str.toCharArray();
            Arrays.sort(chars);
            String key = String.valueOf(chars);
            List<String> tmp = map.getOrDefault(key, new ArrayList<>());
            tmp.add(str);
            map.put(key, tmp);
        }
        return new ArrayList<>(map.values());
    }

    /**
     * 50. Pow(x, n)
     *
     * @param x
     * @param n
     * @return
     */
    public double myPow(double x, int n) {
        if (n == 0) {
            return 1;
        }
        if (n == 1) {
            return x;
        }
        if (n < 0) {
            x = 1 / x;
            n = -n;
        }
        if (x > Integer.MAX_VALUE || x < Integer.MIN_VALUE) {
            return 0;
        }
        return n % 2 == 0 ? myPow(x * x, n / 2) : x * myPow(x * x, n / 2);
    }

    public double myPowV2(double x, int n) {
        double ans = 1.0;
        long abs = Math.abs((long) n);
        while (abs != 0) {
            if ((abs % 2) == 1) {
                ans *= x;
            }
            x *= x;
            abs >>= 1;
        }
        return n < 0 ? 1 / ans : ans;
    }

    /**
     * 51. N-Queens
     *
     * @param n
     * @return
     */
    public List<List<String>> solveNQueens(int n) {
        if (n <= 0) {
            return new ArrayList<>();
        }
        char[][] nQueens = new char[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                nQueens[i][j] = '.';
            }
        }
        List<List<String>> result = new ArrayList<>();

        intervalSolveNQueens(result, 0, nQueens);
        return result;
    }

    private void intervalSolveNQueens(List<List<String>> result, int row, char[][] nQueens) {
        if (row == nQueens.length) {
            List<String> tmp = new ArrayList<>();
            for (char[] word : nQueens) {
                tmp.add(String.valueOf(word));
            }
            result.add(tmp);
            return;
        }
        for (int j = 0; j < nQueens[row].length; j++) {
            if (intervalCheckNQueens(nQueens, j, row)) {
                nQueens[row][j] = 'Q';
                intervalSolveNQueens(result, row + 1, nQueens);
                nQueens[row][j] = '.';
            }
        }
    }

    ;

    private boolean intervalCheckNQueens(char[][] nQueens, int column, int row) {
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
        for (int i = row - 1, j = column + 1; i >= 0 && j < nQueens[row].length; i--, j++) {
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
        Arrays.fill(dp, -1);
        return intervalTotalNQueens(dp, 0, n);
    }


    private int intervalTotalNQueens(int[] dp, int row, int n) {
        int count = 0;
        if (row == n) {
            count++;
            return count;
        }
        for (int j = 0; j < n; j++) {
            if (checkIntervalTotalNQueens(dp, j, row, n)) {
                dp[row] = j;
                count += intervalTotalNQueens(dp, row + 1, n);
                dp[row] = -1;
            }
        }
        return count;
    }

    private boolean checkIntervalTotalNQueens(int[] dp, int column, int row, int n) {
        for (int i = row - 1; i >= 0; i--) {
            if (dp[i] == column || Math.abs(i - row) == Math.abs(dp[i] - column)) {

                return false;
            }
        }
        return true;
    }


    /**
     * 53. Maximum Subarray
     *
     * @param nums
     * @return
     */
    public int maxSubArray(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int local = Integer.MIN_VALUE;
        int result = Integer.MIN_VALUE;
        for (int num : nums) {
            local = local < 0 ? num : local + num;
            result = Math.max(local, result);
        }
        return result;
    }


    /**
     * 54. Spiral Matrix
     *
     * @param matrix
     * @return
     */
    public List<Integer> spiralOrder(int[][] matrix) {
        if (matrix == null || matrix.length == 0) {
            return new ArrayList<>();
        }
        List<Integer> result = new ArrayList<>();
        int top = 0;
        int bottom = matrix.length - 1;
        int left = 0;
        int right = matrix[0].length - 1;
        while (left <= right && top <= bottom) {
            for (int i = left; i <= right; i++) {
                result.add(matrix[top][i]);
            }
            for (int i = top + 1; i <= bottom; i++) {
                result.add(matrix[i][right]);
            }
            if (top != bottom) {
                for (int i = right - 1; i >= left; i--) {
                    result.add(matrix[bottom][i]);
                }
            }
            if (left != right) {
                for (int i = bottom - 1; i > left; i--) {
                    result.add(matrix[i][left]);
                }
            }
            left++;
            bottom--;
            top++;
            right--;
        }
        return result;
    }

    /**
     * 55. Jump Game
     *
     * @param nums
     * @return
     */
    public boolean canJump(int[] nums) {
        if (nums == null || nums.length == 0) {
            return false;
        }
        int reach = 0;
        for (int i = 0; i < nums.length - 1 && i <= reach; i++) {
            reach = Math.max(reach, i + nums[i]);
        }
        return reach >= nums.length - 1;
    }


    /**
     * 56. Merge Intervals
     *
     * @param intervals
     * @return
     */
    public int[][] merge(int[][] intervals) {
        if (intervals == null || intervals.length == 0) {
            return new int[][]{};
        }
        Arrays.sort(intervals, Comparator.comparingInt(o -> o[0]));

        LinkedList<int[]> result = new LinkedList<>();

        for (int i = 0; i < intervals.length; i++) {
            if (result.isEmpty() || result.getLast()[1] < intervals[i][0]) {
                result.offer(intervals[i]);
            } else {
                int[] last = result.peekLast();
                last[1] = Math.max(last[1], intervals[i][1]);
            }
        }
        return result.toArray(new int[result.size()][]);
    }


    public int[][] mergeV2(int[][] intervals) {
        if (intervals == null || intervals.length == 0) {
            return new int[][]{};
        }
        PriorityQueue<int[]> queue = new PriorityQueue<>(intervals.length, Comparator.comparingInt(o -> o[0]));

        for (int[] interval : intervals) {
            queue.offer(interval);
        }
        LinkedList<int[]> result = new LinkedList<>();
        while (!queue.isEmpty()) {
            int[] poll = queue.poll();
            int[] last = result.peekLast();
            if (result.isEmpty() || last[1] < poll[0]) {
                result.offer(poll);
            } else {
                last[1] = Math.max(last[1], poll[1]);
            }
        }
        return result.toArray(new int[result.size()][]);

    }


    /**
     * 57. Insert Interval
     *
     * @param intervals
     * @param newInterval
     * @return
     */
    public int[][] insert(int[][] intervals, int[] newInterval) {
        if (intervals == null || newInterval == null) {
            return new int[][]{};
        }

        List<int[]> result = new ArrayList<>();

        int index = 0;
        while (index < intervals.length && intervals[index][1] < newInterval[0]) {
            result.add(intervals[index++]);
        }
        while (index < intervals.length && intervals[index][0] <= newInterval[1]) {
            int[] poll = intervals[index++];
            newInterval[0] = Math.min(newInterval[0], poll[0]);
            newInterval[1] = Math.max(newInterval[1], poll[1]);
        }
        result.add(newInterval);

        while (index < intervals.length) {
            result.add(intervals[index++]);
        }
        return result.toArray(new int[result.size()][]);
    }


    /**
     * 58. Length of Last Word
     *
     * @param s
     * @return
     */
    public int lengthOfLastWord(String s) {
        if (s == null) {
            return 0;
        }
        s = s.trim();

        if (s.isEmpty()) {
            return 0;
        }
        return s.length() - 1 - s.lastIndexOf(" ");
    }


    /**
     * 59. Spiral Matrix II
     *
     * @param n
     * @return
     */
    public int[][] generateMatrix(int n) {
        if (n <= 0) {
            return new int[][]{};
        }
        int[][] matrix = new int[n][n];
        int left = 0;
        int right = n - 1;
        int top = 0;
        int bottom = n - 1;
        int result = 0;
        while (left <= right && top <= bottom) {
            for (int i = left; i <= right; i++) {
                matrix[top][i] = ++result;
            }
            for (int i = top + 1; i <= bottom; i++) {
                matrix[i][right] = ++result;
            }
            if (top != bottom) {
                for (int i = right - 1; i >= left; i--) {
                    matrix[bottom][i] = ++result;
                }
            }
            if (left != right) {
                for (int i = bottom - 1; i > top; i--) {
                    matrix[i][left] = ++result;
                }
            }
            left++;
            right--;
            top++;
            bottom--;
        }
        return matrix;
    }


    /**
     * todo
     * 60. Permutation Sequence
     *
     * @param n
     * @param k
     * @return
     */
    public String getPermutation(int n, int k) {
        if (n <= 0 || k <= 0) {
            return "";
        }
        List<Integer> ans = new ArrayList<>();
        for (int i = 1; i <= n; i++) {
            ans.add(i);
        }
        int factory = 1;
        int[] nums = new int[n + 1];

        nums[0] = 1;
        for (int i = 1; i <= n; i++) {
            nums[i] = factory;
            factory *= i;
        }
        k--;
        for (int i = 0; i < n; i++) {
        }

        return null;
    }

    /**
     * 61. Rotate List
     *
     * @param head
     * @param k
     * @return
     */
    public ListNode rotateRight(ListNode head, int k) {
        if (head == null || head.next == null) {
            return head;
        }
        int count = 1;
        ListNode fast = head;
        while (fast.next != null) {
            count++;
            fast = fast.next;
        }
        k %= count;
        fast.next = head;

        ListNode slow = head;
        if (k != 0) {
            for (int i = 0; i < count - k; i++) {
                fast = fast.next;
                slow = slow.next;
            }
        }
        fast.next = null;
        return slow;
    }

    /**
     * 62. Unique Paths
     *
     * @param m
     * @param n
     * @return
     */
    public int uniquePaths(int m, int n) {
        if (m <= 0 || n <= 0) {
            return 0;
        }
        int[][] dp = new int[m][n];
        for (int i = 0; i < dp.length; i++) {
            for (int j = 0; j < dp[i].length; j++) {
                if (i == 0 && j == 0) {
                    dp[0][0] = 1;
                } else if (i == 0) {
                    dp[i][j] += dp[i][j - 1];
                } else if (j == 0) {
                    dp[i][j] += dp[i - 1][j];
                } else {
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
                }
            }
        }
        return dp[m - 1][n - 1];
    }

    public int uniquePathsV2(int m, int n) {
        if (m <= 0 || n <= 0) {
            return 0;
        }
        int[] dp = new int[n];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {

                if (i == 0 && j == 0) {
                    dp[j] = 1;
                } else {
                    dp[j] = dp[j] + (j > 0 ? dp[j - 1] : 0);
                }

            }
        }
        return dp[n - 1];

    }

    /**
     * 63. Unique Paths II
     *
     * @param obstacleGrid
     * @return
     */
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        if (obstacleGrid == null || obstacleGrid.length == 0) {
            return 0;
        }
        int m = obstacleGrid.length;
        int n = obstacleGrid[0].length;
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (obstacleGrid[i][j] == 1) {
                    dp[i][j] = 0;
                    continue;
                }
                if (i == 0 && j == 0) {
                    dp[i][j] = 1;
                } else if (i == 0) {
                    dp[i][j] += dp[i][j - 1];
                } else if (j == 0) {
                    dp[i][j] += dp[i - 1][j];
                } else {
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
                }
            }
        }
        return dp[m - 1][n - 1];

    }


    public int uniquePathsWithObstaclesV2(int[][] obstacleGrid) {
        if (obstacleGrid == null || obstacleGrid.length == 0) {
            return 0;
        }
        int m = obstacleGrid.length;
        int n = obstacleGrid[0].length;
        int[] dp = new int[n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (obstacleGrid[i][j] == 1) {
                    dp[j] = 0;
                } else if (i == 0 && j == 0) {
                    dp[j] = 1;
                } else {
                    dp[j] = dp[j] + (j > 0 ? dp[j - 1] : 0);
                }
            }
        }
        return dp[n - 1];
    }


    /**
     * 64. Minimum Path Sum
     *
     * @param grid
     * @return
     */
    public int minPathSum(int[][] grid) {
        if (grid == null || grid.length == 0) {
            return 0;
        }
        int m = grid.length;
        int n = grid[0].length;
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (i == 0 && j == 0) {
                    dp[i][j] = grid[0][0];
                } else if (i == 0) {
                    dp[i][j] = dp[i][j - 1] + grid[i][j];
                } else if (j == 0) {
                    dp[i][j] = dp[i - 1][j] + grid[i][j];
                } else {
                    dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
                }
            }
        }
        return dp[m - 1][n - 1];
    }


    public int minPathSumV2(int[][] grid) {
        if (grid == null || grid.length == 0) {
            return 0;
        }
        int m = grid.length;
        int n = grid[0].length;
        int[] dp = new int[n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (i == 0 && j == 0) {
                    dp[j] = grid[0][0];
                } else if (i == 0) {
                    dp[j] = dp[j - 1] + grid[0][j];
                } else if (j == 0) {
                    dp[j] = dp[j] + grid[i][j];
                } else {
                    dp[j] = Math.min(dp[j], dp[j - 1]) + grid[i][j];
                }
            }
        }

        return dp[n - 1];
    }


    /**
     * 65. Valid Number
     *
     * @param s
     * @return
     */
    public boolean isNumber(String s) {
        if (s == null) {
            return false;
        }
        s = s.trim();
        if (s.isEmpty()) {
            return false;
        }
        boolean numberAfterE = true;
        boolean seenNumber = false;
        boolean seenDit = false;
        boolean seenE = false;

        int len = s.length();
        for (int i = 0; i < len; i++) {
            char word = s.charAt(i);
            if (word >= '0' && word <= '9') {
                seenNumber = true;
                numberAfterE = true;
            } else if (word == 'e' || word == 'E') {

                if (i == 0 || seenE || !seenNumber) {
                    return false;
                }
                seenE = true;
                numberAfterE = false;
            } else if (word == '.') {
                if (seenDit) {
                    return false;
                }
                if (seenE) {
                    return false;
                }
                seenDit = true;
            } else if (word == '-' || word == '+') {
                if (i != 0 && s.charAt(i - 1) != 'e' && s.charAt(i - 1) != 'E') {
                    return false;
                }
            } else {
                return false;
            }
        }
        return numberAfterE && seenNumber;
    }


    /**
     * 66. Plus One
     *
     * @param digits
     * @return
     */
    public int[] plusOne(int[] digits) {
        if (digits == null || digits.length == 0) {
            return new int[]{};
        }
        for (int i = digits.length - 1; i >= 0; i--) {
            if (digits[i] == 9) {
                digits[i] = 0;
            } else {
                digits[i]++;
                return digits;
            }
        }
        int[] ans = new int[digits.length + 1];
        ans[0] = 1;
        return ans;
    }

    /**
     * 67. Add Binary
     *
     * @param a
     * @param b
     * @return
     */
    public String addBinary(String a, String b) {
        if (a == null || b == null) {
            return "";
        }
        StringBuilder builder = new StringBuilder();
        int m = a.length();
        int n = b.length();
        m--;
        n--;
        int carry = 0;
        while (m >= 0 || n >= 0 || carry != 0) {
            int value = (m >= 0 ? Character.getNumericValue(a.charAt(m--)) : 0) + (n >= 0 ? Character.getNumericValue(b.charAt(n--)) : 0) + carry;
            builder.append(value % 2);
            carry = value / 2;
        }
        String result = builder.reverse().toString();
        return result.isEmpty() ? "0" : result;

    }

    /**
     * 68. Text Justification
     *
     * @param words
     * @param maxWidth
     * @return
     */
    public List<String> fullJustify(String[] words, int maxWidth) {
        if (words == null || words.length == 0 || maxWidth == 0) {
            return new ArrayList<>();
        }
        List<String> ans = new ArrayList<>();

        int startIndex = 0;

        while (startIndex < words.length) {

            int endIndex = startIndex;

            int line = 0;

            while (endIndex < words.length && line + words[endIndex].length() <= maxWidth) {

                line += words[endIndex].length() + 1;

                endIndex++;
            }

            boolean lastRow = endIndex == words.length;

            int countOfWord = endIndex - startIndex;

            int blankLine = maxWidth - line + 1;

            StringBuilder builder = new StringBuilder();
            if (countOfWord == 1) {
                builder.append(words[startIndex]);
            } else {
                int blankWord = lastRow ? 1 : 1 + blankLine / (countOfWord - 1);
                int extraWord = lastRow ? 0 : blankLine % (countOfWord - 1);
                builder.append(construct(startIndex, endIndex, words, blankWord, extraWord));
            }
            String tmp = justify(builder, maxWidth);

            startIndex = endIndex;
            ans.add(tmp);
        }
        return ans;
    }

    private String justify(StringBuilder builder, int maxWidth) {
        String result = builder.toString();
        while (result.length() < maxWidth) {
            result = result + " ";
        }
        while (result.length() > maxWidth) {
            result = result.substring(0, result.length() - 1);
        }
        return result;
    }

    private String construct(int startIndex, int endIndex, String[] words, int blankWord, int extraWord) {
        StringBuilder builder = new StringBuilder();

        for (int i = startIndex; i < endIndex; i++) {
            builder.append(words[i]);
            for (int tmp = 0; tmp < blankWord; tmp++) {
                builder.append(" ");
            }
            if (extraWord-- > 0) {
                builder.append(" ");
            }
        }
        return builder.toString();
    }

    public List<String> fullJustifyV2(String[] words, int maxWidth) {
        if (words == null || words.length == 0 || maxWidth <= 0) {
            return new ArrayList<>();
        }
        List<String> result = new ArrayList<>();
        int startIndex = 0;
        while (startIndex < words.length) {
            int line = 0;
            int endIndex = startIndex;
            while (endIndex < words.length && line + words[endIndex].length() <= maxWidth) {
                line += words[endIndex].length() + 1;
                endIndex++;
            }
            boolean lastRow = endIndex == words.length;
            int countOfWord = endIndex - startIndex;
            StringBuilder builder = new StringBuilder();
            if (lastRow) {
                builder.append(words[startIndex]);
            } else {
                if (countOfWord != 0) {
                    line = line - countOfWord + 1;
                }
                for (int i = 0; i < countOfWord; i++) {
                }
            }

            startIndex = endIndex;
        }

        return result;
    }


    /**
     * 69. Sqrt(x)
     *
     * @param x
     * @return
     */
    public int mySqrt(int x) {
        double precision = 0.000001;

        double result = x;
        while (result * result - x > precision) {
            result = (result + x / result) / 2;
        }
        return (int) result;
    }

    /**
     * 70. Climbing Stairs
     *
     * @param n
     * @return
     */
    public int climbStairs(int n) {
        if (n <= 0) {
            return 0;
        }
        if (n <= 1) {
            return 1;
        }
        if (n <= 2) {
            return 2;
        }
        int sum1 = 1;
        int sum2 = 2;
        int sum = 0;
        for (int i = 3; i <= n; i++) {
            sum = sum1 + sum2;
            sum1 = sum2;
            sum2 = sum;
        }
        return sum;
    }


    /**
     * 71. Simplify Path
     *
     * @param path
     * @return
     */
    public String simplifyPath(String path) {
        if (path == null || path.isEmpty()) {
            return "/";
        }
        String[] words = path.split("/");
        Deque<String> deque = new LinkedList<>();
        for (String word : words) {
            if (word.isEmpty() || ".".equals(word)) {
                continue;
            }
            if (!"..".equals(word)) {
                deque.offer(word);
            } else if (!deque.isEmpty()) {
                deque.pollLast();
            }
        }
        if (deque.isEmpty()) {
            return "/";
        }
        StringBuilder result = new StringBuilder();

        for (String word : deque) {
            result.append("/").append(word);
        }
        return result.toString();

    }


    /**
     * 72. Edit Distance
     *
     * @param word1
     * @param word2
     * @return
     */
    public int minDistance(String word1, String word2) {
        if (word1 == null || word2 == null) {
            return 0;
        }
        int m = word1.length();
        int n = word2.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++) {
            dp[i][0] = i;
        }
        for (int j = 1; j <= n; j++) {
            dp[0][j] = j;
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = Math.min(dp[i - 1][j], Math.min(dp[i][j - 1], dp[i - 1][j - 1])) + 1;
                }
            }
        }
        return dp[m][n];
    }


    /**
     * 73. Set Matrix Zeroes
     *
     * @param matrix
     */
    public void setZeroes(int[][] matrix) {
        if (matrix == null || matrix.length == 0) {
            return;
        }
        boolean fr = false;
        boolean fc = false;
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                if (matrix[i][j] == 0) {
                    matrix[i][0] = 0;
                    matrix[0][j] = 0;
                    if (i == 0) {
                        fr = true;
                    }
                    if (j == 0) {
                        fc = true;
                    }
                }
            }
        }

        for (int i = 1; i < matrix.length; i++) {
            for (int j = 1; j < matrix[i].length; j++) {
                if (matrix[i][0] == 0 || matrix[0][j] == 0) {
                    matrix[i][j] = 0;
                }
            }
        }
        if (fr) {
            Arrays.fill(matrix[0], 0);
        }
        if (fc) {
            for (int i = 0; i < matrix.length; i++) {
                matrix[i][0] = 0;
            }
        }

    }


    /**
     * 74. Search a 2D Matrix
     *
     * @param matrix
     * @param target
     * @return
     */
    public boolean searchMatrix(int[][] matrix, int target) {
        if (matrix == null || matrix.length == 0) {
            return false;
        }
        int row = matrix.length;
        int column = matrix[0].length;
        int i = row - 1;
        int j = 0;
        while (i >= 0 && j < column) {
            int value = matrix[i][j];
            if (value == target) {
                return true;
            } else if (value < target) {
                j++;
            } else {
                i--;
            }
        }
        return false;
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
        int redNum = 0;
        int blueNum = nums.length - 1;
        for (int i = 0; i < nums.length; i++) {
            while (nums[i] == 2 && i < blueNum) {
                swapValue(nums, i, blueNum--);
            }
            while (nums[i] == 0 && redNum < i) {
                swapValue(nums, i, redNum++);

            }
        }
    }


    /**
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
        int[] hash = new int[256];
        int m = s.length();
        int count = t.length();
        for (int i = 0; i < count; i++) {
            hash[t.charAt(i) - '0']++;
        }
        int result = Integer.MAX_VALUE;

        int head = 0;

        int end = 0;
        int begin = 0;
        while (end < m) {
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


    /**
     * 77. Combinations
     *
     * @param n
     * @param k
     * @return
     */
    public List<List<Integer>> combine(int n, int k) {
        if (n <= 0 || k <= 0) {
            return new ArrayList<>();
        }
        List<List<Integer>> result = new ArrayList<>();
        intervalCombine(result, new ArrayList<Integer>(), 1, n, k);
        return result;


    }

    private void intervalCombine(List<List<Integer>> result, List<Integer> integers, int start, int n, int k) {
        if (integers.size() == k) {
            result.add(new ArrayList<>(integers));
            return;
        }
        for (int i = start; i <= n; i++) {
            integers.add(i);
            intervalCombine(result, integers, i + 1, n, k);
            integers.remove(integers.size() - 1);
        }

    }


    /**
     * 78. Subsets
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> subsets(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new ArrayList<>();
        }
        List<List<Integer>> result = new ArrayList<>();
        intervalSubSets(result, new ArrayList<Integer>(), 0, nums);
        return result;

    }

    private void intervalSubSets(List<List<Integer>> result, List<Integer> integers, int start, int[] nums) {
        result.add(new ArrayList<>(integers));
        for (int i = start; i < nums.length; i++) {
            integers.add(nums[i]);
            intervalSubSets(result, integers, i + 1, nums);
            integers.remove(integers.size() - 1);
        }
    }


    /**
     * 79. Word Search
     *
     * @param board
     * @param word
     * @return
     */
    public boolean exist(char[][] board, String word) {
        if (board == null || board.length == 0 || word == null) {
            return false;
        }
        int row = board.length;
        int column = board[0].length;
        boolean[][] used = new boolean[row][column];
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (board[i][j] == word.charAt(0) && intervalExist(used, i, j, 0, board, word)) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean intervalExist(boolean[][] used, int i, int j, int start, char[][] board, String word) {
        if (start == word.length()) {
            return true;
        }
        if (i < 0 || i >= board.length || j < 0 || j >= board[i].length || used[i][j] || board[i][j] != word.charAt(start)) {
            return false;
        }
        used[i][j] = true;
        if (intervalExist(used, i - 1, j, start + 1, board, word) ||
                intervalExist(used, i + 1, j, start + 1, board, word) ||
                intervalExist(used, i, j - 1, start + 1, board, word) ||
                intervalExist(used, i, j + 1, start + 1, board, word)) {
            return true;
        }
        used[i][j] = false;
        return false;
    }


    /**
     * 80. Remove Duplicates from Sorted Array II
     *
     * @param nums
     * @return
     */
    public int removeDuplicatesII(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int index = 1;
        int count = 1;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] == nums[i - 1]) {
                count++;
            } else {
                count = 1;
            }
            if (count >= 3) {
                continue;
            }
            nums[index++] = nums[i];
        }
        return index;
    }


    /**
     * 81. Search in Rotated Sorted Array II
     *
     * @param nums
     * @param target
     * @return
     */
    public boolean searchII(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return false;
        }
        int left = 0;
        int right = nums.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;

            if (nums[mid] == target) {
                return true;
            }
            if (nums[left] == nums[right]) {
                left++;
            } else if (nums[left] <= nums[mid]) {
                if (target < nums[mid] && nums[left] <= target) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            } else {
                if (nums[mid] < target && target <= nums[right]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }
        return nums[left] == target;
    }


    /**
     * 84. Largest Rectangle in Histogram
     *
     * @param heights
     * @return
     */
    public int largestRectangleArea(int[] heights) {
        if (heights == null || heights.length == 0) {
            return 0;
        }
        Stack<Integer> stack = new Stack<>();
        int result = 0;
        for (int i = 0; i <= heights.length; i++) {
            int h = i == heights.length ? 0 : heights[i];
            if (stack.isEmpty() || heights[stack.peek()] <= h) {
                stack.push(i);
            } else {
                Integer previous = stack.pop();

                int size = stack.isEmpty() ? i : i - stack.peek() - 1;

                result = Math.max(result, size * heights[previous]);

                i--;
            }
        }
        return result;
    }

    /**
     * 85. Maximal Rectangle
     *
     * @param matrix
     * @return
     */
    public int maximalRectangle(char[][] matrix) {
        if (matrix == null || matrix.length == 0) {
            return 0;
        }
        int row = matrix.length;
        int column = matrix[0].length;
        int result = 0;
        int[][] dp = new int[row][column];
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (matrix[i][j] == '0') {
                    dp[i][j] = -1;
                    continue;
                }

                dp[i][j] = j;

                if (j >= 1 && dp[i][j - 1] != -1) {
                    dp[i][j] = dp[i][j - 1];
                }
                int width = j - dp[i][j] + 1;
                for (int k = i; k >= 0 && matrix[k][j] != '0'; k--) {
                    width = Math.min(width, j - dp[k][j] + 1);
                    result = Math.max(result, width * (i - k + 1));
                }

            }
        }
        return result;
    }


    /**
     * @param matrix
     * @return
     */
    public int maximalRectangleII(char[][] matrix) {
        if (matrix == null || matrix.length == 0) {
            return 0;
        }
        int result = 0;
        int row = matrix.length;

        int column = matrix[0].length;

        int[] left = new int[column];

        int[] right = new int[column];

        int[] height = new int[column];
        for (int j = 0; j < column; j++) {
            right[j] = column;
        }
        for (char[] chars : matrix) {

            int currentLeft = 0;

            int currentRight = column;

            for (int j = 0; j < column; j++) {
                if (chars[j] == '0') {
                    height[j] = 0;
                } else {
                    height[j]++;
                }

            }
            for (int j = 0; j < column; j++) {
                if (chars[j] == '1') {
                    left[j] = Math.max(left[j], currentLeft);
                } else {
                    left[j] = 0;
                    currentLeft = j + 1;
                }
            }

            for (int j = column - 1; j >= 0; j--) {
                if (chars[j] == '1') {
                    right[j] = Math.min(right[j], currentRight);

                } else {
                    right[j] = column;
                    currentRight = j;
                }
            }
            for (int j = 0; j < column; j++) {
                result = Math.max(result, height[j] * (right[j] - left[j]));
            }
        }
        return result;
    }


    /**
     * 86. Partition List     * @param head
     *
     * @param x
     * @return
     */
    public ListNode partition(ListNode head, int x) {
        if (head == null) {
            return null;
        }
        ListNode smallNode = new ListNode(0);
        ListNode bigNode = new ListNode(0);
        ListNode s1 = smallNode;
        ListNode b1 = bigNode;
        while (head != null) {
            if (head.val < x) {
                s1.next = head;
                s1 = s1.next;
            } else {
                b1.next = head;
                b1 = b1.next;
            }
            head = head.next;
        }
        b1.next = null;
        s1.next = bigNode.next;
        bigNode.next = null;
        return smallNode.next;
    }

    /**
     * 87. Scramble String
     *
     * @param s1
     * @param s2
     * @return
     */
    public boolean isScramble(String s1, String s2) {
        if (s1 == null || s2 == null) {
            return false;
        }
        if (s1.equals(s2)) {
            return true;
        }
        int m = s1.length();
        int n = s2.length();
        if (m != n) {
            return false;
        }
        int[] hash = new int[256];

        for (int i = 0; i < m; i++) {
            hash[s1.charAt(i) - 'a']++;
            hash[s2.charAt(i) - 'a']--;
        }
        for (int i = 0; i < hash.length; i++) {
            if (hash[i] != 0) {
                return false;
            }
        }
        for (int i = 1; i < m; i++) {
            if (isScramble(s1.substring(0, i), s2.substring(0, i))
                    && isScramble(s1.substring(i), s2.substring(i))) {
                return true;
            }
            if (isScramble(s1.substring(i), s2.substring(0, m - i))
                    && isScramble(s1.substring(0, i), s2.substring(m - i))) {
                return true;
            }
        }
        return false;

    }


    /**
     * 88. Merge Sorted Array
     *
     * @param nums1
     * @param m
     * @param nums2
     * @param n
     */
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        if (nums1 == null || nums2 == null) {
            return;
        }
        int k = m + n - 1;
        m--;
        n--;
        int i = m;
        int j = n;
        while (i >= 0 && j >= 0) {
            if (nums1[i] <= nums2[j]) {
                nums1[k] = nums2[j--];
            } else {
                nums1[k] = nums1[i--];
            }
            k--;
        }
        while (j >= 0) {
            nums1[k--] = nums2[j--];
        }
    }


    /**
     * 89. Gray Code
     *
     * @param n
     * @return
     */
    public List<Integer> grayCode(int n) {
        return null;
    }


    /**
     * 90. Subsets II
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new ArrayList<>();
        }
        List<List<Integer>> ans = new ArrayList<>();
        Arrays.sort(nums);
        intervalSubSetsWithDup(ans, new ArrayList<Integer>(), 0, nums);
        return ans;
    }

    private void intervalSubSetsWithDup(List<List<Integer>> ans, List<Integer> integers, int start, int[] nums) {
        ans.add(new ArrayList<>(integers));
        for (int i = start; i < nums.length; i++) {
            if (i > start && nums[i] == nums[i - 1]) {
                continue;
            }
            integers.add(nums[i]);
            intervalSubSetsWithDup(ans, integers, i + 1, nums);
            integers.remove(integers.size() - 1);
        }
    }


    /**
     * todo
     * 91. Decode Ways
     *
     * @param s
     * @return
     */
    public int numDecodings(String s) {
        return 0;
    }


    /**
     * 92. Reverse Linked List II
     *
     * @param head
     * @param m
     * @param n
     * @return
     */
    public ListNode reverseBetween(ListNode head, int m, int n) {
        if (head == null) {
            return null;
        }
        ListNode dummy = new ListNode(0);
        dummy.next = head;

        ListNode fast = dummy;

        ListNode slow = dummy;
        for (int i = 0; i < m - 1; i++) {
            slow = slow.next;
        }
        for (int i = 0; i < n; i++) {
            fast = fast.next;
        }
        ListNode node = slow.next;

        ListNode end = fast.next;

        slow.next = reverseListNode(node, end);

//        node.next = end;

        return dummy.next;
    }


    /**
     * 93. Restore IP Addresses
     *
     * @param s
     * @return
     */
    public List<String> restoreIpAddresses(String s) {
        if (s == null || s.isEmpty()) {
            return new ArrayList<>();
        }
        List<String> result = new ArrayList<>();
        if (s.equals("0000")) {
            result.add("0.0.0.0");
            return result;
        }
        int len = s.length();

        for (int i = 1; i < 4; i++) {
            for (int j = 1; j < 4; j++) {
                for (int a = 1; a < 4; a++) {

                    String A = s.substring(0, i);

                    String B = s.substring(i, i + j);

                    String C = s.substring(i + j, i + j + a);

                    String D = s.substring(i + j + a);
                    if (checkValue(A) || checkValue(B) || checkValue(C) || checkValue(D)) {
                        continue;
                    }
                    String tmp = A + "." + B + "." + C + "." + D;

                    if (tmp.length() != 14) {
                        continue;
                    }

                    result.add(tmp);

                }
            }
        }
        return result;
    }

    private boolean checkValue(String s) {
        if (s.isEmpty()) {
            return true;
        }
        if (s.startsWith("0") && s.length() > 1) {
            return true;
        }
        return Integer.parseInt(s) > 255;
    }


    /**
     * @param root
     * @return
     */
    public List<Integer> inorderTraversal(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        List<Integer> result = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        TreeNode p = root;
        while (!stack.isEmpty() || p != null) {
            while (p != null) {
                stack.push(p);
                p = p.left;
            }
            p = stack.pop();
            result.add(p.val);
            p = p.right;
        }
        return result;
    }


}
