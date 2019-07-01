package org.dora.algorithm.leetcode.v2;

import org.dora.algorithm.datastructe.ListNode;
import org.dora.algorithm.datastructe.TreeNode;

import java.util.*;

/**
 * @author dora
 * @date 2019-06-24
 */
public class OneSolution {

    public static void main(String[] args) {
        OneSolution oneSolution = new OneSolution();
        oneSolution.multiply("2", "3");
    }

    private void reverseNums(int[] nums, int start, int end) {
        if (start > end) {
            return;
        }
        for (int i = start; i <= (start + end) / 2; i++) {
            this.reverse(nums, i, start + end - i);
        }
    }

    private void reverse(int[] nums, int i, int j) {
        int tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;
    }

    private ListNode reverseList(ListNode first, ListNode last) {

        ListNode prev = null;


        while (prev != last) {

            ListNode tmp = first.next;

            first.next = prev;

            prev = first;

            first = tmp;
        }
        return prev;

    }

    /**
     * 1.Two Sum
     *
     * @param nums
     * @param target
     * @return
     */
    public int[] twoSum(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return new int[]{};
        }
        int[] ans = new int[2];
        HashMap<Integer, Integer> hashMap = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (hashMap.containsKey(target - nums[i])) {
                ans[0] = hashMap.get(target - nums[i]);
                ans[1] = i;
            }
            hashMap.put(nums[i], i);
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
        int carry = 0;

        ListNode root = new ListNode(0);

        ListNode dummy = root;

        while (l1 != null || l2 != null || carry != 0) {
            int value = (l1 == null ? 0 : l1.val) + (l2 == null ? 0 : l2.val) + carry;

            ListNode node = new ListNode(value % 10);

            carry = value / 10;

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
        if (s == null || s.length() == 0) {
            return 0;
        }
        int left = 0;

        int result = 0;

        int[] hash = new int[256];

        for (int i = 0; i < s.length(); i++) {
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
        if (s == null || s.length() == 0) {
            return "";
        }
        int left = 0;

        int len = 0;

        int m = s.length();

        boolean[][] dp = new boolean[m][m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j <= i; j++) {
                boolean isMatch = s.charAt(j) == s.charAt(i);
                if (isMatch) {
                    if (i - j < 2) {
                        dp[j][i] = true;
                    } else {
                        dp[j][i] = dp[j + 1][i - 1];
                    }
                    if (dp[j][i] && i - j + 1 > len) {
                        left = j;
                        len = i - j + 1;
                    }
                }

            }
        }
        if (len != 0) {
            return s.substring(left, left + len);
        }
        return "";
    }

    /**
     * 6. ZigZag Conversion
     *
     * @param s
     * @param numRows
     * @return
     */
    public String convert(String s, int numRows) {
        if (s == null || s.length() == 0) {
            return "";
        }
        StringBuilder[] sb = new StringBuilder[numRows];
        for (int i = 0; i < numRows; i++) {
            sb[i] = new StringBuilder();
        }
        int index = 0;
        char[] chars = s.toCharArray();
        while (index < chars.length) {
            for (int i = 0; i < numRows && index < chars.length; i++) {
                sb[i].append(chars[index++]);
            }
            for (int i = numRows - 2; i >= 1 && index < chars.length; i--) {
                sb[i].append(chars[index++]);
            }
        }
        for (int i = 1; i < numRows; i++) {
            sb[0].append(sb[i]);
        }
        return sb[0].toString();
    }

    /**
     * 7. Reverse Integer
     *
     * @param x
     * @return
     */
    public int reverse(int x) {
        int ans = 0;
        while (x != 0) {
            if (ans > Integer.MAX_VALUE / 10 || ans < Integer.MIN_VALUE / 10) {
                return 0;
            }
            ans = ans * 10 + x % 10;
            x /= 10;
        }
        return ans;
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
        long result = 0;
        while (index < str.length() && Character.isDigit(str.charAt(index))) {
            int value = str.charAt(index) - '0';
            result = result * 10 + value;
            index++;
            if (result > Integer.MAX_VALUE) {
                return sign == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
            }
        }
        return (int) result * sign;
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
        if (x % 10 == 0) {
            return false;
        }
        int result = 0;
        while (x > result) {
            result = result * 10 + x % 10;
            x /= 10;
        }
        return result == x || result / 10 == x;

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
        int result = 0;
        int left = 0;
        int right = height.length - 1;
        while (left <= right) {
            result = Math.max(result, Math.min(height[left], height[right]) * (right - left));
            if (height[left] <= height[right]) {
                left++;
            } else {
                right--;
            }
        }
        return result;
    }

    /**
     * 13. Roman to Integer
     *
     * @param s
     * @return
     */
    public int romanToInt(String s) {
        if (s == null || s.length() == 0) {
            return 0;
        }
        int[] ans = new int[s.length()];
        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            if (ch == 'I') {
                ans[i] = 1;
            } else if (ch == 'V') {
                ans[i] = 5;
            } else if (ch == 'X') {
                ans[i] = 10;
            } else if (ch == 'L') {
                ans[i] = 50;
            }
        }
        return 0;
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
            while (strs[i].indexOf(prefix) != 0) {
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

        List<List<Integer>> ans = new ArrayList<>();

        Arrays.sort(nums);

        for (int i = 0; i < nums.length - 2; i++) {

            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }


            int left = i + 1;

            int right = nums.length - 1;

            int target = 0 - nums[i];


            while (left < right) {

                int value = nums[left] + nums[right];

                if (value == target) {
                    List<Integer> tmp = Arrays.asList(nums[i], nums[left], nums[right]);
                    ans.add(tmp);
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
        int result = nums[0] + nums[1] + nums[2];
        Arrays.sort(nums);
        for (int i = 0; i < nums.length - 2; i++) {
            int left = i + 1;
            int right = nums.length - 1;

            while (left < right) {

                int value = nums[i] + nums[left] + nums[right];
                if (value == target) {
                    return target;
                } else if (value < target) {
                    left++;
                } else {
                    right--;
                }

                if (Math.abs(value - target) < Math.abs(result - target)) {
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
        if (digits == null || digits.length() == 0) {
            return new ArrayList<>();
        }

        String[] map = new String[]{"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        LinkedList<String> ans = new LinkedList<>();
        ans.add("");
        for (int i = 0; i < digits.length(); i++) {
            int index = digits.charAt(i) - '0';
            String value = map[index];
            while (ans.peek().length() == i) {
                String pop = ans.pop();
                for (char tmp : value.toCharArray()) {
                    ans.add(pop + tmp);
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
        if (head == null) {
            return null;
        }
        ListNode root = new ListNode(0);

        root.next = head;

        ListNode slow = root;

        ListNode fast = root;
        for (int i = 0; i <= n - 1; i++) {
            fast = fast.next;
        }
        while (fast.next != null) {
            fast = fast.next;
            slow = slow.next;
        }
        ListNode node = slow.next;
        slow.next = node.next;

        node = null;

        return root.next;
    }

    /**
     * 20. Valid Parentheses
     *
     * @param s
     * @return
     */
    public boolean isValid(String s) {
        if (s == null || s.length() == 0) {
            return true;
        }
        Stack<Character> stack = new Stack<>();
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') {
                stack.push(')');
            } else if (s.charAt(i) == '[') {
                stack.push(']');
            } else if (s.charAt(i) == '{') {
                stack.push('}');
            } else {
                if (stack.isEmpty() || stack.peek() != s.charAt(i)) {
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
        if (l1 == null) {
            return l2;
        }
        if (l2 == null) {
            return l1;
        }
        if (l1.val <= l2.val) {
            l1.next = this.mergeTwoLists(l1.next, l2);
            return l1;
        } else {
            l2.next = this.mergeTwoLists(l1, l2.next);
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
        List<String> ans = new ArrayList<>();
        this.generateParenthesis(ans, 0, 0, n, "");
        return ans;
    }

    private void generateParenthesis(List<String> ans, int open, int close, int n, String s) {
        if (s.length() == 2 * n) {
            ans.add(s);
        }
        if (open < n) {
            this.generateParenthesis(ans, open + 1, close, n, s + "(");
        }
        if (close < open) {
            this.generateParenthesis(ans, open, close + 1, n, s + ")");
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
        PriorityQueue<ListNode> priorityQueue = new PriorityQueue<>(lists.length, new Comparator<ListNode>() {
            @Override
            public int compare(ListNode o1, ListNode o2) {
                if (o1.val < o2.val) {
                    return -1;
                } else if (o1.val == o2.val) {
                    return 0;
                } else {
                    return 1;
                }
            }
        });
        for (ListNode node : lists) {
            if (node != null) {
                priorityQueue.add(node);
            }
        }
        ListNode root = new ListNode(0);
        ListNode dummy = root;
        while (!priorityQueue.isEmpty()) {
            ListNode node = priorityQueue.poll();
            dummy.next = node;
            dummy = dummy.next;
            if (node.next != null) {
                priorityQueue.add(node.next);
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
        ListNode fast = root;
        ListNode slow = root;
        while (dummy.next != null && dummy.next.next != null) {

            slow = dummy.next;

            fast = dummy.next.next;

            slow.next = fast.next;

            dummy.next = fast;

            fast.next = slow;

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
        if (head == null) {
            return null;
        }

        int count = 0;

        ListNode dummy = head;

        while (dummy != null && count != k) {
            dummy = dummy.next;
            count++;
        }
        if (count == k) {
            dummy = this.reverseKGroup(dummy, k);
            while (count-- > 0) {
                ListNode tmp = head.next;
                head.next = dummy;
                dummy = head;
                head = tmp;
            }
            head = dummy;
        }
        return head;
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
     * 29. Divide Two Integers
     *
     * @param dividend
     * @param divisor
     * @return
     */
    public int divide(int dividend, int divisor) {
        return 0;
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
            this.reverseNums(nums, 0, nums.length - 1);
        } else {
            int value = nums[index - 1];
            int j = nums.length - 1;
            while (j > index - 1) {
                if (nums[j] > value) {
                    break;
                }
                j--;
            }
            this.reverse(nums, j, index - 1);
            this.reverseNums(nums, index, nums.length - 1);
        }
    }

    /**
     * 32. Longest Valid Parentheses
     *
     * @param s
     * @return
     */
    public int longestValidParentheses(String s) {
        if (s == null || s.length() == 0) {
            return 0;
        }
        int left = -1;
        int result = 0;
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') {
                stack.push(i);
            } else {
                if (stack.isEmpty()) {
                    left = i;
                } else {
                    stack.pop();

                    if (stack.isEmpty()) {
                        result = Math.max(result, i - left);
                    } else {
                        result = Math.max(result, i - stack.peek());
                    }
                }
            }
        }
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
            } else if (nums[left] <= nums[mid]) {
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
            return -1;
        }
        int left = 0;
        int right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                return mid;
            }
        }
        return 0;
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
        this.combinationSum(ans, new ArrayList<>(), candidates, 0, target);
        return ans;
    }

    private void combinationSum(List<List<Integer>> ans, List<Integer> tmp, int[] candidates, int start, int target) {
        if (target == 0) {
            ans.add(new ArrayList<>(tmp));
            return;
        }

        for (int i = start; i < candidates.length && candidates[i] <= target; i++) {
            tmp.add(candidates[i]);
            this.combinationSum(ans, tmp, candidates, i, target - candidates[i]);
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
        Arrays.sort(candidates);
        List<List<Integer>> ans = new ArrayList<>();
        this.combinationSum2(ans, new ArrayList<Integer>(), 0, candidates, target);
        return ans;

    }

    private void combinationSum2(List<List<Integer>> ans, List<Integer> tmp, int start, int[] candidates, int target) {
        if (target == 0) {
            ans.add(new ArrayList<>(tmp));
            return;

        }
        for (int i = start; i < candidates.length && target >= candidates[i]; i++) {
            if (i > start && candidates[i] == candidates[i - 1]) {
                continue;
            }
            tmp.add(candidates[i]);
            this.combinationSum2(ans, tmp, i + 1, candidates, target - candidates[i]);
            tmp.remove(tmp.size() - 1);
        }
    }


    /**
     * 41. First Missing Positive
     *
     * @param nums
     * @return
     */
    public int firstMissingPositive(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 1;
        }
        for (int i = 0; i < nums.length; i++) {
            while (nums[i] > 0 && nums[i] <= nums.length && nums[i] != nums[nums[i] - 1]) {
                this.reverse(nums, i, nums[i] - 1);
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
        int leftEdge = 0;
        int rightEdge = 0;
        while (left < right) {
            if (height[left] <= height[right]) {
                if (leftEdge <= height[left]) {
                    leftEdge = height[left];
                } else {
                    result += leftEdge - height[left];
                }
                left++;
            } else {
                if (rightEdge <= height[right]) {
                    rightEdge = height[right];
                } else {
                    result += rightEdge - height[right];
                }
                right--;
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
        if (num1 == null || num2 == null) {
            return "";
        }
        int m = num1.length();

        int n = num2.length();

        int[] pos = new int[m + n];
        for (int i = m - 1; i >= 0; i--) {
            for (int j = n - 1; j >= 0; j--) {
                int value = (num1.charAt(i) - '0') * (num2.charAt(j) - '0') + pos[i + j + 1];

                pos[i + j] += value / 10;

                pos[i + j + 1] = value % 10;

            }
        }
        StringBuilder stringBuilder = new StringBuilder();
        for (int num : pos) {
            if (stringBuilder.length() != 0 || num != 0) {
                stringBuilder.append(num);
            }
        }
        return stringBuilder.length() == 0 ? "0" : stringBuilder.toString();

    }


    /**
     * 44. Wildcard Matching
     *
     * @param s
     * @param p
     * @return
     */
    public boolean isMatchII(String s, String p) {
        if (s == null || p == null) {
            return false;
        }
        int m = s.length();

        int n = p.length();

        boolean[][] dp = new boolean[m + 1][n + 1];

        dp[0][0] = true;


        for (int j = 1; j <= n; j++) {
            dp[0][j] = p.charAt(j - 1) == '*' ? dp[0][j - 1] : false;
        }

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (s.charAt(i - 1) == p.charAt(j - 1) || p.charAt(j - 1) == '?') {
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
            furthest = Math.max(furthest, i + nums[i]);
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
        this.permute(ans, new ArrayList<Integer>(), used, nums);
        return ans;
    }

    private void permute(List<List<Integer>> ans, List<Integer> tmp, boolean[] used, int[] nums) {
        if (tmp.size() == nums.length) {
            ans.add(new ArrayList<>(tmp));
        }
        for (int i = 0; i < nums.length; i++) {
            if (used[i]) {
                continue;
            }
            tmp.add(nums[i]);
            used[i] = true;
            this.permute(ans, tmp, used, nums);

            tmp.remove(tmp.size() - 1);
            used[i] = false;
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
        List<List<Integer>> ans = new ArrayList<>();
        Arrays.sort(nums);
        boolean[] used = new boolean[nums.length];
        this.permuteUnique(ans, new ArrayList<Integer>(), used, nums);
        return ans;
    }

    private void permuteUnique(List<List<Integer>> ans, List<Integer> integers, boolean[] used, int[] nums) {
        if (integers.size() == nums.length) {
            ans.add(new ArrayList<>(integers));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (used[i]) {
                continue;
            }
            if (i > 0 && nums[i] == nums[i - 1] && !used[i - 1]) {
                continue;
            }

            integers.add(nums[i]);

            used[i] = true;

            this.permuteUnique(ans, integers, used, nums);

            integers.remove(integers.size() - 1);

            used[i] = false;
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
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j <= i; j++) {
                this.swapMatrix(matrix, i, j);
            }
        }
        for (int i = 0; i < matrix.length; i++) {
            this.reverseNums(matrix[i], 0, matrix[i].length - 1);
        }
    }

    private void swapMatrix(int[][] matrix, int i, int j) {
        int tmp = matrix[i][j];
        matrix[i][j] = matrix[j][i];
        matrix[j][i] = tmp;
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
        long p = Math.abs((long) n);
        double result = 1.0;
        while (p != 0) {
            if ((p % 2) == 1) {
                result *= x;
            }
            x *= x;
            p >>= 1;
        }
        return n < 0 ? 1 / result : result;
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
        char[][] queens = new char[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                queens[i][j] = '.';
            }
        }
        List<List<String>> ans = new ArrayList<>();
        this.solveNQueens(ans, 0, n, queens);
        return ans;
    }

    private void solveNQueens(List<List<String>> ans, int row, int n, char[][] queens) {
        if (row == n) {
            List<String> tmp = new ArrayList<>();
            for (int i = 0; i < queens.length; i++) {
                String value = String.valueOf(queens[i]);
                tmp.add(value);
            }
            ans.add(new ArrayList<>(tmp));
        }
        for (int i = 0; i < n; i++) {
            if (this.validQueens(i, row, queens)) {
                queens[row][i] = 'Q';
                this.solveNQueens(ans, row + 1, n, queens);
                queens[row][i] = '.';
            }
        }
    }

    private boolean validQueens(int col, int row, char[][] queens) {
        for (int i = row - 1; i >= 0; i--) {
            if (queens[i][col] == 'Q') {
                return false;
            }
        }
        for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--) {
            if (queens[i][j] == 'Q') {
                return false;
            }
        }
        for (int i = row - 1, j = col + 1; i >= 0 && j < queens.length; i--, j++) {
            if (queens[i][j] == 'Q') {
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
            if (this.validQueens(dp, i, row, n)) {
                dp[row] = i;
                count += this.totalNQueens(dp, row + 1, n);
                dp[row] = -1;
            }
        }
        return count;
    }

    private boolean validQueens(int[] dp, int col, int row, int n) {
        for (int i = row - 1; i >= 0; i--) {
            if (dp[i] == col || Math.abs(dp[i] - col) == Math.abs(row - i)) {
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
        int local = 0;
        int result = Integer.MIN_VALUE;
        for (int i = 0; i < nums.length; i++) {
            local = local > 0 ? local + nums[i] : nums[i];
            result = Math.max(result, local);
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
        return null;
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

        int total = 0;

        while (left <= right && top <= bottom) {
            for (int i = left; i <= right; i++) {
                matrix[top][i] = ++total;
            }
            for (int i = top + 1; i <= bottom; i++) {
                matrix[i][right] = ++total;
            }
            if (top != bottom) {
                for (int i = right - 1; i >= left; i--) {
                    matrix[bottom][i] = ++total;
                }
            }
            if (left != right) {
                for (int i = bottom - 1; i > top; i--) {
                    matrix[i][left] = ++total;
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
     * 60. Permutation Sequence
     *
     * @param n
     * @param k
     * @return
     */
    public String getPermutation(int n, int k) {
        if (n <= 0) {
            return "";
        }

        List<Integer> numbers = new ArrayList<>();

        for (int i = 1; i <= n; i++) {
            numbers.add(i);
        }
        int[] base = new int[n + 1];
        base[0] = 1;
        int factory = 1;

        return "";
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
        ListNode fast = head;
        int count = 1;
        while (fast.next != null) {
            fast = fast.next;
            count++;
        }
        fast.next = head;
        ListNode slow = head;
        if ((k %= count) != 0) {
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
        if (m == 0 && n == 0) {
            return 0;
        }
        int[] dp = new int[n];

        dp[0] = 1;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                dp[j] = dp[j] + (j > 0 ? dp[j - 1] : 0);
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
        int row = obstacleGrid.length;

        int column = obstacleGrid[0].length;

        int[] dp = new int[column];

        dp[0] = 1;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (obstacleGrid[i][j] == 1) {
                    dp[j] = 0;
                } else {
                    dp[j] = dp[j] + (j > 0 ? dp[j - 1] : 0);
                }
            }
        }
        return dp[column - 1];
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
        int row = grid.length;

        int column = grid[0].length;

        int[][] dp = new int[row][column];

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
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
        return dp[row - 1][column - 1];
    }


    /**
     * todo
     * 65. Valid Number
     *
     * @param s
     * @return
     */
    public boolean isNumber(String s) {
        if (s == null || s.length() == 0) {
            return false;
        }
        return true;
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

        int m = a.length() - 1;

        int n = b.length() - 1;

        int carry = 0;

        String ans = "";

        while (m >= 0 || n >= 0 || carry != 0) {
            int value = (m >= 0 ? a.charAt(m--) - '0' : 0) + (n >= 0 ? b.charAt(n--) - '0' : 0) + carry;

            carry = value / 2;

            ans = String.valueOf(value % 2) + ans;
        }
        return ans;
    }


    /**
     * 68. Text Justification
     *
     * @param words
     * @param maxWidth
     * @return
     */
    public List<String> fullJustify(String[] words, int maxWidth) {
        if (words == null || words.length == 0) {
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

            StringBuilder sb = new StringBuilder();

            if (countOfWord == 1) {
                sb.append(words[startIndex]);
            } else {
                int space = lastRow ? 1 : 1 + (maxWidth - line + 1) / (countOfWord - 1);
                int extraSpace = lastRow ? 0 : (maxWidth - line + 1) % (countOfWord - 1);
                sb.append(this.constructRow(words, startIndex, endIndex, space, extraSpace));
            }

            startIndex = endIndex;

            ans.add(this.justifyLength(sb.toString(), maxWidth));
        }
        return ans;
    }

    private String justifyLength(String str, int maxWidth) {
        while (str.length() > maxWidth) {
            str = str.substring(0, str.length() - 1);
        }
        while (str.length() < maxWidth) {
            str = str + " ";
        }
        return str;
    }

    private String constructRow(String[] words, int startIndex, int endIndex, int space, int extraSpace) {
        StringBuilder sb = new StringBuilder();
        for (int i = startIndex; i < endIndex; i++) {
            sb.append(words[i]);
            int tmp = space;
            while (tmp-- > 0) {
                sb.append(" ");
            }
            if (extraSpace > 0) {
                sb.append(" ");
                extraSpace--;
            }
        }
        return sb.toString();
    }


    /**
     * 69. Sqrt(x)
     *
     * @param x
     * @return
     */
    public int mySqrt(int x) {
        double result = x;
        double precision = 0.00001;
        while ((result * result - x) > precision) {
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

        if (n <= 2) {
            return n;
        }

        int num1 = 1;

        int num2 = 2;

        int sum = 0;

        for (int i = 3; i <= n; i++) {

            sum = num1 + num2;

            num1 = num2;

            num2 = sum;
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
        if (path == null || path.length() == 0) {
            return "";
        }
        String[] items = path.split("/");
        List<String> skip = Arrays.asList("", "/", ".", "..");
        Stack<String> stack = new Stack<>();
        for (String item : items) {
            if ("..".equals(item) && !stack.isEmpty()) {
                stack.pop();
            } else if (skip.contains(item)) {
                continue;
            } else {
                stack.add(item);
            }
        }
        if (stack.isEmpty()) {
            return "/";
        }
        String sb = "";
        while (!stack.isEmpty()) {
            sb = "/" + stack.pop() + sb;
        }
        return sb;

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

        for (int i = 0; i <= m; i++) {
            for (int j = 0; j <= n; j++) {
                if (i == 0 && j == 0) {
                    dp[i][j] = 0;
                } else if (j == 0) {
                    dp[i][j] = 1 + dp[i - 1][j];
                } else if (i == 0) {
                    dp[i][j] = 1 + dp[i][j - 1];
                } else if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = Math.min(Math.min(dp[i][j - 1], dp[i - 1][j]), dp[i - 1][j - 1]) + 1;
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
        boolean setRow = false;
        boolean setColumn = false;
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                if (matrix[i][j] == 0) {

                    matrix[i][0] = 0;

                    matrix[0][j] = 0;

                    if (i == 0) {
                        setColumn = true;
                    }
                    if (j == 0) {
                        setRow = true;
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

        if (setRow) {
            for (int i = 0; i < matrix.length; i++) {
                matrix[i][0] = 0;
            }
        }
        if (setColumn) {
            for (int j = 0; j < matrix[0].length; j++) {
                matrix[0][j] = 0;
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
            if (matrix[i][j] == target) {
                return true;
            } else if (matrix[i][j] < target) {
                j++;
            } else {
                i--;
            }
        }
        return false;
    }


    /**
     * todo
     * 75. Sort Colors
     *
     * @param nums
     */
    public void sortColors(int[] nums) {
        if (nums == null || nums.length == 0) {
            return;
        }
        int red = 0;
        int blue = nums.length - 1;
        for (int i = 0; i < nums.length; i++) {
            while (i < blue && nums[i] == 2) {
                this.reverse(nums, i, blue--);
            }
            while (i > red && nums[i] == 0) {
                this.reverse(nums, i, red++);
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
        int result = Integer.MAX_VALUE;

        int end = 0;

        int begin = 0;

        int head = 0;

        int[] hash = new int[256];

        int count = t.length();

        for (int i = 0; i < t.length(); i++) {
            hash[t.charAt(i) - '0']++;
        }

        while (end < s.length()) {
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
        if (n <= 0) {
            return new ArrayList<>();
        }
        List<List<Integer>> ans = new ArrayList<>();
        this.combine(ans, new ArrayList<Integer>(), 1, n, k);
        return ans;
    }

    private void combine(List<List<Integer>> ans, List<Integer> tmp, int start, int n, int k) {
        if (tmp.size() == k) {
            ans.add(new ArrayList<>(tmp));
            return;
        }
        for (int i = start; i <= n; i++) {
            tmp.add(i);
            this.combine(ans, tmp, i + 1, n, k);
            tmp.remove(tmp.size() - 1);
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
        List<List<Integer>> ans = new ArrayList<>();
        this.subsets(ans, new ArrayList<Integer>(), 0, nums);
        return ans;

    }

    private void subsets(List<List<Integer>> ans, List<Integer> tmp, int start, int[] nums) {
        ans.add(new ArrayList<>(tmp));
        for (int i = start; i < nums.length; i++) {
            tmp.add(nums[i]);
            this.subsets(ans, tmp, i + 1, nums);
            tmp.remove(tmp.size() - 1);
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
        if (board == null || board.length == 0) {
            return false;
        }
        int row = board.length;
        int column = board[0].length;
        boolean[][] used = new boolean[row][column];
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (board[i][j] == word.charAt(0) && this.verifyExist(board, i, j, used, 0, word)) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean verifyExist(char[][] board, int i, int j, boolean[][] used, int end, String word) {

        if (end == word.length()) {
            return true;
        }

        if (i < 0 || i >= board.length || j < 0 || j >= board[i].length || used[i][j] || board[i][j] != word.charAt(end)) {
            return false;
        }

        used[i][j] = true;

        if (this.verifyExist(board, i - 1, j, used, end + 1, word) || this.verifyExist(board, i + 1, j, used, end + 1, word)
                || this.verifyExist(board, i, j - 1, used, end + 1, word) || this.verifyExist(board, i, j + 1, used, end + 1, word)) {
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
        int count = 1;
        int index = 1;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] == nums[i - 1]) {
                count++;
                if (count > 2) {
                    continue;
                }
            } else {
                count = 1;
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
        while (left <= right) {

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
                if (target > nums[mid] && nums[right] >= target) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }
        return false;
    }

    /**
     * 82. Remove Duplicates from Sorted List II
     *
     * @param head
     * @return
     */
    public ListNode deleteDuplicatesII(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        if (head.val == head.next.val) {
            ListNode node = head.next.next;
            while (node != null && node.val == head.val) {
                node = node.next;
            }
            return this.deleteDuplicatesII(node);
        } else {
            head.next = this.deleteDuplicatesII(head.next);
            return head;
        }
    }

    /**
     * 83. Remove Duplicates from Sorted List
     *
     * @param head
     * @return
     */
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }

        if (head.val == head.next.val) {
            return this.deleteDuplicates(head.next);
        }

        head.next = this.deleteDuplicates(head.next);

        return head;
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
        int result = 0;
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i <= heights.length; i++) {
            int h = i == heights.length ? 0 : heights[i];

            if (stack.isEmpty() || heights[stack.peek()] <= h) {
                stack.push(i);
            } else {
                int value = heights[stack.pop()];
                int side = stack.isEmpty() ? i : i - stack.peek() - 1;
                result = Math.max(result, value * side);
                i--;
            }
        }
        return result;
    }


    /**
     * todo
     * 85. Maximal Rectangle
     *
     * @param matrix
     * @return
     */
    public int maximalRectangle(char[][] matrix) {
        if (matrix == null || matrix.length == 0) {
            return 0;
        }
        int column = matrix[0].length;


        int result = 0;

        int[] left = new int[column];

        int[] right = new int[column];

        int[] height = new int[column];


        for (int j = 0; j < column; j++) {
            right[j] = column;
        }

        for (int i = 0; i < matrix.length; i++) {

            int leftEdge = 0;

            int rightEdge = column;

            for (int j = 0; j < column; j++) {
                if (matrix[i][j] == '0') {
                    height[j] = 0;
                } else {
                    height[j]++;
                }
            }
            for (int j = 0; j < column; j++) {
                if (matrix[i][j] == '1') {
                    left[j] = Math.max(left[j], leftEdge);
                } else {

                    left[j] = 0;

                    leftEdge = j + 1;
                }
            }

            for (int j = column - 1; j >= 0; j--) {
                if (matrix[i][j] == '1') {
                    right[j] = Math.min(rightEdge, right[j]);
                } else {
                    right[j] = column;
                    rightEdge = j;
                }
            }


            for (int j = 0; j < column; j++) {
                result = Math.max(result, (right[j] - left[j]) * height[j]);

            }
        }
        return result;


    }


    /**
     * 86. Partition List
     *
     * @param head
     * @param x
     * @return
     */
    public ListNode partition(ListNode head, int x) {
        if (head == null) {
            return null;
        }

        ListNode dummy1 = new ListNode(0);
        ListNode d1 = dummy1;


        ListNode dummy2 = new ListNode(0);
        ListNode d2 = dummy2;

        while (head != null) {
            if (head.val < x) {
                d1.next = head;
                d1 = d1.next;
            } else {

                d2.next = head;

                d2 = d2.next;
            }
            head = head.next;

        }

        d2.next = null;

        d1.next = dummy2.next;


        return dummy1.next;
    }


    /**
     * todo
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
        int[] hash = new int[256];
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

        while (m >= 0 && n >= 0) {
            if (nums1[m] >= nums2[n]) {
                nums1[k--] = nums1[m--];
            } else {
                nums1[k--] = nums2[n--];
            }
        }

        while (n >= 0) {
            nums1[k--] = nums2[n--];
        }

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

        this.subsetsWithDup(ans, new ArrayList<Integer>(), 0, nums);

        return ans;

    }

    private void subsetsWithDup(List<List<Integer>> ans, List<Integer> tmp, int start, int[] nums) {
        ans.add(new ArrayList<>(tmp));
        for (int i = start; i < nums.length; i++) {
            if (i > start && nums[i] == nums[i - 1]) {
                continue;
            }
            tmp.add(nums[i]);

            this.subsetsWithDup(ans, tmp, i + 1, nums);

            tmp.remove(tmp.size() - 1);
        }
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
        if (head == null || head.next == null) {
            return head;
        }
        ListNode root = new ListNode(0);

        root.next = head;

        ListNode slow = root;

        ListNode fast = root;

        for (int i = 0; i < m - 1; i++) {
            slow = slow.next;
        }

        for (int i = 0; i < n; i++) {
            fast = fast.next;
        }

        ListNode start = slow.next;

        ListNode end = fast.next;

        ListNode reverse = this.reverseList(start, fast);

        slow.next = reverse;


        start.next = end;


        return root.next;

    }

    /**
     * 93. Restore IP Addresses
     *
     * @param s
     * @return
     */
    public List<String> restoreIpAddresses(String s) {
        return null;
    }


    /**
     * 94. Binary Tree Inorder Traversal
     *
     * @param root
     * @return
     */
    public List<Integer> inorderTraversal(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        List<Integer> ans = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        TreeNode p = root;
        while (!stack.isEmpty() || p != null) {

            while (p != null) {

                stack.push(p);

                p = p.left;
            }
            p = stack.pop();

            ans.add(p.val);

            p = p.right;

        }
        return ans;
    }


    /**
     * 95. Unique Binary Search Trees II
     *
     * @param n
     * @return
     */
    public List<TreeNode> generateTrees(int n) {
        if (n <= 0) {
            return new ArrayList<>();
        }
        return this.generateTrees(1, n);
    }

    private List<TreeNode> generateTrees(int start, int end) {
        List<TreeNode> ans = new ArrayList<>();
        if (start == end) {
            ans.add(new TreeNode(start));
            return ans;
        }
        if (start > end) {
            ans.add(null);
            return ans;
        }
        for (int i = start; i <= end; i++) {
            List<TreeNode> leftList = this.generateTrees(start, i - 1);

            List<TreeNode> rightList = this.generateTrees(i + 1, end);

            for (TreeNode left : leftList) {
                for (TreeNode right : rightList) {
                    TreeNode root = new TreeNode(i);
                    root.left = left;
                    root.right = right;
                    ans.add(root);
                }
            }
        }
        return ans;
    }


    /**
     * 96. Unique Binary Search Trees
     *
     * @param n
     * @return
     */
    public int numTrees(int n) {
        if (n <= 0) {
            return 0;
        }
        int[] dp = new int[n + 1];


        dp[0] = dp[1];

        for (int i = 2; i <= n; i++) {
            for (int j = 1; j <= i; j++) {
                dp[i] += dp[j] * dp[i - 1];
            }
        }
        return dp[n];
    }

    /**
     * todo
     * 97. Interleaving String
     *
     * @param s1
     * @param s2
     * @param s3
     * @return
     */
    public boolean isInterleave(String s1, String s2, String s3) {
        if (s1 == null || s2 == null || s3 == null) {
            return false;
        }
        int m = s1.length();

        int n = s2.length();
        if (m + n != s3.length()) {
            return false;
        }
        boolean[][] dp = new boolean[m][n];

        for (int i = 0; i < m; i++) {
            dp[i][0] = s1.charAt(i) == s3.charAt(i);
        }
        for (int j = 0; j < n; j++) {
            dp[0][j] = s2.charAt(j) == s3.charAt(j);
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {

            }
        }
        return false;
    }

    /**
     * 98. Validate Binary Search Tree
     *
     * @param root
     * @return
     */
    public boolean isValidBST(TreeNode root) {
        if (root == null) {
            return false;
        }
        Stack<TreeNode> stack = new Stack<>();
        TreeNode prev = null;
        TreeNode p = root;
        while (!stack.isEmpty() || p != null) {
            while (p != null) {
                stack.push(p);
                p = p.left;
            }
            p = stack.pop();
            if (prev != null && prev.val >= p.val) {
                return false;
            }
            prev = p;

            p = p.right;
        }
        return true;
    }


    /**
     * 99. Recover Binary Search Tree
     *
     * @param root
     */
    public void recoverTree(TreeNode root) {
        if (root == null) {
            return;
        }
        Stack<TreeNode> stack = new Stack<>();
        TreeNode first = null;
        TreeNode second = null;
        TreeNode prev = null;
        TreeNode p = root;
        while (!stack.isEmpty() || p != null) {

            while (p != null) {
                stack.push(p);
                p = p.left;
            }

            p = stack.pop();

            if (prev != null) {
                if (first == null && prev.val >= p.val) {
                    first = prev;
                }
                if (first != null && prev.val >= p.val) {
                    second = p;
                }
            }
            prev = p;

            p = p.right;
        }

        if (first != null && second != null) {
            int tmp = first.val;
            first.val = second.val;
            second.val = tmp;
        }
    }


    /**
     * 100. Same Tree
     *
     * @param p
     * @param q
     * @return
     */
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null && q == null) {
            return true;
        }
        if (p == null || q == null) {
            return false;
        }
        if (p.val == q.val) {
            return this.isSameTree(p.left, q.left) && this.isSameTree(p.right, q.right);
        }
        return false;
    }


}