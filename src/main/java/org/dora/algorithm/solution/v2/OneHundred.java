package org.dora.algorithm.solution.v2;

import org.dora.algorithm.datastructe.ListNode;
import org.dora.algorithm.datastructe.TreeNode;

import java.util.*;

/**
 * @author dora
 * @date 2019-08-19
 */
public class OneHundred {


    public static void main(String[] args) {
        OneHundred oneHundred = new OneHundred();
        oneHundred.myPow(2.00000, -2147483648);
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
            return -1;
        }
        int m = nums1.length;

        int n = nums2.length;

        if (n < m) {
            return this.findMedianSortedArrays(nums2, nums1);
        }
        int imin = 0;
        int imax = nums1.length;
        int leftMax = 0;
        int rightMin = 0;
        while (imin <= imax) {
            int i = imin + (imax - imin) / 2;
            int j = (m + n + 1) / 2 - i;
            if (i < m && nums1[i] < nums2[j - 1]) {
                imin = i + 1;
            }
        }
        return 0;

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
                    if (s.charAt(i - 1) != p.charAt(j - 2) && p.charAt(j - 2) != '.') {
                        dp[i][j] = dp[i][j - 2];
                    } else {
                        dp[i][j] = dp[i][j - 1] || dp[i - 1][j] || dp[i][j - 2];
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
        while (left < right) {
            result = Math.max(result, (right - left) * Math.min(height[left], height[right]));
            if (height[left] < height[right]) {
                left++;
            } else {
                right--;
            }
        }
        return result;
    }

    /**
     * 12. Integer to Roman
     *
     * @param num
     * @return
     */
    public String intToRoman(int num) {
        if (num <= 0) {
            return "";
        }
        return "";
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
        Arrays.sort(nums);
        List<List<Integer>> ans = new ArrayList<>();
        for (int i = 0; i < nums.length - 2; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            int left = i + 1;
            int right = nums.length - 1;
            while (left < right) {
                int val = nums[i] + nums[left] + nums[right];
                if (val == 0) {
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
                } else if (val < 0) {
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
        for (int i = 0; i < nums.length; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;

            }
            int left = i + 1;
            int right = nums.length - 1;
            while (left < right) {
                int val = nums[i] + nums[left] + nums[right];
                if (val == target) {
                    return val;
                }
                if (val < target) {
                    left++;
                } else {
                    right--;
                }
                if (Math.abs(val - target) < Math.abs(result - target)) {
                    result = val;
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
        LinkedList<String> deque = new LinkedList<>();
        deque.add("");
        for (int i = 0; i < digits.length(); i++) {
            int index = Character.getNumericValue(digits.charAt(i));
            String value = map[index];
            while (deque.peek().length() == i) {
                String pop = deque.pop();
                for (Character tmp : value.toCharArray()) {
                    deque.add(pop + tmp);
                }
            }
        }
        return deque;
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
            slow = slow.next;
            fast = fast.next;
        }
        ListNode tmp = slow.next;
        slow.next = tmp.next;
        tmp = null;
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
        if (l1 == null || l2 == null) {
            return l1 == null ? l2 : l1;
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
            return Collections.emptyList();
        }
        List<String> ans = new ArrayList<>();
        this.generateParenthesis(ans, 0, 0, "", n);
        return ans;
    }

    private void generateParenthesis(List<String> ans, int open, int close, String s, int n) {
        if (s.length() == 2 * n) {
            ans.add(s);
        }
        if (open < n) {
            this.generateParenthesis(ans, open + 1, close, s + "(", n);
        }
        if (close < open) {
            this.generateParenthesis(ans, open, close + 1, s + ")", n);
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
        while (dummy.next != null && dummy.next.next != null) {
            ListNode fast = dummy.next.next;
            ListNode slow = dummy.next;

            slow.next = fast.next;
            fast.next = slow;
            dummy.next = fast;

            dummy = dummy.next.next;
        }
        return root.next;
    }

    private ListNode reversListNode(ListNode start, ListNode end) {
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
     * 25. Reverse Nodes in k-Group
     *
     * @param head
     * @param k
     * @return
     */
    public ListNode reverseKGroup(ListNode head, int k) {
        if (head == null || head.next == null || k <= 0) {
            return head;
        }
        ListNode currNode = head;

        int count = 0;

        while (currNode != null && count != k) {
            currNode = currNode.next;
            count++;
        }
        if (count != k) {
            return head;
        }

        ListNode newHead = this.reversListNode(head, currNode);

        head.next = this.reverseKGroup(currNode, k);

        return newHead;
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
     * todo kmp 算法实现
     * 28. Implement strStr()
     *
     * @param haystack
     * @param needle
     * @return
     */
    public int strStr(String haystack, String needle) {
        return 0;
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
            this.reverseArray(nums, 0, nums.length - 1);
        } else {
            int value = nums[index - 1];
            int j = nums.length - 1;
            while (j > index - 1) {
                if (nums[j] > value) {
                    break;
                }
                j--;
            }
            this.reverseValue(nums, index - 1, j);

            this.reverseArray(nums, index, nums.length - 1);
        }

    }

    private void reverseArray(int[] nums, int start, int end) {
        if (start > end) {
            return;
        }
        for (int i = start; i <= (start + end) / 2; i++) {
            this.reverseValue(nums, i, start + end - i);
        }
    }

    private void reverseValue(int[] nums, int i, int j) {
        int val = nums[i];
        nums[i] = nums[j];
        nums[j] = val;
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
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') {
                stack.push(i);
            } else if (!stack.isEmpty() && s.charAt(stack.peek()) == '(') {
                stack.pop();
            } else {
                stack.push(i);
            }
        }
        if (stack.isEmpty()) {
            return s.length();
        } else {
            int a = s.length();
            int result = 0;
            while (!stack.isEmpty()) {
                int pop = stack.pop();
                result = Math.max(result, a - pop - 1);
                a = pop;
            }
            result = Math.max(a, result);
            return result;
        }
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
                if (target < nums[mid] && target >= nums[left]) {
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
        this.combinationSum(ans, new ArrayList<Integer>(), 0, candidates, target);
        return ans;
    }

    private void combinationSum(List<List<Integer>> ans, List<Integer> tmp, int start, int[] candidates, int target) {
        if (target == 0) {
            ans.add(new ArrayList<>(tmp));
        }
        if (target < 0) {
            return;
        }
        for (int i = start; i < candidates.length && target >= candidates[i]; i++) {
            tmp.add(candidates[i]);
            this.combinationSum(ans, tmp, i, candidates, target - candidates[i]);
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
        this.combinationSum2(ans, new ArrayList<Integer>(), candidates, 0, target);
        return ans;
    }

    private void combinationSum2(List<List<Integer>> ans, ArrayList<Integer> integers, int[] candidates, int start, int target) {
        if (target == 0) {
            ans.add(new ArrayList<>(integers));
            return;
        }
        if (target < 0) {
            return;
        }
        for (int i = start; i < candidates.length && candidates[i] <= target; i++) {
            if (i > start && candidates[i] == candidates[i - 1]) {
                continue;
            }
            integers.add(candidates[i]);
            this.combinationSum2(ans, integers, candidates, i + 1, target - candidates[i]);
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
        if (nums == null) {
            return -1;
        }
        for (int i = 0; i < nums.length; i++) {
            while (nums[i] > 0 && nums[i] <= nums.length && nums[i] != nums[nums[i] - 1]) {
                this.reverseValue(nums, i, nums[i] - 1);
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

        int left = 0;

        int right = height.length - 1;

        int result = 0;


        while (left < right) {
            while (left < right && height[left] == 0) {
                left++;

            }
            while (left < right && height[right] == 0) {
                right--;
            }
            int minValue = Math.min(height[left], height[right]);
            for (int i = left; i <= right; i++) {
                if (height[i] >= minValue) {
                    height[i] -= minValue;
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
        if (num1 == null || num2 == null) {
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
        StringBuilder sb = new StringBuilder();
        for (int num : position) {
            if (!(sb.length() == 0 && num == 0)) {
                sb.append(num);
            }
        }
        return sb.length() == 0 ? "0" : sb.toString();
    }

    /**
     * todo
     * 44. Wildcard Matching
     *
     * @param s
     * @param p
     * @return
     */
    public boolean isMatchII(String s, String p) {
        if (s == null && p == null) {
            return false;
        }
        if (p == null) {
            return true;
        }
        int m = s.length();
        int n = p.length();
        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[0][0] = true;
        for (int j = 1; j <= n; j++) {
            dp[0][j] = p.charAt(j - 1) == '*' && dp[0][j - 1];
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (s.charAt(i - 1) == p.charAt(j - 1) || p.charAt(j - 1) == '?') {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = dp[i][j - 1] || dp[i - 1][j];
                }
            }
        }
        return dp[m][n];
    }

    /**
     * 45. Jump Game II
     */
    public int jump(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int step = 0;
        int furthest = 0;
        int currentFurthest = 0;
        for (int i = 0; i < nums.length - 1; i++) {
            furthest = Math.max(furthest, i + nums[i]);
            if (i == currentFurthest) {
                step++;
                currentFurthest = furthest;
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
            return Collections.emptyList();
        }
        boolean[] used = new boolean[nums.length];
        List<List<Integer>> ans = new ArrayList<>();
        this.permute(ans, new ArrayList<Integer>(), used, nums);
        return ans;
    }

    private void permute(List<List<Integer>> ans, ArrayList<Integer> integers, boolean[] used, int[] nums) {
        if (integers.size() == nums.length) {
            ans.add(new ArrayList<>(integers));
        }
        for (int i = 0; i < nums.length; i++) {
            if (used[i]) {
                continue;
            }
            integers.add(nums[i]);
            used[i] = true;
            this.permute(ans, integers, used, nums);
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
        List<List<Integer>> ans = new ArrayList<>();
        Arrays.sort(nums);
        boolean[] used = new boolean[nums.length];
        this.permuteUnique(ans, new ArrayList<Integer>(), used, nums);
        return ans;
    }

    private void permuteUnique(List<List<Integer>> ans, ArrayList<Integer> integers, boolean[] used, int[] nums) {
        if (integers.size() == nums.length) {
            ans.add(new ArrayList<>(integers));
        }
        for (int i = 0; i < nums.length; i++) {
            if (used[i]) {
                continue;
            }
            if (i > 0 && !used[i - 1] && nums[i] == nums[i - 1]) {
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
            this.reverseArray(matrix[i], 0, matrix.length - 1);
        }
    }

    private void swapMatrix(int[][] matrix, int i, int j) {
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
            return Collections.emptyList();
        }
        HashMap<String, List<String>> map = new HashMap<>();
        for (String str : strs) {
            char[] charArray = str.toCharArray();

            Arrays.sort(charArray);

            String key = String.valueOf(charArray);

            List<String> group = map.getOrDefault(key, new ArrayList<>());

            group.add(str);

            map.put(key, group);
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
        double p = 1;

        long base = Math.abs((long) n);

        while (base > 0) {
            if ((base % 2) != 0) {
                p *= x;
            }
            x *= x;
            base /= 2;
        }
        if (p > Integer.MAX_VALUE || p < Integer.MIN_VALUE) {
            return 0;
        }
        return n < 0 ? 1 / p : p;
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
        char[][] queen = new char[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                queen[i][j] = '.';
            }
        }
        List<List<String>> ans = new ArrayList<>();
        this.solveNQueens(ans, 0, n, queen);
        return ans;

    }

    private void solveNQueens(List<List<String>> ans, int row, int n, char[][] queen) {
        if (row == n) {
            List<String> tmp = new ArrayList<>();
            for (char[] rowValue : queen) {
                tmp.add(String.valueOf(rowValue));
            }
            ans.add(tmp);
        }
        for (int i = 0; i < n; i++) {
            if (this.checkQueen(i, row, queen)) {
                queen[row][i] = 'Q';
                this.solveNQueens(ans, row + 1, n, queen);
                queen[row][i] = '.';
            }
        }
    }

    private boolean checkQueen(int col, int row, char[][] queen) {
        for (int i = 0; i < row; i++) {
            if (queen[i][col] == 'Q') {
                return false;

            }
        }
        for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--) {
            if (queen[i][j] == 'Q') {
                return false;
            }
        }
        for (int i = row - 1, j = col + 1; i >= 0 && j < queen[0].length; i--, j++) {
            if (queen[i][j] == 'Q') {
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
            if (this.checkNQueens(i, row, n, dp)) {
                dp[row] = i;
                count += this.totalNQueens(dp, row + 1, n);
                dp[row] = -1;
            }
        }
        return count;
    }

    private boolean checkNQueens(int column, int row, int n, int[] dp) {
        for (int i = row - 1; i >= 0; i--) {
            if (dp[i] == column || Math.abs(dp[i] - column) == Math.abs(i - row)) {
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
        int result = Integer.MIN_VALUE;
        int local = 0;
        for (int i = 0; i < nums.length; i++) {
            local = local >= 0 ? local + nums[i] : nums[i];
            result = Math.max(result, local);
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
            return Collections.emptyList();
        }
        int left = 0;

        int right = matrix[0].length - 1;

        int top = 0;

        int bottom = matrix.length - 1;


        List<Integer> ans = new ArrayList<>();
        while (left <= right && top <= bottom) {

            for (int i = left; i <= right; i++) {

                ans.add(matrix[top][i]);
            }
            for (int i = top + 1; i <= bottom; i++) {
                ans.add(matrix[i][right]);
            }
            if (top != bottom) {
                for (int i = right - 1; i >= left; i--) {
                    ans.add(matrix[bottom][i]);
                }
            }
            if (left != right) {
                for (int i = bottom - 1; i > left; i--) {
                    ans.add(matrix[i][left]);
                }
            }
            left++;
            right--;
            top++;
            bottom--;
        }
        return ans;
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
     * todo
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

        if (s.isEmpty()) {
            return 0;
        }
        int lastIndex = s.lastIndexOf(" ");

        return s.length() - 1 - lastIndex;
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
        int total = 0;

        int left = 0;

        int right = n - 1;

        int top = 0;

        int bottom = n - 1;

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
     * todo
     * 60. Permutation Sequence
     *
     * @param n
     * @param k
     * @return
     */
    public String getPermutation(int n, int k) {
        if (n <= 0 || k < 0) {
            return "";
        }

        List<Integer> numbers = new ArrayList<>();

        for (int i = 1; i <= n; i++) {
            numbers.add(i);
        }

        int base = 1;

        int[] pos = new int[n + 1];

        pos[0] = 1;

        for (int i = 1; i <= n; i++) {
            base *= i;
            pos[i] = base;
        }
        StringBuilder sb = new StringBuilder();

        k--;

        for (int i = 0; i < n; i++) {

            int index = k / pos[n - 1 - i];

            sb.append(numbers.get(index));

            numbers.remove(index);
            k -= index * pos[n - 1 - i];
        }
        return sb.toString();
    }


    /**
     * 61. Rotate List
     *
     * @param head
     * @param k
     * @return
     */
    public ListNode rotateRight(ListNode head, int k) {
        if (head == null || k <= 0) {
            return head;
        }
        ListNode currNode = head;
        int count = 1;
        while (currNode.next != null) {
            count++;
            currNode = currNode.next;
        }
        currNode.next = head;

        ListNode slow = head;

        if ((k %= count) != 0) {
            for (int i = 0; i < count - k; i++) {
                slow = slow.next;
                currNode = currNode.next;
            }
        }
        currNode.next = null;

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
        if (m < 0 || n < 0) {
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
        boolean afterE = true;
        boolean isNumber = false;
        boolean afterDot = false;

        return false;
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
        int carry = 0;
        int m = a.length() - 1;
        int n = b.length() - 1;
        StringBuilder sb = new StringBuilder();
        while (m >= 0 || n >= 0 || carry > 0) {
            int value = (m >= 0 ? Character.getNumericValue(a.charAt(m--)) : 0)
                    + (n >= 0 ? Character.getNumericValue(b.charAt(n--)) : 0) + carry;

            carry = value / 2;

            sb.append(value % 2);
        }
        return sb.reverse().toString();

    }


    /**
     * todo
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
        for (int i = 0, k, l; i < words.length; i += k) {
            for (k = 0, l = 0; words[i].length() + l <= maxWidth - k; k++) {
                l += words[i + k].length();
            }
            String tmp = words[i];
            for (int j = 0; j < k - 1; j++) {

            }
        }

        return ans;
    }

    private String justify(String str, int maxWidth) {
        while (str.length() < maxWidth) {
            str += " ";
        }
        while (str.length() > maxWidth) {
            str = str.substring(0, str.length() - 1);
        }
        return str;
    }

    private String construct(String[] words, int startIndex, int endIndex, int blankOfWord, int extraOfWord) {
        String ans = "";
        for (int i = startIndex; i < endIndex; i++) {
            ans += words[i];
            int tmp = blankOfWord;
            while (tmp-- > 0) {
                ans += " ";
            }
            if (extraOfWord-- > 0) {
                ans += " ";
            }
        }
        return ans;

    }

    /**
     * todo 牛顿平方发
     * 69. Sqrt(x)
     *
     * @param x
     * @return
     */
    public int mySqrt(int x) {
        double precision = 0.00001;
        double result = x;
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
        if (n == 1) {
            return 1;
        }
        if (n == 2) {
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
            return "";
        }
        String[] words = path.split("/");
        List<String> skip = Arrays.asList("/", ".", "", "..");
        Deque<String> ans = new LinkedList<>();
        for (String word : words) {
            if ("..".equals(word) && !ans.isEmpty()) {
                ans.pollLast();
            } else if (!skip.contains(word)) {
                ans.add(word);
            }
        }
        if (ans.isEmpty()) {
            return "/";
        }
        String str = "";
        for (String word : ans) {
            str = str + "/" + word;
        }
        return str;
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
        dp[0][0] = 0;
        for (int i = 1; i <= m; i++) {
            dp[i][0] = i;
        }
        for (int j = 1; j <= n; j++) {
            dp[0][j] = j;
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j = 1];
                } else {
                    dp[i][j] = 1 + Math.min(Math.min(dp[i - 1][j - 1], dp[i - 1][j]), dp[i][j - 1]);
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
        boolean setColumn = false;
        boolean setRow = false;
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
        if (setColumn) {
            for (int j = 0; j < matrix[0].length; j++) {
                matrix[0][j] = 0;
            }
        }
        if (setRow) {
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
        int row = matrix.length - 1;
        int column = matrix[0].length - 1;
        int i = row;
        int j = 0;
        while (i >= 0 && j <= column) {
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
            while (nums[i] == 2 & i < blue) {
                this.reverseValue(nums, i, blue--);
            }
            while (nums[i] == 0 && i > red) {
                this.reverseValue(nums, i, red++);
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
        int count = t.length();

        int[] hash = new int[256];

        for (int i = 0; i < t.length(); i++) {
            hash[t.charAt(i) - '0']++;
        }
        int end = 0;
        int begin = 0;
        int head = 0;
        int result = Integer.MAX_VALUE;
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
        if (n <= 0 || k <= 0) {
            return Collections.emptyList();
        }
        List<List<Integer>> ans = new ArrayList<>();
        this.combine(ans, new ArrayList<Integer>(), 1, n, k);
        return ans;
    }

    private void combine(List<List<Integer>> ans, ArrayList<Integer> integers, int start, int n, int k) {
        if (integers.size() == k) {
            ans.add(new ArrayList<>(integers));
        }
        for (int i = start; i <= n; i++) {
            integers.add(i);
            this.combine(ans, integers, i + 1, n, k);
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
        List<List<Integer>> ans = new ArrayList<>();
        this.subsets(ans, new ArrayList<Integer>(), 0, nums);
        return ans;
    }

    private void subsets(List<List<Integer>> ans, ArrayList<Integer> integers, int start, int[] nums) {
        ans.add(new ArrayList<>(integers));
        for (int i = start; i < nums.length; i++) {
            integers.add(nums[i]);
            this.subsets(ans, integers, i + 1, nums);
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
        if (board == null || board.length == 0) {
            return false;
        }
        int row = board.length;
        int column = board[0].length;
        boolean[][] used = new boolean[row][column];
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (board[i][j] == word.charAt(0) && this.wordSearch(used, word, board, i, j, 0)) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean wordSearch(boolean[][] used, String word, char[][] board, int i, int j, int k) {
        if (k == word.length()) {
            return true;
        }
        if (i < 0 || i >= board.length || j < 0 || j >= board[i].length || used[i][j] || word.charAt(k) != board[i][j]) {
            return false;
        }
        used[i][j] = true;
        if (this.wordSearch(used, word, board, i - 1, j, k + 1) ||
                this.wordSearch(used, word, board, i + 1, j, k + 1) ||
                this.wordSearch(used, word, board, i, j - 1, k + 1) ||
                this.wordSearch(used, word, board, i, j + 1, k + 1)) {
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
            } else {
                count = 1;
            }
            if (count > 2) {
                continue;
            }
            nums[index++] = nums[i];
        }
        return index;
    }

    /**
     * todo 需考虑更多接发
     * 81. Search in Rotated Sorted Array II
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
                if (target < nums[mid] && target >= nums[left]) {
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
        return nums[left] == target;
    }

    /**
     * 82. Remove Duplicates from Sorted List II
     *
     * @param head
     * @return
     */
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        if (head.val == head.next.val) {
            ListNode currNode = head.next;
            while (currNode != null && currNode.val == head.val) {
                currNode = currNode.next;
            }
            return this.deleteDuplicates(currNode);
        } else {
            head.next = this.deleteDuplicates(head.next);
            return head;
        }

    }

    /**
     * 83. Remove Duplicates from Sorted List
     *
     * @param head
     * @return
     */
    public ListNode deleteDuplicatesII(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        if (head.val == head.next.val) {
            ListNode currNode = head.next;
            return this.deleteDuplicatesII(currNode);
        } else {
            head.next = this.deleteDuplicatesII(head.next);
            return head;
        }
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
                int index = stack.pop();

                int side = stack.empty() ? i : i - stack.peek() - 1;

                result = Math.max(result, heights[index] * side);

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
        int result = 0;

        int row = matrix.length;

        int column = matrix[0].length;

        int[] height = new int[column];

        int[] left = new int[column];

        int[] right = new int[column];
        for (int i = 0; i < column; i++) {
            right[i] = column;
        }
        for (int i = 0; i < row; i++) {

            int leftEdge = 0;

            int rightEdge = column;


            for (int j = 0; j < column; j++) {

                if (matrix[i][j] == '0') {
                    height[j] = 0;
                } else {
                    height[j]++;
                }
                if (matrix[i][j] == '1') {
                    left[j] = Math.max(left[j], leftEdge);
                } else {
                    left[j] = 0;
                    leftEdge = j + 1;
                }
            }
            for (int j = column - 1; j >= 0; j--) {
                if (matrix[i][j] == '1') {
                    right[j] = Math.min(right[j], rightEdge);
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
        if (head == null || head.next == null) {
            return head;
        }
        ListNode node1 = new ListNode(0);
        ListNode d1 = node1;

        ListNode node2 = new ListNode(0);
        ListNode d2 = node2;

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
        d1.next = node2.next;
        return node1.next;
    }


    /**
     * todo 可以考虑使用动态规划
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
            hash[s1.charAt(i) - '0']--;
            hash[s2.charAt(i) - '0']++;
        }
        for (int i = 0; i < hash.length; i++) {
            if (hash[i] != 0) {
                return false;
            }
        }
        for (int i = 1; i < m; i++) {
            if (this.isScramble(s1.substring(0, i), s2.substring(0, i)) &&
                    this.isScramble(s1.substring(i), s2.substring(i))) {
                return true;
            }
            if (this.isScramble(s1.substring(i), s2.substring(0, m - i)) && this.isScramble(s1.substring(0, i), s2.substring(m - i))) {
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
     * todo 格雷码
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

        Arrays.sort(nums);

        List<List<Integer>> ans = new ArrayList<>();

        this.subsetsWithDup(ans, new ArrayList<Integer>(), 0, nums);
        return ans;

    }

    private void subsetsWithDup(List<List<Integer>> ans, List<Integer> integers, int start, int[] nums) {
        ans.add(new ArrayList<>(integers));
        for (int i = start; i < nums.length; i++) {
            if (i > start && nums[i] == nums[i - 1]) {
                continue;
            }
            integers.add(nums[i]);
            this.subsetsWithDup(ans, integers, i + 1, nums);
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
        if (s == null || s.isEmpty()) {
            return 0;
        }
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

        ListNode end = fast.next;

        ListNode first = slow.next;


        for (int i = 0; i <= n - m; i++) {
            ListNode tmp = first.next;

            first.next = end;


            end = first;


            first = tmp;

        }

        slow.next = end;

        return root.next;
    }


    /**
     * todo 不会
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
            return Collections.emptyList();
        }
        Stack<TreeNode> stack = new Stack<>();

        List<Integer> ans = new ArrayList<>();

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
            TreeNode root = new TreeNode(start);
            ans.add(root);
            return ans;
        }

        if (start > end) {
            ans.add(null);
            return ans;
        }

        for (int i = start; i <= end; i++) {
            List<TreeNode> leftNodes = this.generateTrees(start, i - 1);
            List<TreeNode> rightNodes = this.generateTrees(i + 1, end);
            for (TreeNode left : leftNodes) {
                for (TreeNode right : rightNodes) {

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
     * todo xuy
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
        dp[0] = dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            for (int j = 1; j <= i; j++) {
                dp[i] += dp[j - 1] * dp[i - j];
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
            return true;
        }
        Stack<TreeNode> stack = new Stack<>();
        TreeNode p = root;
        TreeNode prev = null;
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
        TreeNode p = root;
        TreeNode prev = null;
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
            int val = first.val;
            first.val = second.val;
            second.val = val;
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
