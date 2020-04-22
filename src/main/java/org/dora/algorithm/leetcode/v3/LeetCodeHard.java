package org.dora.algorithm.leetcode.v3;

import org.dora.algorithm.datastructe.ListNode;

import java.util.*;

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

    public static void main(String[] args) {
        LeetCodeHard hard = new LeetCodeHard();
        int[] nums = new int[]{1, 3};
        hard.searchV2(nums, 2);

    }

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
        if (p.isEmpty()) {
            return s.isEmpty();
        }
        boolean firstMatch = !s.isEmpty() && (s.charAt(0) == p.charAt(0) || p.charAt(0) == '.');

        if (s.length() >= 2 && p.charAt(1) == '*') {
            return isMatchV2(s, p.substring(2)) || (firstMatch && isMatchV2(s.substring(1), p));
        } else {
            return firstMatch && isMatchV2(s.substring(1), p.substring(1));
        }
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
        int len = digits.length();
        for (int i = 0; i < len; i++) {
            int index = Character.getNumericValue(digits.charAt(i));
            String word = map[index];
            while (deque.peek().length() == i) {
                String poll = deque.poll();
                for (char tmp : word.toCharArray()) {
                    deque.add(poll + tmp);
                }
            }
        }
        return deque;
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
            if (list != null) {
                priorityQueue.add(list);
            }
        }
        ListNode root = new ListNode(0);

        ListNode dummy = root;
        while (!priorityQueue.isEmpty()) {
            ListNode poll = priorityQueue.poll();
            dummy.next = poll;
            dummy = dummy.next;
            if (poll.next != null) {
                priorityQueue.add(poll.next);
            }
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
        if (head == null || head.next == null || k < 0) {
            return head;
        }
        int count = 0;
        ListNode currentNode = head;
        while (currentNode != null && count != k) {
            count++;
            currentNode = currentNode.next;
        }
        if (count == k) {
            ListNode node = reverseKGroup(currentNode, k);
            while (count-- > 0) {
                ListNode tmp = head.next;
                head.next = node;
                node = head;
                head = tmp;
            }
            head = node;
        }
        return head;
    }


    public ListNode reverseKGroupV2(ListNode head, int k) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode current = head;
        for (int i = 0; i < k; i++) {
            if (current == null) {
                return head;
            }
            current = current.next;
        }
        ListNode newHead = reverseListNode(head, current);

        head.next = reverseKGroup(current, k);

        return newHead;
    }

    private ListNode reverseListNode(ListNode start, ListNode end) {
        ListNode prev = end;
        while (start != end) {
            ListNode next = start.next;
            start.next = prev;
            prev = start;
            start = next;
        }
        return prev;
    }

    /**
     * 29. Divide Two Integers
     *
     * @param dividend
     * @param divisor
     * @return
     */
    public int divide(int dividend, int divisor) {
        if (dividend == 0) {
            return 0;
        }
        if (dividend == Integer.MIN_VALUE && divisor == -1) {
            return Integer.MAX_VALUE;
        }
        int sign = -1;
        if (dividend > 0 && divisor > 0) {
            sign = 1;
        }
        if (dividend < 0 && divisor < 0) {
            sign = 1;
        }
        long dvd = Math.abs((long) dividend);

        long dvs = Math.abs((long) divisor);

        long result = 0;

        while (dvd >= dvs) {
            long tmp = dvs;

            int multi = 1;
            while (dvd >= (tmp << 1)) {
                tmp <<= 1;
                multi <<= 1;
            }
            dvd -= tmp;
            result += multi;
        }
        return sign * (int) result;
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
        int left = 0;
        int right = nums.length - 1;
        while (left <= right) {
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
        return -1;
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
            if (nums[mid] < target) {
                left = mid + 1;
            } else if (nums[mid] > target) {
                right = mid - 1;
            } else {
                int first = mid;
                while (first > 0 && nums[first] == nums[first - 1]) {
                    first = first - 1;
                }
                int second = mid;
                while (second < nums.length - 1 && nums[second] == nums[second + 1]) {
                    second = second + 1;
                }
                ans[0] = first;
                ans[1] = second;
                return ans;
            }
        }
        return ans;

    }

    public int[] searchRangeV2(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return new int[]{-1, -1};
        }

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
        int[] ans = new int[]{-1, -1};
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

    public int[] searchRangeV3(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return new int[]{-1, -1};
        }
        int firstIndex = searchFirstIndex(nums, 0, nums.length - 1, target);
        if (firstIndex == -1) {
            return new int[]{-1, -1};
        }
        int secondIndex = searchSecondIndex(nums, 0, nums.length - 1, target);

        return new int[]{firstIndex, secondIndex};
    }

    private int searchSecondIndex(int[] nums, int start, int end, int target) {
        if (start > end) {
            return -1;
        }
        int mid = start + (end - start) / 2;
        if (nums[mid] < target) {
            return searchSecondIndex(nums, mid + 1, end, target);
        } else if (nums[mid] > target) {
            return searchSecondIndex(nums, start, mid - 1, target);
        } else if (nums[mid] == target) {
            if (mid + 1 < nums.length && nums[mid + 1] == target) {
                return searchSecondIndex(nums, mid + 1, end, target);
            }
            return mid;
        }
        return -1;
    }

    private int searchFirstIndex(int[] nums, int start, int end, int target) {
        if (start > end) {
            return -1;
        }
        while (start <= end) {
            int mid = start + (end - start) / 2;

            if (nums[mid] < target) {
                start = mid + 1;
            } else if (nums[mid] > target) {
                end = mid - 1;
            } else if (mid - 1 >= 0 && nums[mid - 1] == target) {
                end = mid - 1;
            } else {
                return mid;
            }
        }
        return -1;
    }


    /**
     * 41. First Missing Positive
     *
     * @param nums
     * @return
     */
    public int firstMissingPositive(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        for (int i = 0; i < nums.length; i++) {
            while (nums[i] >= 0 && nums[i] <= nums.length && nums[i] != nums[nums[i] - 1]) {
                swap(nums, i, nums[i] - 1);
            }
        }
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != i + 1) {
                return i;
            }
        }
        return nums.length + 1;
    }

    private void swap(int[] nums, int i, int j) {
        int value = nums[i];
        nums[i] = nums[j];
        nums[j] = value;
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

        int minLeft = 0;

        int minRight = 0;

        while (left < right) {
            if (height[left] <= height[right]) {
                if (height[left] > minLeft) {
                    minLeft = height[left];
                } else {
                    result += minLeft - height[left];
                }
                left++;
            } else {
                if (height[right] > minRight) {
                    minRight = height[right];
                } else {
                    result += minRight - height[right];
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
        int left = 0;
        int result = 0;
        int right = height.length - 1;
        while (left < right) {
            while (left < right && height[left] == 0) {
                left++;
            }
            while (left < right && height[right] == 0) {
                right--;
            }
            int value = Math.min(height[left], height[right]);

            for (int i = left; i <= right; i++) {
                if (height[i] > value) {
                    height[i] = height[i] - value;
                } else {
                    result += value - height[i];

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
        if (num1 == null | num2 == null) {
            return "0";
        }
        int m = num1.length();
        int n = num2.length();
        int[] pos = new int[m + n];
        for (int i = m - 1; i >= 0; i--) {
            for (int j = n - 1; j >= 0; j--) {

                int value = Character.getNumericValue(num1.charAt(i)) * Character.getNumericValue(num2.charAt(j)) + pos[i + j + 1];

                pos[i + j + 1] = value % 10;

                pos[i + j] += value / 10;
            }
        }
        StringBuilder builder = new StringBuilder();
        for (int num : pos) {
            if (!(num == 0 && builder.length() == 0)) {
                builder.append(num);
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
    public boolean isMatchV3(String s, String p) {
        if (s == null || s.isEmpty()) {
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
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
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
    public int jumpV2(int[] nums) {
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
        double result = 1.0;
        long pos = Math.abs((long) n);
        while (pos != 0) {
            if (pos % 2 != 0) {
                result *= x;
            }
            x *= x;
            pos >>= 1;
        }
        if (result > Integer.MAX_VALUE) {
            return 0;
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
        char[][] matrix = new char[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                matrix[i][j] = '.';
            }
        }
        List<List<String>> result = new ArrayList<>();
        intervalNQueens(result, matrix, 0);
        return result;
    }

    private void intervalNQueens(List<List<String>> result,
                                 char[][] matrix,
                                 int row) {
        if (row == matrix.length) {
            List<String> tmp = new ArrayList<>();
            for (char[] chars : matrix) {
                tmp.add(String.valueOf(chars));
            }
            result.add(tmp);
            return;
        }
        for (int j = 0; j < matrix[0].length; j++) {
            if (checkQueens(matrix, row, j)) {
                matrix[row][j] = 'Q';
                intervalNQueens(result, matrix, row + 1);
                matrix[row][j] = '.';
            }
        }
    }

    private boolean checkQueens(char[][] matrix, int row, int column) {
        for (int k = row - 1; k >= 0; k--) {
            if (matrix[k][column] == 'Q') {
                return false;
            }
        }
        for (int i = row - 1, j = column - 1; i >= 0 && j >= 0; i--, j--) {
            if (matrix[i][j] == 'Q') {
                return false;
            }
        }
        for (int i = row - 1, j = column + 1; i >= 0 && j < matrix[0].length; i--, j++) {
            if (matrix[i][j] == 'Q') {
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
        return intervalTotalNQueens(dp, 0, n);
    }

    private int intervalTotalNQueens(int[] dp, int row, int n) {
        int count = 0;
        if (row == n) {
            count++;
            return count;
        }
        for (int j = 0; j < dp.length; j++) {
            if (checkNQueens(dp, row, j)) {
                dp[row] = j;
                count += intervalTotalNQueens(dp, row + 1, n);
                dp[row] = -1;
            }
        }
        return count;
    }

    private boolean checkNQueens(int[] dp, int row, int column) {
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
        int local = 0;
        int result = Integer.MIN_VALUE;
        for (int num : nums) {
            local = local < 0 ? num : local + num;

            result = Math.max(result, local);
        }
        return result;
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
        PriorityQueue<int[]> priorityQueue = new PriorityQueue<>(Comparator.comparingInt(o -> o[0]));
        for (int[] interval : intervals) {
            priorityQueue.offer(interval);
        }
        LinkedList<int[]> result = new LinkedList<>();
        while (!priorityQueue.isEmpty()) {
            int[] lastIndex = result.peekLast();

            int[] poll = priorityQueue.poll();
            if (lastIndex == null || lastIndex[1] < poll[0]) {
                result.addLast(poll);
            } else {
                lastIndex[1] = Math.max(lastIndex[1], poll[1]);
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
        PriorityQueue<int[]> queue = new PriorityQueue<>(Comparator.comparingInt(o -> o[0]));
        for (int[] interval : intervals) {
            queue.offer(interval);
        }
        LinkedList<int[]> list = new LinkedList<>();
        while (!queue.isEmpty()) {
            int[] peek = queue.peek();
            if (peek[1] < newInterval[0]) {
                int[] poll = queue.poll();
                list.addLast(poll);
            } else {
                break;
            }
        }
        while (!queue.isEmpty()) {
            int[] peek = queue.peek();
            if (peek[0] <= newInterval[1]) {
                int[] poll = queue.poll();
                newInterval[0] = Math.min(newInterval[0], poll[0]);
                newInterval[1] = Math.max(newInterval[1], poll[1]);
            } else {
                break;
            }
        }
        list.add(newInterval);

        while (!queue.isEmpty()) {
            list.add(queue.poll());
        }
        return list.toArray(new int[list.size()][]);
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
        if (n <= 0 || k <= 0) {
            return "";
        }
        List<Integer> nums = new ArrayList<>();
        for (int i = 1; i <= n; i++) {
            nums.add(i);
        }
        int[] factory = new int[n + 1];
        factory[0] = factory[1] = 1;
        int base = 1;
        for (int i = 2; i <= n; i++) {
            factory[i] = base;
            base *= i;
        }
        k--;
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < n; i++) {
            int index = k / factory[n - 1 - i];
            builder.append(nums.remove(index));
            k -= index * factory[n - 1 - i];
        }
        return builder.toString();
    }


    /**
     * 61. Rotate List
     *
     * @param head
     * @param k
     * @return
     */
    public ListNode rotateRight(ListNode head, int k) {
        if (head == null) {
            return null;
        }
        ListNode current = head;
        int count = 1;
        while (current.next != null) {
            count++;
            current = current.next;
        }
        k %= count;
        if (k != 0) {
            current.next = head;
            for (int i = 0; i < count - k; i++) {
                head = head.next;
                current = current.next;
            }
            current.next = null;

        }
        return head;
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
        for (int j = 0; j < n; j++) {
            dp[j] = 1;
        }
        for (int i = 1; i < m; i++) {
            for (int j = 0; j < n; j++) {
                dp[j] = j == 0 ? dp[j] : dp[j - 1] + dp[j];
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
        int column = obstacleGrid[0].length;
        int row = obstacleGrid.length;
        int[] dp = new int[column];

        dp[0] = 1;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (obstacleGrid[i][j] == 1) {
                    dp[j] = 0;
                } else {
                    dp[j] = j == 0 ? dp[j] : dp[j - 1] + dp[j];
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
        for (int j = 0; j < column; j++) {
            dp[0][j] = j == 0 ? grid[0][0] : dp[0][j - 1] + grid[0][j];
        }
        for (int i = 1; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (j == 0) {
                    dp[i][j] = dp[i - 1][j] + grid[i][j];
                } else {
                    dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
                }
            }
        }
        return dp[row - 1][column - 1];
    }

    public int minPathSumV2(int[][] grid) {
        if (grid == null || grid.length == 0) {
            return 0;
        }
        int column = grid[0].length;
        int row = grid.length;
        int[] dp = new int[column];
        for (int j = 0; j < column; j++) {
            dp[j] = j == 0 ? grid[0][0] : dp[j - 1] + grid[0][j];
        }
        for (int i = 1; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (j == 0) {
                    dp[j] = dp[j] + grid[i][j];
                } else {
                    dp[j] = Math.min(dp[j], dp[j - 1]) + grid[i][j];
                }
            }
        }
        return dp[column - 1];
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
        boolean seenE = false;
        boolean seenNumberAfterE = true;
        boolean seenNumber = false;
        boolean seenDit = false;
        int len = s.length();
        for (int i = 0; i < len; i++) {
            char word = s.charAt(i);
            if (word >= '0' && word <= '9') {
                seenNumber = true;
                seenNumberAfterE = true;
            } else if (word == 'e' || word == 'E') {
                if (i == 0 || (!Character.isDigit(s.charAt(i - 1)) && s.charAt(i - 1) != '.')) {
                    return false;
                }
                if (!seenNumber || seenE) {
                    return false;
                }
                seenE = true;
                seenNumberAfterE = false;
            } else if (word == '-' || word == '+') {
                if (i != 0 && (s.charAt(i - 1) != 'e' && s.charAt(i - 1) != 'E')) {
                    return false;
                }
            } else if (word == '.') {
                if (seenDit || seenE) {
                    return false;
                }
                seenDit = true;
            } else {
                return false;
            }
        }
        return seenNumber && seenNumberAfterE;

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
        StringBuilder builder = new StringBuilder();
        while (m >= 0 || n >= 0 || carry > 0) {
            int value = (m >= 0 ? Character.getNumericValue(a.charAt(m--)) : 0) + (n >= 0 ? Character.getNumericValue(b.charAt(n--)) : 0) + carry;
            carry = value / 2;
            builder.append(value % 2);
        }
        StringBuilder reverse = builder.reverse();
        String result = reverse.toString();
        if (result.isEmpty() || result.startsWith("0")) {
            return "0";
        }
        return result;
    }


    /**
     * 68. Text Justification
     *
     * @param words
     * @param maxWidth
     * @return
     */
    public List<String> fullJustify(String[] words, int maxWidth) {
        if (words == null || words.length == 0 || maxWidth <= 0) {
            return new ArrayList<>();
        }
        List<String> result = new ArrayList<>();
        int startIndex = 0;
        int len = words.length;
        while (startIndex < len) {
            int endIndex = startIndex;
            int line = 0;
            while (endIndex < len && line + words[endIndex].length() <= maxWidth) {
                line += words[endIndex].length() + 1;
                endIndex++;
            }
            StringBuilder builder = new StringBuilder();
            boolean lastRow = endIndex == len;
            int currentLen = maxWidth - line + 1;
            int countOfWord = endIndex - startIndex;
            if (countOfWord == 1) {
                builder.append(words[startIndex]);
            } else {
                int blankOfWord = lastRow ? 1 : currentLen / (countOfWord - 1) + 1;
                int extraOfWord = lastRow ? 0 : currentLen % (countOfWord - 1);
                String construct = construct(startIndex, endIndex, blankOfWord, extraOfWord, words);
                builder.append(construct);
            }

            String word = trimWord(builder.toString(), maxWidth);

            result.add(word);

            startIndex = endIndex;
        }
        return result;

    }

    private String construct(int startIndex, int endIndex, int blankOfWord, int extraOfWord, String[] words) {
        StringBuilder builder = new StringBuilder();
        for (int i = startIndex; i < endIndex; i++) {

            builder.append(words[i]);
            int tmp = blankOfWord;

            while (tmp-- > 0) {
                builder.append(" ");
            }
            if (extraOfWord-- > 0) {
                builder.append(" ");
            }
        }
        return builder.toString();
    }

    private String trimWord(String word, int maxWidth) {
        while (word.length() < maxWidth) {
            word = word + " ";
        }
        while (word.length() > maxWidth) {
            word = word.substring(0, word.length() - 1);
        }
        return word;
    }


    /**
     * todo
     *
     * @param words
     * @param maxWidth
     * @return
     */
    public List<String> fullJustifyV2(String[] words, int maxWidth) {
        if (words == null || words.length == 0) {
            return new ArrayList<>();
        }
        List<String> result = new ArrayList<>();
        for (int i = 0, k; i < words.length; i = i + k) {
            int line = 0;
            for (k = 0; i + k < words.length && line + words[i + k].length() <= maxWidth - k; k++) {
                line += words[i + k].length();
            }
            StringBuilder builder = new StringBuilder();

            boolean lastRow = i + k == words.length;

            for (int j = 0; j < k; j++) {
                builder.append(words[i + j]);
                if (lastRow) {
                    builder.append(" ");
                } else {
                    int countOfBlank = 0;
                    if (k - 1 > 0) {
                        countOfBlank = (maxWidth - line) / (k - 1) + (j < (maxWidth - line) % (k - 1) ? 1 : 0);
                    }
                    while (countOfBlank-- > 0) {
                        builder.append(" ");
                    }
                }
            }
            result.add(trimWord(builder.toString(), maxWidth));
        }
        return result;

    }

    /**
     * 平方根求解
     *
     * @param x
     * @return
     */
    public int mySqrt(int x) {
        double precision = 0.00001;
        double result = x;
        while (result * result - x > precision) {
            result = (result + x / result) / 2;
        }
        return (int) result;
    }


    /**
     * @param path
     * @return
     */
    public String simplifyPath(String path) {
        if (path == null || path.isEmpty()) {
            return "/";
        }
        Deque<String> deque = new LinkedList<>();
        Set<String> skip = new HashSet<>(Arrays.asList("", "..", "."));
        String[] words = path.split("/");
        for (String word : words) {
            if (!skip.contains(word)) {
                deque.offer(word);
            } else if (!deque.isEmpty() && "..".equals(word)) {
                deque.pollLast();
            }
        }
        if (deque.isEmpty()) {
            return "/";
        }
        StringBuilder builder = new StringBuilder();
        for (String word : deque) {
            builder.append("/").append(word);
        }
        return builder.toString();

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
        for (int i = 0; i < dp.length; i++) {
            for (int j = 0; j < dp[i].length; j++) {
                if (i == 0) {
                    dp[i][j] = j;
                } else if (j == 0) {
                    dp[i][j] = i;
                } else if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = Math.min(Math.min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]) + 1;
                }

            }
        }
        return dp[m][n];
    }

    public int minDistanceV2(String word1, String word2) {
        if (word2 == null || word1 == null) {
            return 0;
        }
        int row = word1.length();
        int column = word2.length();
        int[] pre = new int[column + 1];

        for (int j = 1; j <= column; j++) {
            pre[j] = j;
        }
        int[] cur = new int[column + 1];
        for (int i = 1; i <= row; i++) {
            cur[0] = i;
            for (int j = 1; j <= column; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    cur[j] = pre[j - 1];
                } else {
                    cur[j] = Math.min(Math.min(pre[j - 1], pre[j]), cur[j - 1]) + 1;
                }
            }
            for (int k = 0; k < cur.length; k++) {
                pre[k] = cur[k];
            }
        }
        return pre[column];
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
        int blue = nums.length - 1;
        int red = 0;
        for (int i = 0; i < nums.length; i++) {
            while (nums[i] == 2 && i < blue) {
                swap(nums, i, blue--);

            }
            while (nums[i] == 0 && i > red) {
                swap(nums, i, red++);
            }
        }
    }


    public String minWindow(String s, String t) {
        if (s == null || t == null) {
            return "";
        }
        int count = t.length();
        int[] hash = new int[256];
        for (int i = 0; i < count; i++) {
            hash[t.charAt(i) - '0']++;
        }
        int result = Integer.MAX_VALUE;
        int begin = 0;
        int end = 0;
        int head = 0;
        int len = s.length();
        while (end < len) {
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
        return s;
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
                if (board[i][j] == word.charAt(0) && intervalExist(used, i, j, board, 0, word)) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean intervalExist(boolean[][] used, int i, int j, char[][] board, int index, String word) {
        if (index == word.length()) {
            return true;
        }
        if (i < 0 || i >= board.length || j < 0 || j >= board[i].length || used[i][j] || board[i][j] != word.charAt(index)) {
            return false;
        }
        used[i][j] = true;
        if (intervalExist(used, i - 1, j, board, index + 1, word) ||
                intervalExist(used, i + 1, j, board, index + 1, word) ||
                intervalExist(used, i, j - 1, board, index + 1, word) ||
                intervalExist(used, i, j + 1, board, index + 1, word)) {
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
    public int removeDuplicates(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int index = 1;
        int count = 1;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] == nums[i - 1]) {
                if (count == 2) {
                    continue;
                } else {
                    count++;
                }
            } else {
                count = 1;
            }
            nums[index++] = nums[i];
        }
        return index;
    }


    /**
     * todo
     * 81. Search in Rotated Sorted Array II
     *
     * @param nums
     * @param target
     * @return
     */
    public boolean searchV2(int[] nums, int target) {
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
            ListNode current = head.next.next;
            while (current != null && current.val == head.val) {
                current = current.next;
            }
            return deleteDuplicates(current);
        } else {
            head.next = deleteDuplicates(head.next);
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
                Integer pop = stack.pop();
                int value = heights[pop];
                int side = stack.isEmpty() ? i : i - stack.peek() - 1;
                result = Math.max(result, side * value);
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
        int column = matrix[0].length;
        int[] left = new int[column];
        int[] right = new int[column];
        Arrays.fill(right, column);
        int[] height = new int[column];
        int result = 0;
        for (char[] chars : matrix) {
            int leftSide = 0;
            int rightSide = column;
            for (int j = 0; j < column; j++) {
                char word = chars[j];
                if (word == '1') {
                    height[j]++;
                } else {
                    height[j] = 0;
                }

                if (word == '1') {
                    left[j] = Math.max(left[j], leftSide);
                } else {
                    left[j] = 0;
                    leftSide = j + 1;
                }
            }
            for (int j = column - 1; j >= 0; j--) {
                char word = chars[j];

                if (word == '1') {
                    right[j] = Math.min(right[j], rightSide);
                } else {
                    right[j] = column;
                    rightSide = j;
                }
            }

            for (int j = 0; j < column; j++) {
                result = Math.max(result, height[j] * (right[j] - left[j]));
            }
        }
        return result;
    }


    public int maximalSquare(char[][] matrix) {
        if (matrix == null || matrix.length == 0) {
            return 0;
        }
        int row = matrix.length;
        int column = matrix[0].length;
        int result = 0;
        int[][] dp = new int[row][column];
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                char word = matrix[i][j];
                if (word == '1') {
                    if (i == 0 || j == 0) {
                        dp[i][j] = 1;
                    } else {
                        dp[i][j] = Math.min(dp[i - 1][j], Math.min(dp[i][j - 1], dp[i - 1][j - 1])) + 1;
                    }
                    result = Math.max(result, dp[i][j] * dp[i][j]);
                }
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
        ListNode dummy2 = new ListNode(0);
        ListNode small = dummy1;
        ListNode big = dummy2;
        while (head != null) {
            if (head.val < x) {
                small.next = head;
                small = small.next;
            } else {
                big.next = head;
                big = big.next;
            }
            head = head.next;
        }
        big.next = null;
        small.next = dummy2.next;
        dummy2.next = null;
        return dummy1.next;

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
        int[] hash = new int[256];
        for (int i = 0; i < m; i++) {
            hash[s1.charAt(i) - '0']++;
            hash[s2.charAt(i) - 0]--;
        }
        return false;

    }


    /**
     * todo
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
        List<List<Integer>> result = new ArrayList<>();
        intervalSubsetsWithDup(result, new ArrayList<Integer>(), nums, 0);
        return result;
    }

    private void intervalSubsetsWithDup(List<List<Integer>> result, List<Integer> integers, int[] nums, int start) {
        result.add(new ArrayList<>(integers));
        for (int i = start; i < nums.length; i++) {
            if (i > start && nums[i] == nums[i - 1]) {
                continue;
            }
            integers.add(nums[i]);
            intervalSubsetsWithDup(result, integers, nums, i + 1);
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
        StringBuilder builder = new StringBuilder();
        int len = s.length();
        for (int i = 0; i < len; i++) {
            int value = Character.getNumericValue(s.charAt(i));

            int remain = value % 25;
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
        if (head == null) {
            return null;
        }
        ListNode root = new ListNode(0);
        ListNode fast = root;
        ListNode slow = root;
        for (int i = 0; i < n; i++) {
            if (fast == null) {
                return null;
            }
            fast = fast.next;
        }
        for (int i = 0; i < m - 1; i++) {
            slow = slow.next;
        }

    }


}
