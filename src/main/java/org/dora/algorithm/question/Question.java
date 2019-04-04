package org.dora.algorithm.question;

import org.dora.algorithm.datastructe.ListNode;
import org.dora.algorithm.datastructe.Point;
import org.dora.algorithm.datastructe.TreeNode;

import java.util.*;

/**
 * @author liulu
 * @date 2019-03-17
 */
public class Question {

    /**
     * 52. N-Queens II
     *
     * @param n
     * @return
     */
    private int count = 0;

    public static void main(String[] args) {

        Question question = new Question();

        int[][] matrix = new int[][]{{0, -3}};
        question.calculateMinimumHP(matrix);

    }

    /**
     * 4. Median of Two Sorted Arrays
     *
     * @param nums1
     * @param nums2
     * @return
     */
    // todo 不懂
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int m = nums1.length;
        int n = nums2.length;
        if (m > n) {
            return findMedianSortedArrays(nums2, nums1);
        }
        int imin = 0;
        int imax = nums1.length;
        int max_left = 0;
        int min_right = 0;
        while (imin <= imax) {
            int i = (imin + imax) / 2;
            int j = (m + n + 1) / 2 - i;

            if (i < imax && nums1[i] < nums2[j - 1]) {
                imin = i + 1;
            } else if (i > 0 && nums2[i - 1] > nums2[j]) {
                imax = i - 1;
            } else {
                if (i == 0) {
                    max_left = nums2[j - 1];
                } else if (j == 0) {
                    max_left = nums1[i - 1];
                } else {
                    max_left = Math.max(nums1[i - 1], nums2[j - 1]);
                }
                if ((m + n) % 2 != 0) {
                    return max_left;
                }
                if (i == m) {
                    min_right = nums2[j];
                } else if (j == n) {
                    min_right = nums1[i];
                } else {
                    min_right = Math.min(nums2[j], nums1[i]);
                }

                return (max_left + min_right) / 2.0;
            }
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
        if (s == null || s.length() == 0) {
            return "";
        }
        int m = s.length();

        boolean[][] dp = new boolean[m][m];

        int result = 0;

        int left = 0;
        for (int i = 0; i < s.length(); i++) {
            for (int j = 0; j < i; j++) {
                if (i - j <= 2) {
                    dp[i][j] = s.charAt(i) == s.charAt(j);
                } else {
                    dp[i][j] = (s.charAt(j) == s.charAt(i) && dp[i - 1][j + 1]);
                }
                if (dp[i][j] && i - j + 1 > result) {
                    left = j;
                    result = i - j + 1;
                }
            }
        }
        if (result > 0) {
            return s.substring(left, left + result);
        }
        return s;
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
                        dp[i][j] = dp[i - 1][j] || dp[i][j - 2] || dp[i][j - 1];
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
     * @param digits
     * @return
     */
    public List<String> letterCombinations(String digits) {
        if (digits == null || digits.length() == 0) {
            return new ArrayList<>();
        }
        LinkedList<String> ans = new LinkedList<>();
        String[] map = new String[]{"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        ans.add("");
        for (int i = 0; i < digits.length(); i++) {
            int index = digits.charAt(i) - '0';
            String value = map[index];

            while (ans.peek().length() == i) {
                String str = ans.poll();
                for (char c : value.toCharArray()) {
                    ans.add(str + c);
                }

            }
        }
        return ans;
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
        PriorityQueue<ListNode> priorityQueue = new PriorityQueue<>(Comparator.comparing(o -> o.val));
        for (ListNode listNode : lists) {
            if (listNode != null) {
                priorityQueue.add(listNode);
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
        ListNode currNode = head;
        for (int i = 0; i < k; i++) {
            if (currNode == null) {
                return head;
            }
            currNode = currNode.next;
        }
        ListNode newHead = reverseListNode(head, currNode);
        head.next = reverseKGroup(currNode, k);
        return newHead;
    }

    private ListNode reverseListNode(ListNode first, ListNode last) {
        ListNode prev = last;
        while (first != last) {
            ListNode tmp = first.next;
            first.next = prev;
            prev = first;
            first = tmp;
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
    // todo 一直未搞懂
    public int divide(int dividend, int divisor) {
        if (divisor == 0) {
            return Integer.MAX_VALUE;
        }
        if (dividend == Integer.MIN_VALUE && divisor == -1) {
            return Integer.MAX_VALUE;
        }
        int sign = (dividend < 0) ^ (divisor < 0) ? -1 : 1;
        int dvd = Math.abs(dividend);
        int dvs = Math.abs(divisor);

        long result = 0;
        while (dvd >= dvs) {
            long multi = 1;
            long tmp = dvs;
            while (dvd >= (tmp << 1)) {
                multi <<= 1;
                tmp <<= 1;
            }
            result += multi;
            dvd -= tmp;
        }
        return (int) result * sign;
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
            if (nums[index - 1] < nums[index]) {
                break;
            }
            index--;
        }
        if (index == 0) {
            reverseNums(nums, 0, nums.length - 1);
        } else {
            int j = nums.length - 1;
            while (j > index - 1) {
                if (nums[j] > nums[index - 1]) {
                    break;
                }
                j--;
            }
            swap(nums, index - 1, j);
            reverseNums(nums, index, nums.length - 1);
        }
    }

    private void reverseNums(int[] nums, int start, int end) {
        if (start > end) {
            return;
        }
        for (int i = start; i <= (start + end) / 2; i++) {
            swap(nums, i, start + end - i);
        }
    }

    private void swap(int[] nums, int i, int j) {
        int tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;
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
        Stack<Integer> stack = new Stack<>();
        int left = 0;
        int result = 0;
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') {
                stack.push(i);
            } else {
                if (!stack.isEmpty() && s.charAt(stack.peek()) == '(') {
                    stack.pop();
                    if (stack.isEmpty()) {
                        result = Math.max(result, i - left);
                    } else {
                        result = Math.max(result, i - stack.peek());
                    }
                } else {
                    left = i;
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
        return nums[left] == target ? -1 : left;
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
            return new int[2];
        }
        int left = 0;
        int right = nums.length - 1;
        int[] ans = new int[2];
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < target) {
                right = mid + 1;
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
            int mid = left + (right - left) / 2;
            if (nums[mid] < target) {
                right = mid + 1;
            } else {
                right = mid;
            }
        }
        ans[1] = right;
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
            int mid = left + (right - left) >> 1;
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
     * 41. First Missing Positive
     *
     * @param nums
     * @return
     */
    public int firstMissingPositive(int[] nums) {
        if (nums == null || nums.length == 0) {
            return -1;
        }
        for (int i = 0; i < nums.length; i++) {
            while (nums[i] > 0 && nums[i] <= nums.length && nums[nums[i] - 1] != nums[i]) {
                swap(nums, nums[i - 1], i);
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
        int maxLeft = 0;
        int maxRight = 0;
        while (left < right) {
            while (left < right && height[left] == 0) {
                left++;
            }
            while (left < right && height[right] == 0) {
                right--;
            }
            int min = Math.min(height[left], height[right]);
            for (int i = left; i <= right; i++) {
                if (height[i] >= min) {
                    height[i] -= min;
                } else {
                    result += min - height[i];
                    height[i] = 0;
                }
            }

        }
        return result;
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
        for (int i = 0; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (p.charAt(j - 1) == '*') {
                    dp[i][j] = dp[i][j - 1] || (i > 0 && dp[i - 1][j]);
                } else {
                    dp[i][j] = isMatch(s, p, i, j) ? dp[i - 1][j - 1] : false;
                }
            }
        }
        return dp[m][n];
    }

    private boolean isMatch(String s, String p, int i, int j) {
        if (s.charAt(i - 1) == p.charAt(j - 1)) {
            return true;
        } else {
            return p.charAt(j - 1) == '?';
        }
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
        int curr = 0;
        for (int i = 0; i < nums.length - 1; i++) {
            furthest = Math.max(curr, i + nums[i]);
            if (i == curr) {
                step++;
                curr = furthest;
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
        if (n < 0) {
            n = -n;
            x = 1 / x;
        }
        if (x > Integer.MAX_VALUE || x < Integer.MIN_VALUE) {
            return 0;
        }
        return n % 2 == 0 ? myPow(x * x, n / 2) : x * myPow(x * x, n / 2);
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
        List<List<String>> ans = new ArrayList<>();
        solveNQueens(ans, nQueens, 0, n);
        return ans;
    }

    private void solveNQueens(List<List<String>> ans, char[][] nQueens, int row, int n) {
        if (row == n) {

            List<String> queens = construct(nQueens);

            ans.add(queens);

        }
        for (int col = 0; col < n; col++) {
            if (matchQueens(nQueens, col, row, n)) {
                nQueens[row][col] = 'Q';
                solveNQueens(ans, nQueens, row + 1, n);
                nQueens[row][col] = '.';
            }
        }
    }

    private boolean matchQueens(char[][] nQueens, int col, int row, int n) {
        for (int i = row - 1; i >= 0; i--) {
            if (nQueens[i][col] == 'Q') {
                return false;
            }
        }
        for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--) {
            if (nQueens[i][j] == 'Q') {
                return false;
            }
        }
        for (int i = row - 1, j = col + 1; i >= 0 && j < n; i--, j++) {
            if (nQueens[i][j] == 'Q') {
                return false;
            }
        }
        return true;
    }

    private List<String> construct(char[][] nQueens) {
        List<String> ans = new ArrayList<>();
        for (char[] row : nQueens) {
            ans.add(String.valueOf(row));
        }
        return ans;
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
        return totalNQueens(0, n, dp);
    }

    private int totalNQueens(int row, int n, int[] dp) {
        int result = 0;
        if (row >= n) {
            result++;
            return result;
        }
        for (int i = 0; i < n; i++) {
            if (isValid(i, row, n, dp)) {
                dp[row] = i;
                result += totalNQueens(row + 1, n, dp);
                dp[row] = -1;
            }
        }
        return result;
    }

    private boolean isValid(int column, int row, int n, int[] dp) {
        for (int i = 0; i < row; i++) {
            if (dp[i] == column || Math.abs(i - row) == Math.abs(dp[i] - column)) {
                return false;
            }
        }
        return true;
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
        List<Integer> ans = new ArrayList<>();
        int left = 0;
        int right = matrix[0].length - 1;
        int top = 0;
        int bottom = matrix.length - 1;
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
                for (int i = bottom - 1; i > top; i--) {
                    ans.add(matrix[i][left]);
                }
            }
            left++;
            right--;
            bottom--;
            top++;
        }
        return ans;
    }

    /**
     * 59. Spiral Matrix II
     *
     * @param n
     * @return
     */
    public int[][] generateMatrix(int n) {
        if (n <= 0) {
            return new int[0][0];
        }
        int[][] matrix = new int[n][n];
        int left = 0;
        int right = n - 1;
        int top = 0;
        int bottom = n - 1;
        int total = 0;
        while (total != n * n) {
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
                for (int i = bottom - 1; i > left; i--) {
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
        int[] position = new int[n + 1];
        position[0] = 1;
        int factory = 1;
        for (int i = 1; i <= n; i++) {
            factory = factory * i;
            position[i] = factory;
        }
        List<Integer> ans = new ArrayList<>();
        for (int i = 1; i <= n; i++) {
            ans.add(i);
        }
        StringBuilder stringBuilder = new StringBuilder();
        k--;
        for (int i = 0; i < n; i++) {
            int index = k / position[n - 1 - i];
            stringBuilder.append(ans.get(index));
            ans.remove(index);
            k -= index * position[n - 1 - i];
        }
        return stringBuilder.toString();
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
        ListNode fast = head;
        ListNode slow = head;
        int count = 1;
        while (fast.next != null) {
            fast = fast.next;
            count++;
        }
        k %= count;
        fast.next = head;
        for (int i = 0; i < count - k; i++) {
            fast = fast.next;
            slow = slow.next;
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
        int[] dp = new int[n];
        dp[0] = 1;
        for (int i = 0; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[j] = dp[j - 1] + dp[j];
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
        int[] dp = new int[obstacleGrid[0].length];
        dp[0] = 1;
        for (int i = 0; i < obstacleGrid.length; i++) {
            for (int j = 0; j < obstacleGrid[0].length; j++) {
                if (obstacleGrid[i][j] == 1) {
                    dp[j] = 0;
                } else {
                    dp[j] = dp[j] + (j > 0 ? dp[j - 1] : 0);
                }
            }
        }
        return dp[obstacleGrid[0].length - 1];
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

    /**
     * 66. Plus One
     *
     * @param digits
     * @return
     */
    public int[] plusOne(int[] digits) {
        if (digits == null || digits.length == 0) {
            return new int[0];
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
            int line = 0;
            int endIndex = startIndex;
            while (endIndex < words.length && line + words[endIndex].length() <= maxWidth) {
                line += words[endIndex].length() + 1;
                endIndex++;
            }
            boolean isLast = endIndex == words.length;
            int countOfWord = endIndex - startIndex;
            StringBuilder stringBuilder = new StringBuilder();
            if (countOfWord == 1) {
                stringBuilder.append(words[startIndex]);
            } else {
                int blankWord = isLast ? 1 : 1 + (maxWidth - line + 1) / (countOfWord - 1);
                int extraWord = isLast ? 0 : (maxWidth - line + 1) % (countOfWord - 1);
                construct(words, stringBuilder, blankWord, extraWord, startIndex, endIndex);
            }
            ans.add(adjust(stringBuilder, maxWidth));
            startIndex = endIndex;
        }
        return ans;
    }

    private void construct(String[] words, StringBuilder stringBuilder, int blankWord, int extraWord, int startIndex, int endIndex) {
        for (int i = startIndex; i < endIndex; i++) {
            stringBuilder.append(words[i]);
            int tmp = blankWord;
            while (tmp-- > 0) {
                stringBuilder.append(" ");
            }
            if (extraWord-- > 0) {
                stringBuilder.append(" ");
            }
        }
    }


    private String adjust(StringBuilder sb, int maxWidth) {
        String stringBuilder = sb.toString();

        while (stringBuilder.length() < maxWidth) {
            stringBuilder = stringBuilder + " ";
        }
        while (stringBuilder.length() > maxWidth) {
            stringBuilder = stringBuilder.substring(0, stringBuilder.length() - 1);
        }
        return stringBuilder;

    }

    /**
     * 69. Sqrt(x)
     *
     * @param x
     * @return
     */
    public int mySqrt(int x) {
        double precision = 0.00001;
        double ans = x;
        while ((ans * ans - x) > precision) {
            ans = (ans + x / ans) / 2;
        }
        return (int) ans;
    }

    /**
     * 70. Climbing Stairs
     *
     * @param n
     * @return
     */
    public int climbStairs(int n) {
        if (n == 0) {
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
        if (path == null || path.length() == 0) {
            return "";
        }
        Stack<String> stack = new Stack<>();
        String[] strings = path.split("/");
        Set<String> skip = new HashSet<>(Arrays.asList("..", ".", ""));
        for (String node : strings) {
            if (skip.contains(node)) {
                if ("..".equals(node)) {
                    if (!stack.isEmpty()) {
                        stack.pop();
                    }
                }
            } else {
                stack.push(node);
            }
        }
        if (stack.isEmpty()) {
            return "/";
        }
        StringBuilder stringBuilder = new StringBuilder();
        for (String node : stack) {
            stringBuilder = stringBuilder.append("/").append(node);
        }
        return stringBuilder.toString();
    }

    /**
     * 72. Edit Distance
     *
     * @param word1
     * @param word2
     * @return
     */
    public int minDistance(String word1, String word2) {
        int m = word1.length();
        int n = word2.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (i == 0 && j == 0) {
                    dp[i][j] = 0;
                } else if (i == 0) {
                    dp[i][j] = j;
                } else if (j == 0) {
                    dp[i][j] = i;
                } else {
                    if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                        dp[i][j] = dp[i - 1][j - 1];
                    } else {
                        dp[i][j] = 1 + Math.min(Math.min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]);
                    }
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
        if (matrix == null) {
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
                        fc = true;
                    }
                    if (j == 0) {
                        fr = true;
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
            for (int i = 0; i < matrix.length; i++) {
                matrix[i][0] = 0;
            }
        }
        if (fc) {
            for (int j = 0; j < matrix[0].length; j++) {
                matrix[0][j] = 0;
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
        for (Character character : t.toCharArray()) {
            hash[character - 'a']++;
        }
        int endIndex = 0;
        int count = t.length();
        int begin = 0;
        int distance = Integer.MAX_VALUE;
        int head = 0;
        while (endIndex < s.length()) {
            while (hash[s.charAt(endIndex++) - 'a']-- > 0) {
                count--;
            }
            while (count == 0) {
                if (endIndex - begin < distance) {
                    head = begin;
                    distance = endIndex - head;
                }
                if (hash[s.charAt(begin++)]++ > 0) {
                    count++;
                }
            }
        }
        return distance == Integer.MAX_VALUE ? "" : s.substring(head, head + distance);
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
        int value = nums[0];
        int count = 1;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] == value) {
                count++;
                if (count > 2) {
                    continue;
                }
            } else {
                value = nums[i];
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
            if (nums[mid] == nums[left]) {
                left++;
            } else if (nums[mid] > nums[left]) {
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
        return false;
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
            ListNode currNode = head.next.next;
            while (currNode != null && currNode.val == head.val) {
                currNode = currNode.next;
            }
            return deleteDuplicates(currNode);
        } else {
            head.next = deleteDuplicates(head.next);
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
            return deleteDuplicatesII(head.next);
        } else {
            head.next = deleteDuplicatesII(head.next);
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
            if (stack.isEmpty() || heights[stack.peek()] < h) {
                stack.push(i);
            } else {
                int index = stack.pop();
                int size = stack.isEmpty() ? i : i - stack.peek() - 1;
                result = Math.max(result, size * heights[index]);
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
        int result = 0;

        int m = matrix.length;
        int n = matrix[0].length;
        int[] left = new int[n];
        int[] right = new int[n];
        for (int j = 0; j < n; j++) {
            right[j] = n;
        }
        int[] height = new int[n];
        for (int i = 0; i < m; i++) {
            int maxLeft = 0;
            int minRight = n;
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == '1') {
                    height[j]++;
                } else {
                    height[j] = 0;
                }
            }

            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == '1') {
                    left[j] = Math.max(left[j], maxLeft);
                } else {
                    left[j] = 0;
                    maxLeft = j + 1;
                }
            }
            for (int j = n - 1; j >= 0; j--) {
                if (matrix[i][j] == '1') {
                    right[j] = Math.min(right[j], minRight);
                } else {
                    right[j] = n;
                    minRight = j;
                }
            }
            for (int j = 0; j < n; j++) {
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
        int[] hash = new int[26];
        for (int i = 0; i < s1.length(); i++) {
            hash[s1.charAt(i) - 'a']++;
            hash[s2.charAt(i) - 'a']--;
        }
        for (int i = 0; i < 26; i++) {
            if (hash[i] != 0) {
                return false;
            }
        }
        int m = s1.length();
        for (int i = 1; i < m; i++) {
            if (isScramble(s1.substring(0, i), s2.substring(0, i)) && isScramble(s1.substring(i), s2.substring(i))) {
                return true;
            }
            if (isScramble(s1.substring(0, i), s2.substring(m - i)) && isScramble(s1.substring(i), s2.substring(0, m - i))) {
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
        while (m > 0 && n > 0) {
            if (nums1[m - 1] > nums2[n - 1]) {
                nums1[k--] = nums1[m - 1];
                m--;
            } else {
                nums1[k--] = nums2[n - 1];
                n--;
            }
        }
        while (n > 0) {
            nums1[k--] = nums2[n - 1];
            n--;
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
        Arrays.sort(nums);
        List<List<Integer>> ans = new ArrayList<>();
        subsetsWithDup(ans, new ArrayList<>(), 0, nums);
        return ans;
    }

    private <E> void subsetsWithDup(List<List<Integer>> ans, List<Integer> tmp, int start, int[] nums) {
        ans.add(new ArrayList<>(tmp));
        for (int i = start; i < nums.length; i++) {
            if (i > start && nums[i] == nums[i - 1]) {
                continue;
            }
            tmp.add(nums[i]);
            subsetsWithDup(ans, tmp, i + 1, nums);
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
        if (head == null) {
            return null;
        }
        if (m == n) {
            return head;
        }
        ListNode root = new ListNode(0);
        root.next = head;
        ListNode fast = root;
        for (int i = 0; i < m; i++) {
            fast = fast.next;
        }
        ListNode slow = root;
        for (int i = 0; i < m - 1; i++) {
            slow = slow.next;
        }
        ListNode pre = fast.next;
        ListNode start = slow.next;
        for (int i = 0; i <= n - m; i++) {
            ListNode tmp = start.next;
            start.next = pre;
            pre = start;
            start = tmp;
        }
        slow.next = pre;
        return root.next;
    }

    /**
     * 93. Restore IP Addresses
     *
     * @param s
     * @return
     */
    public List<String> restoreIpAddresses(String s) {
        if (s.length() < 12) {
            return new ArrayList<>();
        }
        List<String> ans = new ArrayList<>();
        for (int a = 1; a <= 3; a++) {
            for (int b = 1; b <= 3; b++) {
                for (int c = 1; c <= 3; c++) {
                    for (int d = 1; d <= 3; d++) {
                        int sum = a + b + c + d;
                        if (sum == s.length()) {
                            String s1 = s.substring(0, a);
                            String s2 = s.substring(a, b);
                            String s3 = s.substring(a + b, c);
                            String s4 = s.substring(a + b + c, d);
                            if (isValid(s1) && isValid(s2) && isValid(s3) && isValid(s4)) {
                                ans.add(s1 + "." + s2 + "." + s3 + "." + s4);
                            }
                        }
                    }
                }
            }
        }
        return ans;
    }

    private boolean isValid(String s) {
        if (s.length() > 3 || s.length() == 0 || (s.charAt(0) == '0' && s.length() > 1) || Integer.parseInt(s) > 255) {
            return false;
        }
        return true;
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
        return generateTrees(1, n);
    }

    private List<TreeNode> generateTrees(int start, int end) {
        List<TreeNode> listNode = new ArrayList<>();
        if (start == end) {
            listNode.add(new TreeNode(start));
            return listNode;
        }
        if (start > end) {
            listNode.add(null);
            return listNode;
        }
        for (int i = start; i <= end; i++) {
            List<TreeNode> lefts = generateTrees(start, i - 1);

            List<TreeNode> rights = generateTrees(i + 1, end);
            for (TreeNode left : lefts) {
                for (TreeNode right : rights) {
                    TreeNode root = new TreeNode(i);
                    root.left = left;
                    root.right = right;
                    listNode.add(root);
                }
            }
        }
        return listNode;
    }

    /**
     * 96. Unique Binary Search Trees
     *
     * @param n
     * @return
     */
    public int numTrees(int n) {
        if (n == 0) {
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
     * 97. Interleaving String
     *
     * @param s1
     * @param s2
     * @param s3
     * @return
     */
    public boolean isInterleave(String s1, String s2, String s3) {
        int m = s1.length();
        int n = s2.length();
        boolean[][] dp = new boolean[m + 1][n + 1];
        for (int i = 0; i <= m; i++) {
            for (int j = 0; j <= n; j++) {
                if (i == 0 && j == 0) {
                    dp[i][j] = true;
                } else if (i == 0) {
                    dp[i][j] = dp[i][j - 1] && s2.charAt(j - 1) == s3.charAt(i + j - 1);
                } else if (j == 0) {
                    dp[i][j] = dp[i - 1][j] && s1.charAt(i - 1) == s3.charAt(i + j - 1);
                } else {
                    dp[i][j] = (dp[i - 1][j] && s1.charAt(i - 1) == s3.charAt(i + j - 1)) || (dp[i][j - 1] && s2.charAt(j - 1) == s3.charAt(i + j - 1));
                }
            }
        }
        return dp[m][n];
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
        TreeNode dummy = root;
        TreeNode prev = null;
        while (!stack.isEmpty() || dummy != null) {
            while (dummy != null) {
                stack.push(dummy);
                dummy = dummy.left;
            }
            dummy = stack.pop();
            if (prev == null) {
                prev = dummy;
            } else {
                if (prev.val >= dummy.val) {
                    return false;
                }
                prev = dummy;
            }
            dummy = dummy.right;
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
        TreeNode p = root;
        TreeNode first = null;
        TreeNode second = null;
        TreeNode prev = null;
        while (!stack.isEmpty() || p != null) {
            while (p != null) {
                stack.push(p);
                p = p.left;
            }
            p = stack.pop();
            if (prev == null) {
                prev = p;
            } else {
                if (first == null && prev.val >= p.val) {
                    first = prev;
                }
                if (first != null && prev.val >= p.val) {
                    second = p;
                }
                prev = p;
            }
            p = p.right;
        }
        if (first != null && second != null) {
            int val = first.val;
            first.val = second.val;
            second.val = val;
        }
    }

    /**
     * 145. Binary Tree Postorder Traversal
     *
     * @param root
     * @return
     */
    public List<Integer> postorderTraversal(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        LinkedList<Integer> ans = new LinkedList<>();
        TreeNode p = root;
        Stack<TreeNode> stack = new Stack<>();
        while (!stack.isEmpty() || p != null) {
            if (p != null) {
                ans.addFirst(p.val);
                stack.push(p);
                p = p.right;
            } else {
                p = stack.pop();
                p = p.left;
            }
        }
        return ans;
    }

    /**
     * 103. Binary Tree Zigzag Level Order Traversal
     *
     * @param root
     * @return
     */
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        List<List<Integer>> ans = new ArrayList<>();
        boolean left = true;
        LinkedList<TreeNode> list = new LinkedList<>();
        list.add(root);
        while (!list.isEmpty()) {
            LinkedList<Integer> tmp = new LinkedList<>();
            int size = list.size();
            for (int i = 0; i < size; i++) {
                TreeNode p = list.pop();
                if (p.left != null) {
                    list.add(p.left);
                }
                if (p.right != null) {
                    list.add(p.right);
                }
                if (left) {
                    tmp.addLast(p.val);
                } else {
                    tmp.addFirst(p.val);
                }
            }
            left = !left;
            ans.add(tmp);
        }
        return ans;
    }

    /**
     * 105. Construct Binary Tree from Preorder and Inorder Traversal
     *
     * @param preorder
     * @param inorder
     * @return
     */
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        if (preorder == null || inorder == null) {
            return null;
        }
        return buildTree(0, 0, inorder.length - 1, preorder, inorder);
    }

    private TreeNode buildTree(int preStart, int inStart, int inEnd, int[] preorder, int[] inorder) {
        if (preStart >= preorder.length || inStart > inEnd) {
            return null;
        }
        TreeNode root = new TreeNode(preorder[preStart]);
        int index = 0;
        for (int i = inStart; i <= inEnd; i++) {
            if (inorder[i] == root.val) {
                index = i;
                break;
            }
        }
        root.left = buildTree(preStart + 1, inStart, index - 1, preorder, inorder);
        root.right = buildTree(preStart + index - inStart + 1, index + 1, inEnd, preorder, inorder);
        return root;
    }

    /**
     * 106. Construct Binary Tree from Inorder and Postorder Traversal
     *
     * @param inorder
     * @param postorder
     * @return
     */
    public TreeNode buildTreeII(int[] inorder, int[] postorder) {
        if (inorder == null || postorder == null) {
            return null;
        }
        return buildPostOrder(0, inorder.length - 1, 0, postorder.length - 1, inorder, postorder);
    }

    private TreeNode buildPostOrder(int inStart, int inEnd, int postStart, int postEnd, int[] inorder, int[] postorder) {
        if (inStart > inEnd || postStart > postEnd) {
            return null;
        }
        TreeNode root = new TreeNode(postorder[postEnd]);
        int index = 0;
        for (int i = inStart; i <= inEnd; i++) {
            if (inorder[i] == root.val) {
                index = i;
                break;
            }
        }
        root.left = buildPostOrder(inStart, index - 1, postStart, postStart + (index - inStart) - 1, inorder, postorder);
        root.right = buildPostOrder(index + 1, inEnd, postStart + index - inStart, postEnd - 1, inorder, postorder);
        return root;
    }

    /**
     * 108. Convert Sorted Array to Binary Search Tree
     *
     * @param nums
     * @return
     */
    public TreeNode sortedArrayToBST(int[] nums) {
        if (nums == null || nums.length == 0) {
            return null;
        }
        return buildBst(0, nums.length - 1, nums);
    }

    private TreeNode buildBst(int start, int end, int[] nums) {
        if (start > end) {
            return null;
        }
        int mid = start + (end - start) / 2;
        TreeNode root = new TreeNode(nums[mid]);
        root.left = buildBst(start, mid - 1, nums);
        root.right = buildBst(mid + 1, end, nums);
        return root;
    }

    /**
     * 109. Convert Sorted List to Binary Search Tree
     *
     * @param head
     * @return
     */
    public TreeNode sortedListToBST(ListNode head) {
        if (head == null) {
            return null;
        }
        return buildSortedList(head, null);
    }

    private TreeNode buildSortedList(ListNode head, ListNode tail) {
        if (head == tail) {
            return null;
        }
        ListNode fast = head;
        ListNode slow = head;
        while (fast != tail && fast.next != tail) {
            fast = fast.next.next;
            slow = slow.next;
        }
        TreeNode root = new TreeNode(slow.val);
        root.left = buildSortedList(head, slow);
        root.right = buildSortedList(slow.next, tail);
        return root;
    }


    /**
     * 113. Path Sum II
     *
     * @param root
     * @param sum
     * @return
     */
    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        if (root == null) {
            return new ArrayList<>();
        }
        List<List<Integer>> ans = new ArrayList<>();
        pathSum(ans, new ArrayList<Integer>(), root, sum);
        return ans;
    }

    private void pathSum(List<List<Integer>> ans, List<Integer> integers, TreeNode root, int sum) {
        integers.add(root.val);
        if (root.left == null && root.right == null && root.val == sum) {
            ans.add(new ArrayList<>(integers));
        } else {
            if (root.left != null) {
                pathSum(ans, integers, root.left, sum - root.val);
            }
            if (root.right != null) {
                pathSum(ans, integers, root.right, sum - root.val);
            }
        }
        integers.remove(integers.size() - 1);
    }

    /**
     * 114. Flatten Binary Tree to Linked List
     *
     * @param root
     */
    public void flatten(TreeNode root) {
        if (root == null) {
            return;
        }
        Stack<TreeNode> stack = new Stack<>();
        TreeNode prev = null;
        stack.push(root);
        while (!stack.isEmpty()) {

            TreeNode node = stack.pop();
            if (node.right != null) {
                stack.push(node.right);
            }

            if (node.left != null) {
                stack.push(node.left);
            }

            if (prev == null) {
                prev = node;
            } else {
                prev.right = node;

                prev = node;

            }
            prev.left = null;
        }
    }

    /**
     * 115. Distinct Subsequences
     *
     * @param s
     * @param t
     * @return
     */
    public int numDistinct(String s, String t) {
        if (s == null || t == null) {
            return 0;
        }
        int m = s.length();
        int n = t.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 0; i <= m; i++) {
            dp[i][0] = 1;
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                dp[i][j] = (s.charAt(i - 1) == t.charAt(j - 1) ? dp[i - 1][j - 1] : 0) + dp[i - 1][j];
            }
        }
        return dp[m][n];
    }


    /**
     * 118. Pascal's Triangle
     *
     * @param numRows
     * @return
     */
    public List<List<Integer>> generate(int numRows) {
        if (numRows <= 0) {
            return new ArrayList<>();
        }
        List<List<Integer>> ans = new ArrayList<>();
        for (int i = 0; i < numRows; i++) {
            List<Integer> tmp = new ArrayList<>();
            tmp.add(1);
            for (int j = 1; j < i; j++) {
                int sum = ans.get(i - 1).get(j - 1) + ans.get(i - 1).get(j);
                tmp.add(sum);
            }
            if (i > 0) {
                tmp.add(1);
            }
            ans.add(tmp);
        }
        return ans;
    }

    /**
     * 119. Pascal's Triangle II
     *
     * @param rowIndex
     * @return
     */
    public List<Integer> getRow(int rowIndex) {
        if (rowIndex < 0) {
            return new ArrayList<>();
        }
        List<Integer> ans = new ArrayList<>();
        ans.add(1);

        for (int i = 0; i <= rowIndex; i++) {

            for (int j = i - 1; j >= 1; j--) {
                ans.set(j, ans.get(j - 1) + ans.get(j));
            }
            if (i > 0) {
                ans.add(1);
            }
        }
        return ans;
    }

    /**
     * 120. Triangle
     *
     * @param triangle
     * @return
     */
    public int minimumTotal(List<List<Integer>> triangle) {
        if (triangle.isEmpty()) {
            return 0;
        }
        int size = triangle.size();

        List<Integer> ans = triangle.get(size - 1);

        for (int i = size - 2; i >= 0; i--) {

            for (int j = 0; j < triangle.get(i).size(); j++) {

                int sum = Math.min(ans.get(j), ans.get(j + 1)) + triangle.get(i).get(j);

                ans.set(j, sum);
            }
        }
        return ans.get(0);
    }

    /**
     * 123. Best Time to Buy and Sell Stock III
     *
     * @param prices
     * @return
     */
    public int maxProfitIII(int[] prices) {
        if (prices == null || prices.length == 0) {
            return 0;
        }
        int n = prices.length;
        int[] left = new int[n];
        int minLeft = prices[0];
        int max = 0;
        for (int i = 1; i < n; i++) {
            if (prices[i] < minLeft) {
                minLeft = prices[i];
            }
            max = Math.max(max, prices[i] - minLeft);
            left[i] = max;
        }
        int[] right = new int[n + 1];
        int minRight = prices[n - 1];
        int min = 0;
        for (int i = n - 2; i >= 0; i--) {
            if (prices[i] > minRight) {
                minRight = prices[i];
            }
            min = Math.max(min, minRight - prices[i]);
            right[i] = min;
        }
        int result = 0;
        for (int i = 0; i < n; i++) {
            result = Math.max(result, left[i] + right[i + 1]);
        }
        return result;
    }

    /**
     * 128. Longest Consecutive Sequence
     *
     * @param nums
     * @return
     */
    public int longestConsecutive(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int result = Integer.MIN_VALUE;

        HashMap<Integer, Integer> map = new HashMap<>();

        for (int num : nums) {
            if (map.containsKey(num)) {
                continue;
            }
            int left = map.getOrDefault(num - 1, 0);
            int right = map.getOrDefault(num + 1, 0);
            int sum = 1 + left + right;
            result = Math.max(result, sum);
            map.put(num, sum);
            map.put(num - left, sum);
            map.put(num + right, sum);
        }
        return result;
    }

    /**
     * 129. Sum Root to Leaf Numbers
     *
     * @param root
     * @return
     */
    public int sumNumbers(TreeNode root) {
        if (root == null) {
            return 0;
        }

        if (root.left == null && root.right == null) {
            return root.val;
        }
        return dfs(root.left, root.val) + dfs(root.right, root.val);
    }

    private int dfs(TreeNode root, int val) {
        if (root == null) {
            return 0;
        }
        if (root.left == null && root.right == null) {
            return val * 10 + root.val;
        }
        return dfs(root.left, val * 10 + root.val) + dfs(root.right, val * 10 + root.val);
    }

    /**
     * 130. Surrounded Regions
     *
     * @param board
     */
    public void solve(char[][] board) {
        if (board == null || board.length == 0) {
            return;
        }
        int[][] matrix = new int[][]{{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        LinkedList<Point> list = new LinkedList<>();
        return;
    }


    /**
     * 131. Palindrome Partitioning
     *
     * @param s
     * @return
     */
    public List<List<String>> partition(String s) {
        if (s == null || s.length() == 0) {
            return new ArrayList<>();
        }
        List<List<String>> ans = new ArrayList<>();
        partition(ans, new ArrayList<String>(), 0, s);
        return ans;
    }

    private void partition(List<List<String>> ans, List<String> tmp, int left, String s) {
        if (left == s.length()) {
            ans.add(new ArrayList<>(tmp));
            return;
        }
        for (int i = left; i < s.length(); i++) {
            if (isValid(s, left, i)) {
                tmp.add(s.substring(left, i + 1));
                partition(ans, tmp, i + 1, s);
                tmp.remove(tmp.size() - 1);
            }
        }
    }

    private boolean isValid(String s, int start, int end) {
        if (start > end) {
            return false;
        }
        while (start < end) {
            if (s.charAt(start) == s.charAt(end)) {
                start++;
                end--;
            } else {
                return false;
            }
        }
        return true;
    }

    /**
     * 134. Gas Station
     *
     * @param gas
     * @param cost
     * @return
     */
    public int canCompleteCircuit(int[] gas, int[] cost) {
        if (gas == null || cost == null) {
            return 0;
        }
        int result = 0;
        int total = 0;
        int index = 0;
        for (int i = 0; i < gas.length; i++) {
            total += gas[i] - cost[i];
            result += gas[i] - cost[i];
            if (total < 0) {
                index = i + 1;
                total = 0;
            }
        }
        return result < 0 ? -1 : index;
    }


    /**
     * 135. Candy
     *
     * @param ratings
     * @return
     */
    public int candy(int[] ratings) {
        if (ratings == null || ratings.length == 0) {
            return 0;
        }
        int n = ratings.length;
        int[] dp = new int[n];
        for (int i = 0; i < n; i++) {
            dp[i] = 1;
        }
        for (int i = 1; i < n; i++) {
            if (ratings[i] > ratings[i - 1] && dp[i] < 1 + dp[i - 1]) {
                dp[i] = 1 + dp[i - 1];
            }
        }
        for (int i = n - 2; i >= 0; i--) {
            if (ratings[i] > ratings[i + 1] && dp[i] < 1 + dp[i + 1]) {
                dp[i] = 1 + dp[i + 1];
            }
        }
        int result = 0;
        for (int i = 0; i < n; i++) {
            result += dp[i];
        }
        return result;
    }

    /**
     * 139. Word Break
     *
     * @param s
     * @param wordDict
     * @return
     */
    public boolean wordBreak(String s, List<String> wordDict) {
        if (s == null || s.length() == 0) {
            return false;
        }
        return wordBreak(new HashSet<String>(), s, wordDict);
    }

    private boolean wordBreak(HashSet<String> notIncluded, String s, List<String> wordDict) {
        if (wordDict.contains(s)) {
            return true;
        }
        if (notIncluded.contains(s)) {
            return false;
        }
        for (String word : wordDict) {
            if (s.startsWith(word) && wordBreak(notIncluded, s.substring(word.length()), wordDict)) {
                return true;
            }
        }
        notIncluded.add(s);
        return false;
    }

    /**
     * 140. Word Break II
     *
     * @param s
     * @param wordDict
     * @return
     */
    public List<String> wordBreakII(String s, List<String> wordDict) {
        if (s == null || wordDict.isEmpty()) {
            return new ArrayList<>();
        }
        return wordBreakDFS(s, wordDict, new HashMap<>());
    }

    private List<String> wordBreakDFS(String s, List<String> wordDict, HashMap<String, LinkedList<String>> hashMap) {
        if (hashMap.containsKey(s)) {
            return hashMap.get(s);
        }
        LinkedList<String> ans = new LinkedList<>();
        if (s.length() == 0) {
            ans.add("");
            return ans;
        }
        for (String word : wordDict) {
            if (s.startsWith(word)) {
                List<String> tmp = wordBreakDFS(s.substring(word.length()), wordDict, hashMap);

                for (String value : tmp) {
                    ans.add(word + (value.isEmpty() ? "" : " ") + value);

                }
            }
        }
        hashMap.put(s, ans);
        return ans;
    }

    /**
     * 143. Reorder List
     *
     * @param head
     */
    public void reorderList(ListNode head) {
        if (head == null || head.next == null) {
            return;
        }
        ListNode fast = head;
        ListNode slow = head;
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }

        ListNode middle = slow.next;

    }

    /**
     * 148. Sort List
     *
     * @param head
     * @return
     */
    public ListNode sortList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode fast = head;
        ListNode slow = head;
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        ListNode tmp = slow.next;
        slow.next = null;
        ListNode l1 = sortList(head);
        ListNode l2 = sortList(tmp);
        return merge(l1, l2);

    }

    private ListNode merge(ListNode l1, ListNode l2) {
        ListNode root = new ListNode(0);
        ListNode dummy = root;
        while (l1 != null && l2 != null) {
            if (l1.val <= l2.val) {
                dummy.next = l1;
                l1 = l1.next;
            } else {
                dummy.next = l2;
                l2 = l2.next;
            }
            dummy = dummy.next;
        }
        if (l1 != null) {
            dummy.next = l1;
        }
        if (l2 != null) {
            dummy.next = l2;
        }
        return root.next;
    }

    /**
     * 152. Maximum Product Subarray
     *
     * @param s
     * @return
     */
    public String reverseWords(String s) {
        if (s == null || s.length() == 0) {
            return "";
        }
        s = s.trim();
        String[] strs = s.split(" ");
        StringBuilder stringBuilder = new StringBuilder();
        for (int i = strs.length - 1; i >= 0; i--) {
            if (strs[i].trim().length() == 0) {
                continue;
            }
            stringBuilder.append(strs[i]);
            if (i > 0) {
                stringBuilder.append(" ");
            }
        }
        return stringBuilder.toString();
    }

    /**
     * 152. Maximum Product Subarray
     *
     * @param nums
     * @return
     */
    public int maxProduct(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int max = nums[0];
        int min = nums[0];
        int result = nums[0];
        for (int i = 1; i < nums.length; i++) {
            int tmpMax = Math.max(Math.max(max * nums[i], min * nums[i]), nums[i]);
            int tmpMin = Math.min(Math.min(min * nums[i], max * nums[i]), nums[i]);
            result = Math.max(result, tmpMax);
            max = tmpMax;
            min = tmpMin;
        }
        return result;
    }

    /**
     * 153. Find Minimum in Rotated Sorted Array
     *
     * @param nums
     * @return
     */
    public int findMin(int[] nums) {
        if (nums == null || nums.length == 0) {
            return -1;
        }
        int left = 0;
        int right = nums.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < nums[right]) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return nums[left];
    }

    /**
     * 154. Find Minimum in Rotated Sorted Array II
     *
     * @param nums
     * @return
     */
    public int findMinII(int[] nums) {
        if (nums == null || nums.length == 0) {
            return -1;
        }
        int left = 0;
        int right = nums.length - 1;
        while (left < right) {
            if (nums[left] < nums[right]) {
                return nums[left];
            }
            int mid = left + (right - left) / 2;
            if (nums[left] < nums[mid]) {
                left = mid + 1;
            } else if (nums[left] > nums[mid]) {
                right = mid;
            } else {
                left++;
            }
        }
        return nums[left];
    }

    /**
     * 156、Binary Tree Upside Down
     *
     * @param root
     * @return
     */
    public TreeNode upsideDownBinaryTree(TreeNode root) {
        if (root == null) {
            return null;
        }
        TreeNode prev = null;
        TreeNode tmp = null;
        TreeNode curr = root;
        while (curr != null) {
            TreeNode node = curr.left;
            curr.left = tmp;

            tmp = curr.right;

            curr.right = prev;


            prev = curr;

            curr = node;
        }
        return prev;
    }

    /**
     * 159、Longest Substring with At Most Two Distinct Characters
     *
     * @param s
     * @return
     */
    public int lengthOfLongestSubstringTwoDistinct(String s) {
        if (s == null || s.length() == 0) {
            return 0;
        }
        HashMap<Character, Integer> map = new HashMap<>();
        int left = 0;
        int result = 0;
        for (int i = 0; i < s.length(); i++) {
            map.put(s.charAt(i), i);
            while (map.size() > 2) {
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
     * 160. Intersection of Two Linked Lists
     *
     * @param headA
     * @param headB
     * @return
     */
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if (headA == null || headB == null) {
            return null;
        }
        int lenA = countOfListNode(headA);
        int lenB = countOfListNode(headB);
        while (lenA > lenB) {
            headA = headA.next;
            lenA--;
        }
        while (lenA < lenB) {
            headB = headB.next;
            lenB--;
        }
        while (headA != headB) {
            headA = headA.next;
            headB = headB.next;
        }
        return headA;
    }

    private int countOfListNode(ListNode root) {
        if (root == null) {
            return 0;
        }
        int nodeCount = 0;
        while (root != null) {
            nodeCount++;

            root = root.next;
        }
        return nodeCount;
    }

    /**
     * 161、One Edit Distance
     *
     * @param s
     * @param t
     * @return
     */
    public boolean isOneEditDistance(String s, String t) {

        return false;
    }

    /**
     * 162. Find Peak Element
     *
     * @param nums
     * @return
     */
    public int findPeakElement(int[] nums) {
        if (nums == null || nums.length == 0) {
            return -1;
        }
        int left = 0;
        int right = nums.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < nums[mid + 1]) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return left;
    }

    /**
     * 163、Missing Ranges
     *
     * @param nums
     * @param lower
     * @param upper
     * @return
     */
    public List<String> findMissingRanges(int[] nums, int lower, int upper) {
        if (nums == null || nums.length == 0) {
            return new ArrayList<>();
        }
        List<String> ans = new ArrayList<>();
        int l = lower;
        for (int i = 0; i <= nums.length; i++) {
            int right = (i < nums.length && nums[i] <= upper) ? nums[i] : upper + 1;
            if (l == right) {
                l++;
            } else if (right > l) {
                String tmp = (right - l == 1) ? "" + l : ("" + l) + "->" + ("" + (right - 1));
                ans.add(tmp);
                l = right + 1;
            }
        }
        return ans;
    }

    /**
     * 852. Peak Index in a Mountain Array
     *
     * @param A
     * @return
     */
    public int peakIndexInMountainArray(int[] A) {
        if (A == null || A.length == 0) {
            return -1;
        }
        int left = 0;
        int right = A.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (A[mid] < A[mid + 1]) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return left;
    }


    /**
     * 165、Compare Version Numbers
     *
     * @param version1
     * @param version2
     * @return
     */
    public int compareVersion(String version1, String version2) {
        if (version1 == null || version2 == null) {
            return -1;
        }
        int v1 = 0;
        int v2 = 0;
        while (v1 < version1.length() || v2 < version2.length()) {
            while (v1 < version1.length() && !Character.isDigit(version1.charAt(v1))) {
                v1++;
            }
            while (v2 < version2.length() && !Character.isDigit(version2.charAt(v2))) {
                v2++;
            }
            int tmp1 = 0;
            int tmp2 = 0;
            while (v1 < version1.length() && Character.isDigit(version1.charAt(v1))) {
                tmp1 = tmp1 * 10 + (version1.charAt(v1++) - '0');
            }
            while (v2 < version2.length() && Character.isDigit(version2.charAt(v2))) {
                tmp2 = tmp2 * 10 + (version2.charAt(v2++) - '0');
            }
            if (tmp1 < tmp2) {
                return -1;
            } else if (tmp1 > tmp2) {
                return 1;
            }
        }
        return 0;
    }

    /**
     * 174. Dungeon Game
     *
     * @param dungeon
     * @return
     */
    public int calculateMinimumHP(int[][] dungeon) {
        if (dungeon == null || dungeon.length == 0) {
            return 0;
        }
        int row = dungeon.length;

        int column = dungeon[0].length;

        int[] dp = new int[column + 1];

        dp[column - 1] = 1;

        dp[column] = 1;

        /**
         * the last row initialization
         * the last row is specially
         * so we calculate the row alone;
         */
        for (int j = column - 1; j >= 0; j--) {
            dp[j] = Math.max(1, dp[j + 1] - dungeon[row - 1][j]);
        }
        /**
         * traverse the matrix
         * from the row - 2 to 0 row;
         */
        for (int i = row - 1; i >= 0; i--) {
            for (int j = column - 1; j >= 0; j--) {
                /**
                 * if the column is the right edge
                 */
                if (j == column - 1) {
                    dp[j] = Math.max(1, dp[j] - dungeon[i][j]);
                } else {
                    dp[j] = Math.max(1, Math.min(dp[j], dp[j + 1]) - dungeon[i][j]);
                }
            }
        }
        return dp[0];
    }

    /**
     * 189. Rotate Array
     *
     * @param nums
     * @param k
     */
    public void rotate(int[] nums, int k) {
        if (nums == null || nums.length == 0) {
            return;
        }
        reverseNums(nums, 0, nums.length - 1);
        reverseNums(nums, 0, k - 1);
        reverseNums(nums, k, nums.length - 1);
    }

    /**
     * 198. House Robber
     *
     * @param nums
     * @return
     */
    public int rob(int[] nums) {
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
     * 199. Binary Tree Right Side View
     *
     * @param root
     * @return
     */
    public List<Integer> rightSideView(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        List<Integer> ans = new ArrayList<>();
        LinkedList<TreeNode> list = new LinkedList<>();
        list.add(root);
        while (!list.isEmpty()) {
            int size = list.size();
            for (int i = 0; i < size; i++) {
                TreeNode node = list.pollFirst();
                if (node.left != null) {
                    list.add(node.left);
                }
                if (node.right != null) {
                    list.add(node.right);
                }
                if (i == size - 1) {
                    ans.add(node.val);
                }
            }
        }
        return ans;
    }

    /**
     * 200. Number of Islands
     *
     * @param grid
     * @return
     */
    public int numIslands(char[][] grid) {
        if (grid == null || grid.length == 0) {
            return 0;
        }
        int row = grid.length;
        int column = grid[0].length;
        boolean[][] used = new boolean[row][column];
        int count = 0;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (check(used, i, j, grid)) {
                    count++;
                }
            }
        }
        return count;
    }

    private boolean check(boolean[][] used, int i, int j, char[][] grid) {
        if (i < 0 || i >= grid.length || j < 0 || j >= grid[i].length || used[i][j] || grid[i][j] != '1') {
            return false;
        }
        used[i][j] = true;
        grid[i][j] = '0';
        check(used, i - 1, j, grid);
        check(used, i + 1, j, grid);
        check(used, i, j - 1, grid);
        check(used, i, j + 1, grid);
        return true;
    }
}
