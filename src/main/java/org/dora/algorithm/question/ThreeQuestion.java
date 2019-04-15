package org.dora.algorithm.question;

import org.dora.algorithm.datastructe.ListNode;
import org.dora.algorithm.datastructe.TreeNode;
import org.dora.algorithm.solution.Trie;

import java.util.*;

/**
 * @author liulu
 * @date 2019-04-05
 */
public class ThreeQuestion {

    public static void main(String[] args) {
        ThreeQuestion threeQuestion = new ThreeQuestion();
        int[] nums = new int[]{0, 1, 2, 4, 5, 7};
        threeQuestion.summaryRanges(nums);
    }

    /**
     * 201、Bitwise AND of Numbers Range
     *
     * @param m
     * @param n
     * @return
     */
    public int rangeBitwiseAnd(int m, int n) {
        return n > m ? rangeBitwiseAnd(m, n & n - 1) : n;
    }

    /**
     * 202. Happy Number
     *
     * @param n
     * @return
     */
    public boolean isHappy(int n) {
        if (n <= 0) {
            return false;
        }
        Set<Integer> used = new HashSet<>();
        while (n != 0) {
            int result = 0;
            while (n != 0) {
                int index = n % 10;
                result += index * index;
                n /= 10;
            }
            if (result == 1) {
                return true;
            }
            if (used.contains(result)) {
                return false;
            }
            used.add(result);
            n = result;
        }
        return false;
    }

    /**
     * 203. Remove Linked List Elements
     *
     * @param head
     * @param val
     * @return
     */
    public ListNode removeElements(ListNode head, int val) {
        if (head == null) {
            return null;
        }
        ListNode root = new ListNode(0);
        root.next = head;
        ListNode dummy = head;
        ListNode node = root;
        while (dummy != null) {
            if (dummy.val == val) {
                node.next = dummy.next;
            } else {
                node = node.next;
            }
            dummy = dummy.next;
        }
        return root.next;
    }

    /**
     * 205. Isomorphic Strings
     *
     * @param s
     * @param t
     * @return
     */
    public boolean isIsomorphic(String s, String t) {
        if (s == null || t == null) {
            return false;
        }
        int m = s.length();
        int n = t.length();
        if (m != n) {
            return false;
        }
        int[] hash1 = new int[256];
        int[] hash2 = new int[256];
        for (int i = 0; i < m; i++) {
            if (hash1[s.charAt(i)] != hash2[t.charAt(i)]) {
                return false;
            }
            hash1[s.charAt(i)] = i + 1;
            hash2[t.charAt(i)] = i + 1;
        }
        return true;
    }

    /**
     * 207. Course Schedule
     *
     * @param numCourses
     * @param prerequisites
     * @return
     */
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        return false;
    }

    /**
     * 209. Minimum Size Subarray Sum
     *
     * @param s
     * @param nums
     * @return
     */
    public int minSubArrayLen(int s, int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int sum = 0;
        int left = 0;
        int result = Integer.MAX_VALUE;
        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];
            while (sum >= s) {
                result = Math.min(result, i - left + 1);
                sum -= nums[left++];
            }
        }
        return result == Integer.MAX_VALUE ? 0 : result;
    }

    /**
     * 212. Word Search II
     *
     * @param board
     * @param words
     * @return
     */
    public List<String> findWords(char[][] board, String[] words) {
        if (board == null || board.length == 0) {
            return new ArrayList<>();
        }
        Trie trie = new Trie();
        for (String word : words) {
            trie.insert(word);
        }
        int row = board.length;
        int column = board[0].length;
        boolean[][] visited = new boolean[row][column];
        Set<String> ans = new HashSet<>();
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                dfs(visited, i, j, board, ans, "", trie);
            }
        }
        return new ArrayList<>(ans);

    }

    private void dfs(boolean[][] visited, int i, int j, char[][] board, Set<String> ans, String s, Trie trie) {
        if (i < 0 || i >= board.length || j < 0 || j >= board[i].length || visited[i][j]) {
            return;
        }
        s += board[i][j];
        if (!trie.startsWith(s)) {
            return;
        }
        if (trie.search(s)) {
            ans.add(s);
            return;
        }
        visited[i][j] = true;
        dfs(visited, i - 1, j, board, ans, s, trie);
        dfs(visited, i + 1, j, board, ans, s, trie);
        dfs(visited, i, j - 1, board, ans, s, trie);
        dfs(visited, i, j + 1, board, ans, s, trie);
        visited[i][j] = false;
    }

    /**
     * 213. House Robber II
     *
     * @param nums
     * @return
     */
    public int rob(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        if (nums.length == 1) {
            return nums[0];
        }
        return Math.max(rob(nums, 0, nums.length - 2), rob(nums, 1, nums.length - 1));
    }

    private int rob(int[] nums, int start, int end) {
        int skipCurrent = 0;
        int robCurrent = 0;
        for (int i = start; i <= end; i++) {
            int tmp = skipCurrent;
            skipCurrent = Math.max(skipCurrent, robCurrent);
            robCurrent = nums[i] + tmp;
        }
        return Math.max(skipCurrent, robCurrent);
    }

    /**
     * 215. Kth Largest Element in an Array
     *
     * @param nums
     * @param k
     * @return
     */
    public int findKthLargest(int[] nums, int k) {
        if (nums == null || nums.length == 0) {
            return -1;
        }
        PriorityQueue<Integer> priorityQueue = new PriorityQueue<>();

        for (int num : nums) {
            priorityQueue.offer(num);
            if (priorityQueue.size() > k) {
                priorityQueue.poll();
            }
        }
        return priorityQueue.peek();

    }

    private int partition(int[] nums, int start, int end) {
        int pivot = nums[start];
        while (start < end) {
            while (start < end && nums[end] >= pivot) {
                end--;
            }
            if (start < end) {
                nums[start] = nums[end];
                start++;
            }
            while (start < end && nums[start] <= pivot) {
                start++;
            }
            if (start < end) {
                nums[end] = nums[start];
                end--;
            }
        }
        nums[start] = pivot;
        return start;
    }

    /**
     * 216. Combination Sum III
     *
     * @param k
     * @param n
     * @return
     */
    public List<List<Integer>> combinationSum3(int k, int n) {
        if (k <= 0 || n <= 0) {
            return new ArrayList<>();
        }
        List<List<Integer>> ans = new ArrayList<>();
        combinationSum3(ans, new ArrayList<Integer>(), 1, k, n);
        return ans;
    }

    private void combinationSum3(List<List<Integer>> ans, List<Integer> tmp, int start, int k, int n) {
        if (tmp.size() == k && n == 0) {
            ans.add(new ArrayList<>(tmp));
            return;
        }
        for (int i = start; i <= 9 && i <= n; i++) {
            tmp.add(i);
            combinationSum3(ans, tmp, i + 1, k, n - i);
            tmp.remove(tmp.size() - 1);
        }
    }

    /**
     * 219. Contains Duplicate II
     *
     * @param nums
     * @param k
     * @return
     */
    public boolean containsNearbyDuplicate(int[] nums, int k) {
        if (nums == null || nums.length == 0) {
            return false;
        }
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(nums[i])) {
                int diff = Math.abs(i - map.get(nums[i]));
                if (diff <= k) {
                    return true;
                }
                map.put(nums[i], i);
            }
        }
        return false;
    }

    /**
     * 220. Contains Duplicate III
     *
     * @param nums
     * @param k
     * @param t
     * @return
     */
    public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
        if (nums == null || nums.length == 0) {
            return false;
        }
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {

        }
        return false;
    }

    /**
     * 221. Maximal Square
     *
     * @param matrix
     * @return
     */
    public int maximalSquare(char[][] matrix) {
        if (matrix == null || matrix.length == 0) {
            return 0;
        }
        int row = matrix.length;
        int column = matrix[0].length;
        int result = 0;
        int[][] dp = new int[row + 1][column + 1];
        for (int i = 1; i <= row; i++) {
            for (int j = 1; j <= column; j++) {
                if (matrix[i - 1][j - 1] == '1') {
                    dp[i][j] = Math.min(Math.min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]) + 1;
                    result = Math.max(result, dp[i][j]);
                }
            }
        }
        return result * result;
    }

    /**
     * 222. Count Complete Tree Nodes
     *
     * @param root
     * @return
     */
    public int countNodes(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int left = countNodesOfRoot(root, true);
        int right = countNodesOfRoot(root, false);
        if (left == right) {
            return (1 << left) - 1;
        }
        return 1 + countNodes(root.left) + countNodes(root.right);
    }

    private int countNodesOfRoot(TreeNode root, boolean isLeft) {
        if (root == null) {
            return 0;
        }
        return (isLeft ? countNodesOfRoot(root.left, true) : countNodesOfRoot(root.right, false)) + 1;
    }

    /**
     * 224. Basic Calculator
     *
     * @param s
     * @return
     */
    public int calculate(String s) {
        if (s == null || s.length() == 0) {
            return 0;
        }
        Stack<Integer> stack = new Stack<>();
        int result = 0;
        int sign = 1;
        for (int i = 0; i < s.length(); i++) {
            int num = 0;
            while (i < s.length() && Character.isDigit(s.charAt(i))) {
                num = num * 10 + s.charAt(i++) - '0';
            }
            result += num * sign;
            if (i < s.length()) {
                if (s.charAt(i) == '+') {
                    sign = 1;
                } else if (s.charAt(i) == '-') {
                    sign = -1;
                } else if (s.charAt(i) == '(') {
                    stack.push(result);
                    stack.push(sign);
                    sign = 1;
                    result = 0;
                } else if (s.charAt(i) == ')') {
                    result = result * stack.pop() + stack.pop();
                }
            }
        }
        return result;
    }

    /**
     * 228. Summary Ranges
     *
     * @param nums
     * @return
     */
    public List<String> summaryRanges(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new ArrayList<>();
        }
        List<String> ans = new ArrayList<>();
        for (int i = 0; i < nums.length; i++) {
            int right = i;
            while (right + 1 < nums.length && nums[right + 1] == nums[right] + 1) {
                right++;
            }
            if (right - i > 0) {
                String tmp = nums[i] + "->" + nums[right];
                ans.add(tmp);
                i = right;
            } else if (right == i) {
                String tmp = nums[i] + "";
                ans.add(tmp);
            }
        }
        return ans;
    }

    /**
     * 229. Majority Element II
     *
     * @param nums
     * @return
     */
    public List<Integer> majorityElement(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new ArrayList<>();
        }
        int countA = 0;
        int countB = 0;
        int candidateA = nums[0];
        int candidateB = nums[0];
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == candidateA) {
                countA++;
                continue;
            }
            if (nums[i] == candidateB) {
                countB++;
                continue;
            }
            if (candidateA == 0) {
                candidateA = nums[i];
                countA = 1;
                continue;
            }
            if (candidateB == 0) {
                candidateB = nums[i];
                countB = 1;
                continue;
            }
            countA--;
            countB--;
        }
        countA = 0;
        countB = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == candidateA) {
                countA++;
            } else if (nums[i] == candidateB) {
                countB++;
            }
        }
        List<Integer> ans = new ArrayList<>();
        if (countA > nums.length / 3) {
            ans.add(candidateA);
        }
        if (countB > nums.length / 3) {
            ans.add(candidateB);
        }
        return ans;
    }

    /**
     * 227. Basic Calculator II
     *
     * @param s
     * @return
     */
    public int calculateII(String s) {
        if (s == null || s.length() == 0) {
            return 0;
        }
        s = s.trim();
        char sign = '+';
        int num = 0;
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < s.length(); i++) {
            if (Character.isDigit(s.charAt(i))) {
                num = num * 10 + s.charAt(i) - '0';
                i++;
            }
            if ((!Character.isDigit(s.charAt(i)) && s.charAt(i) != ' ') || i == s.length() - 1) {
                if (sign == '+') {
                    stack.push(num);
                } else if (sign == '-') {
                    stack.push(-num);
                } else if (sign == '*') {
                    stack.push(stack.pop() * num);
                } else if (sign == '/') {
                    stack.push(stack.pop() / num);
                }
                num = 0;
                sign = s.charAt(i);
            }
        }

        int result = 0;
        for (int n : stack) {
            result += n;
        }
        return result;
    }

    /**
     * 230. Kth Smallest Element in a BST
     *
     * @param root
     * @param k
     * @return
     */
    public int kthSmallest(TreeNode root, int k) {
        if (root == null) {
            return -1;
        }
        Stack<TreeNode> stack = new Stack<>();
        int count = 0;
        TreeNode p = root;
        while (!stack.isEmpty() || p != null) {
            while (p != null) {
                stack.push(p);
                p = p.left;
            }
            p = stack.pop();
            count++;
            if (count == k) {
                return p.val;
            }
            p = p.right;
        }
        return -1;
    }

    /**
     * 234. Palindrome Linked List
     *
     * @param head
     * @return
     */
    public boolean isPalindrome(ListNode head) {
        if (head == null) {
            return true;
        }
        ListNode fast = head;
        ListNode slow = head;
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        slow.next = reverseList(slow.next);
        slow = slow.next;
        while (slow != null) {
            if (head.val != slow.val) {
                return false;
            }
            slow = slow.next;
            head = head.next;
        }
        return true;
    }

    private ListNode reverseList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode prev = null;
        ListNode dummy = head;
        while (dummy != null) {
            ListNode tmp = dummy.next;
            dummy.next = prev;
            prev = dummy;
            dummy = tmp;
        }
        return prev;
    }

    /**
     * 235. Lowest Common Ancestor of a Binary Search Tree
     *
     * @param root
     * @param p
     * @param q
     * @return
     */
    public TreeNode lowestCommonAncestorBST(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == q || root == p) {
            return root;
        }
        if (p.val < root.val && q.val < root.val) {
            return lowestCommonAncestorBST(root.left, p, q);
        } else if (p.val > root.val && q.val > root.val) {
            return lowestCommonAncestorBST(root.right, p, q);
        } else {
            return root;
        }
    }

    /**
     * 236. Lowest Common Ancestor of a Binary Tree
     *
     * @param root
     * @param p
     * @param q
     * @return
     */
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == p || root == q) {
            return root;
        }
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if (left != null && right != null) {
            return root;
        }
        return left != null ? left : right;
    }

    /**
     * 237. Delete Node in a Linked List
     *
     * @param node
     */
    public void deleteNode(ListNode node) {
        if (node.next != null && node.next.next != null) {
            node.val = node.next.val;
            node.next = node.next.next;
        } else {
            node.val = node.next.val;
            node.next = null;
        }
    }

    /**
     * 238. Product of Array Except Self
     *
     * @param nums
     * @return
     */
    public int[] productExceptSelf(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new int[]{};
        }
        int[] dp = new int[nums.length];
        int base = 1;
        for (int i = 0; i < nums.length; i++) {
            dp[i] = base;
            base *= nums[i];
        }
        base = 1;
        for (int i = nums.length - 1; i >= 0; i--) {
            dp[i] *= base;
            base *= nums[i];
        }
        return dp;
    }

    /**
     * 240. Search a 2D Matrix II
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
        int i = 0;
        int j = column - 1;
        while (i < row && j >= 0) {
            if (matrix[i][j] == target) {
                return true;
            } else if (matrix[i][j] < target) {
                i++;
            } else {
                j--;
            }
        }
        return false;
    }

    /**
     * 241. Different Ways to Add Parentheses
     *
     * @param input
     * @return
     */
    public List<Integer> diffWaysToCompute(String input) {
        if (input == null || input.length() == 0) {
            return new ArrayList<>();
        }
        List<String> ops = new ArrayList<>();
        for (int i = 0; i < input.length(); i++) {
            int j = i;
            while (i < input.length() && Character.isDigit(input.charAt(i))) {
                i++;
            }
            ops.add(input.substring(j, i));
            if (i != input.length()) {
                ops.add(input.substring(i, i + 1));
            }
        }
        return compute(ops, 0, ops.size() - 1);
    }

    private List<Integer> compute(List<String> ops, int start, int end) {
        List<Integer> ans = new ArrayList<>();
        if (start == end) {
            Integer value = Integer.parseInt(ops.get(start));
            ans.add(value);
            return ans;
        }
        for (int i = start + 1; i <= end - 1; i = i + 2) {
            List<Integer> leftNum = compute(ops, start, i - 1);
            List<Integer> rightNum = compute(ops, i + 1, end);
            String sign = ops.get(i);
            for (Integer left : leftNum) {
                for (Integer right : rightNum) {
                    if (sign.equals("+")) {
                        ans.add(left + right);
                    } else if (sign.equals("-")) {
                        ans.add(left - right);
                    } else if (sign.equals("*")) {
                        ans.add(left * right);
                    }
                }
            }
        }
        return ans;
    }


}
