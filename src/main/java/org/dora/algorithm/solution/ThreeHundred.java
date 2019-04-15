package org.dora.algorithm.solution;

import org.dora.algorithm.datastructe.Interval;
import org.dora.algorithm.datastructe.ListNode;
import org.dora.algorithm.datastructe.TreeNode;

import java.util.*;

/**
 * @author liulu
 * @date 2019-03-16
 */
public class ThreeHundred {
    public static void main(String[] args) {
        ThreeHundred threeHundred = new ThreeHundred();
        TreeNode root = new TreeNode(1);
        root.left = new TreeNode(2);
        root.right = new TreeNode(3);
        threeHundred.binaryTreePaths(root);
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
        Queue<Integer> queue = new PriorityQueue<>();
        Set<Integer> ans = new HashSet<>();
        queue.add(n);
        ans.add(n);
        while (!queue.isEmpty()) {
            int num = queue.poll();
            int result = 0;
            while (num != 0) {
                int digit = num % 10;
                result += digit * digit;
                num /= 10;
            }
            if (result == 1) {
                return true;
            }
            if (ans.contains(result)) {
                return false;
            }
            queue.add(result);
            ans.add(result);
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
            return head;
        }
        ListNode root = new ListNode(0);

        root.next = head;

        ListNode dummy = root;
        ListNode fast = head;
        while (fast != null) {
            if (fast.val == val) {
                dummy.next = fast.next;
            } else {
                dummy.next = fast;
            }
            fast = fast.next;
        }
        return root.next;

    }

    /**
     * 206. Reverse Linked List
     *
     * @param head
     * @return
     */
    public ListNode reverseList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode prev = reverseList(head.next);
        head.next.next = head;
        head.next = null;
        return prev;
    }

    /**
     * 209. Minimum Size Subarray Sum
     *
     * @param s
     * @param nums
     * @return
     */
    public int minSubArrayLen(int s, int[] nums) {
        return 0;
    }

    /**
     * 212. Word Search II
     *
     * @param board
     * @param words
     * @return
     */
    public List<String> findWords(char[][] board, String[] words) {
        if (board == null || words == null) {
            return new ArrayList<>();
        }
        Trie trie = new Trie();
        for (String word : words) {
            trie.insert(word);
        }
        int m = board.length;
        int n = board[0].length;
        boolean[][] visited = new boolean[m][n];
        Set<String> ans = new HashSet<>();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                dfs(board, i, j, trie, visited, ans, "");
            }
        }
        return new ArrayList<>(ans);
    }

    private void dfs(char[][] board, int i, int j, Trie trie, boolean[][] visited, Set<String> ans, String s) {
        if (i < 0 || i >= board.length || j < 0 || j >= board[i].length || visited[i][j]) {
            return;
        }
        s += board[i][j];
        if (!trie.startsWith(s)) {
            return;
        }
        if (trie.search(s)) {
            ans.add(s);
        }
        visited[i][j] = true;
        dfs(board, i - 1, j, trie, visited, ans, s);
        dfs(board, i + 1, j, trie, visited, ans, s);
        dfs(board, i, j - 1, trie, visited, ans, s);
        dfs(board, i, j + 1, trie, visited, ans, s);
        visited[i][j] = false;
    }

    /**
     * 213. House Robber II
     *
     * @param nums
     * @return
     */
    public int robII(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
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
     * 214. Shortest Palindrome
     *
     * @param s
     * @return
     */
    public String shortestPalindrome(String s) {
        if (s == null || s.length() == 0) {
            return "";
        }
        return "";

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
        combinationSum3(ans, new ArrayList<>(), 1, k, n);
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
        int partion = partition(nums, 0, nums.length - 1);
        k = nums.length - k;

        while (partion != k) {

            if (partion < k) {
                partion = partition(nums, partion + 1, nums.length - 1);
            } else {
                partion = partition(nums, 0, partion - 1);
            }
        }
        return nums[partion];
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
     * 231. Power of Two
     *
     * @param n
     * @return
     */
    public boolean isPowerOfTwo(int n) {
        if (n <= 0) {
            return false;
        }
        n = n & (n - 1);
        if (n != 0) {
            return true;
        }
        return false;
    }

    /**
     * 217. Contains Duplicate
     *
     * @param nums
     * @return
     */
    public boolean containsDuplicate(int[] nums) {
        if (nums == null || nums.length == 0) {
            return false;
        }
        Set<Integer> ans = new HashSet<>();
        for (int num : nums) {
            if (ans.contains(num)) {
                return true;
            }
            ans.add(num);
        }
        return false;
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
                int index = map.get(i);
                if (Math.abs(index - i) <= k) {
                    return true;
                }
            }
            map.put(nums[i], i);
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
        return true;
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
        int result = 0;
        int m = matrix.length;
        int n = matrix[0].length;
        int[][] dp = new int[m][n];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                if (matrix[i][j] != '1') {
                    dp[i][j] = -1;
                    continue;
                }
                dp[i][j] = j;
                if (j > 0 && dp[i][j - 1] >= 0) {
                    dp[i][j] = dp[i][j - 1];
                }
                int width = j - dp[i][j] + 1;
                for (int k = i; k >= 0 && matrix[k][j] == '1'; k--) {
                    width = Math.max(width, dp[k][j] - j + 1);
                    int h = i - k + 1;
                    if (width != h) {
                        continue;
                    }
                    result = Math.max(result, width * h);

                }
            }
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
        TreeNode p = root;
        int count = 0;
        while (!stack.isEmpty() || p != null) {
            while (p != null) {
                stack.push(p);
                p = p.left;
            }
            TreeNode node = stack.pop();
            count++;
            if (count == k) {
                return node.val;
            }
            p = node.right;
        }
        return -1;
    }

    /**
     * 235 Lowest Common Ancestor of a BST
     *
     * @param root
     * @param p
     * @param q
     * @return
     */
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == p || root == q) {
            return root;
        }
        if (p.val < root.val && q.val > root.val) {
            return root;
        } else if (p.val < root.val) {
            return lowestCommonAncestor(root.left, p, q);
        } else {
            return lowestCommonAncestor(root.right, p, q);
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
    public TreeNode lowestCommonAncestorII(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null) {
            return null;
        }
        if (root == p || root == q) {
            return root;
        }
        TreeNode left = lowestCommonAncestorII(root.left, p, q);
        TreeNode right = lowestCommonAncestorII(root.right, p, q);

        return left != null && right != null ? root : left != null ? left : right;
    }

    /**
     * 300. Longest Increasing Subsequence
     *
     * @param nums
     * @return
     */
    public int lengthOfLIS(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int[] dp = new int[nums.length];
        for (int i = 0; i < nums.length; i++) {
            dp[i] = 1;
        }
        for (int i = 1; i < nums.length; i++) {

            for (int j = 0; j < i; j++) {
                if (nums[j] < nums[i] && dp[i] < dp[j] + 1) {
                    dp[i] = dp[j] + 1;
                }
            }
        }
        int result = 0;
        for (int num : dp) {
            result = Math.max(result, num);
        }
        return result;

    }

    /**
     * 242. Valid Anagram
     *
     * @param s
     * @param t
     * @return
     */
    public boolean isAnagram(String s, String t) {
        if (s == null || t == null) {
            return false;
        }
        if (s.equals(t)) {
            return true;
        }
        int m = s.length();
        int n = t.length();
        if (m != n) {
            return false;
        }
        int[] hash = new int[256];
        for (int i = 0; i < s.length(); i++) {
            hash[s.charAt(i) - 'a']--;
            hash[t.charAt(i) - 'a']++;
        }
        for (int i = 0; i < hash.length; i++) {
            if (hash[i] != 0) {
                return false;
            }
        }

        return true;
    }

    /**
     * 243、Shortest Word Distance
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
        int result = 0;
        int first = -1;
        int second = -1;
        for (int i = 0; i < words.length; i++) {
            if (words[i].equals(word1)) {
                first = i;
            } else if (words[i].equals(word2)) {
                second = i;
            }
            if (first != -1 && second != -1) {
                result = Math.min(result, Math.abs(first - second));
            }
        }
        return result;

    }

    /**
     * 257. Binary Tree Paths
     *
     * @param root
     * @return
     */
    public List<String> binaryTreePaths(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        List<String> ans = new ArrayList<>();
        dfs(ans, "", root);
        return ans;
    }

    private void dfs(List<String> ans, String tmp, TreeNode root) {
        tmp = tmp + root.val;

        if (root.left == null && root.right == null) {
            ans.add(tmp);
            return;
        }

        if (root.left != null) {
            dfs(ans, tmp + "->", root.left);
        }
        if (root.right != null) {
            dfs(ans, tmp + "->", root.right);
        }
    }

    /**
     * 252、Meeting Rooms
     *
     * @param intervals
     * @return
     */
    public boolean canAttendMeetings(List<Interval> intervals) {
        if (intervals == null || intervals.isEmpty()) {
            return false;
        }
        intervals.sort(new Comparator<Interval>() {
            @Override
            public int compare(Interval o1, Interval o2) {
                return o1.start - o2.start;
            }
        });
        for (int i = 1; i < intervals.size(); i++) {
            if (intervals.get(i).start < intervals.get(i - 1).end) {
                return false;
            }
        }
        return true;
    }

    /**
     * 253、Meeting Rooms II
     *
     * @param intervals
     * @return
     */
    public int minMeetingRooms(List<Interval> intervals) {
        if (intervals == null || intervals.isEmpty()) {
            return 0;
        }
        intervals.sort((o1, o2) -> o1.start - o2.start);
        PriorityQueue<Interval> priorityQueue = new PriorityQueue<>((o1, o2) -> o1.end - o2.end);
        priorityQueue.offer(intervals.get(0));
        for (int i = 1; i < priorityQueue.size(); i++) {
            Interval interval = priorityQueue.peek();
            if (interval.end <= intervals.get(i).start) {
                interval.end = intervals.get(i).end;
            } else {
                priorityQueue.offer(intervals.get(i));
            }
        }
        return priorityQueue.size();
    }

}
