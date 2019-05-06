package org.dora.algorithm.leetcode;

import org.dora.algorithm.datastructe.ListNode;
import org.dora.algorithm.datastructe.TreeNode;

import java.util.*;

/**
 * @author dora
 * @date 2019-05-02
 */
public class ThreePage {


    public static void main(String[] args) {
        ThreePage threePage = new ThreePage();
    }

    /**
     * 201. Bitwise AND of Numbers Range
     * todo 不懂 位运算
     *
     * @param m
     * @param n
     * @return
     */
    public int rangeBitwiseAnd(int m, int n) {
        return 0;
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
            int tmp = n;
            int result = 0;
            while (tmp != 0) {
                int value = tmp % 10;
                result += value * value;

                tmp /= 10;
            }

            if (result == 1) {
                return true;
            }

            if (used.contains(result)) {
                return false;
            }
            n = result;
            used.add(n);
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

        if (head.val == val) {
            return this.removeElements(head.next, val);
        } else {
            head.next = this.removeElements(head.next, val);
            return head;
        }
    }

    /**
     * 204. Count Primes 计算素数个数
     * todo 巧妙设计
     *
     * @param n
     * @return
     */
    public int countPrimes(int n) {
        int count = 0;
        for (int i = 2; i < Math.sqrt(n); i++) {
            if (this.isPrime(i)) {
                count++;
            }
        }
        return count;
    }

    private boolean isPrime(int i) {
        for (int j = 2; j < i; j++) {
            if (i % j == 0) {
                return false;
            }
        }
        return true;
    }

    /**
     * 205. Isomorphic Strings
     * todo 哈希思想 注意遍历退出条件
     *
     * @param s
     * @param t
     * @return
     */
    public boolean isIsomorphic(String s, String t) {
        if (s == null || t == null) {
            return false;
        }
        int[] hash1 = new int[512];
        int[] hash2 = new int[512];
        for (int i = 0; i < s.length(); i++) {
            if (hash1[s.charAt(i)] != hash2[t.charAt(i)]) {
                return false;
            }
            hash1[s.charAt(i)] = i + 1;
            hash2[t.charAt(i)] = i + 1;
        }
        return false;
    }

    /**
     * 反转链表
     *
     * @param head
     * @return
     */
    public ListNode reverseList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode next = this.reverseList(head.next);
        head.next.next = head;
        head.next = null;
        return next;

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
        int begin = 0;
        int end = 0;
        int result = Integer.MAX_VALUE;
        int local = 0;
        while (end < nums.length) {

            local += nums[end];


            while (local >= s) {

                result = Math.min(result, end - begin + 1);

                local -= nums[begin++];
            }
            end++;
        }
        return result == Integer.MAX_VALUE ? 0 : result;
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
        return Math.max(this.houseRob(0, nums.length - 2, nums), this.houseRob(1, nums.length - 1, nums));
    }

    private int houseRob(int start, int end, int[] nums) {
        if (start > end) {
            return 0;
        }
        int robPrev = 0;
        int robCurrent = 0;
        for (int i = start; i <= end; i++) {
            int tmp = robPrev;
            robPrev = Math.max(robPrev, robCurrent);
            robCurrent = tmp + nums[i];
        }
        return Math.max(robPrev, robCurrent);
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
     * 214. Shortest Palindrome
     *
     * @param nums
     * @param k
     * @return
     */
    public int findKthLargest(int[] nums, int k) {
        if (nums == null || nums.length == 0) {
            return -1;
        }

        int partition = this.partition(nums, 0, nums.length - 1);
        k--;
        while (partition != k) {

            if (partition > k) {
                partition = this.partition(nums, 0, k - 1);
            } else {
                partition = this.partition(nums, partition + 1, nums.length - 1);
            }
        }
        return nums[k];
    }

    private int partition(int[] nums, int start, int end) {
        if (start > end) {
            return -1;
        }
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
        this.combinationSum3(ans, new ArrayList<Integer>(), 1, k, n);
        return ans;
    }

    private void combinationSum3(List<List<Integer>> ans, List<Integer> tmp, int start, int k, int n) {
        if (tmp.size() == k && n == 0) {
            ans.add(new ArrayList<>(tmp));
            return;
        }
        for (int i = start; i <= 9 && i <= n; i++) {
            tmp.add(i);
            this.combinationSum3(ans, tmp, i + 1, k, n - i);
            tmp.remove(tmp.size() - 1);
        }
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
        HashSet<Integer> set = new HashSet<>();
        for (int num : nums) {
            if (set.contains(num)) {
                return true;
            }
            set.add(num);
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
                int diff = Math.abs(i - map.get(nums[i]));
                if (diff > k) {
                    return false;
                }
            }
        }
        return true;
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
        TreeSet<Integer> treeSet = new TreeSet<>();
        for (int i = 0; i < nums.length; i++) {
            Integer floor = treeSet.floor(i - t);
            Integer ceil = treeSet.ceiling(i + t);
        }
        return false;

    }

    /**
     * 221. Maximal Square
     * todo 需考虑好方程式
     *
     * @param matrix
     * @return
     */
    public int maximalSquare(char[][] matrix) {
        if (matrix == null | matrix.length == 0) {
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
//                    System.out.println("i:" + i + "j:" + j + "width:" + width);
                    result = Math.max(result, dp[i][j]);
                }
            }
        }
        return result * result;
    }


    /**
     * 224. Basic Calculator
     *
     * @param s
     * @return
     */
    public int calculate(String s) {
        if (s == null) {
            return 0;
        }
        s = s.trim();
        if (s.length() == 0) {
            return 0;
        }
        int sign = 1;
        Stack<Integer> stack = new Stack<>();
        int result = 0;
        int index = 0;
        while (index < s.length()) {

            int tmp = 0;
            if (Character.isDigit(s.charAt(index))) {
                while (index < s.length() && Character.isDigit(s.charAt(index))) {
                    tmp = tmp * 10 + s.charAt(index) - '0';
                    index++;
                }
                result += sign * tmp;
            } else {
                if (s.charAt(index) == '+') {
                    sign = 1;
                } else if (s.charAt(index) == '-') {
                    sign = -1;
                } else if (s.charAt(index) == '(') {
                    stack.push(result);
                    stack.push(sign);
                    result = 0;
                    sign = 1;
                } else if (s.charAt(index) == ')') {
                    result = stack.pop() * result + stack.pop();
                }
                index++;
            }
        }
        return result;
    }


    /**
     * 226. Invert Binary Tree
     *
     * @param root
     * @return
     */
    public TreeNode invertTree(TreeNode root) {
        if (root != null) {
            TreeNode left = root.left;
            root.left = root.right;
            root.right = left;
            this.invertTree(root.left);
            this.invertTree(root.right);
        }
        return root;
    }


    /**
     * 227. Basic Calculator II
     * todo 不太懂 不熟练
     *
     * @param s
     * @return
     */
    public int calculateII(String s) {
        if (s == null) {
            return 0;
        }
        s = s.trim();

        if (s.length() == 0) {
            return 0;
        }
        char sign = '+';

        Stack<Integer> stack = new Stack<>();

        int result = 0;

        for (int i = 0; i < s.length(); i++) {
            if (Character.isDigit(s.charAt(i))) {
                result = result * 10 + s.charAt(i) - '0';
            }

            if ((!Character.isDigit(s.charAt(i)) && s.charAt(i) != ' ') || i == s.length() - 1) {
                if (sign == '+') {
                    stack.push(result);
                } else if (sign == '-') {
                    stack.push(-result);
                } else if (sign == '*') {
                    stack.push(stack.pop() * result);
                } else if (sign == '/') {
                    stack.push(stack.pop() / result);
                }
                result = 0;
                sign = s.charAt(i);
            }
        }

        result = 0;
        for (Integer num : stack) {
            result += num;
        }
        return result;

    }


    /**
     * 228. Summary Ranges
     * todo 巧妙设计
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
            if (right > i) {
                String value = nums[i] + "->" + nums[right];
                ans.add(value);
                i = right;
            } else {
                String value = nums[i] + "";
                ans.add(value);

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
        List<Integer> ans = new ArrayList<>();
        int candidateA = nums[0];
        int candidateB = nums[0];
        int countA = 0;
        int countB = 0;
        for (int num : nums) {
            if (num == candidateA) {
                countA++;
                continue;
            }
            if (num == candidateB) {
                countB++;
                continue;
            }
            if (countA == 0) {
                candidateA = num;
                countA = 1;
                continue;
            }
            if (countB == 0) {
                candidateB = num;
                countB = 1;
                continue;
            }
            countA--;
            countB--;
        }
        countA = 0;
        countB = 0;
        for (int num : nums) {
            if (num == candidateA) {
                countA++;
            } else if (num == candidateB) {
                countB++;
            }
        }
        if (countA * 3 > nums.length) {
            ans.add(candidateA);
        }
        if (countB * 3 > nums.length) {
            ans.add(candidateB);
        }
        return ans;
    }


    /**
     * 230. Kth Smallest Element in a BST
     *
     * @param root
     * @param k
     * @return
     */
    public int kthSmallest(TreeNode root, int k) {
        if (root == null || k <= 0) {
            return -1;
        }
        Stack<TreeNode> stack = new Stack<>();
        int count = 0;
        while (!stack.isEmpty() | root != null) {
            while (root != null) {
                stack.push(root);
                root = root.left;
            }
            root = stack.pop();
            count++;
            if (count == k) {
                return root.val;
            }
            root = root.right;
        }
        return -1;
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
        if (n == 0) {
            return true;
        }
        return false;
    }


    /**
     * 234. Palindrome Linked List
     *
     * @param head
     * @return
     */
    public boolean isPalindrome(ListNode head) {
        if (head == null) {
            return false;
        }
        if (head.next == null) {
            return true;
        }
        ListNode fast = head;
        ListNode slow = head;
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        slow.next = this.reverseList(slow.next);
        slow = slow.next;
        while (slow != null) {
            if (head.val != slow.val) {
                return false;
            }
            head = head.next;
            slow = slow.next;
        }
        return true;

    }


    /**
     * 235. Lowest Common Ancestor of a Binary Search Tree
     *
     * @param root
     * @param p
     * @param q
     * @return
     */
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {

        if (root == null || p == root || q == root) {
            return root;
        }
        if (p.val < root.val && q.val < root.val) {
            return this.lowestCommonAncestor(root.left, p, q);
        } else if (p.val > root.val && q.val > root.val) {
            return this.lowestCommonAncestor(root.right, p, q);
        } else {
            return null;
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
    public TreeNode lowestCommonAncestorTree(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || p == root || q == root) {
            return root;
        }

        TreeNode left = this.lowestCommonAncestorTree(root.left, p, q);

        TreeNode right = this.lowestCommonAncestorTree(root.right, p, q);

        if (left == null && right == null) {
            return root;
        } else {
            return left != null ? left : right != null ? right : null;
        }
    }


    /**
     * 237. Delete Node in a Linked List
     *
     * @param node
     */
    public void deleteNode(ListNode node) {
        if (node == null) {
            return;
        }
        if (node.next == null) {
            node = null;
        } else {
            node.val = node.next.val;

            ListNode tmp = node;

            node.next = tmp.next;

            tmp = null;

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
        int[] ans = new int[nums.length];
        int base = 1;
        for (int i = 0; i < nums.length; i++) {
            ans[i] = base;
            base ^= nums[i];
        }
        base = 1;
        for (int i = nums.length - 1; i >= 0; i--) {
            ans[i] *= base;
            base *= ans[i];
        }
        return ans;
    }


    /**
     * 239. Sliding Window Maximum
     *
     * @param nums
     * @param k
     * @return
     */
    public int[] maxSlidingWindow(int[] nums, int k) {
        if (nums == null || nums.length == 0) {

        }
        LinkedList<Integer> ans = new LinkedList<>();

        List<Integer> data = new ArrayList<>();


        for (int i = 0; i < nums.length; i++) {
            int begin = i - k + 1;

            if (ans.isEmpty()) {
                ans.addLast(i);
            } else if (begin > ans.peekFirst()) {
                ans.removeFirst();
            }
            while (!ans.isEmpty() && nums[ans.peekFirst()] <= nums[i]) {
                ans.pollFirst();
            }
            ans.add(i);
            if (begin >= 0) {
                data.add(nums[ans.peekFirst()]);
            }

        }

        return nums;
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
            } else if (matrix[i][j] > target) {
                j--;
            } else {
                i++;
            }
        }
        return false;
    }

    /**
     * 242. Valid Anagram
     *
     * @param input
     * @return
     */
    public List<Integer> diffWaysToCompute(String input) {
        if (input == null || input.length() == 0) {
            return new ArrayList<>();
        }
        List<String> data = new ArrayList<>();
        for (int i = 0; i < input.length(); i++) {
            int j = i;
            while (i < input.length() && Character.isDigit(i)) {
                i++;
            }
            data.add(input.substring(j, i));
            if (i != input.length()) {
                data.add(input.substring(i, i + 1));
            }
        }
        return this.compute(data, 0, data.size() - 1);
    }

    private List<Integer> compute(List<String> ops, int start, int end) {
        List<Integer> ans = new ArrayList<>();
        if (start == end) {
            int value = Integer.parseInt(ops.get(start));
            ans.add(value);
            return ans;
        }
        for (int i = start + 1; i <= end - 1; i = i + 2) {
            List<Integer> leftNum = this.compute(ops, start, i - 1);
            List<Integer> rightNum = this.compute(ops, i + 1, end);
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
        if (s.length() != t.length()) {
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
        for (int i = 0; i < dp.length; i++) {
            dp[i] = 1;
        }
        for (int i = 1; i < nums.length; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[j] < nums[i] && dp[j] + 1 > dp[i]) {
                    dp[i] = 1 + dp[j];
                }
            }
        }
        int result = 0;
        for (int num : dp) {
            result = Math.max(result, num);
        }
        return result;


    }


}