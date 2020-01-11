package org.dora.algorithm.geeksforgeek;

import org.dora.algorithm.datastructe.ListNode;
import org.dora.algorithm.datastructe.TreeNode;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * 递归其实是动态规划的另种方式
 *
 * @author dora
 * @date 2019-04-26
 */
public class DynamicProgramming {


    /**
     * 124. Binary Tree Maximum Path Sum
     *
     * @param root
     * @return
     */
    private int max_path_sum = Integer.MIN_VALUE;

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
        int n = s.length();
        boolean[][] dp = new boolean[n][n];
        int head = 0;
        int result = Integer.MIN_VALUE;
        for (int i = 0; i < s.length(); i++) {
            for (int j = 0; j <= i; j++) {
                if (s.charAt(i) == s.charAt(j)) {
                    if (i - j < 2) {
                        dp[j][i] = true;
                    } else if (dp[j + 1][i - 1]) {
                        dp[j][i] = true;
                    }
                }
                if (dp[j][i] && i - j + 1 > result) {
                    head = j;
                    result = i - j + 1;
                }
            }
        }
        if (result != Integer.MIN_VALUE) {
            return s.substring(head, head + result);
        }
        return s;
    }

    /**
     * 10. Regular Expression Matching
     * dp[i][j] = dp[i-1][j-1] if s[i] == t[j] || t[j] == '.'
     * = dp[i][j-2] if s[i] != t[j] && t[j-2] != '.' example :  s = a t = ab*
     * = dp[i-1][j] || dp[i][j-2] || dp[i][j-1]
     * e.g case 1: s = aaaaa t = a*;
     * case 2: s = a t = aa*
     * case 3: s = a t = a *
     *
     * @param s
     * @param p
     * @return
     */
    public boolean isMatch(String s, String p) {
        if (s == null && p == null) {
            return false;
        } else if (s == null) {
            return true;
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

        int[][] dp = new int[row][column];

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (obstacleGrid[i][j] == 1) {
                    continue;
                }
                if (i == 0 && j == 0) {
                    dp[i][j] = 1;
                } else if (i == 0) {
                    dp[i][j] = dp[i][j - 1];
                } else if (j == 0) {
                    dp[i][j] = dp[i - 1][j];
                } else {
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
                }
            }

        }
        return dp[row - 1][column - 1];
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

        int[] dp = new int[column];

        for (int i = 0; i < row; i++) {

            for (int j = 0; j < column; j++) {

                if (i == 0 && j == 0) {
                    dp[j] = grid[0][0];
                } else if (i == 0) {
                    dp[j] = dp[j - 1] + grid[0][j];
                } else if (j == 0) {
                    dp[j] = dp[j] + grid[i][0];
                } else {
                    dp[j] = Math.min(dp[j], dp[j - 1]) + grid[i][j];
                }

            }
        }
        return dp[column - 1];
    }

    /**
     * todo
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
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = 1 + Math.min(Math.min(dp[i][j - 1], dp[i - 1][j - 1]), dp[i - 1][j]);
                }
            }
        }
        return dp[m][n];
    }

    /**
     * height[i]: 第i列时的最大高度
     * left[i]: 第i列 的左边界
     * right[i]: 第i列 的右边界
     * 随着层次的遍历 矩形的面积也随着上一层而进行变动
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

        int[] height = new int[column];

        int[] left = new int[column];

        int[] right = new int[column];

        for (int i = 0; i < column; i++) {
            right[i] = column;
        }
        int result = 0;

        for (int i = 0; i < row; i++) {

            int leftSide = 0;

            int rightSide = column;

            for (int j = 0; j < column; j++) {
                if (matrix[i][j] == '1') {
                    height[j]++;
                } else {
                    height[j] = 0;
                }
            }
            for (int j = 0; j < column; j++) {
                if (matrix[i][j] == '1') {
                    left[j] = Math.max(leftSide, left[j]);
                } else {
                    left[j] = 0;
                    leftSide = j + 1;
                }
            }
            for (int j = column - 1; j >= 0; j--) {
                if (matrix[i][j] == '1') {

                    right[j] = Math.min(rightSide, right[j]);
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

    /**
     * todo 未来考虑使用动态规划
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
            if (this.isScramble(s1.substring(i), s2.substring(0, m - i)) &&
                    this.isScramble(s1.substring(0, i), s2.substring(m - i))) {
                return true;
            }
        }
        return false;
    }

    /**
     * todo 需要重新考虑
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
                dp[i] += dp[i - j] * dp[j];
            }

        }
        return dp[n];
    }

    /**
     * todo 动态规划
     * 97. Interleaving String
     *
     * @param s1
     * @param s2
     * @param s3
     * @return
     */
    public boolean isInterleave(String s1, String s2, String s3) {
        if (s1 == null || s2 == null) {
            return false;
        }
        return false;
    }


    /**
     * todo 左右两边姐取最小值
     * 135. Candy
     *
     * @param ratings
     * @return
     */
    public int candy(int[] ratings) {
        if (ratings == null || ratings.length == 0) {
            return 0;
        }
        int result = 0;

        int column = ratings.length;

        int[] left = new int[column];
        for (int i = 0; i < column; i++) {
            left[i] = 1;
        }
        for (int i = 1; i < column; i++) {
            if (ratings[i] > ratings[i - 1] && left[i] < left[i - 1] + 1) {
                left[i] = left[i - 1] + 1;
            }
        }
        int[] right = new int[column];
        for (int i = 0; i < column; i++) {
            right[i] = 1;
        }
        for (int i = column - 2; i >= 0; i--) {
            if (ratings[i] > ratings[i + 1] && right[i] < right[i + 1] + 1) {
                right[i] = right[i + 1] + 1;
            }
        }
        for (int i = 0; i < column; i++) {
            result += Math.max(left[i], right[i]);
        }
        return result;
    }


    // ----------递归--------------//

    /**
     * 120. Triangle
     *
     * @param triangle
     * @return
     */
    public int minimumTotal(List<List<Integer>> triangle) {
        if (triangle == null || triangle.isEmpty()) {
            return 0;
        }
        int size = triangle.size();

        List<Integer> ans = triangle.get(size - 1);

        for (int i = size - 2; i >= 0; i--) {

            for (int j = 0; j < triangle.get(i).size(); j++) {

                int value = Math.min(ans.get(j), ans.get(j + 1)) + triangle.get(i).get(j);

                ans.set(j, value);
            }
        }
        return ans.get(0);
    }

    public List<List<Integer>> subsetsWithDup(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new ArrayList<>();
        }
        Arrays.sort(nums);
        List<List<Integer>> ans = new ArrayList<>();
        this.subsetsWithDup(ans, new ArrayList<>(), 0, nums);
        return ans;
    }

    private <E> void subsetsWithDup(List<List<Integer>> ans, List<Integer> tmp, int start, int[] nums) {
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
        if (start > end) {
            ans.add(null);
            return ans;
        }
        if (start == end) {
            TreeNode node = new TreeNode(start);
            ans.add(node);
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
        return this.buildTree(0, preorder, 0, inorder.length - 1, inorder);
    }

    private TreeNode buildTree(int preStart, int[] preorder, int inStart, int inEnd, int[] inorder) {
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
        root.left = this.buildTree(preStart + 1, preorder, inStart, index - 1, inorder);

        root.right = this.buildTree(preStart + index - inStart + 1, preorder, index + 1, inEnd, inorder);

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
        return this.buildTreeII(0, inorder.length - 1, inorder, 0, postorder.length - 1, postorder);
    }

    private TreeNode buildTreeII(int inStart, int inEnd, int[] inorder, int postStart, int postEnd, int[] postorder) {
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
        root.left = this.buildTreeII(inStart, index - 1, inorder, postStart, postStart + index - inStart - 1, postorder);
        root.right = this.buildTreeII(index + 1, inEnd, inorder, postStart + index - inStart, postEnd - 1, postorder);
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
        return this.sortedArrayToBST(0, nums.length - 1, nums);
    }

    private TreeNode sortedArrayToBST(int start, int end, int[] nums) {
        if (start > end) {
            return null;
        }
        int mid = start + (end - start) / 2;
        TreeNode root = new TreeNode(nums[mid]);
        root.left = this.sortedArrayToBST(start, mid - 1, nums);
        root.right = this.sortedArrayToBST(mid + 1, end, nums);
        return root;
    }

    /**
     * todo
     * 109. Convert Sorted List to Binary Search Tree
     *
     * @param head
     * @return
     */
    public TreeNode sortedListToBST(ListNode head) {
        if (head == null) {
            return null;
        }
        return this.sortedListToBST(head, null);
    }

    private TreeNode sortedListToBST(ListNode start, ListNode end) {
        if (start == end) {
            return null;
        }
        ListNode slow = start;

        ListNode fast = start;

        while (fast != end && fast.next != end) {
            fast = fast.next.next;
            slow = slow.next;
        }
        TreeNode root = new TreeNode(slow.val);

        root.left = this.sortedListToBST(start, slow);

        root.right = this.sortedListToBST(slow.next, end);

        return root;
    }

    /**
     * 110. Balanced Binary Tree
     *
     * @param root
     * @return
     */
    public boolean isBalanced(TreeNode root) {
        if (root == null) {
            return true;
        }
        int left = this.treeDepth(root.left);
        int right = this.treeDepth(root.right);
        if (Math.abs(left - right) <= 1) {
            return this.isBalanced(root.left) && this.isBalanced(root.right);
        }
        return false;

    }

    private int treeDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        return 1 + Math.max(this.treeDepth(root.left), this.treeDepth(root.right));
    }

    /**
     * 112. Path Sum
     *
     * @param root
     * @param sum
     * @return
     */
    public boolean hasPathSum(TreeNode root, int sum) {
        if (root == null) {
            return false;
        }
        if (root.left == null && root.right == null && root.val == sum) {
            return true;
        }
        return this.hasPathSum(root.left, sum - root.val)
                || this.hasPathSum(root.right, sum - root.val);
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

        this.pathSum(ans, new ArrayList<Integer>(), root, sum);

        return ans;
    }

    private void pathSum(List<List<Integer>> ans, List<Integer> tmp, TreeNode root, int sum) {
        if (root == null) {
            return;
        }
        tmp.add(root.val);
        if (root.left == null && root.right == null && root.val == sum) {
            ans.add(new ArrayList<>(tmp));
        } else {
            if (root.left != null) {
                this.pathSum(ans, tmp, root.left, sum - root.val);
            }
            if (root.right != null) {
                this.pathSum(ans, tmp, root.right, sum - root.val);
            }
        }
        tmp.remove(tmp.size() - 1);
    }

    public int maxPathSum(TreeNode root) {
        if (root == null) {
            return 0;
        }
        this.dfsPathSum(root);

        return max_path_sum;
    }

    private int dfsPathSum(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int leftValue = this.dfsPathSum(root.left);

        leftValue = Math.max(leftValue, 0);


        int rightValue = this.dfsPathSum(root.right);


        rightValue = Math.max(rightValue, 0);

        max_path_sum = Math.max(max_path_sum, leftValue + rightValue + root.val);

        return Math.max(leftValue, rightValue) + root.val;

    }


    /**
     * 126. Word Ladder II
     *
     * @param beginWord
     * @param endWord
     * @param wordList
     * @return
     */
    public List<List<String>> findLadders(String beginWord, String endWord, List<String> wordList) {
        return null;
    }


    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        return 0;
    }


    /**
     * 174. Dungeon Game
     *
     * @param dungeon
     * @return
     */
    public int calculateMinimumHP(int[][] dungeon) {
        if (dungeon == null) {
            return 0;
        }
        int row = dungeon.length;

        int column = dungeon[0].length;

        int[] dp = new int[column];

        dp[column - 1] = Math.max(1, 1 - dungeon[row - 1][column - 1]);

        for (int j = column - 2; j >= 0; j--) {
            dp[j] = Math.max(1, dp[j + 1] - dungeon[row - 1][j]);
        }
        for (int i = row - 2; i >= 0; i--) {
            for (int j = column - 1; j >= 0; j--) {
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
     * 221. Maximal Square
     * 层次遍历、每次遍历每一层最大的宽度
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
                    dp[i][j] = Math.min(dp[i - 1][j - 1], Math.min(dp[i - 1][j], dp[i][j - 1])) + 1;

                    result = Math.max(result, dp[i][j] * dp[i][j]);
                }
            }
        }
        return result;
    }


    /**
     * 265. Paint House II
     *
     * @param costs
     * @return
     */
    public int minCostII(int[][] costs) {
        if (costs == null || costs.length == 0) {
            return 0;
        }
        int row = costs.length;
        int column = costs[0].length;

        int min1 = -1;
        int min2 = -1;

        for (int i = 0; i < row; i++) {
            int last1 = min1;

            int last2 = min2;

            min1 = -1;

            min2 = -1;
            for (int j = 0; j < column; j++) {
                if (j != last1) {
                    costs[i][j] += last1 < 0 ? 0 : costs[i - 1][last1];
                } else {
                    costs[i][j] += last2 < 0 ? 0 : costs[i - 1][last2];
                }

                if (min1 < 0 || costs[i][j] < costs[i][min1]) {
                    min2 = min1;
                    min1 = j;
                } else if (min2 < 0 || costs[i][j] < costs[i][min2]) {
                    min2 = j;
                }
            }
        }
        return costs[row - 1][min1];
    }

    public int minCostIIV2(int[][] costs) {
        if (costs == null || costs.length == 0) {
            return 0;
        }
        int column = costs[0].length;
        int row = costs.length;

        int result = Integer.MAX_VALUE;

        if (row == 1) {
            for (int j = 0; j < column; j++) {
                result = Math.min(result, costs[0][j]);
            }
            return result;
        }
        int[][] dp = new int[row][column];

        for (int j = 0; j < column; j++) {
            dp[0][j] = costs[0][j];
        }

        for (int i = 1; i < row; i++) {
            for (int j = 0; j < column; j++) {
                dp[i][j] = Integer.MAX_VALUE;
                for (int k = 0; k < column; k++) {
                    if (j != k) {
                        dp[i][j] = Math.min(dp[i - 1][k] + costs[i][j], dp[i][j]);
                    }
                }
                if (i == row - 1) {
                    result = Math.min(result, dp[i][j]);
                }
            }
        }
        return result;
    }

    public int minCostIIV3(int[][] costs) {
        if (costs == null || costs.length == 0) {
            return 0;
        }
        int column = costs[0].length;

        int row = costs.length;


        int min1 = -1;

        int min2 = -1;

        int index = 0;

        for (int i = 0; i < row; i++) {

            int tmp1 = Integer.MAX_VALUE;
            int tmp2 = tmp1;
            int idx = -1;
            for (int j = 0; j < column; j++) {
                costs[i][j] = costs[i][j] + (j == index ? min2 : min1);

                if (costs[i][j] < tmp1) {
                    tmp2 = tmp1;
                    tmp1 = costs[i][j];
                    idx = j;
                } else if (costs[i][j] < tmp2) {
                    tmp2 = costs[i][j];
                }
            }
            min1 = tmp1;
            min2 = tmp2;
            index = idx;
        }
        return min1;
    }


    /**
     * 274. H-Index
     *
     * @param citations
     * @return
     */
    public int hIndex(int[] citations) {
        return 0;
    }

    /**
     * 275. H-Index II
     *
     * @param citations
     * @return
     */
    public int hIndexII(int[] citations) {
        return 0;
    }

    /**
     * 276 Paint Fence
     *
     * @param n: non-negative integer, n posts
     * @param k: non-negative integer, k colors
     * @return: an integer, the total number of ways
     */
    public int numWays(int n, int k) {
        // write your code here
        return 1;
    }


}
