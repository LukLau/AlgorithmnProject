package org.dora.algorithm.leetcode;

import org.dora.algorithm.datastructe.ListNode;
import org.dora.algorithm.datastructe.TreeNode;

import java.util.*;

/**
 * @author dora
 * @date 2019/11/20
 */
public class SerialQuestion {


    //    --- 卖股票 ----- //

    public static void main(String[] args) {
        SerialQuestion serialQuestion = new SerialQuestion();
        String[] words = new String[]{"a", "c", "a", "a", "a"};
        String word1 = "a";
        String word2 = "a";
        serialQuestion.shortestDistanceIII(words, word1, word2);
    }

    /**
     * 121. Best Time to Buy and Sell Stock
     *
     * @param prices
     * @return
     */
    public int maxProfit(int[] prices) {
        if (prices == null || prices.length == 0) {
            return 0;
        }
        int minPrice = prices[0];
        int result = 0;
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > minPrice) {
                result = Math.max(result, prices[i] - minPrice);
            } else {
                minPrice = prices[i];
            }
        }
        return result;
    }

    /**
     * 122. Best Time to Buy and Sell Stock II
     *
     * @param prices
     * @return
     */
    public int maxProfitII(int[] prices) {
        if (prices == null || prices.length == 0) {
            return 0;
        }
        int result = 0;

        int minPrice = prices[0];
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > minPrice) {
                result += prices[i] - minPrice;
            }
            minPrice = prices[i];
        }
        return result;
    }


    /**
     * key case: 第K次卖出的时候
     * 188. Best Time to Buy and Sell Stock IV
     *
     * @param k
     * @param prices
     * @return
     */
    public int maxProfitIIII(int k, int[] prices) {
        if (prices == null || prices.length == 0) {
            return 0;
        }
        if (k > prices.length / 2) {
            return quickProfit(prices);
        }
        int[][] dp = new int[k + 1][prices.length];
        for (int i = 1; i <= k; i++) {
            int tmp = -prices[0];

            for (int j = 1; j < prices.length; j++) {
                dp[i][j] = Math.max(dp[i][j - 1], prices[j] + tmp);

                tmp = Math.max(tmp, dp[i - 1][j - 1] - prices[j]);
            }
        }
        return dp[k][prices.length - 1];
    }

    private int quickProfit(int[] prices) {
        int result = 0;
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > prices[i - 1]) {
                result += prices[i] - prices[i - 1];
            }
        }
        return result;
    }

    /**
     * 309. Best Time to Buy and Sell Stock with Cooldown
     *
     * @param prices
     * @return
     */
    public int maxProfitIV(int[] prices) {
        if (prices == null || prices.length == 0) {
            return 0;
        }
        return Integer.MAX_VALUE;
    }


    // ------字符串切割-- //

    /**
     * todo 股票交易两次
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

        int[] leftProfit = new int[n];

        int minLeftPrice = prices[0];

        int leftResult = 0;

        for (int i = 1; i < n; i++) {
            if (prices[i] < minLeftPrice) {
                minLeftPrice = prices[i];
            }

            leftResult = Math.max(leftResult, prices[i] - minLeftPrice);

            leftProfit[i] = leftResult;
        }
        int[] rightProfit = new int[n + 1];

        int maxRightPrice = prices[n - 1];

        int rightResult = 0;

        for (int i = n - 2; i >= 1; i--) {
            if (maxRightPrice < prices[i]) {

                maxRightPrice = prices[i];
            }

            rightResult = Math.max(rightResult, maxRightPrice - prices[i]);

            rightProfit[i] = rightResult;

        }

        int result = 0;

        for (int i = 0; i < n; i++) {
            result = Math.max(result, leftProfit[i] + rightProfit[i + 1]);
        }
        return result;


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
        this.partition(ans, new ArrayList<String>(), 0, s);
        return ans;
    }

    private void partition(List<List<String>> ans, List<String> tmp, int k, String s) {
        if (k == s.length()) {
            ans.add(new ArrayList<>(tmp));
        }
        for (int i = k; i < s.length(); i++) {
            if (this.isValid(s, k, i)) {
                tmp.add(s.substring(k, i + 1));
                this.partition(ans, tmp, i + 1, s);
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
     * cut[i] is the minimum of cut[j - 1] + 1 (j <= i), if [j, i] is palindrome.
     * If [j, i] is palindrome, [j + 1, i - 1] is palindrome, and c[j] == c[i].
     * <p>
     * todo
     * 132. Palindrome Partitioning II
     *
     * @param s
     * @return
     */
    public int minCut(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        }
        int n = s.length();
        int[] cut = new int[n];

        boolean[][] dp = new boolean[n][n];
        for (int i = 0; i < n; i++) {
            int cutNum = i;
            for (int j = 0; j <= i; j++) {
                if (s.charAt(j) == s.charAt(i) && (i - j < 2 || dp[j + 1][i - 1])) {
                    dp[j][i] = true;
                    cutNum = j == 0 ? 0 : Math.min(cutNum, cut[j - 1] + 1);
                }
            }
            cut[i] = cutNum;
        }
        return cut[n - 1];


    }

    /**
     * 141. Linked List Cycle
     *
     * @param head
     * @return
     */
    public boolean hasCycle(ListNode head) {
        if (head == null || head.next == null) {
            return false;
        }
        ListNode slow = head;

        ListNode fast = head;
        while (fast.next != null && fast.next.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast) {
                return true;
            }
        }
        return false;
    }


    // -------树的遍历---//

    /**
     * 142. Linked List Cycle II
     *
     * @param head
     * @return
     */
    public ListNode detectCycle(ListNode head) {
        if (head == null || head.next == null) {
            return null;
        }
        ListNode fast = head;
        ListNode slow = head;
        while (fast.next != null && fast.next.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast) {
                fast = head;
                while (fast != slow) {
                    fast = fast.next;
                    slow = slow.next;
                }
                return fast;
            }
        }
        return null;
    }

    // ----树的排序 //

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
                stack.push(p);
                ans.addFirst(p.val);
                p = p.right;
            } else {
                p = stack.pop();
                p = p.left;
            }
        }
        return ans;
    }

    /**
     * 插入排序
     * 147. Insertion Sort List
     *
     * @param head
     * @return
     */
    public ListNode insertionSortList(ListNode head) {
        return null;
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
        ListNode slow = head;
        ListNode fast = head;
        while (fast.next != null && fast.next.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode node = slow.next;
        slow.next = null;
        ListNode l1 = this.sortList(head);
        ListNode l2 = this.sortList(node);
        return this.sortListNode(l1, l2);

    }


    // Olog(N)复杂度 //

    // ------旋转数组 找最小值-----//

    private ListNode sortListNode(ListNode l1, ListNode l2) {
        if (l1 == null) {
            return l2;
        }
        if (l2 == null) {
            return l1;
        }
        if (l1.val < l2.val) {
            l1.next = this.sortListNode(l1.next, l2);
            return l1;
        } else {
            l2.next = this.sortListNode(l1, l2.next);
            return l2;
        }
    }

    /**
     * key point: 左边界可能存在两个区域 所以无法很好的做边界条件。
     * 故使用右边界
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
            if (nums[mid] <= nums[right]) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return nums[left];
    }

    /**
     * todo 未来需要三种
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
            int mid = left + (right - left) / 2;
            if (nums[mid] == nums[right]) {
                right--;
            } else if (nums[mid] < nums[right]) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return nums[left];

    }


    // -------- 逆波兰数---//

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


    // --两个数之和--- //

    /**
     * 150. Evaluate Reverse Polish Notation
     *
     * @param tokens
     * @return
     */
    public int evalRPN(String[] tokens) {
        if (tokens == null || tokens.length == 0) {
            return 0;
        }
        Stack<Integer> stack = new Stack<>();
        for (String token : tokens) {
            if (token.equals("+")) {
                int second = stack.pop();
                int first = stack.pop();
                stack.push(first + second);
            } else if (token.equals("-")) {
                int second = stack.pop();
                int first = stack.pop();
                stack.push(first - second);

            } else if (token.equals("*")) {
                int second = stack.pop();
                int first = stack.pop();
                stack.push(first * second);
            } else if (token.equals("/")) {
                int second = stack.pop();
                int first = stack.pop();
                stack.push(first / second);
            } else {
                stack.push(Integer.parseInt(token));
            }
        }
        return stack.pop();
    }


    // ----滑动窗口系列------- //
    // 滑动窗口固定解题公式-- //

    /**
     * 167. Two Sum II - Input array is sorted
     *
     * @param numbers
     * @param target
     * @return
     */
    public int[] twoSum(int[] numbers, int target) {
        if (numbers == null || numbers.length == 0) {
            return new int[]{};
        }
        int[] ans = new int[2];
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < numbers.length; i++) {
            if (map.containsKey(target - numbers[i])) {
                int first = map.get(target - numbers[i]) + 1;
                int second = i + 1;
                ans[0] = first;
                ans[1] = second;
                return ans;
            }
            map.put(numbers[i], i);
        }
        return ans;
    }

    /**
     * 159 Longest Substring with At Most Two Distinct Characters
     *
     * @param s: a string
     * @return: the length of the longest substring T that contains at most 2 distinct characters
     */
    public int lengthOfLongestSubstringTwoDistinct(String s) {
        // Write your code here
        if (s == null || s.isEmpty()) {
            return 0;
        }
        HashMap<Character, Integer> map = new HashMap<>();
        int left = 0;
        int result = 0;
        for (int i = 0; i < s.length(); i++) {
            char localCharacter = s.charAt(i);
            Integer num = map.getOrDefault(localCharacter, 0);
            num++;
            map.put(localCharacter, num);
            while (map.size() > 2) {
                char leftCharacter = s.charAt(left);

                Integer integer = map.get(leftCharacter);

                integer--;

                map.put(leftCharacter, integer);
                if (integer == 0) {
                    map.remove(leftCharacter);
                }
                left++;
            }
            result = Math.max(result, i - left + 1);
        }
        return result;
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
        int len = Integer.MAX_VALUE;


        int end = 0;

        int start = 0;

        int sum = 0;
        while (end < nums.length) {
            sum += nums[end++];

            while (sum > s) {
                len = Math.min(len, end - start);

                sum -= nums[start++];
            }
        }
        return len;
    }


    // ---字符串反转问题--//

    /**
     * 186 Reverse Words in a String II
     * 不能使用
     *
     * @param str: a string
     * @return: return a string
     */
    public char[] reverseWords(char[] str) {
        // write your code here
        if (str == null || str.length == 0) {
            return new char[]{};
        }
        return null;
    }

    // ----房屋抢劫----//

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
        int column = nums.length;

        int[] dp = new int[column];

        dp[0] = nums[0];

        for (int i = 1; i < nums.length; i++) {
            if (i == 1) {
                dp[i] = Math.max(0, nums[1]);
            } else {
                dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i]);
            }

        }
        return dp[column - 1];

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
        return Math.max(intervalRob(nums, 0, nums.length - 2),
                intervalRob(nums, 1, nums.length - 1));
    }

    private int intervalRob(int[] nums, int start, int end) {
        int robPre = 0;

        int robNow = 0;
        for (int i = start; i <= end; i++) {
            int tmp = robPre;
            robPre = Math.max(robPre, robNow);

            robNow = tmp + nums[i];
        }

        return Math.max(robPre, robNow);
    }


    // ----区间---//

    /**
     * 201. Bitwise AND of Numbers Range
     *
     * @param m
     * @param n
     * @return
     */
    public int rangeBitwiseAnd(int m, int n) {
        return 0;
    }


    // ---课程调度---//

    /**
     * 207. Course Schedule
     *
     * @param numCourses
     * @param prerequisites
     * @return
     */
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        Arrays.sort(prerequisites, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {

                return 0;
            }
        });
        return false;
    }


    /**
     * 210. Course Schedule II
     *
     * @param numCourses
     * @param prerequisites
     * @return
     */
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        return new int[]{};
    }


    //--- 排列组合问题----//

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

    private void combinationSum3(List<List<Integer>> ans, List<Integer> integers, int start, int k, int n) {
        if (n == 0 && integers.size() == k) {
            ans.add(new ArrayList<>(integers));
        }
        for (int i = start; i <= 9 && i <= n; i++) {
            integers.add(i);
            combinationSum3(ans, integers, i + 1, k, n - i);
            integers.remove(integers.size() - 1);
        }
    }

    // ---逆波兰问题---- //

    /**
     * 224. Basic Calculator
     *
     * @param s
     * @return
     */
    public int calculate(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        }
        Stack<Integer> numberStack = new Stack<>();
        int result = 0;
        int sign = 1;
        int endIndex = 0;
        while (endIndex < s.length()) {
            if (Character.isDigit(s.charAt(endIndex))) {
                int tmp = 0;
                while (endIndex < s.length() && Character.isDigit(s.charAt(endIndex))) {
                    tmp = tmp * 10 + Character.getNumericValue(s.charAt(endIndex++));
                }
                result += sign * tmp;
            } else {
                if (s.charAt(endIndex) == '+') {
                    sign = 1;
                } else if (s.charAt(endIndex) == '-') {
                    sign = -1;
                } else if (s.charAt(endIndex) == '(') {
                    numberStack.push(result);
                    numberStack.push(sign);
                    result = 0;
                    sign = 1;
                } else if (s.charAt(endIndex) == ')') {
                    result = numberStack.pop() * result + numberStack.pop();
                }
                endIndex++;
            }
        }
        return result;
    }


    /**
     * 227. Basic Calculator II
     * keyCase : 遍历字符串时 延迟定义操作符号
     *
     * @param s
     * @return
     */
    public int calculateII(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        }
        Stack<Integer> digitStack = new Stack<>();
        char sign = '+';
        int tmp = 0;
        for (int i = 0; i < s.length(); i++) {
            if (Character.isDigit(s.charAt(i))) {
                tmp = tmp * 10 + Character.getNumericValue(s.charAt(i));
            }
            if (i == s.length() - 1 || (!Character.isDigit(s.charAt(i)) && s.charAt(i) != ' ')) {
                if (sign == '+') {
                    digitStack.push(tmp);
                } else if (sign == '-') {
                    digitStack.push(-tmp);
                } else if (sign == '*') {
                    digitStack.push(digitStack.pop() * tmp);
                } else {
                    digitStack.push(digitStack.pop() / tmp);
                }
                tmp = 0;
                sign = s.charAt(i);
            }
        }
        int result = 0;
        for (Integer integer : digitStack) {
            result += integer;
        }
        return result;
    }


    // -----树的公共祖先问题----//

    /**
     * 235. Lowest Common Ancestor of a Binary Search Tree
     *
     * @param root
     * @param p
     * @param q
     * @return
     */
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || p == null || q == null) {
            return null;
        }
        if (root == p || root == q) {
            return root;
        }
        if (root.val > p.val && root.val > q.val) {
            return lowestCommonAncestor(root.left, p, q);
        } else if (root.val < p.val && root.val < q.val) {
            return lowestCommonAncestor(root.right, p, q);
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
    public TreeNode lowestCommonAncestorII(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || p == null || q == null) {
            return null;
        }
        if (root == p || root == q) {
            return root;
        }
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestorII(root.right, p, q);
        if (left != null && right != null) {
            return root;
        } else if (left != null) {
            return left;
        } else {
            return right;
        }
    }

    // -----滑动窗口系列----///

    /**
     * todo 太难
     * 239. Sliding Window Maximum
     *
     * @param nums
     * @param k
     * @return
     */
    public int[] maxSlidingWindow(int[] nums, int k) {
        if (nums == null || nums.length == 0) {
            return new int[]{};
        }
        // 用来存放满足窗口限制条件的值
        List<Integer> tmp = new ArrayList<>();

        // 用作窗口  窗口坐标对应的值 从大到小
        // 窗口 具有边界
        LinkedList<Integer> list = new LinkedList<>();
        for (int i = 0; i < nums.length; i++) {
            int begin = i - k + 1;


            // 当前计算出得左边界 与 窗口的左边界比较是否超过
            while (!list.isEmpty() && begin > list.peekFirst()) {
                list.pollFirst();
            }

            // 更新窗口内的值　
            while (!list.isEmpty() && nums[list.peekLast()] <= nums[i]) {
                list.pollLast();
            }
            list.add(i);

            // 当前下标到达了窗口的左边界
            if (begin >= 0) {
                tmp.add(nums[list.peekFirst()]);
            }
        }
        int[] result = new int[tmp.size()];
        for (int i = 0; i < tmp.size(); i++) {
            result[i] = tmp.get(i);
        }
        return result;
    }


    // --排序矩阵搜索目标值----//


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
            int value = matrix[i][j];
            if (value == target) {
                return true;
            } else if (value < target) {
                i++;
            } else {
                j--;
            }
        }
        return false;
    }


    // ---字符之间的差距系列问题---- //

    /**
     * #243 Shortest Word Distance
     *
     * @param words: a list of words
     * @param word1: a string
     * @param word2: a string
     * @return: the shortest distance between word1 and word2 in the list
     */
    public int shortestDistance(String[] words, String word1, String word2) {
        if (words == null || words.length == 0) {
            return -1;
        }
        int result = Integer.MAX_VALUE;
        int index1 = -1;
        int index2 = -1;
        for (int i = 0; i < words.length; i++) {
            String tmp = words[i];
            if (tmp.equals(word1)) {
                index1 = i;
            }
            if (tmp.equals(word2)) {
                index2 = i;
            }
            if (index1 != -1 && index2 != -1) {
                result = Math.min(result, Math.abs(index1 - index2));
            }
        }
        // Write your code here
        return result;
    }


    /**
     * #243 Shortest Word DistanceII
     * 在243基础上优化
     *
     * @param words
     * @param word1
     * @param word2
     * @return
     */
    public int shortestDistanceII(String[] words, String word1, String word2) {
        if (words == null || words.length == 0) {
            return 0;
        }
        HashMap<String, List<Integer>> map = new HashMap<>();
        for (int i = 0; i < words.length; i++) {
            String word = words[i];
            boolean match = word.equals(word1) || word.equals(word2);
            if (match) {
                List<Integer> index = map.getOrDefault(word, new ArrayList<>());
                index.add(i);
                map.put(word, index);
            }
        }
        List<Integer> index1 = map.get(word1);
        List<Integer> index2 = map.get(word2);

        int result = Integer.MAX_VALUE;
        for (Integer integer : index1) {
            for (Integer value : index2) {
                result = Math.min(result, Math.abs(integer - value));
            }
        }
        return result;
    }


    /**
     * #244 Shortest Word DistanceII
     * * 在243基础上优化
     *
     * @param words
     * @param word1
     * @param word2
     * @return
     */
    public int shortestDistanceIII(String[] words, String word1, String word2) {
        if (words == null || words.length == 0) {
            return 0;
        }
        int result = Integer.MAX_VALUE;

        int index1 = -1;

        int index2 = -1;

        for (int i = 0; i < words.length; i++) {
            int tmp = index1;

            String word = words[i];

            if (word.equals(word1)) {
                index1 = i;
            }
            if (word.equals(word2)) {
                index2 = i;
            }
            if (index1 != -1 && index2 != -1) {
                if (word1.equals(word2) && tmp != -1 && tmp != index1) {
                    result = Math.min(result, Math.abs(tmp - index2));
                } else if (index1 != index2) {
                    result = Math.min(result, Math.abs(index1 - index2));
                }
            }
        }
        return result;
    }


}