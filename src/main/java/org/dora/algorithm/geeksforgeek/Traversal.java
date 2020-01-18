package org.dora.algorithm.geeksforgeek;

import org.dora.algorithm.datastructe.ListNode;
import org.dora.algorithm.datastructe.Node;
import org.dora.algorithm.datastructe.TreeNode;

import java.util.*;

/**
 * 各种遍历方法
 *
 * @author dora
 * @date 2019/11/5
 */
public class Traversal {

    public static void main(String[] args) {
        Traversal traversal = new Traversal();
//        String num = "105";
//        int target = 5;
//        traversal.addOperatorsV2(num, target);
        int[] ans = new int[]{0, 1, 0, 3, 12};
        traversal.moveZeroes(ans);
    }

    /**
     * 79. Word Search
     * <p>
     * 深度优先遍历
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
                if (board[i][j] == word.charAt(0) && this.checkExist(used, board, i, j, 0, word)) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean checkExist(boolean[][] used, char[][] board,
                               int i, int j, int index, String word) {
        if (index == word.length()) {
            return true;
        }
        if (i < 0 || i >= board.length || j < 0
                || j >= board[0].length || used[i][j] || board[i][j] != word.charAt(index)) {
            return false;
        }
        used[i][j] = true;
        if (this.checkExist(used, board, i - 1, j, index + 1, word) ||
                this.checkExist(used, board, i + 1, j, index + 1, word) ||
                this.checkExist(used, board, i, j - 1, index + 1, word) ||
                this.checkExist(used, board, i, j + 1, index + 1, word)) {
            return true;
        }
        used[i][j] = false;
        return false;
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
        TreeNode p = root;
        TreeNode prev = null;
        TreeNode first = null;
        TreeNode second = null;
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
        if (first != null) {
            int val = first.val;
            first.val = second.val;
            second.val = val;
        }
    }

    /**
     * 102. Binary Tree Level Order Traversal
     */
    public List<List<Integer>> levelOrder(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        List<List<Integer>> ans = new ArrayList<>();

        Deque<TreeNode> deque = new LinkedList<>();

        deque.push(root);

        while (!deque.isEmpty()) {

            int size = deque.size();

            List<Integer> tmp = new ArrayList<>();

            for (int i = 0; i < size; i++) {

                TreeNode poll = deque.poll();

                tmp.add(poll.val);

                if (poll.left != null) {
                    deque.add(poll.left);
                }

                if (poll.right != null) {
                    deque.add(poll.right);
                }
            }
            ans.add(tmp);
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

        Deque<TreeNode> list = new LinkedList<>();

        list.add(root);

        boolean leftToRight = true;

        while (!list.isEmpty()) {
            LinkedList<Integer> tmp = new LinkedList<>();
            int size = list.size();
            for (int i = 0; i < size; i++) {
                TreeNode poll = list.poll();
                if (leftToRight) {
                    tmp.addLast(poll.val);
                } else {
                    tmp.addFirst(poll.val);
                }
                if (poll.left != null) {
                    list.add(poll.left);
                }
                if (poll.right != null) {
                    list.add(poll.right);
                }

            }
            ans.add(tmp);
            leftToRight = !leftToRight;
        }
        return ans;
    }

    /**
     * 107. Binary Tree Level Order Traversal II
     *
     * @param root
     * @return
     */
    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        LinkedList<List<Integer>> ans = new LinkedList<>();

        Deque<TreeNode> deque = new LinkedList<>();

        deque.add(root);

        while (!deque.isEmpty()) {
            List<Integer> tmp = new ArrayList<>();

            int size = deque.size();

            for (int i = 0; i < size; i++) {
                TreeNode poll = deque.poll();
                tmp.add(poll.val);
                if (poll.left != null) {
                    deque.add(poll.left);
                }
                if (poll.right != null) {
                    deque.add(poll.right);
                }
            }
            ans.addFirst(tmp);
        }
        return ans;
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


        TreeNode p = root;

        stack.push(p);

        TreeNode prev = null;

        while (!stack.isEmpty()) {
            p = stack.pop();
            if (p.right != null) {
                stack.push(p.right);
            }
            if (p.left != null) {
                stack.push(p.left);
            }

            if (prev != null) {
                prev.right = p;

                prev.left = null;
            }
            prev = p;
        }
    }

    /**
     * todo
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

        for (int i = 1; i <= m; i++) {
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
     * 116. Populating Next Right Pointers in Each Node
     *
     * @param root
     * @return
     */
    public Node connect(Node root) {
        if (root == null) {
            return null;
        }
        Node p = root;
        while (p.left != null) {
            Node nextLevel = p.left;

            while (p != null) {
                p.left.next = p.right;

                if (p.next != null) {
                    p.right.next = p.next.left;
                }
                p = p.next;
            }

            p = nextLevel;
        }
        return root;
    }

    /**
     * todo O1空间
     * 关键点在于: 找到并设置每一层的头结点
     * 117. Populating Next Right Pointers in Each Node II
     *
     * @param root
     * @return
     */
    public Node connectII(Node root) {
        if (root == null) {
            return null;
        }
        Node p = root;

        Node levelHead = null;

        Node levelPrev = null;

        while (p != null) {

            while (p != null) {
                if (p.left != null) {
                    if (levelPrev != null) {
                        levelPrev.next = p.left;
                    } else {
                        levelHead = p.left;
                    }
                    levelPrev = p.left;
                }
                if (p.right != null) {
                    if (levelPrev != null) {
                        levelPrev.next = p.right;
                    } else {
                        levelHead = p.right;
                    }
                    levelPrev = p.right;
                }
                p = p.next;
            }
            p = levelHead;

            levelHead = null;

            levelPrev = null;

        }
        return root;
    }

    /**
     * 118. Pascal's Triangle
     *
     * @param numRows
     * @return
     */
    public List<List<Integer>> generate(int numRows) {
        if (numRows < 0) {
            return new ArrayList<>();
        }
        List<List<Integer>> ans = new ArrayList<>();

        for (int i = 0; i <= numRows - 1; i++) {

            List<Integer> tmp = new ArrayList<>();

            tmp.add(1);


            for (int j = 1; j < i; j++) {
                int value = ans.get(i - 1).get(j - 1) + ans.get(i - 1).get(j);

                tmp.add(value);

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
                int value = ans.get(j) + ans.get(j - 1);
                ans.set(j, value);
            }
            if (i > 0) {
                ans.add(1);
            }
        }
        return ans;
    }

    /**
     * 125. Valid Palindrome
     *
     * @param s
     * @return
     */
    public boolean isPalindrome(String s) {
        if (s == null) {
            return true;
        }
        s = s.trim();
        if (s.isEmpty()) {
            return true;
        }
        int left = 0;

        int right = s.length() - 1;
        while (left < right) {
            while (left < right && !Character.isLetterOrDigit(s.charAt(left))) {
                left++;
            }
            while (left < right && !Character.isLetterOrDigit(s.charAt(right))) {
                right--;
            }
            if (Character.toLowerCase(s.charAt(left)) == Character.toLowerCase(s.charAt(right))) {
                left++;
                right--;
            } else {
                return false;
            }
        }
        return true;
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
        return this.sumNumbers(root.left, root.val) + this.sumNumbers(root.right, root.val);
    }

    private int sumNumbers(TreeNode root, int val) {
        if (root == null) {
            return 0;
        }
        if (root.left == null && root.right == null) {
            return val * 10 + root.val;
        }
        return this.sumNumbers(root.left, val * 10 + root.val)
                + this.sumNumbers(root.right, val * 10 + root.val);
    }

    /**
     * 138. Copy List with Random Pointer
     *
     * @param head
     * @return
     */
    public Node copyRandomList(Node head) {
        if (head == null) {
            return null;
        }
        Node current = head;
        while (current != null) {
            Node tmp = current.next;

            Node next = new Node(current.val, current.next, null);

            current.next = next;

            current = tmp;
        }
        current = head;
        while (current != null) {
            Node random = current.random;

            if (random != null) {
                current.next.random = random.next;
            }
            current = current.next.next;
        }

        current = head;

        Node copyHead = head.next;
        while (current.next != null) {
            Node tmp = current.next;

            current.next = tmp.next;

            current = tmp;
        }
        return copyHead;
    }


    /**
     * todo
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

        ListNode middle = slow;

        /**
         * 根据LeetCode
         */
        ListNode current = slow.next;


        ListNode prev = this.reverseListNode(current, null);


        slow.next = prev;


        slow = head;


        fast = middle.next;


        while (slow != middle) {
            middle.next = fast.next;

            fast.next = slow.next;

            slow.next = fast;


            slow = fast.next;

            fast = middle.next;

        }
    }

    private ListNode reverseListNode(ListNode start, ListNode end) {
        if (start == end) {
            return null;
        }
        ListNode prev = null;
        while (start != end) {
            ListNode tmp = start.next;
            start.next = prev;
            prev = start;
            start = tmp;
        }
        return prev;
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
        ListNode p1 = headA;

        ListNode p2 = headB;

        while (p1 != p2) {

            p1 = p1 == null ? headB : p1.next;
            p2 = p2 == null ? headB : p2.next;
        }
        return p1;
    }

    /**
     * 165. Compare Version Numbers
     *
     * @param version1
     * @param version2
     * @return
     */
    public int compareVersion(String version1, String version2) {
        if (version1 == null || version2 == null) {
            return -1;
        }
        String[] split1 = version1.split("\\.");
        String[] split2 = version2.split("\\.");
        int index1 = 0;
        int index2 = 0;
        while (index1 < split1.length || index2 < split2.length) {
            Integer value1 = index1 == split1.length ? 0 : Integer.parseInt(split1[index1++]);
            Integer value2 = index2 == split2.length ? 0 : Integer.parseInt(split2[index2++]);
            if (!value1.equals(value2)) {
                return value1.compareTo(value2);
            }
        }
        return 0;
    }

    /**
     * 168. Excel Sheet Column Title
     *
     * @param n
     * @return
     */
    public String convertToTitle(int n) {
        if (n <= 0) {
            return "";
        }
        StringBuilder builder = new StringBuilder();
        while (n != 0) {
            char val = (char) (((n - 1) % 26) + 'A');
            builder.append(val);
            n = (n - 1) / 26;
        }
        return builder.reverse().toString();
    }


    /**
     * 171. Excel Sheet Column Number
     *
     * @param s
     * @return
     */
    public int titleToNumber(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        }
        int result = 0;
        for (char tmp : s.toCharArray()) {
            result = result * 26 + (tmp - 'A' + 1);
        }
        return result;
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
        k %= nums.length;
        reverseArray(nums, 0, nums.length - 1);
        reverseArray(nums, 0, k - 1);
        reverseArray(nums, k, nums.length - 1);

    }

    private void reverseArray(int[] nums, int start, int end) {
        if (start > end) {
            return;
        }
        for (int i = start; i <= (start + end) / 2; i++) {
            swap(nums, i, start + end - i);
        }
    }

    private void swap(int[] nums, int start, int end) {
        int val = nums[start];
        nums[start] = nums[end];
        nums[end] = val;
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

        LinkedList<TreeNode> deque = new LinkedList<>();

        deque.add(root);

        while (!deque.isEmpty()) {

            int size = deque.size();

            for (int i = 0; i < size; i++) {
                TreeNode poll = deque.poll();

                if (i == size - 1) {
                    ans.add(poll.val);
                }
                if (poll.left != null) {
                    deque.add(poll.left);
                }
                if (poll.right != null) {
                    deque.add(poll.right);
                }
            }
        }
        return ans;
    }


    /**
     * 203. Remove Linked List Element
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

        ListNode dummy = root;

        while (head != null) {

            if (head.val != val) {

                dummy.next = head;

                dummy = dummy.next;
            }
            head = head.next;
        }
        dummy.next = null;
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
        if (s.length() != t.length()) {
            return false;
        }
        if (s.equals(t)) {
            return true;
        }
        int[] hash1 = new int[256];

        int[] hash2 = new int[256];

        for (int i = 0; i < s.length(); i++) {
            if (hash1[s.charAt(i)] != hash2[t.charAt(i)]) {
                return false;
            }
            hash1[s.charAt(i)] = i;
            hash2[t.charAt(i)] = i;
        }
        return true;
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

        ListNode node = reverseList(head.next);

        head.next.next = head;

        head.next = null;

        return node;
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
            return 0;
        }
        PriorityQueue<Integer> priorityQueue = new PriorityQueue<Integer>(k, (o1, o2) -> o2.compareTo(o1));
        for (int num : nums) {
            priorityQueue.add(num);
        }
        for (int i = 0; i < k - 1; i++) {
            priorityQueue.remove();
        }
        return priorityQueue.remove();
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
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            if (map.containsKey(num)) {
                return true;
            }
            map.put(num, 1);
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
        HashMap<Integer, Integer> hashMap = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (hashMap.containsKey(nums[i])) {
                int diff = i - hashMap.get(nums[i]);
                if (diff <= k) {
                    return true;
                }
            }
            hashMap.put(nums[i], i);
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
        for (int i = t - 1; i < nums.length; i++) {
            for (int j = 0; j < i; j++) {
                long diff = (long) nums[i] - (long) nums[j];
                if (Math.abs(diff) <= t) {
                    return true;
                }

            }
        }
        return false;
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
        return 1 + countNodes(root.left) + countNodes(root.right);
    }

    public TreeNode invertTree(TreeNode root) {
        if (root == null) {
            return null;
        }
        TreeNode tmp = root.left;
        root.left = root.right;
        root.right = tmp;
        invertTree(root.left);
        invertTree(root.right);
        return root;
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
            ans.add(getRange(nums[i], nums[right]));
            i = right;
        }
        return ans;
    }


    private String getRange(long start, long end) {
        if (start == end) {
            return String.valueOf(start);
        }
        return start + "->" + end;
    }


    public List<String> findMissingRanges(int[] nums, int lower, int upper) {
        List<String> ans = new ArrayList<>();
        if (nums == null || nums.length == 0) {
            ans.add(getRange(lower, upper));
            return ans;
        }
        if (nums[0] > lower) {
            ans.add(getRange(lower, nums[0] - 1));
        }
        long prev = nums[0];
        for (int i = 1; i <= nums.length; i++) {
            long current = i == nums.length ? upper + 1 : nums[i];
            long diff = current - prev;
            if (diff > 1) {
                ans.add(getRange(prev + 1, current - 1));
            }
            prev = current;
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

        if (head == null || head.next == null) {
            return true;
        }
        ListNode fast = head;
        ListNode mid = head;
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            mid = mid.next;
        }
        ListNode reverseNode = reverseList(mid.next);
        mid.next = reverseNode;

        mid = mid.next;
        while (mid != null) {
            if (head.val != mid.val) {
                return false;
            }

            head = head.next;
            mid = mid.next;
        }
        return true;
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
        ListNode next = node.next;
        if (next == null) {
            node = null;
            return;
        }
        node.val = next.val;
        if (next.next == null) {
            node.next = null;
            next = null;
        } else {
            node.next = next.next;
            next = null;
        }

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
            hash[s.charAt(i) - 'a']++;
            hash[t.charAt(i) - 'a']--;
        }
        for (int i = 0; i < 256; i++) {
            if (hash[i] != 0) {
                return false;
            }
        }
        return true;
    }


    /**
     * 241. Different Ways to Add Parentheses
     *
     * @param input
     * @return
     */
    public List<Integer> diffWaysToCompute(String input) {
        if (input == null || input.isEmpty()) {
            return new ArrayList<>();
        }
        List<String> ans = new ArrayList<>();
        for (int i = 0; i < input.length(); i++) {
            int tmp = 0;
            while (i < input.length() && Character.isDigit(input.charAt(i))) {
                tmp = tmp * 10 + Character.getNumericValue(input.charAt(i));
                i++;
            }
            ans.add(String.valueOf(tmp));
            if (i != input.length()) {
                ans.add(input.substring(i, i + 1));
            }
        }
        return computeWays(ans, 0, ans.size() - 1);
    }

    private List<Integer> computeWays(List<String> ans, int start, int end) {
        List<Integer> result = new ArrayList<>();
        if (start == end) {
            result.add(Integer.parseInt(ans.get(start)));
            return result;
        }
        for (int i = start + 1; i <= end - 1; i = i + 2) {
            List<Integer> leftWays = computeWays(ans, start, i - 1);
            List<Integer> rightWays = computeWays(ans, i + 1, end);
            String sign = ans.get(i);
            for (Integer leftWay : leftWays) {
                for (Integer rightWay : rightWays) {
                    if (sign.equals("+")) {
                        result.add(leftWay + rightWay);
                    } else if (sign.equals("-")) {
                        result.add(leftWay - rightWay);
                    } else if (sign.equals("*")) {
                        result.add(leftWay * rightWay);
                    } else {
                        result.add(leftWay / rightWay);
                    }

                }
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
        intervalPaths(ans, root, "");
        return ans;
    }

    private void intervalPaths(List<String> ans, TreeNode root, String s) {
        String tmp = s + root.val;

        if (root.left == null && root.right == null) {
            ans.add(tmp);
            return;
        }
        if (root.left != null) {
            intervalPaths(ans, root.left, tmp + "->");
        }
        if (root.right != null) {
            intervalPaths(ans, root.right, tmp + "->");
        }

    }


    /**
     * 278. First Bad Version
     *
     * @param n
     * @return
     */
    public int firstBadVersion(int n) {
        if (n <= 0) {
            return -1;
        }
        int left = 1;
        int right = n;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (!isBadVersion(mid)) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return left;
    }

    private boolean isBadVersion(int version) {
        return true;
    }

    /**
     * todo 优化间复杂度
     * 转化成 O(1) 时间复杂度
     * <p>
     * result = 1 + (n-1) mod 9;
     * 258. Add Digits
     *
     * @param num
     * @return
     */
    public int addDigits(int num) {
        if (num <= 0) {
            return 0;
        }
        while (num / 10 > 0) {
            int result = 0;
            int tmp = num;
            while (tmp != 0) {
                result += tmp % 10;
                tmp /= 10;
            }
            num = result;
        }
        return num;
    }


    /**
     * 280 Wiggle Sort
     *
     * @param nums: A list of integers
     * @return: nothing
     */
    public void wiggleSort(int[] nums) {
        // write your code here
        if (nums == null || nums.length == 0) {
            return;
        }
        for (int i = 1; i < nums.length; i++) {
            boolean odd = nums[i] % 2 == 1;

            boolean correctFormat = true;

            if (odd && nums[i] < nums[i - 1]) {
                correctFormat = false;
            }
            if (!odd && nums[i] > nums[i - 1]) {
                correctFormat = false;
            }

            if (!correctFormat) {
                swap(nums, i - 1, i);
            }
        }
    }

    /**
     * 283. Move Zeroes
     *
     * @param nums
     */
    public void moveZeroes(int[] nums) {
        if (nums == null || nums.length == 0) {
            return;
        }
        int index = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != 0) {
                swap(nums, index++, i);
            }
        }
    }

    /**
     * 259 3Sum Smaller
     *
     * @param nums:   an array of n integers
     * @param target: a target
     * @return: the number of index triplets satisfy the condition nums[i] + nums[j] + nums[k] < target
     */
    public int threeSumSmaller(int[] nums, int target) {
        // Write your code here
        if (nums == null || nums.length == 0) {
            return Integer.MIN_VALUE;
        }
        Arrays.sort(nums);

        int count = 0;

        for (int i = 0; i < nums.length - 2; i++) {

            if (nums[i] > target) {
                break;
            }
            int begin = i + 1;

            int end = nums.length - 1;

            while (begin < end) {
                int result = nums[i] + nums[begin] + nums[end];
                if (result <= target) {
                    count++;
                    begin++;
                } else {
                    end--;
                }
            }
        }
        return count;
    }


    /**
     * 282. Expression Add Operators
     *
     * @param num
     * @param target
     * @return
     */
    public List<String> addOperators(String num, int target) {
        if (num == null || num.isEmpty()) {
            return new ArrayList<>();
        }
        List<String> ans = new ArrayList<>();
        intervalHelper(ans, "", num, target, 0, 0, 0);
        return ans;
    }

    private void intervalHelper(List<String> ans, String s, String num, int target, int pos, long eval, long multi) {
        if (pos == num.length() && eval == target) {
            ans.add(s);
            return;
        }
        for (int i = pos; i < num.length(); i++) {
            if (i != pos && num.charAt(pos) == '0') {
                break;
            }
            long current = Long.parseLong(num.substring(pos, i + 1));
            if (pos == 0) {
                intervalHelper(ans, s + current, num, target, i + 1, eval + current, current);
            } else {
                intervalHelper(ans, s + "+" + current, num, target, i + 1, eval + current, current);
                intervalHelper(ans, s + "-" + current, num, target, i + 1, eval - current, -current);
                intervalHelper(ans, s + "*" + current, num, target, i + 1, eval - multi + multi * current, multi * current);
            }
        }
    }

    public List<String> addOperatorsV2(String num, int target) {
        if (num == null || num.isEmpty()) {
            return new ArrayList<>();
        }
        List<String> ans = new ArrayList<>();
        char[] path = new char[num.length() * 2 - 1];
        char[] digits = num.toCharArray();
        long n = 0;
        for (int i = 0; i < digits.length; i++) {
            n = n * 10 + digits[i] - '0';
            path[i] = digits[i];
            operatorsDfs(ans, path, i + 1, 0, n, digits, i + 1, target);
            if (n == 0) {
                break;
            }
        }
        return ans;
    }

    private void operatorsDfs(List<String> ans, char[] path, int len, long left, long cur, char[] digits, int pos, int target) {
        if (pos == path.length) {
            if (left + cur == target) {
                ans.add(new String(path, 0, len));
            }
            return;
        }
        long n = 0;
        int j = len + 1;
        for (int i = pos; i < digits.length; i++) {
            n = n * 10 + digits[i] - '0';
            path[j++] = digits[i];
            path[len] = '+';
            operatorsDfs(ans, path, j, left + cur, n, digits, i + 1, target);
            path[len] = '-';
            operatorsDfs(ans, path, j, left + cur, -n, digits, i + 1, target);
            path[len] = '*';
            operatorsDfs(ans, path, j, left, cur * n, digits, i + 1, target);
            if (digits[pos] == '0') {
                break;
            }
        }
    }


    /**
     * 285 Inorder Successor in BST
     *
     * @param root: The root of the BST.
     * @param p:    You need find the successor node of p.
     * @return: Successor of p.
     */
    public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
        // write your code here
        if (root == null || p == null) {
            return null;
        }
        Stack<TreeNode> stack = new Stack<>();
        boolean match = false;
        TreeNode tmp = root;
        while (!stack.isEmpty() || tmp != null) {
            while (tmp != null) {
                stack.push(tmp);
                tmp = tmp.left;
            }
            tmp = stack.pop();

            if (match) {
                return tmp;
            }

            if (tmp == p) {
                match = true;
            }
            tmp = tmp.right;
        }
        return null;
    }


    // ---------- 深度优先遍历DFS---------//

    /**
     * 139. Word Break
     *
     * @param s
     * @param wordDict
     * @return
     */
    public boolean wordBreak(String s, List<String> wordDict) {
        if (s == null) {
            return false;
        }
        if (wordDict.contains(s) || s.isEmpty()) {
            return true;
        }
        HashMap<String, Boolean> notIncluded = new HashMap<>();
        return this.wordBreakDFS(notIncluded, s, wordDict);
    }

    private boolean wordBreakDFS(HashMap<String, Boolean> notIncluded, String s, List<String> wordDict) {
        if (notIncluded.containsKey(s)) {
            return false;
        }
        if (s.isEmpty()) {
            return true;
        }
        for (String word : wordDict) {
            if (s.startsWith(word) && this.wordBreakDFS(notIncluded, s.substring(word.length()), wordDict)) {
                return true;
            }
        }
        notIncluded.put(s, false);
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
        if (s == null || wordDict == null) {
            return new ArrayList<>();
        }
        HashMap<String, List<String>> map = new HashMap<>();
        return this.wordBreakIIDFS(map, s, wordDict);
    }

    private List<String> wordBreakIIDFS(HashMap<String, List<String>> map, String s, List<String> wordDict) {
        if (map.containsKey(s)) {
            return map.get(s);
        }
        List<String> ans = new ArrayList<>();
        if (s.isEmpty()) {
            ans.add("");
            return ans;
        }
        for (String word : wordDict) {
            if (s.startsWith(word)) {
                List<String> list = this.wordBreakIIDFS(map, s.substring(word.length()), wordDict);
                for (String val : list) {
                    ans.add(word + (val.isEmpty() ? "" : " ") + val);
                }
            }
        }
        return ans;
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
        int row = board.length;
        int column = board[0].length;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                boolean isEdge = i == 0 || i == row - 1 || j == 0 || j == column - 1;
                if (board[i][j] == 'O' && isEdge) {
                    this.solveBoard(i, j, board);
                }
            }
        }
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (board[i][j] == 'o') {
                    board[i][j] = 'O';
                } else if (board[i][j] == 'O') {
                    board[i][j] = 'X';
                }
            }
        }
    }

    private void solveBoard(int i, int j, char[][] board) {
        if (i < 0 || i >= board.length || j < 0 || j >= board.length || board[i][j] != 'O') {
            return;
        }
        board[i][j] = 'o';
        this.solveBoard(i - 1, j, board);

        this.solveBoard(i + 1, j, board);

        this.solveBoard(i, j - 1, board);

        this.solveBoard(i, j + 1, board);
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

        int count = 0;

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (grid[i][j] == '1') {
                    checkIslands(grid, i, j);
                    count++;
                }
            }
        }
        return count;
    }

    private void checkIslands(char[][] grid, int i, int j) {
        if (i < 0 || i >= grid.length || j < 0 || j >= grid[i].length || grid[i][j] == '0') {
            return;
        }
        grid[i][j] = '0';
        checkIslands(grid, i - 1, j);
        checkIslands(grid, i + 1, j);
        checkIslands(grid, i, j - 1);
        checkIslands(grid, i, j + 1);
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
        List<String> ans = new ArrayList<>();

        boolean[][] used = new boolean[row][column];
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                intervalFindWords(i, j, board, used, trie, "", ans);
            }
        }
        return ans;
    }

    private void intervalFindWords(int i, int j, char[][] board, boolean[][] used, Trie trie,
                                   String s, List<String> ans) {
        if (i < 0 || i >= board.length || j < 0 || j >= board[i].length || used[i][j]) {
            return;
        }
        s += board[i][j];
        if (!trie.startsWith(s)) {
            return;
        }
        if (trie.search(s)) {
            ans.add(s);
        }
        used[i][j] = true;

    }


    /**
     * 268. Missing Number
     *
     * @param nums
     * @return
     */
    public int missingNumber(int[] nums) {
        if (nums == null || nums.length == 0) {
            return -1;
        }
        Set<Integer> set = new HashSet<>();

        for (int num : nums) {
            set.add(num);

        }

        for (int i = 0; i <= nums.length; i++) {
            if (!set.contains(i)) {
                return i;
            }
        }
        return -1;
    }


    /**
     * 269. Alien Dictionary
     * tod:
     *
     * @param words
     * @return
     */
    public String alienOrder(List<String> words) {
        return "";
    }


    /**
     * 270 Closest Binary Search Tree Value
     *
     * @param root:   the given BST
     * @param target: the given target
     * @return: the value in the BST that is closest to the target
     */
    public int closestValue(TreeNode root, double target) {
        // write your code here

        Stack<TreeNode> stack = new Stack<>();
        TreeNode p = root;
        double result = 0;
        while (!stack.isEmpty() || p != null) {
            while (p != null) {
                stack.push(p);
                p = p.left;
            }
            p = stack.pop();

            if (p.val == target) {
                return p.val;
            }
            if (Math.abs(p.val - target) < Math.abs(result - target)) {
                result = p.val;
            }
            p = p.right;
        }
        return (int) result;
    }


    public int closestValueV2(TreeNode root, double target) {
        double result = 0;

        while (root != null) {
            if (root.val == target) {
                return root.val;
            }
            if (Math.abs(root.val - target) < Math.abs(result - target)) {
                result = root.val;
            }
            if (root.val < target) {
                root = root.right;
            } else {
                root = root.left;
            }
        }
        return (int) result;
    }


    public int closestValueV3(TreeNode root, double target) {
        if (root.val == target) {
            return root.val;
        }
        int result = root.val;

        if (root.val > target && root.left != null) {

            int tmp = closestValueV3(root.left, target);

            if (Math.abs(root.val - target) >= Math.abs(tmp - target)) {
                result = tmp;
            }
        } else if (root.val < target && root.right != null) {

            int tmp = closestValueV3(root.right, target);

            if (Math.abs(root.val - target) >= Math.abs(tmp - target)) {
                result = tmp;
            }

        }
        return result;
    }

    public List<Integer> closestKValues(TreeNode root, double target, int k) {
        if (root == null) {
            return new ArrayList<>();
        }
        List<Integer> values = new ArrayList<>();

        inorderTraversal(values, root);

        int index = 0;

        int size = values.size();

        while (index < size) {

            Integer tmp = values.get(index);

            if (tmp >= target) {
                break;
            }
            index++;
        }
        if (index >= size) {
            return values.subList(size - k, size);
        }
        int left = index - 1;

        int right = index;

        List<Integer> ans = new ArrayList<>();

        for (int i = 0; i < k; i++) {

            if (left >= 0 && (right >= size || target - values.get(left) < values.get(right) - target)) {
                ans.add(values.get(left));
                left--;
            } else {
                ans.add(values.get(right));
                right++;
            }
        }
        return ans;

    }

    private void inorderTraversal(List<Integer> values, TreeNode root) {
        if (root == null) {
            return;
        }
        inorderTraversal(values, root.left);
        values.add(root.val);
        inorderTraversal(values, root.right);

    }


    /**
     * 273. Integer to English Words
     *
     * @param num
     * @return
     */
    public String numberToWords(int num) {
        if (num < 0) {
            return "";
        }
        if (num == 0) {
            return "Zero";
        }
        String[] v = new String[]{"Thousand", "Million", "Billion"};

        String res = convertNumber(num % 1000);

        for (int i = 0; i < 3; i++) {

            num = num / 1000;

            res = num % 1000 > 0 ? convertNumber(num % 1000) + " " + v[i] + " " + res : res;
        }

        res = res.trim();

        if (res.isEmpty()) {
            return "ZERO";
        }
        return res;
    }

    private String convertNumber(int num) {
        String[] v1 = new String[]{"", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"};

        String[] v2 = new String[]{"", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"};

        String res = "";

        int a = num / 100;

        int b = num % 100;

        int c = num % 10;

        res = b < 20 ? v1[b] : (v2[b / 10] + (c == 0 ? "" : " " + v1[c]));

        if (a != 0) {
            res = v1[a] + " Hundred" + (b != 0 ? " " + res : "");
        }
        return res;
    }


    /**
     * 286 Walls and Gates
     *
     * @param rooms: m x n 2D grid
     * @return: nothing
     */
    public void wallsAndGates(int[][] rooms) {
        // write your code here
        if (rooms == null || rooms.length == 0) {
            return;
        }
        int row = rooms.length;

        int column = rooms[0].length;

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                int val = rooms[i][j];
            }
        }


    }

}

