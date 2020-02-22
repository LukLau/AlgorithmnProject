package org.dora.algorithm.swordoffer;

import org.dora.algorithm.datastructe.ListNode;
import org.dora.algorithm.datastructe.RandomListNode;
import org.dora.algorithm.datastructe.TreeLinkNode;
import org.dora.algorithm.datastructe.TreeNode;

import java.util.*;

/**
 * @author liulu12@xiaomi.com
 * @date 2020/1/29
 */
public class SwordOfferV2 {

    public static void main(String[] args) {
        SwordOfferV2 swordOfferV2 = new SwordOfferV2();
        swordOfferV2.cutRope(5);
    }


    /**
     * 二维数组中的查找
     *
     * @param target
     * @param array
     * @return
     */
    public boolean Find(int target, int[][] array) {
        if (array == null || array.length == 0) {
            return false;
        }
        int row = array.length;
        int column = array[0].length;
        int i = row - 1;
        int j = 0;
        while (i >= 0 && j < column) {
            int val = array[i][j];
            if (val == target) {
                return true;
            } else if (val < target) {
                j++;
            } else {
                i--;
            }
        }
        return false;
    }

    /**
     * 替换空格
     *
     * @param str
     * @return
     */
    public String replaceSpace(StringBuffer str) {
        if (str == null) {
            return "";
        }
//        String s = str.toString();
//        int blankCount = 0;
//        for (int i = 0; i < s.length(); i++) {
//            if (s.charAt(i) == ' ') {
//                blankCount++;
//            }
//        }
//        if (blankCount == 0) {
//            return s;
//        }
        StringBuilder builder = new StringBuilder();
        int len = str.length() - 1;
        while (len >= 0) {
            if (str.charAt(len) != ' ') {
                builder.append(str.charAt(len));
            } else {
                builder.append("02%");
            }
            len--;
        }
        return builder.reverse().toString();
    }

    public String replaceSpaceV2(StringBuffer str) {
        if (str == null) {
            return "";
        }

        int len = str.length();

        if (len == 0) {
            return "";
        }
        int blankWordCount = 0;
        for (int i = 0; i < len; i++) {
            if (str.charAt(i) == ' ') {
                blankWordCount++;
            }
        }
        if (blankWordCount == 0) {
            return str.toString();
        }
        int newWordCount = len - blankWordCount + blankWordCount * 3;

        char[] words = new char[newWordCount];

        int index = len - 1;

        int side = newWordCount - 1;

        while (index >= 0) {
            if (str.charAt(index) != ' ') {
                words[side--] = str.charAt(index);
            } else {
                words[side--] = '0';
                words[side--] = '2';
                words[side--] = '%';
            }
            index--;
        }
        return String.valueOf(words);
    }


    /**
     * 从尾到头打印链表
     *
     * @param listNode
     * @return
     */
    public ArrayList<Integer> printListFromTailToHeadV2(ListNode listNode) {
        if (listNode == null) {
            return new ArrayList<>();
        }
        ListNode prev = null;

        ListNode head = listNode;
        while (head != null) {
            ListNode tmp = head.next;
            head.next = prev;
            prev = head;
            head = tmp;
        }
        ArrayList<Integer> ans = new ArrayList<>();
        while (prev != null) {
            ans.add(prev.val);
            prev = prev.next;
        }
        return ans;
    }


    public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {

        if (listNode == null) {
            return new ArrayList<>();
        }
        if (listNode.next == null) {
            ArrayList<Integer> ans = new ArrayList<>();
            ans.add(listNode.val);
            return ans;
        }
        ArrayList<Integer> integers = printListFromTailToHead(listNode.next);
        integers.add(listNode.val);
        return integers;

    }


    /**
     * 重建二叉树
     *
     * @param pre
     * @param in
     * @return
     */
    public TreeNode reConstructBinaryTree(int[] pre, int[] in) {
        if (pre == null || in == null) {
            return null;
        }
        if (pre.length != in.length) {
            return null;
        }
        return buildConstruct(0, pre, 0, in.length - 1, in);
    }

    private TreeNode buildConstruct(int preStart, int[] pre, int inStart, int inEnd, int[] in) {
        if (preStart < 0 || preStart >= pre.length || inStart > inEnd) {
            return null;
        }
        TreeNode root = new TreeNode(pre[preStart]);
        int index = 0;
        for (int i = inStart; i <= inEnd; i++) {
            if (in[i] == root.val) {
                index = i;
                break;
            }
        }
        root.left = buildConstruct(preStart + 1, pre, inStart, index - 1, in);
        root.right = buildConstruct(preStart + index - inStart + 1, pre, index + 1, inEnd, in);
        return root;
    }


    /**
     * todo: 跑完全部测试用例
     * 旋转数组的最小数字
     *
     * @param array
     * @return
     */
    public int minNumberInRotateArray(int[] array) {
        if (array == null || array.length == 0) {
            return 0;
        }
        int left = 0;
        int right = array.length - 1;
        while (left < right) {
            if (array[left] < array[right]) {
                return array[left];
            }
            int mid = left + (right - left) / 2;

            if (array[left] <= array[mid]) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return array[left];
    }


    /**
     * [1,2,3,1,1,1,1,1,1]
     * <p>
     * [2,3,1,1,1]
     *
     * @param array
     * @return
     */
    public int minNumberInRotateArrayV2(int[] array) {
        if (array == null || array.length == 0) {
            return 0;
        }
        int left = 0;
        int right = array.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (array[mid] <= array[right]) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return array[left];
    }


    /**
     * 斐波那切数列
     *
     * @param n
     * @return
     */
    public int Fibonacci(int n) {
        if (n == 0) {
            return 0;
        }
        if (n == 1) {
            return 1;
        }
        if (n == 2) {
            return 2;
        }
        return Fibonacci(n - 1) + Fibonacci(n - 2);
    }

    public int FibonacciV2(int n) {
        if (n == 0) {
            return 0;
        }
        if (n <= 2) {
            return 1;
        }
        int sum1 = 1;
        int sum2 = 1;
        int result = 0;
        for (int i = 3; i <= n; i++) {
            result = sum1 + sum2;
            sum1 = sum2;
            sum2 = result;
        }
        return result;
    }

    /**
     * 跳台阶
     *
     * @param target
     * @return
     */
    public int JumpFloor(int target) {
        if (target == 1) {
            return 1;
        }
        if (target == 2) {
            return 2;
        }
        return JumpFloor(target - 1) + JumpFloor(target - 2);
    }


    /**
     * 变态跳台阶
     *
     * @param target
     * @return
     */
    public int JumpFloorII(int target) {
        if (target <= 0) {
            return 0;
        }
        if (target == 1) {
            return 1;
        }
        if (target == 2) {
            return 2;
        }
        return 2 * JumpFloor(target - 1);
    }

    /**
     * 二进制中1的个数
     *
     * @param n
     * @return
     */
    public int NumberOf1(int n) {
        int result = 0;
        while (n != 0) {
            result++;
            n = n & (n - 1);
        }
        return result;
    }


    /**
     * 数值的整数次方
     *
     * @param base
     * @param exponent
     * @return
     */
    public double Power(double base, int exponent) {
        if (exponent == 0) {
            return 1;
        }
        if (exponent == 1) {
            return base;
        }
        if (exponent < 0) {
            exponent = -exponent;
            base = 1 / base;
        }
        return (exponent % 2 == 0) ? this.Power(base * base, exponent / 2) : base * this.Power(base * base, exponent / 2);
    }


    public double PowerV2(double base, int exponent) {
        if (exponent == 0) {
            return 1;
        }
        double result = 1.0;
        int absExponent = Math.abs(exponent);
        while (absExponent != 0) {
            if ((absExponent % 2) != 0) {
                result *= base;
            }
            base *= base;
            absExponent >>= 1;
        }
        return exponent < 0 ? 1 / result : result;
    }


    /**
     * 调整数组顺序使奇数位于偶数前面
     *
     * @param array
     */
    public void reOrderArray(int[] array) {
        if (array == null || array.length == 0) {
            return;
        }
        int len = array.length;
        int[] result = new int[len];
        int index = 0;
        for (int num : array) {
            if (num % 2 != 0) {
                result[index++] = num;
            }
        }
        for (int num : array) {
            if (num % 2 == 0) {
                result[index++] = num;
            }
        }
        array = result;
    }

    public void reOrderArrayV2(int[] array) {
        if (array == null || array.length == 0) {
            return;
        }
        for (int i = 0; i < array.length; i++) {
            for (int j = array.length - 1; j > i; j--) {
                if (array[j] % 2 == 1 && array[j - 1] % 2 == 0) {
                    swap(array, j, j - 1);
                }
            }
        }
    }

    private void swap(int[] nums, int i, int j) {
        int val = nums[i];
        nums[i] = nums[j];
        nums[j] = val;
    }


    /**
     * 1 - > 2 - > 3 - > 4
     * 2
     * 链表中倒数第k个结点
     *
     * @param head
     * @param k
     * @return
     */
    public ListNode FindKthToTail(ListNode head, int k) {
        if (head == null || k <= 0) {
            return null;
        }
        ListNode fast = head;
        for (int i = 0; i < k - 1; i++) {
            fast = fast.next;
            if (fast == null) {
                return null;
            }
        }
        ListNode slow = head;
        while (fast.next != null) {
            fast = fast.next;
            slow = slow.next;
        }
        return slow;
    }


    /**
     * 反转链表
     *
     * @param head
     * @return
     */
    public ListNode ReverseListV2(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode prev = null;
        ListNode current = head;
        while (current != null) {
            ListNode tmp = current.next;
            current.next = prev;
            prev = current;
            current = tmp;
        }
        return prev;
    }


    public ListNode ReverseList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode listNode = ReverseList(head.next);
        head.next.next = head;
        head.next = null;
        return listNode;
    }


    /**
     * 合并两个排序的链表
     *
     * @param list1
     * @param list2
     * @return
     */
    public ListNode Merge(ListNode list1, ListNode list2) {
        if (list1 == null && list2 == null) {
            return null;
        }
        if (list1 == null) {
            return list2;
        }
        if (list2 == null) {
            return list1;
        }
        if (list1.val <= list2.val) {
            list1.next = this.Merge(list1.next, list2);
            return list1;
        } else {
            list2.next = this.Merge(list1, list2.next);
            return list2;
        }
    }


    /**
     * 树的子结构
     *
     * @param root1
     * @param root2
     * @return
     */
    public boolean HasSubtree(TreeNode root1, TreeNode root2) {
        if (root1 == null || root2 == null) {
            return false;
        }
        return isSubTree(root1, root2) || HasSubtree(root1.left, root2) || HasSubtree(root1.right, root2);

    }

    private boolean isSubTree(TreeNode root, TreeNode node) {
        if (root == null) {
            return true;
        }
        if (node == null) {
            return false;
        }
        if (root.val != node.val) {
            return false;
        }
        return isSubTree(root.left, node.left) && isSubTree(root.right, node.right);
    }


    /**
     * 树的反转
     *
     * @param root
     */
    public void Mirror(TreeNode root) {
        if (root == null) {
            return;
        }
        TreeNode tmp = root.left;
        root.left = root.right;
        root.right = tmp;
        Mirror(root.left);
        Mirror(root.right);
    }


    /**
     * 打印螺旋矩阵
     *
     * @param matrix
     * @return
     */
    public ArrayList<Integer> printMatrix(int[][] matrix) {
        ArrayList<Integer> result = new ArrayList<>();
        if (matrix == null || matrix.length == 0) {
            return result;
        }
        int left = 0;
        int right = matrix[0].length - 1;
        int top = 0;
        int bottom = matrix.length - 1;
        while (left <= right && top <= bottom) {
            for (int j = left; j <= right; j++) {
                result.add(matrix[top][j]);
            }
            for (int i = top + 1; i <= bottom; i++) {
                result.add(matrix[i][right]);
            }
            if (top != bottom) {
                for (int j = right - 1; j >= left; j--) {
                    result.add(matrix[bottom][j]);
                }
            }
            if (left != right) {
                for (int i = bottom - 1; i > top; i--) {
                    result.add(matrix[i][left]);
                }
            }
            left++;
            top++;
            bottom--;
            right--;
        }
        return result;
    }


    /**
     * 栈的压入、弹出序列
     *
     * @param pushA
     * @param popA
     * @return
     */
    public boolean IsPopOrder(int[] pushA, int[] popA) {
        if (pushA == null || popA == null) {
            return false;
        }
        Stack<Integer> stack = new Stack<>();
        int j = 0;
        for (int i = 0; i < pushA.length; i++) {
            stack.push(pushA[i]);
            while (!stack.isEmpty() && stack.peek() == popA[j]) {
                stack.pop();
                j++;
            }

        }
        return stack.isEmpty();
    }


    /**
     * 层次打印树
     *
     * @param root
     * @return
     */
    public ArrayList<Integer> PrintFromTopToBottom(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        ArrayList<Integer> result = new ArrayList<>();
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            result.add(node.val);
            if (node.left != null) {
                queue.add(node.left);
            }
            if (node.right != null) {
                queue.add(node.right);
            }
        }
        return result;
    }


    /**
     * 二叉搜索树的后序遍历序列
     *
     * @param sequence
     * @return
     */
    public boolean VerifySequenceOfBST(int[] sequence) {
        if (sequence == null || sequence.length == 0) {
            return false;
        }
        int end = sequence.length - 1;
        while (end >= 0) {
            int index = 0;
            while (index < end && sequence[index] < sequence[end]) {
                index++;
            }
            while (index < end && sequence[index] > sequence[end]) {
                index++;
            }
            if (index != end) {
                return false;
            }
            end--;
        }
        return true;
    }


    public boolean VerifySequenceOfBSTV2(int[] sequence) {
        if (sequence == null || sequence.length == 0) {
            return false;
        }
        return intervalVerify(sequence, 0, sequence.length - 1);
    }

    private boolean intervalVerify(int[] sequence, int start, int end) {

        if (start >= end) {
            return true;
        }
        int begin = start;
        int tmp = 0;
        while (begin < end && sequence[begin] < sequence[end]) {
            begin++;
        }
        tmp = begin;
        while (tmp < end && sequence[tmp] > sequence[end]) {
            tmp++;
        }
        if (tmp == end) {
            return intervalVerify(sequence, start, begin - 1) && intervalVerify(sequence, begin, end - 1);
        }
        return false;
    }


    /**
     * 二叉树中和为某一值的路径
     *
     * @param root
     * @param target
     * @return
     */
    public ArrayList<ArrayList<Integer>> FindPath(TreeNode root, int target) {
        if (root == null) {
            return new ArrayList<>();
        }
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        intervalFindPath(result, new ArrayList<Integer>(), root, target);
        return result;
    }

    private void intervalFindPath(ArrayList<ArrayList<Integer>> result, ArrayList<Integer> integers, TreeNode root, int target) {
        integers.add(root.val);
        if (root.left == null && root.right == null && root.val == target) {
            result.add(new ArrayList<>(integers));
        } else {
            if (root.left != null) {
                intervalFindPath(result, integers, root.left, target - root.val);
            }
            if (root.right != null) {
                intervalFindPath(result, integers, root.right, target - root.val);
            }
        }
        integers.remove(integers.size() - 1);
    }


    /**
     * 复杂链表的复制
     *
     * @param pHead
     * @return
     */
    public RandomListNode Clone(RandomListNode pHead) {
        if (pHead == null) {
            return null;
        }
        RandomListNode node = pHead;
        while (node != null) {
            RandomListNode next = new RandomListNode(node.label);
            if (node.next != null) {
                next.next = node.next;
            }
            node.next = next;
            node = next.next;
        }
        node = pHead;
        while (node != null) {
            RandomListNode next = node.next;

            if (node.random != null) {

                next.random = node.random.next;
            }
            node = next.next;
        }
        node = pHead;

        RandomListNode randomHead = pHead.next;

        while (node.next != null) {
            RandomListNode tmp = node.next;
            node.next = tmp.next;
            node = tmp;
        }
        return randomHead;
    }


    /**
     * 二叉搜索树与双向链表
     *
     * @param pRootOfTree
     * @return
     */
    public TreeNode Convert(TreeNode pRootOfTree) {
        if (pRootOfTree == null) {
            return null;
        }
        Stack<TreeNode> stack = new Stack<>();
        TreeNode p = pRootOfTree;
        TreeNode root = null;
        TreeNode prev = null;
        while (!stack.isEmpty() || p != null) {
            while (p != null) {
                stack.push(p);
                p = p.left;
            }
            p = stack.pop();
            if (prev == null) {
                root = p;
                prev = p;
            } else {
                prev.right = p;
                p.left = prev;
            }
            prev = p;
            p = p.right;
        }
        return root;
    }


    /**
     * 字符串的排列
     *
     * @param str
     * @return
     */
    public ArrayList<String> Permutation(String str) {
        ArrayList<String> result = new ArrayList<>();
        if (str == null || str.isEmpty()) {
            return result;
        }
        int len = str.length();
        char[] chars = str.toCharArray();
        Arrays.sort(chars);
        str = String.valueOf(chars);
        boolean[] used = new boolean[len];
        StringBuilder builder = new StringBuilder();
        intervalPermutation(result, used, builder, str);
        return result;

    }

    private void intervalPermutation(ArrayList<String> result, boolean[] used, StringBuilder builder, String str) {
        if (builder.length() == str.length()) {
            result.add(builder.toString());
            return;
        }
        for (int i = 0; i < str.length(); i++) {
            if (used[i]) {
                continue;
            }
            if (i > 0 && str.charAt(i) == str.charAt(i - 1) && !used[i - 1]) {
                continue;
            }
            used[i] = true;
            intervalPermutation(result, used, builder.append(str.charAt(i)), str);
            used[i] = false;
            builder.deleteCharAt(builder.length() - 1);
        }
    }


    public ArrayList<String> PermutationV2(String str) {
        if (str == null || str.length() == 0) {
            return new ArrayList<>();
        }
        ArrayList<String> result = new ArrayList<>();
        char[] chars = str.toCharArray();
        Arrays.sort(chars);
        intervalPermutationV2(result, 0, chars);

        result.sort(String::compareTo);

        return result;


    }

    private void intervalPermutationV2(ArrayList<String> result, int start, char[] tmp) {
        if (start == tmp.length - 1) {
            result.add(String.valueOf(tmp));
        }
        for (int i = start; i < tmp.length; i++) {
            if (i != start && tmp[i] == tmp[start]) {
                continue;
            }
            swap(tmp, i, start);

            intervalPermutationV2(result, start + 1, tmp);

            swap(tmp, i, start);
        }
    }

    private void swap(char[] words, int start, int end) {
        char tmp = words[start];
        words[start] = words[end];
        words[end] = tmp;
    }


    /**
     * 数组中出现次数超过一半的数字
     *
     * @param array
     * @return
     */
    public int MoreThanHalfNumSolution(int[] array) {
        if (array == null || array.length == 0) {
            return 0;
        }
        int candidate = array[0];
        int count = 1;
        for (int i = 1; i < array.length; i++) {
            int num = array[i];

            if (num == candidate) {
                count++;
            } else {
                count--;
                if (count == 0) {
                    candidate = num;
                    count = 1;
                }
            }

        }
        count = 0;
        for (int num : array) {
            if (num == candidate) {
                count++;
            }
        }
        if (count * 2 > array.length) {
            return candidate;
        }
        return 0;

    }


    /**
     * 最小的K个数
     *
     * @param input
     * @param k
     * @return
     */
    public ArrayList<Integer> GetLeastNumbersSolution(int[] input, int k) {
        if (input == null || input.length == 0 || k > input.length) {
            return new ArrayList<>();
        }
        if (k <= 0 || k > input.length) {
            return new ArrayList<>();
        }

        k--;
        int index = getPartition(input, 0, input.length - 1);
        while (index != k) {
            if (index > k) {
                index = getPartition(input, 0, index - 1);
            } else {
                index = getPartition(input, index + 1, input.length - 1);
            }
        }
        ArrayList<Integer> result = new ArrayList<>();
        for (int i = 0; i <= k; i++) {
            result.add(input[i]);
        }
        return result;
    }

    private int getPartition(int[] input, int start, int end) {
        int pivot = input[start];
        while (start < end) {
            while (start < end && input[end] >= pivot) {
                end--;
            }
            if (start < end) {
                input[start] = input[end];
                start++;
            }
            while (start < end && input[start] < pivot) {
                start++;
            }
            if (start < end) {
                input[end] = input[start];
                end--;
            }
        }
        input[start] = pivot;
        return start;
    }


    public ArrayList<Integer> GetLeastNumbersSolutionV2(int[] input, int k) {
        ArrayList<Integer> result = new ArrayList<>();
        if (input == null || input.length == 0) {
            return result;
        }
        if (k <= 0 || k > input.length) {
            return result;
        }
        Arrays.sort(input);
        for (int i = 0; i < k; i++) {
            result.add(input[i]);
        }
        return result;
    }


    /**
     * 连续子数组的最大和
     *
     * @param array
     * @return
     */
    public int FindGreatestSumOfSubArray(int[] array) {
        if (array == null || array.length == 0) {
            return -1;
        }
        int local = 0;
        int global = Integer.MIN_VALUE;
        for (int num : array) {
            local = local < 0 ? num : local + num;
            global = Math.max(local, global);
        }
        return global;
    }


    /**
     * todo 剑指offer
     * 整数中1出现的次数（从1到n整数中1出现的次数）
     *
     * @param n
     * @return
     */
    public int NumberOf1Between1AndN_Solution(int n) {
        return 0;
    }


    /**
     * 把数组排成最小的数
     *
     * @param numbers
     * @return
     */
    public String PrintMinNumber(int[] numbers) {
        if (numbers == null || numbers.length == 0) {
            return "";
        }
        String[] words = new String[numbers.length];
        for (int i = 0; i < numbers.length; i++) {
            words[i] = String.valueOf(numbers[i]);
        }
        Arrays.sort(words, (o1, o2) -> {
            String tmp1 = o1 + o2;
            String tmp2 = o2 + o1;
            return tmp1.compareTo(tmp2);
        });
        StringBuilder builder = new StringBuilder();
        for (String word : words) {
            builder.append(word);
        }
        if (builder.charAt(0) == '0') {
            return "0";
        }
        return builder.toString();
    }


    /**
     * 获取第N个丑数
     *
     * @param index
     * @return
     */
    public int GetUglyNumberSolution(int index) {
        return 0;
    }


    /**
     * 第一个只出现一次的字符
     *
     * @param str
     * @return
     */
    public int FirstNotRepeatingChar(String str) {
        if (str == null || str.length() == 0) {
            return -1;
        }
        HashMap<Character, Integer> map = new HashMap<>();
        for (int i = 0; i < str.length(); i++) {
            char c = str.charAt(i);
            Integer count = map.getOrDefault(c, 0);
            count++;
            map.put(c, count);
        }
        for (int i = 0; i < str.length(); i++) {
            if (map.get(str.charAt(i)) == 1) {
                return i;
            }
        }
        return -1;
    }


    /**
     * todo
     * 数组中的逆序对
     *
     * @param array
     * @return
     */
    public int InversePairs(int[] array) {
        return 0;
    }


    /**
     * 两个链表的第一个公共结点
     *
     * @param pHead1
     * @param pHead2
     * @return
     */
    public ListNode FindFirstCommonNode(ListNode pHead1, ListNode pHead2) {
        if (pHead1 == null || pHead2 == null) {
            return null;
        }
        ListNode p1 = pHead1;
        ListNode p2 = pHead2;
        while (p1 != p2) {
            p1 = p1 == null ? pHead2 : p1.next;
            p2 = p2 == null ? pHead1 : p2.next;
        }
        return p1;
    }


    /**
     * 数字在排序数组中出现的次数
     * ˚
     *
     * @param array
     * @param k
     * @return
     */
    public int GetNumberOfKV2(int[] array, int k) {
        if (array == null || array.length == 0) {
            return 0;
        }
        int count = 0;
        for (int num : array) {
            if (num == k) {
                count++;
            }
        }
        return count;
    }


    public int GetNumberOfK(int[] array, int k) {
        if (array == null || array.length == 0) {
            return 0;
        }
        int leftSide = getLeftSide(array, 0, array.length - 1, k);

        return -1;
    }

    private int getLeftSide(int[] array, int left, int right, int k) {
        if (left > right) {
            return -1;
        }
        int mid = -1;

        while (left < right) {
            mid = left + (right - left) / 2;

            int value = array[mid];

            if (value < k) {

                left = mid + 1;

            } else if (value > k) {

                right = mid - 1;
            } else if (mid > 0 && array[mid] == array[mid - 1]) {
                mid = mid - 1;
            }
        }
        return mid;
    }


    /**
     * 二叉树的深度
     *
     * @param root
     * @return
     */
    public int TreeDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        return 1 + Math.max(TreeDepth(root.left), TreeDepth(root.right));
    }


    /**
     * 平衡二叉树
     *
     * @param root
     * @return
     */
    public boolean IsBalanced_Solution(TreeNode root) {
        if (root == null) {
            return true;
        }
        int leftDepth = TreeDepth(root.left);
        int rightDepth = TreeDepth(root.right);
        if (Math.abs(leftDepth - rightDepth) <= 1) {
            return IsBalanced_Solution(root.left) && IsBalanced_Solution(root.right);
        }
        return false;
    }


    /**
     * 数组中只出现一次的数字
     *
     * @param array
     * @param num1
     * @param num2
     */
    public void FindNumsAppearOnce(int[] array, int[] num1, int[] num2) {
        if (array == null || array.length == 0) {
            return;
        }
        int result = 0;
        for (int num : array) {
            result ^= num;
        }
        int index = 0;
        for (int i = 0; i < 32; i++) {
            if (((1 << i) & result) != 0) {
                index = i;
                break;
            }
        }
        for (int num : array) {
            if ((num & (1 << index)) != 0) {
                num1[0] ^= num;
            } else {
                num2[0] ^= num;
            }
        }
    }


    /**
     * todo
     * 和为S的连续正数序列
     *
     * @param sum
     * @return
     */
    public ArrayList<ArrayList<Integer>> FindContinuousSequence(int sum) {
        return new ArrayList<>();
    }


    /**
     * 和为S的两个数字
     *
     * @param array
     * @param sum
     * @return
     */
    public ArrayList<Integer> FindNumbersWithSum(int[] array, int sum) {
        ArrayList<Integer> result = new ArrayList<>();
        if (array == null || array.length == 0) {
            return result;
        }
        int begin = 0;
        int end = array.length - 1;
        while (begin < end) {
            if (array[begin] + array[end] == sum) {
                result.add(array[begin]);
                result.add(array[end]);
                return result;
            }
            while (begin < end && array[begin] + array[end] < sum) {
                begin++;
            }
            while (begin < end && array[begin] + array[end] > sum) {
                end--;
            }
        }
        return result;
    }


    /**
     * 左旋转字符串
     *
     * @param str
     * @param n
     * @return
     */
    public String LeftRotateString(String str, int n) {
        if (str == null || str.isEmpty()) {
            return "";
        }
        int len = str.length();
        str += str;
        return str.substring(n, n + len);
    }


    /**
     * 翻转单词顺序列
     *
     * @param str
     * @return
     */
    public String ReverseSentence(String str) {
        if (str == null || str.isEmpty()) {
            return "";
        }

        String[] words = str.split(" ");
        StringBuilder builder = new StringBuilder();
        for (int i = words.length - 1; i >= 0; i--) {
            builder.append(words[i]);
            if (i > 0) {
                builder.append(" ");
            }
        }
        return builder.length() != 0 ? builder.toString() : str;
    }


    /**
     * 扑克牌顺子
     *
     * @param numbers
     * @return
     */
    public boolean isContinuous(int[] numbers) {
        if (numbers == null || numbers.length == 0) {
            return true;
        }
        int min = 14;
        int max = -1;

        int countOfZero = 0;
        int[] used = new int[14];
        for (int num : numbers) {
            if (num == 0) {
                countOfZero++;
                continue;
            }
            if (num >= 14 || num <= -1) {
                return false;
            }
            if (used[num] >= 1) {
                return false;
            }
            used[num]++;
            max = Math.max(max, num);
            min = Math.min(min, num);

        }
        if (countOfZero >= 5) {
            return true;
        }
        if (max - min > 4) {
            return false;
        }
        return true;
    }


    /**
     * todo 约瑟夫环
     * 孩子们的游戏(圆圈中最后剩下的数)
     *
     * @param n
     * @param m
     * @return
     */
    public int LastRemainingSolution(int n, int m) {
        if (n <= 0 || m <= 0) {
            return -1;
        }
        return -1;
    }

    /**
     * 禁止使用循环语句以及三元语句
     * 求1+2+3+...+n
     *
     * @param n
     * @return
     */
    public int Sum_Solution(int n) {
        int result = n;

        boolean flag = n > 0 && ((result += Sum_Solution(n - 1)) > 0);

        return result;
    }


    /**
     * 禁止使用加减法
     *
     * @param num1
     * @param num2
     * @return
     */
    public int Add(int num1, int num2) {
        return -1;
    }


    /**
     * 把字符串转换成整数
     *
     * @param str
     * @return
     */
    public int StrToInt(String str) {
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
        int len = str.length();

        long result = 0L;
        while (index < len && Character.isDigit(str.charAt(index))) {

            result = result * 10 + Character.getNumericValue(str.charAt(index));

            index++;
        }
        if (index != len) {
            return 0;
        }
        result = sign * result;

        if (result > Integer.MAX_VALUE || result < Integer.MIN_VALUE) {
            return 0;
        }
        return (int) result;
    }


    /**
     * 构建乘积数组
     *
     * @param A
     * @return
     */
    public int[] multiply(int[] A) {
        if (A == null || A.length == 0) {
            return new int[]{};
        }
        int[] result = new int[A.length];
        int base = 1;
        for (int i = 0; i < A.length; i++) {
            result[i] = base;
            base *= A[i];
        }
        base = 1;
        for (int i = result.length - 1; i >= 0; i--) {
            result[i] *= base;
            base *= A[i];
        }
        return result;
    }


    /**
     * 正则表达式匹配
     *
     * @param str
     * @param pattern
     * @return
     */
    public boolean match(char[] str, char[] pattern) {
        if (str == null || pattern == null) {
            return false;
        }
        int m = str.length;
        int n = pattern.length;
        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[0][0] = true;
        for (int j = 1; j <= pattern.length; j++) {
            dp[0][j] = pattern[j - 1] == '*' && dp[0][j - 2];
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (str[i - 1] == pattern[j - 1] || pattern[j - 1] == '.') {
                    dp[i][j] = dp[i - 1][j - 1];
                } else if (pattern[j - 1] == '*') {
                    if (pattern[j - 2] != '.' && str[i - 1] != pattern[j - 2]) {
                        dp[i][j] = dp[i][j - 2];
                    } else {
                        dp[i][j] = dp[i][j - 2] || dp[i - 1][j - 1] || dp[i - 1][j];
                    }

                }
            }
        }
        return dp[m][n];
    }


    /**
     * 数组中获取最小的K个数
     *
     * @param input
     * @param k
     * @return
     */
    public ArrayList<Integer> GetLeastNumbersSolutionV3(int[] input, int k) {
        ArrayList<Integer> result = new ArrayList<>();
        if (input == null || input.length == 0) {
            return result;
        }
        if (k <= 0 || k > input.length) {
            return result;
        }
        PriorityQueue<Integer> queue = new PriorityQueue<>(k, Comparator.reverseOrder());
        for (int number : input) {
            if (queue.size() != k) {
                queue.add(number);
            } else if (queue.peek() >= number) {
                queue.poll();
                queue.offer(number);
            }
        }
        result.addAll(queue);

        Collections.sort(result);
        return result;
    }


    /**
     * 表示数值的字符串
     *
     * @param str
     * @return
     */
    public boolean isNumeric(char[] str) {
        if (str == null || str.length == 0) {
            return false;
        }
        // 是否看到 点
        boolean pointSeen = false;

        // 是否看到 E
        boolean eSeen = false;

        // 是否看到数字
        boolean numberSeen = false;

        // 是否在e后面数字
        boolean numberAfterE = true;

        for (int i = 0; i < str.length; i++) {
            char word = str[i];
            if (Character.isDigit(word)) {
                numberSeen = true;
                numberAfterE = true;
            } else if (word == 'e' || word == 'E') {
                if (eSeen || !numberSeen) {
                    return false;
                }
                eSeen = true;
                numberAfterE = false;
            } else if (word == '.') {
                if (pointSeen || eSeen || i == 0) {
                    return false;
                }
                pointSeen = true;
            } else if (word == '-' || word == '+') {
                boolean preWordE = i != 0 && (str[i - 1] == 'e' || str[i - 1] == 'E');
                if (i != 0 && !preWordE) {
                    return false;
                }

            } else {
                return false;
            }
        }
        return numberSeen && numberAfterE;

    }


    public boolean isNumericV2(char[] str) {
        if (str == null || str.length == 0) {
            return false;
        }
        boolean numberAfterE = true;
        boolean seenE = false;
        boolean seenDit = false;
        boolean seenNumber = false;
        for (int i = 0; i < str.length; i++) {
            char word = str[i];
            if (Character.isDigit(word)) {
                numberAfterE = true;
                seenNumber = true;
            } else if (word == 'e' || word == 'E') {
                if (seenE || !seenNumber) {
                    return false;
                }
                seenE = true;
                numberAfterE = false;
            } else if (word == '-' || word == '+') {
                boolean preWord = i != 0 && (str[i - 1] == 'e' || str[i - 1] == 'E');

                if (i != 0 && !preWord) {
                    return false;
                }
            } else if (word == '.') {
                if (i == 0 || seenDit || seenE) {
                    return false;
                }
                seenDit = true;
            } else {
                return false;
            }
        }
        return seenNumber && numberAfterE;
    }


    /**
     * 链表中环的入口结点
     *
     * @param pHead
     * @return
     */
    public ListNode EntryNodeOfLoop(ListNode pHead) {
        if (pHead == null || pHead.next == null) {
            return null;
        }
        ListNode fast = pHead;
        ListNode slow = pHead;
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (slow == fast) {
                fast = pHead;
                while (fast != slow) {
                    fast = fast.next;
                    slow = slow.next;
                }
                return fast;
            }
        }
        return null;
    }


    /**
     * 删除链表中重复的结点
     *
     * @param pHead
     * @return
     */
    public ListNode deleteDuplication(ListNode pHead) {
        if (pHead == null || pHead.next == null) {
            return pHead;
        }
        if (pHead.next.val == pHead.val) {
            ListNode thirdNode = pHead.next.next;
            while (thirdNode != null && thirdNode.val == pHead.val) {
                thirdNode = thirdNode.next;
            }
            return deleteDuplication(thirdNode);
        }
        pHead.next = deleteDuplication(pHead.next);
        return pHead;
    }


    /**
     * 二叉树的中序下一个结点
     *
     * @param pNode
     * @return
     */
    public TreeLinkNode GetNext(TreeLinkNode pNode) {
        if (pNode == null) {
            return null;
        }
        TreeLinkNode right = pNode.right;
        if (right != null) {
            while (right.left != null) {
                right = right.left;
            }
            return right;
        }
        while (pNode.next != null) {
            if (pNode.next.left == pNode) {
                return pNode.next;
            }
            pNode = pNode.next;
        }
        return null;
    }

    /**
     * 对称的二叉树
     *
     * @param pRoot
     * @return
     */
    public boolean isSymmetrical(TreeNode pRoot) {
        if (pRoot == null) {
            return true;
        }
        TreeNode left = pRoot.left;
        TreeNode right = pRoot.right;
        return intervalSymmetrical(left, right);
    }

    private boolean intervalSymmetrical(TreeNode left, TreeNode right) {
        if (left == null && right == null) {
            return true;
        }
        if (left == null || right == null) {
            return false;
        }
        if (left.val != right.val) {
            return false;
        }
        return this.intervalSymmetrical(left.left, right.right) && this.intervalSymmetrical(left.right, right.left);
    }


    /**
     * 按之字形顺序打印二叉树
     *
     * @param pRoot
     * @return
     */
    public ArrayList<ArrayList<Integer>> Print(TreeNode pRoot) {
        if (pRoot == null) {
            return new ArrayList<>();
        }
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        LinkedList<TreeNode> deque = new LinkedList<>();
        deque.offer(pRoot);

        boolean leftToRight = true;

        while (!deque.isEmpty()) {

            int size = deque.size();

            LinkedList<Integer> tmp = new LinkedList<>();

            for (int i = 0; i < size; i++) {

                TreeNode poll = deque.poll();


                if (leftToRight) {
                    tmp.addLast(poll.val);
                } else {
                    tmp.addFirst(poll.val);
                }


                if (poll.left != null) {
                    deque.add(poll.left);
                }
                if (poll.right != null) {
                    deque.add(poll.right);
                }
            }
            result.add(new ArrayList<>(tmp));
            leftToRight = !leftToRight;
        }
        return result;
    }


    /**
     * 把二叉树打印成多行
     *
     * @param pRoot
     * @return
     */
    public ArrayList<ArrayList<Integer>> PrintII(TreeNode pRoot) {
        if (pRoot == null) {
            return new ArrayList<>();
        }
        ArrayList<ArrayList<Integer>> ans = new ArrayList<>();
        LinkedList<TreeNode> deque = new LinkedList<>();
        deque.add(pRoot);
        while (!deque.isEmpty()) {
            int size = deque.size();
            ArrayList<Integer> tmp = new ArrayList<>();
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
     * 二叉搜索树的第k个结点
     *
     * @param pRoot
     * @param k
     * @return
     */
    TreeNode KthNode(TreeNode pRoot, int k) {
        if (pRoot == null || k <= 0) {
            return null;
        }
        Stack<TreeNode> stack = new Stack<>();
        TreeNode p = pRoot;
        int index = 0;
        while (!stack.isEmpty() || p != null) {
            while (p != null) {
                stack.push(p);
                p = p.left;
            }
            p = stack.pop();

            index++;

            if (index == k) {
                return p;
            }
            p = p.right;
        }
        return null;
    }


    /**
     * 滑动窗口的最大值
     *
     * @param num
     * @param size
     * @return
     */
    public ArrayList<Integer> maxInWindows(int[] num, int size) {
        if (num == null || num.length == 0 || size <= 0 || size > num.length) {
            return new ArrayList<>();
        }
        ArrayList<Integer> result = new ArrayList<>();
        LinkedList<Integer> deque = new LinkedList<>();
        for (int i = 0; i < num.length; i++) {
            int index = i - size + 1;

            if (!deque.isEmpty() && index > deque.peekFirst()) {
                deque.pollFirst();
            }
            while (!deque.isEmpty() && num[deque.peekLast()] <= num[i]) {
                deque.pollLast();
            }
            deque.add(i);
            if (index >= 0) {
                result.add(num[deque.peekFirst()]);
            }
        }
        return result;
    }

    /**
     * 矩阵中的路径
     *
     * @param matrix
     * @param rows
     * @param cols
     * @param str
     * @return
     */
    public boolean hasPath(char[] matrix, int rows, int cols, char[] str) {
        if (matrix == null || matrix.length == 0) {
            return false;
        }
        boolean[][] used = new boolean[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int index = i * cols + j;
                if (matrix[index] == str[0] && checkPath(matrix, used, rows, cols, i, j, 0, str)) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean checkPath(char[] matrix, boolean[][] used, int rows, int cols, int i, int j, int start, char[] str) {
        if (start >= str.length) {
            return true;
        }
        if (i < 0 || i >= rows || j < 0 || j >= cols || used[i][j]) {
            return false;
        }
        int index = i * cols + j;
        if (matrix[index] != str[start]) {
            return false;
        }
        used[i][j] = true;

        if (checkPath(matrix, used, rows, cols, i - 1, j, start + 1, str) ||
                checkPath(matrix, used, rows, cols, i + 1, j, start + 1, str) ||
                checkPath(matrix, used, rows, cols, i, j - 1, start + 1, str) ||
                checkPath(matrix, used, rows, cols, i, j + 1, start + 1, str)) {
            return true;
        }
        used[i][j] = false;
        return false;
    }


    /**
     * 机器人的运动范围
     *
     * @param threshold
     * @param rows
     * @param cols
     * @return
     */
    public int movingCount(int threshold, int rows, int cols) {
        if (rows <= 0 || cols <= 0) {
            return 0;
        }
        boolean[][] used = new boolean[rows][cols];

        return intervalCount(0, 0, rows, cols, threshold, used);
    }

    private int intervalCount(int i, int j, int rows, int cols, int threshold, boolean[][] used) {
        if (i < 0 || i >= rows || j < 0 || j >= cols) {
            return 0;
        }
        if (used[i][j]) {
            return 0;
        }
        used[i][j] = true;

        boolean notExceed = (this.calculateSum(i) + this.calculateSum(j)) <= threshold;

        if (!notExceed) {
            return 0;
        }
        int count = 1;
        count += intervalCount(i - 1, j, rows, cols, threshold, used);
        count += intervalCount(i + 1, j, rows, cols, threshold, used);
        count += intervalCount(i, j - 1, rows, cols, threshold, used);
        count += intervalCount(i, j + 1, rows, cols, threshold, used);

        return count;
    }

    private int calculateSum(int number) {
        int sum = 0;
        while (number != 0) {
            sum += number % 10;
            number /= 10;
        }
        return sum;
    }


    /**
     * 剪绳子
     *
     * @param target
     * @return
     */
    public int cutRope(int target) {
        if (target == 0) {
            return 0;
        }
        if (target <= 2) {
            return 1;
        }
        if (target == 3) {
            return 2;
        }
        int[] dp = new int[target + 1];


        dp[0] = dp[1] = 1;

        dp[2] = 2;

        dp[3] = 3;

        dp[4] = 4;

        for (int i = 5; i <= target; i++) {
            int result = 0;
            for (int j = 1; j <= i / 2; j++) {
                result = Math.max(result, dp[j] * dp[i - j]);
            }
            dp[i] = result;
        }
        return dp[target];
    }


}
